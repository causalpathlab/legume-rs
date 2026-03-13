use crate::atoi::sifter::*;
use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::data::dna::Dna;
use crate::data::dna_stat_map::*;
use crate::data::methylation::*;
use crate::data::util_htslib::*;
use crate::pipeline_util::*;

use dashmap::DashMap as HashMap;
use dashmap::DashSet as HashSet;
use genomic_data::gff::{GeneId, GffRecordMap};
use rust_htslib::faidx;
use std::sync::{Arc, Mutex};

/// Parameters for A-to-I discovery and quantification, decoupled from DartSeqCountArgs.
pub struct AtoIParams {
    pub genome_file: Box<str>,
    pub wt_bam_files: Vec<Box<str>>,
    pub mut_bam_files: Vec<Box<str>>,
    pub gene_barcode_tag: Box<str>,
    pub cell_barcode_tag: Box<str>,
    pub include_missing_barcode: bool,
    pub min_coverage: usize,
    pub min_conversion: usize,
    pub pseudocount: usize,
    pub pvalue_cutoff: f64,
    pub resolution_kb: Option<f32>,
    pub backend: SparseIoBackend,
    pub output: Box<str>,
    pub output_value_type: MethFeatureType,
    pub row_nnz_cutoff: Option<usize>,
    pub column_nnz_cutoff: Option<usize>,
    pub cell_membership_file: Option<Box<str>>,
    pub membership_barcode_col: usize,
    pub membership_celltype_col: usize,
    pub exact_barcode_match: bool,
}

impl AtoIParams {
    /// Create AtoISifter with these parameters
    pub fn create_atoi_sifter<'a>(
        &self,
        faidx: &'a faidx::Reader,
        chr: &'a str,
        capacity: usize,
    ) -> AtoISifter<'a> {
        AtoISifter {
            faidx,
            chr,
            min_coverage: self.min_coverage,
            min_conversion: self.min_conversion,
            pseudocount: self.pseudocount,
            max_pvalue_cutoff: self.pvalue_cutoff,
            candidate_sites: Vec::with_capacity(capacity),
        }
    }

    pub fn backend_file_path(&self, batch_name: &str) -> Box<str> {
        match self.backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &self.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &self.output, batch_name),
        }
        .into_boxed_str()
    }

    pub fn qc_cutoffs(&self) -> SqueezeCutoffs {
        SqueezeCutoffs {
            row: self.row_nnz_cutoff.unwrap_or(0),
            column: self.column_nnz_cutoff.unwrap_or(0),
        }
    }

    pub fn value_extractor(&self) -> impl Fn(&MethylationData) -> f32 {
        let output_type = self.output_value_type.clone();
        move |dat: &MethylationData| -> f32 {
            match output_type {
                MethFeatureType::Beta => {
                    let tot = (dat.methylated + dat.unmethylated) as f32;
                    (dat.methylated as f32) / tot.max(1.)
                }
                MethFeatureType::Methylated => dat.methylated as f32,
                MethFeatureType::Unmethylated => dat.unmethylated as f32,
            }
        }
    }
}

/// Minimum number of positions required to attempt A-to-I detection
const MIN_LENGTH_FOR_TESTING: usize = 1;

/// Padding around target region when reading BAM files
const BAM_READ_PADDING: i64 = 1;

///////////////////////////////////////////
// A-to-I editing site discovery         //
///////////////////////////////////////////

pub fn find_all_atoi_sites(
    gff_map: &GffRecordMap,
    params: &AtoIParams,
) -> anyhow::Result<HashMap<GeneId, Vec<AtoISite>>> {
    let njobs = gff_map.len();
    info!("Searching A-to-I editing sites over {} blocks", njobs);

    load_fasta_index(&params.genome_file)?;

    let arc_gene_sites = Arc::new(HashMap::<GeneId, Vec<AtoISite>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_count(njobs as u64)
        .try_for_each(|rec| -> anyhow::Result<()> {
            find_atoi_sites_in_gene(rec, params, arc_gene_sites.clone())
        })?;

    Arc::try_unwrap(arc_gene_sites)
        .map_err(|_| anyhow::anyhow!("failed to release atoi gene_sites"))
}

fn find_atoi_sites_in_gene(
    gff_record: &GffRecord,
    params: &AtoIParams,
    arc_gene_sites: Arc<HashMap<GeneId, Vec<AtoISite>>>,
) -> anyhow::Result<()> {
    let gene_id = gff_record.gene_id.clone();
    let strand = &gff_record.strand;
    let chr = gff_record.seqname.as_ref();

    let faidx_reader = load_fasta_index(&params.genome_file)?;

    let mut wt_base_freq_map = DnaBaseFreqMap::new();
    for wt_file in &params.wt_bam_files {
        wt_base_freq_map.update_from_gene(
            wt_file,
            gff_record,
            &params.gene_barcode_tag,
            params.include_missing_barcode,
        )?;
    }

    let positions = wt_base_freq_map.sorted_positions();
    if positions.len() < MIN_LENGTH_FOR_TESTING {
        return Ok(());
    }

    let mut mut_base_freq_map = DnaBaseFreqMap::new();
    for mut_file in &params.mut_bam_files {
        mut_base_freq_map.update_from_gene(
            mut_file,
            gff_record,
            &params.gene_barcode_tag,
            params.include_missing_barcode,
        )?;
    }

    let wt_freq = wt_base_freq_map
        .marginal_frequency_map()
        .ok_or_else(|| anyhow::anyhow!("failed to count wt freq"))?;
    let mut_freq = mut_base_freq_map
        .marginal_frequency_map()
        .ok_or_else(|| anyhow::anyhow!("failed to count mut freq"))?;

    let mut sifter = params.create_atoi_sifter(&faidx_reader, chr, positions.len());

    match strand {
        Strand::Forward => {
            sifter.forward_scan(&positions, wt_freq, Some(mut_freq));
        }
        Strand::Backward => {
            sifter.backward_scan(&positions, wt_freq, Some(mut_freq));
        }
    }

    let mut candidate_sites = sifter.candidate_sites;
    candidate_sites.sort();
    candidate_sites.dedup();

    if !candidate_sites.is_empty() {
        arc_gene_sites.insert(gene_id, candidate_sites);
    }

    Ok(())
}

///////////////////////////////////////////
// A-to-I count matrix (second pass)     //
///////////////////////////////////////////

pub fn process_all_bam_files_to_backend_atoi(
    params: &AtoIParams,
    atoi_sites: &HashMap<GeneId, Vec<AtoISite>>,
    gff_map: &GffRecordMap,
) -> anyhow::Result<()> {
    let membership = if let Some(ref path) = params.cell_membership_file {
        let m = CellMembership::from_file(
            path,
            params.membership_barcode_col,
            params.membership_celltype_col,
            !params.exact_barcode_match,
        )?;
        Some(m)
    } else {
        None
    };

    let gene_key = crate::pipeline_util::create_gene_key_function(gff_map);
    let site_key = |x: &BedWithGene| -> Box<str> {
        let gene_part = gene_key(x);
        format!("{}_{}_{}_{}/A2I", gene_part, x.chr, x.start, x.stop).into_boxed_str()
    };
    let take_value = params.value_extractor();
    let cutoffs = params.qc_cutoffs();

    let mut sites = HashSet::<Box<str>>::default();
    let mut site_data_files: Vec<Box<str>> = vec![];

    let wt_batch_names = uniq_batch_names(&params.wt_bam_files)?;

    for (bam_file, batch_name) in params.wt_bam_files.iter().zip(wt_batch_names) {
        info!(
            "collecting A-to-I data over {} sites from {} ...",
            atoi_sites.iter().map(|x| x.value().len()).sum::<usize>(),
            bam_file
        );

        let stats = gather_atoi_stats(atoi_sites, params, gff_map, bam_file, membership.as_ref())?;

        let atoi_batch_name = format!("{}_atoi", batch_name);
        let site_data_file = params.backend_file_path(&atoi_batch_name);
        let triplets = crate::pipeline_util::summarize_stats(&stats, site_key, &take_value);
        let data = triplets.to_backend(&site_data_file)?;
        data.qc(cutoffs.clone())?;
        sites.extend(data.row_names()?);
        info!("created A-to-I site-level data: {}", &site_data_file);
        site_data_files.push(site_data_file);
    }

    // Reorder rows
    let mut sites_sorted: Vec<_> = sites.into_iter().collect();
    sites_sorted.sort();

    let backend = &params.backend;
    for data_file in site_data_files {
        open_sparse_matrix(&data_file, backend)?.reorder_rows(&sites_sorted)?;
    }

    Ok(())
}

pub fn gather_atoi_stats(
    atoi_sites: &HashMap<GeneId, Vec<AtoISite>>,
    params: &AtoIParams,
    gff_map: &GffRecordMap,
    bam_file: &str,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let ndata = atoi_sites.iter().map(|x| x.value().len()).sum::<usize>();
    let arc_ret = Arc::new(Mutex::new(Vec::with_capacity(ndata)));

    atoi_sites
        .into_iter()
        .par_bridge()
        .progress_count(atoi_sites.len() as u64)
        .try_for_each(|gs| -> anyhow::Result<()> {
            let gene = gs.key();
            let sites = gs.value();

            if let Some(gff) = gff_map.get(gene) {
                let stats =
                    collect_gene_atoi_stats(params, bam_file, &gff, sites, cell_membership)?;
                arc_ret.lock().expect("lock").extend(stats);
            }
            Ok(())
        })?;

    Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release atoi stats"))?
        .into_inner()
        .map_err(Into::into)
}

fn collect_gene_atoi_stats(
    params: &AtoIParams,
    bam_file: &str,
    gff_record: &GffRecord,
    sites: &[AtoISite],
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut all_stats = Vec::new();

    for site in sites {
        let stats = estimate_atoi_stat(params, bam_file, gff_record, site, cell_membership)?;
        all_stats.extend(stats);
    }

    Ok(all_stats)
}

fn estimate_atoi_stat(
    params: &AtoIParams,
    bam_file: &str,
    gff_record: &GffRecord,
    atoi_site: &AtoISite,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut stat_map =
        DnaBaseFreqMap::new_with_cell_barcode(&params.cell_barcode_tag, cell_membership);
    let editing_pos = atoi_site.editing_pos;

    let mut gff = gff_record.clone();
    gff.start = (editing_pos - BAM_READ_PADDING).max(0);
    gff.stop = editing_pos + BAM_READ_PADDING;
    stat_map.update_from_gene(
        bam_file,
        &gff,
        &params.gene_barcode_tag,
        params.include_missing_barcode,
    )?;

    let gene = gff.gene_id;
    let chr = gff.seqname.as_ref();
    let strand = &gff.strand;

    // A-to-I: fwd = A->G, rev = T->C
    let (unmutated_base, mutated_base) = match strand {
        Strand::Forward => (Dna::A, Dna::G),
        Strand::Backward => (Dna::T, Dna::C),
    };

    let methylation_stat = stat_map.stratified_frequency_at(editing_pos);

    let Some(meth_stat) = methylation_stat else {
        return Ok(Vec::new());
    };

    let (start, stop) = bin_position_kb(editing_pos, params.resolution_kb);

    let stats = meth_stat
        .iter()
        .filter_map(|(cb, counts)| {
            let edited = counts.get(Some(&mutated_base));
            let unedited = counts.get(Some(&unmutated_base));

            if (params.include_missing_barcode || cb != &CellBarcode::Missing) && edited > 0 {
                Some((
                    cb.clone(),
                    BedWithGene {
                        chr: chr.into(),
                        start,
                        stop,
                        gene: gene.clone(),
                        strand: *strand,
                    },
                    MethylationData {
                        methylated: edited,
                        unmethylated: unedited,
                        m6a_pos: editing_pos,
                    },
                ))
            } else {
                None
            }
        })
        .collect();

    Ok(stats)
}
