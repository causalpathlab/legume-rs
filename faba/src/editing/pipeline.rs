use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::data::dna::Dna;
use crate::data::dna_stat_map::*;
use crate::data::methylation::*;
use crate::data::util_htslib::*;
use crate::editing::sifter::*;
use crate::editing::ConversionSite;
use crate::pipeline_util::*;

use dashmap::DashMap as HashMap;
use dashmap::DashSet as HashSet;
use fnv;
use genomic_data::gff::{GeneId, GffRecordMap};
use rust_htslib::faidx;
use std::sync::{Arc, Mutex};

/// Padding around target region when reading BAM files
const BAM_READ_PADDING: i64 = 1;

/// Unified parameters for base conversion (m6A and A-to-I) discovery and quantification.
pub struct ConversionParams {
    pub genome_file: Box<str>,
    pub wt_bam_files: Vec<Box<str>>,
    pub mut_bam_files: Vec<Box<str>>,
    pub gene_barcode_tag: Box<str>,
    pub cell_barcode_tag: Box<str>,
    pub include_missing_barcode: bool,
    pub min_coverage: usize,
    pub min_conversion: usize,
    pub pseudocount: usize,
    pub pvalue_cutoff: f32,
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
    pub mod_type: ModificationType,
}

impl ConversionParams {
    /// Create a ConversionSifter with these parameters
    pub fn create_sifter<'a>(
        &self,
        faidx: &'a faidx::Reader,
        chr: &'a str,
        capacity: usize,
    ) -> ConversionSifter<'a> {
        ConversionSifter {
            faidx,
            chr,
            min_coverage: self.min_coverage,
            min_conversion: self.min_conversion,
            pseudocount: self.pseudocount,
            max_pvalue_cutoff: self.pvalue_cutoff,
            mod_type: self.mod_type.clone(),
            candidate_sites: Vec::with_capacity(capacity),
        }
    }

    /// Create backend file path for a given batch name
    pub fn backend_file_path(&self, batch_name: &str) -> Box<str> {
        match self.backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &self.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &self.output, batch_name),
        }
        .into_boxed_str()
    }

    /// Get QC cutoffs
    pub fn qc_cutoffs(&self) -> SqueezeCutoffs {
        SqueezeCutoffs {
            row: self.row_nnz_cutoff.unwrap_or(0),
            column: self.column_nnz_cutoff.unwrap_or(0),
        }
    }

    /// Create value extraction function based on output type
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

    /// Minimum number of positions required to attempt site discovery.
    /// m6A requires a triplet (3 consecutive positions), A-to-I needs only 1.
    fn min_length_for_testing(&self) -> usize {
        match self.mod_type {
            ModificationType::M6A { .. } => 3,
            ModificationType::AtoI => 1,
        }
    }

    /// Load cell membership from file if configured
    pub fn load_membership(&self) -> anyhow::Result<Option<CellMembership>> {
        if let Some(ref path) = self.cell_membership_file {
            let m = CellMembership::from_file(
                path,
                self.membership_barcode_col,
                self.membership_celltype_col,
                !self.exact_barcode_match,
            )?;
            info!(
                "Loaded {} cell barcodes from membership file: {}",
                m.num_cells(),
                path
            );
            info!("Prefix matching: {}", !self.exact_barcode_match);
            Ok(Some(m))
        } else {
            Ok(None)
        }
    }
}

///////////////////////////////////////////
// FIRST PASS: Site discovery            //
///////////////////////////////////////////

/// Find all conversion sites across the genome.
///
/// For m6A with cell_membership: uses per-cell-type discovery.
/// For m6A without membership or for AtoI: uses bulk statistics.
pub fn find_all_conversion_sites(
    gff_map: &GffRecordMap,
    params: &ConversionParams,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<HashMap<GeneId, Vec<ConversionSite>>> {
    let njobs = gff_map.len();
    info!(
        "Searching {} conversion sites over {} blocks",
        params.mod_type.label(),
        njobs
    );

    // Validate reference genome
    info!("Loading reference genome: {}", params.genome_file);
    load_fasta_index(&params.genome_file)?;

    let arc_gene_sites = Arc::new(HashMap::<GeneId, Vec<ConversionSite>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_count(njobs as u64)
        .try_for_each(|rec| -> anyhow::Result<()> {
            find_sites_in_gene(rec, params, arc_gene_sites.clone(), cell_membership)
        })?;

    Arc::try_unwrap(arc_gene_sites).map_err(|_| anyhow::anyhow!("failed to release gene_sites"))
}

/// Per-gene site discovery: reads WT and MUT BAM files, creates sifter, dispatches via scan().
fn find_sites_in_gene(
    gff_record: &GffRecord,
    params: &ConversionParams,
    arc_gene_sites: Arc<HashMap<GeneId, Vec<ConversionSite>>>,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<()> {
    let gene_id = gff_record.gene_id.clone();
    let strand = &gff_record.strand;
    let chr = gff_record.seqname.as_ref();

    // Each thread creates its own reader (faidx is not thread-safe)
    let faidx_reader = load_fasta_index(&params.genome_file)?;

    let candidate_sites = match (&params.mod_type, cell_membership) {
        // m6A with cell membership: per-cell-type discovery
        (ModificationType::M6A { .. }, Some(membership)) => find_sites_with_celltype_stats(
            gff_record,
            params,
            &faidx_reader,
            chr,
            strand,
            membership,
        )?,
        // m6A without membership or AtoI: bulk discovery
        _ => find_sites_with_bulk_stats(gff_record, params, &faidx_reader, chr, strand)?,
    };

    if !candidate_sites.is_empty() {
        arc_gene_sites.insert(gene_id, candidate_sites);
    }

    Ok(())
}

/// Find conversion sites using bulk/marginal statistics (no cell type info).
fn find_sites_with_bulk_stats(
    gff_record: &GffRecord,
    params: &ConversionParams,
    faidx_reader: &faidx::Reader,
    chr: &str,
    strand: &Strand,
) -> anyhow::Result<Vec<ConversionSite>> {
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

    if positions.len() < params.min_length_for_testing() {
        return Ok(Vec::new());
    }

    let mut sifter = params.create_sifter(faidx_reader, chr, positions.len());

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

    let forward = matches!(strand, Strand::Forward);
    sifter.scan(&positions, wt_freq, Some(mut_freq), forward);

    let mut candidate_sites = sifter.candidate_sites;
    candidate_sites.sort();
    candidate_sites.dedup();
    Ok(candidate_sites)
}

/// Find conversion sites using per-cell-type statistics (m6A only).
fn find_sites_with_celltype_stats(
    gff_record: &GffRecord,
    params: &ConversionParams,
    faidx_reader: &faidx::Reader,
    chr: &str,
    strand: &Strand,
    membership: &CellMembership,
) -> anyhow::Result<Vec<ConversionSite>> {
    let cell_types = membership.cell_types();

    if cell_types.is_empty() {
        return Ok(Vec::new());
    }

    // Collect bulk frequencies from mut BAM files (background/null distribution)
    let mut mut_base_freq_map = DnaBaseFreqMap::new();
    for mut_file in &params.mut_bam_files {
        mut_base_freq_map.update_from_gene(
            mut_file,
            gff_record,
            &params.gene_barcode_tag,
            params.include_missing_barcode,
        )?;
    }
    let mut_freq = mut_base_freq_map.marginal_frequency_map();

    // Find sites for each cell type
    let mut all_candidate_sites = Vec::new();

    for cell_type in &cell_types {
        // Create frequency map for this cell type only
        let mut wt_base_freq_map =
            DnaBaseFreqMap::new_for_celltype(&params.cell_barcode_tag, membership, cell_type);

        for wt_file in &params.wt_bam_files {
            wt_base_freq_map.update_from_gene(
                wt_file,
                gff_record,
                &params.gene_barcode_tag,
                params.include_missing_barcode,
            )?;
        }

        let positions = wt_base_freq_map.sorted_positions();

        if positions.len() < params.min_length_for_testing() {
            continue;
        }

        let wt_freq = match wt_base_freq_map.marginal_frequency_map() {
            Some(freq) => freq,
            None => continue,
        };

        let mut sifter = params.create_sifter(faidx_reader, chr, positions.len());

        let forward = matches!(strand, Strand::Forward);
        sifter.scan(&positions, wt_freq, mut_freq, forward);

        all_candidate_sites.extend(sifter.candidate_sites);
    }

    all_candidate_sites.sort();
    all_candidate_sites.dedup();
    Ok(all_candidate_sites)
}

///////////////////////////////////////////
// SECOND PASS: Collect cell-level data  //
///////////////////////////////////////////

/// Gather conversion statistics for all sites in all genes from a single BAM file.
pub fn gather_conversion_stats(
    gene_sites: &HashMap<GeneId, Vec<ConversionSite>>,
    params: &ConversionParams,
    gff_map: &GffRecordMap,
    bam_file: &str,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let ndata = gene_sites.iter().map(|x| x.value().len()).sum::<usize>();
    let arc_ret = Arc::new(Mutex::new(Vec::with_capacity(ndata)));

    gene_sites
        .into_iter()
        .par_bridge()
        .progress_count(gene_sites.len() as u64)
        .try_for_each(|gs| -> anyhow::Result<()> {
            let gene = gs.key();
            let sites = gs.value();

            if let Some(gff) = gff_map.get(gene) {
                let stats =
                    collect_gene_conversion_stats(params, bam_file, &gff, sites, cell_membership)?;
                arc_ret.lock().expect("lock").extend(stats);
            }
            Ok(())
        })?;

    Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()
        .map_err(Into::into)
}

fn collect_gene_conversion_stats(
    params: &ConversionParams,
    bam_file: &str,
    gff_record: &GffRecord,
    sites: &[ConversionSite],
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut all_stats = Vec::new();

    for site in sites {
        let stats = estimate_conversion_stat(params, bam_file, gff_record, site, cell_membership)?;
        all_stats.extend(stats);
    }

    Ok(all_stats)
}

/// Estimate conversion statistics for a single site.
///
/// For M6A sites: sets anchor via `stat_map.set_anchor_position()`, reads 2 positions
/// (m6a + conversion), uses C->T (fwd) or G->A (rev).
/// For AtoI sites: no anchor, reads 1 position, uses A->G (fwd) or T->C (rev).
fn estimate_conversion_stat(
    params: &ConversionParams,
    bam_file: &str,
    gff_record: &GffRecord,
    site: &ConversionSite,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut stat_map =
        DnaBaseFreqMap::new_with_cell_barcode(&params.cell_barcode_tag, cell_membership);

    let mut gff = gff_record.clone();
    let gene = gff.gene_id.clone();
    let chr = gff.seqname.as_ref();
    let strand = gff.strand;

    match site {
        ConversionSite::M6A {
            m6a_pos,
            conversion_pos,
            ..
        } => {
            let lb = (*m6a_pos).min(*conversion_pos);
            let ub = (*conversion_pos).max(*m6a_pos);

            gff.start = (lb - BAM_READ_PADDING).max(0);
            gff.stop = ub + BAM_READ_PADDING;
            stat_map.update_from_gene(
                bam_file,
                &gff,
                &params.gene_barcode_tag,
                params.include_missing_barcode,
            )?;

            let (unmutated_base, mutated_base) = match strand {
                Strand::Forward => (Dna::C, Dna::T),
                Strand::Backward => (Dna::G, Dna::A),
            };

            // Set the anchor position for m6A
            let anchor_base = match strand {
                Strand::Forward => Dna::A,
                Strand::Backward => Dna::T,
            };
            stat_map.set_anchor_position(*m6a_pos, anchor_base);

            let methylation_stat = stat_map.stratified_frequency_at(*conversion_pos);

            let Some(meth_stat) = methylation_stat else {
                return Ok(Vec::new());
            };

            let (start, stop) = bin_position_kb(*conversion_pos, params.resolution_kb);

            let stats = meth_stat
                .iter()
                .filter_map(|(cb, counts)| {
                    let methylated = counts.get(Some(&mutated_base));
                    let unmethylated = counts.get(Some(&unmutated_base));

                    if (params.include_missing_barcode || cb != &CellBarcode::Missing)
                        && methylated > 0
                    {
                        Some((
                            cb.clone(),
                            BedWithGene {
                                chr: chr.into(),
                                start,
                                stop,
                                gene: gene.clone(),
                                strand,
                            },
                            MethylationData {
                                methylated,
                                unmethylated,
                                site_pos: *m6a_pos,
                            },
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            Ok(stats)
        }
        ConversionSite::AtoI { editing_pos, .. } => {
            gff.start = (*editing_pos - BAM_READ_PADDING).max(0);
            gff.stop = *editing_pos + BAM_READ_PADDING;
            stat_map.update_from_gene(
                bam_file,
                &gff,
                &params.gene_barcode_tag,
                params.include_missing_barcode,
            )?;

            // A-to-I: fwd = A->G, rev = T->C
            let (unmutated_base, mutated_base) = match strand {
                Strand::Forward => (Dna::A, Dna::G),
                Strand::Backward => (Dna::T, Dna::C),
            };

            let methylation_stat = stat_map.stratified_frequency_at(*editing_pos);

            let Some(meth_stat) = methylation_stat else {
                return Ok(Vec::new());
            };

            let (start, stop) = bin_position_kb(*editing_pos, params.resolution_kb);

            let stats = meth_stat
                .iter()
                .filter_map(|(cb, counts)| {
                    let edited = counts.get(Some(&mutated_base));
                    let unedited = counts.get(Some(&unmutated_base));

                    if (params.include_missing_barcode || cb != &CellBarcode::Missing) && edited > 0
                    {
                        Some((
                            cb.clone(),
                            BedWithGene {
                                chr: chr.into(),
                                start,
                                stop,
                                gene: gene.clone(),
                                strand,
                            },
                            MethylationData {
                                methylated: edited,
                                unmethylated: unedited,
                                site_pos: *editing_pos,
                            },
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            Ok(stats)
        }
    }
}

///////////////////////////////////////////
// Backend output                        //
///////////////////////////////////////////

/// Unified backend output for conversion sites (m6A or A-to-I).
///
/// Uses `site_suffix()` from ConversionSite for feature IDs.
/// Gene-level output is only supported for m6A.
pub fn process_all_bam_files_to_backend(
    params: &ConversionParams,
    gene_sites: &HashMap<GeneId, Vec<ConversionSite>>,
    gff_map: &GffRecordMap,
    gene_level_output: bool,
    output_null_data: bool,
) -> anyhow::Result<()> {
    let membership = params.load_membership()?;

    let gene_key = create_gene_key_function(gff_map);

    // Determine site suffix from modification type
    let suffix = match params.mod_type {
        ModificationType::M6A { .. } => "m6A",
        ModificationType::AtoI => "A2I",
    };

    let site_key = |x: &BedWithGene| -> Box<str> {
        let gene_part = gene_key(x);
        format!("{}_{}_{}_{}/{}", gene_part, x.chr, x.start, x.stop, suffix).into_boxed_str()
    };
    let take_value = params.value_extractor();
    let cutoffs = params.qc_cutoffs();

    let mut genes = HashSet::<Box<str>>::default();
    let mut sites = HashSet::<Box<str>>::default();
    let mut gene_data_files: Vec<Box<str>> = vec![];
    let mut site_data_files: Vec<Box<str>> = vec![];
    let mut null_gene_data_files: Vec<Box<str>> = vec![];
    let mut null_site_data_files: Vec<Box<str>> = vec![];

    let wt_batch_names = uniq_batch_names(&params.wt_bam_files)?;

    for (bam_file, batch_name) in params.wt_bam_files.iter().zip(wt_batch_names) {
        process_bam_to_backend(
            bam_file,
            &batch_name,
            gene_sites,
            params,
            gff_map,
            &gene_key,
            &site_key,
            &take_value,
            &cutoffs,
            &mut genes,
            &mut sites,
            &mut gene_data_files,
            &mut site_data_files,
            membership.as_ref(),
            gene_level_output,
        )?;
    }

    if output_null_data {
        info!("output null data");
        let mut_batch_names = uniq_batch_names(&params.mut_bam_files)?;

        for (bam_file, batch_name) in params.mut_bam_files.iter().zip(mut_batch_names) {
            process_bam_to_backend(
                bam_file,
                &batch_name,
                gene_sites,
                params,
                gff_map,
                &gene_key,
                &site_key,
                &take_value,
                &cutoffs,
                &mut genes,
                &mut sites,
                &mut null_gene_data_files,
                &mut null_site_data_files,
                membership.as_ref(),
                gene_level_output,
            )?;
        }
    }

    // Log match statistics if membership was used
    if let Some(ref m) = membership {
        let (matched, total) = m.match_stats();
        info!(
            "Cell barcode matching: {}/{} BAM barcodes matched membership ({:.1}%)",
            matched,
            total,
            if total > 0 {
                100.0 * matched as f32 / total as f32
            } else {
                0.0
            }
        );
    }

    // Reorder rows to ensure consistency across files
    reorder_all_matrices(
        params,
        genes,
        sites,
        gene_data_files,
        site_data_files,
        null_gene_data_files,
        null_site_data_files,
        output_null_data,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_bam_to_backend(
    bam_file: &str,
    batch_name: &str,
    gene_sites: &HashMap<GeneId, Vec<ConversionSite>>,
    params: &ConversionParams,
    gff_map: &GffRecordMap,
    gene_key: &(impl Fn(&BedWithGene) -> Box<str> + Send + Sync),
    site_key: &(impl Fn(&BedWithGene) -> Box<str> + Send + Sync),
    take_value: &(impl Fn(&MethylationData) -> f32 + Send + Sync),
    cutoffs: &SqueezeCutoffs,
    genes: &mut HashSet<Box<str>>,
    sites: &mut HashSet<Box<str>>,
    gene_data_files: &mut Vec<Box<str>>,
    site_data_files: &mut Vec<Box<str>>,
    cell_membership: Option<&CellMembership>,
    gene_level_output: bool,
) -> anyhow::Result<()> {
    info!(
        "collecting data over {} sites from {} ...",
        gene_sites.iter().map(|x| x.value().len()).sum::<usize>(),
        bam_file
    );

    let stats = gather_conversion_stats(gene_sites, params, gff_map, bam_file, cell_membership)?;

    info!(
        "aggregating the '{}' triplets over {} stats...",
        params.output_value_type,
        stats.len()
    );

    if gene_level_output {
        let gene_data_file = params.backend_file_path(batch_name);
        let triplets = summarize_stats(&stats, gene_key, take_value);
        let data = triplets.to_backend(&gene_data_file)?;
        data.qc(cutoffs.clone())?;
        genes.extend(data.row_names()?);
        info!("created gene-level data: {}", &gene_data_file);
        gene_data_files.push(gene_data_file);
    } else {
        let site_data_file = params.backend_file_path(batch_name);
        let triplets = summarize_stats(&stats, site_key, take_value);
        let data = triplets.to_backend(&site_data_file)?;
        data.qc(cutoffs.clone())?;
        sites.extend(data.row_names()?);
        info!("created site-level data: {}", &site_data_file);
        site_data_files.push(site_data_file);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reorder_all_matrices(
    params: &ConversionParams,
    genes: HashSet<Box<str>>,
    sites: HashSet<Box<str>>,
    gene_data_files: Vec<Box<str>>,
    site_data_files: Vec<Box<str>>,
    null_gene_data_files: Vec<Box<str>>,
    null_site_data_files: Vec<Box<str>>,
    output_null_data: bool,
) -> anyhow::Result<()> {
    let mut genes_sorted: Vec<_> = genes.into_iter().collect();
    genes_sorted.sort();

    let backend = &params.backend;

    for data_file in gene_data_files {
        open_sparse_matrix(&data_file, backend)?.reorder_rows(&genes_sorted)?;
    }

    if output_null_data {
        for data_file in null_gene_data_files {
            open_sparse_matrix(&data_file, backend)?.reorder_rows(&genes_sorted)?;
        }
    }

    let mut sites_sorted: Vec<_> = sites.into_iter().collect();
    sites_sorted.sort();

    for data_file in site_data_files {
        open_sparse_matrix(&data_file, backend)?.reorder_rows(&sites_sorted)?;
    }

    if output_null_data {
        for data_file in null_site_data_files {
            open_sparse_matrix(&data_file, backend)?.reorder_rows(&sites_sorted)?;
        }
    }

    Ok(())
}

///////////////////////////////////////////
// BED output (m6A only)                 //
///////////////////////////////////////////

pub fn process_all_bam_files_to_bed(
    params: &ConversionParams,
    gene_sites: &HashMap<GeneId, Vec<ConversionSite>>,
    gff_map: &GffRecordMap,
    output_cell_types: bool,
    output_null_data: bool,
) -> anyhow::Result<()> {
    let membership = params.load_membership()?;

    let wt_batch_names = uniq_batch_names(&params.wt_bam_files)?;

    for (bam_file, batch_name) in params.wt_bam_files.iter().zip(wt_batch_names) {
        let mut stats =
            gather_conversion_stats(gene_sites, params, gff_map, bam_file, membership.as_ref())?;
        let bed_path = format!("{}/{}.bed.gz", &params.output, batch_name);
        write_bed(
            &mut stats,
            gff_map,
            &bed_path,
            membership.as_ref(),
            output_cell_types,
        )?;
    }

    if output_null_data {
        info!("output null data");
        let mut_batch_names = uniq_batch_names(&params.mut_bam_files)?;

        for (bam_file, batch_name) in params.mut_bam_files.iter().zip(mut_batch_names) {
            let mut stats = gather_conversion_stats(
                gene_sites,
                params,
                gff_map,
                bam_file,
                membership.as_ref(),
            )?;
            let bed_path = format!("{}/{}.bed.gz", &params.output, batch_name);
            write_bed(
                &mut stats,
                gff_map,
                &bed_path,
                membership.as_ref(),
                output_cell_types,
            )?;
        }
    }

    // Log match statistics if membership was used
    if let Some(ref m) = membership {
        let (matched, total) = m.match_stats();
        info!(
            "Cell barcode matching: {}/{} BAM barcodes matched membership ({:.1}%)",
            matched,
            total,
            if total > 0 {
                100.0 * matched as f32 / total as f32
            } else {
                0.0
            }
        );
    }

    Ok(())
}

pub fn write_bed(
    stats: &mut [(CellBarcode, BedWithGene, MethylationData)],
    gff_map: &GffRecordMap,
    file_path: &str,
    cell_membership: Option<&CellMembership>,
    output_cell_types: bool,
) -> anyhow::Result<()> {
    use rust_htslib::bgzf::Writer as BWriter;
    use std::io::Write;

    stats.par_sort_by(|a, b| a.1.cmp(&b.1));

    let lines: Vec<_> = stats
        .iter()
        .map(|(cb, bg, data)| {
            let gene_string = gff_map
                .get(&bg.gene)
                .map(|gff| match gff.gene_name {
                    GeneSymbol::Symbol(x) => format!("{}_{}", &bg.gene, x),
                    GeneSymbol::Missing => format!("{}", &bg.gene),
                })
                .unwrap_or_else(|| format!("{}", &bg.gene));

            if output_cell_types {
                if let Some(membership) = cell_membership {
                    let cell_type = membership
                        .matches_barcode(cb)
                        .unwrap_or_else(|| "unknown".into());
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        bg.chr,
                        bg.start,
                        bg.stop,
                        bg.strand,
                        gene_string,
                        data.methylated,
                        data.unmethylated,
                        cb,
                        data.site_pos,
                        cell_type
                    )
                    .into_boxed_str()
                } else {
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tunknown",
                        bg.chr,
                        bg.start,
                        bg.stop,
                        bg.strand,
                        gene_string,
                        data.methylated,
                        data.unmethylated,
                        cb,
                        data.site_pos
                    )
                    .into_boxed_str()
                }
            } else {
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    bg.chr,
                    bg.start,
                    bg.stop,
                    bg.strand,
                    gene_string,
                    data.methylated,
                    data.unmethylated,
                    cb,
                    data.site_pos
                )
                .into_boxed_str()
            }
        })
        .collect();

    let header: &[u8] = if output_cell_types {
        b"#chr\tstart\tstop\tstrand\tgene\tmethylated\tunmethylated\tbarcode\tsite_pos\tcell_type\n"
    } else {
        b"#chr\tstart\tstop\tstrand\tgene\tmethylated\tunmethylated\tbarcode\tsite_pos\n"
    };

    let mut writer = BWriter::from_path(file_path)?;
    writer.write_all(header)?;
    for line in lines {
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;

    Ok(())
}

///////////////////////////////////////////
// Mixture model pipeline                //
///////////////////////////////////////////

/// Run per-gene 1D Gaussian mixture model on discovered sites and output results.
///
/// For each gene, collects cell-level observations from the second-pass stats,
/// fits a GMM via BIC model selection, and outputs:
/// - A sparse (cells x mixture_components) count matrix
///   with feature IDs like `GENE/m6A/0`, `GENE/A2I/1`
/// - A mixture_annotations.parquet file
pub fn run_mixture_model(
    params: &ConversionParams,
    gene_sites: &HashMap<GeneId, Vec<ConversionSite>>,
    gff_map: &GffRecordMap,
    mixture_params: &crate::editing::mixture::MixtureParams,
) -> anyhow::Result<()> {
    use crate::editing::mixture::{fit_gene_mixture, CellObservation, MixtureComponentAnnotation};

    let membership = params.load_membership()?;

    // Collect cell-level stats from all WT BAM files
    let mut all_stats: Vec<(CellBarcode, BedWithGene, MethylationData)> = Vec::new();
    for bam_file in &params.wt_bam_files {
        let stats =
            gather_conversion_stats(gene_sites, params, gff_map, bam_file, membership.as_ref())?;
        all_stats.extend(stats);
    }

    info!(
        "Mixture model: collected {} cell-level observations",
        all_stats.len()
    );

    if all_stats.is_empty() {
        info!("No observations for mixture model");
        return Ok(());
    }

    // Build a global cell index
    let mut unique_cells: Vec<CellBarcode> =
        all_stats.iter().map(|(cb, _, _)| cb.clone()).collect();
    unique_cells.sort();
    unique_cells.dedup();
    let cell_to_idx: fnv::FnvHashMap<CellBarcode, usize> = unique_cells
        .iter()
        .enumerate()
        .map(|(i, cb)| (cb.clone(), i))
        .collect();

    // Group observations by gene, converting to strand-aware relative position
    let mut gene_obs: fnv::FnvHashMap<GeneId, Vec<(usize, f32, usize)>> =
        fnv::FnvHashMap::default();
    for (cb, bed, meth) in &all_stats {
        let cell_idx = cell_to_idx[cb];
        let count = meth.methylated + meth.unmethylated;
        // Convert absolute site_pos to strand-aware relative position
        let rel_pos = if let Some(gff) = gff_map.get(&bed.gene) {
            let lb = (gff.start - 1).max(0); // GFF 1-based -> 0-based
            let ub = gff.stop;
            match gff.strand {
                Strand::Forward => (meth.site_pos - lb) as f32,
                Strand::Backward => (ub - meth.site_pos - 1) as f32,
            }
        } else {
            meth.site_pos as f32
        };
        gene_obs
            .entry(bed.gene.clone())
            .or_default()
            .push((cell_idx, rel_pos, count));
    }

    // Fit mixture per gene in parallel
    let gene_entries: Vec<_> = gene_obs.into_iter().collect();
    type Triplet = (CellBarcode, Box<str>, f32);
    let arc_triplets: Arc<Mutex<Vec<Triplet>>> = Arc::new(Mutex::new(Vec::new()));
    let arc_annotations: Arc<Mutex<Vec<MixtureComponentAnnotation>>> =
        Arc::new(Mutex::new(Vec::new()));
    let arc_pdui: Arc<Mutex<Vec<Triplet>>> = Arc::new(Mutex::new(Vec::new()));

    gene_entries.par_iter().for_each(|(gene_id, obs_list)| {
        let gene_name: Box<str> = gff_map
            .get(gene_id)
            .map(|gff| match &gff.gene_name {
                GeneSymbol::Symbol(s) => format!("{}_{}", gene_id, s),
                GeneSymbol::Missing => format!("{}", gene_id),
            })
            .unwrap_or_else(|| format!("{}", gene_id))
            .into();

        let gene_length = gff_map
            .get(gene_id)
            .map(|gff| (gff.stop - gff.start) as f32)
            .unwrap_or(1000.0);

        // Expand observations by count
        let mut cell_observations = Vec::new();
        for &(cell_idx, position, count) in obs_list {
            for _ in 0..count {
                cell_observations.push(CellObservation { cell_idx, position });
            }
        }

        if let Some(result) = fit_gene_mixture(&cell_observations, gene_length, mixture_params) {
            // Build component annotations
            let mut local_annotations = Vec::new();
            for (j, (&mu, &sigma)) in result
                .gmm
                .mus
                .iter()
                .zip(result.gmm.sigmas.iter())
                .enumerate()
            {
                let pi = result.gmm.weights[j + 1]; // skip noise
                if pi > 0.0 {
                    local_annotations.push(MixtureComponentAnnotation {
                        gene_name: gene_name.clone(),
                        component_idx: j,
                        mu,
                        sigma,
                        pi,
                    });
                }
            }

            // Build triplets: (cell_barcode, feature_id, count)
            // Feature IDs: GENE/m6A/0, GENE/A2I/1, etc.
            let mod_suffix = match params.mod_type {
                ModificationType::M6A { .. } => "m6A",
                ModificationType::AtoI => "A2I",
            };
            let mut local_triplets = Vec::new();
            for (cell_idx, component, count) in &result.cell_component_counts {
                if *component == 0 {
                    continue; // skip noise
                }
                let feature_id: Box<str> =
                    format!("{}/{}/{}", gene_name, mod_suffix, component - 1).into();
                let cb = &unique_cells[*cell_idx];
                local_triplets.push((cb.clone(), feature_id, *count as f32));
            }

            // Compute PDUI for genes with 2+ active components
            // Distal component = largest mu (furthest from gene start)
            if local_annotations.len() >= 2 {
                let distal_idx = local_annotations
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.mu.partial_cmp(&b.mu).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                // distal_component is the 1-based component index (0 = noise in GMM)
                let distal_component = local_annotations[distal_idx].component_idx + 1;

                // Per-cell: PDUI = distal_count / total_non_noise_count
                let mut cell_total: fnv::FnvHashMap<usize, usize> = fnv::FnvHashMap::default();
                let mut cell_distal: fnv::FnvHashMap<usize, usize> = fnv::FnvHashMap::default();
                for &(cell_idx, component, count) in &result.cell_component_counts {
                    if component == 0 {
                        continue;
                    }
                    *cell_total.entry(cell_idx).or_default() += count;
                    if component == distal_component {
                        *cell_distal.entry(cell_idx).or_default() += count;
                    }
                }

                let mut local_pdui = Vec::new();
                for (&cell_idx, &total) in &cell_total {
                    if total > 0 {
                        let distal = *cell_distal.get(&cell_idx).unwrap_or(&0);
                        let pdui = distal as f32 / total as f32;
                        let cb = &unique_cells[cell_idx];
                        local_pdui.push((cb.clone(), gene_name.clone(), pdui));
                    }
                }
                arc_pdui.lock().expect("lock").extend(local_pdui);
            }

            arc_triplets.lock().expect("lock").extend(local_triplets);
            arc_annotations
                .lock()
                .expect("lock")
                .extend(local_annotations);
        }
    });

    let triplets_data = Arc::try_unwrap(arc_triplets)
        .map_err(|_| anyhow::anyhow!("failed to unwrap triplets"))?
        .into_inner()?;

    let annotations = Arc::try_unwrap(arc_annotations)
        .map_err(|_| anyhow::anyhow!("failed to unwrap annotations"))?
        .into_inner()?;

    info!(
        "Mixture model: {} triplets, {} component annotations",
        triplets_data.len(),
        annotations.len()
    );

    if triplets_data.is_empty() {
        info!("No mixture results to output");
        return Ok(());
    }

    // Write sparse matrix
    let triplets = format_data_triplets(triplets_data);
    let suffix = match params.mod_type {
        ModificationType::M6A { .. } => "mixture_m6a",
        ModificationType::AtoI => "mixture_atoi",
    };
    let output_file = params.backend_file_path(suffix);
    let data = triplets.to_backend(&output_file)?;
    data.qc(params.qc_cutoffs())?;
    info!("Mixture model: created {}", &output_file);

    // Write annotations parquet
    if !annotations.is_empty() {
        write_mixture_annotations(
            &annotations,
            &format!("{}/mixture_annotations.parquet", &params.output),
        )?;
    }

    // Write PDUI sparse matrix (cells × genes)
    let pdui_data = Arc::try_unwrap(arc_pdui)
        .map_err(|_| anyhow::anyhow!("failed to unwrap pdui"))?
        .into_inner()?;

    if !pdui_data.is_empty() {
        info!("PDUI: {} cell-gene entries", pdui_data.len());
        let pdui_triplets = format_data_triplets(pdui_data);
        let pdui_suffix = match params.mod_type {
            ModificationType::M6A { .. } => "pdui_m6a",
            ModificationType::AtoI => "pdui_atoi",
        };
        let pdui_file = params.backend_file_path(pdui_suffix);
        let pdui_out = pdui_triplets.to_backend(&pdui_file)?;
        pdui_out.qc(params.qc_cutoffs())?;
        info!("PDUI: created {}", &pdui_file);
    }

    Ok(())
}

fn write_mixture_annotations(
    annotations: &[crate::editing::mixture::MixtureComponentAnnotation],
    path: &str,
) -> anyhow::Result<()> {
    use arrow::array::{ArrayRef, Float32Array, StringArray, UInt64Array};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let gene_names: Vec<&str> = annotations.iter().map(|a| a.gene_name.as_ref()).collect();
    let component_idxs: Vec<u64> = annotations.iter().map(|a| a.component_idx as u64).collect();
    let mus: Vec<f32> = annotations.iter().map(|a| a.mu).collect();
    let sigmas: Vec<f32> = annotations.iter().map(|a| a.sigma).collect();
    let pis: Vec<f32> = annotations.iter().map(|a| a.pi).collect();

    let schema = arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("gene_name", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("component_idx", arrow::datatypes::DataType::UInt64, false),
        arrow::datatypes::Field::new("mu", arrow::datatypes::DataType::Float32, false),
        arrow::datatypes::Field::new("sigma", arrow::datatypes::DataType::Float32, false),
        arrow::datatypes::Field::new("pi", arrow::datatypes::DataType::Float32, false),
    ]);

    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![
            std::sync::Arc::new(StringArray::from(gene_names)) as ArrayRef,
            std::sync::Arc::new(UInt64Array::from(component_idxs)) as ArrayRef,
            std::sync::Arc::new(Float32Array::from(mus)) as ArrayRef,
            std::sync::Arc::new(Float32Array::from(sigmas)) as ArrayRef,
            std::sync::Arc::new(Float32Array::from(pis)) as ArrayRef,
        ],
    )?;

    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    info!(
        "Wrote {} mixture annotations to {}",
        annotations.len(),
        path
    );
    Ok(())
}

impl ModificationType {
    /// Human-readable label for log messages
    fn label(&self) -> &'static str {
        match self {
            ModificationType::M6A { .. } => "m6A",
            ModificationType::AtoI => "A-to-I",
        }
    }
}
