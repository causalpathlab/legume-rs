use crate::common::*;
use crate::dartseq_sifter::*;
use crate::data::dna::Dna;
use crate::data::dna_stat_map::*;
use crate::data::dna_stat_traits::*;
use crate::data::gff::FeatureType as GffFeatureType;
use crate::data::gff::GeneType as GffGeneType;
use crate::data::methylation::*;
use crate::data::util_htslib::*;

use dashmap::DashMap as HashMap;
use rayon::ThreadPoolBuilder;
use std::sync::{Arc, Mutex};

#[derive(Args, Debug)]
pub struct DartSeqCountArgs {
    /// Observed (WT) `.bam` files where `C->U` (`C->T`) conversions happen
    #[arg(short = 'w', long = "wt", value_delimiter = ',', required = true)]
    wt_bam_files: Vec<Box<str>>,

    /// Control (MUT) `.bam` files where `C->U` (`C->T`) conversion is disrupted
    #[arg(short = 'm', long = "mut", value_delimiter = ',', required = true)]
    mut_bam_files: Vec<Box<str>>,

    /// Gene annotation (`GFF`) file
    #[arg(short = 'g', long = "gff", required = true)]
    gff_file: Box<str>,

    /// resolution (in kb)
    #[arg(short = 'r', long)]
    resolution_kb: Option<f32>,

    /// (10x) cell/sample barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "CB")]
    cell_barcode_tag: Box<str>,

    /// gene barcode tag. [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)
    #[arg(long, default_value = "GX")]
    gene_barcode_tag: Box<str>,

    /// minimum number of total reads per site
    #[arg(long, default_value_t = 3)]
    min_coverage: usize,

    /// minimum number of reads at `C->U` edit events
    #[arg(long, default_value_t = 3)]
    min_conversion: usize,

    /// maximum detection p-value cutoff
    #[arg(short, long, default_value_t = 0.05)]
    pvalue_cutoff: f64,

    /// selectively choose bam record type (gene, transcript, exon, utr)
    #[arg(long, value_enum)]
    record_type: Option<GffFeatureType>,

    /// gene type (protein_coding, pseudogene, lncRNA)
    #[arg(long, value_enum)]
    gene_type: Option<GffGeneType>,

    /// maximum number of threads
    #[arg(long, default_value_t = 16)]
    max_threads: usize,

    /// number of non-zero cutoff for rows/features
    #[arg(long, default_value_t = 10)]
    row_nnz_cutoff: usize,

    /// number of non-zero cutoff for columns/cells
    #[arg(long, default_value_t = 10)]
    column_nnz_cutoff: usize,

    /// output value type
    #[arg(short = 't', long, value_enum, default_value = "methylated")]
    output_value_type: MethFeatureType,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "hdf5")]
    backend: SparseIoBackend,

    /// include reads missing gene and cell barcode
    #[arg(long, default_value_t = false)]
    include_missing_barcode: bool,

    /// output mut signals
    #[arg(long, default_value_t = false)]
    output_mut: bool,

    /// output bed file
    #[arg(long, default_value_t = false)]
    output_bed_file: bool,

    /// output header for `data-beans` files
    #[arg(short, long, required = true)]
    output: Box<str>,
}

/// Count possibly methylated A positions in DART-seq bam files to
/// quantify m6A β values
///
pub fn run_count_dartseq(args: &DartSeqCountArgs) -> anyhow::Result<()> {
    if args.wt_bam_files.is_empty() || args.mut_bam_files.is_empty() {
        return Err(anyhow::anyhow!("need pairs of bam files"));
    }

    for x in args.wt_bam_files.iter() {
        info!("checking .bai file for {}...", x);
        check_bam_index(x, None)?;
    }

    for x in args.mut_bam_files.iter() {
        info!("checking .bai file for {}...", x);
        check_bam_index(x, None)?;
    }

    ThreadPoolBuilder::new()
        .num_threads(args.max_threads)
        .build_global()?;

    info!("parsing GFF file: {}", args.gff_file);

    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref(), args.record_type.as_ref())?;

    if let Some(gene_type) = args.gene_type.clone() {
        gff_map.subset(gene_type);
    }

    info!("found {} features", gff_map.len(),);

    if gff_map.is_empty() {
        info!("empty gff map");
        return Ok(());
    }

    /////////////////////////////////////
    // figure out potential edit sites //
    /////////////////////////////////////

    let njobs = gff_map.len();
    info!("Searching possible edit sites over {} blocks", njobs);

    let arc_gene_sites = Arc::new(HashMap::<GeneId, Vec<MethylatedSite>>::default());

    gff_map
        .records()
        .into_iter()
        .par_bridge()
        .progress_count(njobs as u64)
        .for_each(|rec| {
            find_methylated_sites_in_gene(rec, args, arc_gene_sites.clone())
                .expect("failed in find_methylated_sites")
        });

    if arc_gene_sites.is_empty() {
        info!("no sites found");
        return Ok(());
    }

    let gene_sites = Arc::try_unwrap(arc_gene_sites)
        .map_err(|_| anyhow::anyhow!("failed to release gene_sites"))?;

    let take_value = |dat: &MethylationData| -> f32 {
        match args.output_value_type {
            MethFeatureType::Beta => {
                let tot = (dat.methylated + dat.unmethylated) as f32;
                (dat.methylated as f32) / tot.max(1.)
            }
            MethFeatureType::Methylated => dat.methylated as f32,
            MethFeatureType::Unmethylated => dat.unmethylated as f32,
        }
    };

    // todo: motif output
    // collect_frequencies_nearby

    // let bed_file =
    //     |name: &str| -> Box<str> { format!("{}.{}.bed.gz", &args.output, name).into_boxed_str() };

    let gene_key = |x: &BedWithGene| -> GeneId { x.gene.clone() };
    let site_key = |x: &BedWithGene| -> BedWithGene { x.clone() };

    let backend = args.backend.clone();
    let backend_file = |name: &str| -> Box<str> {
        match backend {
            SparseIoBackend::HDF5 => format!("{}.{}.h5", &args.output, name),
            SparseIoBackend::Zarr => format!("{}.{}.zarr", &args.output, name),
        }
        .into_boxed_str()
    };

    let cutoffs = SqueezeCutoffs {
        row: args.row_nnz_cutoff,
        column: args.column_nnz_cutoff,
    };

    let mut genes = HashSet::<Box<str>>::default();
    let mut sites = HashSet::<Box<str>>::default();
    let mut gene_data_files = vec![];
    let mut site_data_files = vec![];

    for bam_file in args.wt_bam_files.iter() {
        ////////////////////////////
        // collect the statistics //
        ////////////////////////////

        info!(
            "collecting data over {} sites from {} ...",
            gene_sites.iter().map(|x| x.value().len()).sum::<usize>(),
            bam_file
        );

        let stats = gather_m6a_stats(&gene_sites, args, &gff_map, &bam_file)?;

        //////////////////////////////////
        // Aggregate them into triplets //
        //////////////////////////////////

        info!(
            "aggregating the '{}' triplets over {} stats...",
            args.output_value_type,
            stats.len()
        );

        let batch_name = basename(&bam_file)?;
        let gene_data_file = backend_file(&format!("gene_{}", batch_name));
        let triplets = summarize_stats(&stats, gene_key, take_value);
        let data = triplets.to_backend(&gene_data_file)?;
        data.qc(cutoffs.clone())?;
        genes.extend(data.row_names()?);
        info!("created gene-level data: {}", &gene_data_file);
        gene_data_files.push(gene_data_file);

        let site_data_file = backend_file(&format!("site_{}", batch_name));
        let triplets = summarize_stats(&stats, site_key, take_value);
        let data = triplets.to_backend(&site_data_file)?;
        data.qc(cutoffs.clone())?;
        sites.extend(data.row_names()?);
        info!("created site-level data: {}", &site_data_file);
        site_data_files.push(site_data_file);
    }

    let mut genes_sorted = genes.into_iter().collect::<Vec<_>>();
    genes_sorted.sort();

    for data_file in gene_data_files {
        open_sparse_matrix(&data_file, &backend)?.reorder_rows(&genes_sorted)?;
    }

    let mut sites_sorted = sites.into_iter().collect::<Vec<_>>();
    sites_sorted.sort();

    for data_file in site_data_files {
        open_sparse_matrix(&data_file, &backend)?.reorder_rows(&sites_sorted)?;
    }

    info!("done");
    Ok(())
}

////////////////////////////////////////////////
// Step 1: find possibly methylated positions //
////////////////////////////////////////////////

fn find_methylated_sites_in_gene(
    rec: &GffRecord,
    args: &DartSeqCountArgs,
    arc_gene_sites: Arc<HashMap<GeneId, Vec<MethylatedSite>>>,
) -> anyhow::Result<()> {
    let strand = rec.strand.clone();
    let gene_id = rec.gene_id.clone();

    // 1. sweep each pair bam files to find variable sites
    let mut wt_freq_map = DnaBaseFreqMap::new(&args.cell_barcode_tag);

    for wt_file in args.wt_bam_files.iter() {
        wt_freq_map.update_bam_by_gene(wt_file, rec, &args.gene_barcode_tag)?;
    }

    let positions = wt_freq_map.sorted_positions();

    if positions.len() >= 3 {
        // 2. find AC/T patterns: Using mutant statistics as null
        // distribution, it will keep possible C->U edit positions.
        let mut sifter = DartSeqSifter {
            min_coverage: args.min_coverage,
            min_conversion: args.min_conversion,
            max_pvalue_cutoff: args.pvalue_cutoff,
            candidate_sites: Vec::with_capacity(positions.len()),
        };

        // gather background frequency map
        let mut mut_freq_map = DnaBaseFreqMap::new(&args.cell_barcode_tag);

        for mut_file in args.mut_bam_files.iter() {
            mut_freq_map.update_bam_by_gene(mut_file, rec, &args.gene_barcode_tag)?;
        }

        let wt_freq = wt_freq_map.marginal_frequency_map();
        let mut_freq = mut_freq_map.marginal_frequency_map();

        match &strand {
            Strand::Forward => {
                sifter.forward_rac_sweep(&positions, &wt_freq, &mut_freq);
            }
            Strand::Backward => {
                sifter.backward_car_sweep(&positions, &wt_freq, &mut_freq);
            }
        };

        let mut ret = sifter.candidate_sites;
        if !ret.is_empty() {
            ret.sort();
            ret.dedup();
            arc_gene_sites.insert(gene_id, ret);
        }
    }
    Ok(())
}

///////////////////////////////////////////////////////////////////
// Step 2: revisit possible C2U positions and collect m6A sites, //
// locations and samples.                                        //
///////////////////////////////////////////////////////////////////

fn gather_m6a_stats(
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    args: &DartSeqCountArgs,
    gff_map: &GffRecordMap,
    bam_file: &str,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let ndata = gene_sites.iter().map(|x| x.value().len()).sum::<usize>();

    let arc_ret = Arc::new(Mutex::new(Vec::with_capacity(ndata)));

    gene_sites
        .into_iter()
        .par_bridge()
        .progress_count(gene_sites.len() as u64)
        .for_each(|gs| {
            let gene = gs.key();
            let sites = gs.value();

            let mut ret = Vec::with_capacity(sites.len());
            for x in sites {
                ret.extend(
                    estimate_m6a_stat(args, bam_file, &gff_map, gene, x)
                        .expect("failed to collect stat"),
                );
            }
            arc_ret.lock().expect("lock").extend(ret);
        });

    let stats = Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()?;
    Ok(stats)
}

fn collect_frequencies_nearby(gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>) {

    // todo
}

fn estimate_m6a_stat(
    args: &DartSeqCountArgs,
    bam_file: &str,
    gff_map: &GffRecordMap,
    gene: &GeneId,
    m6a_c2u: &MethylatedSite,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut stat_map = DnaBaseFreqMap::new(&args.cell_barcode_tag);
    let m6apos = m6a_c2u.m6a_pos;
    let c2upos = m6a_c2u.conversion_pos;
    let strand = &m6a_c2u.strand;

    let lb = m6apos.min(c2upos);
    let ub = c2upos.max(m6apos);

    // need to read bam files with the matching gene contexts
    let gff_record = gff_map
        .get(gene)
        .ok_or(anyhow::anyhow!("unable to find this gene"))?;

    let mut gff = gff_record.clone();
    gff.start = (lb - 1).max(0); // padding
    gff.stop = ub + 1; // padding
    stat_map.update_bam_by_gene(bam_file, &gff, &args.gene_barcode_tag)?;

    let chr = gff.seqname.as_ref();
    let methylation_stat = stat_map.stratified_frequency_at(c2upos);

    let unmethylated_base = match strand {
        Strand::Forward => Dna::C,
        Strand::Backward => Dna::G,
    };

    let methylated_base = match strand {
        Strand::Forward => Dna::T,
        Strand::Backward => Dna::A,
    };

    let (lb, ub) = if let Some(r) = args.resolution_kb {
        // report reduced kb resolution
        let r = (r * 1000.0) as usize;
        (
            ((m6apos as usize) / r * r + 1) as i64,
            ((m6apos as usize).div_ceil(r) * r) as i64,
        )
    } else {
        // report bp resolution
        (m6apos, m6apos + 1)
    };

    let mut ret = vec![];

    if let Some(meth_stat) = methylation_stat {
        for (cb, counts) in meth_stat {
            let methylated = counts.get(Some(&methylated_base));
            let unmethylated = counts.get(Some(&unmethylated_base));

            if (args.include_missing_barcode || cb != &CellBarcode::Missing) && methylated > 0 {
                ret.push((
                    cb.clone(),
                    BedWithGene {
                        chr: chr.into(),
                        start: lb,
                        stop: ub,
                        gene: gene.clone(),
                        strand: strand.clone(),
                    },
                    MethylationData {
                        methylated,
                        unmethylated,
                    },
                ));
            }
        }
    }

    Ok(ret)
}

//////////////////////////////////////////////////////////
// Step 3: repackaging them into desired output formats //
//////////////////////////////////////////////////////////

fn summarize_stats<F, V, T>(
    stats: &[(CellBarcode, BedWithGene, MethylationData)],
    feature_key_func: F,
    value_func: V,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> T + Send + Sync,
    T: Clone + Send + Sync + ToString + std::hash::Hash + std::cmp::Eq + std::cmp::Ord,
    V: Fn(&MethylationData) -> f32,
{
    let combined_data: HashMap<(CellBarcode, T), MethylationData> = HashMap::default();
    stats.par_iter().for_each(|(cb, k, dat)| {
        let key = (cb.clone(), feature_key_func(k));
        combined_data.entry(key).or_default().add_assign(dat);
    });

    let combined_data = combined_data
        .into_iter()
        .map(|((c, k), v)| (c, k, value_func(&v)))
        .collect::<Vec<_>>();

    format_data_triplets(combined_data)
}

// fn write_bed(
//     stats: &mut [(CellBarcode, BedWithGene, MethylationData)],
//     file_path: &str,
// ) -> anyhow::Result<()> {
//     use std::io::Write;

//     stats.par_sort_by(|a, b| a.1.cmp(&b.1));

//     let lines = stats
//         .into_iter()
//         .map(|(cb, bg, data)| {
//             format!(
//                 "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
//                 bg.chr,
//                 bg.start,
//                 bg.stop,
//                 bg.strand,
//                 bg.gene,
//                 data.methylated,
//                 data.unmethylated,
//                 cb
//             )
//             .into_boxed_str()
//         })
//         .collect::<Vec<_>>();

//     use rust_htslib::bgzf::Writer as BWriter;

//     let mut writer = BWriter::from_path(file_path)?;
//     for l in lines {
//         writer.write_all(l.as_bytes())?;
//         writer.write_all(b"\n")?;
//     }
//     writer.flush()?;
//     Ok(())
// }
