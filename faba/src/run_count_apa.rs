use crate::apa_mix::cell_assign::*;
use crate::apa_mix::em::*;
use crate::apa_mix::fragment::*;
use crate::apa_mix::likelihood::*;
use crate::apa_mix::site_discovery::*;
use crate::apa_mix::utr_region::*;
use crate::common::*;
use crate::data::poly_a_stat_map::{PolyASiteArgs, PolyASiteMap};
use crate::data::util_htslib::*;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use genomic_data::bed::BedWithGene;
use genomic_data::gff::FeatureType as GffFeatureType;
use genomic_data::gff::GeneType as GffGeneType;
use genomic_data::gff::{build_union_gene_model, read_gff_record_vec, GeneId, GffRecordMap};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::ThreadPoolBuilder;
use std::sync::{Arc, Mutex};

/// APA quantification method
#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum ApaMethod {
    /// Pileup-based poly-A site counting (fast, no EM)
    Simple,
    /// EM mixture model based on the SCAPE framework (Zhou et al., NAR 2022)
    #[default]
    Mixture,
}

#[derive(Args, Debug)]
#[command(after_long_help = "CITATION:\n  \
        The mixture model is based on the SCAPE framework:\n  \
        Zhou et al., \"SCAPE: a mixture model revealing single-cell\n  \
        polyadenylation diversity and cellular dynamics during cell\n  \
        differentiation and reprogramming\",\n  \
        Nucleic Acids Research, 50(11):e66, 2022.\n  \
        https://doi.org/10.1093/nar/gkac167")]
pub struct CountApaArgs {
    /// Input BAM file(s), comma-separated
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input BAM file(s)",
        long_help = "Comma-separated list of BAM files to quantify.\n\
                     Each file produces a separate output matrix."
    )]
    bam_files: Vec<Box<str>>,

    /// Gene annotation file (GFF/GTF)
    #[arg(
        short = 'g',
        long = "gff",
        help = "Gene annotation file (GFF/GTF)",
        long_help = "Path to gene annotation file in GFF/GTF format.\n\
                     Required for simple mode; in mixture mode, used to\n\
                     extract 3'-UTR regions unless --utr-bed is provided."
    )]
    gff_file: Option<Box<str>>,

    /// Cell barcode BAM tag
    #[arg(
        long,
        default_value = "CB",
        help = "Cell barcode BAM tag",
        long_help = "BAM tag for cell/sample barcode identification.\n\
                     Standard 10x Genomics tag is \"CB\"."
    )]
    cell_barcode_tag: Box<str>,

    /// Minimum soft-clipped poly(A) tail length
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum poly(A) tail length (bp)",
        long_help = "Minimum number of soft-clipped A/T bases required to\n\
                     call a read as a poly(A) junction read."
    )]
    polya_min_tail_length: usize,

    /// Maximum non-A/T bases allowed in poly(A) tail
    #[arg(
        long,
        default_value_t = 3,
        help = "Max non-A/T bases in poly(A) tail",
        long_help = "Maximum number of non-A (forward) or non-T (reverse)\n\
                     bases allowed in the soft-clipped tail."
    )]
    polya_max_non_a_or_t: usize,

    /// Internal priming check window size (bp)
    #[arg(
        long,
        default_value_t = 10,
        help = "Internal priming check window (bp)",
        long_help = "Window size in base pairs around the cleavage site to\n\
                     check for genomic A/T-rich stretches (internal priming)."
    )]
    polya_internal_prime_window: usize,

    /// A/T count threshold to flag internal priming
    #[arg(
        long,
        default_value_t = 7,
        help = "A/T count threshold for internal priming",
        long_help = "If the number of A/T bases in the internal priming\n\
                     window meets or exceeds this threshold, the site is\n\
                     flagged as likely internal priming and discarded."
    )]
    polya_internal_prime_count: usize,

    /// Minimum read coverage at a candidate site
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum read coverage per site",
        long_help = "Candidate poly(A) sites with fewer than this many\n\
                     supporting reads are discarded."
    )]
    min_coverage: usize,

    /// Maximum number of threads
    #[arg(
        long,
        default_value_t = 16,
        help = "Maximum number of threads",
        long_help = "Maximum number of threads for parallel processing.\n\
                     Capped by the number of available CPUs."
    )]
    max_threads: usize,

    /// Minimum non-zero entries per row (site) to keep
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum non-zeros per row (site)",
        long_help = "Sites with fewer than this many non-zero cells are\n\
                     removed from the output matrix."
    )]
    row_nnz_cutoff: usize,

    /// Minimum non-zero entries per column (cell) to keep
    #[arg(
        long,
        default_value_t = 1,
        help = "Minimum non-zeros per column (cell)",
        long_help = "Cells with fewer than this many non-zero sites are\n\
                     removed from the output matrix."
    )]
    column_nnz_cutoff: usize,

    /// Output directory
    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Directory for output files.\n\
                     One sparse matrix file per input BAM is created here."
    )]
    output: Box<str>,

    /// Sparse matrix output backend
    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix output backend",
        long_help = "File format for the output sparse matrix.\n\
                     Supported: zarr, hdf5."
    )]
    backend: SparseIoBackend,

    // --- Method selection ---
    /// APA quantification method
    #[arg(
        long,
        value_enum,
        default_value = "mixture",
        help = "APA quantification method",
        long_help = "Algorithm for poly(A) site quantification.\n\
                     \"simple\": pileup-based counting (fast, no EM).\n\
                     \"mixture\": EM mixture model based on SCAPE."
    )]
    method: ApaMethod,

    // --- Simple-mode args ---
    /// Gene barcode BAM tag (simple mode)
    #[arg(
        long,
        default_value = "GX",
        help = "Gene barcode BAM tag (simple mode)",
        long_help = "BAM tag for gene identification. Only used in simple mode.\n\
                     Standard 10x Genomics tag is \"GX\"."
    )]
    gene_barcode_tag: Box<str>,

    /// Bin resolution in bp for poly(A) site grouping (simple mode)
    #[arg(
        long,
        default_value_t = 10,
        help = "Bin resolution in bp (simple mode)",
        long_help = "Nearby poly(A) sites within this distance in base pairs\n\
                     are grouped into a single bin. Only used in simple mode."
    )]
    resolution_bp: usize,

    /// Include reads without cell barcode (simple mode)
    #[arg(
        long,
        default_value_t = false,
        help = "Include reads without barcode (simple mode)",
        long_help = "Include reads missing a cell barcode tag in the count.\n\
                     Only used in simple mode."
    )]
    include_missing_barcode: bool,

    /// GFF record type filter (simple mode)
    #[arg(
        long,
        value_enum,
        help = "GFF record type filter (simple mode)",
        long_help = "Filter GFF records by feature type. Only used in simple mode.\n\
                     Common values: gene, transcript, exon, utr."
    )]
    record_type: Option<GffFeatureType>,

    /// Gene biotype filter (simple mode)
    #[arg(
        long,
        value_enum,
        help = "Gene biotype filter (simple mode)",
        long_help = "Filter genes by biotype. Only used in simple mode.\n\
                     Common values: protein_coding, pseudogene, lncRNA."
    )]
    gene_type: Option<GffGeneType>,

    // --- Mixture-mode (SCAPE) args ---
    /// 3'-UTR regions BED file (mixture mode)
    #[arg(
        short = 'u',
        long = "utr-bed",
        help = "3'-UTR regions BED file (mixture mode)",
        long_help = "BED file defining 3'-UTR regions. Alternative to --gff\n\
                     for mixture mode. Each row should be a UTR interval."
    )]
    utr_bed: Option<Box<str>>,

    /// Minimum 3'-UTR length in bp (mixture mode)
    #[arg(
        long,
        default_value_t = 200,
        help = "Minimum 3'-UTR length in bp (mixture mode)",
        long_help = "UTRs shorter than this are skipped. Only used in mixture mode."
    )]
    min_utr_length: usize,

    /// Pre-identified pA sites BED file (mixture mode)
    #[arg(
        long,
        help = "Pre-identified pA sites BED (mixture mode)",
        long_help = "BED file of known poly(A) sites. When provided, skips\n\
                     de novo site discovery. Only used in mixture mode."
    )]
    pre_sites: Option<Box<str>>,

    /// UMI BAM tag (mixture mode)
    #[arg(
        long,
        default_value = "UB",
        help = "UMI BAM tag (mixture mode)",
        long_help = "BAM tag for unique molecular identifiers.\n\
                     Standard 10x Genomics tag is \"UB\".\n\
                     Only used in mixture mode."
    )]
    umi_tag: Box<str>,

    /// Expected fragment length mean (mixture mode)
    #[arg(
        long,
        default_value_t = 300.0,
        help = "Fragment length mean, mu_f (mixture mode)",
        long_help = "Expected mean fragment length in base pairs (mu_f in SCAPE).\n\
                     Only used in mixture mode."
    )]
    mu_f: f64,

    /// Fragment length standard deviation (mixture mode)
    #[arg(
        long,
        default_value_t = 50.0,
        help = "Fragment length s.d., sigma_f (mixture mode)",
        long_help = "Expected fragment length standard deviation (sigma_f in SCAPE).\n\
                     Only used in mixture mode."
    )]
    sigma_f: f64,

    /// pA site position step size in bp (mixture mode)
    #[arg(
        long,
        default_value_t = 10,
        help = "pA site enumeration step (bp, mixture mode)",
        long_help = "Step size in base pairs for enumerating candidate pA site\n\
                     positions along each UTR. Only used in mixture mode."
    )]
    theta_step: usize,

    /// Maximum pA site dispersion (mixture mode)
    #[arg(
        long,
        default_value_t = 70.0,
        help = "Max pA dispersion beta (mixture mode)",
        long_help = "Upper bound on the dispersion parameter (beta) for each\n\
                     poly(A) site component. Only used in mixture mode."
    )]
    max_beta: f64,

    /// Minimum pA site dispersion (mixture mode)
    #[arg(
        long,
        default_value_t = 10.0,
        help = "Min pA dispersion beta (mixture mode)",
        long_help = "Lower bound on the dispersion parameter (beta) for each\n\
                     poly(A) site component. Only used in mixture mode."
    )]
    min_beta: f64,

    /// Minimum component weight (mixture mode)
    #[arg(
        long,
        default_value_t = 0.01,
        help = "Min component weight (mixture mode)",
        long_help = "Components with weight below this threshold are pruned\n\
                     during EM iterations. Only used in mixture mode."
    )]
    min_ws: f64,

    /// Minimum fragments per UTR (mixture mode)
    #[arg(
        long,
        default_value_t = 50,
        help = "Min fragments per UTR (mixture mode)",
        long_help = "UTRs with fewer than this many fragments are skipped.\n\
                     Only used in mixture mode."
    )]
    min_fragments: usize,

    /// Merge distance for nearby pA sites in bp (mixture mode)
    #[arg(
        long,
        default_value_t = 50.0,
        help = "pA site merge distance (bp, mixture mode)",
        long_help = "Candidate pA sites within this distance are merged into\n\
                     a single site. Only used in mixture mode."
    )]
    merge_distance: f64,
}

impl CountApaArgs {
    fn lik_params(&self) -> LikelihoodParams {
        LikelihoodParams {
            mu_f: self.mu_f,
            sigma_f: self.sigma_f,
            theta_step: self.theta_step,
            ..Default::default()
        }
    }

    fn em_params(&self) -> EmParams {
        EmParams {
            min_weight: self.min_ws,
            ..Default::default()
        }
    }

    fn backend_file_path(&self, name: &str) -> Box<str> {
        match self.backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &self.output, name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &self.output, name),
        }
        .into_boxed_str()
    }

    fn qc_cutoffs(&self) -> SqueezeCutoffs {
        SqueezeCutoffs {
            row: self.row_nnz_cutoff,
            column: self.column_nnz_cutoff,
        }
    }

    fn polya_site_args(&self) -> PolyASiteArgs {
        PolyASiteArgs {
            min_tail_length: self.polya_min_tail_length,
            max_non_a_or_t_bases: self.polya_max_non_a_or_t,
            internal_prime_in: self.polya_internal_prime_window,
            internal_prime_a_or_t_count: self.polya_internal_prime_count,
        }
    }
}

/// Main entry point for `count-apa`.
pub fn run_count_apa(args: &CountApaArgs) -> anyhow::Result<()> {
    mkdir(&args.output)?;

    let max_threads = num_cpus::get().min(args.max_threads);
    ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global()
        .ok();
    info!("will use {} threads", rayon::current_num_threads());

    if args.bam_files.is_empty() {
        return Err(anyhow::anyhow!("need at least one BAM file"));
    }

    for bam_file in &args.bam_files {
        info!("checking .bai file for {}...", bam_file);
        check_bam_index(bam_file, None)?;
    }

    match args.method {
        ApaMethod::Simple => run_simple(args),
        ApaMethod::Mixture => run_mixture(args),
    }
}

// ─────────────────────────────────────────────────────────
// Simple mode (pileup-based, migrated from run_polya_count)
// ─────────────────────────────────────────────────────────

fn run_simple(args: &CountApaArgs) -> anyhow::Result<()> {
    let gff_file = args
        .gff_file
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--gff is required for simple mode"))?;

    info!("parsing GFF file: {}", gff_file);
    let mut gff_map = GffRecordMap::from(gff_file)?;

    if let Some(gene_type) = args.gene_type.clone() {
        gff_map.subset(gene_type);
    }

    info!("found {} genes", gff_map.len());
    if gff_map.is_empty() {
        info!("empty gff map");
        return Ok(());
    }

    // FIRST PASS: identify poly-A sites
    let gene_sites = find_all_polya_sites(&gff_map, args)?;
    if gene_sites.is_empty() {
        info!("no poly-A sites found");
        return Ok(());
    }

    let ndata: usize = gene_sites.iter().map(|x| x.value().len()).sum();
    info!("Found {} poly-A sites", ndata);

    // SECOND PASS: collect cell-level counts at sites
    let site_key = |x: &BedWithGene, gff_map: &GffRecordMap| -> Box<str> {
        let gene_part = gff_map
            .get(&x.gene)
            .map(|gff| format!("{}_{}", gff.gene_id, gff.gene_name))
            .unwrap_or_else(|| format!("{}", x.gene));
        format!("{}_{}_{}_{}/pA", gene_part, x.chr, x.start, x.stop).into_boxed_str()
    };

    let cutoffs = args.qc_cutoffs();
    let batch_names = uniq_batch_names(&args.bam_files)?;

    for (bam_file, batch_name) in args.bam_files.iter().zip(batch_names) {
        process_simple_bam(
            bam_file,
            &batch_name,
            &gene_sites,
            args,
            &gff_map,
            &site_key,
            &cutoffs,
        )?;
    }

    info!("done");
    Ok(())
}

/// Bin a position based on resolution
#[inline]
fn bin_position(position: i64, resolution: usize) -> (i64, i64) {
    let start = ((position as usize) / resolution * resolution) as i64;
    let stop = start + resolution as i64;
    (start, stop)
}

fn find_all_polya_sites(
    gff_map: &GffRecordMap,
    args: &CountApaArgs,
) -> anyhow::Result<DashMap<GeneId, Vec<i64>>> {
    let njobs = gff_map.len();
    info!("Searching poly-A sites over {} genes", njobs);

    let arc_gene_sites = Arc::new(DashMap::<GeneId, Vec<i64>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_count(njobs as u64)
        .try_for_each(|rec| -> anyhow::Result<()> {
            find_polya_sites_in_gene(rec, args, arc_gene_sites.clone())
        })?;

    Arc::try_unwrap(arc_gene_sites).map_err(|_| anyhow::anyhow!("failed to release gene_sites"))
}

fn find_polya_sites_in_gene(
    gff_record: &GffRecord,
    args: &CountApaArgs,
    arc_gene_sites: Arc<DashMap<GeneId, Vec<i64>>>,
) -> anyhow::Result<()> {
    let mut polya_map = PolyASiteMap::new(args.polya_site_args());

    for bam_file in &args.bam_files {
        polya_map.update_from_gene(
            bam_file,
            gff_record,
            &args.gene_barcode_tag,
            args.include_missing_barcode,
        )?;
    }

    let filtered_positions: Vec<i64> = polya_map
        .positions_with_counts()
        .into_iter()
        .filter(|(_, count)| *count >= args.min_coverage)
        .map(|(pos, _)| pos)
        .collect();

    if !filtered_positions.is_empty() {
        arc_gene_sites.insert(gff_record.gene_id.clone(), filtered_positions);
    }

    Ok(())
}

fn process_simple_bam(
    bam_file: &str,
    batch_name: &str,
    gene_sites: &DashMap<GeneId, Vec<i64>>,
    args: &CountApaArgs,
    gff_map: &GffRecordMap,
    site_key: &(impl Fn(&BedWithGene, &GffRecordMap) -> Box<str> + Send + Sync),
    cutoffs: &SqueezeCutoffs,
) -> anyhow::Result<()> {
    info!(
        "collecting cell-level data over {} sites from {} ...",
        gene_sites.iter().map(|x| x.value().len()).sum::<usize>(),
        bam_file
    );

    let stats = gather_polya_stats(gene_sites, args, gff_map, bam_file)?;
    info!("collected {} cell-level poly-A counts", stats.len());

    let site_data_file = args.backend_file_path(batch_name);
    let triplets = summarize_simple_stats(&stats, |bed| site_key(bed, gff_map));
    let data = triplets.to_backend(&site_data_file)?;
    data.qc(cutoffs.clone())?;
    info!("created data backend: {}", &site_data_file);

    Ok(())
}

fn gather_polya_stats(
    gene_sites: &DashMap<GeneId, Vec<i64>>,
    args: &CountApaArgs,
    gff_map: &GffRecordMap,
    bam_file: &str,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, usize)>> {
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
                let stats = collect_gene_stats(args, bam_file, &gff, gene, sites)?;
                arc_ret.lock().expect("lock").extend(stats);
            }
            Ok(())
        })?;

    Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()
        .map_err(Into::into)
}

fn collect_gene_stats(
    args: &CountApaArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    gene_id: &GeneId,
    positions: &[i64],
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, usize)>> {
    let mut all_stats = Vec::new();

    for &position in positions {
        let stats = collect_polya_counts_at_site(args, bam_file, gff_record, gene_id, position)?;
        all_stats.extend(stats);
    }

    Ok(all_stats)
}

fn collect_polya_counts_at_site(
    args: &CountApaArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    gene_id: &GeneId,
    position: i64,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, usize)>> {
    const PADDING: i64 = 100;

    let mut polya_map =
        PolyASiteMap::new_with_cell_barcode(args.polya_site_args(), &args.cell_barcode_tag);

    let mut gff = gff_record.clone();
    gff.start = (position - PADDING).max(0);
    gff.stop = position + PADDING;

    polya_map.update_from_gene(
        bam_file,
        &gff,
        &args.gene_barcode_tag,
        args.include_missing_barcode,
    )?;

    let Some(cell_counts) = polya_map.get_cell_counts_at(position) else {
        return Ok(Vec::new());
    };

    let (start, stop) = bin_position(position, args.resolution_bp);
    let chr = gff_record.seqname.as_ref();
    let strand = &gff_record.strand;

    let stats = cell_counts
        .iter()
        .filter(|(cb, _)| args.include_missing_barcode || *cb != &CellBarcode::Missing)
        .map(|(cell_barcode, count)| {
            (
                cell_barcode.clone(),
                BedWithGene {
                    chr: chr.into(),
                    start,
                    stop,
                    gene: gene_id.clone(),
                    strand: *strand,
                },
                *count,
            )
        })
        .collect();

    Ok(stats)
}

fn summarize_simple_stats<F, T>(
    stats: &[(CellBarcode, BedWithGene, usize)],
    feature_key_func: F,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> T + Send + Sync,
    T: Clone + Send + Sync + ToString + std::hash::Hash + std::cmp::Eq + std::cmp::Ord,
{
    let combined_data: DashMap<(CellBarcode, T), usize> = DashMap::default();

    stats.par_iter().for_each(|(cb, bed, count)| {
        let key = (cb.clone(), feature_key_func(bed));
        combined_data
            .entry(key)
            .and_modify(|c| *c += count)
            .or_insert(*count);
    });

    let combined_data = combined_data
        .into_iter()
        .map(|((c, k), v)| (c, k, v as f32))
        .collect::<Vec<_>>();

    format_data_triplets(combined_data)
}

// ─────────────────────────────────────────────────────────
// Mixture mode (SCAPE EM, migrated from run_apa_mix)
// ─────────────────────────────────────────────────────────

fn run_mixture(args: &CountApaArgs) -> anyhow::Result<()> {
    let utrs = load_utrs(args)?;

    if utrs.is_empty() {
        info!("no UTR regions found");
        return Ok(());
    }

    let pre_sites = if let Some(ref pre_path) = args.pre_sites {
        info!("loading pre-identified sites from {}", pre_path);
        let sites = load_pre_sites(pre_path)?;
        info!("loaded sites for {} UTRs", sites.len());
        Some(sites)
    } else {
        None
    };

    let njobs = utrs.len();
    info!("processing {} UTRs...", njobs);

    let arc_results: Arc<Mutex<Vec<CellSiteCount>>> =
        Arc::new(Mutex::new(Vec::with_capacity(njobs * 10)));
    let arc_annotations: Arc<Mutex<Vec<ApaSiteAnnotation>>> =
        Arc::new(Mutex::new(Vec::with_capacity(njobs * 3)));

    utrs.par_iter()
        .progress_count(njobs as u64)
        .try_for_each(|utr| -> anyhow::Result<()> {
            let (cell_counts, site_annots) =
                process_utr(utr, &args.bam_files, pre_sites.as_ref(), args)?;
            if !cell_counts.is_empty() {
                arc_results.lock().expect("lock").extend(cell_counts);
            }
            if !site_annots.is_empty() {
                arc_annotations.lock().expect("lock").extend(site_annots);
            }
            Ok(())
        })?;

    let all_counts = Arc::try_unwrap(arc_results)
        .map_err(|_| anyhow::anyhow!("failed to unwrap results"))?
        .into_inner()?;

    let all_annotations = Arc::try_unwrap(arc_annotations)
        .map_err(|_| anyhow::anyhow!("failed to unwrap annotations"))?
        .into_inner()?;

    info!("collected {} cell-site counts", all_counts.len());

    if all_counts.is_empty() {
        info!("no counts to output");
        return Ok(());
    }

    // Rows=sites, cols=cells
    let triplets_data: Vec<(CellBarcode, Box<str>, f32)> = all_counts
        .into_iter()
        .map(|c| (c.cell_barcode, c.site_id, c.count as f32))
        .collect();

    let triplets = format_data_triplets(triplets_data);
    let output_file = args.backend_file_path("count_apa");
    let data = triplets.to_backend(&output_file)?;
    data.qc(args.qc_cutoffs())?;
    info!("created output: {}", &output_file);

    // Write APA site annotation Parquet
    if !all_annotations.is_empty() {
        let parquet_path = format!("{}/apa_site_annotations.parquet", &args.output);
        write_apa_annotations(&all_annotations, &parquet_path)?;
        info!(
            "wrote {} site annotations to {}",
            all_annotations.len(),
            parquet_path
        );
    }

    info!("done");
    Ok(())
}

/// Load UTR regions from --gff or --utr-bed.
fn load_utrs(args: &CountApaArgs) -> anyhow::Result<Vec<UtrRegion>> {
    if let Some(ref gff_file) = args.gff_file {
        info!("parsing GFF/GTF file: {}", gff_file);
        let records = read_gff_record_vec(gff_file)?;
        info!("read {} GFF records", records.len());

        let model = build_union_gene_model(&records)?;
        info!(
            "found {} 3'-UTR regions in gene model",
            model.three_prime_utr.len()
        );

        let mut utrs = Vec::new();
        for entry in model.three_prime_utr.iter() {
            let gene_id = entry.key();
            let rec = entry.value();
            let utr_length = (rec.stop - rec.start) as usize;
            if utr_length < args.min_utr_length {
                continue;
            }
            let name: Box<str> = match &rec.gene_name {
                genomic_data::gff::GeneSymbol::Symbol(s) => format!("{}_{}", gene_id, s).into(),
                genomic_data::gff::GeneSymbol::Missing => format!("{}", gene_id).into(),
            };
            utrs.push(UtrRegion {
                chr: rec.seqname.clone(),
                start: rec.start,
                end: rec.stop,
                strand: rec.strand,
                name,
                utr_length,
            });
        }

        info!("kept {} 3'-UTRs (>= {}bp)", utrs.len(), args.min_utr_length);
        Ok(utrs)
    } else if let Some(ref utr_bed) = args.utr_bed {
        info!("loading UTR regions from {}", utr_bed);
        let utrs = load_utr_regions_from_bed(utr_bed)?;
        info!("loaded {} UTR regions", utrs.len());
        Ok(utrs)
    } else {
        Err(anyhow::anyhow!(
            "must provide either --gff or --utr-bed for mixture mode"
        ))
    }
}

/// Load pre-identified sites from a BED file.
fn load_pre_sites(path: &str) -> anyhow::Result<fnv::FnvHashMap<Box<str>, Vec<f64>>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut sites: fnv::FnvHashMap<Box<str>, Vec<f64>> = fnv::FnvHashMap::default();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() >= 4 {
            let name = fields[3];
            if let Some((utr_name, pos_str)) = name.rsplit_once('@') {
                if let Ok(pos) = pos_str.parse::<f64>() {
                    sites.entry(utr_name.into()).or_default().push(pos);
                }
            }
        }
    }

    Ok(sites)
}

/// Process a single UTR: extract fragments, discover/load sites, run EM, assign cells.
fn process_utr(
    utr: &UtrRegion,
    bam_files: &[Box<str>],
    pre_sites: Option<&fnv::FnvHashMap<Box<str>, Vec<f64>>>,
    args: &CountApaArgs,
) -> anyhow::Result<(Vec<CellSiteCount>, Vec<ApaSiteAnnotation>)> {
    let mut all_fragments = Vec::new();
    for bam_file in bam_files {
        let polya_params = PolyAFilterParams {
            min_tail: args.polya_min_tail_length,
            max_non_at: args.polya_max_non_a_or_t,
            internal_prime_window: args.polya_internal_prime_window,
            internal_prime_count: args.polya_internal_prime_count,
        };
        let frags = extract_fragments(
            bam_file,
            utr,
            args.cell_barcode_tag.as_bytes(),
            args.umi_tag.as_bytes(),
            &polya_params,
        )?;
        all_fragments.extend(frags);
    }

    info!(
        "UTR {} ({}:{}-{}, L={}): {} fragments extracted",
        utr.name,
        utr.chr,
        utr.start,
        utr.end,
        utr.utr_length,
        all_fragments.len()
    );

    if all_fragments.len() < args.min_fragments {
        return Ok((Vec::new(), Vec::new()));
    }

    let candidate_sites = if let Some(pre) = pre_sites {
        pre.get(&utr.name).cloned().unwrap_or_default()
    } else {
        let raw_sites = discover_sites_from_junctions(&all_fragments, args.min_coverage);
        if !raw_sites.is_empty() {
            merge_nearby_sites(&raw_sites, &all_fragments, args.merge_distance)
        } else {
            let bandwidth = 100.0;
            let coverage_sites =
                discover_sites_from_coverage(&all_fragments, utr.utr_length as f64, bandwidth);
            merge_nearby_sites(&coverage_sites, &all_fragments, args.merge_distance)
        }
    };

    let n_junction = all_fragments.iter().filter(|f| f.is_junction).count();
    info!(
        "UTR {}: {} candidate sites, {} junction reads",
        utr.name,
        candidate_sites.len(),
        n_junction
    );

    if candidate_sites.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let lik_params = args.lik_params();
    let (theta_lik_matrix, theta_grid) =
        precompute_theta_lik_matrix(&all_fragments, utr.utr_length as f64, &lik_params);

    let default_beta = (args.min_beta + args.max_beta) / 2.0;
    let beta_arr: Vec<f64> = vec![default_beta; candidate_sites.len()];

    let em_params = args.em_params();
    let em_result = fixed_inference(
        &candidate_sites,
        &beta_arr,
        &theta_lik_matrix,
        &theta_grid,
        utr.utr_length as f64,
        lik_params.max_polya,
        &em_params,
    );

    let (cell_counts, annotations) = assign_fragments_to_sites(&all_fragments, &em_result, utr);

    Ok((cell_counts, annotations))
}

/// Write APA site annotations to a Parquet file.
fn write_apa_annotations(annotations: &[ApaSiteAnnotation], path: &str) -> anyhow::Result<()> {
    let mut site_ids = Vec::with_capacity(annotations.len());
    let mut gene_names = Vec::with_capacity(annotations.len());
    let mut chrs = Vec::with_capacity(annotations.len());
    let mut genomic_alphas = Vec::with_capacity(annotations.len());
    let mut betas = Vec::with_capacity(annotations.len());
    let mut genomic_starts = Vec::with_capacity(annotations.len());
    let mut genomic_stops = Vec::with_capacity(annotations.len());
    let mut pi_weights = Vec::with_capacity(annotations.len());
    let mut expected_tails = Vec::with_capacity(annotations.len());

    for a in annotations {
        site_ids.push(a.site_id.as_ref());
        gene_names.push(a.gene_name.as_ref());
        chrs.push(a.chr.as_ref());
        genomic_alphas.push(a.genomic_alpha);
        betas.push(a.beta);
        genomic_starts.push(a.genomic_start);
        genomic_stops.push(a.genomic_stop);
        pi_weights.push(a.pi_weight);
        expected_tails.push(a.expected_tail_length);
    }

    let schema = arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("site_id", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("gene_name", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("chr", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("genomic_alpha", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("beta", arrow::datatypes::DataType::Float64, false),
        arrow::datatypes::Field::new("genomic_start", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("genomic_stop", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("pi_weight", arrow::datatypes::DataType::Float64, false),
        arrow::datatypes::Field::new(
            "expected_tail_length",
            arrow::datatypes::DataType::Float64,
            false,
        ),
    ]);

    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![
            Arc::new(StringArray::from(site_ids)) as ArrayRef,
            Arc::new(StringArray::from(gene_names)) as ArrayRef,
            Arc::new(StringArray::from(chrs)) as ArrayRef,
            Arc::new(Int64Array::from(genomic_alphas)) as ArrayRef,
            Arc::new(Float64Array::from(betas)) as ArrayRef,
            Arc::new(Int64Array::from(genomic_starts)) as ArrayRef,
            Arc::new(Int64Array::from(genomic_stops)) as ArrayRef,
            Arc::new(Float64Array::from(pi_weights)) as ArrayRef,
            Arc::new(Float64Array::from(expected_tails)) as ArrayRef,
        ],
    )?;

    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}
