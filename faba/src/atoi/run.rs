use crate::common::*;
use crate::editing::io::ToParquet;
use crate::editing::mixture::MixtureParams;
use crate::editing::mixture_pipeline::run_mixture_model;
use crate::editing::pipeline::*;
use crate::editing::sifter::ModificationType;
use crate::gene_count::splice::CountReadOpts;
use crate::quant::{
    check_all_bam_indices, resolve_modality_gene_qc, resolve_umi_tag, GeneMatrixSink, GeneQcRequest,
};

use genomic_data::gff::GeneType as GffGeneType;
use genomic_data::gff::GffRecordMap;
use rayon::ThreadPoolBuilder;

#[derive(Args, Debug)]
pub struct AtoICountArgs {
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input BAM files (comma-separated)"
    )]
    pub bam_files: Vec<Box<str>>,

    #[arg(
        short = 'g',
        long = "gff",
        required = true,
        help = "Gene annotation (GFF) file"
    )]
    pub gff_file: Box<str>,

    #[arg(
        short = 'f',
        long = "genome",
        help = "Reference genome FASTA file",
        long_help = "Path to reference genome in FASTA format (.fa or .fasta).\n\
                     Used to validate base calls at editing sites. File must be indexed (.fai)."
    )]
    pub genome_file: Box<str>,

    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Output directory for A-to-I detection results.\n\
                     Writes atoi_sites.parquet (every putative site with its statistics),\n\
                     and per input BAM: {batch}_atoi (gene-level), {batch}_atoi_site (per-site),\n\
                     {batch}_count and {batch}_cells.tsv.gz from the gene QC, genes_kept.tsv.gz,\n\
                     and {batch}_atoi_mixture (+ atoi_components.parquet) with --mixture."
    )]
    pub output: Box<str>,

    #[arg(long, default_value = "CB", help = "Cell barcode tag")]
    pub cell_barcode_tag: Box<str>,

    #[arg(long, default_value = "GX", help = "Gene barcode tag")]
    pub gene_barcode_tag: Box<str>,

    #[arg(
        long,
        default_value_t = crate::editing::pipeline::DEFAULT_ATOI_MIN_COVERAGE,
        help = "Minimum coverage (ref + alt reads) for an A-to-I site to be written",
        long_help = "Minimum coverage (ref + alt reads) for an A-to-I site to be written.\n\
                     A candidacy floor, shared with `faba all --atoi-min-coverage`.\n\
                     A-to-I has no motif anchor, so this bounds the candidate set;\n\
                     no p-value cutoff is applied here (see `faba qc --site-max-pv`)."
    )]
    pub min_coverage: usize,

    #[arg(
        long = "min-conversion",
        default_value_t = crate::editing::pipeline::DEFAULT_ATOI_MIN_CONVERSION,
        help = "Minimum A-to-G (alt) reads for an A-to-I site to be written"
    )]
    pub min_conversion: usize,

    #[arg(
        long = "min-base-quality",
        default_value_t = 20,
        help = "Minimum base quality (Phred score) to include a base"
    )]
    pub min_base_quality: u8,

    #[arg(
        long = "min-mapping-quality",
        default_value_t = 20,
        help = "Minimum mapping quality (MAPQ) to include a read"
    )]
    pub min_mapping_quality: u8,

    #[arg(
        long = "error-rate",
        default_value_t = 0.01,
        help = "Sequencing-error rate ε for the beta-binomial editing null"
    )]
    pub error_rate: f64,

    #[arg(
        long = "overdispersion",
        default_value_t = 0.1,
        help = "Beta-binomial overdispersion ρ for the editing null (0 ⇒ binomial)"
    )]
    pub overdispersion: f64,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix backend (zarr or hdf5)"
    )]
    pub backend: SparseIoBackend,

    #[arg(
        long = "no-zip",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive",
        long_help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive.\n\
                     Zarr backend only; no effect on hdf5."
    )]
    pub zip: bool,

    #[arg(long, default_value_t = false, help = "Include reads w/o barcode info")]
    pub include_missing_barcode: bool,

    #[arg(
        long = "cell-membership",
        alias = "barcode-membership",
        alias = "membership",
        help = "Cell barcode membership file for filtering cells (TSV, CSV, or Parquet)"
    )]
    pub cell_membership_file: Option<Box<str>>,

    #[arg(
        long = "membership-barcode-col",
        default_value_t = 0,
        help = "Column index for cell barcodes in membership file"
    )]
    pub membership_barcode_col: usize,

    #[arg(
        long = "membership-celltype-col",
        default_value_t = 1,
        help = "Column index for cell types in membership file"
    )]
    pub membership_celltype_col: usize,

    #[arg(
        long = "exact-barcode-match",
        default_value_t = false,
        help = "Require exact cell barcode matching"
    )]
    pub exact_barcode_match: bool,

    #[arg(long, value_enum, help = "Gene type filter")]
    gene_type: Option<GffGeneType>,

    #[arg(
        long,
        alias = "threads",
        default_value_t = 16,
        help = "Maximum number of threads"
    )]
    max_threads: usize,

    ///////////////////////////
    // Mixture model options //
    ///////////////////////////
    #[arg(
        long = "mixture",
        default_value_t = false,
        help = "Also fit the per-gene 1D Gaussian mixture of editing sites (slow EM; off by default)"
    )]
    pub mixture: bool,

    #[arg(
        long = "mixture-min-sites",
        default_value_t = 3,
        help = "Min distinct positions per gene to attempt mixture"
    )]
    pub mixture_min_sites: usize,

    #[arg(
        long = "mixture-max-k",
        default_value_t = 5,
        help = "Cap on components per gene (modes of the smoothed site density; not a BIC selection)"
    )]
    pub mixture_max_k: usize,

    #[arg(
        long = "mixture-bandwidth",
        alias = "mixture-initial-sigma",
        default_value_t = 0.0,
        help = "Gaussian bandwidth (nt) for component calling; 0 = auto (data-derived)"
    )]
    pub mixture_bandwidth: f32,

    #[arg(
        long = "drop-single-component",
        default_value_t = false,
        help = "Drop genes with a single mixture component (no relative signal)"
    )]
    pub drop_single_component: bool,

    #[arg(
        long = "mixture-weight",
        value_enum,
        default_value_t = crate::editing::pipeline::MixtureWeightMode::Posterior,
        help = "How to weight each (cell, site) observation in the mixture EM",
        long_help = "Per-observation weighting for the per-gene Gaussian mixture.\n\
                     `posterior` (default) uses the Beta-posterior regularized effective count w = n · (c + α) / (n + α + β),\n\
                     where n is the per-site coverage and c the converted-read count.\n\
                     `converted` uses the raw converted-read count c (legacy)."
    )]
    pub mixture_weight: crate::editing::pipeline::MixtureWeightMode,

    #[arg(
        long = "mixture-prior-alpha",
        default_value_t = 1.0,
        help = "Beta prior α for posterior-rate weighting"
    )]
    pub mixture_prior_alpha: f32,

    #[arg(
        long = "mixture-prior-beta",
        default_value_t = 1.0,
        help = "Beta prior β for posterior-rate weighting"
    )]
    pub mixture_prior_beta: f32,

    ////////////////////////
    // Gene expression QC //
    ////////////////////////
    #[arg(
        long = "gene-min-cells",
        default_value_t = 1,
        help = "Min cells per gene for expression QC; 1 = drop only empty rows (stricter floors: `faba qc`)"
    )]
    pub gene_min_cells: usize,

    #[arg(
        long = "gene-min-counts",
        default_value_t = 0,
        help = "Min total UMI counts per gene for expression QC (0 disables)"
    )]
    pub gene_min_counts: usize,

    #[arg(
        long = "cell-min-genes",
        default_value_t = 1,
        help = "Min genes per cell for expression QC; 1 = drop only empty columns (stricter floors: `faba qc`)"
    )]
    pub cell_min_genes: usize,

    #[arg(
        long = "skip-gene-qc",
        default_value_t = false,
        help = "Skip gene expression QC step"
    )]
    pub skip_gene_qc: bool,

    #[command(flatten)]
    pub cell_qc: crate::cell_qc::CellQcArgs,

    #[command(flatten)]
    pub mito_qc: crate::quant::MitoQcArgs,

    /// Reuse a per-batch cell set from `faba count` instead of recomputing QC
    #[arg(
        long = "valid-cells",
        help = "Directory of `faba count` outputs ({batch}_cells.tsv.gz) to reuse"
    )]
    pub valid_cells_file: Option<Box<str>>,

    /// Reuse the retained-gene set from `faba count` (its pooled `genes_kept.tsv.gz`)
    #[arg(long = "valid-genes")]
    pub valid_genes_file: Option<Box<str>>,

    #[arg(
        long = "umi-tag",
        default_value = "UB",
        help = "UMI BAM tag (for read dedup)"
    )]
    pub umi_tag: Box<str>,

    #[arg(
        long = "no-umi-dedup",
        default_value_t = false,
        help = "Disable UMI deduplication"
    )]
    pub no_umi_dedup: bool,
}

impl From<&AtoICountArgs> for ConversionParams {
    fn from(args: &AtoICountArgs) -> Self {
        ConversionParams {
            genome_file: args.genome_file.clone(),
            wt_bam_files: args.bam_files.clone(),
            gene_barcode_tag: args.gene_barcode_tag.clone(),
            cell_barcode_tag: args.cell_barcode_tag.clone(),
            include_missing_barcode: args.include_missing_barcode,
            min_coverage: args.min_coverage,
            min_conversion: args.min_conversion,
            error_rate: args.error_rate,
            overdispersion: args.overdispersion,
            backend: args.backend.clone(),
            zip: args.zip,
            output: args.output.clone(),
            cell_membership_file: args.cell_membership_file.clone(),
            membership_barcode_col: args.membership_barcode_col,
            membership_celltype_col: args.membership_celltype_col,
            exact_barcode_match: args.exact_barcode_match,
            mod_type: ModificationType::AtoI,
            min_base_quality: args.min_base_quality,
            min_mapping_quality: args.min_mapping_quality,
            mixture_weight_mode: args.mixture_weight,
            mixture_prior_alpha: args.mixture_prior_alpha,
            mixture_prior_beta: args.mixture_prior_beta,
            umi_tag: if args.no_umi_dedup {
                None
            } else {
                Some(args.umi_tag.clone())
            },
            // A-to-I is single-sample (ADAR is active in the YTHmut too); no control.
            mut_bam_files: Vec::new(),
            competent_cells: None,
        }
    }
}

/// Standalone A-to-I editing site detection and quantification
pub fn run_atoi(args: &AtoICountArgs) -> anyhow::Result<()> {
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

    check_all_bam_indices(&args.bam_files)?;

    info!("parsing GFF file: {}", args.gff_file);
    // One parse, two views: the gene map and the merged-exon model. A second
    // `GffRecordMap::from` here would re-read GENCODE (~9s) for records we
    // already hold.
    let gff_records = read_gff_record_vec(args.gff_file.as_ref())?;
    let mut gff_map = GffRecordMap::from_map(build_gene_map(
        &gff_records,
        Some(&genomic_data::gff::FeatureType::Gene),
    )?);
    // Transcript coordinates for the sites parquet's rel_pos column, from the
    // same parse that built the gene map above.
    let spliced = crate::data::gene_model::SplicedGenes::from_records(&gff_records);

    if let Some(gene_type) = args.gene_type.clone() {
        gff_map.subset(gene_type);
    }

    info!("found {} genes", gff_map.len());
    if gff_map.is_empty() {
        info!("empty gff map");
        return Ok(());
    }

    // Gene expression QC: reuse a passed cell/gene set from `faba count`, or
    // recompute it (per-batch cell calling).
    let gene_qc = resolve_modality_gene_qc(
        &mut gff_map,
        &GeneQcRequest {
            bam_files: &args.bam_files,
            count: CountReadOpts {
                cell_barcode_tag: &args.cell_barcode_tag,
                gene_barcode_tag: &args.gene_barcode_tag,
                umi_tag: resolve_umi_tag(args.no_umi_dedup, &args.umi_tag),
                min_mapping_quality: args.min_mapping_quality,
            },
            gff_file: Some(&args.gff_file),
            output_dir: &args.output,
            // Biotype is applied by subsetting the gff for site discovery; QC counts
            // every biotype (and the gene set it freezes is intersected with that subset).
            gene_type: "",
            gene_min_cells: args.gene_min_cells,
            gene_min_counts: args.gene_min_counts,
            cell_min_genes: args.cell_min_genes,
            cell_call: args.cell_qc.params(),
            mito: args.mito_qc.params(),
            valid_cells_file: args.valid_cells_file.as_deref(),
            valid_genes_file: args.valid_genes_file.as_deref(),
            skip_gene_qc: args.skip_gene_qc,
            persist: Some(GeneMatrixSink {
                backend: &args.backend,
                zip: args.zip,
            }),
        },
    )?;
    if gene_qc.is_some() && gff_map.is_empty() {
        info!("no genes passed QC");
        return Ok(());
    }

    let params = ConversionParams::from(args);

    // Load cell membership for filtering
    let membership = params.load_membership()?;

    // FIRST PASS: every putative A-to-I site, with its p-value. No cutoff is
    // applied here; `faba qc` thresholds the parquet columns.
    let atoi_sites = find_all_conversion_sites(&gff_map, &params, membership.as_ref())?;
    let n_atoi: usize = atoi_sites.iter().map(|x| x.value().len()).sum();
    info!("Found {} putative A-to-I editing sites", n_atoi);

    if atoi_sites.is_empty() {
        info!("no A-to-I sites found");
        return Ok(());
    }

    // Write sites parquet
    atoi_sites.to_parquet(
        &gff_map,
        &spliced,
        format!("{}/atoi_sites.parquet", args.output),
    )?;
    info!("wrote atoi_sites.parquet");

    // SECOND PASS: quantify into sparse matrix
    info!("Second pass: A-to-I count matrix");
    let valid_cells = gene_qc.as_ref().map(|qc| &qc.cells_by_batch);
    process_all_bam_files_to_backend(&params, &atoi_sites, &gff_map, valid_cells)?;

    // Mixture model (opt-in): cluster editing sites per gene
    if args.mixture {
        info!("Running 1D Gaussian mixture model on A-to-I sites...");
        let mix_params = MixtureParams {
            min_sites: args.mixture_min_sites,
            max_k: args.mixture_max_k,
            bandwidth: args.mixture_bandwidth,
            drop_single_component: args.drop_single_component,
            ..Default::default()
        };
        run_mixture_model(
            &params,
            &atoi_sites,
            &gff_map,
            &spliced,
            &mix_params,
            valid_cells,
        )?;
    }

    info!("done");
    Ok(())
}
