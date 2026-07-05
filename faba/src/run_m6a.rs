use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::editing::bed_output::process_all_bam_files_to_bed;
use crate::editing::io::{load_atoi_mask_from_parquet, ToParquet};
use crate::editing::mask::{build_atoi_mask, filter_m6a_by_mask};
use crate::editing::mixture::MixtureParams;
use crate::editing::mixture_pipeline::run_mixture_model;
use crate::editing::pipeline::{
    find_all_conversion_sites, process_all_bam_files_to_backend, ConversionParams, M6aContrastArgs,
};
use crate::editing::sifter::ModificationType;
use crate::pipeline_util::mass_enrichment::MassEnrichmentArgs;
use crate::pipeline_util::{
    check_all_bam_indices, resolve_modality_gene_qc, resolve_umi_tag, GeneMatrixSink, GeneQcRequest,
};
use crate::snp::io::load_snp_mask_from_parquet;

use genomic_data::gff::GeneType as GffGeneType;
use genomic_data::gff::GffRecordMap;

use rayon::ThreadPoolBuilder;

#[derive(Args, Debug)]
pub struct DartSeqCountArgs {
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Signal BAM files (APOBEC1-YTH fusion)",
        long_help = "Comma-separated list of signal (APOBEC1-YTH fusion) BAM files.\n\
                     These contain the C->T conversions at m6A sites. Each motif C\n\
                     is called by a WT-vs-MUT contrast against the --control-bam\n\
                     samples (a genomic C/T variant converts equally in both arms\n\
                     and is rejected); calls are FDR-corrected."
    )]
    pub wt_bam_files: Vec<Box<str>>,

    #[arg(
        short = 'm',
        long = "control-bam",
        alias = "mut",
        alias = "control",
        alias = "background",
        value_delimiter = ',',
        required = true,
        help = "Control BAM files (catalytically-dead YTHmut)",
        long_help = "Comma-separated list of control (catalytically-dead YTHmut) BAM\n\
                     files, pooled into one background. m6A is called where the\n\
                     signal BAMs show significantly higher C->T conversion than\n\
                     these controls (two-sample test). Required: m6A cannot be\n\
                     distinguished from genomic variation without a control."
    )]
    pub control_bam_files: Vec<Box<str>>,

    #[command(flatten)]
    pub m6a_contrast: M6aContrastArgs,

    #[arg(
        short = 'g',
        long = "gff",
        required = true,
        help = "Gene annotation in GFF format (e.g. genes.gff)"
    )]
    pub gff_file: Box<str>,

    #[arg(
        long,
        default_value = "CB",
        help = "Cell barcode tag",
        long_help = "Cell barcode tag used for cell/sample identification in 10x Genomics\n\
                     BAM files. [See here](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam)"
    )]
    pub cell_barcode_tag: Box<str>,

    #[arg(
        long,
        default_value = "GX",
        help = "Gene barcode tag",
        long_help = "Barcode tag used for gene identification in 10x Genomics BAM files.\n\
		    [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)"
    )]
    pub gene_barcode_tag: Box<str>,

    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum number of total reads per site",
        long_help = "Minimum number of total reads required per site for inclusion in\n\
                     the analysis. Filters out low-coverage sites."
    )]
    pub min_coverage: usize,

    #[arg(
        long = "min-conversion",
        default_value_t = 5,
        help = "Minimum converted (C->T) reads per site"
    )]
    pub min_conversion: usize,

    #[arg(
        long = "min-base-quality",
        default_value_t = 20,
        help = "Minimum base quality (Phred) to include a base"
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
        short = 'p',
        long = "pval",
        alias = "pvalue",
        alias = "p-val",
        alias = "p-value",
        default_value_t = 0.05,
        help = "Detection FDR target (Benjamini-Hochberg q-value)"
    )]
    pub pvalue_cutoff: f32,

    #[arg(
        long,
        value_enum,
        help = "Gene type (protein_coding, pseudogene, lncRNA)",
        long_help = "Filter analysis by gene type. \n\
		     Options include protein_coding, pseudogene, or lncRNA."
    )]
    gene_type: Option<GffGeneType>,

    #[arg(
        long,
        alias = "threads",
        default_value_t = 16,
        help = "Maximum number of threads",
        long_help = "Maximum number of threads to use for parallel processing.\n\
                     Choose the right number in HPC environments."
    )]
    max_threads: usize,

    #[arg(
        long = "site-min-cells",
        default_value_t = crate::editing::pipeline::DEFAULT_SITE_MIN_CELLS,
        help = "Min cells per site for the per-site matrix feature QC (0 disables)",
        long_help = "Unit-aware feature QC for the per-site (`_site`) output matrix:\n\
                     a site is kept only if detected in at least this many cells,\n\
                     and both of its channels (methylated/unmethylated) are kept\n\
                     together. The gene-level matrix is unaffected. 0 disables.\n\
                     Sites are a distinct feature space not covered by the upstream\n\
                     gene expression QC (--gene-min-cells)."
    )]
    pub site_min_cells: usize,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix backend (zarr or hdf5)",
        long_help = "File format for the output sparse matrix. Supported: zarr, hdf5."
    )]
    pub backend: SparseIoBackend,

    #[arg(
        long = "no-zip",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive",
        long_help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive\n\
                     (zarr backend only; no effect on hdf5)"
    )]
    pub zip: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Include reads w/o barcode info",
        long_help = "Include reads that are missing gene and cell barcode information\n\
                     in the analysis."
    )]
    pub include_missing_barcode: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Output results in BED",
        long_help = "Output results in BED file format for genomic intervals."
    )]
    output_bed_file: bool,

    #[arg(short, long, required = true, help = "Output directory")]
    pub output: Box<str>,

    #[arg(
        short = 'f',
        long = "genome",
        help = "Reference genome FASTA file",
        long_help = "Path to reference genome in FASTA format (.fa or .fasta). \n\
		     Used to validate base calls at editing sites. \n\
		     File must be indexed (.fai). If index doesn't exist, one will be created. \n\
		     Example: genome.fa"
    )]
    pub genome_file: Box<str>,

    #[arg(
        long = "cell-membership",
        alias = "barcode-membership",
        alias = "membership",
        help = "Cell barcode membership file for filtering cells (TSV, CSV, or Parquet)",
        long_help = "Path to cell barcode membership file for restricting analysis to\n\
                     specific cells.\n\
                     Format: First column = cell barcode, Second column = cell type.\n\
                     Supports .tsv, .csv, .parquet, and .gz variants.\n\
                     Only cells (barcodes) present in this file will be included in analysis.\n\
                     By default, barcodes are prefix-matched (use --exact-barcode-match to change)."
    )]
    pub cell_membership_file: Option<Box<str>>,

    #[arg(
        long = "membership-barcode-col",
        default_value_t = 0,
        help = "Column index for cell barcodes in membership file",
        long_help = "Zero-based column index for the cell barcode field in\n\
                     the membership file."
    )]
    pub membership_barcode_col: usize,

    #[arg(
        long = "membership-celltype-col",
        default_value_t = 1,
        help = "Column index for cell types in membership file",
        long_help = "Zero-based column index for the cell type field in\n\
                     the membership file."
    )]
    pub membership_celltype_col: usize,

    #[arg(
        long = "exact-barcode-match",
        default_value_t = false,
        help = "Require exact cell barcode matching",
        long_help = "By default, membership barcodes are matched as prefixes\n\
                     of BAM barcodes (handles suffixes like \"-1\").\n\
                     Enable this flag to require exact string matching."
    )]
    pub exact_barcode_match: bool,

    #[arg(
        long = "output-cell-types",
        default_value_t = false,
        help = "Include cell type annotation in BED output",
        long_help = "Append a cell type column to BED output lines.\n\
                     Requires --cell-membership or --n-clusters > 1."
    )]
    pub output_cell_types: bool,

    #[arg(
        long = "no-check-r-site",
        default_value_t = false,
        help = "Disable R site (RAC/GTY) validation in reference",
        long_help = "By default, faba validates the R position in RAC/GTY\n\
                     motifs against the reference genome (requires R=A/G on\n\
                     forward strand and Y=C/T on reverse strand). Use this\n\
                     flag to disable that check."
    )]
    pub no_check_r_site: bool,

    //////////////////////////////
    // A-to-I editing detection //
    //////////////////////////////
    #[arg(
        long = "detect-atoi",
        default_value_t = false,
        help = "Detect A-to-I editing sites and mask them from m6A calling",
        long_help = "Detect A-to-I (adenosine-to-inosine) RNA editing sites\n\
                     via A→G conversions. Detected sites are output to a\n\
                     separate parquet file and used as a mask to exclude\n\
                     false-positive m6A candidates whose RAC/GTY triplet\n\
                     overlaps an A-to-I site."
    )]
    pub detect_atoi: bool,

    #[arg(
        long = "atoi-min-coverage",
        default_value_t = 10,
        help = "Minimum coverage for A-to-I site detection"
    )]
    pub atoi_min_coverage: usize,

    #[arg(
        long = "atoi-min-conversion",
        default_value_t = 5,
        help = "Minimum A-to-G conversions for A-to-I detection"
    )]
    pub atoi_min_conversion: usize,

    #[arg(
        long = "atoi-pval",
        default_value_t = 0.05,
        help = "A-to-I detection FDR target (Benjamini-Hochberg q-value)"
    )]
    pub atoi_pvalue_cutoff: f32,

    #[arg(
        long = "atoi-mask",
        help = "Pre-computed A-to-I mask parquet (from `faba atoi` or `--detect-atoi`)",
        long_help = "Path to a pre-computed A-to-I sites parquet file.\n\
                     When provided, skips A-to-I discovery and loads the mask\n\
                     directly from this file to filter m6A candidates.\n\
                     Implies --detect-atoi behavior for masking."
    )]
    pub atoi_mask_file: Option<Box<str>>,

    #[arg(
        long = "snp-mask",
        help = "SNP mask parquet from `faba snp` to filter genetic variants",
        long_help = "Path to snp_sites.parquet from `faba snp`. m6A candidates\n\
                     at known SNP positions (het or hom-alt) are removed.\n\
                     Applied after A-to-I masking (if any)."
    )]
    pub snp_mask_file: Option<Box<str>>,

    ///////////////////////////
    // Mixture model options //
    ///////////////////////////
    #[arg(
        long = "no-mixture",
        default_value_t = false,
        help = "Disable 1D Gaussian mixture clustering of modification sites",
        long_help = "Disable 1D Gaussian mixture clustering of modification sites.\n\
                     By default, faba fits a mixture of Gaussians + uniform noise\n\
                     to the discovered site positions per gene, selecting K by BIC.\n\
                     This outputs a sparse (cells x mixture_components) count matrix\n\
                     and a m6a_components.parquet file."
    )]
    pub no_mixture: bool,

    #[arg(
        long = "mixture-min-sites",
        default_value_t = 3,
        help = "Min distinct positions per gene to attempt mixture (default: 3)"
    )]
    pub mixture_min_sites: usize,

    #[arg(
        long = "mixture-max-k",
        default_value_t = 5,
        help = "Max components to test via BIC (default: 5)"
    )]
    pub mixture_max_k: usize,

    #[arg(
        long = "mixture-bandwidth",
        alias = "mixture-initial-sigma",
        default_value_t = 0.0,
        help = "Gaussian bandwidth (nt) for component calling; 0 = auto (data-derived)",
        long_help = "Gaussian smoothing bandwidth in nucleotides used to call mixture\n\
                     components: the per-gene signal pileup is smoothed at this\n\
                     bandwidth and its modes become components. 0 (default) derives\n\
                     a global per-modality bandwidth from the empirical site spacing."
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
                     `posterior` (default) uses the Beta-posterior regularized\n\
                     effective count w = n · (c + α) / (n + α + β), where n is\n\
                     the per-site coverage and c the converted-read count. This\n\
                     prevents low-coverage 1-of-1 sites from dominating μ/σ.\n\
                     `converted` uses the raw converted-read count c (legacy)."
    )]
    pub mixture_weight: crate::editing::pipeline::MixtureWeightMode,

    #[arg(
        long = "mixture-prior-alpha",
        default_value_t = 1.0,
        help = "Beta prior α for posterior-rate weighting (default: 1.0)"
    )]
    pub mixture_prior_alpha: f32,

    #[arg(
        long = "mixture-prior-beta",
        default_value_t = 1.0,
        help = "Beta prior β for posterior-rate weighting (default: 1.0)"
    )]
    pub mixture_prior_beta: f32,

    ///////////////////////////////
    // Mass-enrichment grouping   //
    ///////////////////////////////
    #[command(flatten)]
    enrich: MassEnrichmentArgs,

    ////////////////////////
    // Gene expression QC //
    ////////////////////////
    #[arg(
        long = "gene-min-cells",
        default_value_t = 10,
        help = "Min cells per gene for expression QC",
        long_help = "Minimum number of cells with non-zero expression for a gene\n\
                     to pass QC. Genes below this threshold are excluded before\n\
                     site discovery."
    )]
    pub gene_min_cells: usize,

    #[arg(
        long = "gene-min-counts",
        default_value_t = 0,
        help = "Min total UMI counts per gene for expression QC (0 disables)",
        long_help = "Minimum total UMI counts (summed across all cells) for a gene\n\
                     to pass QC. Genes below this threshold are excluded before\n\
                     site discovery. 0 disables the threshold."
    )]
    pub gene_min_counts: usize,

    #[arg(
        long = "cell-min-genes",
        default_value_t = 10,
        help = "Min genes per cell for expression QC",
        long_help = "Minimum number of genes with non-zero expression for a cell\n\
                     to pass QC. Cells below this threshold are excluded from\n\
                     quantification."
    )]
    pub cell_min_genes: usize,

    #[arg(
        long = "skip-gene-qc",
        default_value_t = false,
        help = "Skip gene expression QC step",
        long_help = "Skip the a priori gene expression QC step.\n\
                     By default, faba counts reads per gene and filters to\n\
                     expressed genes/cells before site discovery."
    )]
    pub skip_gene_qc: bool,

    #[command(flatten)]
    pub cell_qc: crate::cell_qc::CellQcArgs,

    /// Reuse a per-batch cell set from `faba genes` instead of recomputing QC
    #[arg(
        long = "valid-cells",
        help = "Directory of `faba genes` outputs ({batch}_cells.tsv.gz) to reuse"
    )]
    pub valid_cells_file: Option<Box<str>>,

    /// Reuse the retained-gene set from `faba genes` ({batch}_genes_kept.tsv.gz)
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

/// Create m6A ConversionParams from DartSeqCountArgs
impl From<&DartSeqCountArgs> for ConversionParams {
    fn from(args: &DartSeqCountArgs) -> Self {
        ConversionParams {
            genome_file: args.genome_file.clone(),
            wt_bam_files: args.wt_bam_files.clone(),
            gene_barcode_tag: args.gene_barcode_tag.clone(),
            cell_barcode_tag: args.cell_barcode_tag.clone(),
            include_missing_barcode: args.include_missing_barcode,
            min_coverage: args.min_coverage,
            min_conversion: args.min_conversion,
            pvalue_cutoff: args.pvalue_cutoff,
            error_rate: args.error_rate,
            overdispersion: args.overdispersion,
            backend: args.backend.clone(),
            zip: args.zip,
            output: args.output.clone(),
            cell_membership_file: args.cell_membership_file.clone(),
            membership_barcode_col: args.membership_barcode_col,
            membership_celltype_col: args.membership_celltype_col,
            exact_barcode_match: args.exact_barcode_match,
            mod_type: ModificationType::M6A {
                check_r_site: !args.no_check_r_site,
                contrast: args.m6a_contrast.to_contrast(),
            },
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
            mut_bam_files: args.control_bam_files.clone(),
            site_min_cells: args.site_min_cells,
        }
    }
}

impl DartSeqCountArgs {
    /// Create A-to-I ConversionParams from DartSeqCountArgs (single-sample).
    fn atoi_params(&self) -> ConversionParams {
        ConversionParams {
            genome_file: self.genome_file.clone(),
            wt_bam_files: self.wt_bam_files.clone(),
            gene_barcode_tag: self.gene_barcode_tag.clone(),
            cell_barcode_tag: self.cell_barcode_tag.clone(),
            include_missing_barcode: self.include_missing_barcode,
            min_coverage: self.atoi_min_coverage,
            min_conversion: self.atoi_min_conversion,
            pvalue_cutoff: self.atoi_pvalue_cutoff,
            error_rate: self.error_rate,
            overdispersion: self.overdispersion,
            backend: self.backend.clone(),
            zip: self.zip,
            output: self.output.clone(),
            cell_membership_file: self.cell_membership_file.clone(),
            membership_barcode_col: self.membership_barcode_col,
            membership_celltype_col: self.membership_celltype_col,
            exact_barcode_match: self.exact_barcode_match,
            mod_type: ModificationType::AtoI,
            min_base_quality: self.min_base_quality,
            min_mapping_quality: self.min_mapping_quality,
            mixture_weight_mode: self.mixture_weight,
            mixture_prior_alpha: self.mixture_prior_alpha,
            mixture_prior_beta: self.mixture_prior_beta,
            umi_tag: if self.no_umi_dedup {
                None
            } else {
                Some(self.umi_tag.clone())
            },
            // A-to-I is single-sample (ADAR is active in the YTHmut too); no control.
            mut_bam_files: Vec::new(),
            site_min_cells: self.site_min_cells,
        }
    }
}

/// Detect and quantify DART-seq m6A sites
pub fn run_m6a(args: &DartSeqCountArgs) -> anyhow::Result<()> {
    mkdir(&args.output)?;

    // Setup thread pool
    let max_threads = num_cpus::get().min(args.max_threads);
    ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global()?;
    info!("will use {} threads", rayon::current_num_threads());

    // Validate inputs
    if args.wt_bam_files.is_empty() {
        return Err(anyhow::anyhow!("need at least one signal BAM file"));
    }
    if args.control_bam_files.is_empty() {
        return Err(anyhow::anyhow!(
            "m6A requires control BAMs (--control-bam): the WT-vs-MUT contrast \
             cannot separate m6A from genomic C/T variation without a control"
        ));
    }

    // Check all BAM indices (signal + control)
    check_all_bam_indices(&args.wt_bam_files)?;
    check_all_bam_indices(&args.control_bam_files)?;

    // m6A is a WT-vs-MUT contrast, so the signal (wt) arm for SITE DISCOVERY is
    // the positional BAMs MINUS any control listed in --control-bam; otherwise a
    // both-listed control would be pooled into the wt side and dilute its own
    // contrast (mirrors `faba all`). Controls are still QUANTIFIED in the second
    // pass via `quant_bam_files` (signal ∪ control) — only discovery drops them.
    let control_set: rustc_hash::FxHashSet<&str> =
        args.control_bam_files.iter().map(|s| s.as_ref()).collect();
    let signal_bam_files: Vec<Box<str>> = args
        .wt_bam_files
        .iter()
        .filter(|b| !control_set.contains(b.as_ref()))
        .cloned()
        .collect();
    if signal_bam_files.is_empty() {
        return Err(anyhow::anyhow!(
            "no m6A signal BAMs: every positional BAM is also in --control-bam"
        ));
    }
    info!(
        "m6A contrast: {} signal (wt) vs {} control (mut) BAMs \
         (controls excluded from site discovery, still quantified)",
        signal_bam_files.len(),
        args.control_bam_files.len()
    );

    // Load and filter GFF
    info!("parsing GFF file: {}", args.gff_file);
    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref())?;

    if let Some(gene_type) = args.gene_type.clone() {
        gff_map.subset(gene_type);
    }

    info!("found {} genes", gff_map.len());
    if gff_map.is_empty() {
        info!("empty gff map");
        return Ok(());
    }

    // Gene expression QC: reuse a passed cell/gene set from `faba genes`, or
    // recompute it (per-batch cell calling).
    // Cell-calling QC must cover EVERY quantified BAM — signal (wt) AND control
    // (mut) — so control cells are filtered by their own per-library knee in the
    // second pass (see `quant_bam_files`). The map is keyed by BAM path, so it
    // stays correct regardless of BAM ordering.
    let (qc_bam_files, _) = unique_bam_files(
        args.wt_bam_files
            .iter()
            .chain(args.control_bam_files.iter())
            .cloned(),
    );
    let gene_qc = resolve_modality_gene_qc(
        &mut gff_map,
        &GeneQcRequest {
            bam_files: &qc_bam_files,
            cell_barcode_tag: &args.cell_barcode_tag,
            gene_barcode_tag: &args.gene_barcode_tag,
            umi_tag: resolve_umi_tag(args.no_umi_dedup, &args.umi_tag),
            gff_file: Some(&args.gff_file),
            gene_min_cells: args.gene_min_cells,
            gene_min_counts: args.gene_min_counts,
            cell_min_genes: args.cell_min_genes,
            cell_call: args.cell_qc.params(),
            valid_cells_file: args.valid_cells_file.as_deref(),
            valid_genes_file: args.valid_genes_file.as_deref(),
            skip_gene_qc: args.skip_gene_qc,
            persist: Some(GeneMatrixSink {
                output_dir: &args.output,
                backend: &args.backend,
                zip: args.zip,
            }),
        },
    )?;
    if gene_qc.is_some() && gff_map.is_empty() {
        info!("no genes passed QC");
        return Ok(());
    }

    // Load cell membership: from file, from clustering, or none
    let membership = if let Some(ref path) = args.cell_membership_file {
        // Load from file
        let m = CellMembership::from_file(
            path,
            args.membership_barcode_col,
            args.membership_celltype_col,
            !args.exact_barcode_match,
        )?;
        info!(
            "Loaded {} cell barcodes from membership file: {}",
            m.num_cells(),
            path
        );
        info!("Prefix matching: {}", !args.exact_barcode_match);
        Some(m)
    } else {
        // Generate membership by grouping cells for mass enrichment (returns None
        // when n_clusters <= 1). Reuse the gene-count matrices QC just persisted so
        // enrichment does not re-scan the BAMs. The instrument is built over the
        // signal (wt) cells here by design; the pipeline builds it over the
        // all-quant union instead (shared with ATOI) — see run_pipeline.
        let matrix_paths: Vec<Box<str>> = gene_qc
            .as_ref()
            .map(|q| q.matrix_by_batch.values().cloned().collect())
            .unwrap_or_default();
        args.enrich.build_membership(
            &signal_bam_files,
            &gff_map,
            &matrix_paths,
            &args.cell_barcode_tag,
            &args.gene_barcode_tag,
            !args.exact_barcode_match,
        )?
    };

    /////////////////////////////////
    // FIRST PASS: Find edit sites //
    /////////////////////////////////

    // Detect A-to-I editing sites first (if requested), or load pre-computed mask
    let atoi_params = args.atoi_params();
    let atoi_mask = if let Some(ref mask_file) = args.atoi_mask_file {
        // Load pre-computed A-to-I mask from parquet
        info!("Loading A-to-I mask from {}", mask_file);
        let mask = load_atoi_mask_from_parquet(mask_file.as_ref())?;
        info!("Loaded A-to-I mask with {} positions", mask.len());
        Some((None, mask))
    } else if args.detect_atoi {
        // Stratify the A-to-I masking pass by the same groups when enabled, so
        // cell-type-specific A-to-I edits are also masked out of m6A candidates.
        let atoi_sites = find_all_conversion_sites(&gff_map, &atoi_params, membership.as_ref())?;
        let n_atoi: usize = atoi_sites.iter().map(|x| x.value().len()).sum();
        info!("Found {} A-to-I editing sites", n_atoi);

        if !atoi_sites.is_empty() {
            ToParquet::to_parquet(
                &atoi_sites,
                &gff_map,
                format!("{}/atoi_sites.parquet", args.output),
            )?;
        }

        let mask = build_atoi_mask(&atoi_sites, &gff_map);
        info!("Built A-to-I mask with {} positions", mask.len());

        // Store atoi_sites for second-pass quantification
        Some((Some(atoi_sites), mask))
    } else {
        None
    };

    let mut m6a_params = ConversionParams::from(args);
    // Discovery contrasts signal vs control: override the wt arm to exclude
    // controls. `quant_bam_files` still unions the controls back in, so the
    // second pass quantifies them (once each) — see above.
    m6a_params.wt_bam_files = signal_bam_files;
    let gene_sites = find_all_conversion_sites(&gff_map, &m6a_params, membership.as_ref())?;

    // Apply A-to-I mask to filter m6A candidates
    if let Some((_, ref mask)) = atoi_mask {
        if !mask.is_empty() {
            let n_before: usize = gene_sites.iter().map(|x| x.value().len()).sum();
            filter_m6a_by_mask(&gene_sites, mask, &gff_map);
            let n_after: usize = gene_sites.iter().map(|x| x.value().len()).sum();
            info!(
                "A-to-I masking: {} → {} m6A sites ({} removed)",
                n_before,
                n_after,
                n_before - n_after
            );
        }
    }

    // Apply SNP mask if provided (after A-to-I masking)
    if let Some(ref mask_file) = args.snp_mask_file {
        info!("Loading SNP mask from {}", mask_file);
        let snp_mask = load_snp_mask_from_parquet(mask_file.as_ref())?;
        let n_before: usize = gene_sites.iter().map(|x| x.value().len()).sum();
        filter_m6a_by_mask(&gene_sites, &snp_mask, &gff_map);
        let n_after: usize = gene_sites.iter().map(|x| x.value().len()).sum();
        info!(
            "SNP masking: {} → {} m6A sites ({} removed)",
            n_before,
            n_after,
            n_before - n_after
        );
    }

    if gene_sites.is_empty() {
        info!("no sites found");
        return Ok(());
    }

    let ndata: usize = gene_sites.iter().map(|x| x.value().len()).sum();
    info!("Found {} m6A sites", ndata);

    gene_sites.to_parquet(&gff_map, format!("{}/m6a_sites.parquet", args.output))?;

    //////////////////////////////////////////
    // SECOND PASS: Collect cell-level data //
    //////////////////////////////////////////

    if args.output_bed_file {
        process_all_bam_files_to_bed(&m6a_params, &gene_sites, &gff_map, args.output_cell_types)?;
    } else {
        let valid_cells = gene_qc.as_ref().map(|qc| &qc.cells_by_batch);
        process_all_bam_files_to_backend(&m6a_params, &gene_sites, &gff_map, valid_cells)?;
    }

    // A-to-I second pass: quantify editing sites into sparse matrices
    if let Some((Some(ref atoi_sites), _)) = atoi_mask {
        if !atoi_sites.is_empty() {
            info!("Second pass: A-to-I count matrix");
            let valid_cells = gene_qc.as_ref().map(|qc| &qc.cells_by_batch);
            process_all_bam_files_to_backend(&atoi_params, atoi_sites, &gff_map, valid_cells)?;
        }
    }

    // Mixture model: cluster modification sites per gene
    if !args.no_mixture {
        info!("Running 1D Gaussian mixture model on m6A sites...");
        let mix_params = MixtureParams {
            min_sites: args.mixture_min_sites,
            max_k: args.mixture_max_k,
            bandwidth: args.mixture_bandwidth,
            drop_single_component: args.drop_single_component,
            ..Default::default()
        };
        let valid_cells = gene_qc.as_ref().map(|qc| &qc.cells_by_batch);
        run_mixture_model(&m6a_params, &gene_sites, &gff_map, &mix_params, valid_cells)?;
    }

    info!("done");
    Ok(())
}
