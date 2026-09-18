use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::editing::bed_output::process_all_bam_files_to_bed;
use crate::editing::io::ToParquet;
use crate::editing::mixture::MixtureParams;
use crate::editing::mixture_pipeline::run_mixture_model;
use crate::editing::pipeline::{
    find_all_conversion_sites, process_all_bam_files_to_backend, ConversionParams,
};
use crate::editing::sifter::ModificationType;
use crate::gene_count::splice::CountReadOpts;
use crate::quant::{
    check_all_bam_indices, resolve_modality_gene_qc, resolve_umi_tag, GeneMatrixSink, GeneQcRequest,
};

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
                     These contain the C->T conversions at m6A sites.\n\
                     Each motif C is called by a WT-vs-MUT contrast against the --control-bam samples.\n\
                     A genomic C/T variant converts equally in both arms and is rejected."
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
        long_help = "Comma-separated list of control (catalytically-dead YTHmut) BAM files,\n\
                     pooled into one background.\n\
                     m6A is called where the signal BAMs show significantly higher C->T conversion than these controls,\n\
                     by a two-sample test. Required:\n\
                     m6A cannot be distinguished from genomic variation without a control."
    )]
    pub control_bam_files: Vec<Box<str>>,

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
        long_help = "Cell barcode tag used for cell/sample identification in 10x Genomics BAM files.\n\
                     [See here](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam)"
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
        default_value_t = crate::editing::pipeline::DEFAULT_M6A_MIN_COVERAGE,
        help = "Minimum total reads (signal + control) per site",
        long_help = "Minimum total reads (signal + control) at a site for it to be written.\n\
                     \n\
                     This is the ONLY floor the producer applies, and it defaults to 1.\n\
                     Together with --min-conversion 1 it excludes exactly the sites\n\
                     with no converted read at all, i.e. empty rows.\n\
                     Every other decision (p-value, odds ratio, control depth,\n\
                     edit ratio, cells per site) is a column in m6a_sites.parquet\n\
                     and a flag on `faba qc`, so it can be revisited without a rerun.\n\
                     \n\
                     Null-cell QC removes cells that never edit before discovery runs,\n\
                     so the signal-arm depth here is the de-diluted depth."
    )]
    pub min_coverage: usize,

    #[arg(
        long = "min-conversion",
        default_value_t = crate::editing::pipeline::DEFAULT_M6A_MIN_CONVERSION,
        help = "Minimum converted (C->T) signal reads per site",
        long_help = "Minimum converted (C->T) signal reads at a site for it to be written.\n\
                     At the default of 1 a single converted read makes a site.\n\
                     Raise it only to bound the candidate set; thresholding evidence\n\
                     is `faba qc`'s job (--site-min-converted, --site-max-pv, ...)."
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
        long,
        value_enum,
        help = "Gene type (protein_coding, pseudogene, lncRNA)",
        long_help = "Filter analysis by gene type. Options include protein_coding, pseudogene,\n\
                     or lncRNA."
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
        long_help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive.\n\
                     Zarr backend only; no effect on hdf5."
    )]
    pub zip: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Include reads w/o barcode info",
        long_help = "Include reads that are missing gene and cell barcode information in the analysis."
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
        long_help = "Path to reference genome in FASTA format (.fa or .fasta).\n\
                     Used to validate base calls at editing sites. File must be indexed (.fai).\n\
                     If index doesn't exist, one will be created. Example: genome.fa"
    )]
    pub genome_file: Box<str>,

    #[arg(
        long = "cell-membership",
        alias = "barcode-membership",
        alias = "membership",
        help = "Cell barcode membership file for filtering cells (TSV, CSV, or Parquet)",
        long_help = "Path to cell barcode membership file for restricting analysis to specific cells.\n\
                     Format: First column = cell barcode, Second column = cell type.\n\
                     Supports .tsv, .csv, .parquet, and .gz variants.\n\
                     Only cells (barcodes) present in this file will be included in analysis.\n\
                     By default,\n\
                     barcodes are prefix-matched (use --exact-barcode-match to change)."
    )]
    pub cell_membership_file: Option<Box<str>>,

    #[arg(
        long = "membership-barcode-col",
        default_value_t = 0,
        help = "Column index for cell barcodes in membership file",
        long_help = "Zero-based column index for the cell barcode field in the membership file."
    )]
    pub membership_barcode_col: usize,

    #[arg(
        long = "membership-celltype-col",
        default_value_t = 1,
        help = "Column index for cell types in membership file",
        long_help = "Zero-based column index for the cell type field in the membership file."
    )]
    pub membership_celltype_col: usize,

    #[arg(
        long = "exact-barcode-match",
        default_value_t = false,
        help = "Require exact cell barcode matching",
        long_help = "By default, membership barcodes are matched as prefixes of BAM barcodes\n\
                     (handles suffixes like \"-1\").\n\
                     Enable this flag to require exact string matching."
    )]
    pub exact_barcode_match: bool,

    #[arg(
        long = "output-cell-types",
        default_value_t = false,
        help = "Include cell type annotation in BED output",
        long_help = "Append a cell type column to BED output lines (--output-bed-file only).\n\
                     Requires --cell-membership: the column is read from that file.\n\
                     Every row is written as \"unknown\" without it."
    )]
    pub output_cell_types: bool,

    #[arg(
        long = "no-check-r-site",
        default_value_t = false,
        help = "Disable R site (RAC/GTY) validation in reference",
        long_help = "By default,\n\
                     faba validates the R position in RAC/GTY motifs against the reference genome.\n\
                     It requires R=A/G on the forward strand, Y=C/T on the reverse.\n\
                     Use this flag to disable that check."
    )]
    pub no_check_r_site: bool,

    ///////////////////////////
    // Mixture model options //
    ///////////////////////////
    #[arg(
        long = "mixture",
        default_value_t = false,
        help = "Also fit the per-gene 1D Gaussian mixture of site positions (slow EM)",
        long_help = "Also fit a 1D Gaussian mixture (+ uniform noise) to each gene's putative site positions,\n\
                     with components called as modes of the bandwidth-smoothed site pileup
\
                     (--mixture-bandwidth, capped by --mixture-max-k), and write a sparse (cells x components) matrix\n\
                     plus m6a_components.parquet. Off by default, matching `faba all`.\n\
                     The fit runs over EVERY putative site, posterior-weighted by evidence,\n\
                     since the producer applies no p-value cutoff."
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
        help = "Gaussian bandwidth (nt) for component calling; 0 = auto (data-derived)",
        long_help = "Gaussian smoothing bandwidth in nucleotides used to call mixture components:\n\
                     the per-gene signal pileup is smoothed at this bandwidth and its modes become components.\n\
                     0 (default) derives a global per-modality bandwidth from the empirical site spacing."
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
                     This prevents low-coverage 1-of-1 sites from dominating μ/σ.\n\
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
        help = "Min cells per gene for expression QC; 1 = drop only empty rows",
        long_help = "Minimum number of cells with non-zero expression for a gene to pass QC.\n\
                     Genes below this threshold are excluded before site discovery.\n\
                     The default of 1 drops only genes with no counts at all;\n\
                     an opinionated floor belongs to `faba qc --row-nnz-cutoff`."
    )]
    pub gene_min_cells: usize,

    #[arg(
        long = "gene-min-counts",
        default_value_t = 0,
        help = "Min total UMI counts per gene for expression QC (0 disables)",
        long_help = "Minimum total UMI counts (summed across all cells) for a gene to pass QC.\n\
                     Genes below this threshold are excluded before site discovery.\n\
                     0 disables the threshold."
    )]
    pub gene_min_counts: usize,

    #[arg(
        long = "cell-min-genes",
        default_value_t = 1,
        help = "Min genes per cell for expression QC; 1 = drop only empty columns",
        long_help = "Minimum number of genes with non-zero expression for a cell to pass QC.\n\
                     Cells below this threshold are excluded from quantification.\n\
                     The default of 1 drops only cells with no counts at all;\n\
                     an opinionated floor belongs to `faba qc`."
    )]
    pub cell_min_genes: usize,

    #[arg(
        long = "skip-gene-qc",
        default_value_t = false,
        help = "Skip gene expression QC step",
        long_help = "Skip the a priori gene expression QC step. By default,\n\
                     faba counts reads per gene and filters to expressed genes/cells before site discovery."
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

    #[command(flatten)]
    pub cell_scan: crate::editing::cell_activity::CellScanArgs,

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
            error_rate: crate::editing::pipeline::DEFAULT_EDIT_ERROR_RATE,
            overdispersion: crate::editing::pipeline::DEFAULT_EDIT_OVERDISPERSION,
            backend: args.backend.clone(),
            zip: args.zip,
            output: args.output.clone(),
            cell_membership_file: args.cell_membership_file.clone(),
            membership_barcode_col: args.membership_barcode_col,
            membership_celltype_col: args.membership_celltype_col,
            exact_barcode_match: args.exact_barcode_match,
            mod_type: ModificationType::M6A {
                check_r_site: !args.no_check_r_site,
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
            competent_cells: None,
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

    // Cell membership restricts every pass to the listed barcodes, and comes only
    // from `--cell-membership`. There is no derived grouping: cells were once
    // auto-grouped by expression (Leiden, `--cluster-resolution`) on the theory
    // that a cell-type-specific edit is diluted by pooling, but discovery pools
    // the marginal either way (see `find_sites_with_celltype_stats`), so the
    // groups changed no test. The dilution that mattered was catalytic
    // competence, not cell type, and `cell_activity` removes it directly.
    let membership = match args.cell_membership_file {
        Some(ref path) => {
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
        }
        None => None,
    };

    /////////////////////////////////
    // FIRST PASS: Find edit sites //
    /////////////////////////////////

    let mut m6a_params = ConversionParams::from(args);
    // Discovery contrasts signal vs control: override the wt arm to exclude
    // controls. `quant_bam_files` still unions the controls back in, so the
    // second pass quantifies them (once each) — see above.
    m6a_params.wt_bam_files = signal_bam_files;

    // Null-cell QC, BEFORE discovery. Cells that edit no more than the control
    // does are dropped from the SIGNAL arm only, so every site's WT counts are
    // de-diluted as they are first computed and nothing downstream changes.
    m6a_params.competent_cells = crate::editing::cell_activity::call_and_report(
        &gff_map,
        &m6a_params,
        &args.cell_scan,
        // Null-cell calling is the LAST cell-QC stage: it asks "does this real
        // cell edit?", which only means anything for barcodes that already
        // passed droplet calling. Scanning every barcode would score competence
        // on ambient droplets and let their reads into discovery.
        gene_qc.as_ref(),
        &args.output,
        "m6a",
    )?;

    // Every putative site, with its statistics. Nothing is tested here: the
    // p-value, odds ratio, control depth and edit ratio are columns, and
    // `faba qc` is where they become thresholds.
    let gene_sites = find_all_conversion_sites(&gff_map, &m6a_params, membership.as_ref())?;

    if gene_sites.is_empty() {
        info!("no sites found");
        return Ok(());
    }

    let ndata: usize = gene_sites.iter().map(|x| x.value().len()).sum();
    info!("Found {} putative m6A sites", ndata);

    gene_sites.to_parquet(
        &gff_map,
        &spliced,
        format!("{}/m6a_sites.parquet", args.output),
    )?;

    //////////////////////////////////////////
    // SECOND PASS: Collect cell-level data //
    //////////////////////////////////////////

    if args.output_bed_file {
        process_all_bam_files_to_bed(&m6a_params, &gene_sites, &gff_map, args.output_cell_types)?;
    } else {
        // Default: the matrices keep every QC cell, so they stay
        // column-compatible with the other modalities. `--quantify-competent-only`
        // trades that for a de-diluted matrix.
        let restricted = crate::editing::cell_activity::cells_for_quantification(
            m6a_params.competent_cells.as_ref(),
            gene_qc.as_ref().map(|qc| &qc.cells_by_batch),
            args.cell_scan.quantify_competent_only,
        );
        let valid_cells = restricted
            .as_ref()
            .or_else(|| gene_qc.as_ref().map(|qc| &qc.cells_by_batch));
        process_all_bam_files_to_backend(&m6a_params, &gene_sites, &gff_map, valid_cells)?;
    }

    // Mixture model (opt-in): cluster modification sites per gene
    if args.mixture {
        info!("Running 1D Gaussian mixture model on m6A sites...");
        let mix_params = MixtureParams {
            min_sites: args.mixture_min_sites,
            max_k: args.mixture_max_k,
            bandwidth: args.mixture_bandwidth,
            drop_single_component: args.drop_single_component,
            ..Default::default()
        };
        let valid_cells = gene_qc.as_ref().map(|qc| &qc.cells_by_batch);
        run_mixture_model(
            &m6a_params,
            &gene_sites,
            &gff_map,
            &spliced,
            &mix_params,
            valid_cells,
        )?;
    }

    info!("done");
    Ok(())
}
