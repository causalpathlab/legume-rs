use crate::apa::em::*;
use crate::apa::likelihood::*;
use crate::common::*;
use crate::data::poly_a_stat_map::PolyASiteArgs;
use crate::data::util_htslib::*;

use genomic_data::gff::FeatureType as GffFeatureType;
use genomic_data::gff::GeneType as GffGeneType;
use rayon::ThreadPoolBuilder;

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
    pub(crate) bam_files: Vec<Box<str>>,

    /// Gene annotation file (GFF/GTF)
    #[arg(
        short = 'g',
        long = "gff",
        help = "Gene annotation file (GFF/GTF)",
        long_help = "Path to gene annotation file in GFF/GTF format.\n\
                     Required for simple mode; in mixture mode, used to\n\
                     extract 3'-UTR regions unless --utr-bed is provided."
    )]
    pub(crate) gff_file: Option<Box<str>>,

    /// Cell barcode BAM tag
    #[arg(
        long,
        default_value = "CB",
        help = "Cell barcode BAM tag",
        long_help = "BAM tag for cell/sample barcode identification.\n\
                     Standard 10x Genomics tag is \"CB\"."
    )]
    pub(crate) cell_barcode_tag: Box<str>,

    /// Minimum soft-clipped poly(A) tail length
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum poly(A) tail length (bp)",
        long_help = "Minimum number of soft-clipped A/T bases required to\n\
                     call a read as a poly(A) junction read."
    )]
    pub(crate) polya_min_tail_length: usize,

    /// Maximum non-A/T bases allowed in poly(A) tail
    #[arg(
        long,
        default_value_t = 3,
        help = "Max non-A/T bases in poly(A) tail",
        long_help = "Maximum number of non-A (forward) or non-T (reverse)\n\
                     bases allowed in the soft-clipped tail."
    )]
    pub(crate) polya_max_non_a_or_t: usize,

    /// Internal priming check window size (bp)
    #[arg(
        long,
        default_value_t = 10,
        help = "Internal priming check window (bp)",
        long_help = "Window size in base pairs around the cleavage site to\n\
                     check for genomic A/T-rich stretches (internal priming)."
    )]
    pub(crate) polya_internal_prime_window: usize,

    /// A/T count threshold to flag internal priming
    #[arg(
        long,
        default_value_t = 7,
        help = "A/T count threshold for internal priming",
        long_help = "If the number of A/T bases in the internal priming\n\
                     window meets or exceeds this threshold, the site is\n\
                     flagged as likely internal priming and discarded."
    )]
    pub(crate) polya_internal_prime_count: usize,

    /// Minimum read coverage at a candidate site
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum read coverage per site",
        long_help = "Candidate poly(A) sites with fewer than this many\n\
                     supporting reads are discarded."
    )]
    pub(crate) min_coverage: usize,

    /// Maximum number of threads
    #[arg(
        long,
        default_value_t = 16,
        help = "Maximum number of threads",
        long_help = "Maximum number of threads for parallel processing.\n\
                     Capped by the number of available CPUs."
    )]
    pub(crate) max_threads: usize,

    /// Minimum non-zero entries per row (site) to keep
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum non-zeros per row (site)",
        long_help = "Sites with fewer than this many non-zero cells are\n\
                     removed from the output matrix."
    )]
    pub(crate) row_nnz_cutoff: usize,

    /// Minimum non-zero entries per column (cell) to keep
    #[arg(
        long,
        default_value_t = 1,
        help = "Minimum non-zeros per column (cell)",
        long_help = "Cells with fewer than this many non-zero sites are\n\
                     removed from the output matrix."
    )]
    pub(crate) column_nnz_cutoff: usize,

    /// Output directory
    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Directory for output files.\n\
                     In simple mode, one sparse matrix per input BAM is created.\n\
                     In mixture mode, a single sparse matrix and a site\n\
                     annotation parquet are created for all inputs."
    )]
    pub(crate) output: Box<str>,

    /// Sparse matrix output backend
    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix output backend",
        long_help = "File format for the output sparse matrix.\n\
                     Supported: zarr, hdf5."
    )]
    pub(crate) backend: SparseIoBackend,

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
    pub(crate) method: ApaMethod,

    // --- A-to-I mask (shared between simple and mixture) ---
    /// Pre-computed A-to-I mask parquet file
    #[arg(
        long = "atoi-mask",
        help = "A-to-I mask parquet (from `faba atoi` or `faba dart --detect-atoi`)",
        long_help = "Path to a pre-computed A-to-I sites parquet file.\n\
                     When provided, poly(A) sites that overlap A-to-I editing\n\
                     positions are removed before quantification."
    )]
    pub(crate) atoi_mask_file: Option<Box<str>>,

    // --- Simple-mode args ---
    /// Gene barcode BAM tag (simple mode)
    #[arg(
        long,
        default_value = "GX",
        help = "Gene barcode BAM tag (simple mode)",
        long_help = "BAM tag for gene identification. Only used in simple mode.\n\
                     Standard 10x Genomics tag is \"GX\"."
    )]
    pub(crate) gene_barcode_tag: Box<str>,

    /// Bin resolution in bp for poly(A) site grouping (simple mode)
    #[arg(
        long,
        default_value_t = 10,
        help = "Bin resolution in bp (simple mode)",
        long_help = "Nearby poly(A) sites within this distance in base pairs\n\
                     are grouped into a single bin. Only used in simple mode."
    )]
    pub(crate) resolution_bp: usize,

    /// Include reads without cell barcode (simple mode)
    #[arg(
        long,
        default_value_t = false,
        help = "Include reads without barcode (simple mode)",
        long_help = "Include reads missing a cell barcode tag in the count.\n\
                     Only used in simple mode."
    )]
    pub(crate) include_missing_barcode: bool,

    /// GFF record type filter (simple mode)
    #[arg(
        long,
        value_enum,
        help = "GFF record type filter (simple mode)",
        long_help = "Filter GFF records by feature type. Only used in simple mode.\n\
                     Common values: gene, transcript, exon, utr."
    )]
    pub(crate) record_type: Option<GffFeatureType>,

    /// Gene biotype filter (simple mode)
    #[arg(
        long,
        value_enum,
        help = "Gene biotype filter (simple mode)",
        long_help = "Filter genes by biotype. Only used in simple mode.\n\
                     Common values: protein_coding, pseudogene, lncRNA."
    )]
    pub(crate) gene_type: Option<GffGeneType>,

    // --- Mixture-mode (SCAPE) args ---
    /// 3'-UTR regions BED file (mixture mode)
    #[arg(
        short = 'u',
        long = "utr-bed",
        help = "3'-UTR regions BED file (mixture mode)",
        long_help = "BED file defining 3'-UTR regions. Alternative to --gff\n\
                     for mixture mode. Each row should be a UTR interval."
    )]
    pub(crate) utr_bed: Option<Box<str>>,

    /// Minimum 3'-UTR length in bp (mixture mode)
    #[arg(
        long,
        default_value_t = 200,
        help = "Minimum 3'-UTR length in bp (mixture mode)",
        long_help = "UTRs shorter than this are skipped. Only used in mixture mode."
    )]
    pub(crate) min_utr_length: usize,

    /// Pre-identified pA sites BED file (mixture mode)
    #[arg(
        long,
        help = "Pre-identified pA sites BED (mixture mode)",
        long_help = "BED file of known poly(A) sites. When provided, skips\n\
                     de novo site discovery. Only used in mixture mode."
    )]
    pub(crate) pre_sites: Option<Box<str>>,

    /// UMI BAM tag (mixture mode)
    #[arg(
        long,
        default_value = "UB",
        help = "UMI BAM tag (mixture mode)",
        long_help = "BAM tag for unique molecular identifiers.\n\
                     Standard 10x Genomics tag is \"UB\".\n\
                     Only used in mixture mode."
    )]
    pub(crate) umi_tag: Box<str>,

    /// Expected fragment length mean (mixture mode)
    #[arg(
        long,
        default_value_t = 300.0,
        help = "Fragment length mean, mu_f (mixture mode)",
        long_help = "Expected mean fragment length in base pairs (mu_f in SCAPE).\n\
                     Only used in mixture mode."
    )]
    pub(crate) mu_f: f32,

    /// Fragment length standard deviation (mixture mode)
    #[arg(
        long,
        default_value_t = 50.0,
        help = "Fragment length s.d., sigma_f (mixture mode)",
        long_help = "Expected fragment length standard deviation (sigma_f in SCAPE).\n\
                     Only used in mixture mode."
    )]
    pub(crate) sigma_f: f32,

    /// pA site position step size in bp (mixture mode)
    #[arg(
        long,
        default_value_t = 10,
        help = "pA site enumeration step (bp, mixture mode)",
        long_help = "Step size in base pairs for enumerating candidate pA site\n\
                     positions along each UTR. Only used in mixture mode."
    )]
    pub(crate) theta_step: usize,

    /// Maximum pA site dispersion (mixture mode)
    #[arg(
        long,
        default_value_t = 70.0,
        help = "Max pA dispersion beta (mixture mode)",
        long_help = "Upper bound on the dispersion parameter (beta) for each\n\
                     poly(A) site component. Only used in mixture mode."
    )]
    pub(crate) max_beta: f32,

    /// Minimum pA site dispersion (mixture mode)
    #[arg(
        long,
        default_value_t = 10.0,
        help = "Min pA dispersion beta (mixture mode)",
        long_help = "Lower bound on the dispersion parameter (beta) for each\n\
                     poly(A) site component. Only used in mixture mode."
    )]
    pub(crate) min_beta: f32,

    /// Minimum component weight (mixture mode)
    #[arg(
        long,
        default_value_t = 0.01,
        help = "Min component weight (mixture mode)",
        long_help = "Components with weight below this threshold are pruned\n\
                     during EM iterations. Only used in mixture mode."
    )]
    pub(crate) min_ws: f32,

    /// Minimum fragments per UTR (mixture mode)
    #[arg(
        long,
        default_value_t = 50,
        help = "Min fragments per UTR (mixture mode)",
        long_help = "UTRs with fewer than this many fragments are skipped.\n\
                     Only used in mixture mode."
    )]
    pub(crate) min_fragments: usize,

    /// Merge distance for nearby pA sites in bp (mixture mode)
    #[arg(
        long,
        default_value_t = 50.0,
        help = "pA site merge distance (bp, mixture mode)",
        long_help = "Candidate pA sites within this distance are merged into\n\
                     a single site. Only used in mixture mode."
    )]
    pub(crate) merge_distance: f32,

    /// Compute PDUI (Percentage of Distal poly(A) site Usage Index)
    #[arg(
        long = "compute-pdui",
        default_value_t = false,
        help = "Compute PDUI for genes with exactly 2 active pA sites",
        long_help = "Compute PDUI (Percentage of Distal poly(A) site Usage Index)\n\
                     for genes with exactly 2 active poly(A) sites after EM.\n\
                     PDUI = distal_count / (distal + proximal).\n\
                     Outputs a sparse (genes x cells) PDUI matrix.\n\
                     Only used in mixture mode."
    )]
    pub(crate) compute_pdui: bool,
}

impl CountApaArgs {
    pub(crate) fn lik_params(&self) -> LikelihoodParams {
        LikelihoodParams {
            mu_f: self.mu_f,
            sigma_f: self.sigma_f,
            theta_step: self.theta_step,
            ..Default::default()
        }
    }

    pub(crate) fn em_params(&self) -> EmParams {
        EmParams {
            min_weight: self.min_ws,
            ..Default::default()
        }
    }

    pub(crate) fn backend_file_path(&self, name: &str) -> Box<str> {
        match self.backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &self.output, name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &self.output, name),
        }
        .into_boxed_str()
    }

    pub(crate) fn qc_cutoffs(&self) -> SqueezeCutoffs {
        SqueezeCutoffs {
            row: self.row_nnz_cutoff,
            column: self.column_nnz_cutoff,
        }
    }

    pub(crate) fn polya_site_args(&self) -> PolyASiteArgs {
        PolyASiteArgs {
            min_tail_length: self.polya_min_tail_length,
            max_non_a_or_t_bases: self.polya_max_non_a_or_t,
            internal_prime_in: self.polya_internal_prime_window,
            internal_prime_a_or_t_count: self.polya_internal_prime_count,
        }
    }
}

/// Detect and quantify alternative polyadenylation (APA) sites
pub fn run_apa(args: &CountApaArgs) -> anyhow::Result<()> {
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
        ApaMethod::Simple => crate::apa::pipeline::run_simple(args),
        ApaMethod::Mixture => crate::apa::pipeline::run_mixture(args),
    }
}
