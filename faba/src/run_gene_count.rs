use crate::common::*;
use crate::data::util_htslib::*;

/// simply count the occurence of gene and cell barcode
#[derive(Args, Debug)]
pub struct GeneCountArgs {
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
        required = true,
        help = "Gene annotation file (GFF/GTF)",
        long_help = "Path to gene annotation file in GFF/GTF format.\n\
                     Used to define gene boundaries for read counting."
    )]
    pub(crate) gff_file: Box<str>,

    /// Cell barcode BAM tag
    #[arg(
        long,
        default_value = "CB",
        help = "Cell barcode BAM tag",
        long_help = "BAM tag for cell/sample barcode identification.\n\
                     Standard 10x Genomics tag is \"CB\"."
    )]
    pub(crate) cell_barcode_tag: Box<str>,

    /// Gene barcode BAM tag
    #[arg(
        long,
        default_value = "GX",
        help = "Gene barcode BAM tag",
        long_help = "BAM tag for gene identification.\n\
                     Standard 10x Genomics tag is \"GX\"."
    )]
    pub(crate) gene_barcode_tag: Box<str>,

    /// GFF record type filter
    #[arg(
        long,
        default_value = "gene",
        help = "GFF record type filter",
        long_help = "GFF feature type to use for counting.\n\
                     Common values: gene, transcript, exon, utr."
    )]
    pub(crate) record_type: Box<str>,

    /// Gene biotype filter
    #[arg(
        long,
        default_value = "",
        help = "Gene biotype filter (empty = all biotypes)",
        long_help = "Filter genes by biotype. Empty (default) keeps all biotypes.\n\
                     Pass a value to restrict: protein_coding, pseudogene, lncRNA."
    )]
    pub(crate) gene_type: Box<str>,

    /// Minimum non-zero entries per row (gene) to keep
    #[arg(
        short,
        long,
        default_value_t = 10,
        help = "Minimum non-zeros per row (gene)",
        long_help = "Genes with fewer than this many non-zero cells are removed\n\
                     from the output matrix."
    )]
    pub(crate) row_nnz_cutoff: usize,

    /// Minimum non-zero entries per column (cell) to keep
    #[arg(
        short,
        long,
        default_value_t = 10,
        help = "Minimum non-zeros per column (cell)",
        long_help = "Cells with fewer than this many non-zero genes are removed\n\
                     from the output matrix."
    )]
    pub(crate) column_nnz_cutoff: usize,

    #[command(flatten)]
    pub(crate) cell_qc: crate::cell_qc::CellQcArgs,

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

    #[arg(
        long = "no-zip",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive",
        long_help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive\n\
                     (zarr backend only; no effect on hdf5)"
    )]
    pub(crate) zip: bool,

    /// Output directory
    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Directory for output files.\n\
                     One sparse matrix file per input BAM is created here."
    )]
    pub(crate) output: Box<str>,

    /// Disable spliced/unspliced separation (output total counts only)
    #[arg(
        long = "no-splice",
        default_value_t = false,
        help = "Disable spliced/unspliced separation",
        long_help = "By default, faba produces separate spliced and unspliced\n\
                     count matrices. Use this flag to output a single total\n\
                     count matrix instead."
    )]
    pub(crate) no_splice: bool,

    /// Exon boundary buffer zone (bp) for splice classification
    #[arg(
        long,
        default_value_t = 3,
        help = "Intron buffer zone (bp) for splice classification",
        long_help = "Number of base pairs around exon boundaries to treat as\n\
                     ambiguous when classifying reads as spliced or unspliced.\n\
                     Reads falling entirely within this buffer are discarded."
    )]
    pub(crate) intron_buffer: i64,

    /// UMI BAM tag used for read deduplication
    #[arg(
        long = "umi-tag",
        default_value = "UB",
        help = "UMI BAM tag (for read dedup)",
        long_help = "BAM tag holding the corrected UMI. Counts collapse to one\n\
                     per (cell, gene, UMI) — matching Cell Ranger's molecule\n\
                     counting. Standard 10x tag is \"UB\". Reads without this tag\n\
                     are counted individually."
    )]
    pub(crate) umi_tag: Box<str>,

    /// Disable UMI deduplication (count reads instead of molecules)
    #[arg(
        long = "no-umi-dedup",
        default_value_t = false,
        help = "Disable UMI deduplication (count reads, not molecules)",
        long_help = "By default faba collapses reads sharing a (cell, gene, UMI)\n\
                     into a single count (molecule counting, like Cell Ranger).\n\
                     Use this flag to count every non-duplicate read instead."
    )]
    pub(crate) no_umi_dedup: bool,

    /// Mitochondrial chromosome name(s), comma-separated
    #[arg(
        long = "mito-chr",
        default_value = "chrM,chrMT,MT,M",
        help = "Mitochondrial chromosome name(s) (comma-separated)",
        long_help = "Genes on these chromosomes are treated as mitochondrial:\n\
                     excluded from the count matrix (unless --keep-mito) and\n\
                     summarized in the per-cell MT-fraction QC. Matched\n\
                     case-insensitively against the GFF seqname."
    )]
    pub(crate) mito_chr: Box<str>,

    /// Keep mitochondrial genes in the count matrix (default: exclude)
    #[arg(
        long = "keep-mito",
        default_value_t = false,
        help = "Keep mitochondrial genes in the count matrix",
        long_help = "By default mitochondrial genes are dropped from the output\n\
                     matrix (their per-cell MT fraction is still reported as QC).\n\
                     Use this flag to retain them in the matrix."
    )]
    pub(crate) keep_mito: bool,

    /// Max mitochondrial fraction per cell (0 = data-driven elbow cutoff)
    #[arg(
        long = "max-mito-frac",
        default_value_t = 0.0,
        help = "Max MT fraction per cell: >0 = fixed cutoff; 0 = elbow cutoff",
        long_help = "Cells whose mitochondrial UMI fraction exceeds the cutoff are\n\
                     removed during QC. A value > 0 is a fixed cutoff; the default 0\n\
                     uses a data-driven elbow cutoff on the MT% distribution (drops\n\
                     the high-MT burst tail). See --no-mito-cell-qc to disable."
    )]
    pub(crate) max_mito_frac: f64,

    /// Disable mitochondrial cell QC (report MT% only, drop no cells)
    #[arg(
        long = "no-mito-cell-qc",
        default_value_t = false,
        help = "Disable MT cell QC (report MT% only, drop no cells)",
        long_help = "Report per-cell MT% but drop no cells. Mitochondrial genes are\n\
                     still excluded from the matrix unless --keep-mito."
    )]
    pub(crate) no_mito_cell_qc: bool,
}

impl GeneCountArgs {
    /// Resolve the UMI tag for dedup: `None` disables it (count reads).
    pub(crate) fn umi_dedup_tag(&self) -> Option<&[u8]> {
        crate::pipeline_util::resolve_umi_tag(self.no_umi_dedup, &self.umi_tag)
    }
}

pub fn run_gene_count(args: &GeneCountArgs) -> anyhow::Result<()> {
    if args.bam_files.is_empty() {
        return Err(anyhow::anyhow!("need bam files"));
    }

    for x in args.bam_files.iter() {
        check_bam_index(x, None)?;
    }

    info!("data files:");
    for x in args.bam_files.iter() {
        info!("{}", x);
    }

    let batch_names = uniq_batch_names(&args.bam_files)?;
    std::fs::create_dir_all(args.output.as_ref())?;

    let backend = args.backend.clone();
    info!("parsing GFF file: {}", args.gff_file);

    if args.no_splice {
        crate::gene_count::pipeline::run_simple(args, &backend, &batch_names)
    } else {
        crate::gene_count::pipeline::run_splice_aware(args, &backend, &batch_names)
    }
}
