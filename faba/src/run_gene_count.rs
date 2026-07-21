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
        long_help = "By default, faba produces separate spliced and unspliced count matrices.\n\
                     Use this flag to output a single total count matrix instead."
    )]
    pub(crate) no_splice: bool,

    /// UMI BAM tag used for read deduplication
    #[arg(
        long = "umi-tag",
        default_value = "UB",
        help = "UMI BAM tag (for read dedup)",
        long_help = "BAM tag holding the corrected UMI.\n\
                     Counts collapse to one per (cell, gene, UMI)\n\
                     — matching Cell Ranger's molecule counting.\n\
                     Standard 10x tag is \"UB\".\n\
                     Reads without this tag are counted individually."
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

    #[command(flatten)]
    pub(crate) mito_qc: crate::pipeline_util::MitoQcArgs,
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
