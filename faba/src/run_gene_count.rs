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
        default_value = "protein_coding",
        help = "Gene biotype filter",
        long_help = "Filter genes by biotype.\n\
                     Common values: protein_coding, pseudogene, lncRNA."
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

    /// Count spliced and unspliced reads separately
    #[arg(
        long,
        default_value_t = false,
        help = "Count spliced/unspliced separately (RNA velocity)",
        long_help = "Produce separate spliced and unspliced count matrices\n\
                     for RNA velocity analysis. Requires exon annotations in\n\
                     the GFF file."
    )]
    pub(crate) splice: bool,

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

    if args.splice {
        crate::gene_count::pipeline::run_splice_aware(args, &backend, &batch_names)
    } else {
        crate::gene_count::pipeline::run_simple(args, &backend, &batch_names)
    }
}
