use crate::common::*;
use crate::data::util_htslib::*;
use crate::gene_count::splice::CountReadOpts;

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
        long_help = "Genes with fewer than this many non-zero cells are removed from the output matrix."
    )]
    pub(crate) row_nnz_cutoff: usize,

    /// Minimum non-zero entries per column (cell) to keep
    #[arg(
        short,
        long,
        default_value_t = 10,
        help = "Minimum non-zeros per column (cell)",
        long_help = "Cells with fewer than this many non-zero genes are removed from the output matrix."
    )]
    pub(crate) column_nnz_cutoff: usize,

    /// Minimum mapping quality for a read to be counted
    #[arg(
        long = "min-mapping-quality",
        default_value_t = 20,
        help = "Minimum mapping quality (MAPQ) to count a read",
        long_help = "Reads below this MAPQ are not counted.\n\
                     Secondary and supplementary alignments are always skipped.\n\
                     Cell Ranger marks a unique, confident alignment MAPQ 255,\n\
                     so the default admits those and drops multi-mappers.\n\
                     Pass 0 to count every alignment regardless of MAPQ."
    )]
    pub(crate) min_mapping_quality: u8,

    #[command(flatten)]
    pub(crate) cell_qc: crate::cell_qc::CellQcArgs,

    /// Sparse matrix output backend
    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix output backend",
        long_help = "File format for the output sparse matrix. Supported: zarr, hdf5."
    )]
    pub(crate) backend: SparseIoBackend,

    #[arg(
        long = "no-zip",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive",
        long_help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive.\n\
                     Zarr backend only; no effect on hdf5."
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
        long_help = "By default faba collapses reads that share a (cell, gene, UMI),\n\
                     into a single count: molecule counting, like Cell Ranger.\n\
                     Use this flag to count every non-duplicate read instead."
    )]
    pub(crate) no_umi_dedup: bool,

    #[command(flatten)]
    pub(crate) mito_qc: crate::quant::MitoQcArgs,
}

impl GeneCountArgs {
    /// Resolve the UMI tag for dedup: `None` disables it (count reads).
    pub(crate) fn umi_dedup_tag(&self) -> Option<&[u8]> {
        crate::quant::resolve_umi_tag(self.no_umi_dedup, &self.umi_tag)
    }

    /// The admission policy this run hands the shared counting loop, so
    /// `faba genes` and the gene QC pass behind each modality cannot diverge on
    /// tags or on which reads they trust.
    pub(crate) fn count_read_opts(&self) -> CountReadOpts<'_> {
        CountReadOpts {
            cell_barcode_tag: &self.cell_barcode_tag,
            gene_barcode_tag: &self.gene_barcode_tag,
            umi_tag: self.umi_dedup_tag(),
            min_mapping_quality: self.min_mapping_quality,
        }
    }
}

/// Count genes into one `{batch}_genes` matrix per BAM, spliced and unspliced
/// rows in the same feature axis.
///
/// This is [`crate::quant::run_gene_count_qc`] with the standalone command's
/// knobs — the same call `faba all` and the modality QC make, so all three
/// agree on which cells and genes survive and on what a gene-count matrix
/// looks like. It used to be a second copy of that loop with its own writers,
/// emitting a `{batch}` (total), `{batch}_spliced` and `{batch}_unspliced`
/// triple per BAM; the total was just the other two summed, nothing read the
/// split halves, and a `{gene}/count/total` row is rejected downstream anyway
/// (see `auxiliary_data::feature_rows::split_count_row`) precisely because it
/// double-counts the gene.
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

    std::fs::create_dir_all(args.output.as_ref())?;
    info!("parsing GFF file: {}", args.gff_file);

    crate::quant::run_gene_count_qc(
        args.gff_file.as_ref(),
        &crate::quant::GeneQcRequest {
            bam_files: &args.bam_files,
            count: args.count_read_opts(),
            gff_file: Some(args.gff_file.as_ref()),
            output_dir: &args.output,
            gene_type: &args.gene_type,
            gene_min_cells: args.row_nnz_cutoff,
            // No total-count floor on this command; `-r/--row-nnz-cutoff` is the
            // only gene threshold it exposes.
            gene_min_counts: 0,
            cell_min_genes: args.column_nnz_cutoff,
            cell_call: args.cell_qc.params(),
            mito: args.mito_qc.params(),
            valid_cells_file: None,
            valid_genes_file: None,
            skip_gene_qc: false,
            persist: Some(crate::quant::GeneMatrixSink {
                backend: &args.backend,
                zip: args.zip,
            }),
        },
    )?;

    Ok(())
}
