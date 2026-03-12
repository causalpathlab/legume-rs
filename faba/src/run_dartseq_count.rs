use crate::cell_clustering::{cluster_cells_from_bam, ClusteringParams};
use crate::common::*;
use crate::dartseq::io::ToParquet;
use crate::dartseq::sifter::*;
use crate::dartseq::stat::Histogram;
use crate::dartseq_pipeline::*;
use crate::data::methylation::*;

use crate::data::cell_membership::CellMembership;
use genomic_data::gff::FeatureType as GffFeatureType;
use genomic_data::gff::GeneType as GffGeneType;
use genomic_data::gff::GffRecordMap;
use rust_htslib::faidx;

use rayon::ThreadPoolBuilder;

#[derive(Args, Debug)]
pub struct DartSeqCountArgs {
    #[arg(
        short = 'w',
        long = "wt",
        alias = "observed",
        alias = "deamination",
        value_delimiter = ',',
        required = true,
        help = "Observed (wild-type) BAM files.",
        long_help = "Comma-separated list of observed (wild-type) BAM files. \n\
		       These files contain C->U (C->T) conversions, \n\
		       representing the editing events in the wild-type sample. \n\
		       Example: file1.bam,file2.bam"
    )]
    pub wt_bam_files: Vec<Box<str>>,

    #[arg(
        short = 'm',
        long = "mut",
        alias = "background",
        value_delimiter = ',',
        required = true,
        help = "Background/control (mutant) BAM files.",
        long_help = "Comma-separated list of control (mutant) BAM files. \n\
		     These files are used to calibrate background mutation rates \n\
		     to identify disrupted C->U (C->T) conversions in the mutant sample. \n\
		     Example: mut1.bam,mut2.bam"
    )]
    pub mut_bam_files: Vec<Box<str>>,

    #[arg(
        short = 'g',
        long = "gff",
        required = true,
        help = "Gene annotation (`GFF`) file",
        long_help = "Path to the gene annotation file in GFF format. \n\
		     This file provides genomic feature information required for analysis. \n\
		     Example: genes.gff"
    )]
    pub gff_file: Box<str>,

    #[arg(
        short = 'r',
        long,
        help = "resolution (in kb)",
        long_help = "Resolution for binning in kilobases (kb). \n\
		     Determines the size of site-level reports. \n\
		     Example: 1.0"
    )]
    pub resolution_kb: Option<f32>,

    #[arg(
        long = "genome-bins",
        default_value_t = 57,
        help = "#bins for genomic locations in histogram",
        long_help = "Number of bins for genomic locations in the histogram. \n\
		     Controls the granularity of the histogram across the genome."
    )]
    num_genomic_bins_histogram: usize,

    #[arg(
        long = "print_width",
        default_value_t = 40,
        help = "#bins in histogram when printing on the screen",
        long_help = "Approximate number of bins in the output histogram. \n\
		     Adjusts the print width for visualization."
    )]
    histogram_print_width: usize,

    #[arg(
        long,
        default_value = "CB",
        help = "Cell barcode tag",
        long_help = "Cell barcode tag used for cell/sample identification in 10x Genomics BAM files. \n\
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
        default_value_t = 10,
        help = "minimum number of total reads per site",
        long_help = "Minimum number of total reads required per site for inclusion in the analysis. \n\
		     Filters out low-coverage sites."
    )]
    pub min_coverage: usize,

    #[arg(
        long = "min-conversion",
        default_value_t = 5,
        help = "Minimum number of converted reads (T) in wild-type",
        long_help = "Minimum number of converted reads (C->T) required in wild-type sample. \n\
		     Ensures sufficient conversion signal beyond just frequency."
    )]
    pub min_conversion: usize,

    #[arg(
        long = "pseudocount",
        default_value_t = 1,
        help = "Pseudocount for null distribution in binomial test",
        long_help = "Pseudocount added to background/mutant counts for regularization. \n\
		     Helps avoid overly confident p-values with zero background counts."
    )]
    pub pseudocount: usize,

    #[arg(
        short = 'p',
        long = "pval",
        alias = "pvalue",
        alias = "p-val",
        alias = "p-value",
        default_value_t = 0.05,
        help = "Maximum detection p-value cutoff",
        long_help = "Maximum p-value cutoff for detection. \n\
		     Sites with p-values above this threshold will be excluded."
    )]
    pub pvalue_cutoff: f64,

    #[arg(
        long,
        value_enum,
        help = "Bam record type (gene, transcript, exon, utr)",
        long_help = "Selectively choose BAM record type for analysis. \n\
		     Options include gene, transcript, exon, or UTR."
    )]
    record_type: Option<GffFeatureType>,

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
        default_value_t = 16,
        help = "Maximum number of threads",
        long_help = "Maximum number of threads to use for parallel processing. \n\
		     Choose the right number in HPC environments."
    )]
    max_threads: usize,

    #[arg(
        long,
        help = "Number of non-zero cutoff for rows/features",
        long_help = "Minimum number of non-zero entries required for rows/features to be included in the output. If not set, no filtering is applied."
    )]
    row_nnz_cutoff: Option<usize>,

    #[arg(
        long,
        help = "Minimum number of non-zero entries for the columns/cells",
        long_help = "Minimum number of non-zero entries required for columns/cells to be included in the output. If not set, no filtering is applied."
    )]
    column_nnz_cutoff: Option<usize>,

    #[arg(
        short = 't',
        long,
        value_enum,
        default_value = "beta",
        help = "Type of output value to report",
        long_help = "Type of output value to report. Options include beta, count, or other supported types."
    )]
    pub output_value_type: MethFeatureType,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for the output file",
        long_help = "Backend format for the output file. Options include zarr, hdf5, or other supported sparse IO backends."
    )]
    pub backend: SparseIoBackend,

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
        help = "Output mutant signals",
        long_help = "Output mutant signals (null data) in addition to wild-type signals."
    )]
    pub output_null_data: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Output results in BED",
        long_help = "Output results in BED file format for genomic intervals."
    )]
    output_bed_file: bool,

    #[arg(
        long = "gene-level",
        default_value_t = false,
        help = "Output results at a gene level (default: a site level)",
        long_help = "Output results at a gene level (default: a site level).\n\
		     The counts will be aggregated within a gene."
    )]
    pub gene_level_output: bool,

    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Output directory for the output files. \n\
		     This file will contain the results in the selected format."
    )]
    pub output: Box<str>,

    #[arg(
        long,
        default_value_t = false,
        help = "Print histogram to stdout",
        long_help = "Print histogram of gene features to stdout. \n\
		     The histogram will be saved to a file regardless of this option."
    )]
    print_histogram: bool,

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
        long_help = "Path to cell barcode membership file for restricting analysis to specific cells.\n\
                     Format: First column = cell barcode, Second column = cell type.\n\
                     Supports .tsv, .csv, .parquet, and .gz variants.\n\
                     Only cells (barcodes) present in this file will be included in analysis,\n\
                     based on --cell-filter-stage setting."
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
        long = "check-r-site",
        default_value_t = false,
        help = "Validate R site (A/G for RAC, C/T for GTY) in reference",
        long_help = "Validate the R position in RAC/GTY motifs against the\n\
                     reference genome. When enabled, requires R=A/G on the\n\
                     forward strand and Y=C/T on the reverse strand."
    )]
    pub check_r_site: bool,

    // ========== Cell clustering options ==========
    #[arg(
        long = "n-clusters",
        default_value_t = 1,
        help = "Number of cell clusters (1 = no clustering)",
        long_help = "Number of cell clusters for automatic cell type assignment.\n\
                     When set to 1 (default), all cells are treated as one group.\n\
                     When > 1, cells are clustered via random projection + SVD +\n\
                     k-means on gene expression profiles."
    )]
    n_clusters: usize,

    #[arg(
        long = "cluster-proj-dim",
        default_value_t = 50,
        help = "Random projection dimension for clustering",
        long_help = "Dimensionality of the random projection used to compress\n\
                     the gene expression matrix before SVD."
    )]
    cluster_proj_dim: usize,

    #[arg(
        long = "cluster-svd-dim",
        default_value_t = 10,
        help = "Number of SVD components for clustering",
        long_help = "Number of leading singular vectors to retain after SVD.\n\
                     These form the feature space for k-means."
    )]
    cluster_svd_dim: usize,

    #[arg(
        long = "cluster-block-size",
        default_value_t = 100,
        help = "Block size for clustering parallel processing",
        long_help = "Number of columns processed per parallel block during\n\
                     the random projection step."
    )]
    cluster_block_size: usize,

    #[arg(
        long = "cluster-max-iter",
        default_value_t = 100,
        help = "Maximum k-means iterations for clustering",
        long_help = "Maximum number of k-means iterations before convergence\n\
                     is declared."
    )]
    cluster_max_iter: usize,

    #[arg(
        long = "cluster-min-row-nnz",
        default_value_t = 1,
        help = "Minimum non-zeros per gene for clustering QC",
        long_help = "Genes with fewer non-zero cells than this are removed\n\
                     before clustering."
    )]
    cluster_min_row_nnz: usize,

    #[arg(
        long = "cluster-min-col-nnz",
        default_value_t = 1,
        help = "Minimum non-zeros per cell for clustering QC",
        long_help = "Cells with fewer non-zero genes than this are removed\n\
                     before clustering."
    )]
    cluster_min_col_nnz: usize,
}

impl DartSeqCountArgs {
    /// Create backend file path for a given batch name
    pub fn backend_file_path(&self, batch_name: &str) -> Box<str> {
        match self.backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &self.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &self.output, batch_name),
        }
        .into_boxed_str()
    }

    /// Create BED file path for a given batch name
    pub fn bed_file_path(&self, batch_name: &str) -> Box<str> {
        format!("{}/{}.bed.gz", &self.output, batch_name).into_boxed_str()
    }

    /// Get QC cutoffs
    pub fn qc_cutoffs(&self) -> SqueezeCutoffs {
        SqueezeCutoffs {
            row: self.row_nnz_cutoff.unwrap_or(0),
            column: self.column_nnz_cutoff.unwrap_or(0),
        }
    }

    /// Create value extraction function based on output type
    pub fn value_extractor(&self) -> impl Fn(&MethylationData) -> f32 {
        let output_type = self.output_value_type.clone();
        move |dat: &MethylationData| -> f32 {
            match output_type {
                MethFeatureType::Beta => {
                    let tot = (dat.methylated + dat.unmethylated) as f32;
                    (dat.methylated as f32) / tot.max(1.)
                }
                MethFeatureType::Methylated => dat.methylated as f32,
                MethFeatureType::Unmethylated => dat.unmethylated as f32,
            }
        }
    }

    /// Create DartSeqSifter with current arguments
    pub fn create_sifter<'a>(
        &self,
        faidx: &'a faidx::Reader,
        chr: &'a str,
        capacity: usize,
    ) -> DartSeqSifter<'a> {
        DartSeqSifter {
            faidx,
            chr,
            min_coverage: self.min_coverage,
            min_conversion: self.min_conversion,
            pseudocount: self.pseudocount,
            max_pvalue_cutoff: self.pvalue_cutoff,
            check_r_site: self.check_r_site,
            candidate_sites: Vec::with_capacity(capacity),
        }
    }
}

/// Count possibly methylated A positions in DART-seq bam files to
/// quantify m6A β values
pub fn run_count_dartseq(args: &DartSeqCountArgs) -> anyhow::Result<()> {
    mkdir(&args.output)?;

    // Setup thread pool
    let max_threads = num_cpus::get().min(args.max_threads);
    ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global()?;
    info!("will use {} threads", rayon::current_num_threads());

    // Validate inputs
    if args.wt_bam_files.is_empty() || args.mut_bam_files.is_empty() {
        return Err(anyhow::anyhow!("need pairs of bam files"));
    }

    // Check all BAM indices
    check_all_bam_indices(&args.wt_bam_files)?;
    check_all_bam_indices(&args.mut_bam_files)?;

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
    } else if args.n_clusters > 1 {
        // Generate membership via clustering
        let params = ClusteringParams {
            n_clusters: args.n_clusters,
            proj_dim: args.cluster_proj_dim,
            svd_dim: args.cluster_svd_dim,
            block_size: args.cluster_block_size,
            kmeans_max_iter: args.cluster_max_iter,
            cell_barcode_tag: &args.cell_barcode_tag,
            gene_barcode_tag: &args.gene_barcode_tag,
            allow_prefix_matching: !args.exact_barcode_match,
            min_row_nnz: args.cluster_min_row_nnz,
            min_col_nnz: args.cluster_min_col_nnz,
        };
        let m = cluster_cells_from_bam(&args.wt_bam_files, &gff_map, &params)?;
        info!("Generated {} cell clusters via clustering", args.n_clusters);
        Some(m)
    } else {
        None
    };

    /////////////////////////////////
    // FIRST PASS: Find edit sites //
    /////////////////////////////////

    let gene_sites = find_all_methylated_sites(&gff_map, args, membership.as_ref())?;

    if gene_sites.is_empty() {
        info!("no sites found");
        return Ok(());
    }

    let ndata: usize = gene_sites.iter().map(|x| x.value().len()).sum();
    info!("Found {} m6A sites", ndata);

    gene_sites.to_parquet(&gff_map, format!("{}/sites.parquet", args.output))?;

    ////////////////////////////////
    // Output marginal statistics //
    ////////////////////////////////

    let gene_feature_count =
        gene_sites.count_gene_features(&args.gff_file, args.num_genomic_bins_histogram)?;

    if args.print_histogram {
        gene_feature_count.print(args.histogram_print_width);
    }

    gene_feature_count.to_tsv(&format!("{}/gene_feature_count.tsv.gz", args.output))?;

    //////////////////////////////////////////
    // SECOND PASS: Collect cell-level data //
    //////////////////////////////////////////

    if args.output_bed_file {
        process_all_bam_files_to_bed(args, &gene_sites, &gff_map)?;
    } else {
        process_all_bam_files_to_backend(args, &gene_sites, &gff_map)?;
    }

    info!("done");
    Ok(())
}
