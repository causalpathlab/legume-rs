use crate::cell_clustering::{cluster_cells_from_bam, ClusteringParams};
use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::data::methylation::*;
use crate::editing::io::{load_atoi_mask_from_parquet, ToParquet};
use crate::editing::mask::{build_atoi_mask, filter_m6a_by_atoi_mask};
use crate::editing::mixture::MixtureParams;
use crate::editing::pipeline::{
    find_all_conversion_sites, process_all_bam_files_to_backend, process_all_bam_files_to_bed,
    run_mixture_model, ConversionParams,
};
use crate::editing::sifter::ModificationType;
use crate::pipeline_util::*;

use genomic_data::gff::FeatureType as GffFeatureType;
use genomic_data::gff::GeneType as GffGeneType;
use genomic_data::gff::GffRecordMap;

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
    pub pvalue_cutoff: f32,

    #[arg(
        long,
        value_enum,
        help = "GFF feature type filter (gene, transcript, exon, utr)",
        long_help = "Filter GFF records by feature type.\n\
		     Common values: gene, transcript, exon, utr.\n\
		     Note: currently unused, reserved for future use."
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
        long_help = "Type of output value to report.\n\
		     beta: methylation fraction (methylated / total),\n\
		     methylated: raw methylated read count,\n\
		     unmethylated: raw unmethylated read count."
    )]
    pub output_value_type: MethFeatureType,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for the output file",
        long_help = "File format for the output sparse matrix.\n\
		     Supported: zarr, hdf5."
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

    // ========== A-to-I editing detection ==========
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
        help = "P-value cutoff for A-to-I detection"
    )]
    pub atoi_pvalue_cutoff: f32,

    #[arg(
        long = "atoi-mask",
        help = "Pre-computed A-to-I mask parquet file (from `faba atoi` or `faba dart --detect-atoi`)",
        long_help = "Path to a pre-computed A-to-I sites parquet file.\n\
                     When provided, skips A-to-I discovery and loads the mask\n\
                     directly from this file to filter m6A candidates.\n\
                     Implies --detect-atoi behavior for masking."
    )]
    pub atoi_mask_file: Option<Box<str>>,

    // ========== Mixture model options ==========
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
        long = "mixture-initial-sigma",
        default_value_t = 0.0,
        help = "Initial sigma, or 0 for auto (default: 0)"
    )]
    pub mixture_initial_sigma: f32,

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

/// Create m6A ConversionParams from DartSeqCountArgs
impl From<&DartSeqCountArgs> for ConversionParams {
    fn from(args: &DartSeqCountArgs) -> Self {
        ConversionParams {
            genome_file: args.genome_file.clone(),
            wt_bam_files: args.wt_bam_files.clone(),
            mut_bam_files: args.mut_bam_files.clone(),
            gene_barcode_tag: args.gene_barcode_tag.clone(),
            cell_barcode_tag: args.cell_barcode_tag.clone(),
            include_missing_barcode: args.include_missing_barcode,
            min_coverage: args.min_coverage,
            min_conversion: args.min_conversion,
            pseudocount: args.pseudocount,
            pvalue_cutoff: args.pvalue_cutoff,
            backend: args.backend.clone(),
            output: args.output.clone(),
            output_value_type: args.output_value_type.clone(),
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            cell_membership_file: args.cell_membership_file.clone(),
            membership_barcode_col: args.membership_barcode_col,
            membership_celltype_col: args.membership_celltype_col,
            exact_barcode_match: args.exact_barcode_match,
            mod_type: ModificationType::M6A {
                check_r_site: !args.no_check_r_site,
            },
            min_base_quality: 20,
            min_mapping_quality: 20,
        }
    }
}

impl DartSeqCountArgs {
    /// Create A-to-I ConversionParams from DartSeqCountArgs
    fn atoi_params(&self) -> ConversionParams {
        ConversionParams {
            genome_file: self.genome_file.clone(),
            wt_bam_files: self.wt_bam_files.clone(),
            mut_bam_files: self.mut_bam_files.clone(),
            gene_barcode_tag: self.gene_barcode_tag.clone(),
            cell_barcode_tag: self.cell_barcode_tag.clone(),
            include_missing_barcode: self.include_missing_barcode,
            min_coverage: self.atoi_min_coverage,
            min_conversion: self.atoi_min_conversion,
            pseudocount: self.pseudocount,
            pvalue_cutoff: self.atoi_pvalue_cutoff,
            backend: self.backend.clone(),
            output: self.output.clone(),
            output_value_type: self.output_value_type.clone(),
            row_nnz_cutoff: self.row_nnz_cutoff,
            column_nnz_cutoff: self.column_nnz_cutoff,
            cell_membership_file: self.cell_membership_file.clone(),
            membership_barcode_col: self.membership_barcode_col,
            membership_celltype_col: self.membership_celltype_col,
            exact_barcode_match: self.exact_barcode_match,
            mod_type: ModificationType::AtoI,
            min_base_quality: 20,
            min_mapping_quality: 20,
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

    // Detect A-to-I editing sites first (if requested), or load pre-computed mask
    let atoi_params = args.atoi_params();
    let atoi_mask = if let Some(ref mask_file) = args.atoi_mask_file {
        // Load pre-computed A-to-I mask from parquet
        info!("Loading A-to-I mask from {}", mask_file);
        let mask = load_atoi_mask_from_parquet(mask_file.as_ref())?;
        info!("Loaded A-to-I mask with {} positions", mask.len());
        Some((None, mask))
    } else if args.detect_atoi {
        let atoi_sites = find_all_conversion_sites(&gff_map, &atoi_params, None)?;
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

    let m6a_params = ConversionParams::from(args);
    let gene_sites = find_all_conversion_sites(&gff_map, &m6a_params, membership.as_ref())?;

    // Apply A-to-I mask to filter m6A candidates
    if let Some((_, ref mask)) = atoi_mask {
        if !mask.is_empty() {
            let n_before: usize = gene_sites.iter().map(|x| x.value().len()).sum();
            filter_m6a_by_atoi_mask(&gene_sites, mask, &gff_map);
            let n_after: usize = gene_sites.iter().map(|x| x.value().len()).sum();
            info!(
                "A-to-I masking: {} → {} m6A sites ({} removed)",
                n_before,
                n_after,
                n_before - n_after
            );
        }
    }

    if gene_sites.is_empty() {
        info!("no sites found");
        return Ok(());
    }

    let ndata: usize = gene_sites.iter().map(|x| x.value().len()).sum();
    info!("Found {} m6A sites", ndata);

    gene_sites.to_parquet(&gff_map, format!("{}/sites.parquet", args.output))?;

    //////////////////////////////////////////
    // SECOND PASS: Collect cell-level data //
    //////////////////////////////////////////

    if args.output_bed_file {
        process_all_bam_files_to_bed(
            &m6a_params,
            &gene_sites,
            &gff_map,
            args.output_cell_types,
            args.output_null_data,
        )?;
    } else {
        process_all_bam_files_to_backend(
            &m6a_params,
            &gene_sites,
            &gff_map,
            args.gene_level_output,
            args.output_null_data,
        )?;
    }

    // A-to-I second pass: quantify editing sites into sparse matrices
    if let Some((Some(ref atoi_sites), _)) = atoi_mask {
        if !atoi_sites.is_empty() {
            info!("Second pass: A-to-I count matrix");
            process_all_bam_files_to_backend(&atoi_params, atoi_sites, &gff_map, false, false)?;
        }
    }

    // Mixture model: cluster modification sites per gene
    if !args.no_mixture {
        info!("Running 1D Gaussian mixture model on m6A sites...");
        let mix_params = MixtureParams {
            min_sites: args.mixture_min_sites,
            max_k: args.mixture_max_k,
            initial_sigma: args.mixture_initial_sigma,
            ..Default::default()
        };
        run_mixture_model(&m6a_params, &gene_sites, &gff_map, &mix_params)?;
    }

    info!("done");
    Ok(())
}
