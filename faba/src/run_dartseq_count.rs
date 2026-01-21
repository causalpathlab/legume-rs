use crate::cell_clustering::{cluster_cells_from_bam, ClusteringParams};
use crate::common::*;
use crate::dartseq_io::ToParquet;
use crate::dartseq_sifter::*;
use crate::dartseq_stat::Histogram;
use crate::data::dna::Dna;
use rust_htslib::faidx;

use crate::data::cell_membership::CellMembership;
use crate::data::dna_stat_map::*;
use crate::data::dna_stat_traits::*;
use crate::data::gff::FeatureType as GffFeatureType;
use crate::data::gff::GeneType as GffGeneType;
use crate::data::gff::{GeneId, GffRecordMap};
use crate::data::methylation::*;

use crate::data::util_htslib::*;
use clap::ValueEnum;

use dashmap::DashMap as HashMap;
use dashmap::DashSet as HashSet;
use rayon::ThreadPoolBuilder;
use std::sync::{Arc, Mutex};

/// Minimum number of positions required to attempt finding methylated sites
const MIN_LENGTH_FOR_TESTING: usize = 3;

/// Padding around target region when reading BAM files
const BAM_READ_PADDING: i64 = 1;

/// When to apply cell barcode membership filtering
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum FilterStage {
    /// Filter only during quantification (second pass) - discovers sites from all cells
    QuantificationOnly,
    /// Filter during both discovery and quantification passes
    Both,
    /// Filter only during discovery (first pass) - rare use case
    DiscoveryOnly,
}

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
    wt_bam_files: Vec<Box<str>>,

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
    mut_bam_files: Vec<Box<str>>,

    #[arg(
        short = 'g',
        long = "gff",
        required = true,
        help = "Gene annotation (`GFF`) file",
        long_help = "Path to the gene annotation file in GFF format. \n\
		     This file provides genomic feature information required for analysis. \n\
		     Example: genes.gff"
    )]
    gff_file: Box<str>,

    #[arg(
        short = 'r',
        long,
        help = "resolution (in kb)",
        long_help = "Resolution for binning in kilobases (kb). \n\
		     Determines the size of site-level reports. \n\
		     Example: 1.0"
    )]
    resolution_kb: Option<f32>,

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
    cell_barcode_tag: Box<str>,

    #[arg(
        long,
        default_value = "GX",
        help = "Gene barcode tag",
        long_help = "Barcode tag used for gene identification in 10x Genomics BAM files.\n\
		    [See here](`https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/bam`)"
    )]
    gene_barcode_tag: Box<str>,

    #[arg(
        long,
        default_value_t = 10,
        help = "minimum number of total reads per site",
        long_help = "Minimum number of total reads required per site for inclusion in the analysis. \n\
		     Filters out low-coverage sites."
    )]
    min_coverage: usize,

    #[arg(
        long = "min-conversion",
        default_value_t = 5,
        help = "Minimum number of converted reads (T) in wild-type",
        long_help = "Minimum number of converted reads (C->T) required in wild-type sample. \n\
		     Ensures sufficient conversion signal beyond just frequency."
    )]
    min_conversion: usize,

    #[arg(
        long = "pseudocount",
        default_value_t = 1,
        help = "Pseudocount for null distribution in binomial test",
        long_help = "Pseudocount added to background/mutant counts for regularization. \n\
		     Helps avoid overly confident p-values with zero background counts."
    )]
    pseudocount: usize,

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
    pvalue_cutoff: f64,

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
    output_value_type: MethFeatureType,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for the output file",
        long_help = "Backend format for the output file. Options include zarr, hdf5, or other supported sparse IO backends."
    )]
    backend: SparseIoBackend,

    #[arg(
        long,
        default_value_t = false,
        help = "Include reads w/o barcode info",
        long_help = "Include reads that are missing gene and cell barcode information in the analysis."
    )]
    include_missing_barcode: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Output mutant signals",
        long_help = "Output mutant signals (null data) in addition to wild-type signals."
    )]
    output_null_data: bool,

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
    gene_level_output: bool,

    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Output directory for the output files. \n\
		     This file will contain the results in the selected format."
    )]
    output: Box<str>,

    #[arg(
        long,
        short,
        help = "verbosity",
        long_help = "Enable verbose output `RUST_LOG=info`"
    )]
    verbose: bool,

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
    genome_file: Box<str>,

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
    cell_membership_file: Option<Box<str>>,

    #[arg(
        long = "membership-barcode-col",
        default_value_t = 0,
        help = "Column index for cell barcodes in membership file (0-based)"
    )]
    membership_barcode_col: usize,

    #[arg(
        long = "membership-celltype-col",
        default_value_t = 1,
        help = "Column index for cell types in membership file (0-based)"
    )]
    membership_celltype_col: usize,

    #[arg(
        long = "cell-filter-stage",
        value_enum,
        default_value = "both",
        help = "When to apply cell barcode membership filtering",
        long_help = "Control when cell barcode membership filtering is applied:\n\
                     - quantification-only: Filter cells during second pass only (default, recommended)\n\
                       Discovers methylation sites from all cells, but only quantify cells in membership file\n\
                     - both: Filter cells during both discovery and quantification passes\n\
                       Only discover and quantify sites in cells from membership file\n\
                     - discovery-only: Filter cells during first pass only (rare use case)\n\
                       Only discover sites in membership cells, but quantify all cells"
    )]
    cell_filter_stage: FilterStage,

    #[arg(
        long = "exact-barcode-match",
        default_value_t = false,
        help = "Require exact cell barcode matching (disable prefix matching)",
        long_help = "By default, membership cell barcodes are matched as prefixes of BAM barcodes.\n\
                     This handles cases where BAM barcodes have suffixes like '-1'.\n\
                     Enable this flag to require exact string matching for cell barcodes."
    )]
    exact_barcode_match: bool,

    #[arg(
        long = "output-cell-types",
        default_value_t = false,
        help = "Include cell type annotation in BED output"
    )]
    output_cell_types: bool,

    #[arg(
        long = "check-r-site",
        default_value_t = false,
        help = "Validate R site (A/G for RAC, C/T for GTY) in reference",
        long_help = "Whether to validate the R site in RAC/GTY patterns.\n\
                     When enabled, requires R=A/G for forward strand and Y=C/T for reverse strand.\n\
                     Enable to test if R site validation is meaningful for your data."
    )]
    check_r_site: bool,

    // ========== Cell clustering options ==========
    #[arg(
        long = "n-clusters",
        default_value_t = 1,
        help = "Number of cell clusters (1 = no clustering)",
        long_help = "Number of cell clusters for automatic cell type assignment.\n\
                     When set to 1 (default), all cells are treated as a single group.\n\
                     When > 1, cells are clustered using random projection + SVD + k-means."
    )]
    n_clusters: usize,

    #[arg(
        long = "cluster-proj-dim",
        default_value_t = 50,
        help = "Random projection dimension for clustering"
    )]
    cluster_proj_dim: usize,

    #[arg(
        long = "cluster-svd-dim",
        default_value_t = 10,
        help = "Number of SVD components for clustering"
    )]
    cluster_svd_dim: usize,

    #[arg(
        long = "cluster-block-size",
        default_value_t = 100,
        help = "Block size for parallel processing during clustering"
    )]
    cluster_block_size: usize,

    #[arg(
        long = "cluster-max-iter",
        default_value_t = 100,
        help = "Maximum iterations for k-means clustering"
    )]
    cluster_max_iter: usize,
}

impl DartSeqCountArgs {
    /// Create backend file path for a given batch name
    fn backend_file_path(&self, batch_name: &str) -> Box<str> {
        match self.backend {
            SparseIoBackend::HDF5 => format!("{}/{}.h5", &self.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}.zarr", &self.output, batch_name),
        }
        .into_boxed_str()
    }

    /// Create BED file path for a given batch name
    fn bed_file_path(&self, batch_name: &str) -> Box<str> {
        format!("{}/{}.bed.gz", &self.output, batch_name).into_boxed_str()
    }

    /// Get QC cutoffs
    fn qc_cutoffs(&self) -> SqueezeCutoffs {
        SqueezeCutoffs {
            row: self.row_nnz_cutoff.unwrap_or(0),
            column: self.column_nnz_cutoff.unwrap_or(0),
        }
    }

    /// Create value extraction function based on output type
    fn value_extractor(&self) -> impl Fn(&MethylationData) -> f32 {
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
    fn create_sifter<'a>(
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

/// Bin a position based on resolution (in kb)
/// Returns (start, stop) where start is inclusive and stop is exclusive: [start, stop)
#[inline]
fn bin_position_kb(position: i64, resolution_kb: Option<f32>) -> (i64, i64) {
    if let Some(r) = resolution_kb {
        let r = (r * 1000.0) as usize;
        let start = ((position as usize) / r * r) as i64;
        let stop = start + r as i64;
        (start, stop)
    } else {
        (position, position + 1)
    }
}

/// Generate unique batch names from BAM files
fn uniq_batch_names(bam_files: &[Box<str>]) -> anyhow::Result<Vec<Box<str>>> {
    let batch_names: Vec<Box<str>> = bam_files
        .iter()
        .map(|x| basename(x))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let unique_bams: HashSet<_> = bam_files.iter().cloned().collect();
    let unique_names: HashSet<_> = batch_names.iter().cloned().collect();

    if unique_names.len() == bam_files.len() && unique_bams.len() == bam_files.len() {
        Ok(batch_names)
    } else {
        info!("bam file (base) names are not unique");
        Ok(batch_names
            .iter()
            .enumerate()
            .map(|(i, name)| format!("{}_{}", name, i).into_boxed_str())
            .collect())
    }
}

/// Check BAM indices for all files
fn check_all_bam_indices(bam_files: &[Box<str>]) -> anyhow::Result<()> {
    for bam_file in bam_files {
        info!("checking .bai file for {}...", bam_file);
        check_bam_index(bam_file, None)?;
    }
    Ok(())
}

/// Count possibly methylated A positions in DART-seq bam files to
/// quantify m6A β values
pub fn run_count_dartseq(args: &DartSeqCountArgs) -> anyhow::Result<()> {
    // Setup logging and environment
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

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
        info!("Cell filter stage: {:?}", args.cell_filter_stage);
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
        };
        let m = cluster_cells_from_bam(&args.wt_bam_files, &gff_map, &params)?;
        info!(
            "Generated {} cell clusters via clustering",
            args.n_clusters
        );
        info!("Cell filter stage: {:?}", args.cell_filter_stage);
        Some(m)
    } else {
        None
    };

    // Determine when to apply cell membership filter for discovery based on filter stage
    let membership_for_discovery = match args.cell_filter_stage {
        FilterStage::DiscoveryOnly | FilterStage::Both => membership.as_ref(),
        FilterStage::QuantificationOnly => None,
    };

    /////////////////////////////////
    // FIRST PASS: Find edit sites //
    /////////////////////////////////

    let gene_sites = find_all_methylated_sites(&gff_map, args, membership_for_discovery)?;

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

///////////////////////////////////////
// FIRST PASS: Find methylated sites //
///////////////////////////////////////

fn find_all_methylated_sites(
    gff_map: &GffRecordMap,
    args: &DartSeqCountArgs,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<HashMap<GeneId, Vec<MethylatedSite>>> {
    let njobs = gff_map.len();
    info!("Searching possible edit sites over {} blocks", njobs);

    // Validate reference genome
    info!("Loading reference genome: {}", args.genome_file);
    load_fasta_index(&args.genome_file)?;

    let arc_gene_sites = Arc::new(HashMap::<GeneId, Vec<MethylatedSite>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_count(njobs as u64)
        .try_for_each(|rec| -> anyhow::Result<()> {
            find_methylated_sites_in_gene(rec, args, arc_gene_sites.clone(), cell_membership)
        })?;

    Arc::try_unwrap(arc_gene_sites).map_err(|_| anyhow::anyhow!("failed to release gene_sites"))
}

fn find_methylated_sites_in_gene(
    gff_record: &GffRecord,
    args: &DartSeqCountArgs,
    arc_gene_sites: Arc<HashMap<GeneId, Vec<MethylatedSite>>>,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<()> {
    let gene_id = gff_record.gene_id.clone();
    let strand = &gff_record.strand;
    let chr = gff_record.seqname.as_ref();

    // Each thread creates its own reader (faidx is not thread-safe)
    let faidx_reader = load_fasta_index(&args.genome_file)?;

    // Sweep all BAM files to find variable sites
    let mut wt_base_freq_map = DnaBaseFreqMap::new(cell_membership);

    for wt_file in &args.wt_bam_files {
        wt_base_freq_map.update_bam_file_by_gene(wt_file, gff_record, &args.gene_barcode_tag)?;
    }

    let positions = wt_base_freq_map.sorted_positions();

    if positions.len() < MIN_LENGTH_FOR_TESTING {
        return Ok(());
    }

    // Find AC/T patterns using mutant statistics as null distribution
    let mut sifter = args.create_sifter(&faidx_reader, chr, positions.len());

    // Gather background frequency map
    let mut mut_base_freq_map = DnaBaseFreqMap::new(cell_membership);

    for mut_file in &args.mut_bam_files {
        mut_base_freq_map.update_bam_file_by_gene(mut_file, gff_record, &args.gene_barcode_tag)?;
    }

    let wt_freq = wt_base_freq_map
        .marginal_frequency_map()
        .ok_or_else(|| anyhow::anyhow!("failed to count wt freq"))?;
    let mut_freq = mut_base_freq_map
        .marginal_frequency_map()
        .ok_or_else(|| anyhow::anyhow!("failed to count mut freq"))?;

    match strand {
        Strand::Forward => {
            sifter.forward_sweep(&positions, &wt_freq, Some(&mut_freq));
        }
        Strand::Backward => {
            sifter.backward_sweep(&positions, &wt_freq, Some(&mut_freq));
        }
    }

    let mut candidate_sites = sifter.candidate_sites;

    if !candidate_sites.is_empty() {
        candidate_sites.sort();
        candidate_sites.dedup();
        arc_gene_sites.insert(gene_id, candidate_sites);
    }

    Ok(())
}

//////////////////////////////////////////
// SECOND PASS: Collect cell-level data //
//////////////////////////////////////////

fn process_all_bam_files_to_bed(
    args: &DartSeqCountArgs,
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    gff_map: &GffRecordMap,
) -> anyhow::Result<()> {
    // Load cell membership file if provided
    let membership = if let Some(ref path) = args.cell_membership_file {
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
        info!("Cell filter stage: {:?}", args.cell_filter_stage);
        info!("Prefix matching: {}", !args.exact_barcode_match);
        Some(m)
    } else {
        None
    };

    // Determine when to apply cell membership filter based on filter stage
    let membership_for_quantification = match args.cell_filter_stage {
        FilterStage::QuantificationOnly | FilterStage::Both => membership.as_ref(),
        FilterStage::DiscoveryOnly => None,
    };

    let wt_batch_names = uniq_batch_names(&args.wt_bam_files)?;

    for (bam_file, batch_name) in args.wt_bam_files.iter().zip(wt_batch_names) {
        let mut stats = gather_m6a_stats(
            gene_sites,
            args,
            gff_map,
            bam_file,
            membership_for_quantification,
        )?;
        write_bed(
            &mut stats,
            gff_map,
            &args.bed_file_path(&batch_name),
            membership_for_quantification,
            args,
        )?;
    }

    if args.output_null_data {
        info!("output null data");
        let mut_batch_names = uniq_batch_names(&args.mut_bam_files)?;

        for (bam_file, batch_name) in args.mut_bam_files.iter().zip(mut_batch_names) {
            let mut stats = gather_m6a_stats(
                gene_sites,
                args,
                gff_map,
                bam_file,
                membership_for_quantification,
            )?;
            write_bed(
                &mut stats,
                gff_map,
                &args.bed_file_path(&batch_name),
                membership_for_quantification,
                args,
            )?;
        }
    }

    // Log match statistics if membership was used
    if let Some(ref m) = membership {
        let (matched, total) = m.match_stats();
        info!(
            "Cell barcode matching: {}/{} BAM barcodes matched membership ({:.1}%)",
            matched,
            total,
            if total > 0 {
                100.0 * matched as f64 / total as f64
            } else {
                0.0
            }
        );
    }

    Ok(())
}

fn process_all_bam_files_to_backend(
    args: &DartSeqCountArgs,
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    gff_map: &GffRecordMap,
) -> anyhow::Result<()> {
    // Load cell membership file if provided
    let membership = if let Some(ref path) = args.cell_membership_file {
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
        info!("Cell filter stage: {:?}", args.cell_filter_stage);
        info!("Prefix matching: {}", !args.exact_barcode_match);
        Some(m)
    } else {
        None
    };

    // Determine when to apply cell membership filter based on filter stage
    let membership_for_quantification = match args.cell_filter_stage {
        FilterStage::QuantificationOnly | FilterStage::Both => membership.as_ref(),
        FilterStage::DiscoveryOnly => None,
    };

    let gene_key = create_gene_key_function(gff_map);
    let site_key = |x: &BedWithGene| -> Box<str> { format!("{}@m6A", x).into_boxed_str() };
    let take_value = args.value_extractor();
    let cutoffs = args.qc_cutoffs();

    let mut genes = HashSet::<Box<str>>::default();
    let mut sites = HashSet::<Box<str>>::default();
    let mut gene_data_files: Vec<Box<str>> = vec![];
    let mut site_data_files: Vec<Box<str>> = vec![];
    let mut null_gene_data_files: Vec<Box<str>> = vec![];
    let mut null_site_data_files: Vec<Box<str>> = vec![];

    let wt_batch_names = uniq_batch_names(&args.wt_bam_files)?;

    for (bam_file, batch_name) in args.wt_bam_files.iter().zip(wt_batch_names) {
        process_bam_to_backend(
            bam_file,
            &batch_name,
            gene_sites,
            args,
            gff_map,
            &gene_key,
            &site_key,
            &take_value,
            &cutoffs,
            &mut genes,
            &mut sites,
            &mut gene_data_files,
            &mut site_data_files,
            membership_for_quantification,
        )?;
    }

    if args.output_null_data {
        info!("output null data");
        let mut_batch_names = uniq_batch_names(&args.mut_bam_files)?;

        for (bam_file, batch_name) in args.mut_bam_files.iter().zip(mut_batch_names) {
            process_bam_to_backend(
                bam_file,
                &batch_name,
                gene_sites,
                args,
                gff_map,
                &gene_key,
                &site_key,
                &take_value,
                &cutoffs,
                &mut genes,
                &mut sites,
                &mut null_gene_data_files,
                &mut null_site_data_files,
                membership_for_quantification,
            )?;
        }
    }

    // Log match statistics if membership was used
    if let Some(ref m) = membership {
        let (matched, total) = m.match_stats();
        info!(
            "Cell barcode matching: {}/{} BAM barcodes matched membership ({:.1}%)",
            matched,
            total,
            if total > 0 {
                100.0 * matched as f64 / total as f64
            } else {
                0.0
            }
        );
    }

    // Reorder rows to ensure consistency across files
    reorder_all_matrices(
        args,
        genes,
        sites,
        gene_data_files,
        site_data_files,
        null_gene_data_files,
        null_site_data_files,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_bam_to_backend(
    bam_file: &str,
    batch_name: &str,
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    args: &DartSeqCountArgs,
    gff_map: &GffRecordMap,
    gene_key: &(impl Fn(&BedWithGene) -> Box<str> + Send + Sync),
    site_key: &(impl Fn(&BedWithGene) -> Box<str> + Send + Sync),
    take_value: &(impl Fn(&MethylationData) -> f32 + Send + Sync),
    cutoffs: &SqueezeCutoffs,
    genes: &mut HashSet<Box<str>>,
    sites: &mut HashSet<Box<str>>,
    gene_data_files: &mut Vec<Box<str>>,
    site_data_files: &mut Vec<Box<str>>,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<()> {
    info!(
        "collecting data over {} sites from {} ...",
        gene_sites.iter().map(|x| x.value().len()).sum::<usize>(),
        bam_file
    );

    let stats = gather_m6a_stats(gene_sites, args, gff_map, bam_file, cell_membership)?;

    info!(
        "aggregating the '{}' triplets over {} stats...",
        args.output_value_type,
        stats.len()
    );

    if args.gene_level_output {
        let gene_data_file = args.backend_file_path(batch_name);
        let triplets = summarize_stats(&stats, gene_key, take_value);
        let data = triplets.to_backend(&gene_data_file)?;
        data.qc(cutoffs.clone())?;
        genes.extend(data.row_names()?);
        info!("created gene-level data: {}", &gene_data_file);
        gene_data_files.push(gene_data_file);
    } else {
        let site_data_file = args.backend_file_path(batch_name);
        let triplets = summarize_stats(&stats, site_key, take_value);
        let data = triplets.to_backend(&site_data_file)?;
        data.qc(cutoffs.clone())?;
        sites.extend(data.row_names()?);
        info!("created site-level data: {}", &site_data_file);
        site_data_files.push(site_data_file);
    }

    Ok(())
}

fn reorder_all_matrices(
    args: &DartSeqCountArgs,
    genes: HashSet<Box<str>>,
    sites: HashSet<Box<str>>,
    gene_data_files: Vec<Box<str>>,
    site_data_files: Vec<Box<str>>,
    null_gene_data_files: Vec<Box<str>>,
    null_site_data_files: Vec<Box<str>>,
) -> anyhow::Result<()> {
    let mut genes_sorted: Vec<_> = genes.into_iter().collect();
    genes_sorted.sort();

    let backend = &args.backend;

    for data_file in gene_data_files {
        open_sparse_matrix(&data_file, backend)?.reorder_rows(&genes_sorted)?;
    }

    if args.output_null_data {
        for data_file in null_gene_data_files {
            open_sparse_matrix(&data_file, backend)?.reorder_rows(&genes_sorted)?;
        }
    }

    let mut sites_sorted: Vec<_> = sites.into_iter().collect();
    sites_sorted.sort();

    for data_file in site_data_files {
        open_sparse_matrix(&data_file, backend)?.reorder_rows(&sites_sorted)?;
    }

    if args.output_null_data {
        for data_file in null_site_data_files {
            open_sparse_matrix(&data_file, backend)?.reorder_rows(&sites_sorted)?;
        }
    }

    Ok(())
}

fn gather_m6a_stats(
    gene_sites: &HashMap<GeneId, Vec<MethylatedSite>>,
    args: &DartSeqCountArgs,
    gff_map: &GffRecordMap,
    bam_file: &str,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let ndata = gene_sites.iter().map(|x| x.value().len()).sum::<usize>();
    let arc_ret = Arc::new(Mutex::new(Vec::with_capacity(ndata)));

    gene_sites
        .into_iter()
        .par_bridge()
        .progress_count(gene_sites.len() as u64)
        .try_for_each(|gs| -> anyhow::Result<()> {
            let gene = gs.key();
            let sites = gs.value();

            if let Some(gff) = gff_map.get(gene) {
                let stats = collect_gene_m6a_stats(args, bam_file, &gff, sites, cell_membership)?;
                arc_ret.lock().expect("lock").extend(stats);
            }
            Ok(())
        })?;

    Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()
        .map_err(Into::into)
}

fn collect_gene_m6a_stats(
    args: &DartSeqCountArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    sites: &[MethylatedSite],
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut all_stats = Vec::new();

    for site in sites {
        let stats = estimate_m6a_stat(args, bam_file, gff_record, site, cell_membership)?;
        all_stats.extend(stats);
    }

    Ok(all_stats)
}

fn estimate_m6a_stat(
    args: &DartSeqCountArgs,
    bam_file: &str,
    gff_record: &GffRecord,
    m6a_c2u: &MethylatedSite,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, MethylationData)>> {
    let mut stat_map =
        DnaBaseFreqMap::new_with_cell_barcode(&args.cell_barcode_tag, cell_membership);
    let m6apos = m6a_c2u.m6a_pos;
    let c2upos = m6a_c2u.conversion_pos;

    let lb = m6apos.min(c2upos);
    let ub = c2upos.max(m6apos);

    // Read BAM file for region around the m6A site
    let mut gff = gff_record.clone();
    gff.start = (lb - BAM_READ_PADDING).max(0);
    gff.stop = ub + BAM_READ_PADDING;
    stat_map.update_bam_file_by_gene(bam_file, &gff, &args.gene_barcode_tag)?;

    let gene = gff.gene_id;
    let chr = gff.seqname.as_ref();
    let strand = &gff.strand;

    let (unmutated_base, mutated_base) = match strand {
        Strand::Forward => (Dna::C, Dna::T),
        Strand::Backward => (Dna::G, Dna::A),
    };

    // Set the anchor position for m6A
    let anchor_base = match strand {
        Strand::Forward => Dna::A,
        Strand::Backward => Dna::T,
    };
    stat_map.set_anchor_position(m6apos, anchor_base);

    let methylation_stat = stat_map.stratified_frequency_at(c2upos);

    let Some(meth_stat) = methylation_stat else {
        return Ok(Vec::new());
    };

    let (start, stop) = bin_position_kb(m6apos, args.resolution_kb);

    let stats = meth_stat
        .iter()
        .filter_map(|(cb, counts)| {
            let methylated = counts.get(Some(&mutated_base));
            let unmethylated = counts.get(Some(&unmutated_base));

            if (args.include_missing_barcode || cb != &CellBarcode::Missing) && methylated > 0 {
                Some((
                    cb.clone(),
                    BedWithGene {
                        chr: chr.into(),
                        start,
                        stop,
                        gene: gene.clone(),
                        strand: strand.clone(),
                    },
                    MethylationData {
                        methylated,
                        unmethylated,
                    },
                ))
            } else {
                None
            }
        })
        .collect();

    Ok(stats)
}

//////////////////////////////////////////////////////////
// Step 3: Repackaging into desired output formats     //
//////////////////////////////////////////////////////////

fn create_gene_key_function(
    gff_map: &GffRecordMap,
) -> impl Fn(&BedWithGene) -> Box<str> + Send + Sync + '_ {
    |x: &BedWithGene| -> Box<str> {
        gff_map
            .get(&x.gene)
            .map(|gff| format!("{}_{}", gff.gene_id, gff.gene_name))
            .unwrap_or_else(|| format!("{}", x.gene))
            .into_boxed_str()
    }
}

fn summarize_stats<F, V, T>(
    stats: &[(CellBarcode, BedWithGene, MethylationData)],
    feature_key_func: F,
    value_func: V,
) -> TripletsRowsCols
where
    F: Fn(&BedWithGene) -> T + Send + Sync,
    T: Clone + Send + Sync + ToString + std::hash::Hash + std::cmp::Eq + std::cmp::Ord,
    V: Fn(&MethylationData) -> f32 + Send + Sync,
{
    let combined_data: HashMap<(CellBarcode, T), MethylationData> = HashMap::default();

    stats.par_iter().for_each(|(cb, k, dat)| {
        let key = (cb.clone(), feature_key_func(k));
        combined_data.entry(key).or_default().add_assign(dat);
    });

    let combined_data = combined_data
        .into_iter()
        .map(|((c, k), v)| (c, k, value_func(&v)))
        .collect::<Vec<_>>();

    format_data_triplets(combined_data)
}

fn write_bed(
    stats: &mut [(CellBarcode, BedWithGene, MethylationData)],
    gff_map: &GffRecordMap,
    file_path: &str,
    cell_membership: Option<&CellMembership>,
    args: &DartSeqCountArgs,
) -> anyhow::Result<()> {
    use rust_htslib::bgzf::Writer as BWriter;
    use std::io::Write;

    stats.par_sort_by(|a, b| a.1.cmp(&b.1));

    let lines: Vec<_> = stats
        .iter()
        .map(|(cb, bg, data)| {
            let gene_string = gff_map
                .get(&bg.gene)
                .map(|gff| match gff.gene_name {
                    GeneSymbol::Symbol(x) => format!("{}_{}", &bg.gene, x),
                    GeneSymbol::Missing => format!("{}", &bg.gene),
                })
                .unwrap_or_else(|| format!("{}", &bg.gene));

            if args.output_cell_types {
                if let Some(membership) = cell_membership {
                    let cell_type = membership
                        .matches_barcode(cb)
                        .unwrap_or_else(|| "unknown".into());
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        bg.chr,
                        bg.start,
                        bg.stop,
                        bg.strand,
                        gene_string,
                        data.methylated,
                        data.unmethylated,
                        cb,
                        cell_type
                    )
                    .into_boxed_str()
                } else {
                    // No membership provided but cell types requested - just output unknown
                    format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tunknown",
                        bg.chr,
                        bg.start,
                        bg.stop,
                        bg.strand,
                        gene_string,
                        data.methylated,
                        data.unmethylated,
                        cb
                    )
                    .into_boxed_str()
                }
            } else {
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    bg.chr,
                    bg.start,
                    bg.stop,
                    bg.strand,
                    gene_string,
                    data.methylated,
                    data.unmethylated,
                    cb
                )
                .into_boxed_str()
            }
        })
        .collect();

    let header: &[u8] = if args.output_cell_types {
        b"#chr\tstart\tstop\tstrand\tgene\tmethylated\tunmethylated\tbarcode\tcell_type\n"
    } else {
        b"#chr\tstart\tstop\tstrand\tgene\tmethylated\tunmethylated\tbarcode\n"
    };

    let mut writer = BWriter::from_path(file_path)?;
    writer.write_all(header)?;
    for line in lines {
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;

    Ok(())
}
