use crate::common::*;
use crate::data::conversion::*;
use crate::editing::io::ToParquet;
use crate::editing::mask::filter_conversion_sites_by_mask;
use crate::editing::mixture::MixtureParams;
use crate::editing::pipeline::*;
use crate::editing::sifter::ModificationType;
use crate::pipeline_util::{check_all_bam_indices, run_gene_count_qc};
use crate::snp::io::load_snp_mask_from_parquet;

use genomic_data::gff::GeneType as GffGeneType;
use genomic_data::gff::GffRecordMap;
use rayon::ThreadPoolBuilder;

#[derive(Args, Debug)]
pub struct AtoICountArgs {
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input BAM files (comma-separated)"
    )]
    pub bam_files: Vec<Box<str>>,

    #[arg(
        short = 'g',
        long = "gff",
        required = true,
        help = "Gene annotation (GFF) file"
    )]
    pub gff_file: Box<str>,

    #[arg(
        short = 'f',
        long = "genome",
        help = "Reference genome FASTA file",
        long_help = "Path to reference genome in FASTA format (.fa or .fasta).\n\
                     Used to validate base calls at editing sites.\n\
                     File must be indexed (.fai)."
    )]
    pub genome_file: Box<str>,

    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Output directory for A-to-I detection results.\n\
                     Creates atoi_sites.parquet (detected sites) and one\n\
                     sparse count matrix per input BAM (with _atoi suffix)."
    )]
    pub output: Box<str>,

    #[arg(long, default_value = "CB", help = "Cell barcode tag")]
    pub cell_barcode_tag: Box<str>,

    #[arg(long, default_value = "GX", help = "Gene barcode tag")]
    pub gene_barcode_tag: Box<str>,

    #[arg(
        long,
        default_value_t = 5,
        help = "Minimum coverage for A-to-I site detection"
    )]
    pub min_coverage: usize,

    #[arg(
        long = "min-conversion",
        default_value_t = 3,
        help = "Minimum A-to-G conversions for A-to-I detection"
    )]
    pub min_conversion: usize,

    #[arg(
        long = "min-base-quality",
        default_value_t = 20,
        help = "Minimum base quality (Phred score) to include a base"
    )]
    pub min_base_quality: u8,

    #[arg(
        long = "min-mapping-quality",
        default_value_t = 20,
        help = "Minimum mapping quality (MAPQ) to include a read"
    )]
    pub min_mapping_quality: u8,

    #[arg(
        long = "pseudocount",
        default_value_t = 1,
        help = "Pseudocount for null distribution in binomial test"
    )]
    pub pseudocount: usize,

    #[arg(
        short = 'p',
        long = "pval",
        alias = "pvalue",
        default_value_t = 0.05,
        help = "P-value cutoff for A-to-I detection"
    )]
    pub pvalue_cutoff: f32,

    #[arg(
        short = 't',
        long,
        value_enum,
        default_value = "ratio",
        help = "Type of output value to report"
    )]
    pub output_value_type: ConversionValueType,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for the output file"
    )]
    pub backend: SparseIoBackend,

    #[arg(long, default_value_t = false, help = "Include reads w/o barcode info")]
    pub include_missing_barcode: bool,

    #[arg(long, help = "Number of non-zero cutoff for rows/features")]
    pub row_nnz_cutoff: Option<usize>,

    #[arg(long, help = "Minimum number of non-zero entries for columns/cells")]
    pub column_nnz_cutoff: Option<usize>,

    #[arg(
        long = "cell-membership",
        alias = "barcode-membership",
        alias = "membership",
        help = "Cell barcode membership file for filtering cells (TSV, CSV, or Parquet)"
    )]
    pub cell_membership_file: Option<Box<str>>,

    #[arg(
        long = "membership-barcode-col",
        default_value_t = 0,
        help = "Column index for cell barcodes in membership file"
    )]
    pub membership_barcode_col: usize,

    #[arg(
        long = "membership-celltype-col",
        default_value_t = 1,
        help = "Column index for cell types in membership file"
    )]
    pub membership_celltype_col: usize,

    #[arg(
        long = "exact-barcode-match",
        default_value_t = false,
        help = "Require exact cell barcode matching"
    )]
    pub exact_barcode_match: bool,

    #[arg(
        long = "snp-mask",
        help = "SNP mask parquet from `faba snp` to filter genetic variants",
        long_help = "Path to snp_sites.parquet from `faba snp`. A-to-I candidates\n\
                     at known SNP positions (het or hom-alt) are removed before\n\
                     quantification, eliminating A/G SNPs that mimic editing."
    )]
    pub snp_mask_file: Option<Box<str>>,

    #[arg(long, value_enum, help = "Gene type filter")]
    gene_type: Option<GffGeneType>,

    #[arg(long, default_value_t = 16, help = "Maximum number of threads")]
    max_threads: usize,

    // ========== Mixture model options ==========
    #[arg(
        long = "no-mixture",
        default_value_t = false,
        help = "Disable 1D Gaussian mixture clustering of editing sites"
    )]
    pub no_mixture: bool,

    #[arg(
        long = "mixture-min-sites",
        default_value_t = 3,
        help = "Min distinct positions per gene to attempt mixture"
    )]
    pub mixture_min_sites: usize,

    #[arg(
        long = "mixture-max-k",
        default_value_t = 5,
        help = "Max components to test via BIC"
    )]
    pub mixture_max_k: usize,

    #[arg(
        long = "mixture-initial-sigma",
        default_value_t = 0.0,
        help = "Initial sigma, or 0 for auto"
    )]
    pub mixture_initial_sigma: f32,

    // ========== Gene expression QC ==========
    #[arg(
        long = "gene-min-cells",
        default_value_t = 10,
        help = "Min cells per gene for expression QC"
    )]
    pub gene_min_cells: usize,

    #[arg(
        long = "cell-min-genes",
        default_value_t = 10,
        help = "Min genes per cell for expression QC"
    )]
    pub cell_min_genes: usize,

    #[arg(
        long = "skip-gene-qc",
        default_value_t = false,
        help = "Skip gene expression QC step"
    )]
    pub skip_gene_qc: bool,
}

impl From<&AtoICountArgs> for ConversionParams {
    fn from(args: &AtoICountArgs) -> Self {
        ConversionParams {
            genome_file: args.genome_file.clone(),
            wt_bam_files: args.bam_files.clone(),
            mut_bam_files: Vec::new(),
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
            mod_type: ModificationType::AtoI,
            min_base_quality: args.min_base_quality,
            min_mapping_quality: args.min_mapping_quality,
        }
    }
}

/// Standalone A-to-I editing site detection and quantification
pub fn run_atoi(args: &AtoICountArgs) -> anyhow::Result<()> {
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

    check_all_bam_indices(&args.bam_files)?;

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

    // Gene expression QC: filter to expressed genes and valid cells
    let gene_qc = if !args.skip_gene_qc {
        let qc = run_gene_count_qc(
            &args.gff_file,
            &args.bam_files,
            &args.cell_barcode_tag,
            &args.gene_barcode_tag,
            args.gene_min_cells,
            args.cell_min_genes,
        )?;
        gff_map.retain_by_ids(&qc.gene_ids);
        info!("After gene QC: {} genes retained", gff_map.len());
        if gff_map.is_empty() {
            info!("no genes passed QC");
            return Ok(());
        }
        Some(qc)
    } else {
        None
    };

    let params = ConversionParams::from(args);

    // Load cell membership for filtering
    let membership = params.load_membership()?;

    // FIRST PASS: discover A-to-I sites
    let atoi_sites = find_all_conversion_sites(&gff_map, &params, membership.as_ref())?;
    let n_atoi: usize = atoi_sites.iter().map(|x| x.value().len()).sum();
    info!("Found {} A-to-I editing sites", n_atoi);

    // Apply SNP mask if provided
    if let Some(ref mask_file) = args.snp_mask_file {
        info!("Loading SNP mask from {}", mask_file);
        let snp_mask = load_snp_mask_from_parquet(mask_file.as_ref())?;
        let n_before: usize = atoi_sites.iter().map(|x| x.value().len()).sum();
        filter_conversion_sites_by_mask(&atoi_sites, &snp_mask, &gff_map);
        let n_after: usize = atoi_sites.iter().map(|x| x.value().len()).sum();
        info!(
            "SNP masking: {} → {} A-to-I sites ({} removed)",
            n_before,
            n_after,
            n_before - n_after
        );
    }

    if atoi_sites.is_empty() {
        info!("no A-to-I sites found");
        return Ok(());
    }

    // Write sites parquet
    atoi_sites.to_parquet(&gff_map, format!("{}/atoi_sites.parquet", args.output))?;
    info!("wrote atoi_sites.parquet");

    // SECOND PASS: quantify into sparse matrix
    info!("Second pass: A-to-I count matrix");
    let valid_cells = gene_qc.as_ref().map(|qc| &qc.cell_barcodes);
    process_all_bam_files_to_backend(&params, &atoi_sites, &gff_map, false, valid_cells)?;

    // Mixture model: cluster editing sites per gene
    if !args.no_mixture {
        info!("Running 1D Gaussian mixture model on A-to-I sites...");
        let mix_params = MixtureParams {
            min_sites: args.mixture_min_sites,
            max_k: args.mixture_max_k,
            initial_sigma: args.mixture_initial_sigma,
            ..Default::default()
        };
        run_mixture_model(&params, &atoi_sites, &gff_map, &mix_params)?;
    }

    info!("done");
    Ok(())
}
