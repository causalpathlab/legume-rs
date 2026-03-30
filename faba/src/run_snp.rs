use crate::common::*;
use crate::pipeline_util::check_all_bam_indices;
use crate::snp::genotyper::GenotypeParams;
use crate::snp::io::load_known_snps;
use crate::snp::pipeline::{run_snp_pipeline, SnpParams};

use genomic_data::gff::GeneType as GffGeneType;
use genomic_data::gff::GffRecordMap;
use rayon::ThreadPoolBuilder;

#[derive(Args, Debug)]
pub struct SnpArgs {
    /// Input BAM files (comma-separated).
    /// For 10x data, these should be the possorted_genome_bam.bam files.
    /// For bulk WGS/RNA-seq, any coordinate-sorted, indexed BAM.
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input BAM files (comma-separated)",
        long_help = "One or more coordinate-sorted BAM files with .bai index.\n\
                     Multiple files are merged for pileup (e.g., biological replicates).\n\
                     For 10x single-cell data, use possorted_genome_bam.bam."
    )]
    pub bam_files: Vec<Box<str>>,

    /// Reference genome FASTA file (indexed, .fai).
    /// Required for de novo variant discovery. Used to determine the
    /// reference allele at each genomic position.
    #[arg(
        short = 'f',
        long = "genome",
        required = true,
        help = "Reference genome FASTA (.fa, indexed with .fai)",
        long_help = "Path to the reference genome in FASTA format.\n\
                     Must be indexed (.fai); if missing, the index is created automatically.\n\
                     Used for de novo discovery (compare reads to reference) and to\n\
                     validate ref alleles at known sites."
    )]
    pub genome_file: Box<str>,

    /// Known SNP sites in VCF/BCF format. Optional.
    /// When provided, genotypes are force-called at these positions.
    /// When omitted, only de novo discovery is performed.
    #[arg(
        long = "known-snps",
        help = "Known SNP sites VCF/BCF (optional)",
        long_help = "Path to a VCF/BCF file containing known SNP sites (e.g., dbSNP).\n\
                     Only biallelic SNPs are used; indels and multi-allelic sites are skipped.\n\
                     When provided, genotypes are force-called at these positions regardless\n\
                     of alt allele evidence. Can be combined with de novo discovery.\n\
                     When omitted, only de novo discovery from the reference genome is used."
    )]
    pub known_snps_vcf: Option<Box<str>>,

    /// Gene annotation in GFF format. Optional.
    /// Enables gene-centric processing and per-cell output.
    /// When omitted, operates in region-centric mode (bulk WGS).
    #[arg(
        short = 'g',
        long = "gff",
        help = "Gene annotation GFF file (optional, enables gene-centric mode)",
        long_help = "Gene annotation in GFF/GTF format.\n\
                     When provided: processes SNPs within gene boundaries, enables\n\
                     per-cell allele count output with gene_key/SNP/chr:pos naming.\n\
                     When omitted: processes SNPs by chromosome region (bulk WGS mode),\n\
                     no per-cell sparse matrix output."
    )]
    pub gff_file: Option<Box<str>>,

    /// Output directory for results.
    /// Creates: snp_sites.parquet (genotype calls), and optionally
    /// *_snp_alt.zarr + *_snp_depth.zarr (per-cell matrices for BAF).
    #[arg(
        short,
        long,
        required = true,
        help = "Output directory",
        long_help = "Output directory for SNP genotyping results. Created if needed.\n\
                     Outputs:\n\
                     - snp_sites.parquet: all genotyped sites with allele counts and GQ\n\
                     - {batch}_snp_alt.zarr: per-cell alt allele count matrix (10x mode)\n\
                     - {batch}_snp_depth.zarr: per-cell total depth matrix (10x mode)\n\
                     BAF per cell = alt / depth."
    )]
    pub output: Box<str>,

    /// Cell barcode BAM tag. Standard 10x tag is CB.
    #[arg(
        long,
        default_value = "CB",
        help = "Cell barcode BAM tag",
        long_help = "BAM auxiliary tag for cell barcodes.\n\
                     Standard 10x Genomics tag is \"CB\".\n\
                     Only used in single-cell mode (without --bulk)."
    )]
    pub cell_barcode_tag: Box<str>,

    /// Gene barcode BAM tag. Standard 10x tag is GX.
    #[arg(
        long,
        default_value = "GX",
        help = "Gene barcode BAM tag",
        long_help = "BAM auxiliary tag for gene identification.\n\
                     Standard 10x Genomics tag is \"GX\".\n\
                     Only used with --gff for gene-centric processing."
    )]
    pub gene_barcode_tag: Box<str>,

    /// Bulk mode: only produce genotype calls, no per-cell matrices.
    /// Use for bulk WGS/RNA-seq or when only the SNP mask is needed.
    #[arg(
        long,
        default_value_t = false,
        help = "Bulk mode (genotype calls only, no per-cell output)",
        long_help = "When set, only snp_sites.parquet is produced.\n\
                     No per-cell allele count or depth matrices are written.\n\
                     Use for bulk WGS/RNA-seq, or when you only need the SNP mask\n\
                     for --snp-mask in faba atoi/dartseq/apa."
    )]
    pub bulk: bool,

    /// Skip de novo variant discovery. Only genotype at --known-snps positions.
    /// Requires --known-snps when set.
    #[arg(
        long,
        default_value_t = false,
        help = "Skip de novo discovery (known sites only)",
        long_help = "When set, no reference-based variant discovery is performed.\n\
                     Only positions from --known-snps are genotyped.\n\
                     Requires --known-snps to be provided."
    )]
    pub skip_discovery: bool,

    // ========== De novo discovery parameters ==========
    /// Minimum total read depth to consider a position for de novo discovery.
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for de novo discovery",
        long_help = "Minimum total read depth at a position to consider it for\n\
                     de novo variant discovery. Positions below this threshold are\n\
                     skipped. Does not affect known-site genotyping (see --min-depth)."
    )]
    pub min_coverage: usize,

    /// Minimum alt allele read count for de novo discovery.
    #[arg(
        long,
        default_value_t = 3,
        help = "Minimum alt allele reads for discovery",
        long_help = "Minimum number of reads supporting the non-reference allele\n\
                     to consider a position as a candidate variant. Applied before\n\
                     genotype calling."
    )]
    pub min_alt_count: usize,

    /// Minimum alt allele frequency for de novo discovery.
    #[arg(
        long,
        default_value_t = 0.1,
        help = "Minimum alt allele frequency for discovery",
        long_help = "Minimum fraction of reads supporting the non-reference allele\n\
                     (alt_count / total_depth) to consider a position as a candidate.\n\
                     Default 0.1 (10%) balances sensitivity with false positive rate."
    )]
    pub min_alt_freq: f64,

    // ========== Genotyping parameters ==========
    /// Minimum read depth to call a genotype at known sites.
    #[arg(
        long,
        default_value_t = 5,
        help = "Minimum depth for genotype calling",
        long_help = "Minimum total read depth to attempt genotype calling at a\n\
                     known variant site. Sites below this threshold get NoCall (./.).\n\
                     For de novo discovery, --min-coverage is used instead."
    )]
    pub min_depth: usize,

    /// Minimum genotype quality (Phred-scaled) to report a call.
    #[arg(
        long,
        default_value_t = 20.0,
        help = "Minimum genotype quality (Phred)",
        long_help = "Minimum Phred-scaled genotype quality to emit a genotype call.\n\
                     GQ = -10*log10(P(wrong genotype)). GQ >= 20 means >= 99% confidence.\n\
                     Sites below this threshold get NoCall (./.) in the output."
    )]
    pub min_gq: f32,

    /// Base error rate for the genotype likelihood model.
    #[arg(
        long,
        default_value_t = 0.01,
        help = "Base error rate for GL model",
        long_help = "Probability that an observed base is a sequencing error.\n\
                     Used in the binomial genotype likelihood model:\n\
                     P(data|HomRef) ~ Binom(n_alt; depth, epsilon).\n\
                     Default 0.01 corresponds to Q20 base quality."
    )]
    pub base_error_rate: f64,

    // ========== Quality filters ==========
    /// Minimum Phred base quality score to include a base in pileup.
    #[arg(
        long = "min-base-quality",
        default_value_t = 20,
        help = "Minimum base quality (Phred)",
        long_help = "Bases with quality below this threshold are excluded from pileup.\n\
                     Default 20 means >= 99% base call accuracy."
    )]
    pub min_base_quality: u8,

    /// Minimum mapping quality (MAPQ) to include a read in pileup.
    #[arg(
        long = "min-mapping-quality",
        default_value_t = 20,
        help = "Minimum mapping quality (MAPQ)",
        long_help = "Reads with MAPQ below this threshold are excluded from pileup.\n\
                     Default 20. Also filters duplicates, secondary, and supplementary alignments."
    )]
    pub min_mapping_quality: u8,

    // ========== Output options ==========
    /// Backend format for sparse matrix output.
    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix backend (zarr or hdf5)",
        long_help = "Format for per-cell sparse matrices.\n\
                     zarr: Zarr v2 store (default, faster for large datasets)\n\
                     hdf5: HDF5 file (.h5)"
    )]
    pub backend: SparseIoBackend,

    /// Include reads without cell barcode information.
    #[arg(
        long,
        default_value_t = false,
        help = "Include reads without barcodes",
        long_help = "When set, reads without a cell barcode tag are included\n\
                     in pileup and counted under CellBarcode::Missing."
    )]
    pub include_missing_barcode: bool,

    /// Gene type filter (protein_coding, lncRNA, etc.).
    #[arg(
        long,
        value_enum,
        help = "Gene type filter",
        long_help = "Filter GFF genes by type (e.g., protein_coding, lncRNA).\n\
                     Only genes matching this type are processed.\n\
                     Ignored when --gff is not provided."
    )]
    gene_type: Option<GffGeneType>,

    /// Maximum number of threads for parallel processing.
    #[arg(
        long,
        default_value_t = 16,
        help = "Maximum threads",
        long_help = "Maximum number of threads for parallel gene/region processing.\n\
                     Capped at the number of available CPU cores."
    )]
    max_threads: usize,
}

/// Run SNP genotyping at known variant sites and/or de novo discovery.
pub fn run_snp(args: &SnpArgs) -> anyhow::Result<()> {
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

    // Determine discovery mode
    let discover = !args.skip_discovery;
    if args.skip_discovery && args.known_snps_vcf.is_none() {
        return Err(anyhow::anyhow!(
            "--skip-discovery requires --known-snps to be provided"
        ));
    }
    if !discover && args.known_snps_vcf.is_none() {
        return Err(anyhow::anyhow!(
            "either --known-snps or de novo discovery (default) must be enabled"
        ));
    }

    // Load known SNPs if provided
    let known_snps = if let Some(ref vcf_path) = args.known_snps_vcf {
        info!("Loading known SNPs from: {}", vcf_path);
        let snps = load_known_snps(vcf_path)?;
        info!("{} known biallelic SNPs loaded", snps.num_sites());
        Some(snps)
    } else {
        None
    };

    // Load GFF if provided
    let gff_map = if let Some(ref gff_file) = args.gff_file {
        info!("parsing GFF file: {}", gff_file);
        let mut gff = GffRecordMap::from(gff_file.as_ref())?;
        if let Some(gene_type) = args.gene_type.clone() {
            gff.subset(gene_type);
        }
        info!("found {} genes", gff.len());
        Some(gff)
    } else {
        info!("no GFF provided, using region-centric mode");
        None
    };

    let params = SnpParams {
        bam_files: args.bam_files.clone(),
        genome_file: args.genome_file.clone(),
        cell_barcode_tag: args.cell_barcode_tag.clone(),
        gene_barcode_tag: args.gene_barcode_tag.clone(),
        include_missing_barcode: args.include_missing_barcode,
        min_base_quality: args.min_base_quality,
        min_mapping_quality: args.min_mapping_quality,
        genotype_params: GenotypeParams {
            min_depth: args.min_depth,
            min_gq: args.min_gq,
            base_error_rate: args.base_error_rate,
            min_coverage: args.min_coverage,
            min_alt_count: args.min_alt_count,
            min_alt_freq: args.min_alt_freq,
            ..GenotypeParams::default()
        },
        backend: args.backend.clone(),
        output: args.output.clone(),
        bulk: args.bulk,
    };

    let snp_mask = run_snp_pipeline(known_snps.as_ref(), gff_map.as_ref(), &params, discover)?;
    info!("done: {} variant positions in SNP mask", snp_mask.len());

    Ok(())
}
