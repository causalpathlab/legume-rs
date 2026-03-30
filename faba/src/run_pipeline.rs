use crate::common::*;
use crate::data::conversion::ConversionValueType;
use crate::editing::io::ToParquet;
use crate::editing::mask::{build_atoi_mask, filter_conversion_sites_by_mask, filter_m6a_by_mask};
use crate::editing::mixture::MixtureParams;
use crate::editing::pipeline::{
    find_all_conversion_sites, process_all_bam_files_to_backend, run_mixture_model,
    ConversionParams,
};
use crate::editing::sifter::ModificationType;
use crate::gene_count::splice::{count_read_per_gene_splice, format_gene_key};
use crate::pipeline_util::{check_all_bam_indices, extract_gene_key, qc_passing_keys, GeneCountQc};
use crate::snp::genotyper::GenotypeParams;
use crate::snp::io::load_known_snps_auto;
use crate::snp::pipeline::{run_snp_pipeline, SnpParams};

use genomic_data::gff::GffRecordMap;
use genomic_data::sam::CellBarcode;
use log::info;
use rayon::ThreadPoolBuilder;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Args, Debug)]
#[command(
    about = "Run unified RNA-seq pipeline: SNP → genes → ATOI → APA → DART",
    long_about = "Orchestrates the complete RNA-seq analysis pipeline:\n\n\
        0. SNP genotyping (de novo discovery + optional known sites)\n\
        1. Gene expression filtering (identify expressed genes)\n\
        2. ATOI detection (A-to-I editing, masked by SNP)\n\
        3. APA quantification (alternative polyadenylation, masked by SNP+ATOI)\n\
        4. DART analysis (m6A methylation, masked by SNP+ATOI, requires --mut)\n\n\
        Step 0 discovers variants de novo and optionally force-calls at known\n\
        sites (--known-snps, VCF/BCF/Parquet). De novo variants are VAF-filtered\n\
        (--snp-mask-min-vaf) so RNA editing sites are preserved in the mask.\n\
        UMI deduplication is applied to all pileup steps (disable with --no-umi-dedup)."
)]
pub struct PipelineArgs {
    // === Required inputs ===
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "BAM files (WT/observed)",
        long_help = "Comma-separated list of observed (wild-type) BAM files.\n\
                     Used for gene counting, ATOI, APA, and DART quantification."
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
        required = true,
        help = "Reference genome FASTA file (.fa/.fasta, must be indexed)"
    )]
    pub genome_file: Box<str>,

    #[arg(
        short = 'o',
        long = "output",
        required = true,
        help = "Output directory (flat structure)"
    )]
    pub output: Box<str>,

    // === Optional mutant BAMs (if missing, skip DART) ===
    #[arg(
        long = "mut",
        value_delimiter = ',',
        help = "Mutant/control BAM files for DART (skip DART if omitted)"
    )]
    pub mut_bam_files: Option<Vec<Box<str>>>,

    // === Shared parameters ===
    #[arg(long, default_value = "CB", help = "Cell barcode tag")]
    pub cell_barcode_tag: Box<str>,

    #[arg(long, default_value = "GX", help = "Gene barcode tag")]
    pub gene_barcode_tag: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix backend (zarr or hdf5)"
    )]
    pub backend: SparseIoBackend,

    #[arg(long, default_value_t = 16, help = "Maximum number of threads")]
    pub max_threads: usize,

    // === Gene expression filtering ===
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum cells per gene (gene filtering)"
    )]
    pub gene_min_cells: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum UMI per gene (gene filtering)"
    )]
    pub gene_min_counts: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum genes per cell (cell filtering)"
    )]
    pub cell_min_genes: usize,

    // === ATOI parameters ===
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for ATOI detection"
    )]
    pub atoi_min_coverage: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Minimum A-to-G conversions for ATOI"
    )]
    pub atoi_min_conversion: usize,

    #[arg(long, default_value_t = 0.05, help = "ATOI detection p-value cutoff")]
    pub atoi_pvalue_cutoff: f32,

    // === APA parameters ===
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for APA detection"
    )]
    pub apa_min_coverage: usize,

    #[arg(long, default_value_t = 10, help = "Minimum poly(A) tail length")]
    pub polya_min_tail_length: usize,

    // === DART parameters ===
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for m6A detection"
    )]
    pub m6a_min_coverage: usize,

    #[arg(long, default_value_t = 5, help = "Minimum C-to-T conversions for m6A")]
    pub m6a_min_conversion: usize,

    #[arg(long, default_value_t = 0.05, help = "m6A detection p-value cutoff")]
    pub m6a_pvalue_cutoff: f32,

    // === SNP parameters ===
    #[arg(
        long = "known-snps",
        help = "Known SNP sites VCF/BCF/Parquet for SNP masking",
        long_help = "Path to known SNP sites. Accepts:\n\
                     - VCF/BCF (.vcf, .vcf.gz, .bcf): standard variant calls\n\
                     - Parquet (.parquet): output from a previous `faba snp` run\n\
                     When provided, force-calls genotypes at these positions in addition\n\
                     to de novo discovery, and builds a mask for ATOI/APA/DART filtering."
    )]
    pub known_snps: Option<Box<str>>,

    #[arg(long, default_value_t = 5, help = "Minimum depth for SNP calling")]
    pub snp_min_depth: usize,

    #[arg(
        long,
        default_value_t = 20.0,
        help = "Minimum genotype quality for SNP mask"
    )]
    pub snp_min_gq: f32,

    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for de novo SNP discovery"
    )]
    pub snp_min_coverage: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Minimum alt allele reads for SNP discovery"
    )]
    pub snp_min_alt_count: usize,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Minimum alt allele frequency for SNP discovery"
    )]
    pub snp_min_alt_freq: f64,

    #[arg(
        long,
        default_value_t = 0.35,
        help = "Minimum VAF for SNP mask (filters RNA editing from de novo variants)",
        long_help = "Minimum variant allele fraction for a de novo discovered variant\n\
                     to enter the SNP mask. Het sites need VAF in [min_vaf, 1-min_vaf],\n\
                     hom-alt sites need VAF >= 1-min_vaf. Sites with lower VAF are likely\n\
                     RNA editing (A-to-I or m6A) rather than germline SNPs. Set to 0 to\n\
                     disable VAF filtering (mask all called variants)."
    )]
    pub snp_mask_min_vaf: f32,

    // === UMI deduplication (applies to all steps) ===
    #[arg(
        long = "umi-tag",
        default_value = "UB",
        help = "UMI barcode BAM tag for deduplication (all steps)"
    )]
    pub umi_tag: Box<str>,

    #[arg(
        long = "no-umi-dedup",
        default_value_t = false,
        help = "Disable UMI deduplication (for bulk data without UMIs)"
    )]
    pub no_umi_dedup: bool,

    // === Step control ===
    #[arg(long, default_value_t = false, help = "Skip SNP genotyping step")]
    pub skip_snp: bool,

    #[arg(long, default_value_t = false, help = "Skip gene counting step")]
    pub skip_genes: bool,

    #[arg(long, default_value_t = false, help = "Skip ATOI detection step")]
    pub skip_atoi: bool,

    #[arg(long, default_value_t = false, help = "Skip APA quantification step")]
    pub skip_apa: bool,
}

pub fn run_pipeline(args: &PipelineArgs) -> anyhow::Result<()> {
    // 0. Setup
    info!("=== faba pipeline: unified RNA-seq analysis ===");
    ThreadPoolBuilder::new()
        .num_threads(args.max_threads)
        .build_global()?;
    std::fs::create_dir_all(&*args.output)?;

    // Validate inputs
    check_all_bam_indices(&args.bam_files)?;
    if let Some(ref mut_bams) = args.mut_bam_files {
        check_all_bam_indices(mut_bams)?;
    }

    let n_steps = 5;

    // Step 0: SNP genotyping (de novo discovery + optional known sites).
    // VAF filtering prevents masking true RNA editing sites from de novo variants.
    let snp_mask = if !args.skip_snp {
        info!("=== Step 0/{}: SNP genotyping ===", n_steps);
        match run_snp_step(args) {
            Ok(mask) => {
                info!("SNP complete: {} variant positions in mask", mask.len());
                Some(mask)
            }
            Err(e) => {
                log::warn!("SNP step failed: {}. Continuing without SNP mask.", e);
                None
            }
        }
    } else {
        info!("=== Step 0/{}: SKIPPED (--skip-snp) ===", n_steps);
        None
    };

    // Step 1: Gene Expression Filtering
    let gene_count_qc = if !args.skip_genes {
        info!("=== Step 1/{}: Gene expression filtering ===", n_steps);
        run_gene_counting_step(args)?
    } else {
        info!("=== Step 1/{}: SKIPPED (--skip-genes) ===", n_steps);
        None
    };

    // Step 2: ATOI Detection
    let atoi_mask = if !args.skip_atoi {
        info!("=== Step 2/{}: ATOI detection ===", n_steps);
        match run_atoi_step(args, &gene_count_qc, &snp_mask) {
            Ok(mask_data) => {
                info!(
                    "ATOI complete: {} sites, {} mask positions",
                    mask_data.n_sites,
                    mask_data.mask.len()
                );
                Some(mask_data)
            }
            Err(e) => {
                log::warn!("ATOI step failed: {}. Continuing without mask.", e);
                None
            }
        }
    } else {
        info!("=== Step 2/{}: SKIPPED (--skip-atoi) ===", n_steps);
        None
    };

    // Step 3: APA Analysis
    if !args.skip_apa {
        info!("=== Step 3/{}: APA analysis ===", n_steps);
        match run_apa_step(args, &atoi_mask, &snp_mask, &gene_count_qc) {
            Ok(_) => info!("APA complete"),
            Err(e) => log::warn!("APA step failed: {}. Continuing to DART.", e),
        }
    } else {
        info!("=== Step 3/{}: SKIPPED (--skip-apa) ===", n_steps);
    }

    // Step 4: DART Analysis (conditional on mutant data)
    if let Some(ref mut_bams) = args.mut_bam_files {
        info!("=== Step 4/{}: DART analysis ===", n_steps);
        match run_dart_step(args, mut_bams, &atoi_mask, &snp_mask, &gene_count_qc) {
            Ok(_) => info!("DART complete"),
            Err(e) => log::warn!("DART step failed: {}", e),
        }
    } else {
        info!("=== Step 4/{}: SKIPPED (no --mut BAM files) ===", n_steps);
    }

    write_pipeline_summary(args)?;
    info!("Pipeline complete! Results in: {}", args.output);
    Ok(())
}

struct AtoiMaskData {
    mask: rustc_hash::FxHashSet<(Box<str>, i64)>,
    n_sites: usize,
}

// Step 0: SNP genotyping (discovery + optional known sites).
// VAF filter ensures de novo variants only enter the mask if they have
// germline-like allele fractions, preserving RNA editing sites.
fn run_snp_step(args: &PipelineArgs) -> anyhow::Result<FxHashSet<(Box<str>, i64)>> {
    let known_snps = if let Some(ref path) = args.known_snps {
        info!("Loading known SNPs from: {}", path);
        let snps = load_known_snps_auto(path)?;
        info!("{} known biallelic SNPs loaded", snps.num_sites());
        Some(snps)
    } else {
        None
    };

    let gff_map = GffRecordMap::from(args.gff_file.as_ref())?;

    let umi_tag = if args.no_umi_dedup {
        None
    } else {
        Some(args.umi_tag.clone())
    };

    let min_vaf = if args.snp_mask_min_vaf > 0.0 {
        Some(args.snp_mask_min_vaf)
    } else {
        None
    };

    let params = SnpParams {
        bam_files: args.bam_files.clone(),
        genome_file: args.genome_file.clone(),
        cell_barcode_tag: args.cell_barcode_tag.clone(),
        gene_barcode_tag: args.gene_barcode_tag.clone(),
        include_missing_barcode: false,
        min_base_quality: 20,
        min_mapping_quality: 20,
        genotype_params: GenotypeParams {
            min_depth: args.snp_min_depth,
            min_gq: args.snp_min_gq,
            min_coverage: args.snp_min_coverage,
            min_alt_count: args.snp_min_alt_count,
            min_alt_freq: args.snp_min_alt_freq,
            ..GenotypeParams::default()
        },
        backend: args.backend.clone(),
        output: args.output.clone(),
        bulk: true,
        umi_tag,
        use_base_quality: true,
        min_vaf,
    };

    // Discover de novo + force-call known sites if provided
    run_snp_pipeline(known_snps.as_ref(), Some(&gff_map), &params, true)
}

// Step 1: Gene expression filtering (splice-aware: spliced + unspliced in one backend)
fn run_gene_counting_step(args: &PipelineArgs) -> anyhow::Result<Option<GeneCountQc>> {
    // Parse GFF once to get both gene map and exon intervals
    let all_records = read_gff_record_vec(args.gff_file.as_ref())?;
    let gene_map = build_gene_map(&all_records, Some(&FeatureType::Gene))?;
    let exon_map = build_exon_intervals(&all_records);
    let exon_intervals: FxHashMap<GeneId, Vec<(i64, i64)>> = exon_map.into_iter().collect();

    let gff_map = GffRecordMap::from_map(gene_map);
    info!("Loaded {} genes for expression filtering", gff_map.len());

    // Combine all BAM files (WT + mut)
    let mut all_bam_files = args.bam_files.clone();
    if let Some(ref mut_files) = args.mut_bam_files {
        all_bam_files.extend_from_slice(mut_files);
    }

    info!("Gene counting across {} BAM files:", all_bam_files.len());
    for bam in &all_bam_files {
        info!("  {}", bam);
    }

    let batch_names = uniq_batch_names(&all_bam_files)?;
    let records = gff_map.records();

    // Build gene_key → GeneId mapping for reverse lookup after QC
    let gene_key_to_id: FxHashMap<Box<str>, GeneId> = records
        .iter()
        .map(|rec| (format_gene_key(rec), rec.gene_id.clone()))
        .collect();

    let mut expressed_gene_ids: rustc_hash::FxHashSet<GeneId> = rustc_hash::FxHashSet::default();
    let mut valid_cell_barcodes: rustc_hash::FxHashSet<CellBarcode> =
        rustc_hash::FxHashSet::default();

    for (bam_file, batch_name) in all_bam_files.iter().zip(batch_names.iter()) {
        let njobs = records.len() as u64;
        info!(
            "Counting genes (splice-aware) in {} ({} genes)",
            batch_name, njobs
        );

        // Splice-aware counting via par_iter
        let results: Vec<_> = records
            .par_iter()
            .progress_count(njobs)
            .map(|rec| {
                count_read_per_gene_splice(
                    bam_file,
                    rec,
                    &exon_intervals,
                    &args.cell_barcode_tag,
                    &args.gene_barcode_tag,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut spliced_triplets = Vec::new();
        let mut unspliced_triplets = Vec::new();
        for r in results {
            spliced_triplets.extend(r.spliced);
            unspliced_triplets.extend(r.unspliced);
        }

        info!(
            "{}: {} spliced, {} unspliced triplets",
            batch_name,
            spliced_triplets.len(),
            unspliced_triplets.len()
        );

        // In-memory QC on total counts
        let (passing_genes, passing_cells) = qc_passing_keys(
            &spliced_triplets,
            &unspliced_triplets,
            args.gene_min_cells,
            args.cell_min_genes,
        );

        info!(
            "{}: {} genes, {} cells passed QC",
            batch_name,
            passing_genes.len(),
            passing_cells.len()
        );

        // Filter triplets to QC-passing genes and cells (par_iter)
        let filter_triplets =
            |triplets: Vec<(CellBarcode, Box<str>, f32)>| -> Vec<(CellBarcode, Box<str>, f32)> {
                triplets
                    .into_par_iter()
                    .filter(|(cb, feat, _)| {
                        let gk: Box<str> = extract_gene_key(feat).into();
                        passing_genes.contains(&gk) && passing_cells.contains(cb)
                    })
                    .collect()
            };

        let spliced_triplets = filter_triplets(spliced_triplets);
        let unspliced_triplets = filter_triplets(unspliced_triplets);

        // Merge spliced + unspliced into one backend
        let UnionNames {
            col_names,
            cell_to_index,
            row_names,
            feature_to_index,
        } = collect_union_names(&spliced_triplets, &unspliced_triplets);

        let backend_file = match args.backend {
            SparseIoBackend::HDF5 => format!("{}/{}_genes.h5", &args.output, batch_name),
            SparseIoBackend::Zarr => format!("{}/{}_genes.zarr", &args.output, batch_name),
        };

        let merged: Vec<_> = spliced_triplets
            .into_iter()
            .chain(unspliced_triplets)
            .collect();

        format_data_triplets_shared(
            merged,
            &feature_to_index,
            &cell_to_index,
            row_names,
            col_names,
        )
        .to_backend(&backend_file)?;

        info!(
            "{}: wrote spliced + unspliced to {}",
            batch_name, backend_file
        );

        // Map passing gene keys back to GeneIds
        for gene_key in &passing_genes {
            if let Some(gene_id) = gene_key_to_id.get(gene_key) {
                expressed_gene_ids.insert(gene_id.clone());
            }
        }

        // Collect QC-passing cell barcodes
        valid_cell_barcodes.extend(passing_cells);
    }

    info!(
        "Gene filtering: {} genes passed QC across {} BAM files",
        expressed_gene_ids.len(),
        all_bam_files.len()
    );

    info!(
        "Cell filtering: {} cells passed QC across {} BAM files",
        valid_cell_barcodes.len(),
        all_bam_files.len()
    );

    Ok(Some(GeneCountQc {
        gene_ids: expressed_gene_ids,
        cell_barcodes: valid_cell_barcodes,
    }))
}

// Step 2: ATOI detection
fn run_atoi_step(
    args: &PipelineArgs,
    gene_count_qc: &Option<GeneCountQc>,
    snp_mask: &Option<FxHashSet<(Box<str>, i64)>>,
) -> anyhow::Result<AtoiMaskData> {
    // Load GFF and filter to expressed genes
    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref())?;
    if let Some(ref eg) = gene_count_qc {
        gff_map.retain_by_ids(&eg.gene_ids);
        info!("Filtered to {} expressed genes", gff_map.len());
    } else {
        info!("Loaded {} genes (no expression filter)", gff_map.len());
    }

    // Build ConversionParams for ATOI
    let mut_bams = args.mut_bam_files.as_ref().cloned().unwrap_or_default();
    let has_mut_files = !mut_bams.is_empty();

    let params = ConversionParams {
        mod_type: ModificationType::AtoI,
        genome_file: args.genome_file.clone(),
        wt_bam_files: args.bam_files.clone(),
        mut_bam_files: mut_bams,
        gene_barcode_tag: args.gene_barcode_tag.clone(),
        cell_barcode_tag: args.cell_barcode_tag.clone(),
        include_missing_barcode: false,
        min_coverage: args.atoi_min_coverage,
        min_conversion: args.atoi_min_conversion,
        pseudocount: 1,
        pvalue_cutoff: args.atoi_pvalue_cutoff,
        backend: args.backend.clone(),
        output: args.output.clone(),
        output_value_type: ConversionValueType::Ratio,
        row_nnz_cutoff: None,
        column_nnz_cutoff: None,
        cell_membership_file: None,
        membership_barcode_col: 0,
        membership_celltype_col: 1,
        exact_barcode_match: false,
        min_base_quality: 20,
        min_mapping_quality: 20,
    };

    // Find ATOI sites (first pass)
    info!("Discovering ATOI sites...");
    let atoi_sites = find_all_conversion_sites(&gff_map, &params, None)?;

    // Apply SNP mask if available
    if let Some(ref mask) = snp_mask {
        let n_before: usize = atoi_sites.iter().map(|e| e.value().len()).sum();
        filter_conversion_sites_by_mask(&atoi_sites, mask, &gff_map);
        let n_after: usize = atoi_sites.iter().map(|e| e.value().len()).sum();
        info!(
            "SNP masking: {} → {} ATOI sites ({} removed)",
            n_before,
            n_after,
            n_before - n_after
        );
    }

    // Count total sites
    let n_sites: usize = atoi_sites.iter().map(|entry| entry.value().len()).sum();
    info!("Found {} ATOI sites", n_sites);

    // Build mask
    let mask = build_atoi_mask(&atoi_sites, &gff_map);

    // Save site annotations
    let sites_output = format!("{}/atoi_sites.parquet", args.output);
    atoi_sites.to_parquet(&gff_map, &sites_output)?;
    info!("Saved ATOI sites to {}", sites_output);

    // Second pass: quantification (WT + mut if available)
    if has_mut_files {
        info!("Quantifying ATOI sites per cell (WT + mut files)...");
    } else {
        info!("Quantifying ATOI sites per cell (WT files only)...");
    }
    let valid_cells = gene_count_qc.as_ref().map(|qc| &qc.cell_barcodes);
    process_all_bam_files_to_backend(&params, &atoi_sites, &gff_map, has_mut_files, valid_cells)?;

    // Mixture model: cluster editing sites per gene
    info!("Running 1D Gaussian mixture model on A-to-I sites...");
    let mix_params = MixtureParams::default();
    run_mixture_model(&params, &atoi_sites, &gff_map, &mix_params)?;

    Ok(AtoiMaskData { mask, n_sites })
}

// Step 3: APA analysis
fn run_apa_step(
    args: &PipelineArgs,
    atoi_mask: &Option<AtoiMaskData>,
    snp_mask: &Option<FxHashSet<(Box<str>, i64)>>,
    gene_count_qc: &Option<GeneCountQc>,
) -> anyhow::Result<()> {
    use crate::run_apa::{run_apa, ApaMethod, CountApaArgs};

    // Save ATOI mask to file if available
    let atoi_mask_file = if let Some(ref _mask_data) = atoi_mask {
        let mask_path = format!("{}/atoi_sites.parquet", args.output);
        info!("APA will use ATOI mask from: {}", mask_path);
        Some(mask_path.into_boxed_str())
    } else {
        None
    };

    // SNP mask file path if SNP step was run
    let snp_mask_file = if snp_mask.is_some() {
        let mask_path = format!("{}/snp_sites.parquet", args.output);
        info!("APA will use SNP mask from: {}", mask_path);
        Some(mask_path.into_boxed_str())
    } else {
        None
    };

    // Combine WT + mut BAM files for APA quantification
    let mut all_bam_files = args.bam_files.clone();
    if let Some(ref mut_files) = args.mut_bam_files {
        if !mut_files.is_empty() {
            info!(
                "APA will quantify {} WT + {} mut BAM files",
                all_bam_files.len(),
                mut_files.len()
            );
            all_bam_files.extend_from_slice(mut_files);
        }
    }

    // Extract valid gene IDs and cell barcodes from gene count QC
    let (valid_gene_ids, valid_cell_barcodes) = match gene_count_qc {
        Some(qc) => {
            info!(
                "APA will restrict to {} genes, {} cells",
                qc.gene_ids.len(),
                qc.cell_barcodes.len()
            );
            (Some(qc.gene_ids.clone()), Some(qc.cell_barcodes.clone()))
        }
        None => (None, None),
    };

    // Build CountApaArgs from PipelineArgs
    let mut apa_args = CountApaArgs {
        bam_files: all_bam_files,
        gff_file: Some(args.gff_file.clone()),
        cell_barcode_tag: args.cell_barcode_tag.clone(),
        polya_min_tail_length: args.polya_min_tail_length,
        polya_max_non_a_or_t: 3,
        polya_internal_prime_window: 10,
        polya_internal_prime_count: 7,
        min_coverage: args.apa_min_coverage,
        max_threads: args.max_threads,
        row_nnz_cutoff: 10,
        column_nnz_cutoff: 1,
        output: args.output.clone(),
        backend: args.backend.clone(),
        method: ApaMethod::Mixture, // Always use mixture mode (more robust)
        atoi_mask_file,
        snp_mask_file,
        gene_barcode_tag: args.gene_barcode_tag.clone(),
        resolution_bp: 10,
        include_missing_barcode: false,
        record_type: None,
        gene_type: None,
        utr_bed: None,
        min_utr_length: 200,
        pre_sites: None,
        umi_tag: "UB".into(),
        mu_f: 300.0,
        sigma_f: 50.0,
        theta_step: 10,
        max_beta: 70.0,
        min_beta: 10.0,
        min_ws: 0.01,
        min_fragments: 50,
        merge_distance: 50.0,
        compute_pdui: true,
        gene_min_cells: args.gene_min_cells,
        cell_min_genes: args.cell_min_genes,
        skip_gene_qc: true, // Pipeline already did gene QC in step 1
        valid_gene_ids,
        valid_cell_barcodes,
    };

    run_apa(&mut apa_args)?;
    Ok(())
}

// Step 4: DART analysis
fn run_dart_step(
    args: &PipelineArgs,
    mut_bams: &[Box<str>],
    atoi_mask: &Option<AtoiMaskData>,
    snp_mask: &Option<FxHashSet<(Box<str>, i64)>>,
    gene_count_qc: &Option<GeneCountQc>,
) -> anyhow::Result<()> {
    // Load GFF and filter to expressed genes
    let mut gff_map = GffRecordMap::from(args.gff_file.as_ref())?;
    if let Some(ref eg) = gene_count_qc {
        gff_map.retain_by_ids(&eg.gene_ids);
        info!("Filtered to {} expressed genes", gff_map.len());
    } else {
        info!("Loaded {} genes (no expression filter)", gff_map.len());
    }

    // Build ConversionParams for m6A (DART)
    let params = ConversionParams {
        mod_type: ModificationType::M6A { check_r_site: true },
        genome_file: args.genome_file.clone(),
        wt_bam_files: args.bam_files.clone(),
        mut_bam_files: mut_bams.to_vec(),
        gene_barcode_tag: args.gene_barcode_tag.clone(),
        cell_barcode_tag: args.cell_barcode_tag.clone(),
        include_missing_barcode: false,
        min_coverage: args.m6a_min_coverage,
        min_conversion: args.m6a_min_conversion,
        pseudocount: 1,
        pvalue_cutoff: args.m6a_pvalue_cutoff,
        backend: args.backend.clone(),
        output: args.output.clone(),
        output_value_type: ConversionValueType::Ratio,
        row_nnz_cutoff: None,
        column_nnz_cutoff: None,
        cell_membership_file: None,
        membership_barcode_col: 0,
        membership_celltype_col: 1,
        exact_barcode_match: false,
        min_base_quality: 20,
        min_mapping_quality: 20,
    };

    // Find m6A sites (first pass)
    info!("Discovering m6A sites...");
    let m6a_sites = find_all_conversion_sites(&gff_map, &params, None)?;

    let n_sites_before: usize = m6a_sites.iter().map(|e| e.value().len()).sum();
    info!("Found {} m6A sites before masking", n_sites_before);

    // Apply ATOI mask if available
    if let Some(ref mask_data) = atoi_mask {
        info!("Applying ATOI mask to m6A sites...");
        filter_m6a_by_mask(&m6a_sites, &mask_data.mask, &gff_map);
        let n_sites_after: usize = m6a_sites.iter().map(|e| e.value().len()).sum();
        info!(
            "Retained {} m6A sites after ATOI masking (removed {})",
            n_sites_after,
            n_sites_before - n_sites_after
        );
    }

    // Apply SNP mask if available
    if let Some(ref mask) = snp_mask {
        let n_before: usize = m6a_sites.iter().map(|e| e.value().len()).sum();
        filter_m6a_by_mask(&m6a_sites, mask, &gff_map);
        let n_after: usize = m6a_sites.iter().map(|e| e.value().len()).sum();
        info!(
            "SNP masking: {} → {} m6A sites ({} removed)",
            n_before,
            n_after,
            n_before - n_after
        );
    }

    // Save site annotations
    let sites_output = format!("{}/m6a_sites.parquet", args.output);
    m6a_sites.to_parquet(&gff_map, &sites_output)?;
    info!("Saved m6A sites to {}", sites_output);

    // Second pass: quantification (wt + mut)
    info!("Quantifying m6A sites per cell...");
    let valid_cells = gene_count_qc.as_ref().map(|qc| &qc.cell_barcodes);
    process_all_bam_files_to_backend(&params, &m6a_sites, &gff_map, true, valid_cells)?;

    // Mixture model: cluster modification sites per gene
    info!("Running 1D Gaussian mixture model on m6A sites...");
    let mix_params = MixtureParams::default();
    run_mixture_model(&params, &m6a_sites, &gff_map, &mix_params)?;

    Ok(())
}

// Write pipeline summary
fn write_pipeline_summary(args: &PipelineArgs) -> anyhow::Result<()> {
    use std::io::Write;
    let summary_path = format!("{}/pipeline_summary.json", args.output);
    let mut file = std::fs::File::create(&summary_path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"bam_files\": {:?},", args.bam_files)?;
    writeln!(file, "  \"gff_file\": {:?},", args.gff_file)?;
    writeln!(file, "  \"genome_file\": {:?},", args.genome_file)?;
    writeln!(
        file,
        "  \"mut_bam_files\": {},",
        if args.mut_bam_files.is_some() {
            "provided"
        } else {
            "null"
        }
    )?;
    writeln!(file, "  \"output\": {:?}", args.output)?;
    writeln!(file, "}}")?;
    info!("Wrote pipeline summary to {}", summary_path);
    Ok(())
}
