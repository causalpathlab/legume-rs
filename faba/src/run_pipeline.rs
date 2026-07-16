use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::editing::io::ToParquet;
use crate::editing::mask::{build_atoi_mask, filter_conversion_sites_by_mask, filter_m6a_by_mask};
use crate::editing::mixture::MixtureParams;
use crate::editing::mixture_pipeline::run_mixture_model;
use crate::editing::pipeline::{
    find_all_conversion_sites, process_all_bam_files_to_backend, ConversionParams, M6aContrastArgs,
};
use crate::editing::sifter::ModificationType;
use crate::pipeline_util::mass_enrichment::MassEnrichmentArgs;
use crate::pipeline_util::{check_all_bam_indices, GeneCountQc};
use crate::snp::genotyper::GenotypeParams;
use crate::snp::io::load_known_snps_auto;
use crate::snp::pipeline::{run_snp_pipeline, SnpParams};

use anyhow::Context;
use genomic_data::gff::GffRecordMap;
use log::info;
use rayon::ThreadPoolBuilder;
use rustc_hash::FxHashSet;

/// Serialize a field by its `Debug` form — for foreign enums that carry no `Serialize`.
fn ser_debug<T: std::fmt::Debug, S: serde::Serializer>(v: &T, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&format!("{v:?}"))
}

#[derive(Args, Debug, serde::Serialize)]
#[command(
    about = "Run unified RNA-seq pipeline: SNP → genes → ATOI → m6A → APA",
    long_about = "Orchestrates the complete RNA-seq analysis pipeline:\n\n\
        0. SNP genotyping (de novo discovery + optional known sites)\n\
        1. Gene expression filtering (identify expressed genes)\n\
        2. ATOI detection (A-to-I editing, masked by SNP)\n\
        3. APA quantification (alternative polyadenylation, masked by SNP+ATOI)\n\
        4. m6A detection (DART C→T, WT-vs-MUT contrast; skipped without --control-bam)\n\n\
        ATOI is reference-anchored and tested per site\n\
        against a beta-binomial sequencing-error null (--edit-error-rate/--edit-overdispersion),\n\
        no control sample.\n\
        m6A instead requires a catalytically-dead control (--control-bam):\n\
        each motif C is tested for higher conversion in the positional BAMs\n\
        than the pooled control (so a genomic C/T variant is rejected);\n\
        the step is skipped when no control is given.\n\
        The SNP mask is off by default for m6A\n\
        (the contrast already rejects variants; opt back in with --m6a-snp-mask).\n\
        Step 0 discovers variants de novo and optionally force-calls at known sites\n\
        (--known-snps, VCF/BCF/Parquet).\n\
        De novo variants are VAF-filtered (--snp-mask-min-vaf)\n\
        so RNA editing sites are preserved in the mask.\n\
        UMI deduplication is applied to all pileup steps (disable with --no-umi-dedup)."
)]
pub struct PipelineArgs {
    // Required inputs
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input BAM files (comma-separated)",
        long_help = "Comma-separated BAM files used across every modality:\n\
                     gene counting, ATOI, APA and m6A quantification."
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

    #[arg(
        long = "control-bam",
        alias = "mut",
        alias = "control",
        value_delimiter = ',',
        help = "Control BAM files (catalytically-dead YTHmut) for the m6A contrast",
        long_help = "Comma-separated control (catalytically-dead YTHmut) BAM files.\n\
                     m6A is called by a WT-vs-MUT contrast:\n\
                     the signal arm is the positional BAMs MINUS these controls.\n\
                     That split is used only for m6A site discovery;\n\
                     otherwise these controls are quantified like positional samples.\n\
                     Their SNP, gene, ATOI, APA and m6A per-cell matrices are produced too,\n\
                     with cells frozen per control BAM in step 1.\n\
                     Optional, but the m6A (DART) step is skipped without it:\n\
                     m6A cannot be separated from genomic C/T variation without a control."
    )]
    pub control_bam_files: Vec<Box<str>>,

    ///////////////////////
    // Shared parameters //
    ///////////////////////
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
    // `SparseIoBackend` is data-beans' and does not implement `Serialize`; it is a plain
    // enum, so its `Debug` form ("Zarr" / "Hdf5") is exactly what belongs in the summary.
    #[serde(serialize_with = "ser_debug")]
    pub backend: SparseIoBackend,

    #[arg(
        long = "no-zip",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive",
        long_help = "Keep a `.zarr` directory instead of producing a `.zarr.zip` archive\n\
                     (zarr backend only; no effect on hdf5)"
    )]
    pub zip: bool,

    #[arg(
        long,
        alias = "threads",
        default_value_t = 16,
        help = "Maximum number of threads"
    )]
    pub max_threads: usize,

    ///////////////////////////////
    // Gene expression filtering //
    ///////////////////////////////
    #[arg(
        long,
        default_value_t = 0,
        help = "Minimum cells per gene (gene filtering; 0 = off, matching \
                Cell Ranger, which keeps every gene/feature)"
    )]
    pub gene_min_cells: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Minimum UMI per gene (gene filtering; 0 = off, matching \
                Cell Ranger, which keeps every gene/feature)"
    )]
    pub gene_min_counts: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Minimum detected genes (nnz) per cell; cells below this are \
                dropped from gene counts and every downstream modality. \
                0 = off (default), so cell calling is pure Cell Ranger \
                EmptyDrops/OrdMag with no extra min-genes floor"
    )]
    pub cell_min_genes: usize,

    #[command(flatten)]
    pub cell_qc: crate::cell_qc::CellQcArgs,

    //////////////////////////////////////////
    // Gene biotype (quantification subset) //
    //////////////////////////////////////////
    #[arg(
        long,
        default_value = "",
        help = "Gene biotype to quantify. Empty (default) keeps all biotypes. \
                Pass a value to restrict: protein_coding, lncRNA, pseudogene. \
                QC/cell-calling always uses ALL biotypes; only the quantified \
                gene set (gene counts + ATOI/APA/m6A) is restricted to this type."
    )]
    pub gene_type: Box<str>,

    //////////////////////
    // Mitochondrial QC //
    //////////////////////
    #[command(flatten)]
    pub mito_qc: crate::pipeline_util::MitoQcArgs,

    ////////////////////////////////////////////////////
    // Shared read-quality filters (ATOI / m6A / SNP) //
    ////////////////////////////////////////////////////
    #[arg(
        long,
        default_value_t = 20,
        help = "Minimum base quality for editing/SNP base calls (ATOI/m6A/SNP)"
    )]
    pub min_base_quality: u8,

    #[arg(
        long,
        default_value_t = 20,
        help = "Minimum mapping quality for editing/SNP reads (ATOI/m6A/SNP)"
    )]
    pub min_mapping_quality: u8,

    #[arg(
        long = "no-apa-pdui",
        default_value_t = false,
        help = "Skip the APA PDUI (proximal/distal count) matrix output ({batch}_apa)"
    )]
    pub no_apa_pdui: bool,

    /////////////////////
    // ATOI parameters //
    /////////////////////
    #[arg(
        long,
        default_value_t = 5,
        help = "Minimum coverage for ATOI detection (matches `faba atoi`)"
    )]
    pub atoi_min_coverage: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Minimum A-to-G conversions for ATOI (matches `faba atoi`)"
    )]
    pub atoi_min_conversion: usize,

    #[arg(
        long = "atoi-fdr",
        default_value_t = 0.05,
        help = "ATOI detection FDR target (Benjamini-Hochberg q-value)"
    )]
    pub atoi_fdr_cutoff: f32,

    ///////////////////////////////////////////////////////
    // Editing statistical null (shared by ATOI and m6A) //
    ///////////////////////////////////////////////////////
    #[arg(
        long = "edit-error-rate",
        alias = "error-rate",
        default_value_t = 0.01,
        help = "Sequencing-error rate ε: the beta-binomial null mean the edited \
                fraction is tested against (reference-anchored, no control sample)"
    )]
    pub edit_error_rate: f64,

    #[arg(
        long = "edit-overdispersion",
        alias = "overdispersion",
        default_value_t = 0.1,
        help = "Beta-binomial overdispersion ρ for the editing null (0 ⇒ binomial)"
    )]
    pub edit_overdispersion: f64,

    ////////////////////
    // APA parameters //
    ////////////////////
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for APA detection"
    )]
    pub apa_min_coverage: usize,

    #[arg(
        long = "apa-max-sites",
        default_value_t = 20,
        help = "Cap candidate poly-A sites per UTR for APA BIC selection \
                (top-N by coverage; 0 = unlimited). Bounds EM cost on long 3'UTRs."
    )]
    pub apa_max_sites: usize,

    #[arg(
        long = "apa-em-pdui",
        default_value_t = false,
        help = "Use the full SCAPE EM for PDUI instead of the fast top-2 nearest-site \
                assignment (slower; --mixture also forces the EM)"
    )]
    pub apa_em_pdui: bool,

    #[arg(long, default_value_t = 10, help = "Minimum poly(A) tail length")]
    pub polya_min_tail_length: usize,

    /////////////////////
    // DART parameters //
    /////////////////////
    #[arg(
        long,
        default_value_t = 10,
        help = "Minimum coverage for m6A detection"
    )]
    pub m6a_min_coverage: usize,

    #[arg(long, default_value_t = 5, help = "Minimum C-to-T conversions for m6A")]
    pub m6a_min_conversion: usize,

    #[arg(
        long = "m6a-fdr",
        default_value_t = 0.05,
        help = "m6A detection FDR target (Benjamini-Hochberg q-value)"
    )]
    pub m6a_fdr_cutoff: f32,

    #[command(flatten)]
    pub m6a_contrast: M6aContrastArgs,

    /// Apply the SNP mask to m6A calls. Off by default: with the WT-vs-MUT
    /// contrast a genomic variant is rejected automatically, so the mask is
    /// redundant (and was over-aggressive).
    #[arg(long = "m6a-snp-mask", default_value_t = false)]
    pub m6a_snp_mask: bool,

    ////////////////////////////////////////////////////////
    // Mixture model weighting (shared by m6A and A-to-I) //
    ////////////////////////////////////////////////////////
    #[arg(
        long = "mixture-weight",
        value_enum,
        default_value_t = crate::editing::pipeline::MixtureWeightMode::Posterior,
        help = "How to weight each (cell, site) observation in the mixture EM"
    )]
    pub mixture_weight: crate::editing::pipeline::MixtureWeightMode,

    #[arg(
        long = "mixture-prior-alpha",
        default_value_t = 1.0,
        help = "Beta prior α for posterior-rate weighting (default: 1.0)"
    )]
    pub mixture_prior_alpha: f32,

    #[arg(
        long = "mixture-prior-beta",
        default_value_t = 1.0,
        help = "Beta prior β for posterior-rate weighting (default: 1.0)"
    )]
    pub mixture_prior_beta: f32,

    #[arg(
        long = "drop-single-component",
        default_value_t = false,
        help = "Drop genes with a single mixture component across m6A/ATOI/APA"
    )]
    pub drop_single_component: bool,

    #[arg(
        long = "mixture",
        default_value_t = false,
        help = "Also produce the per-gene component-mixture `_{modality}_mixture` \
                matrices (EM; slow). Off by default — only the gene-level \
                `{gene}/{modality}/{channel}` counts are produced.",
        long_help = "Also produce the per-gene component-mixture matrices (EM; slow).\n\
                     Off by default — only the gene-level\n\
                     `{gene}/{modality}/{channel}` counts are produced.\n\n\
                     For m6A / A-to-I this SKIPS the 1-D Gaussian mixture EM entirely when off.\n\
                     For APA the SCAPE poly-A fit always runs\n\
                     (PDUI needs it to identify proximal vs distal),\n\
                     so this gates only the extra `_apa_mixture` component-matrix output."
    )]
    pub mixture: bool,

    ////////////////////
    // SNP parameters //
    ////////////////////
    #[arg(
        long = "known-snps",
        help = "Known SNP sites VCF/BCF/Parquet for SNP masking",
        long_help = "Path to known SNP sites. Accepts:\n\
                     - VCF/BCF (.vcf, .vcf.gz, .bcf): standard variant calls\n\
                     - Parquet (.parquet): output from a previous `faba snp` run\n\
                     When provided, force-calls genotypes at these positions\n\
                     in addition to de novo discovery,\n\
                     and builds a mask for ATOI/APA/DART filtering."
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
                     to enter the SNP mask.\n\
                     A het site needs VAF in the range [min_vaf, 1-min_vaf];\n\
                     a hom-alt site needs VAF >= 1-min_vaf.\n\
                     Sites with lower VAF are likely RNA editing (A-to-I or m6A)\n\
                     rather than germline SNPs.\n\
                     Set to 0 to disable VAF filtering (mask all called variants)."
    )]
    pub snp_mask_min_vaf: f32,

    //////////////////////////////////////////////
    // UMI deduplication (applies to all steps) //
    //////////////////////////////////////////////
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

    //////////////////
    // Step control //
    //////////////////
    #[arg(long, default_value_t = false, help = "Skip SNP genotyping step")]
    pub skip_snp: bool,

    #[arg(long, default_value_t = false, help = "Skip gene counting step")]
    pub skip_genes: bool,

    #[arg(long, default_value_t = false, help = "Skip ATOI detection step")]
    pub skip_atoi: bool,

    #[arg(long, default_value_t = false, help = "Skip APA quantification step")]
    pub skip_apa: bool,

    //////////////////////////////
    // Mass-enrichment grouping //
    //////////////////////////////
    #[command(flatten)]
    pub enrich: MassEnrichmentArgs,
}

/// The full set of samples to QUANTIFY in every modality: the positional
/// (signal/WT) BAMs together with the `--control-bam` (MUT/YTHmut) BAMs,
/// deduplicated (a BAM may legitimately be listed in both roles). The WT-vs-MUT
/// split is used ONLY for the m6A discovery contrast (step 4); SNP, gene counts,
/// ATOI, APA and m6A per-cell matrices are all produced for EVERY one of these
/// samples, so the control background is fully quantified — not merely consumed
/// as an m6A reference. This also freezes a cell set for each control BAM in
/// step 1, so the control m6A matrices reuse it instead of the ambient superset.
fn all_quant_bam_files(args: &PipelineArgs) -> Vec<Box<str>> {
    let (files, dropped) = unique_bam_files(
        args.bam_files
            .iter()
            .chain(args.control_bam_files.iter())
            .cloned(),
    );
    if dropped > 0 {
        log::warn!(
            "{dropped} BAM file(s) listed both positionally and in --control-bam; \
             quantifying each once to avoid double counting"
        );
    }
    files
}

pub fn run_pipeline(args: &PipelineArgs) -> anyhow::Result<()> {
    // 0. Setup
    info!("faba pipeline: unified RNA-seq analysis");
    ThreadPoolBuilder::new()
        .num_threads(args.max_threads)
        .build_global()?;
    std::fs::create_dir_all(&*args.output)?;

    // Validate inputs
    check_all_bam_indices(&args.bam_files)?;
    check_all_bam_indices(&args.control_bam_files)?;

    let n_steps = 5;

    // Step 0: SNP genotyping (de novo discovery + optional known sites).
    // VAF filtering prevents masking true RNA editing sites from de novo variants.
    let snp_mask = if !args.skip_snp {
        info!("Step 0/{}: SNP genotyping", n_steps);
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
        info!("Step 0/{}: SKIPPED (--skip-snp)", n_steps);
        None
    };

    // Step 1: Gene Expression Filtering
    let gene_count_qc = if !args.skip_genes {
        info!("Step 1/{}: Gene expression filtering", n_steps);
        run_gene_counting_step(args)?
    } else {
        info!("Step 1/{}: SKIPPED (--skip-genes)", n_steps);
        None
    };

    // Mass-enrichment grouping (shared instrument for stratified discovery).
    // Built once over all quantified cells so ATOI (all samples) and m6A (signal
    // arm ⊆ all samples) stratify on the same groups; `None` when disabled
    // (`--n-clusters <= 1`), which restores bulk discovery.
    let enrich_membership: Option<CellMembership> = if args.enrich.enabled() {
        info!("Grouping cells for mass enrichment (shared across ATOI + m6A)");
        // Reuse the gene-count matrices Step 1 persisted (no BAM re-scan). The gff
        // is only consulted on the fallback path (no persisted matrix).
        let matrix_paths: Vec<Box<str>> = gene_count_qc
            .as_ref()
            .map(|q| q.matrix_by_batch.values().cloned().collect())
            .unwrap_or_default();
        let gff_map = filtered_gff(&args.gff_file, &gene_count_qc)?;
        args.enrich.build_membership(
            &all_quant_bam_files(args),
            &gff_map,
            &matrix_paths,
            &args.cell_barcode_tag,
            &args.gene_barcode_tag,
            true,
        )?
    } else {
        None
    };

    // Step 2: ATOI Detection
    let atoi_mask = if !args.skip_atoi {
        info!("Step 2/{}: ATOI detection", n_steps);
        match run_atoi_step(args, &gene_count_qc, &snp_mask, enrich_membership.as_ref()) {
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
        info!("Step 2/{}: SKIPPED (--skip-atoi)", n_steps);
        None
    };

    // Step 3: m6A (DART) detection — WT-vs-MUT contrast at motif Cs (signal arm =
    // positional BAMs minus --control-bam, tested against the pooled control).
    // m6A discovery uses only the SNP + ATOI masks (NOT APA), so it runs BEFORE
    // the heavy APA EM — the fast modalities all finish first. Requires a
    // control; skipped (not failed) when none is supplied.
    if args.control_bam_files.is_empty() {
        info!(
            "Step 3/{}: SKIPPED (m6A needs --control-bam for the WT-vs-MUT contrast)",
            n_steps
        );
    } else {
        info!("Step 3/{}: m6A detection", n_steps);
        match run_dart_step(
            args,
            &atoi_mask,
            &snp_mask,
            &gene_count_qc,
            enrich_membership.as_ref(),
        ) {
            Ok(_) => info!("m6A complete"),
            Err(e) => log::warn!("m6A step failed: {}", e),
        }
    }

    // Step 4: APA analysis — the heavy SCAPE EM, run LAST so it never blocks the
    // fast modalities (genes / ATOI / m6A) that downstream work needs first.
    if !args.skip_apa {
        info!("Step 4/{}: APA analysis", n_steps);
        match run_apa_step(args, &atoi_mask, &snp_mask, &gene_count_qc) {
            Ok(_) => info!("APA complete"),
            Err(e) => log::warn!("APA step failed: {}", e),
        }
    } else {
        info!("Step 4/{}: SKIPPED (--skip-apa)", n_steps);
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
        // Genotype over WT + control: same genome, so pooling deepens coverage
        // and the shared mask, and each control BAM gets its own SNP output.
        bam_files: all_quant_bam_files(args),
        genome_file: args.genome_file.clone(),
        cell_barcode_tag: args.cell_barcode_tag.clone(),
        gene_barcode_tag: args.gene_barcode_tag.clone(),
        include_missing_barcode: false,
        min_base_quality: args.min_base_quality,
        min_mapping_quality: args.min_mapping_quality,
        genotype_params: GenotypeParams {
            min_depth: args.snp_min_depth,
            min_gq: args.snp_min_gq,
            min_coverage: args.snp_min_coverage,
            min_alt_count: args.snp_min_alt_count,
            min_alt_freq: args.snp_min_alt_freq,
            ..GenotypeParams::default()
        },
        backend: args.backend.clone(),
        zip: args.zip,
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
/// Parse the GFF and retain to the QC-passing genes when a resolved `GeneCountQc`
/// is present. Shared by the enrichment, ATOI, and m6A steps, which each need the
/// same expressed-gene-filtered map.
fn filtered_gff(gff_file: &str, qc: &Option<GeneCountQc>) -> anyhow::Result<GffRecordMap> {
    let mut gff_map = GffRecordMap::from(gff_file)?;
    if let Some(eg) = qc {
        gff_map.retain_by_ids(&eg.gene_ids);
        info!("Filtered to {} expressed genes", gff_map.len());
    } else {
        info!("Loaded {} genes (no expression filter)", gff_map.len());
    }
    Ok(gff_map)
}

/// Step 1 is [`run_gene_count_qc`] with the pipeline's knobs — the same call the
/// standalone modalities make, so `faba all` and `faba dartseq` cannot disagree
/// about which cells and genes survive.
fn run_gene_counting_step(args: &PipelineArgs) -> anyhow::Result<Option<GeneCountQc>> {
    // Count genes (and freeze cells) for WT + control samples alike.
    let all_bam_files = all_quant_bam_files(args);

    info!("Gene counting across {} BAM files:", all_bam_files.len());
    for bam in &all_bam_files {
        info!("  {}", bam);
    }

    let qc = crate::pipeline_util::run_gene_count_qc(
        args.gff_file.as_ref(),
        &crate::pipeline_util::GeneQcRequest {
            bam_files: &all_bam_files,
            cell_barcode_tag: &args.cell_barcode_tag,
            gene_barcode_tag: &args.gene_barcode_tag,
            umi_tag: crate::pipeline_util::resolve_umi_tag(args.no_umi_dedup, &args.umi_tag),
            gff_file: Some(args.gff_file.as_ref()),
            output_dir: &args.output,
            gene_type: &args.gene_type,
            gene_min_cells: args.gene_min_cells,
            gene_min_counts: args.gene_min_counts,
            cell_min_genes: args.cell_min_genes,
            cell_call: args.cell_qc.params(),
            mito: args.mito_qc.params(),
            valid_cells_file: None,
            valid_genes_file: None,
            skip_gene_qc: false,
            persist: Some(crate::pipeline_util::GeneMatrixSink {
                backend: &args.backend,
                zip: args.zip,
            }),
        },
    )?;
    Ok(Some(qc))
}

// Step 2: ATOI detection
fn run_atoi_step(
    args: &PipelineArgs,
    gene_count_qc: &Option<GeneCountQc>,
    snp_mask: &Option<FxHashSet<(Box<str>, i64)>>,
    membership: Option<&CellMembership>,
) -> anyhow::Result<AtoiMaskData> {
    // Load GFF and filter to expressed genes
    let gff_map = filtered_gff(args.gff_file.as_ref(), gene_count_qc)?;

    // Build ConversionParams for ATOI
    let params = ConversionParams {
        mod_type: ModificationType::AtoI,
        genome_file: args.genome_file.clone(),
        // ADAR is active in WT and YTHmut alike, so A-to-I is quantified across
        // all samples (signal-only test, no control arm).
        wt_bam_files: all_quant_bam_files(args),
        gene_barcode_tag: args.gene_barcode_tag.clone(),
        cell_barcode_tag: args.cell_barcode_tag.clone(),
        include_missing_barcode: false,
        min_coverage: args.atoi_min_coverage,
        min_conversion: args.atoi_min_conversion,
        fdr_cutoff: args.atoi_fdr_cutoff,
        error_rate: args.edit_error_rate,
        overdispersion: args.edit_overdispersion,
        backend: args.backend.clone(),
        zip: args.zip,
        output: args.output.clone(),
        cell_membership_file: None,
        membership_barcode_col: 0,
        membership_celltype_col: 1,
        exact_barcode_match: false,
        min_base_quality: args.min_base_quality,
        min_mapping_quality: args.min_mapping_quality,
        mixture_weight_mode: args.mixture_weight,
        mixture_prior_alpha: args.mixture_prior_alpha,
        mixture_prior_beta: args.mixture_prior_beta,
        umi_tag: if args.no_umi_dedup {
            None
        } else {
            Some(args.umi_tag.clone())
        },
        // A-to-I is single-sample (ADAR is active in the YTHmut too); no control.
        mut_bam_files: Vec::new(),
        site_min_cells: crate::editing::pipeline::DEFAULT_SITE_MIN_CELLS,
    };

    // Find ATOI sites (first pass): reference-anchored A→G / T→C calls, each
    // tested against the beta-binomial sequencing-error null (no control sample).
    info!("Discovering ATOI sites (reference-anchored)...");
    let atoi_sites = find_all_conversion_sites(&gff_map, &params, membership)?;

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

    // Second pass: quantification per cell across all input samples.
    info!("Quantifying ATOI sites per cell...");
    let valid_cells = gene_count_qc.as_ref().map(|qc| &qc.cells_by_batch);
    process_all_bam_files_to_backend(&params, &atoi_sites, &gff_map, valid_cells)?;

    // Mixture model (opt-in): cluster editing sites per gene. Skipped by
    // default — the gene-level {gene}/atoi/{channel} counts don't need it.
    if args.mixture {
        info!("Running 1D Gaussian mixture model on A-to-I sites...");
        let mix_params = MixtureParams {
            drop_single_component: args.drop_single_component,
            ..MixtureParams::default()
        };
        run_mixture_model(&params, &atoi_sites, &gff_map, &mix_params, valid_cells)?;
    }

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

    // APA is pure quantification (no contrast): produce it for WT + control.
    let all_bam_files = all_quant_bam_files(args);

    // Extract valid gene IDs and cell barcodes from gene count QC
    let (valid_gene_ids, valid_cell_barcodes) = match gene_count_qc {
        Some(qc) => {
            let n_cells: usize = qc.cells_by_batch.values().map(|s| s.len()).sum();
            info!(
                "APA will restrict to {} genes, {} cells",
                qc.gene_ids.len(),
                n_cells
            );
            (Some(qc.gene_ids.clone()), Some(qc.cells_by_batch.clone()))
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
        min_mapping_quality: args.min_mapping_quality,
        max_threads: args.max_threads,
        // row = poly-A-site feature QC (keep sites seen in >=10 cells) — a
        // distinct feature space from genes, so not redundant with step 1.
        row_nnz_cutoff: 10,
        // column = cells: 0 = no cell filter. The cell set was frozen in step 1
        // and every modality restricts to it; re-dropping frozen cells that
        // happen to lack APA signal would make APA's cell axis inconsistent with
        // genes/ATOI/m6A (which keep the full set).
        column_nnz_cutoff: 0,
        output: args.output.clone(),
        backend: args.backend.clone(),
        zip: args.zip,
        method: ApaMethod::Mixture, // Always use mixture mode (more robust)
        write_mixture: args.mixture,
        apa_max_sites: args.apa_max_sites,
        apa_em_pdui: args.apa_em_pdui,
        drop_single_component: args.drop_single_component,
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
        umi_tag: args.umi_tag.clone(),
        no_umi_dedup: args.no_umi_dedup,
        mu_f: 300.0,
        sigma_f: 50.0,
        theta_step: 10,
        max_beta: 70.0,
        min_beta: 10.0,
        min_ws: 0.01,
        min_fragments: 50,
        merge_distance: 50.0,
        skirt_eta: 0.05,
        skirt_mult: 3.0,
        merge_beta_mult: 2.0,
        compute_pdui: !args.no_apa_pdui,
        gene_min_cells: args.gene_min_cells,
        gene_min_counts: args.gene_min_counts,
        cell_min_genes: args.cell_min_genes,
        skip_gene_qc: true, // Pipeline already did gene QC in step 1
        valid_gene_ids,
        valid_cell_barcodes,
        cell_qc: args.cell_qc.clone(),
        mito_qc: args.mito_qc.clone(),
        valid_cells_file: None,
        valid_genes_file: None,
    };

    run_apa(&mut apa_args)?;
    Ok(())
}

// Step 4: DART analysis
fn run_dart_step(
    args: &PipelineArgs,
    atoi_mask: &Option<AtoiMaskData>,
    snp_mask: &Option<FxHashSet<(Box<str>, i64)>>,
    gene_count_qc: &Option<GeneCountQc>,
    membership: Option<&CellMembership>,
) -> anyhow::Result<()> {
    // Load GFF and filter to expressed genes
    let gff_map = filtered_gff(args.gff_file.as_ref(), gene_count_qc)?;

    // m6A is a WT-vs-MUT contrast: the signal (wt) arm is the positional BAMs
    // MINUS the control set; the control (mut) arm is --control-bam. (SNP/genes/
    // ATOI/APA quantified the full positional+control union upstream; only this
    // discovery contrast distinguishes the two arms.)
    let control_set: FxHashSet<&str> = args.control_bam_files.iter().map(|s| s.as_ref()).collect();
    // Controls are quantified as full samples in steps 0-3 (they are part of the
    // step-1 cell-calling union), so cells_by_batch carries a frozen cell set for
    // each control BAM and the per-cell {control}_m6a_* matrices reuse it (no
    // ambient superset). The WT-vs-MUT split below is only for site discovery.
    let signal_bam_files: Vec<Box<str>> = args
        .bam_files
        .iter()
        .filter(|b| !control_set.contains(b.as_ref()))
        .cloned()
        .collect();
    if signal_bam_files.is_empty() {
        anyhow::bail!("no m6A signal BAMs: every positional BAM is also in --control-bam");
    }
    info!(
        "m6A contrast: {} signal (wt) vs {} control (mut) BAMs",
        signal_bam_files.len(),
        args.control_bam_files.len()
    );

    // Build ConversionParams for m6A (DART)
    let params = ConversionParams {
        mod_type: ModificationType::M6A {
            check_r_site: true,
            contrast: args.m6a_contrast.to_contrast(),
        },
        genome_file: args.genome_file.clone(),
        wt_bam_files: signal_bam_files,
        gene_barcode_tag: args.gene_barcode_tag.clone(),
        cell_barcode_tag: args.cell_barcode_tag.clone(),
        include_missing_barcode: false,
        min_coverage: args.m6a_min_coverage,
        min_conversion: args.m6a_min_conversion,
        fdr_cutoff: args.m6a_fdr_cutoff,
        error_rate: args.edit_error_rate,
        overdispersion: args.edit_overdispersion,
        backend: args.backend.clone(),
        zip: args.zip,
        output: args.output.clone(),
        cell_membership_file: None,
        membership_barcode_col: 0,
        membership_celltype_col: 1,
        exact_barcode_match: false,
        min_base_quality: args.min_base_quality,
        min_mapping_quality: args.min_mapping_quality,
        mixture_weight_mode: args.mixture_weight,
        mixture_prior_alpha: args.mixture_prior_alpha,
        mixture_prior_beta: args.mixture_prior_beta,
        umi_tag: if args.no_umi_dedup {
            None
        } else {
            Some(args.umi_tag.clone())
        },
        mut_bam_files: args.control_bam_files.clone(),
        site_min_cells: crate::editing::pipeline::DEFAULT_SITE_MIN_CELLS,
    };

    // Find m6A sites (first pass)
    info!("Discovering m6A sites...");
    let m6a_sites = find_all_conversion_sites(&gff_map, &params, membership)?;

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

    // Apply SNP mask only if explicitly requested. Off by default: the WT-vs-MUT
    // contrast already rejects genomic variants (equal in both arms), so the SNP
    // mask is redundant here and was over-aggressive.
    if args.m6a_snp_mask {
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
    }

    // Save site annotations
    let sites_output = format!("{}/m6a_sites.parquet", args.output);
    m6a_sites.to_parquet(&gff_map, &sites_output)?;
    info!("Saved m6A sites to {}", sites_output);

    // Second pass: quantification per cell across all input samples.
    info!("Quantifying m6A sites per cell...");
    let valid_cells = gene_count_qc.as_ref().map(|qc| &qc.cells_by_batch);
    process_all_bam_files_to_backend(&params, &m6a_sites, &gff_map, valid_cells)?;

    // Mixture model (opt-in): cluster modification sites per gene. Skipped by
    // default — the gene-level {gene}/m6a/{channel} counts don't need it.
    if args.mixture {
        info!("Running 1D Gaussian mixture model on m6A sites...");
        let mix_params = MixtureParams {
            drop_single_component: args.drop_single_component,
            ..MixtureParams::default()
        };
        run_mixture_model(&params, &m6a_sites, &gff_map, &mix_params, valid_cells)?;
    }

    Ok(())
}

/// Write `{output}/pipeline_summary.json`: the faba version, the exact command line, and the
/// **effective** value of every pipeline option.
///
/// "Effective" is the whole point, and it is why this serializes [`PipelineArgs`] itself
/// rather than re-listing fields by hand. A run is defined as much by the defaults it did not
/// override as by the flags it passed — and faba's defaults have changed between builds
/// (`--cluster-resolution` used to be 0.5 and is now 0; `--n-bootstrap` existed and then did
/// not). Recording only the command line would leave a rerun unable to tell whether an output
/// was produced with grouping on or off, and `faba --version` cannot settle it either (the
/// version has gone 0.10.3 → 0.13.0 → 0.11.0 → 0.12.0, non-monotonic). The previous summary
/// recorded four input paths and no parameters at all, so it could not answer the question it
/// existed to answer.
///
/// Serializing the struct also means a new option cannot be silently *omitted* here: it
/// appears the moment it is added to `PipelineArgs`, with no second list to keep in sync.
fn write_pipeline_summary(args: &PipelineArgs) -> anyhow::Result<()> {
    let summary_path = format!("{}/pipeline_summary.json", args.output);
    let summary = serde_json::json!({
        "faba_version": env!("CARGO_PKG_VERSION"),
        "command_line": std::env::args().collect::<Vec<_>>(),
        "options": args,
    });
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("writing {summary_path}"))?;
    info!("Wrote pipeline summary (version + argv + effective options) to {summary_path}");
    Ok(())
}

#[cfg(test)]
mod tests;
