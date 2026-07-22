//! The per-modality steps `faba all` chains, in the order it runs them.
//!
//! Each one builds the standalone subcommand's args from [`super::args::PipelineArgs`]
//! and calls the same entry point the user would — never a private copy of the work.

use crate::data::cell_membership::CellMembership;
use crate::editing::io::ToParquet;
use crate::editing::mask::{build_atoi_mask, filter_conversion_sites_by_mask, filter_m6a_by_mask};
use crate::editing::mixture::MixtureParams;
use crate::editing::mixture_pipeline::run_mixture_model;
use crate::editing::pipeline::{
    find_all_conversion_sites, process_all_bam_files_to_backend, ConversionParams,
};
use crate::editing::sifter::ModificationType;
use crate::quant::GeneCountQc;
use crate::snp::genotyper::GenotypeParams;
use crate::snp::io::load_known_snps_auto;
use crate::snp::pipeline::{run_snp_pipeline, SnpParams};

use super::args::*;
use super::run::*;
use genomic_data::gff::GffRecordMap;
use log::info;
use rustc_hash::FxHashSet;

pub(super) struct AtoiMaskData {
    pub(super) mask: rustc_hash::FxHashSet<(Box<str>, i64)>,
    pub(super) n_sites: usize,
}

// Step 0: SNP genotyping (discovery + optional known sites).
// VAF filter ensures de novo variants only enter the mask if they have
// germline-like allele fractions, preserving RNA editing sites.
pub(super) fn run_snp_step(args: &PipelineArgs) -> anyhow::Result<FxHashSet<(Box<str>, i64)>> {
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
pub(super) fn filtered_gff(
    gff_file: &str,
    qc: &Option<GeneCountQc>,
) -> anyhow::Result<GffRecordMap> {
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
pub(super) fn run_gene_counting_step(args: &PipelineArgs) -> anyhow::Result<Option<GeneCountQc>> {
    // Count genes (and freeze cells) for WT + control samples alike.
    let all_bam_files = all_quant_bam_files(args);

    info!("Gene counting across {} BAM files:", all_bam_files.len());
    for bam in &all_bam_files {
        info!("  {}", bam);
    }

    let qc = crate::quant::run_gene_count_qc(
        args.gff_file.as_ref(),
        &crate::quant::GeneQcRequest {
            bam_files: &all_bam_files,
            cell_barcode_tag: &args.cell_barcode_tag,
            gene_barcode_tag: &args.gene_barcode_tag,
            umi_tag: crate::quant::resolve_umi_tag(args.no_umi_dedup, &args.umi_tag),
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
            persist: Some(crate::quant::GeneMatrixSink {
                backend: &args.backend,
                zip: args.zip,
            }),
        },
    )?;
    Ok(Some(qc))
}

// Step 2: ATOI detection
pub(super) fn run_atoi_step(
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
pub(super) fn run_apa_step(
    args: &PipelineArgs,
    atoi_mask: &Option<AtoiMaskData>,
    snp_mask: &Option<FxHashSet<(Box<str>, i64)>>,
    gene_count_qc: &Option<GeneCountQc>,
) -> anyhow::Result<()> {
    use crate::apa::run::{run_apa, ApaMethod, CountApaArgs};

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
pub(super) fn run_dart_step(
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
