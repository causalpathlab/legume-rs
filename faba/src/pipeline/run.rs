use super::args::*;
use super::steps::*;
use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::quant::check_all_bam_indices;

use anyhow::Context;
use log::info;
use rayon::ThreadPoolBuilder;

/// The full set of samples to QUANTIFY in every modality: the positional
/// (signal/WT) BAMs together with the `--control-bam` (MUT/YTHmut) BAMs,
/// deduplicated (a BAM may legitimately be listed in both roles). The WT-vs-MUT
/// split is used ONLY for the m6A discovery contrast (step 4); SNP, gene counts,
/// ATOI, APA and m6A per-cell matrices are all produced for EVERY one of these
/// samples, so the control background is fully quantified — not merely consumed
/// as an m6A reference. This also freezes a cell set for each control BAM in
/// step 1, so the control m6A matrices reuse it instead of the ambient superset.
pub(super) fn all_quant_bam_files(args: &PipelineArgs) -> Vec<Box<str>> {
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
