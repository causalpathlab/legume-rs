//! Entry point for `faba gem` (alias `gem-embedding`).

use anyhow::Context;
use data_beans::qc::collect_column_stat_across_vec;
use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util::stop::setup_stop_handler;
use graph_embedding_util::{load_unified_data, FeatureNameKind, LoadUnifiedArgs};
use log::info;
use matrix_util::common_io::{basename, mkdir_parent};
use matrix_util::traits::RunningStatOps;
use rayon::ThreadPoolBuilder;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use faba::gem::args::GemArgs;
use faba::gem::feature_table::FeatureTable;
use faba::gem::manifest::{write_outputs, CellQcOutputs};
use faba::gem::model::{GemModel, PARAM_INIT_STD};
use faba::gem::pseudobulk::{build_pseudobulk, RefineContext};
use faba::gem::region::{load_component_annotations, ComponentAnnotation, RegionMap};
use faba::gem::train::train;

pub fn run_gem_embedding(args: &GemArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    validate_args(args)?;

    let n_threads = if args.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        args.threads
    };
    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok(); // ignore error if pool already initialised
    info!(
        "rayon thread pool: {} threads",
        rayon::current_num_threads()
    );

    // Collect modality files in faba's conventional order: count first,
    // then m6A, A2I, pA. Order seeds modality_id assignment in the
    // FeatureTable (slot 0 is forced to "count" regardless). Each flag
    // accepts a comma-separated list of prefixes; all are stacked under
    // Union (cells merged by barcode, batches resolved from the barcodes'
    // `@batch` tags or a single --batch-files file). Modality is inferred
    // from the row name, not the flag, so multiple files per flag are fine.
    // Collect files and, in lockstep, each file's per-cell sample id (its
    // basename with the per-flag suffix stripped). The sample id tags the
    // file's barcodes (`barcode@sample`) before the Union merge so distinct
    // samples stay apart and a sample's modalities merge into one joint cell.
    let mut data_files: Vec<Box<str>> = Vec::new();
    let mut sample_ids: Vec<Box<str>> = Vec::new();
    let flags: [(&[Box<str>], &str); 4] = [
        (&args.genes, args.genes_sample_strip.as_ref()),
        (
            args.dartseq.as_deref().unwrap_or(&[]),
            args.dartseq_sample_strip.as_ref(),
        ),
        (
            args.atoi.as_deref().unwrap_or(&[]),
            args.atoi_sample_strip.as_ref(),
        ),
        (
            args.apa.as_deref().unwrap_or(&[]),
            args.apa_sample_strip.as_ref(),
        ),
    ];
    for (files, strip) in flags {
        for f in files {
            sample_ids.push(file_sample_id(f, strip)?);
            data_files.push(f.clone());
        }
    }

    let batch_files: Option<&[Box<str>]> = if args.ignore_batch {
        if args.batch_files.is_some() {
            info!("--ignore-batch: dropping batch labels; treating all cells as one batch");
        }
        None
    } else {
        args.batch_files.as_deref()
    };

    // Tag barcodes with the per-file sample id only when it disambiguates and
    // the user hasn't supplied explicit identity. With a single input file
    // there's nothing to keep apart; with `--batch-files` the caller owns
    // batch/cell identity, so leave barcodes untagged (avoids `@donor@sample`
    // double tags when barcodes already carry an `@` tag).
    let per_file_barcode_suffix: Option<Vec<Option<Box<str>>>> =
        if batch_files.is_some() || data_files.len() <= 1 {
            None
        } else {
            info!(
                "tagging barcodes with per-file sample id (e.g. {} → @{})",
                data_files[0], sample_ids[0]
            );
            Some(sample_ids.into_iter().map(Some).collect())
        };

    let feature_kind = if args.feature_name_exact {
        FeatureNameKind::Exact
    } else {
        FeatureNameKind::Gene {
            delim: args.feature_name_delim,
        }
    };
    info!(
        "loading {} modality file(s) under Union column alignment \
         (feature_kind={:?}, preload={})",
        data_files.len(),
        feature_kind,
        args.preload_data
    );
    let mut unified = load_unified_data(LoadUnifiedArgs {
        data_files,
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        feature_kind: Some(feature_kind),
        preload: args.preload_data,
        column_alignment: ColumnAlignment::Union,
        // gem inputs already carry `{gene}/{modality}/{detail}` row names, so
        // no per-file feature suffix; barcodes get the sample tag instead.
        per_file_barcode_suffix,
        ..Default::default()
    })
    .context("load_unified_data")?;

    // `--ignore-batch` semantics: under Union with multiple modality
    // files, `load_unified_data` defaults to auto_modality_batch (one
    // batch per cell-modality-presence pattern). That contradicts
    // --ignore-batch's "one batch" promise — explicitly collapse here.
    if args.ignore_batch {
        let n = unified.n_cells();
        unified.batch_membership = vec![0_u32; n];
        unified.batch_names = vec!["all".into()];
    }

    info!(
        "unified {} features × {} cells × {} batches",
        unified.n_features(),
        unified.n_cells(),
        unified.n_batches()
    );

    // ---- Cell QC: inclusive, modality-agnostic, no data rewritten ----
    // Per-cell detected-feature count over ALL modalities (count +
    // m6A/A2I/APA), via the data-beans column-stat infra. Cells below
    // `--min-cell-nnz` are near-empty — typically a barcode seen in one
    // sparse modality with a single read and no counts; the count-anchored
    // phase-2 projection maps these to ~0. We keep them through pb collapse
    // and feature training (their signal still counts) but drop them from
    // the per-cell outputs (a write-time selection, NOT a squeeze).
    let cell_nnz = collect_column_stat_across_vec(unified.count_backend(), None, None)
        .context("cell QC column statistics")?
        .count_positives();
    let keep_idx: Vec<usize> = (0..unified.n_cells())
        .filter(|&c| (cell_nnz[c] as usize) >= args.min_cell_nnz)
        .collect();
    info!(
        "cell QC: {} / {} cells pass --min-cell-nnz {} ({} dropped as near-empty)",
        keep_idx.len(),
        unified.n_cells(),
        args.min_cell_nnz,
        unified.n_cells() - keep_idx.len()
    );

    // Load component annotations (region binning). Modality labels must
    // match the modifier row names emitted by faba m6a/atoi/apa.
    let region_map = build_region_map(args)?;
    let table = FeatureTable::build(&unified.feature_names, &region_map);
    info!(
        "feature_table: {} genes, {} modalities, {} regions, {} count-comp + {} modifier-comp rows",
        table.n_genes(),
        table.n_modalities(),
        table.n_regions,
        table.count_comp_rows.len(),
        table.modifier_comp_rows.len(),
    );
    anyhow::ensure!(
        table.n_genes() > 0,
        "no genes parsed from feature axis — check row name convention (`gene/modality/detail`)"
    );

    let n_cells = unified.n_cells();
    let pb = build_pseudobulk(&mut unified, &table, args, None).context("build pseudobulk")?;

    // Persist the per-gene NB-Fisher housekeeping weights (senna
    // convention: `{out}.fisher_weights.parquet`) so the suppression is
    // inspectable. Skipped when the penalty is disabled.
    if args.housekeeping_penalty > 0.0 {
        data_beans_alg::gene_weighting::save_fisher_weights(
            &args.out,
            &pb.gene_fisher_weights,
            &table.gene_names,
        )
        .context("save fisher weights")?;
        info!(
            "wrote {}.fisher_weights.parquet ({} genes)",
            args.out,
            pb.gene_fisher_weights.len()
        );
    }

    // Persist per-gene ubiquity (fraction of cells expressing) — a
    // diagnostic / inverse-propensity signal (breadth complement to the
    // NB-Fisher magnitude weight), inspectable next to the Fisher weights.
    data_beans_alg::gene_weighting::save_per_gene_weights(
        &pb.gene_ubiquity,
        &table.gene_names,
        &format!("{}.ubiquity.parquet", args.out),
    )
    .context("save gene ubiquity")?;
    info!(
        "wrote {}.ubiquity.parquet ({} genes)",
        args.out,
        pb.gene_ubiquity.len()
    );

    let n_pbs_per_level: Vec<usize> = pb.pb_pools_per_level.iter().map(|l| l.n_units).collect();

    let dev = args
        .device
        .to_device(args.device_no)
        .context("candle device init")?;
    info!("compute device = {:?}", dev);
    let mut model = GemModel::new(
        table.n_genes(),
        table.n_modalities(),
        args.n_programs,
        table.n_regions,
        args.embedding_dim,
        n_cells,
        &n_pbs_per_level,
        &dev,
    )
    .context("init model")?;

    let stop = setup_stop_handler();
    let cell_nrms = train(args, &table, &pb, &mut model, &stop).context("training loop")?;

    // When --refine is active, pass-1 prior-score parquets go under
    // `{out}.pass1.*` so they survive the pass-2 overwrite.
    let score_prefix_p1: String = if args.refine {
        format!("{}.pass1", args.out)
    } else {
        args.out.to_string()
    };

    // Durable embeddings first, so a force-abort (second Ctrl+C) during the
    // optional topic step never loses the trained model.
    write_outputs(
        &args.out,
        &score_prefix_p1,
        &table,
        &pb,
        &model,
        &unified,
        CellQcOutputs {
            keep_idx: &keep_idx,
            cell_nrms: &cell_nrms,
        },
    )
    .context("write outputs")?;

    // Refinement pass: filter dead genes + dead cells identified from pass-1,
    // rebuild everything, retrain.  Skipped when stopped early or disabled.
    if args.refine && !stop.load(std::sync::atomic::Ordering::Relaxed) {
        // Extract β_g as a flat row-major slice [G × H] from the trained model.
        let beta_rows: Vec<f32> = model
            .beta
            .to_vec2::<f32>()
            .context("extract beta for refine")?
            .into_iter()
            .flatten()
            .collect();

        let ctx = Pass1Context {
            beta_rows,
            cell_nrms: &cell_nrms,
            cell_nnz: &cell_nnz,
            table: &table,
            region_map: &region_map,
        };
        let out2 = run_refine_pass(args, ctx, &mut unified, &stop).context("refinement pass")?;

        // Only write the final manifest and outputs if refine produced a
        // new model (returns None when nothing was filtered).
        if let Some(Pass2Outputs {
            model: model2,
            table: table2,
            pb: pb2,
            keep_idx: keep_idx2,
            cell_nrms: cell_nrms2,
        }) = out2
        {
            write_outputs(
                &args.out,
                &args.out,
                &table2,
                &pb2,
                &model2,
                &unified,
                CellQcOutputs {
                    keep_idx: &keep_idx2,
                    cell_nrms: &cell_nrms2,
                },
            )
            .context("write refined outputs")?;

            let topics2 = if args.resolve_topics {
                Some(
                    faba::gem::topics::resolve_topics(
                        &args.out, &model2, &table2, &unified, args, &stop, &keep_idx2,
                    )
                    .context("resolve topics (refined)")?,
                )
            } else {
                None
            };

            faba::gem::manifest::write_manifest(
                &args.out,
                &model2,
                keep_idx2.len(),
                topics2.as_ref(),
                !cell_nrms2.is_empty(),
            )
            .context("write manifest (refined)")?;

            info!("done (refined) — prefix '{}'", args.out);
            return Ok(());
        }
        // Nothing was filtered: fall through to write the pass-1 manifest.
        info!("refine: no genes or cells filtered — using pass-1 model");
    }

    // Single-pass or refine-no-op: write manifest for pass-1 results.
    let topics = if args.resolve_topics {
        Some(
            faba::gem::topics::resolve_topics(
                &args.out, &model, &table, &unified, args, &stop, &keep_idx,
            )
            .context("resolve topics")?,
        )
    } else {
        None
    };

    // Manifest last, so it records the resolved-topic artifacts. n_cells
    // reflects the QC-passed cells actually written to the per-cell outputs.
    faba::gem::manifest::write_manifest(
        &args.out,
        &model,
        keep_idx.len(),
        topics.as_ref(),
        !cell_nrms.is_empty(),
    )
    .context("write manifest")?;

    info!("done — prefix '{}'", args.out);
    Ok(())
}

/// Outputs produced by a successful `run_refine_pass` invocation.
struct Pass2Outputs {
    model: GemModel,
    table: FeatureTable,
    pb: faba::gem::pseudobulk::PseudobulkData,
    keep_idx: Vec<usize>,
    cell_nrms: Vec<f32>,
}

/// Pass-1 scoring inputs bundled for `run_refine_pass`.
struct Pass1Context<'a> {
    /// β_g rows from the trained model, flat row-major [G × H] in f32.
    beta_rows: Vec<f32>,
    /// Pre-L2-normalisation cell norms from phase-2 (one per original cell).
    cell_nrms: &'a [f32],
    /// Per-cell detected-feature count from `count_positives` (f32, one per cell).
    cell_nnz: &'a [f32],
    /// Pass-1 feature table — used for the gene→feature-row mapping.
    table: &'a FeatureTable,
    /// Region map (unchanged across passes).
    region_map: &'a faba::gem::region::RegionMap,
}

/// Run the pass-2 refinement: filter dead genes/cells from `unified`,
/// rebuild pseudobulk + model, retrain.  Modifies `unified` in place.
///
/// Returns `Some(Pass2Outputs)` when at least one gene or cell was filtered
/// and retraining completed, or `None` when nothing was filtered (caller uses
/// pass-1 results).
///
/// Scores are computed directly from `ctx.beta_rows` (pass-1 β_g) and
/// `ctx.cell_nrms` (pre-L2-norm from cell_solve) — no parquet round-trip.
fn run_refine_pass(
    args: &GemArgs,
    ctx: Pass1Context<'_>,
    unified: &mut graph_embedding_util::data::UnifiedData,
    stop: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<Option<Pass2Outputs>> {
    let Pass1Context {
        beta_rows,
        cell_nrms,
        cell_nnz,
        table: table_p1,
        region_map,
    } = ctx;
    let h = args.embedding_dim;
    let var = PARAM_INIT_STD * PARAM_INIT_STD;
    let dist = ChiSquared::new(h as f64).context("chi-squared dof")?;

    let n_genes = table_p1.n_genes();
    let n_cells_old = unified.n_cells();

    // ---- Score genes from β norms ----
    let live_gene_mask: Vec<bool> = (0..n_genes)
        .map(|g| {
            let row = &beta_rows[g * h..(g + 1) * h];
            let sq_norm: f64 = row.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let pval = dist.sf(sq_norm / var);
            pval <= args.feature_prior_pval_max as f64
        })
        .collect();
    let n_dead_genes = live_gene_mask.iter().filter(|&&v| !v).count();

    // ---- Score cells from pre-L2 norms ----
    // live_cell_mask covers all n_cells_old cells.  Cells that never ran
    // phase-2 (empty cell_nrms) are kept iff they pass --min-cell-nnz.
    let live_cell_mask: Vec<bool> = if cell_nrms.is_empty() {
        (0..n_cells_old)
            .map(|c| (cell_nnz[c] as usize) >= args.min_cell_nnz)
            .collect()
    } else {
        (0..n_cells_old)
            .map(|c| {
                if (cell_nnz[c] as usize) < args.min_cell_nnz {
                    return false;
                }
                let nrm = cell_nrms[c] as f64;
                let pval = dist.sf((nrm * nrm) / var);
                pval <= args.cell_prior_pval_max as f64
            })
            .collect()
    };
    let n_dead_cells = live_cell_mask.iter().filter(|&&v| !v).count();

    info!(
        "refine: {} / {} genes dead (feature_prior_pval > {})",
        n_dead_genes, n_genes, args.feature_prior_pval_max
    );
    info!(
        "refine: {} / {} cells dead (cell_prior_pval > {})",
        n_dead_cells, n_cells_old, args.cell_prior_pval_max
    );

    if n_dead_genes == 0 && n_dead_cells == 0 {
        return Ok(None);
    }

    // ---- Filter unified ----
    let live_feature_rows: Vec<usize> = (0..unified.n_features())
        .filter(|&r| {
            table_p1.row_gene[r]
                .map(|g| live_gene_mask[g as usize])
                .unwrap_or(false)
        })
        .collect();
    let live_cell_old_ids: Vec<usize> = (0..n_cells_old).filter(|&c| live_cell_mask[c]).collect();

    info!(
        "refine: keeping {} / {} feature rows, {} / {} cells",
        live_feature_rows.len(),
        unified.n_features(),
        live_cell_old_ids.len(),
        n_cells_old,
    );

    // Capture the full N_old per-cell batch labels *before* subsetting: the
    // refine pass's projection + collapse run on the still-full backend (N_old
    // columns), but `subset_cells` shrinks `unified.batch_membership` to N_new.
    let full_batch_labels: Vec<Box<str>> = unified.batch_labels();

    unified.subset_features(&live_feature_rows);
    unified.subset_cells(&live_cell_old_ids);

    // ---- Rebuild ----
    let table2 = FeatureTable::build(&unified.feature_names, region_map);
    anyhow::ensure!(
        table2.n_genes() > 0,
        "refine: no genes remain after dead-gene filter"
    );

    let pb2 = build_pseudobulk(
        unified,
        &table2,
        args,
        Some(RefineContext {
            live_cell_old_ids: &live_cell_old_ids,
            full_batch_labels: &full_batch_labels,
        }),
    )
    .context("build pseudobulk (refined)")?;

    // Diagnostic weights for pass-2.
    if args.housekeeping_penalty > 0.0 {
        data_beans_alg::gene_weighting::save_fisher_weights(
            args.out.as_ref(),
            &pb2.gene_fisher_weights,
            &table2.gene_names,
        )
        .context("save fisher weights (refined)")?;
    }
    data_beans_alg::gene_weighting::save_per_gene_weights(
        &pb2.gene_ubiquity,
        &table2.gene_names,
        &format!("{}.ubiquity.parquet", args.out),
    )
    .context("save gene ubiquity (refined)")?;

    let n_cells2 = unified.n_cells();
    let n_pbs2: Vec<usize> = pb2.pb_pools_per_level.iter().map(|l| l.n_units).collect();
    let dev = args
        .device
        .to_device(args.device_no)
        .context("candle device init (refined)")?;
    let mut model2 = GemModel::new(
        table2.n_genes(),
        table2.n_modalities(),
        args.n_programs,
        table2.n_regions,
        args.embedding_dim,
        n_cells2,
        &n_pbs2,
        &dev,
    )
    .context("init model (refined)")?;

    let cell_nrms2 =
        train(args, &table2, &pb2, &mut model2, stop).context("training loop (refined)")?;

    // All cells in `unified` already passed the combined QC (subset_cells
    // applied both min_cell_nnz and the embedding threshold), so keep_idx2
    // is the identity partition.
    let keep_idx2: Vec<usize> = (0..n_cells2).collect();

    Ok(Some(Pass2Outputs {
        model: model2,
        table: table2,
        pb: pb2,
        keep_idx: keep_idx2,
        cell_nrms: cell_nrms2,
    }))
}

/// Per-file sample id: the file's basename (sparse-data extension stripped)
/// with the per-flag `strip` suffix removed. `rep1_wt_genes.zarr.zip` with
/// `strip = "_genes"` → `rep1_wt`. Empty `strip` (or a non-matching one)
/// keeps the full basename, so two modality files of one sample merge only
/// when their stripped basenames agree.
fn file_sample_id(file: &str, strip: &str) -> anyhow::Result<Box<str>> {
    let base = basename(file)?;
    let sid = if strip.is_empty() {
        base.as_ref()
    } else {
        base.as_ref().strip_suffix(strip).unwrap_or(base.as_ref())
    };
    Ok(sid.into())
}

/// Load any supplied `*_components.parquet` sidecars and build the
/// transcript-position `RegionMap`. Each sidecar is tagged with the
/// modality label that matches its modifier row names (`m6A`, `A2I`,
/// `pA`). With no sidecars the map is empty and every satellite falls
/// back to region 0 (γ collapses to one per-modality offset).
fn build_region_map(args: &GemArgs) -> anyhow::Result<RegionMap> {
    let sidecars: [(&Option<Box<str>>, &str); 3] = [
        (&args.dartseq_components, "m6A"),
        (&args.atoi_components, "A2I"),
        (&args.apa_components, "pA"),
    ];
    let mut records: Vec<ComponentAnnotation> = Vec::new();
    for (path, modality) in sidecars {
        if let Some(path) = path.as_ref() {
            let recs = load_component_annotations(path, modality)
                .with_context(|| format!("loading {modality} component annotations from {path}"))?;
            info!(
                "region: {} {} component annotations from {}",
                recs.len(),
                modality,
                path
            );
            records.extend(recs);
        }
    }
    Ok(RegionMap::from_records(&records, args.n_regions))
}

/// Argument-level sanity checks before any I/O or training. Surfaces
/// configuration mistakes (zero-dim model, NaN/out-of-range tempering,
/// stratum-fraction overflow) as a clear `anyhow::Error` rather than a
/// candle panic or silently zero-loss training.
fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.embedding_dim
    );
    anyhow::ensure!(
        args.n_programs > 0,
        "--num-programs must be > 0 (got {})",
        args.n_programs
    );
    anyhow::ensure!(
        args.n_regions > 0,
        "--num-regions must be > 0 (got {})",
        args.n_regions
    );
    anyhow::ensure!(
        args.tau.is_finite() && (0.0..=1.0).contains(&args.tau),
        "--tau must be a finite value in [0, 1] (got {})",
        args.tau
    );
    anyhow::ensure!(
        args.tau_modality.is_finite() && (0.0..=1.0).contains(&args.tau_modality),
        "--tau-modality must be a finite value in [0, 1] (got {})",
        args.tau_modality
    );
    anyhow::ensure!(
        args.housekeeping_penalty.is_finite() && args.housekeeping_penalty >= 0.0,
        "--housekeeping-penalty must be a finite value ≥ 0 (got {})",
        args.housekeeping_penalty
    );
    if args.resolve_topics {
        // --no-cell-axis leaves e_cell at its random init (never trained),
        // so resolving topics from it would yield archetypes of noise.
        anyhow::ensure!(
            !args.no_cell_axis,
            "--resolve-topics requires a trained cell embedding, but --no-cell-axis \
             leaves e_cell at its random init; drop one of the two flags"
        );
        if let Some(k) = args.num_topics {
            anyhow::ensure!(k >= 2, "--num-topics must be ≥ 2 (got {})", k);
        }
        // Else: K is auto-swept over 2..=H+1, which is always ≥ 2.
    }
    anyhow::ensure!(
        args.f_agg.is_finite() && (0.0..=1.0).contains(&args.f_agg),
        "--f-agg must be in [0, 1] (got {})",
        args.f_agg
    );
    anyhow::ensure!(
        args.f_count.is_finite() && (0.0..=1.0).contains(&args.f_count),
        "--f-count must be in [0, 1] (got {})",
        args.f_count
    );
    anyhow::ensure!(
        args.f_agg + args.f_count <= 1.0,
        "--f-agg + --f-count must be ≤ 1.0 (got {} + {} = {}); modifier stratum gets the remainder",
        args.f_agg,
        args.f_count,
        args.f_agg + args.f_count
    );
    // At least one negative-source must be active or training is a no-op:
    // log_softmax over a single positive column is identically zero.
    anyhow::ensure!(
        args.n_rand + args.n_swap_gene_mode > 0,
        "at least one of --n-rand, --n-swap-gene-mode must be > 0 \
         (else NCE loss collapses to zero and AdamW makes no progress)"
    );
    if args.refine {
        anyhow::ensure!(
            args.feature_prior_pval_max > 0.0 && args.feature_prior_pval_max < 1.0,
            "--feature-prior-pval-max must be in (0, 1) (got {})",
            args.feature_prior_pval_max
        );
        anyhow::ensure!(
            args.cell_prior_pval_max > 0.0 && args.cell_prior_pval_max < 1.0,
            "--cell-prior-pval-max must be in (0, 1) (got {})",
            args.cell_prior_pval_max
        );
    }
    Ok(())
}
