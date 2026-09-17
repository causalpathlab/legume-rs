//! Public entry point for `graph-embedding`. Callers translate their
//! own CLI args into a [`FitConfig`] and pass already-loaded
//! [`UnifiedData`] (so this crate stays free of file/path concerns).

mod axes;
pub mod batch_fold;
mod config;
pub mod hier;
mod models;
pub mod module_args;
pub mod module_warm;
pub mod pb_readout;
pub mod projection;
pub mod resolve_embedding;
mod samplers;
mod setup;

pub use batch_fold::BatchGeneFold;
pub use config::{
    validate_offset_rank, FeatureModuleConfig, FitConfig, FitOutput, ParentModulesOwned, TrackInfo,
    TrackSpec,
};
pub use module_args::FeatureModuleArgs;
pub use module_warm::{parent_module_logits, warm_start_module_labels};
pub use pb_readout::{majority_batch_per_pb, PbLevelEmbedding};
pub use projection::{CellEncoder, CellEncoders, TrackEncoder};
pub use resolve_embedding::{train_rest, RestConfig, RestTrainInputs, TrainedRest};

use crate::data::{Triplet, UnifiedData};
use anyhow::Context;
use candle_util::candle_core::Tensor;
use candle_util::candle_nn::VarMap;
use log::info;
use matrix_param::traits::Inference;
use nalgebra::DMatrix;

use matrix_util::traits::ConvertMatOps;
use projection::{project_cells_phase2, CellBatchFold, DistillLevel, DistillSpec, PHASE2_RIDGE};
pub use projection::{
    FrozenProjection, FrozenProjectionArgs, FrozenProjector, PHASE2_RIDGE as PROJECTION_RIDGE_SGD,
};

/// Two-phase fit, shared by `senna bge` and `senna gem` through the same
/// driver: multilevel-pseudobulk phase 1 over the feature axis (one track,
/// or several when the caller names tracks), then per-cell phase 2 against
/// the frozen dictionary.
///
/// The bilinear score is `E_feat[f]·E_cell[c] + b_feat[f] + b_cell[c]`; the
/// per-cell bias `b_cell` absorbs library size.
///
/// **Phase 1 — features + pseudobulks.** Train only the pseudobulk axes
/// (coarsest..finest from `collapse_columns_multilevel_vec`, pseudobulk-
/// feature triplets) with `Sum`. They share — and learn — `E_feat /
/// b_feat` and per-level pb cell-side embeddings.
///
/// **Phase 2 — dense per-cell embedding.** Freeze the entire feature side
/// and fit ONLY `E_cell` against it. With a single axis the objective is
/// separable per cell — each row's gradient depends only on that cell's
/// own edges (embarrassingly parallel) — and the auto per-epoch budget
/// (sized by `n_units` = `n_cells`) sweeps every cell ~once per epoch.
///
/// This replaces the old single joint pass, in which the per-cell axis was
/// starved: the per-epoch budget was sized by the pseudobulk count, so
/// `E_cell` received ~1 step/epoch across all cells and never left random
/// init (all useful training happened at the pb level).
pub fn fit(unified: &mut UnifiedData, config: FitConfig) -> anyhow::Result<FitOutput> {
    let n_cells = unified.n_cells();
    let h = config.embedding_dim;
    let stop = crate::stop::stop_flag();

    ///////////////////////////////////////////////
    // Shared upstream: projection → pseudobulks //
    ///////////////////////////////////////////////
    let n_features = unified.n_features();
    let feature_to_backend = unified.feature_to_backend_row.clone();
    // Row structure of the feature axis: plain genes (every row its own gene)
    // unless the caller named tracks. Validated HERE, before anything reads it:
    // the projection below sketches on the base track's rows, so an unchecked
    // spec would reach the collapse before the error did.
    let tracks = config
        .tracks
        .clone()
        .unwrap_or_else(|| TrackSpec::base(n_features));
    tracks
        .validate(n_features)
        .context("the fit's track spec does not describe this feature axis")?;
    let n_tracks = tracks.n_tracks();
    let pb = setup::build_pseudobulks(unified, &config, &tracks)?;
    let setup::Pseudobulks {
        collapsed_levels,
        cell_to_pb_per_level,
        blobs: pb_blobs,
    } = pb;
    // Per-batch gene fold for phase 2, from the finest collapse's `δ`. The count
    // backend numbers batches by sorted name; the unified data by first appearance
    // — matched by name inside.
    let batch_gene_fold: Option<BatchGeneFold> =
        match collapsed_levels.last().and_then(|c| c.delta.as_ref()) {
            Some(delta) => {
                let collapse_batch_names =
                    unified.count_backend().batch_names().ok_or_else(|| {
                        anyhow::anyhow!("collapse fit a δ but the backend has no batch names")
                    })?;
                batch_fold::batch_gene_fold(batch_fold::FoldSource {
                    delta: delta.posterior_mean(),
                    collapse_batch_names: &collapse_batch_names,
                    unified_batch_names: &unified.batch_names,
                    n_features,
                    feature_to_backend: &feature_to_backend,
                })?
            }
            None => None,
        };
    // Levels run coarsest..finest here, so the finest is `.last()`. Cloned
    // rather than moved: the level list feeds training below. The clone is
    // cheap relative to the fit and only happens when a reference is emitted.
    let finest_collapse = config.emit_finest_collapse.then(|| {
        (
            collapsed_levels.last().expect("at least one level").clone(),
            cell_to_pb_per_level
                .last()
                .expect("membership per level")
                .clone(),
        )
    });

    ////////////////////////////////
    // VarMap and embedding heads //
    ////////////////////////////////
    let varmap = VarMap::new();
    // Phase 1 trains by the exact hierarchical softmax (see `hier`), which owns
    // its own module warm start and composes the dictionary itself.
    // The finest collapse's feature profile (batch-corrected pseudobulk rates,
    // gathered onto the unified feature axis) — seeds the hier engine's module
    // partition.
    let finest_profile = || -> DMatrix<f32> {
        let finest = collapsed_levels.last().expect("at least one level");
        let pb_full = match &finest.mu_adjusted {
            Some(adj) => adj.posterior_mean(),
            None => finest.mu_observed.posterior_mean(),
        };
        setup::gather_to_unified_axis(pb_full, n_features, &feature_to_backend)
    };
    let models::Heads {
        mut cell_model,
        mut level_models,
    } = models::build_heads(unified, &pb_blobs, &config, &varmap)?;

    /////////////////
    // Axis data   //
    /////////////////
    let ax = axes::build_axis_data(unified, &cell_to_pb_per_level, &config)?;
    let (use_cell_axis, cell_samplers) = (ax.use_cell_axis, &ax.cell_samplers);

    /////////////////////////////
    // Phase 1: joint training //
    /////////////////////////////

    // Units = every level's pseudobulks + the phase-1 cell subsample, the cells
    // batch-folded like phase 2 folds them.
    let cell_fold = batch_gene_fold
        .as_ref()
        .map(|fold| fold.cell_fold(&unified.batch_membership));
    let blobs: Vec<&[Triplet]> = pb_blobs.iter().map(|b| b.triplets.as_slice()).collect();
    let n_pb_per_level: Vec<usize> = pb_blobs.iter().map(|b| b.n_cells()).collect();
    // `--phase1-cells-per-pb 0` ⇒ `use_cell_axis == false` (pure-pb phase 1,
    // logged above as "cell axis SUPPRESSED"); `phase1_cell_samplers()` would
    // otherwise fall back to the FULL cell samplers, silently making every
    // cell a unit.
    let cell_rows: Vec<(u32, &[u32], &[f32])> = if use_cell_axis {
        ax.phase1_cell_samplers()
            .iter()
            .flat_map(|s| {
                s.active_cells
                    .iter()
                    .zip(&s.per_cell)
                    .map(|(&c, cf)| (c, cf.features.as_slice(), cf.counts.as_slice()))
            })
            .collect()
    } else {
        Vec::new()
    };
    let units = hier::UnitTable::from_pseudobulks_and_cells_tracked(
        &blobs,
        &n_pb_per_level,
        &cell_rows,
        cell_fold,
        n_features,
        tracks.clone(),
    );
    // Module labels: under `senna update`, seeded from the parent's membership
    // (the argmax of `parent_module_logits`, i.e. the partition `senna update`
    // claims to carry — matched features take the parent's module, unmatched
    // ones are initialized through the parent's modules); otherwise the
    // k-means warm start over the finest level's profiles.
    let profile = finest_profile();
    let (labels, n_modules) = match config
        .feature_modules
        .as_ref()
        .and_then(|g| g.parent.as_ref())
    {
        Some(parent) => {
            anyhow::ensure!(
                tracks.is_base(),
                "module warm start from a parent needs a single-track feature axis"
            );
            anyhow::ensure!(
                parent.mu.ncols() == h,
                "parent modules are {}-dimensional but this fit uses H={h}",
                parent.mu.ncols()
            );
            let logits = module_warm::parent_module_logits(parent, &profile);
            info!(
                "Phase 1 (hier) — module partition seeded from the parent's membership ({} \
                 modules)",
                parent.mu.nrows()
            );
            (
                hier::partition::labels_from_membership(&logits),
                parent.mu.nrows(),
            )
        }
        None => {
            let n = config
                .feature_modules
                .as_ref()
                .context("the hierarchical phase 1 needs a module count (feature_modules)")?
                .n_modules;
            (
                // The partition is over GENES, so the warm start reads the
                // base track's rows re-keyed by gene; identity on one track.
                module_warm::warm_start_module_labels(
                    &module_warm::base_track_profile(&profile, &tracks),
                    n,
                    config.seed,
                ),
                n,
            )
        }
    };
    let out = hier::train(
        &units,
        &labels,
        h,
        &hier::HierConfig {
            n_modules,
            epochs: config.epochs,
            units_per_step: config.hier_units_per_step,
            modules_per_unit: config.hier_modules_per_unit,
            lr: config.learning_rate as f32,
            weight_decay: config.weight_decay as f32,
            seed: config.seed,
            offset_l2: config.offset_l2,
            offset_rank: config.offset_rank,
            device: config.device.clone(),
        },
        config.preset_features.as_ref(),
        &config.preset_offsets,
        &stop,
    )?;
    // The composed dictionary into the shared feature Vars; each level's
    // pseudobulk rows into that level head's cell table.
    let rho_row_major: Vec<f32> = out.rho.transpose().as_slice().to_vec();
    let rho_t = Tensor::from_slice(&rho_row_major, (n_features, h), &config.device)?;
    let b_feat_t = Tensor::from_slice(out.b_feat.as_slice(), n_features, &config.device)?;
    {
        let vars = varmap.data().lock().unwrap();
        vars.get(crate::model::E_FEAT_VAR_NAME)
            .context("e_feat var missing")?
            .set(&rho_t)?;
        vars.get("b_feat")
            .context("b_feat var missing")?
            .set(&b_feat_t)?;
    }
    cell_model.e_feat = rho_t;
    cell_model.b_feat = b_feat_t;
    // Scatter the pseudobulk rows onto their level heads in one pass over
    // the units (levels first, then cells, per `UnitTable`).
    let mut level_rows: Vec<Vec<f32>> = level_models
        .iter()
        .map(|lm| lm.e_cell.dim(0).map(|n_pb| vec![0f32; n_pb * h]))
        .collect::<Result<_, _>>()?;
    for u in 0..units.n_units() {
        let l = units.level[u] as usize;
        if l < level_rows.len() {
            let p = units.source_index[u] as usize;
            let dst = &mut level_rows[l][p * h..(p + 1) * h];
            for (k, x) in out.e_u.row(u).iter().enumerate() {
                dst[k] = *x;
            }
        }
    }
    for (l, (lm, rows)) in level_models.iter_mut().zip(level_rows).enumerate() {
        let n_pb = lm.e_cell.dim(0)?;
        let e_cell_t = Tensor::from_vec(rows, (n_pb, h), &config.device)?;
        {
            let vars = varmap.data().lock().unwrap();
            vars.get(&format!("pb_l{l}_e_cell"))
                .context("pb level e_cell var missing")?
                .set(&e_cell_t)?;
        }
        lm.e_cell = e_cell_t;
    }
    info!(
        "Phase 1 (hier) — done: loss/unit {:.4}; dictionary {} × {h} composed from {n_modules} \
         modules over {n_tracks} track(s)",
        out.final_loss_per_unit, n_features
    );

    // The trained pseudobulk tables, read before phase 2 takes `&mut cell_model`:
    // one `[n_pb × H]` per level with each pseudobulk's batch.
    let pb_embeddings: Vec<pb_readout::PbLevelEmbedding> = level_models
        .iter()
        .zip(&cell_to_pb_per_level)
        .map(
            |(m, c2pb)| -> anyhow::Result<pb_readout::PbLevelEmbedding> {
                let e_pb = DMatrix::<f32>::from_tensor(&m.e_cell)?;
                let n_pb = e_pb.nrows();
                Ok(pb_readout::PbLevelEmbedding {
                    e_pb,
                    batch: pb_readout::majority_batch_per_pb(c2pb, &unified.batch_membership, n_pb),
                })
            },
        )
        .collect::<anyhow::Result<_>>()?;

    // Snapshot β+δ into the `e_feat` field so phase 2 and every
    // output/co-embed reader see a fixed materialized dictionary. No-op for a free,
    // ungated model. This MUST come after the training above: `e_feat` is what phase 2
    // projects against and what the dictionary output writes, so materializing before
    // the fit would leave the whole pass invisible downstream — cells projected onto a
    // dictionary that does not match the parameters just fitted, which looks exactly
    // like a model that trained badly rather than one whose output was never refreshed.
    cell_model.materialize_e_feat()?;

    // The first Ctrl+C stops the *major SGD loop* — phase 1 above — and nothing
    // else. Every stage from here down is a follow-up routine that turns the
    // trained dictionary into the run's deliverables, so gating it on `stop`
    // does not save the user time, it destroys the output. Phase 2 especially:
    // below `--phase1-cells-per-pb n_cells` most cells never train in phase 1,
    // so their `e_cell` rows are still randn init until phase 2 runs — skipping
    // it wrote a `cell_embedding.parquet` of pure noise, silently. A second
    // Ctrl+C aborts the process outright (`matrix_util::stop`), which is the
    // escape hatch for a user who really does want out now.
    if stop.load(std::sync::atomic::Ordering::Relaxed) {
        log::warn!(
            "phase 1 was interrupted — the feature dictionary is short-trained. The follow-up \
             stage (phase-2 projection) still runs, so the outputs are complete but fit against \
             a partially trained dictionary; treat this run as a draft. Ctrl+C again to abort \
             outright."
        );
    }

    // Phase 2: per-cell projection onto the fixed feature side. With
    // E_feat/b_feat held fixed each cell's embedding is independent, so this is
    // a cell-block Poisson SGD over `e_cell`/`b_cell` alone — see
    // `projection::project_cells_phase2`. The per-cell intercept `b_cell` is fitted
    // and kept.
    let phase2 = {
        // Phase-2 batch correction: divide each cell's counts by its batch's
        // per-gene fold `δ` from the finest collapse (see `batch_fold`), so the
        // solve runs in the batch-free frame the dictionary was trained in. `None`
        // on single-batch data.
        let batch_fold: Option<CellBatchFold> = batch_gene_fold
            .as_ref()
            .map(|f| f.cell_fold(&unified.batch_membership));

        // The phase-1 pseudobulk tables are the distillation targets of the
        // encoder that replaces the per-cell solve.
        anyhow::ensure!(
            pb_embeddings.len() == cell_to_pb_per_level.len(),
            "phase 2: {} pseudobulk tables for {} membership levels",
            pb_embeddings.len(),
            cell_to_pb_per_level.len()
        );
        let distill_levels: Vec<DistillLevel> = pb_embeddings
            .iter()
            .zip(&cell_to_pb_per_level)
            .map(|(pb, c2pb)| DistillLevel {
                e_pb: &pb.e_pb,
                cell_to_pb: c2pb,
            })
            .collect();
        let spec = DistillSpec {
            levels: &distill_levels,
            seed: config.seed,
        };
        project_cells_phase2(
            &mut cell_model,
            &varmap,
            cell_samplers,
            n_cells,
            f64::from(PHASE2_RIDGE),
            &config.device,
            batch_fold,
            Some(&spec),
            &tracks,
        )?
    };

    // The pseudobulk tables leave in the CELLS' frame: phase 2 moved the cells
    // by −θ̄ (folded into `b_feat`), and a reader putting the two tables in one
    // layout — the anchors over the cells they placed — needs them shifted
    // alike. The in-memory tables served phase 2 (distillation targets) in the
    // as-trained frame before this point.
    let mut pb_embeddings = pb_embeddings;
    for level in &mut pb_embeddings {
        for mut row in level.e_pb.row_iter_mut() {
            for (k, x) in row.iter_mut().enumerate() {
                *x -= phase2.theta_mean[k];
            }
        }
    }

    Ok(FitOutput {
        batch_gene_fold,
        model: cell_model,
        finest_collapse,
        varmap,
        cell_nrms: phase2.cell_nrms,
        pb_embeddings,
        cell_encoder: phase2.cell_encoder,
        // One fitted intercept per NON-base track; empty on a one-track axis.
        track_intercepts: phase2.other_intercepts,
    })
}
