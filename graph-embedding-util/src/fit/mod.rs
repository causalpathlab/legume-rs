//! Public entry point for `graph-embedding`. Callers translate their
//! own CLI args into a [`FitConfig`] and pass already-loaded
//! [`UnifiedData`] (so this crate stays free of file/path concerns).

mod config;
mod projection;
mod samplers;

pub use config::{
    load_feature_network, FeatureNetworkArgs, FeatureNetworkConfig, FitConfig, FitOutput,
};

use crate::coarsen::{identity_axis, AxisCoarsenings};
use crate::data::UnifiedData;
use crate::loss::{build_stratified_sampler, PerBatchStratifiedCellSampler, StratifiedSampler};
use crate::model::{JointEmbedModel, ModelArgs, ModelInit, ShareFeaturesArgs};
use crate::stop::setup_stop_handler;
use crate::training::{
    train_composite, AxisSampler, CompositeAxis, CompositeMode, CompositeTrainContext,
};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use log::info;
use matrix_param::traits::Inference;
use nalgebra::DMatrix;

use config::{
    load_or_compute_fisher_weights, stage_params, DEFAULT_AXIS_LAMBDA, DEFAULT_STRATIFY_ALPHA_CELL,
    DEFAULT_STRATIFY_ALPHA_PB,
};
use projection::{project_cells_phase2, PHASE2_RIDGE};
use samplers::{build_active_samplers, build_smoother, subsample_cell_samplers_multilevel};

/// Composite-objective gbe fit — trained in **two phases**.
///
/// The bilinear score is `E_feat[f]·E_cell[c] + b_feat[f] + b_cell[c]` —
/// the per-cell bias `b_cell` absorbs library size (consistent with
/// `faba gem`).
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
pub fn fit(unified: &mut UnifiedData, mut config: FitConfig) -> anyhow::Result<FitOutput> {
    let n_cells = unified.n_cells();
    let h = config.embedding_dim;
    let alpha_neg = 0.75_f32;
    let stop = config.stop.take().unwrap_or_else(setup_stop_handler);

    // ---- Shared upstream: batch-corrected projection + Fisher weights ----
    info!(
        "Batch-corrected projection (proj_dim={}, {} batches)...",
        config.proj_dim,
        unified.n_batches()
    );
    let batch_labels: Vec<Box<str>> = unified.batch_labels();
    let batch_arg = (unified.n_batches() > 1).then_some(batch_labels.as_slice());
    let proj_out = if let Some(w) = config.hvg_weights.as_deref() {
        anyhow::ensure!(
            w.len() == unified.n_features(),
            "hvg_weights length {} != n_features {} (HVG mask must be aligned to the unified \
             feature axis BEFORE any subset/coarsening — pass full-axis weights from the wrapper)",
            w.len(),
            unified.n_features()
        );
        info!(
            "HVG-weighted projection: {} weighted features (>= 1.0)",
            w.iter().filter(|&&x| x > 0.0).count()
        );
        // The projection runs on the full backend row axis, which may be
        // wider than the compact feature axis when a prior pass dropped
        // features (e.g. the two-pass null-QC refine in `senna bge`). Scatter
        // the compact weights to backend rows via `feature_to_backend_row`;
        // rows not in the current feature axis get 0 so they sit out the
        // projection basis. Identity (and a no-op) when no subset has happened.
        let backend_rows = unified.count_backend().num_rows();
        let mut backend_w = vec![0.0f32; backend_rows];
        for (compact_i, &brow) in unified.feature_to_backend_row.iter().enumerate() {
            backend_w[brow] = w[compact_i];
        }
        unified.count_backend_mut().project_columns_weighted(
            config.proj_dim,
            config.block_size,
            batch_arg,
            &backend_w,
        )?
    } else {
        unified
            .count_backend_mut()
            .project_columns_with_batch_correction(config.proj_dim, config.block_size, batch_arg)?
    };

    let feat_weights = load_or_compute_fisher_weights(
        unified,
        config.block_size,
        config.fisher_weights_cache.as_deref(),
    )?;

    // ---- Multilevel collapse → batch-corrected pseudobulks ----
    //
    // `sort_dim` controls how many bits of the binary-sketched projection
    // are used to hash cells into the *finest* pb-sample partition (so
    // `2^sort_dim` is the max number of distinct codes / pb-samples at
    // that level). Exposed directly via `FitConfig.sort_dim` for parity
    // with `senna topic` / `svd` rather than derived from a target count.
    info!(
        "Multilevel collapse (sort_dim={}, {} levels requested)...",
        config.sort_dim, config.num_levels
    );
    let collapse_out = collapse_columns_multilevel_with_hierarchy(
        unified.count_backend_mut(),
        &proj_out.proj,
        &batch_labels,
        &MultilevelParams {
            knn_pb_samples: config.knn_pb_samples,
            num_levels: config.num_levels.max(1),
            sort_dim: config.sort_dim,
            num_opt_iter: config.num_opt_iter,
            refine: config.refine.clone(),
            // bge only reads `posterior_mean()` off the collapse output
            // (see the pb_full loop below), so skip the sd / log_mean /
            // log_sd planes entirely — that's the bulk of the coarsen-stage
            // memory at high pb-sample counts.
            output_calibration: matrix_param::traits::CalibrateTarget::MeanOnly,
        },
    )?;
    let mut collapsed_levels = collapse_out.levels;
    // Per-level cell→pb (finest-first, matching `collapsed_levels`
    // pre-reverse). Surfaced for the future nested chain sampler;
    // currently informational only — use it to derive the pb-tree
    // parent map between adjacent levels via `derive_parent_map`.
    let mut cell_to_pb_per_level = collapse_out.cell_to_pb_per_level;
    // After this reverse, levels are ordered coarsest..finest. Senna
    // topic uses the same order so the curriculum trains coarse first.
    collapsed_levels.reverse();
    cell_to_pb_per_level.reverse();
    let num_levels = collapsed_levels.len();

    let n_features = unified.n_features();
    let feature_to_backend = unified.feature_to_backend_row.clone();

    ///////////////////////////////
    // Pseudobulk data per level //
    ///////////////////////////////
    //
    // pb counts live on the unified feature axis directly. If the
    // backend (per_file_data[0]) holds more rows than the unified axis
    // — e.g. an HVG mask narrowed `unified.feature_names` — gather the
    // unified rows out of the backend's pb matrix. Otherwise reuse it.
    let mut pb_blobs: Vec<UnifiedData> = Vec::with_capacity(num_levels);
    for collapsed in &collapsed_levels {
        let pb_full: &DMatrix<f32> = match &collapsed.mu_adjusted {
            Some(adj) => adj.posterior_mean(),
            None => collapsed.mu_observed.posterior_mean(),
        };
        let n_pb = pb_full.ncols();
        let pb_count_ds: DMatrix<f32> = if pb_full.nrows() == n_features {
            pb_full.clone()
        } else {
            let mut subset = DMatrix::<f32>::zeros(n_features, n_pb);
            for (new_i, &old_i) in feature_to_backend.iter().enumerate() {
                for s in 0..n_pb {
                    subset[(new_i, s)] = pb_full[(old_i, s)];
                }
            }
            subset
        };
        pb_blobs.push(UnifiedData::from_pseudobulks(
            &pb_count_ds,
            unified.feature_names.clone(),
            unified.feature_to_backend_row.clone(),
        )?);
    }

    // NOTE: the flat cell↔feature edge list is intentionally *not* built.
    // The cell axis is always `PerBatchStratified`, whose sampler is built by
    // streaming columns in `build_active_samplers` and is self-contained at
    // sample time — so `unified.triplets` stays empty. `materialize_cell_triplets`
    // remains available only for reviving the flat `PerBatch` path.

    ////////////////////////////////
    // VarMap and embedding heads //
    ////////////////////////////////
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(
        &varmap,
        candle_util::candle_core::DType::F32,
        &config.device,
    );
    let zeros_features = vec![0f32; n_features];
    let zeros_cells = vec![0f32; n_cells];

    // The cell head allocates the canonical "e_feat" / "b_feat" /
    // "e_cell" / "b_cell" Vars; every level head then clones those
    // shared `e_feat` / `b_feat` Tensors and registers its own cell
    // side under a unique `pb_l{idx}` prefix.
    let mut cell_model = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features,
            n_cells,
            embedding_dim: h,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &zeros_features,
            b_cell: &zeros_cells,
        },
        &varmap,
        vs,
        &config.device,
    )?;

    let mut level_models: Vec<JointEmbedModel> = Vec::with_capacity(num_levels);
    for (level_idx, pb) in pb_blobs.iter().enumerate() {
        let n_pb = pb.n_cells();
        level_models.push(JointEmbedModel::new_sharing_features(
            ShareFeaturesArgs {
                n_cells: n_pb,
                embedding_dim: h,
                shared_e_feat: cell_model.e_feat.clone(),
                shared_b_feat: cell_model.b_feat.clone(),
                e_cell_init: None,
                b_cell_init: &vec![0f32; n_pb],
                var_prefix: &format!("pb_l{level_idx}"),
            },
            &varmap,
            &config.device,
        )?);
    }

    ////////////////////////////////
    // Composite axes and trainer //
    ////////////////////////////////
    let cell_axis_coarsening = identity_axis(n_cells);
    let cell_samplers = build_active_samplers(
        unified,
        &feat_weights,
        DEFAULT_STRATIFY_ALPHA_CELL,
        alpha_neg,
        config.cell_weight_mult.as_deref(),
    )?;
    info!(
        "Composite axis cell ({} cells × {} features, strat-cell α={}, {} active batch(es))",
        n_cells,
        n_features,
        DEFAULT_STRATIFY_ALPHA_CELL,
        cell_samplers.len()
    );

    // Phase-1 cell-axis mode (`phase1_cells_per_pb` = k). The full
    // `cell_samplers` above are always kept for the phase-2 projection (which
    // visits every cell); k only controls what shapes `E_feat` in phase 1:
    //   k == 0           → suppress the cell axis entirely (pure-pb phase 1);
    //                      `E_feat` is driven by pb aggregates only.
    //   1 ≤ k < n_cells  → subsample a *separate, smaller* view keeping ≤k cells
    //                      per pb-sample at every collapse level (union),
    //                      shrinking the per-epoch step budget from `n_cells`
    //                      to ≈ k × pb-samples while preserving rare-cell coverage.
    //   k ≥ n_cells      → no pb-sample can exceed k, so subsampling is a no-op:
    //                      use the full set (legacy all-cells behaviour).
    let use_cell_axis = config.phase1_cells_per_pb != 0;
    let phase1_cell_samplers_owned: Option<Vec<PerBatchStratifiedCellSampler>> =
        (config.phase1_cells_per_pb >= 1 && config.phase1_cells_per_pb < n_cells).then(|| {
            subsample_cell_samplers_multilevel(
                &cell_samplers,
                &cell_to_pb_per_level,
                config.phase1_cells_per_pb,
                DEFAULT_STRATIFY_ALPHA_CELL,
                config.cell_weight_mult.as_deref(),
                config.seed,
            )
        });
    let phase1_cell_samplers: &[PerBatchStratifiedCellSampler] =
        if let Some(sub) = &phase1_cell_samplers_owned {
            let kept: usize = sub.iter().map(|s| s.active_cells.len()).sum();
            info!(
                "Phase-1 cell subsampling: ≤{} cells per pb-sample (all {} levels) → \
             {} of {} cells shape E_feat (phase 2 still projects all {})",
                config.phase1_cells_per_pb, num_levels, kept, n_cells, n_cells
            );
            sub
        } else {
            // k == 0 → cell axis suppressed (logged); k ≥ n_cells → legacy all-cells.
            if !use_cell_axis {
                info!(
                    "Phase-1 cell axis SUPPRESSED (pure-pb): E_feat shaped by pb aggregates \
                 only; phase 2 still projects all {n_cells} cells"
                );
            }
            &cell_samplers
        };

    let mut level_axes_data: Vec<(AxisCoarsenings, StratifiedSampler)> =
        Vec::with_capacity(num_levels);
    for (level_idx, pb) in pb_blobs.iter().enumerate() {
        let n_pb = pb.n_cells();
        let axis = identity_axis(n_pb);
        let stratified = build_stratified_sampler(
            &pb.triplets,
            n_pb,
            n_features,
            &feat_weights,
            DEFAULT_STRATIFY_ALPHA_PB,
            alpha_neg,
        )
        .ok_or_else(|| {
            anyhow::anyhow!(
                "pb_l{level_idx}: stratified sampler build failed (no positives or empty feature pool)"
            )
        })?;
        info!(
            "Composite axis pb_l{} ({} pseudobulks × {} features, stratified α={}, {} active pb(s))",
            level_idx,
            n_pb,
            n_features,
            DEFAULT_STRATIFY_ALPHA_PB,
            stratified.active_pbs.len()
        );
        level_axes_data.push((axis, stratified));
    }

    let mut smoother = build_smoother(config.feature_network.take(), n_features, h)?;

    // Note on biases: the per-CELL bias `b_cell` and the per-PB biases
    // (`pb_l*_b_cell`) BOTH train in phase 1 — a per-sample bias absorbs
    // that sample's depth so the shared `E_feat` captures composition, not
    // library size. `b_cell` is re-fitted analytically in phase 2 and
    // written alongside `e_cell` (consistent with `faba gem`).

    // Two-phase training (always — `ge::fit` is the bge driver only); see
    // the `fit()` doc for the rationale. Shared AdamW hyperparameters:
    let adamw_params = || ParamsAdamW {
        lr: config.learning_rate,
        weight_decay: config.weight_decay,
        ..Default::default()
    };

    // Cell axis (per-cell embedding). Trained jointly in phase 1 (to shape
    // `E_feat`) and recalibrated in phase 2 against the fixed feature side.
    let cell_axis = CompositeAxis {
        model: &cell_model,
        unified,
        cell_axis: &cell_axis_coarsening,
        sampler: AxisSampler::PerBatchStratified(phase1_cell_samplers),
        lambda: DEFAULT_AXIS_LAMBDA,
        label: "cell",
    };
    // Pseudobulk axes (coarsest→finest).
    let mut pb_axes: Vec<CompositeAxis> = Vec::with_capacity(num_levels);
    for (i, model) in level_models.iter().enumerate() {
        let (axis, stratified) = &level_axes_data[i];
        pb_axes.push(CompositeAxis {
            model,
            unified: &pb_blobs[i],
            cell_axis: axis,
            sampler: AxisSampler::Stratified(stratified),
            lambda: DEFAULT_AXIS_LAMBDA,
            label: "pb",
        });
    }

    /////////////////////////////
    // Phase 1: joint training //
    /////////////////////////////

    // The cell axis is trained HERE (e_cell + b_cell trainable, as are the
    // pb `pb_l*_b_cell`) so the per-cell stratified sampler —
    // which guarantees coverage of rare/shallow cells — shapes `E_feat`.
    // Without the cell axis, `E_feat` is driven only by pb aggregates and rare
    // compartments (DC/NK/HSPC) collapse into abundant ones. Phase 2 then
    // recalibrates e_cell for *every* cell against the fixed `E_feat`.
    // The axes borrow `cell_model` / `cell_samplers`; confine them to this
    // block so those borrows are released before the phase-2 projection
    // takes `&mut cell_model`.
    {
        let mut joint_axes: Vec<CompositeAxis> = Vec::with_capacity(1 + pb_axes.len());
        // `use_cell_axis == false` (phase1_cells_per_pb == 0) trains E_feat from
        // pb aggregates only; `cell_axis` is left unused (its borrow ends here).
        if use_cell_axis {
            joint_axes.push(cell_axis);
        }
        joint_axes.append(&mut pb_axes);
        let mut opt1 = AdamW::new(varmap.all_vars(), adamw_params())?;
        let mut p1 = stage_params(&config);
        p1.composite_mode = CompositeMode::Sum;
        info!(
            "Phase 1 (joint) — features + {}{} pb level(s) [Sum], {} epochs",
            if use_cell_axis { "cell + " } else { "" },
            joint_axes.len() - usize::from(use_cell_axis),
            config.epochs
        );
        train_composite(
            &CompositeTrainContext {
                axes: &joint_axes,
                feat_weights: &feat_weights,
                dev: &config.device,
                stop: &stop,
                cell_to_pb_per_level: None,
            },
            &mut opt1,
            &p1,
            smoother.as_mut(),
        )?;
    }
    // `cell_axis` / `pb_axes` borrows of `cell_model` / `cell_samplers` end
    // here, freeing them for the phase-2 `&mut` projection below.

    // ---- Phase 2: analytical per-cell projection onto the fixed feature
    // side. With E_feat/b_feat/z/δ held fixed, each cell's embedding is
    // independent — so rather than SGD over `e_cell`, project every cell
    // directly (Poisson MAP, ridge prior) in parallel. The per-cell
    // intercept `b_cell` is fitted and kept. See `crate::cell_projection`.
    let mut cell_nrms: Vec<f32> = Vec::new();
    if !stop.load(std::sync::atomic::Ordering::Relaxed) {
        info!(
            "Phase 2 — analytical per-cell projection ({n_cells} cells, feature side fixed, ridge λ={PHASE2_RIDGE})"
        );
        cell_nrms = project_cells_phase2(
            &mut cell_model,
            &varmap,
            &cell_samplers,
            n_cells,
            f64::from(PHASE2_RIDGE),
            &config.device,
        )?;
    }

    Ok(FitOutput {
        model: cell_model,
        varmap,
        cell_nrms,
    })
}
