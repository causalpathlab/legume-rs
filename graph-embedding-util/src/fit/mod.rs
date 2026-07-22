//! Public entry point for `graph-embedding`. Callers translate their
//! own CLI args into a [`FitConfig`] and pass already-loaded
//! [`UnifiedData`] (so this crate stays free of file/path concerns).

mod config;
mod feature_projection;
pub mod lift;
pub mod lineage;
pub mod projection;
pub mod resolve_embedding;
mod samplers;

pub use config::{
    load_feature_network, FeatFactorSpec, FeatureNetworkArgs, FeatureNetworkConfig, FitConfig,
    FitOutput, SoftmaxGateConfig,
};
pub use feature_projection::{
    CalibrationDiag, CalibrationKind, FeatureProjection, FeatureProjectionConfig,
    DEFAULT_PROJECTION_CALIB_RIDGE, DEFAULT_PROJECTION_RIDGE,
};
pub use lift::{CellLineage, LineageQc};
pub use projection::PbLevelVelocity;
pub use resolve_embedding::{train_rest, RestConfig, RestTrainInputs, TrainedRest};

use crate::coarsen::{identity_axis, AxisCoarsenings};
use crate::data::UnifiedData;
use crate::loss::{
    build_stratified_sampler, FeatPairing, PerBatchStratifiedCellSampler, StratifiedSampler,
};
use crate::model::{
    FactoredInit, JointEmbedModel, ModelArgs, ModelInit, ShareFeaturesArgs, SoftmaxGateSpec,
};
use crate::training::{
    train_composite, AxisSampler, CompositeAxis, CompositeMode, CompositeTrainContext, PbDagParams,
    PbDagTerm, PbDagTermSpec, PbSemTerm,
};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use log::info;
use matrix_param::traits::Inference;
use nalgebra::DMatrix;

use config::{
    stage_params, DEFAULT_AXIS_LAMBDA, DEFAULT_STRATIFY_ALPHA_CELL, DEFAULT_STRATIFY_ALPHA_PB,
};
use projection::{project_cells_phase2, project_pbs_phase2, CellBatchDivisor, PHASE2_RIDGE};
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
    let stop = crate::stop::stop_flag();

    ///////////////////////////////////////////////////
    // Shared upstream: batch-corrected projection //
    ///////////////////////////////////////////////////
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
        unified
            .count_backend_mut()
            .project_columns_weighted_seeded(
                config.proj_dim,
                config.block_size,
                batch_arg,
                &backend_w,
                config.seed,
            )?
    } else {
        unified
            .count_backend_mut()
            .project_columns_with_batch_correction_seeded(
                config.proj_dim,
                config.block_size,
                batch_arg,
                config.seed,
            )?
    };

    ///////////////////////////////////////////////////////
    // Multilevel collapse → batch-corrected pseudobulks //
    ///////////////////////////////////////////////////////
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
        let pb_count_ds = gather_to_unified_axis(pb_full, n_features, &feature_to_backend);
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

    // The cell head allocates the canonical feature-side Vars ("e_feat"/"b_feat"
    // for a free model, or "beta"/"b_feat" when β-sharing factored); every level
    // head then SHARES that feature side and registers its own cell side under a
    // unique `pb_l{idx}` prefix.
    let mut cell_model = match &config.feat_factor {
        Some(spec) => {
            // β-sharing is incompatible with the SGC smoother and the free-E_feat
            // L2 (both assume a single free feature table per row).
            anyhow::ensure!(
                config.feature_network.is_none(),
                "feat_factor (β-sharing) is not supported together with --feature-network"
            );
            anyhow::ensure!(
                config.feature_embedding_l2 == 0.0,
                "feat_factor (β-sharing) is not supported with feature_embedding_l2 > 0"
            );
            anyhow::ensure!(
                spec.row_to_gene.len() == n_features && spec.unspliced_rows.len() == n_features,
                "feat_factor row maps (row_to_gene {}, unspliced_rows {}) must match n_features {}",
                spec.row_to_gene.len(),
                spec.unspliced_rows.len(),
                n_features
            );
            // Dense gene ids → count is the max + 1 (single source of truth: the
            // row→gene map; no separately-supplied n_genes to keep in sync).
            let n_genes = spec
                .row_to_gene
                .iter()
                .copied()
                .max()
                .map_or(0, |m| m as usize + 1);
            info!(
                "per-gene β-sharing factorization: {} features → {} genes ({} unspliced rows); \
                 splice δ recovered post-hoc on the cell axis (dual phase-2 projection)",
                n_features,
                n_genes,
                spec.unspliced_rows.iter().filter(|&&b| b).count(),
            );
            // Allocate the ridge-shrunk per-gene splice offset δ_g only when its
            // L2 penalty is on; otherwise plain β-sharing (spliced ≡ unspliced ≡ β_g).
            let unspliced_rows = (config.delta_l2 > 0.0).then_some(spec.unspliced_rows.as_slice());
            if unspliced_rows.is_some() {
                info!(
                    "δ_g splice offset ON (L2={}): unspliced rows embed as β_g + δ_g",
                    config.delta_l2
                );
            }
            JointEmbedModel::new_factored(
                FactoredInit {
                    n_features,
                    n_cells,
                    embedding_dim: h,
                    n_genes,
                    row_to_gene: &spec.row_to_gene,
                    b_feat: &zeros_features,
                    b_cell: &zeros_cells,
                    seed: config.seed,
                    unspliced_rows,
                },
                &varmap,
                vs,
                &config.device,
            )?
        }
        None => JointEmbedModel::new_with_init(
            ModelArgs {
                n_features,
                n_cells,
                embedding_dim: h,
                seed: config.seed,
            },
            &ModelInit {
                e_feat: None,
                e_cell: None,
                b_feat: &zeros_features,
                b_cell: &zeros_cells,
            },
            &varmap,
            &config.device,
        )?,
    };

    // Enable the per-gene softmax gate (SuSiE single-effect prior + graceful feature
    // selection) on the primary model BEFORE the sharing heads are built, so every
    // head references the one shared gate Var (`s_feat`/`s_beta`).
    if let Some(g) = config.softmax_gate {
        cell_model.enable_softmax_gate(
            SoftmaxGateSpec {
                temperature: g.temperature,
            },
            &varmap,
            &config.device,
        )?;
        info!(
            "Softmax feature gate ON (variational spike-and-slab) — τ={}",
            g.temperature
        );
    }

    let mut level_models: Vec<JointEmbedModel> = Vec::with_capacity(num_levels);
    for (level_idx, pb) in pb_blobs.iter().enumerate() {
        let n_pb = pb.n_cells();
        let prefix = format!("pb_l{level_idx}");
        // Each level's cell side is keyed by its unique `{prefix}_e_cell` name,
        // so one base seed yields an independent reproducible init per level.
        let level_model = if cell_model.factor.is_some() {
            cell_model.new_sharing_factor(n_pb, &prefix, &varmap, &config.device, config.seed)?
        } else {
            JointEmbedModel::new_sharing_features(
                ShareFeaturesArgs {
                    n_cells: n_pb,
                    embedding_dim: h,
                    shared_e_feat: cell_model.e_feat.clone(),
                    shared_b_feat: cell_model.b_feat.clone(),
                    e_cell_init: None,
                    b_cell_init: &vec![0f32; n_pb],
                    var_prefix: &prefix,
                    seed: config.seed,
                    // Share the free-model gate (if enabled) so every head reweights
                    // the SAME feature side and AdamW updates one `s_feat`.
                    shared_s_feat: cell_model.s_feat.clone(),
                    shared_e_feat_raw: cell_model.e_feat_raw.clone(),
                    shared_e_feat_logstd: cell_model.e_feat_logstd.clone(),
                    gate: cell_model.gate,
                },
                &varmap,
                &config.device,
            )?
        };
        level_models.push(level_model);
    }

    ////////////////////////////////
    // Composite axes and trainer //
    ////////////////////////////////
    let cell_axis_coarsening = identity_axis(n_cells);
    let cell_samplers = build_active_samplers(
        unified,
        DEFAULT_STRATIFY_ALPHA_CELL,
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

    // β-sharing (gem): sample phase-1 positives by GENE at the spliced count, and
    // emit the paired unspliced edge so δ_g trains at that frequency (identity stays
    // spliced-driven, no double-bite from nascent abundance). `None` for bge (per-row).
    let pairing = config.feat_factor.as_ref().map(|spec| FeatPairing {
        row_to_gene: &spec.row_to_gene,
        unspliced_rows: &spec.unspliced_rows,
    });

    let mut level_axes_data: Vec<(AxisCoarsenings, StratifiedSampler)> =
        Vec::with_capacity(num_levels);
    for (level_idx, pb) in pb_blobs.iter().enumerate() {
        let n_pb = pb.n_cells();
        let axis = identity_axis(n_pb);
        let stratified = build_stratified_sampler(
            &pb.triplets,
            n_pb,
            n_features,
            DEFAULT_STRATIFY_ALPHA_PB,
            pairing.as_ref(),
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
                dev: &config.device,
                stop: &stop,
                cell_to_pb_per_level: None,
                lineage_sem: None,
                lineage_sem_theta: None,
                lineage_dag: None,
                lineage_dag_stride: 1,
            },
            &mut opt1,
            &p1,
            smoother.as_mut(),
        )?;
    }
    // `cell_axis` / `pb_axes` borrows of `cell_model` / `cell_samplers` end
    // here, freeing them for the phase-2 `&mut` projection below.

    // Snapshot the deltaTopic β+δ into the `e_feat` field so phase-2 (and every
    // output/co-embed reader) sees a fixed materialized dictionary. No-op for a
    // free model.
    cell_model.materialize_e_feat()?;

    // Lineage-DAG refine (gem β-sharing only; fixed velocity-KNN structure). The warm-up
    // phase 1 above yields a trained-enough dictionary: read pb-level velocity
    // (identity θ_pb + velocity δ_pb, reusing the phase-2 dual solver on the
    // already-batch-corrected pb aggregates), build a fixed velocity-oriented pb
    // graph, and run a SECOND phase-1 pass with the velocity-drift SEM residual on
    // so the shared E_feat picks up lineage geometry. The returned `pb_velocity` is
    // the FINAL readout (post-refine), consumed by the phase-2 cell lift. Flag off
    // or non-β-sharing ⇒ `None` and byte-identical training.
    let mut pb_velocity: Option<Vec<PbLevelVelocity>> = None;
    let mut pb_dag_w: Option<Vec<Vec<f32>>> = None;
    let mut refine_loss = 0f32; // final refine loss → QC likelihood-hygiene signal
    if config.lineage_dag && !stop.load(std::sync::atomic::Ordering::Relaxed) {
        match config.feat_factor.as_ref() {
            Some(spec) => {
                // Warm-up pb velocity, optionally smoothed + confidence-gated (①+②) so
                // `sign(δ_pb)` is stabilized before it orients the graph / SEM drift.
                let warmup_vel = maybe_smooth(
                    pb_velocity_readout(&cell_model, &pb_blobs, &spec.unspliced_rows)?,
                    h,
                    config.lineage_smooth,
                );

                // Rebuild the phase-1 axes for the refine pass (warm-up axes were
                // consumed). Same axis set; only the lineage term differs.
                let mut refine_axes: Vec<CompositeAxis> = Vec::with_capacity(1 + num_levels);
                if use_cell_axis {
                    refine_axes.push(CompositeAxis {
                        model: &cell_model,
                        unified,
                        cell_axis: &cell_axis_coarsening,
                        sampler: AxisSampler::PerBatchStratified(phase1_cell_samplers),
                        lambda: DEFAULT_AXIS_LAMBDA,
                        label: "cell",
                    });
                }
                for (i, model) in level_models.iter().enumerate() {
                    let (axis, stratified) = &level_axes_data[i];
                    refine_axes.push(CompositeAxis {
                        model,
                        unified: &pb_blobs[i],
                        cell_axis: axis,
                        sampler: AxisSampler::Stratified(stratified),
                        lambda: DEFAULT_AXIS_LAMBDA,
                        label: "pb",
                    });
                }
                let mut p2 = stage_params(&config);
                p2.composite_mode = CompositeMode::Sum;

                if config.dag_learnable {
                    // Sample the DATA side densely (NCE every batch, ~16 batches/epoch)
                    // and the STRUCTURE side sparsely (`W`/SEM every DAG_STRIDE steps) —
                    // a warm-started `W` is a light prior, not a from-scratch structure to
                    // hammer. `DAG_STRIDE = 8` ⇒ ~2 structure updates/epoch vs 16 NCE.
                    const DAG_STRIDE: usize = 8;
                    p2.batches_per_epoch = Some(p2.batches_per_epoch.unwrap_or(1).max(16));
                    // Per-axis term slices lead with the optional cell axis, so pb level
                    // `i`'s term sits at `offset + i`.
                    let offset = usize::from(use_cell_axis);
                    // Warm-start each `W` from the velocity-oriented KNN so SGD refines a
                    // correctly-oriented structure instead of learning one from zeros
                    // (zero-init `W` is the unstable, non-monotone start).
                    let knn_init =
                        lineage::build_pb_lineage(&warmup_vel, h, lineage::DEFAULT_LINEAGE_KNN);
                    // θ-pseudotime DAG per level (τ-forward θ-KNN + gradient ĝ): supplies the
                    // unified `W`'s θ-topology (weighted L1) and the δ-sparse-fallback drift.
                    let theta_dag = lineage::build_theta_dag(
                        &warmup_vel,
                        &knn_init,
                        h,
                        lineage::DEFAULT_LINEAGE_KNN,
                    );
                    // learned-DAG: one UNIFIED `PbDagTerm` per pb level — a single `W`
                    // explaining BOTH structures (θ-weighted-L1 topology + δ/θ-gated drift).
                    // Registers a `W` Var (must precede the optimizer), aligned to the refine
                    // axes ([cell?] + pb levels).
                    let mut dag_terms: Vec<Option<PbDagTerm>> = Vec::with_capacity(1 + num_levels);
                    if use_cell_axis {
                        dag_terms.push(None);
                    }
                    for (i, lvl) in warmup_vel.iter().enumerate() {
                        let w0 = knn_init[i].to_dense();
                        dag_terms.push(PbDagTerm::new(PbDagTermSpec {
                            vel: lvl,
                            theta_dag: &theta_dag[i],
                            h,
                            params: PbDagParams::default(),
                            var_name: &format!("dag_l{i}_w"),
                            varmap: &varmap,
                            dev: &config.device,
                            w_init: Some(&w0),
                        })?);
                    }
                    let n_dag = dag_terms.iter().filter(|t| t.is_some()).count();
                    info!(
                        "Lineage-DAG refine (unified learned DAG) — {} pb-level DAG(s), warm-started \
                         from the velocity-KNN graph [SEM + θ-weighted L1 + DAGMA acyclicity; δ/θ-gated \
                         drift], structure every {} steps, {} epochs",
                        n_dag, DAG_STRIDE, config.epochs
                    );
                    let mut opt2 = AdamW::new(varmap.all_vars(), adamw_params())?;
                    refine_loss = train_composite(
                        &CompositeTrainContext {
                            axes: &refine_axes,
                            dev: &config.device,
                            stop: &stop,
                            cell_to_pb_per_level: None,
                            lineage_sem: None,
                            // Unified single `W` carries θ too (θ-weighted L1 + fused drift),
                            // so no separate θ-SEM term on the learned-DAG path.
                            lineage_sem_theta: None,
                            lineage_dag: Some(&dag_terms),
                            lineage_dag_stride: DAG_STRIDE,
                        },
                        &mut opt2,
                        &p2,
                        smoother.as_mut(),
                    )?;
                    // Extract the learned `W` per pb level (aligned to `pb_blobs`).
                    let mut ws = Vec::with_capacity(num_levels);
                    for i in 0..num_levels {
                        ws.push(match &dag_terms[offset + i] {
                            Some(d) => d.w_dense()?,
                            None => Vec::new(),
                        });
                    }
                    pb_dag_w = Some(ws);
                    drop(refine_axes);
                } else {
                    // velocity-KNN: fixed velocity-oriented KNN graph + velocity-drift SEM residual.
                    // The dense KNN graph (each node → its velocity-forward neighbours),
                    // built once from the warm-up readout, shapes E_feat in a single refine
                    // pass. `pb_dag_w` stays `None`; the cell-lift rebuilds the same graph from
                    // the final `pb_velocity`.
                    let levels =
                        lineage::build_pb_lineage(&warmup_vel, h, lineage::DEFAULT_LINEAGE_KNN);
                    let n_edges: usize = levels.iter().map(|l| l.edges.len()).sum();
                    let mut terms: Vec<Option<PbSemTerm>> = Vec::with_capacity(1 + num_levels);
                    if use_cell_axis {
                        terms.push(None);
                    }
                    for lvl in &levels {
                        terms.push(PbSemTerm::new(
                            lvl,
                            h,
                            lineage::DEFAULT_SEM_STEP,
                            lineage::DEFAULT_SEM_WEIGHT,
                            &config.device,
                        )?);
                    }
                    info!(
                        "Lineage-DAG refine (fixed velocity-KNN) — {} oriented pb edge(s) across \
                         {} level(s); velocity-drift SEM residual ON",
                        n_edges,
                        levels.len()
                    );
                    // Second lineage term: the θ-pseudotime DAG (see the learned branch).
                    let theta_terms = build_theta_sem_terms(
                        &warmup_vel,
                        &levels,
                        h,
                        use_cell_axis,
                        num_levels,
                        &config.device,
                    )?;
                    let mut opt2 = AdamW::new(varmap.all_vars(), adamw_params())?;
                    refine_loss = train_composite(
                        &CompositeTrainContext {
                            axes: &refine_axes,
                            dev: &config.device,
                            stop: &stop,
                            cell_to_pb_per_level: None,
                            lineage_sem: Some(&terms),
                            lineage_sem_theta: Some(&theta_terms),
                            lineage_dag: None,
                            lineage_dag_stride: 1,
                        },
                        &mut opt2,
                        &p2,
                        smoother.as_mut(),
                    )?;
                    drop(refine_axes);
                }

                // Refresh the dictionary and read the FINAL pb velocity (post-refine),
                // smoothed the same way so the cell-lift orients off a denoised field.
                cell_model.materialize_e_feat()?;
                pb_velocity = Some(maybe_smooth(
                    pb_velocity_readout(&cell_model, &pb_blobs, &spec.unspliced_rows)?,
                    h,
                    config.lineage_smooth,
                ));
            }
            None => {
                log::warn!(
                    "lineage_dag set but the model is not β-sharing (feat_factor = None); \
                     skipping lineage refine"
                );
            }
        }
    }

    // The first Ctrl+C stops the *major SGD loops* — phase 1 above and the lineage
    // refine — and nothing else. Every stage from here down is a follow-up routine
    // that turns the trained dictionary into the run's deliverables, so gating them
    // on `stop` does not save the user time, it destroys the output. Phase 2
    // especially: at the default `--phase1-cells-per-pb 0` the cell axis is never
    // trained in phase 1, so `e_cell` is still its randn init until phase 2 runs —
    // skipping it wrote a `cell_embedding.parquet` of pure noise, silently. A
    // second Ctrl+C aborts the process outright (`matrix_util::stop`), which is the
    // escape hatch for a user who really does want out now.
    if stop.load(std::sync::atomic::Ordering::Relaxed) {
        log::warn!(
            "phase 1 was interrupted — the feature dictionary is short-trained. The follow-up \
             stages (phase-2 projection, held-out feature projection, cell-lift) still run, so \
             the outputs are complete but fit against a partially trained dictionary; treat \
             this run as a draft. Ctrl+C again to abort outright."
        );
    }

    // Phase 2: per-cell projection onto the fixed feature side. With
    // E_feat/b_feat/z/δ held fixed each cell's embedding is independent, so this is
    // a cell-block Poisson SGD over `e_cell`/`b_cell` alone — see
    // `projection::project_cells_phase2`. The per-cell intercept `b_cell` is fitted
    // and kept.
    let phase2 = {
        // Phase-2 batch correction (mirrors senna svd/topic): divide each cell's
        // counts by its finest-pb μ_residual fold-factor. μ_residual is gathered
        // onto the unified feature axis so a feature id indexes a row directly;
        // built only when the collapse fit one (>1 batch).
        let phase2_mu_residual: Option<DMatrix<f32>> = collapsed_levels
            .last()
            .and_then(|c| c.mu_residual.as_ref())
            .map(|mr| gather_to_unified_axis(mr.posterior_mean(), n_features, &feature_to_backend));
        let batch_divisor = phase2_mu_residual.as_ref().map(|mu| CellBatchDivisor {
            mu_residual: mu,
            // `.last()` is always `Some` here: the divisor only exists when the
            // collapse produced a μ_residual, i.e. ≥1 level (num_levels.max(1)),
            // and `cell_to_pb_per_level` has the same length.
            cell_to_pb: cell_to_pb_per_level
                .last()
                .map(Vec::as_slice)
                .expect("collapse always produces ≥1 level"),
        });

        // β-sharing (gem): identity is resolved by the SPLICED edges (stored raw),
        // and a second pass emits the raw velocity increment δ on the cell axis.
        // Plain (bge): one combined projection = identity (stored as dir), no splice
        // output.
        let unspliced = config
            .feat_factor
            .as_ref()
            .map(|s| s.unspliced_rows.as_slice());
        project_cells_phase2(
            &mut cell_model,
            &varmap,
            &cell_samplers,
            n_cells,
            f64::from(PHASE2_RIDGE),
            &config.device,
            batch_divisor,
            unspliced,
        )?
    };

    // Post-hoc held-out feature projection. Solve `β_g` for every backend
    // feature the training axis never saw (the `--n-hvg` remainder, plus
    // anything the feature-null QC dropped) against the frozen pseudobulk side.
    // Strictly read-only on the cell side, so every cell output above is
    // unaffected. See `crate::fit::feature_projection`.
    // Not gated on `stop` — see the phase-2 note above.
    let feature_projection = match &config.feature_projection {
        Some(fp_cfg) => {
            let pb = stacked_pb_view(&varmap, &collapsed_levels, &cell_to_pb_per_level, h)?;
            let trained = trained_gene_beta(
                &cell_model,
                config.feat_factor.as_ref(),
                &feature_to_backend,
                &fp_cfg.backend_row_to_gene,
                h,
            )?;
            Some(feature_projection::project_held_out_features(
                &pb, h, &trained, fp_cfg,
            ))
        }
        _ => None,
    };

    // cell-lift: phase-2 cell-lineage lift (evaluation only). Runs on the FINAL pb
    // velocity readout + the now-projected per-cell identity θ_c. Integrate a pb
    // pseudotime/fate along the oriented pb-DAG (learned `W` for the learned DAG, the fixed
    // velocity-oriented graph for the velocity-KNN) at the finest level, then landmark-blend it to
    // every cell. `None` when lineage-DAG is off or the readout is empty.
    // Not gated on `stop` — see the phase-2 note above.
    let mut lineage_qc: Option<LineageQc> = None;
    let cell_lineage = match &pb_velocity {
        Some(pbv) if !pbv.is_empty() => {
            let level = pbv.len() - 1; // finest level: densest landmark tiling
            let vel = &pbv[level];
            let edges = match &pb_dag_w {
                // learned-DAG: dense learned W (forward-masked) → positive-mass edges.
                Some(ws) => lift::dense_to_edges(&ws[level], vel.n_pb),
                // velocity-KNN: rebuild the fixed velocity-oriented graph from the readout.
                None => lineage::build_pb_lineage(
                    std::slice::from_ref(vel),
                    h,
                    lineage::DEFAULT_LINEAGE_KNN,
                )
                .pop()
                .map(|l| {
                    l.edges
                        .into_iter()
                        .map(|(i, j, w)| (i as usize, j as usize, w))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            };
            let traj = lift::pb_trajectory(vel, &edges, h, lineage::DEFAULT_SEM_STEP);
            // Put the cells back in the LANDMARKS' frame before comparing. `vel` was
            // read by the Newton pb readout above, i.e. before phase 2 gauge-fixed
            // the cell latents, and pb θ is never re-gauged — so `lift_cells`, which
            // takes `dist2(θ_c, θ_p)` and projects `θ_c − θ_p` onto the pb velocity,
            // would otherwise be differencing two different frames and displacing
            // every cell by `‖θ̄‖` (88 on the reference fit, against a post-centring
            // median `‖θ‖` of 5.6). See `Phase2Result::theta_mean`.
            let mut theta_c: Vec<f32> = cell_model.e_cell.flatten_all()?.to_vec1()?;
            for (k, x) in theta_c.iter_mut().enumerate() {
                *x += phase2.theta_mean[k % h];
            }
            let lin = lift::lift_cells(&theta_c, n_cells, vel, &traj, h, level);
            // Unsupervised per-run structural diagnostics (decisiveness, coherence, fate
            // count, ambiguity, likelihood) — for run inspection, not a validated quality
            // ranker.
            let qc = lift::compute_lineage_qc(
                &traj,
                vel,
                &lin,
                refine_loss,
                h,
                lineage::DEFAULT_LINEAGE_KNN,
            );
            info!(
                "cell-lift — finest pb level {}: {} nodes, {} root(s), {} fate(s), \
                 top-source reach {:.2}, velocity-coherence {:.2}",
                level,
                vel.n_pb,
                traj.roots.len(),
                traj.terminals.len(),
                qc.root_decisiveness,
                qc.velocity_coherence,
            );
            lineage_qc = Some(qc);
            Some(lin)
        }
        _ => None,
    };

    Ok(FitOutput {
        model: cell_model,
        varmap,
        cell_nrms: phase2.cell_nrms,
        cell_velocity: phase2.velocity,
        pb_velocity,
        pb_dag_w,
        cell_lineage,
        lineage_qc,
        feature_projection,
    })
}

/// The trained per-gene `β`, plus each trained gene's id on the caller's backend
/// gene axis. Returns `(beta [n_trained_genes × H] row-major, backend_gene_id)`.
///
/// For a **factored** (β-sharing) model the source is the `beta` Var itself: a
/// gene's `e_feat` rows are `β_g` (spliced) and `β_g + δ_g` (unspliced), so only
/// `β` is a clean calibration target. For a **free** model (bge) every compact
/// feature row is its own gene and `e_feat` is exactly `β`.
///
/// **Softmax gate.** When the gate is on, the held-out solve estimates each gene's
/// *effective* embedding, so the calibration target must be the GATED per-gene
/// `β_g ⊙ softmax(s_beta_g)` (matching what `θ_pb` was trained against). The free
/// branch needs no special handling: `model.e_feat` was already overwritten with the
/// gated dictionary by `materialize_e_feat` before this runs.
fn trained_gene_beta(
    model: &JointEmbedModel,
    spec: Option<&FeatFactorSpec>,
    feature_to_backend: &[usize],
    backend_row_to_gene: &[u32],
    h: usize,
) -> anyhow::Result<feature_projection::TrainedBeta> {
    let (beta, backend_gene_id) = match (spec, &model.factor) {
        (Some(spec), Some(factor)) => {
            // `s_beta` is `Some` iff the gate is on (both set together by
            // `enable_softmax_gate`), so checking it alone is sufficient.
            let beta_t = match &factor.s_beta {
                Some(s_beta) => model.apply_softmax_gate(&factor.beta, s_beta)?,
                None => factor.beta.clone(),
            };
            let beta = beta_t.flatten_all()?.to_vec1::<f32>()?;
            let n_trained_genes = beta.len() / h;
            // compact gene id → backend gene id, via any of the gene's rows.
            let mut backend_gene_id = vec![u32::MAX; n_trained_genes];
            for (r, &g) in spec.row_to_gene.iter().enumerate() {
                backend_gene_id[g as usize] = backend_row_to_gene[feature_to_backend[r]];
            }
            anyhow::ensure!(
                backend_gene_id.iter().all(|&g| g != u32::MAX),
                "trained gene without any feature row — β-sharing factor is inconsistent"
            );
            (beta, backend_gene_id)
        }
        _ => {
            let beta = model.e_feat.flatten_all()?.to_vec1::<f32>()?;
            let backend_gene_id = feature_to_backend
                .iter()
                .map(|&row| backend_row_to_gene[row])
                .collect();
            (beta, backend_gene_id)
        }
    };
    Ok(feature_projection::TrainedBeta {
        beta,
        backend_gene_id,
    })
}

/// Stack every collapse level's **trained** pseudobulk embedding into one frozen
/// table, paired with that level's full-backend count matrix.
///
/// Phase 1 shapes `β` against exactly these axes — one per level, combined with
/// `CompositeMode::Sum` at uniform [`DEFAULT_AXIS_LAMBDA`] — and with the default
/// `phase1_cells_per_pb == 0` the cell axis is suppressed entirely, so this stack
/// *is* the objective `β` was fit under. Solving a held-out gene against it puts
/// the result in `β`'s native frame.
///
/// The pb Vars are read out of the `VarMap` by name rather than off
/// `level_models[l].e_cell`: the latter is a `Tensor` aliasing the `Var`'s
/// storage, and whether it tracks in-place `Var::set` updates is a candle
/// implementation detail.
///
/// Counts come from `mu_adjusted` when the collapse produced one, matching the
/// `pb_blobs` the model actually trained on, so held-out and trained genes are on
/// the same scale.
fn stacked_pb_view<'a>(
    varmap: &VarMap,
    collapsed_levels: &'a [data_beans_alg::collapse_data::CollapsedOut],
    cell_to_pb_per_level: &[Vec<usize>],
    h: usize,
) -> anyhow::Result<feature_projection::StackedPb<'a>> {
    let vars = varmap.data().lock().expect("varmap poisoned");
    let (mut theta, mut bias, mut counts, mut sizes, mut offsets) =
        (vec![], vec![], vec![], vec![], vec![]);
    for (level, collapsed) in collapsed_levels.iter().enumerate() {
        let get = |suffix: &str| -> anyhow::Result<Vec<f32>> {
            let name = format!("pb_l{level}_{suffix}");
            let var = vars
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("pb var {name} missing from the varmap"))?;
            Ok(var.as_tensor().flatten_all()?.to_vec1::<f32>()?)
        };
        let level_bias = get("b_cell")?;
        let level_theta = get("e_cell")?;
        let n_pb = level_bias.len();
        let pb_full = match &collapsed.mu_adjusted {
            Some(adj) => adj.posterior_mean(),
            None => collapsed.mu_observed.posterior_mean(),
        };
        anyhow::ensure!(
            level_theta.len() == n_pb * h && pb_full.ncols() == n_pb,
            "pb_l{level}: embedding ({} × {h}) and counts ({} pb) disagree on the pseudobulk count {n_pb}",
            level_theta.len() / h.max(1),
            pb_full.ncols(),
        );

        // Exposure: cells per pseudobulk. The collapse stores per-cell RATES, so a
        // Poisson likelihood on them is mis-scaled unless each column carries its
        // `size_p` — see `StackedPb`. Empty pseudobulks are clamped to 1 so the
        // `log(size)` offset stays finite; their counts are zero anyway.
        let mut level_sizes = vec![0f32; n_pb];
        for &p in &cell_to_pb_per_level[level] {
            if p < n_pb {
                level_sizes[p] += 1.0;
            }
        }
        for s in &mut level_sizes {
            *s = s.max(1.0);
        }

        offsets.push(bias.len());
        theta.extend(level_theta);
        bias.extend(level_bias.iter().zip(&level_sizes).map(|(b, s)| b + s.ln()));
        counts.push(pb_full);
        sizes.push(level_sizes);
    }
    Ok(feature_projection::StackedPb {
        theta,
        bias,
        counts,
        sizes,
        offsets,
    })
}

/// Build the θ-pseudotime DAG's per-axis SEM terms for the lineage refine, aligned
/// 1:1 with the refine axes ([cell?] + pb levels) like the velocity terms. `vel_levels`
/// is the velocity-oriented graph, used only to pick each level's root; orientation and
/// drift come from θ. See [`lineage::build_theta_dag`].
fn build_theta_sem_terms(
    warmup_vel: &[PbLevelVelocity],
    vel_levels: &[lineage::PbLineageLevel],
    h: usize,
    use_cell_axis: bool,
    num_levels: usize,
    dev: &candle_util::candle_core::Device,
) -> anyhow::Result<Vec<Option<PbSemTerm>>> {
    let theta_levels =
        lineage::build_theta_dag(warmup_vel, vel_levels, h, lineage::DEFAULT_LINEAGE_KNN);
    let mut terms: Vec<Option<PbSemTerm>> = Vec::with_capacity(1 + num_levels);
    if use_cell_axis {
        terms.push(None);
    }
    for lvl in &theta_levels {
        terms.push(PbSemTerm::new(
            lvl,
            h,
            lineage::DEFAULT_SEM_STEP,
            lineage::DEFAULT_THETA_SEM_WEIGHT,
            dev,
        )?);
    }
    Ok(terms)
}

/// Apply the velocity-graph smoothing + confidence gating (①+②) to a pb readout when
/// `on`; an identity pass-through otherwise. Kept here so both readout sites share one
/// call and the default constants stay in one place.
fn maybe_smooth(vel: Vec<PbLevelVelocity>, h: usize, on: bool) -> Vec<PbLevelVelocity> {
    if on {
        lineage::smooth_pb_velocity_levels(&vel, h, lineage::DEFAULT_SMOOTH_KNN)
    } else {
        vel
    }
}

/// Analytic pb-level velocity readout: identity `θ_pb` + velocity `δ_pb` per pb
/// node per level, reusing the phase-2 dual solver on the (already
/// batch-corrected) pb aggregates. Requires a materialized `e_feat` dictionary.
/// Called twice on the lineage-DAG path — once on the warm-up dictionary (to
/// orient the fixed pb graph) and once after the refine pass (the returned readout).
fn pb_velocity_readout(
    model: &JointEmbedModel,
    pb_blobs: &[UnifiedData],
    unspliced_rows: &[bool],
) -> anyhow::Result<Vec<PbLevelVelocity>> {
    let feat_flat = model.e_feat.flatten_all()?.to_vec1()?;
    let b_feat_v = model.b_feat.to_vec1()?;
    project_pbs_phase2(
        &feat_flat,
        &b_feat_v,
        model.embedding_dim,
        pb_blobs,
        unspliced_rows,
        f64::from(PHASE2_RIDGE),
    )
}

/// Gather a backend-axis `[backend_rows × cols]` matrix onto the unified feature
/// axis `[n_features × cols]` via `feature_to_backend` (a clone when the axes
/// already match, e.g. no HVG mask narrowed the feature set). Shared by the
/// per-level pb counts and the phase-2 `μ_residual` divisor.
fn gather_to_unified_axis(
    backend: &DMatrix<f32>,
    n_features: usize,
    feature_to_backend: &[usize],
) -> DMatrix<f32> {
    if backend.nrows() == n_features {
        return backend.clone();
    }
    let cols = backend.ncols();
    let mut out = DMatrix::<f32>::zeros(n_features, cols);
    for (new_i, &old_i) in feature_to_backend.iter().enumerate() {
        for s in 0..cols {
            out[(new_i, s)] = backend[(old_i, s)];
        }
    }
    out
}
