//! Public entry point for `graph-embedding`. Callers translate their
//! own CLI args into a [`FitConfig`] and pass already-loaded
//! [`UnifiedData`] (so this crate stays free of file/path concerns).

mod config;
pub mod lift;
pub mod lineage;
pub mod projection;
pub mod resolve_embedding;
mod samplers;
pub(crate) mod stacked_pb;

pub use config::{FeatFactorSpec, FeatureGateConfig, FitConfig, FitOutput};
pub use lift::{CellLineage, LineageQc};
pub use projection::PbLevelVelocity;
pub use resolve_embedding::{train_rest, RestConfig, RestTrainInputs, TrainedRest};

use crate::coarsen::{identity_axis, AxisCoarsenings};
use crate::data::UnifiedData;
use crate::loss::{
    build_stratified_sampler, FeatPairing, PerBatchStratifiedCellSampler, StratifiedSampler,
};
use crate::model::{
    FactoredInit, GateKind, JointEmbedModel, ModelArgs, ModelInit, ShareFeaturesArgs,
};
use crate::training::{
    train_composite, AxisSampler, CompositeAxis, CompositeMode, CompositeTrainContext, PbSemTerm,
};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use log::info;
use matrix_param::traits::Inference;
use nalgebra::DMatrix;

use config::{
    stage_params, DEFAULT_AXIS_LAMBDA, DEFAULT_STRATIFY_ALPHA_CELL, DEFAULT_STRATIFY_ALPHA_PB,
    LINEAGE_WARMUP_FRAC,
};
use projection::{project_cells_phase2, project_pbs_phase2, CellBatchDivisor, PHASE2_RIDGE};
use samplers::{build_active_samplers, subsample_cell_samplers_multilevel};

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
pub fn fit(unified: &mut UnifiedData, config: FitConfig) -> anyhow::Result<FitOutput> {
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
            // β-sharing is incompatible with the free-E_feat L2, which assumes a
            // single free feature table per row.
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

    // Enable the softmax feature gate on the primary model BEFORE the sharing heads
    // are built, so every head references the one shared gate Var (`s_feat`/`s_beta`).
    if let Some(g) = config.feature_gate {
        cell_model.enable_feature_gate(g, &varmap, &config.device)?;
        info!(
            "Softmax feature gate ON (per-dim distribution over genes) — τ={}",
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

    // `--lineage-dag` reallocates the ONE `config.epochs` budget across the warm-up
    // (phase 1) and the refine instead of doubling it. The DAG can only be oriented
    // from a *trained* velocity readout (chicken-and-egg), so a warm-up before the
    // lineage term is required — but the refine is warm-started, so it needs a
    // refinement, not a second full-length fit. Off the lineage path phase 1 keeps
    // the whole budget (`refine_epochs == 0`) and the run is byte-identical.
    // Phase 1 is SGD **xor** sampling. An initialization cannot bias a *converged*
    // chain — it only sets burn-in — so warm-starting the sampler from an SGD optimum is
    // either harmless or fatal, and the recorded rank collapse says fatal: with an
    // effective rank of ~2.5 of 128 and max VIF ~105, the surplus directions have a flat
    // likelihood, the chain random-walks in them, and nothing washes out at any sweep
    // count anyone will pay for. `pip` and `π₀h` would then describe curvature around
    // the SGD optimum rather than a posterior. So when the posterior is requested it
    // *replaces* phase 1 rather than refining it.
    //
    // Phase **2** is unaffected: it is an analytical Poisson-MAP projection, not SGD.
    let sample_phase1 = config.pb_posterior.is_some();
    if sample_phase1 {
        // Constraints, enforced rather than discovered. Neither is reachable by
        // accident — `phase1_cells_per_pb` defaults to 0 and the lineage refine is
        // opt-in — but silently leaving a cell axis at its random init, or silently
        // skipping a refine the caller asked for, would both be worse than stopping.
        anyhow::ensure!(
            !use_cell_axis,
            "--posterior samples the PSEUDOBULK phase, and there is no cell block to \
             sample: --phase1-cells-per-pb {} would add a cell axis that only SGD \
             trains, leaving it at its random initialization. Pass \
             --phase1-cells-per-pb 0 (the default) to sample, or drop --posterior to \
             train that axis.",
            config.phase1_cells_per_pb
        );
        anyhow::ensure!(
            !(config.lineage_dag && config.feat_factor.is_some()),
            "--lineage-dag refines a TRAINED fit by SGD, so it cannot be combined with \
             --posterior, which replaces that training. Run one or the other."
        );
        // The sampler's likelihood is the profiled Poisson, whose normalizer
        // `T_a · ln Σ exp(s)` IS the sampled-softmax estimand: dividing it by `T_a` gives
        // `E_{o~p̂}[s_o] − ln Σ exp(s_o)`, which is InfoNCE for the anchor with its
        // positives weighted by count. Against `NceObjective::Logistic` that identity
        // simply does not hold — SGNS is a sum of independent per-pair decisions,
        // `Σ log σ(s) + Σ log σ(−s)`, with no `logsumexp` anywhere.
        //
        // So sampling a logistic fit with this likelihood would report a posterior for a
        // model nobody asked for, and would do it silently, with a full set of
        // plausible-looking tables. Refuse instead. Sampling the logistic objective is a
        // real option and the design is settled — it stays `affine_in_anchor`, so the
        // rank-1 column update still applies, and its intercept has no closed-form
        // profile but does have a monotone score equation solvable per sweep by bisection,
        // exactly the Bayes-EM step the Poisson path already takes for its live
        // intercepts. It is simply not implemented, and an error is the honest way to say
        // that.
        anyhow::ensure!(
            config.nce_objective != crate::loss::NceObjective::Logistic,
            "--posterior samples the profiled-Poisson likelihood, which is the same \
             estimand as --nce-objective softmax but NOT as logistic (SGNS is a sum of \
             per-pair decisions, with no logsumexp). Sampling a logistic fit with it would \
             report a posterior for a different model. Use --nce-objective softmax with \
             --posterior, or drop --posterior to train with logistic."
        );
    }

    let lineage_on = config.lineage_dag && config.feat_factor.is_some();
    let warmup_epochs = if lineage_on && config.epochs > 0 {
        ((LINEAGE_WARMUP_FRAC * config.epochs as f64).round() as usize).clamp(1, config.epochs)
    } else {
        config.epochs
    };
    let refine_epochs = config.epochs - warmup_epochs;
    if sample_phase1 {
        // One unmissable line, because "did SGD run?" must never be a guess. `--epochs`
        // is deliberately named in it: it still has a value and it no longer governs
        // this phase, which is exactly the kind of thing a reader assumes otherwise.
        info!(
            "Phase 1 = SAMPLED, not trained — SGD is SKIPPED on this path, so --epochs \
             ({}) does not apply to it. The pseudobulk model is initialized from the \
             data and then sampled; phase 2 still runs (it is an analytical projection, \
             not SGD).",
            config.epochs
        );
    } else {
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
        p1.epochs = warmup_epochs;
        let cell_prefix = if use_cell_axis { "cell + " } else { "" };
        let n_pb_levels = joint_axes.len() - usize::from(use_cell_axis);
        if refine_epochs > 0 {
            info!(
                "Phase 1 (joint) = LINEAGE WARM-UP — {}/{} epochs; the DAG refine gets the other \
                 {} (ONE shared epoch budget, NOT doubled). Training features + {}{} pb level(s) [Sum]",
                warmup_epochs, config.epochs, refine_epochs, cell_prefix, n_pb_levels,
            );
        } else {
            info!(
                "Phase 1 (joint) — features + {}{} pb level(s) [Sum], {} epochs",
                cell_prefix, n_pb_levels, warmup_epochs,
            );
        }
        train_composite(
            &CompositeTrainContext {
                axes: &joint_axes,
                dev: &config.device,
                stop: &stop,
                cell_to_pb_per_level: None,
                lineage_sem: None,
                lineage_sem_theta: None,
            },
            &mut opt1,
            &p1,
        )?;
    }
    // `cell_axis` / `pb_axes` borrows of `cell_model` / `cell_samplers` end
    // here, freeing them for the phase-2 `&mut` projection below.

    // Phase-1 posterior. Runs HERE — after the SGD that warm-starts it, before
    // `materialize_e_feat` bakes the dictionary — so writing the posterior means
    // back into the Vars makes this a refinement of phase 1 rather than a second
    // set of tables nothing reads. Everything downstream (phase 2, the
    // dictionary, the co-embed) then sees the sampled fit.
    let mut splice_posterior = None;
    let pb_posterior = match config.pb_posterior.as_ref() {
        None => None,
        Some(pcfg) if config.feat_factor.is_some() => {
            // gem's β-sharing side: one `β_g` per gene plus `δ_g` on the unspliced
            // rows, i.e. TWO gates over a row→gene grouping. Sampled by the splice
            // variant; the write-back targets `beta` / `delta`, not `e_feat`.
            let spec = config.feat_factor.as_ref().expect("checked");
            let pb = stacked_pb_view(&varmap, &collapsed_levels, &cell_to_pb_per_level, h)?;
            // NOT `cell_model.e_feat`: on a factored model the training loss never
            // touches that Var, and `materialize_e_feat` has not run yet — it is
            // still the randn snapshot from init. Build the per-row MAP from the
            // trained `beta`/`delta` Vars instead, so both the warm start and the
            // fallback loading for rows no gene claims come from the actual fit.
            let e_feat_map = materialized_splice_rows(&varmap, spec, unified.n_features(), h)?;
            let b_feat_map: Vec<f32> = cell_model.b_feat.to_vec1()?;
            let feat = crate::posterior::pb_index::FeatureSide {
                e_feat: &e_feat_map,
                b_feat: &b_feat_map,
                feature_to_backend_row: &unified.feature_to_backend_row,
            };
            let tracks = crate::posterior::pb_gibbs::SpliceTracks {
                row_to_gene: &spec.row_to_gene,
                unspliced_rows: &spec.unspliced_rows,
                n_genes: spec
                    .row_to_gene
                    .iter()
                    .copied()
                    .max()
                    .map_or(0, |m| m as usize + 1),
                nested: config.pb_posterior_nested_delta,
            };
            let res = crate::posterior::pb_gibbs::pb_gibbs_splice(&pb, &feat, &tracks, h, pcfg)?;
            write_back_splice(&varmap, &res, num_levels)?;
            splice_posterior = Some(res);
            None
        }
        Some(pcfg) => {
            let pb = stacked_pb_view(&varmap, &collapsed_levels, &cell_to_pb_per_level, h)?;
            let e_feat_map: Vec<f32> = cell_model.e_feat.flatten_all()?.to_vec1()?;
            let b_feat_map: Vec<f32> = cell_model.b_feat.to_vec1()?;
            let feat = crate::posterior::pb_index::FeatureSide {
                e_feat: &e_feat_map,
                b_feat: &b_feat_map,
                feature_to_backend_row: &unified.feature_to_backend_row,
            };
            // Free model ⇒ a feature row IS an anchor, so no grouping.
            let res = crate::posterior::pb_gibbs::pb_gibbs(&pb, &feat, None, h, pcfg)?;
            write_back_posterior(&varmap, &res, num_levels)?;
            Some(res)
        }
    };

    // Snapshot the deltaTopic β+δ into the `e_feat` field so phase-2 (and every
    // output/co-embed reader) sees a fixed materialized dictionary. No-op for a
    // free model.
    cell_model.materialize_e_feat()?;

    // The posterior's feature side lands HERE, on the materialized field, not on
    // the raw Var above — `E[z·β]` is already gated by its own selection, so
    // feeding it through `materialize_e_feat` would apply the trained gate a
    // second time. See `overwrite_feature_side`.
    //
    // JITTER SKIPS THIS. `mean_beta` is `E[z·β]` — already gated — so installing it as
    // the SGD starting point and then applying the mask on top would gate twice, the
    // exact double-application this comment warns about. Under jitter the posterior
    // contributes the SELECTION and SGD fits the loading from its own initialization.
    // The posterior gate is now UNCONDITIONAL: `--posterior` samples the selection and
    // SGD then fits the loading under it. It no longer replaces SGD.
    //
    // Measured on BM1, 3 seeds, paired: `pip ⊙ β` beat plain SGD on kNN label purity in
    // every seed (0.6739 ± 0.0074 vs 0.6664 ± 0.0068) with a 5.7x sparser dictionary
    // (1531 vs 8785 effective genes/dim). A STOCHASTIC `z ~ Bern(pip)` mask redrawn per
    // epoch was also tried and lost 3/3 — 0.6632, below plain SGD — so the mask is held
    // at its mean rather than sampled. Dropout's noise costs more here than its
    // decorrelation buys.
    let jitter = config.pb_posterior.is_some();
    if !jitter {
        if let Some(res) = pb_posterior.as_ref() {
            overwrite_feature_side(&mut cell_model, &res.mean_beta, &res.mean_b_feat, h)?;
        }
        if let Some(res) = splice_posterior.as_ref() {
            let rows = scatter_gene_to_rows(res, &config, unified.n_features(), h);
            overwrite_feature_side(&mut cell_model, &rows.0, &rows.1, h)?;
        }
    }

    /////////////////////////////////////////////////////////////////
    // Jitter: posterior-informed dropout, then SGD for the loading //
    /////////////////////////////////////////////////////////////////
    //
    // The sampler above produced `pip`; SGD now fits the loading under draws from it.
    //
    // Deliberately NOT Monte-Carlo EM: `pip` is estimated once and frozen, where MCEM
    // would re-estimate it against the updated `β` each outer round. So the selection
    // is fixed at what a cold-started chain saw, and cannot be refined by the loading
    // SGD goes on to find. That is the known limitation of this design, not an
    // oversight — buying it back costs another sampler run per round.
    if jitter && !stop.load(std::sync::atomic::Ordering::Relaxed) {
        let (pip, rows) = match (pb_posterior.as_ref(), splice_posterior.as_ref()) {
            // Free model: an anchor IS a feature row, so `pip` is already row-indexed.
            (Some(res), _) => (res.pip.clone(), unified.n_features()),
            // Factored: `beta_pip` is per GENE, which is the axis `s_beta` lives on —
            // `gathered_gate_weights` gathers rows from it via `row_to_gene`.
            (_, Some(res)) => {
                let n_genes = res.beta_pip.len() / h.max(1);
                (res.beta_pip.clone(), n_genes)
            }
            (None, None) => unreachable!("jitter requires a posterior"),
        };
        let nz = pip.iter().filter(|&&p| p <= 0.0).count();
        info!(
            "Jitter — SGD under posterior dropout: pip over {rows} unit(s) x {h} dim(s),              mean {:.3}, {} entrie(s) at exactly 0 (permanently masked; their loading              never trains and the dictionary carries exact zeros). z ~ Bern(pip) is              redrawn ONCE PER EPOCH, not per minibatch — z is a latent for the dataset.",
            pip.iter().sum::<f32>() / pip.len().max(1) as f32,
            nz,
        );
        cell_model.set_gate_pip(GateKind::Identity, &pip, rows, &config.device)?;
        // gem's SECOND gate gets its OWN table. Masking δ with β's inclusion would tie a
        // gene's motion to its identity selection — the conflation `GateKind` exists to
        // prevent. `delta_pip` is NaN where δ is unidentified (a gene missing one splice
        // track), and NaN must not reach the mask: an unidentified δ is exactly one that
        // should be masked OFF, so it maps to 0.
        // Scrubbed ONCE and reused by every model below: `delta_pip` is NaN where δ is
        // unidentified (a gene missing one splice track), and NaN must not reach a mask.
        // An unidentified δ is exactly one that should be OFF, so it maps to 0.
        let dpip: Option<Vec<f32>> = splice_posterior.as_ref().map(|res| {
            res.delta_pip
                .iter()
                .map(|p| if p.is_finite() { *p } else { 0.0 })
                .collect()
        });
        if let (Some(res), Some(dpip)) = (splice_posterior.as_ref(), dpip.as_ref()) {
            let n_unident = res.delta_pip.iter().filter(|p| !p.is_finite()).count();
            cell_model.set_gate_pip(GateKind::Velocity, dpip, rows, &config.device)?;
            info!(
                "Jitter — velocity gate: separate δ pip over {rows} gene(s), mean {:.3}; \
                 {n_unident} unidentified (masked off)",
                dpip.iter().sum::<f32>() / dpip.len().max(1) as f32,
            );
        }
        let cell_cell = cell_model.gate_mask_cell();
        for m in &mut level_models {
            m.set_gate_pip(GateKind::Identity, &pip, rows, &config.device)?;
            if let Some(dp) = dpip.as_ref() {
                m.set_gate_pip(GateKind::Velocity, dp, rows, &config.device)?;
            }
            m.share_gate_mask(&cell_cell);
        }

        let mut jitter_axes: Vec<CompositeAxis> = Vec::with_capacity(num_levels);
        for (i, model) in level_models.iter().enumerate() {
            let (axis, stratified) = &level_axes_data[i];
            jitter_axes.push(CompositeAxis {
                model,
                unified: &pb_blobs[i],
                cell_axis: axis,
                sampler: AxisSampler::Stratified(stratified),
                lambda: DEFAULT_AXIS_LAMBDA,
                label: "pb",
            });
        }
        let mut optj = AdamW::new(varmap.all_vars(), adamw_params())?;
        let mut pj = stage_params(&config);
        pj.composite_mode = CompositeMode::Sum;
        pj.epochs = config.epochs;
        info!(
            "Phase 1 (jitter) — features + {} pb level(s) [Sum], {} epochs",
            num_levels, pj.epochs
        );
        train_composite(
            &CompositeTrainContext {
                axes: &jitter_axes,
                dev: &config.device,
                stop: &stop,
                cell_to_pb_per_level: None,
                lineage_sem: None,
                lineage_sem_theta: None,
            },
            &mut optj,
            &pj,
        )?;
        drop(jitter_axes);
        // Back to the mean for everything downstream: training averaged over draws, so
        // `E[z ⊙ β] = pip ⊙ β` is the dictionary the fit actually implies. Leaving a
        // draw installed would ship ONE random sub-model as if it were the answer.
        cell_model.clear_gate_mask();
        // RE-MATERIALIZE. `materialize_e_feat` already ran above, BEFORE this training,
        // so `e_feat` is a snapshot of the pre-jitter Vars — and it is what phase 2
        // projects against and what the dictionary output writes. Without this the whole
        // jitter pass is invisible downstream: the cells get projected onto a dictionary
        // that does not match the parameters just fitted, which looks exactly like a
        // model that trained badly rather than one whose output was never refreshed.
        cell_model.materialize_e_feat()?;
    }

    // Lineage-DAG refine (gem β-sharing only; fixed velocity-KNN structure). The warm-up
    // phase 1 above yields a trained-enough dictionary: read pb-level velocity
    // (identity θ_pb + velocity δ_pb, reusing the phase-2 dual solver on the
    // already-batch-corrected pb aggregates), build a fixed velocity-oriented pb
    // graph, and run a SECOND phase-1 pass with the velocity-drift SEM residual on
    // so the shared E_feat picks up lineage geometry. The returned `pb_velocity` is
    // the FINAL readout (post-refine), consumed by the phase-2 cell lift. Flag off
    // or non-β-sharing ⇒ `None` and byte-identical training.
    let mut pb_velocity: Option<Vec<PbLevelVelocity>> = None;
    let mut refine_loss = 0f32; // final refine loss → QC likelihood-hygiene signal
    if config.lineage_dag && !stop.load(std::sync::atomic::Ordering::Relaxed) {
        match config.feat_factor.as_ref() {
            Some(spec) => {
                // Warm-up pb velocity, optionally smoothed + confidence-gated (①+②) so
                // `sign(δ_pb)` is stabilized before it orients the graph / SEM drift.
                let warmup_vel = maybe_smooth(
                    pb_velocity_readout(
                        &cell_model,
                        &pb_blobs,
                        &spec.unspliced_rows,
                        &config.device,
                    )?,
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
                // Share the `config.epochs` budget: the refine gets what the warm-up
                // (phase 1) did not, so `--lineage-dag` reallocates rather than doubles.
                p2.epochs = refine_epochs;

                // Fixed velocity-oriented KNN graph + velocity-drift SEM residual. The
                // dense KNN graph (each node → its velocity-forward neighbours), built
                // once from the warm-up readout, shapes E_feat in this single refine
                // pass; the cell-lift rebuilds the same graph from the final `pb_velocity`.
                let levels = lineage::build_pb_lineage(
                    &warmup_vel,
                    h,
                    lineage::DEFAULT_LINEAGE_KNN,
                    config.lineage_mst,
                );
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
                    "Lineage refine (velocity-KNN) = SECOND pass — {}/{} epochs (phase 1's \
                     remaining budget; SHARED with the warm-up, NOT a second full training). \
                     Baking lineage into E_feat: {} oriented pb edge(s) across {} level(s); \
                     velocity-drift SEM residual ON",
                    refine_epochs,
                    config.epochs,
                    n_edges,
                    levels.len()
                );
                // Second lineage term: the θ-pseudotime DAG.
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
                    },
                    &mut opt2,
                    &p2,
                )?;
                drop(refine_axes);

                // Refresh the dictionary and read the FINAL pb velocity (post-refine),
                // smoothed the same way so the cell-lift orients off a denoised field.
                cell_model.materialize_e_feat()?;
                pb_velocity = Some(maybe_smooth(
                    pb_velocity_readout(
                        &cell_model,
                        &pb_blobs,
                        &spec.unspliced_rows,
                        &config.device,
                    )?,
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
             stages (phase-2 projection, cell-lift) still run, so \
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
            config.joint_velocity,
        )?
    };

    // cell-lift: phase-2 cell-lineage lift (evaluation only). Runs on the FINAL pb
    // velocity readout + the now-projected per-cell identity θ_c. Integrate a pb
    // pseudotime/fate along the fixed velocity-oriented graph at the finest level, then
    // landmark-blend it to every cell. `None` when lineage-DAG is off or the readout is empty.
    // Not gated on `stop` — see the phase-2 note above.
    let mut lineage_qc: Option<LineageQc> = None;
    let cell_lineage = match &pb_velocity {
        Some(pbv) if !pbv.is_empty() => {
            let level = pbv.len() - 1; // finest level: densest landmark tiling
            let vel = &pbv[level];
            // Rebuild the velocity-oriented pb graph from the final readout — the same
            // fixed velocity-KNN the refine used to shape E_feat.
            let edges = lineage::build_pb_lineage(
                std::slice::from_ref(vel),
                h,
                lineage::DEFAULT_LINEAGE_KNN,
                config.lineage_mst,
            )
            .pop()
            .map(|l| {
                l.edges
                    .into_iter()
                    .map(|(i, j, w)| (i as usize, j as usize, w))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
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
        cell_lineage,
        lineage_qc,
        pb_posterior,
        splice_posterior,
    })
}

/// Stack every collapse level's **trained** pseudobulk embedding into one frozen
/// table, paired with that level's full-backend count matrix.
///
/// Phase 1 shapes `β` against exactly these axes — one per level, combined with
/// `CompositeMode::Sum` at uniform [`DEFAULT_AXIS_LAMBDA`] — and with the default
/// `phase1_cells_per_pb == 0` the cell axis is suppressed entirely, so this stack
/// *is* the objective `β` was fit under — which is what makes it the right frame for
/// the posterior sampler to condition on.
///
/// The pb Vars are read out of the `VarMap` by name rather than off
/// `level_models[l].e_cell`: the latter is a `Tensor` aliasing the `Var`'s
/// storage, and whether it tracks in-place `Var::set` updates is a candle
/// implementation detail.
///
/// Counts come from `mu_adjusted` when the collapse produced one, matching the
/// `pb_blobs` the model actually trained on, so the sampler sees the same scale the
/// fit did.
/// Splice-model write-back: `beta` and `delta` are the gene-side Vars under
/// β-sharing, not `e_feat` (which is a materialized snapshot rebuilt from them).
///
/// `delta` exists only when the model was built with a velocity offset; a run
/// without one simply has no Var to write and its `delta_mean` is all zeros.
fn write_back_splice(
    varmap: &VarMap,
    res: &crate::posterior::pb_gibbs::SpliceGibbsResult,
    num_levels: usize,
) -> anyhow::Result<()> {
    use candle_util::candle_core::{Device, Tensor};
    let vars = varmap.data().lock().expect("varmap poisoned");
    let set = |name: &str, values: &[f32]| -> anyhow::Result<bool> {
        let Some(var) = vars.get(name) else {
            return Ok(false);
        };
        anyhow::ensure!(
            var.elem_count() == values.len(),
            "splice write-back for {name}: have {} values, var holds {}",
            values.len(),
            var.elem_count()
        );
        let rows = values.len() / res.h.max(1);
        let t = Tensor::from_slice(values, (rows, res.h), &Device::Cpu)?.to_device(var.device())?;
        var.set(&t)?;
        Ok(true)
    };
    anyhow::ensure!(
        set("beta", &res.beta_mean)?,
        "splice write-back: the model has no `beta` var"
    );
    if !set("delta", &res.delta_mean)? {
        log::debug!("splice write-back: no `delta` var (velocity off) — β only");
    }
    drop(vars);
    write_back_pb_levels(varmap, &res.mean_pb, &res.mean_b_pb, res.h, num_levels)
}

/// Write the **pseudobulk** half of the phase-1 posterior back into the `VarMap`.
///
/// Written through the `VarMap` by name, not through `level_models[l].e_cell`:
/// those are `Tensor`s aliasing the `Var`s' storage, and whether they observe an
/// in-place `Var::set` is a candle implementation detail — the same reason
/// [`stacked_pb_view`] reads by name.
///
/// The FEATURE half is deliberately not written here; see
/// [`overwrite_feature_side`] for why it has to wait until after
/// `materialize_e_feat`.
fn write_back_posterior(
    varmap: &VarMap,
    res: &crate::posterior::pb_gibbs::PbGibbsResult,
    num_levels: usize,
) -> anyhow::Result<()> {
    write_back_pb_levels(varmap, &res.mean_pb, &res.mean_b_pb, res.h, num_levels)
}

/// The per-row phase-1 MAP for a β-sharing model, read from the trained Vars:
/// `β_g` on a spliced row, `β_g + δ_g` on an unspliced one.
///
/// This exists because `cell_model.e_feat` is NOT the fit on this path — the
/// factored loss trains `beta`/`delta` and `e_feat` stays at its random
/// initialisation until `materialize_e_feat`, which runs after the posterior.
/// Reading it there would warm-start the sampler from noise.
fn materialized_splice_rows(
    varmap: &VarMap,
    spec: &FeatFactorSpec,
    n_features: usize,
    h: usize,
) -> anyhow::Result<Vec<f32>> {
    let vars = varmap.data().lock().expect("varmap poisoned");
    let get = |name: &str| -> anyhow::Result<Option<Vec<f32>>> {
        match vars.get(name) {
            None => Ok(None),
            Some(v) => Ok(Some(v.as_tensor().flatten_all()?.to_vec1::<f32>()?)),
        }
    };
    let beta = get("beta")?
        .ok_or_else(|| anyhow::anyhow!("factored model has no `beta` var to warm-start from"))?;
    let delta = get("delta")?;
    let mut out = vec![0f32; n_features * h];
    for (uid, (&g, &unspliced)) in spec
        .row_to_gene
        .iter()
        .zip(&spec.unspliced_rows)
        .enumerate()
    {
        if g == u32::MAX || uid >= n_features {
            continue;
        }
        let (src, dst) = (g as usize * h, uid * h);
        for k in 0..h {
            let d = if unspliced {
                delta.as_ref().map_or(0.0, |v| v[src + k])
            } else {
                0.0
            };
            out[dst + k] = beta[src + k] + d;
        }
    }
    Ok(out)
}

/// Expand gem's per-GENE splice posterior onto the per-ROW feature axis the
/// materialized dictionary lives on: `β_g` on a spliced row, `β_g + δ_g` on an
/// unspliced one, and each row's own profiled intercept.
///
/// Genes whose `δ` is not identified contribute `β_g` alone — a prior draw has no
/// business entering the dictionary just because it has the right shape.
fn scatter_gene_to_rows(
    res: &crate::posterior::pb_gibbs::SpliceGibbsResult,
    config: &FitConfig,
    n_features: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut e = vec![0f32; n_features * h];
    let mut b = vec![0f32; n_features];
    let Some(spec) = config.feat_factor.as_ref() else {
        return (e, b);
    };
    for (uid, (&g, &unspliced)) in spec
        .row_to_gene
        .iter()
        .zip(&spec.unspliced_rows)
        .enumerate()
    {
        if g == u32::MAX || uid >= n_features {
            continue;
        }
        let (src, dst) = (g as usize * h, uid * h);
        let use_delta = unspliced
            && res
                .delta_identified
                .get(g as usize)
                .copied()
                .unwrap_or(false);
        for k in 0..h {
            e[dst + k] = res.beta_mean[src + k]
                + if use_delta {
                    res.delta_mean[src + k]
                } else {
                    0.0
                };
        }
        b[uid] = res.mean_b_feat.get(g as usize).copied().unwrap_or(0.0);
    }
    (e, b)
}

/// Replace the materialized dictionary with the posterior mean, AFTER the gate has
/// been baked in.
///
/// This cannot be a `Var::set` on `"e_feat"` before `materialize_e_feat`. For a
/// free + gated model — which is every `senna bge` run — `e_feat_raw` aliases that
/// Var and holds the **ungated** loading; `materialize_e_feat` then writes
/// `gate ⊙ raw` into the `e_feat` field. The posterior mean is `E[z·β]`, already
/// gated by its own selection, so putting it in the raw slot makes the shipped
/// dictionary `gate² ⊙ β`: every partially down-weighted gene shrunk quadratically,
/// no error, and identical shapes throughout.
///
/// The bias moves with it. `multinomial_ll` profiles the intercept out, so the
/// sampled loading is only identified alongside `b_a*`; leaving the NCE-fitted bias
/// in place pairs a sampled embedding with an intercept from a different objective.
fn overwrite_feature_side(
    model: &mut JointEmbedModel,
    e_feat: &[f32],
    b_feat: &[f32],
    h: usize,
) -> anyhow::Result<()> {
    use candle_util::candle_core::{Device, Tensor};
    let dev = model.e_feat.device().clone();
    let rows = e_feat.len() / h.max(1);
    anyhow::ensure!(
        model.e_feat.elem_count() == e_feat.len() && model.b_feat.elem_count() == b_feat.len(),
        "posterior feature write-back: {} loadings / {} biases against a {}×{} dictionary",
        e_feat.len(),
        b_feat.len(),
        model.e_feat.elem_count() / h.max(1),
        h
    );
    model.e_feat = Tensor::from_slice(e_feat, (rows, h), &Device::Cpu)?.to_device(&dev)?;
    model.b_feat = Tensor::from_slice(b_feat, rows, &Device::Cpu)?.to_device(&dev)?;
    Ok(())
}

/// Slice the stacked pb posterior mean back into each level's own Var.
///
/// Shared by both feature-side parameterizations: the pb heads are the same
/// objects either way, and the stacked axis they came from is level-ordered, so
/// consuming it exactly is also the check that the level sizes still agree.
fn write_back_pb_levels(
    varmap: &VarMap,
    mean_pb: &[f32],
    mean_b_pb: &[f32],
    h: usize,
    num_levels: usize,
) -> anyhow::Result<()> {
    use candle_util::candle_core::{Device, Tensor};
    let vars = varmap.data().lock().expect("varmap poisoned");
    let mut off = 0usize;
    for level in 0..num_levels {
        let name = format!("pb_l{level}_e_cell");
        let Some(var) = vars.get(&name) else { continue };
        let n_pb = var.elem_count() / h.max(1);
        let end = off + n_pb * h;
        anyhow::ensure!(
            end <= mean_pb.len(),
            "pb write-back for {name}: stacked axis holds {} values, level needs {end}",
            mean_pb.len()
        );
        let t = Tensor::from_slice(&mean_pb[off..end], (n_pb, h), &Device::Cpu)?
            .to_device(var.device())?;
        var.set(&t)?;
        // The paired bias moves with the loading: the profile likelihood
        // maximised it out, so they are only identified together.
        let bias_name = format!("pb_l{level}_b_cell");
        if let Some(bvar) = vars.get(&bias_name) {
            let (bs, be) = (off / h, end / h);
            anyhow::ensure!(
                bvar.elem_count() == be - bs,
                "pb bias write-back for {bias_name}: var holds {}, level needs {}",
                bvar.elem_count(),
                be - bs
            );
            let bt = Tensor::from_slice(&mean_b_pb[bs..be], be - bs, &Device::Cpu)?
                .to_device(bvar.device())?;
            bvar.set(&bt)?;
        }
        off = end;
    }
    anyhow::ensure!(
        off == mean_pb.len(),
        "pb write-back consumed {off} of {} stacked values — level sizes disagree",
        mean_pb.len()
    );
    Ok(())
}

pub(crate) fn stacked_pb_view<'a>(
    varmap: &VarMap,
    collapsed_levels: &'a [data_beans_alg::collapse_data::CollapsedOut],
    cell_to_pb_per_level: &[Vec<usize>],
    h: usize,
) -> anyhow::Result<stacked_pb::StackedPb<'a>> {
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
    Ok(stacked_pb::StackedPb {
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
    dev: &candle_util::candle_core::Device,
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
        dev,
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
