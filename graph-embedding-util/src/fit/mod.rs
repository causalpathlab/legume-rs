//! Public entry point for `graph-embedding`. Callers translate their
//! own CLI args into a [`FitConfig`] and pass already-loaded
//! [`UnifiedData`] (so this crate stays free of file/path concerns).

mod axes;
pub mod batch_fold;
mod config;
pub mod lift;
pub mod lineage;
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
    FeatFactorSpec, FitConfig, FitOutput, GeneModuleConfig, GenePairConfig, ParentModulesOwned,
};
pub use lift::{CellLineage, LineageQc};
pub use module_args::GeneModuleArgs;
pub use module_warm::{parent_module_logits, warm_start_module_labels};
pub use pb_readout::{majority_batch_per_pb, PbLevelEmbedding};
pub use projection::PbLevelVelocity;
pub use resolve_embedding::{train_rest, RestConfig, RestTrainInputs, TrainedRest};

use crate::data::UnifiedData;
use crate::loss::{GenePairSampler, PerBatchStratifiedCellSampler};
use crate::model::JointEmbedModel;
use crate::training::{train_composite, CompositeTrainContext, GenePairAxis, PbSemTerm};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use log::info;
use matrix_param::traits::Inference;
use nalgebra::DMatrix;

use config::{stage_params, LINEAGE_WARMUP_FRAC};
use matrix_util::traits::ConvertMatOps;
use projection::{project_cells_phase2, project_pbs_phase2, CellBatchFold, PHASE2_RIDGE};
pub use projection::{
    FrozenProjection, FrozenProjectionArgs, FrozenProjector, PHASE2_RIDGE as PROJECTION_RIDGE_SGD,
};

/// Composite-objective gbe fit — trained in **two phases**.
///
/// The bilinear score is `E_feat[f]·E_cell[c] + b_feat[f] + b_cell[c]` —
/// the per-cell bias `b_cell` absorbs library size (consistent with
/// `senna gem`).
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
    let pb = setup::build_pseudobulks(unified, &config)?;
    let num_levels = pb.num_levels();
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
    // Module warm start: k-means over the feature profiles at the finest collapse
    // level, on the same batch-corrected pseudobulk counts phase 1 trains on.
    // Either the k-means labels over this fit's own profiles, or — under a parent
    // (`senna update`) — explicit logits carrying the parent's membership for the
    // matched features and initializing the rest through the parent's modules.
    let module_warm: Option<models::ModuleWarm> = config.gene_modules.as_ref().map(|g| {
        let finest = collapsed_levels.last().expect("at least one level");
        let pb_full = match &finest.mu_adjusted {
            Some(adj) => adj.posterior_mean(),
            None => finest.mu_observed.posterior_mean(),
        };
        let profile = setup::gather_to_unified_axis(pb_full, n_features, &feature_to_backend);
        match &g.parent {
            Some(parent) => models::ModuleWarm::Parent {
                logits: module_warm::parent_module_logits(parent, &profile),
                mu: parent.mu.clone(),
            },
            None => models::ModuleWarm::Labels(module_warm::warm_start_module_labels(
                &profile,
                g.n_modules,
                config.seed,
            )),
        }
    });
    let models::Heads {
        mut cell_model,
        level_models,
    } = models::build_heads(unified, &pb_blobs, &config, module_warm.as_ref(), &varmap)?;

    ////////////////////////////////
    // Composite axes and trainer //
    ////////////////////////////////
    let ax = axes::build_axis_data(unified, &pb_blobs, &cell_to_pb_per_level, &config)?;
    let (use_cell_axis, cell_samplers) = (ax.use_cell_axis, &ax.cell_samplers);

    // Gene-gene co-occurrence edges: positives are two genes of one cell, drawn
    // from the FULL cell samplers (every cell is a co-occurrence context, whatever
    // phase 1's cell axis keeps); negatives come from a cell under a shared
    // collapse ancestor a random number of hops up.
    let gene_pair_sampler: Option<GenePairSampler> = match config.gene_pairs.as_ref() {
        Some(gp) => {
            let s = GenePairSampler::new(cell_samplers, &cell_to_pb_per_level, &gp.hops)?;
            info!(
                "Gene-pair edges ON: {} negatives/pair, λ={}, {} hops with mass {:?}, {} finest \
                 groups",
                gp.n_negatives,
                gp.lambda,
                s.n_hops(),
                s.hop_weights(),
                s.tree().n_finest_groups(),
            );
            Some(s)
        }
        None => None,
    };

    // Note on biases: the per-CELL bias `b_cell` and the per-PB biases
    // (`pb_l*_b_cell`) BOTH train in phase 1 — a per-sample bias absorbs
    // that sample's depth so the shared `E_feat` captures composition, not
    // library size. `b_cell` is re-fitted analytically in phase 2 and
    // written alongside `e_cell` (consistent with `senna gem`).

    // Two-phase training (always — `ge::fit` is the bge driver only); see
    // the `fit()` doc for the rationale. Shared AdamW hyperparameters:
    let adamw_params = || ParamsAdamW {
        lr: config.learning_rate,
        weight_decay: config.weight_decay,
        ..Default::default()
    };

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
    let lineage_on = config.lineage_dag && config.feat_factor.is_some();
    let warmup_epochs = if lineage_on && config.epochs > 0 {
        ((LINEAGE_WARMUP_FRAC * config.epochs as f64).round() as usize).clamp(1, config.epochs)
    } else {
        config.epochs
    };
    let refine_epochs = config.epochs - warmup_epochs;
    {
        let joint_axes = ax.composite_axes(&cell_model, &level_models, unified, &pb_blobs);
        let mut opt1 = AdamW::new(varmap.all_vars(), adamw_params())?;
        let mut p1 = stage_params(&config);
        p1.epochs = warmup_epochs;
        let cell_prefix = if use_cell_axis { "cell + " } else { "" };
        let n_pb_levels = num_levels;
        if refine_epochs > 0 {
            info!(
                "Phase 1 (joint) = LINEAGE WARM-UP — {}/{} epochs; the DAG refine gets the other \
                 {} (ONE shared epoch budget, NOT doubled). Training features + {}{} pb \
                 level(s)",
                warmup_epochs, config.epochs, refine_epochs, cell_prefix, n_pb_levels,
            );
        } else {
            info!(
                "Phase 1 (joint) — features + {}{} pb level(s), {} epochs",
                cell_prefix, n_pb_levels, warmup_epochs,
            );
        }
        let gp_axis = gene_pair_axis(
            &config,
            gene_pair_sampler.as_ref(),
            cell_samplers,
            &cell_model,
        );
        train_composite(
            &CompositeTrainContext {
                axes: &joint_axes,
                dev: &config.device,
                stop: &stop,
                cell_to_pb_per_level: None,
                gene_pairs: gp_axis.as_ref(),
                lineage_sem: None,
                lineage_sem_theta: None,
            },
            &mut opt1,
            &p1,
        )?;
    }
    // `cell_axis` / `pb_axes` borrows of `cell_model` / `cell_samplers` end
    // here, freeing them for the phase-2 `&mut` projection below.

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

                // The SAME axis set the warm-up trained — that identity is the point
                // of the refine, which differs only by its SEM term.
                let refine_axes = ax.composite_axes(&cell_model, &level_models, unified, &pb_blobs);
                let mut p2 = stage_params(&config);
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
                // The SAME gene-pair term as the warm-up, for the same reason as the axes.
                let gp_axis = gene_pair_axis(
                    &config,
                    gene_pair_sampler.as_ref(),
                    cell_samplers,
                    &cell_model,
                );
                refine_loss = train_composite(
                    &CompositeTrainContext {
                        axes: &refine_axes,
                        dev: &config.device,
                        stop: &stop,
                        cell_to_pb_per_level: None,
                        gene_pairs: gp_axis.as_ref(),
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
    // especially: below `--phase1-cells-per-pb n_cells` most cells never train in
    // phase 1, so their `e_cell` rows are still randn init until phase 2 runs —
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
        // Phase-2 batch correction: divide each cell's counts by its batch's
        // per-gene fold `δ` from the finest collapse (see `batch_fold`), so the
        // solve runs in the batch-free frame the dictionary was trained in. `None`
        // on single-batch data.
        let batch_fold: Option<CellBatchFold> = batch_gene_fold
            .as_ref()
            .map(|f| f.cell_fold(&unified.batch_membership));

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
            cell_samplers,
            n_cells,
            f64::from(PHASE2_RIDGE),
            &config.device,
            batch_fold,
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
        batch_gene_fold,
        model: cell_model,
        finest_collapse,
        varmap,
        cell_nrms: phase2.cell_nrms,
        cell_velocity: phase2.velocity,
        pb_velocity,
        cell_lineage,
        lineage_qc,
        pb_embeddings,
    })
}

/// The gene-pair term for one training pass, or `None` when the config has no
/// gene edges. Borrows are taken per pass so the warm-up and the lineage refine
/// train the same term the same way the axes are rebuilt per pass.
fn gene_pair_axis<'a>(
    config: &'a FitConfig,
    sampler: Option<&'a GenePairSampler>,
    cell_samplers: &'a [PerBatchStratifiedCellSampler],
    model: &'a JointEmbedModel,
) -> Option<GenePairAxis<'a>> {
    let gp = config.gene_pairs.as_ref()?;
    let sampler = sampler?;
    Some(GenePairAxis {
        model,
        sampler,
        cell_samplers,
        n_negatives: gp.n_negatives,
        lambda: gp.lambda,
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
