//! Public entry point for `graph-embedding`. Callers translate their
//! own CLI args into a [`FitConfig`] and pass already-loaded
//! [`UnifiedData`] (so this crate stays free of file/path concerns).

use crate::coarsen::identity_axis;
use crate::data::UnifiedData;
use crate::feature_network::FeatureNetworkSmoother;
use crate::loss::{build_per_batch_cell_samplers, build_per_batch_samplers, PerBatchSampler};
use crate::model::{JointEmbedModel, ModelArgs, ModelInit, WarmStartArgs};
use crate::stop::setup_stop_handler;
use crate::training::{train, CellCellTraining, TrainingContext, TrainingParams};
use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans_alg::collapse_data::{MultilevelCollapsingOps, MultilevelParams};
use data_beans_alg::gene_weighting::{
    compute_nb_fisher_weights, load_per_gene_weights, save_per_gene_weights,
};
use data_beans_alg::random_projection::RandProjOps;
use log::{info, warn};
use matrix_param::traits::Inference;
use matrix_util::pair_graph::FeaturePairGraph;
use nalgebra::DMatrix;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Hyperparameter / configuration bundle for [`fit`]. Constructed by
/// each caller from its own CLI arguments — this crate doesn't import
/// `clap`.
pub struct FitConfig {
    pub embedding_dim: usize,
    pub num_coarsen_seeds: usize,
    pub super_cells: usize,
    pub sketch_dim: usize,
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub num_negatives: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub device: Device,
    /// Streaming block size for the per-feature NB-Fisher pass and
    /// other column-block I/O. `None` falls back to
    /// `matrix_util::utils::default_block_size(n_features)` which
    /// clamps to 100 for large feature counts — that's tiny on
    /// rotational disks. Pass `Some(1024)` or higher when you have
    /// the RAM, especially without `--preload-data`.
    pub block_size: Option<usize>,
    /// Optional path for caching the NB-Fisher per-gene weights as
    /// parquet (one row per gene). When set, [`fit`] looks for an
    /// existing file and reuses it (skipping the streaming pass)
    /// when its gene names line up with the unified feature axis;
    /// otherwise it computes the weights and writes them to this
    /// path so subsequent runs are instant. `None` always recomputes.
    pub fisher_weights_cache: Option<Box<str>>,
    pub feature_network: Option<FeatureNetworkConfig>,
    /// Optional cell-cell loss term — positive cell pairs from a
    /// caller-provided graph (e.g. spatial KNN), with negatives drawn
    /// within each pair's batch. Combined with the bipartite loss
    /// additively: `L = L_bip + λ · L_cc`. `None` (or `lambda == 0`)
    /// disables the term.
    pub cell_cell: Option<CellCellConfig>,
    /// Optional caller-provided stop flag (so a single SIGINT handler
    /// can be shared with surrounding orchestration). When `None`,
    /// [`fit`] installs its own. Callers running `fit` more than once
    /// in the same process MUST pass `Some(...)` with a single shared
    /// flag — `ctrlc::set_handler` panics on the second registration.
    pub stop: Option<Arc<AtomicBool>>,
}

/// Cell-cell NCE configuration. Positives = neighbor pairs from a
/// caller-provided graph; negatives = within-batch random non-neighbors.
pub struct CellCellConfig {
    /// Positive cell pairs as global cell ids (canonical (i, j) with i < j).
    pub edges: Vec<(u32, u32)>,
    /// Loss mixing weight λ. 0.0 → bipartite-only (term skipped);
    /// 1.0 → equal weight with the bipartite loss; > 1 emphasizes cell-cell.
    pub lambda: f32,
    /// Negative cells per positive pair.
    pub n_negatives: usize,
}

/// Optional SGC feature-network smoother configuration. The graph is
/// already loaded and aligned to the model's feature axis — caller
/// owns the file → graph translation.
pub struct FeatureNetworkConfig {
    pub graph: FeaturePairGraph,
    pub k_hops: usize,
    pub alpha: f32,
    pub refresh_epochs: usize,
}

/// CLI-flag-shaped helper that resolves an edge-list file against the
/// unified feature axis and packages it as a [`FeatureNetworkConfig`].
/// Both `senna gbe` and `pinto gbe` reach this with identical args, so
/// the resolution + zero-edge guard live here to keep error messages
/// consistent.
pub struct FeatureNetworkArgs<'a> {
    pub path: &'a str,
    pub feature_names: &'a [Box<str>],
    pub prefix_match: bool,
    pub delim: Option<char>,
    pub k_hops: usize,
    pub alpha: f32,
    pub refresh_epochs: usize,
}

/// Load NB-Fisher per-gene weights from the cache parquet when its
/// gene names match the unified feature axis byte-for-byte; otherwise
/// stream-compute them and (if a cache path was provided) save them
/// for next time. Caches that don't match are warned about and
/// recomputed (typical cause: feature axis changed because of a
/// different `--feature-name-delim` or a different file set).
fn load_or_compute_fisher_weights(
    unified: &UnifiedData,
    block_size: Option<usize>,
    cache_path: Option<&str>,
) -> anyhow::Result<Vec<f32>> {
    if let Some(path) = cache_path {
        // An earlier run that bombed mid-save can leave a 0-byte stub;
        // skip those so we don't spam a confusing parquet-EOF warning.
        let usable = std::fs::metadata(path)
            .map(|m| m.len() > 0)
            .unwrap_or(false);
        if usable {
            match load_per_gene_weights(path) {
                Ok((cached_names, cached_weights))
                    if cached_names == unified.feature_names
                        && cached_weights.len() == unified.n_features() =>
                {
                    info!("Reusing cached NB-Fisher weights from {}", path);
                    return Ok(cached_weights);
                }
                Ok((cached_names, _)) => {
                    warn!(
                        "Fisher cache {} is stale ({} cached genes vs {} unified) — recomputing",
                        path,
                        cached_names.len(),
                        unified.n_features()
                    );
                }
                Err(e) => warn!("Failed to load Fisher cache {} ({}); recomputing", path, e),
            }
        }
    }

    info!(
        "Computing NB-Fisher weights (block_size={:?})...",
        block_size
    );
    let mut full_weights: Vec<f32> = Vec::new();
    for (i, data) in unified.per_file_data.iter().enumerate() {
        let w = compute_nb_fisher_weights(data, block_size)?;
        info!(
            "  backend {}: {} features, mean Fisher weight {:.3}",
            i,
            w.len(),
            w.iter().sum::<f32>() / w.len().max(1) as f32
        );
        full_weights.extend(w);
    }
    // Subset to the (possibly HVG-reduced) compact feature axis so the
    // returned vec is aligned 1:1 with `unified.feature_names`.
    let feat_weights: Vec<f32> = unified
        .feature_to_backend_row
        .iter()
        .map(|&i| full_weights[i])
        .collect();

    if let Some(path) = cache_path {
        if let Err(e) = save_per_gene_weights(&feat_weights, &unified.feature_names, path) {
            warn!("Failed to save Fisher cache to {}: {}", path, e);
        } else {
            info!("Saved NB-Fisher weights to {}", path);
        }
    }

    Ok(feat_weights)
}

pub fn load_feature_network(args: FeatureNetworkArgs) -> anyhow::Result<FeatureNetworkConfig> {
    info!("Loading feature network from {}...", args.path);
    let graph = FeaturePairGraph::from_edge_list(
        args.path,
        args.feature_names.to_vec(),
        args.prefix_match,
        args.delim,
    )?;
    if graph.num_edges() == 0 {
        anyhow::bail!(
            "Feature network has 0 matched edges — check name resolution \
             (--feature-network-delim / --feature-network-prefix-match)."
        );
    }
    Ok(FeatureNetworkConfig {
        graph,
        k_hops: args.k_hops,
        alpha: args.alpha,
        refresh_epochs: args.refresh_epochs,
    })
}

/// Trained model + its `VarMap`. The varmap is exposed so callers can
/// save checkpoints or re-run inference; current callers (`senna gbe`,
/// `pinto gbe`) only consume `model`, so it sits unused but kept alive.
pub struct FitOutput {
    pub model: JointEmbedModel,
    pub varmap: VarMap,
}

/// Two-stage gbe fit:
///
/// **Stage 1** trains `E_feat` and `b_feat` on batch-corrected
/// pseudobulks built via `collapse_columns_multilevel_vec` (the same
/// pipeline `senna topic` uses). The "cells" of stage 1 are the
/// super-cells, so `E_pseudobulk` and `b_pseudobulk` come out
/// alongside; they're discarded after stage 1 except as warm-start
/// values for stage 2.
///
/// **Stage 2** trains `E_cell` (n_cells × H) and `b_cell` (n_cells)
/// against fine-resolution per-cell triplets, with `E_feat` /
/// `b_feat` from stage 1 **frozen** (held as plain `Tensor`s outside
/// the `VarMap` so AdamW doesn't update them). `E_cell[c]` is
/// initialized from `E_pseudobulk[cell→pb_map[c]]` so stage 2 starts
/// from a sensible biological location instead of randn.
///
/// Both stages share the existing NCE training loop. The cell-cell
/// loss term and SGC feature-network smoother only attach in their
/// natural stage (cell-cell in stage 2 because it operates on real
/// per-cell pairs; smoother in stage 1 because that's when `E_feat`
/// is being learned).
pub fn fit(unified: &mut UnifiedData, mut config: FitConfig) -> anyhow::Result<FitOutput> {
    let n_cells = unified.n_cells();
    let n_features = unified.n_features();
    let h = config.embedding_dim;
    let alpha_neg = 0.75_f32;
    let stop = config.stop.take().unwrap_or_else(setup_stop_handler);

    // ---- Shared upstream: batch-corrected projection + Fisher weights ----
    info!(
        "Batch-corrected projection (sketch_dim={}, {} batches)...",
        config.sketch_dim,
        unified.n_batches()
    );
    let batch_labels: Vec<Box<str>> = unified
        .batch_membership
        .iter()
        .map(|&b| unified.batch_names[b as usize].clone())
        .collect();
    let proj_out = unified.per_file_data[0].project_columns_with_batch_correction(
        config.sketch_dim,
        config.block_size,
        (unified.n_batches() > 1).then_some(batch_labels.as_slice()),
    )?;

    let feat_weights = load_or_compute_fisher_weights(
        unified,
        config.block_size,
        config.fisher_weights_cache.as_deref(),
    )?;

    // ---- Multilevel collapse → batch-corrected pseudobulks ----
    info!(
        "Multilevel collapse (target {} super-cells, {} levels)...",
        config.super_cells, config.num_coarsen_seeds
    );
    let mut collapsed_levels = unified.per_file_data[0].collapse_columns_multilevel_vec(
        &proj_out.proj,
        &batch_labels,
        &MultilevelParams {
            knn_super_cells: 10,
            num_levels: config.num_coarsen_seeds.max(1),
            sort_dim: config.sketch_dim.min(12),
            num_opt_iter: 100,
            refine: None,
        },
    )?;
    // After this reverse, levels are ordered coarsest..finest. Senna
    // topic uses the same order so the curriculum trains coarse first.
    collapsed_levels.reverse();
    let num_levels = collapsed_levels.len();

    // cell → super-cell membership stored on the SparseIoVec by the
    // last level processed (= finest after reverse).
    let cell_to_pb: Vec<usize> = unified.per_file_data[0].get_group_membership(0..n_cells)?;

    // ============== STAGE 1: progressive curriculum across levels ==============
    // Coarsest level gets the most epochs (more cells per pseudobulk →
    // lower variance → cleaner E_feat init), finest gets the fewest
    // (refinement on biological detail). Mirrors senna topic's
    // `compute_level_epochs(total, num_levels)` weighting.
    let level_epochs = compute_level_epochs(config.epochs, num_levels);
    info!(
        "Stage 1 curriculum: {} levels, per-level epochs {:?}",
        num_levels, level_epochs
    );

    let mut e_feat_init: Option<DMatrix<f32>> = None;
    let mut b_feat_init: Vec<f32> = vec![0f32; n_features];
    let mut last_e_pb: DMatrix<f32> = DMatrix::zeros(0, h);
    let mut last_b_pb: Vec<f32> = Vec::new();
    let mut last_n_pb: usize = 0;

    // Smoother is built once and reused across levels — its frozen
    // residual refreshes every `refresh_epochs` epochs anyway, so it
    // self-realigns to the current E_feat after each level transition.
    let mut stage1_smoother = build_smoother(config.feature_network.take(), n_features, h)?;

    for (level_idx, collapsed) in collapsed_levels.iter().enumerate() {
        let pb_full: &DMatrix<f32> = match &collapsed.mu_adjusted {
            Some(adj) => adj.posterior_mean(),
            None => collapsed.mu_observed.posterior_mean(),
        };
        let n_pb = pb_full.ncols();
        let pb_count_ds: DMatrix<f32> = if pb_full.nrows() == n_features {
            pb_full.clone()
        } else {
            let mut subset = DMatrix::<f32>::zeros(n_features, n_pb);
            for (new_i, &old_i) in unified.feature_to_backend_row.iter().enumerate() {
                for s in 0..n_pb {
                    subset[(new_i, s)] = pb_full[(old_i, s)];
                }
            }
            subset
        };

        let pb_unified = UnifiedData::from_pseudobulks(
            &pb_count_ds,
            unified.feature_names.clone(),
            unified.feature_to_backend_row.clone(),
        )?;

        let stage1_varmap = VarMap::new();
        let stage1_vs = VarBuilder::from_varmap(
            &stage1_varmap,
            candle_util::candle_core::DType::F32,
            &config.device,
        );
        let zeros_pb = vec![0f32; n_pb];
        let stage1_model = JointEmbedModel::new_with_init(
            ModelArgs {
                n_features,
                n_cells: n_pb,
                embedding_dim: h,
            },
            &ModelInit {
                e_feat: e_feat_init.as_ref(),
                e_cell: None, // pseudobulk embedding starts random each level (size differs)
                b_feat: &b_feat_init,
                b_cell: &zeros_pb,
            },
            &stage1_varmap,
            stage1_vs,
            &config.device,
        )?;
        let mut stage1_opt = AdamW::new(
            stage1_varmap.all_vars(),
            ParamsAdamW {
                lr: config.learning_rate,
                ..Default::default()
            },
        )?;

        let stage1_axis = identity_axis(n_pb);
        let pb_samplers = build_active_samplers(&pb_unified, &feat_weights, alpha_neg)?;

        let epochs_this_level = level_epochs[level_idx];
        info!(
            "Stage 1 level {}/{}: {} pseudobulks × {} features, {} epochs",
            level_idx + 1,
            num_levels,
            n_pb,
            n_features,
            epochs_this_level,
        );
        let mut stage1_params = stage_params(&config);
        stage1_params.epochs = epochs_this_level;
        let stage1_ctx = TrainingContext {
            unified: &pb_unified,
            cell_axis: &stage1_axis,
            feat_weights: &feat_weights,
            batch_samplers: &pb_samplers,
            cell_cell: None,
            dev: &config.device,
            stop: &stop,
        };
        train(
            &stage1_model,
            &mut stage1_opt,
            &stage1_ctx,
            &stage1_params,
            stage1_smoother.as_mut(),
        )?;

        // Hand over to next level / stage 2.
        e_feat_init = Some(tensor_to_mat(&stage1_model.e_feat, n_features, h)?);
        b_feat_init = stage1_model.b_feat.to_vec1::<f32>()?;
        last_e_pb = tensor_to_mat(&stage1_model.e_cell, n_pb, h)?;
        last_b_pb = stage1_model.b_cell.to_vec1::<f32>()?;
        last_n_pb = n_pb;

        if stop.load(std::sync::atomic::Ordering::SeqCst) {
            warn!("Stop requested mid-curriculum at level {}; advancing to stage 2 with whatever is trained.", level_idx + 1);
            break;
        }
    }

    // The finest level's `cell_to_pb` (captured pre-loop) drives stage-2
    // init. If we broke out early, we still use the same finest mapping
    // — only the trained `e_pb` is from the level we made it to.
    let n_pb = last_n_pb;
    debug_assert!(n_pb > 0, "stage 1 produced no pseudobulks");
    let frozen_e_feat = e_feat_init
        .as_ref()
        .expect("stage 1 trained at least one level")
        .clone();
    let _frozen_b_feat_host = &b_feat_init;

    // Re-materialize the trained E_feat / b_feat as device tensors so
    // stage 2 holds them as plain (frozen) Tensors, not Vars.
    let frozen_e_feat = mat_to_tensor(&frozen_e_feat, &config.device)?.detach();
    let frozen_b_feat = Tensor::from_vec(b_feat_init.clone(), n_features, &config.device)?.detach();

    // E_cell[c] = E_pseudobulk[cell_to_pb[c]]. The mapping comes from
    // the FINEST level (set on the SparseIoVec by collapse). When the
    // curriculum runs to completion, `last_e_pb` is from that same
    // finest level so indices line up. If SIGINT broke us out at an
    // earlier level, `last_e_pb` has fewer rows than the finest cell→pb
    // map references — bail with a clear error rather than silently
    // remapping cells to wrong pseudobulks.
    let max_pb_id = cell_to_pb.iter().copied().max().unwrap_or(0);
    anyhow::ensure!(
        max_pb_id < last_n_pb,
        "stage 1 exited with {} pseudobulks but the finest cell→pb map references id {}; \
         likely SIGINT mid-curriculum. Re-run without interruption.",
        last_n_pb,
        max_pb_id
    );
    let mut e_cell_init: DMatrix<f32> = DMatrix::zeros(n_cells, h);
    for (c, &pb) in cell_to_pb.iter().enumerate() {
        for d in 0..h {
            e_cell_init[(c, d)] = last_e_pb[(pb, d)];
        }
    }
    let b_cell_init: Vec<f32> = cell_to_pb.iter().map(|&pb| last_b_pb[pb]).collect();

    // ============== STAGE 2: train E_cell with frozen features ==============
    let stage2_varmap = VarMap::new();
    let stage2_vs = VarBuilder::from_varmap(
        &stage2_varmap,
        candle_util::candle_core::DType::F32,
        &config.device,
    );
    let stage2_model = JointEmbedModel::new_with_warm_start(
        WarmStartArgs {
            n_cells,
            embedding_dim: h,
            frozen_e_feat,
            frozen_b_feat,
            e_cell_init: Some(&e_cell_init),
            b_cell_init: &b_cell_init,
        },
        &stage2_varmap,
        stage2_vs,
        &config.device,
    )?;
    let mut stage2_opt = AdamW::new(
        stage2_varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            ..Default::default()
        },
    )?;

    let stage2_axis = identity_axis(n_cells);
    let stage2_samplers = build_active_samplers(unified, &feat_weights, alpha_neg)?;

    // Cell-cell loss attaches only here.
    struct CellCellPrepared {
        samplers: Vec<Option<crate::loss::PerBatchCellSampler>>,
        edges: Vec<(u32, u32)>,
        lambda: f32,
        n_negatives: usize,
    }
    let cell_cell_built: Option<CellCellPrepared> = match config.cell_cell.take() {
        Some(cc) if cc.lambda > 0.0 && !cc.edges.is_empty() => {
            let (samplers, cross_batch) = build_per_batch_cell_samplers(
                &cc.edges,
                &unified.batch_membership,
                unified.n_batches(),
                n_cells,
                alpha_neg,
            );
            let n_active = samplers.iter().filter(|s| s.is_some()).count();
            if cross_batch > 0 {
                info!(
                    "Cell-cell loss: dropped {} cross-batch edges; {} batch(es) have within-batch edges",
                    cross_batch, n_active
                );
            }
            if n_active == 0 {
                warn!(
                    "Cell-cell loss requested (λ={}) but no batch retained any \
                     within-batch edges — falling back to bipartite-only.",
                    cc.lambda
                );
                None
            } else {
                info!(
                    "Cell-cell loss enabled: λ={}, K={}, {} active batch(es), {} edges total",
                    cc.lambda,
                    cc.n_negatives,
                    n_active,
                    cc.edges.len()
                );
                Some(CellCellPrepared {
                    samplers,
                    edges: cc.edges,
                    lambda: cc.lambda,
                    n_negatives: cc.n_negatives,
                })
            }
        }
        _ => None,
    };
    let cell_cell_training = cell_cell_built.as_ref().map(|p| CellCellTraining {
        samplers: &p.samplers,
        edges: &p.edges,
        lambda: p.lambda,
        n_negatives: p.n_negatives,
    });

    info!(
        "Stage 2: per-cell NCE with frozen E_feat ({} cells × {} features, {} epochs)",
        n_cells, n_features, config.epochs
    );
    let stage2_params = stage_params(&config);
    let stage2_ctx = TrainingContext {
        unified,
        cell_axis: &stage2_axis,
        feat_weights: &feat_weights,
        batch_samplers: &stage2_samplers,
        cell_cell: cell_cell_training,
        dev: &config.device,
        stop: &stop,
    };
    train(
        &stage2_model,
        &mut stage2_opt,
        &stage2_ctx,
        &stage2_params,
        None,
    )?;

    Ok(FitOutput {
        model: stage2_model,
        varmap: stage2_varmap,
    })
}

fn build_active_samplers(
    unified: &UnifiedData,
    feat_weights: &[f32],
    alpha_neg: f32,
) -> anyhow::Result<Vec<PerBatchSampler>> {
    let samplers_all = build_per_batch_samplers(
        &unified.triplets,
        &unified.batch_membership,
        unified.n_batches(),
        unified.n_features(),
        feat_weights,
        alpha_neg,
    );
    let mut empty: Vec<&str> = Vec::new();
    let active: Vec<PerBatchSampler> = samplers_all
        .into_iter()
        .enumerate()
        .filter_map(|(b, s)| {
            if s.is_none() {
                empty.push(unified.batch_names[b].as_ref());
            }
            s
        })
        .collect();
    if !empty.is_empty() {
        warn!(
            "Skipping {} batch(es) with no observed edges: {}",
            empty.len(),
            empty.join(", ")
        );
    }
    if active.is_empty() {
        anyhow::bail!("no non-empty batches available for sampling");
    }
    Ok(active)
}

fn build_smoother(
    feature_network: Option<FeatureNetworkConfig>,
    n_features: usize,
    embedding_dim: usize,
) -> anyhow::Result<Option<FeatureNetworkSmoother>> {
    let Some(FeatureNetworkConfig {
        graph,
        k_hops,
        alpha,
        refresh_epochs,
    }) = feature_network
    else {
        return Ok(None);
    };
    if graph.num_edges() == 0 {
        anyhow::bail!("feature network has 0 matched edges — check name resolution at the caller");
    }
    info!(
        "SGC smoothing: K={}, α={}, refresh={} epochs over {} edges",
        k_hops,
        alpha,
        refresh_epochs,
        graph.num_edges()
    );
    Ok(Some(FeatureNetworkSmoother::new(
        &graph,
        n_features,
        embedding_dim,
        alpha,
        k_hops,
        refresh_epochs,
    )?))
}

fn stage_params(config: &FitConfig) -> TrainingParams {
    TrainingParams {
        epochs: config.epochs,
        batches_per_epoch: config.batches_per_epoch,
        batch_size: config.batch_size,
        num_negatives: config.num_negatives,
        seed: config.seed,
    }
}

/// Per-level epoch budget for the stage-1 curriculum. Coarser levels
/// (lower index after `collapsed_levels.reverse()`) get more epochs:
/// `w[i] = num_levels - i`. Mirrors
/// `senna::topic::common::compute_level_epochs` so the same curve
/// applies across senna topic and gbe.
fn compute_level_epochs(total_epochs: usize, num_levels: usize) -> Vec<usize> {
    if num_levels == 0 {
        return Vec::new();
    }
    let total_weight: usize = (1..=num_levels).sum();
    (0..num_levels)
        .map(|i| {
            let w = num_levels - i;
            (total_epochs * w / total_weight).max(1)
        })
        .collect()
}

/// Read a candle [rows, cols] tensor back to a host `nalgebra::DMatrix`
/// (column-major). Used between stage-1 levels (and at the boundary
/// to stage 2) so the trained `E_feat` / `b_feat` survives across
/// VarMap rebuilds.
fn tensor_to_mat(t: &Tensor, rows: usize, cols: usize) -> anyhow::Result<DMatrix<f32>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1::<f32>()?;
    debug_assert_eq!(flat.len(), rows * cols);
    let mut m = DMatrix::<f32>::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            m[(i, j)] = flat[i * cols + j];
        }
    }
    Ok(m)
}

fn mat_to_tensor(mat: &DMatrix<f32>, dev: &Device) -> anyhow::Result<Tensor> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    let mut row_major = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            row_major.push(mat[(i, j)]);
        }
    }
    Ok(Tensor::from_vec(row_major, (rows, cols), dev)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_tensor_round_trip_preserves_values() {
        // nalgebra is column-major and candle Tensor is row-major; both
        // helpers mediate via the same row-major layout. Locks in the
        // round-trip so a future "optimization" in either direction can't
        // silently transpose E_feat between stages.
        let dev = Device::Cpu;
        let m = DMatrix::<f32>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = mat_to_tensor(&m, &dev).unwrap();
        let back = tensor_to_mat(&t, 2, 3).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (m[(i, j)] - back[(i, j)]).abs() < 1e-6,
                    "mismatch at ({i},{j}): {} vs {}",
                    m[(i, j)],
                    back[(i, j)]
                );
            }
        }
    }
}
