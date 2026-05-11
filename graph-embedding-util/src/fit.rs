//! Public entry point for `graph-embedding`. Callers translate their
//! own CLI args into a [`FitConfig`] and pass already-loaded
//! [`UnifiedData`] (so this crate stays free of file/path concerns).

use crate::coarsen::{identity_axis, AxisCoarsenings};
use crate::data::UnifiedData;
use crate::feature_network::FeatureNetworkSmoother;
use crate::loss::{
    build_per_batch_cell_samplers, build_per_batch_stratified_cell_samplers,
    build_stratified_sampler, PerBatchStratifiedCellSampler, StratifiedSampler,
};
use crate::model::{JointEmbedModel, ModelArgs, ModelInit, ShareFeaturesArgs};
use crate::stop::setup_stop_handler;
use crate::training::{
    train_composite, AxisSampler, CellCellTraining, CompositeAxis, CompositeMode,
    CompositeTrainContext, TrainingParams,
};
use candle_util::candle_core::Device;
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::gene_weighting::{
    compute_nb_fisher_weights, load_per_gene_weights, save_per_gene_weights,
};
use data_beans_alg::random_projection::RandProjOps;
use data_beans_alg::refine_multilevel::RefineParams;
use log::{info, warn};
use matrix_param::traits::Inference;
use matrix_util::pair_graph::FeaturePairGraph;
use nalgebra::DMatrix;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Per-axis mixing weight in the composite loss. Defaults to 1.0 for
/// every axis (uniform); callers can override by passing a different
/// `lambda_per_axis` shape via [`FitConfig`] in the future.
const DEFAULT_AXIS_LAMBDA: f32 = 1.0;

/// Stratification exponent for pb-axis positive sampling: `q(p) ∝
/// pb_size(p)^alpha`. `0` is uniform (every pb equal coverage); `1`
/// is count-proportional (matches the old flat sampler). `0.5`
/// (sublinear, mirrors the `count^0.75` we use for negatives) gives
/// rare cell types meaningful coverage without starving the dominant
/// strata.
const DEFAULT_STRATIFY_ALPHA_PB: f32 = 0.5;

/// Stratification exponent for cell-axis positive sampling: outer pick
/// is `q(c) ∝ degree(c)^alpha_cell` within each batch. Same shape as
/// `alpha_pb`. `0.5` gives rare/shallow cells real coverage without
/// starving deeply sequenced cells.
const DEFAULT_STRATIFY_ALPHA_CELL: f32 = 0.5;

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
    /// How the composite training loop mixes per-axis NCE losses each
    /// step. `Sum` = one minibatch per axis per step (lower variance,
    /// `O(n_axes)` work per step); `Sample` = one axis per step picked
    /// proportional to `λ` (same expected gradient, `O(1)` per step,
    /// roughly `n_axes×` faster epochs at the cost of a noisier
    /// estimator). Defaults to `Sum` for backwards compatibility.
    pub composite_mode: CompositeMode,
    /// Optional per-row HVG weights for the random projection (length =
    /// full feature axis). When `Some(w)`, the RP uses
    /// `project_columns_weighted` with these weights — uninformative
    /// genes are down-weighted but still contribute to the sketch and
    /// every downstream pass. When `None`, falls back to plain batch-
    /// corrected RP (every gene weight = 1).
    pub hvg_weights: Option<Vec<f32>>,
    /// Hard cap on the number of features trained. When `> 0` and less
    /// than `n_features`, keeps the top-`max_features` genes by
    /// NB-Fisher weight and drops the rest before the multilevel
    /// collapse. Shrinks `E_feat`, the triplet vec, every per-batch
    /// sampler, and pb-blob storage proportionally — the dominant
    /// large-data speed knob now that supergene coarsening is gone.
    /// `0` keeps every gene.
    pub max_features: usize,
    /// BBKNN + DC-Poisson refinement on the multi-level pseudobulk
    /// partition. `Some(RefineParams::default())` enables it (parity
    /// with senna topic / svd / postprocess); `None` falls back to the
    /// raw hash partition. Setting `num_gibbs == 0 && num_greedy == 0`
    /// inside `Some(..)` is equivalent to disabling.
    pub refine: Option<RefineParams>,
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

/// Cap `unified.n_features()` at `max_features` by Fisher rank (largest
/// kept). No-op when `max_features == 0` or already <= cap. Subsets
/// triplets + feature_names + feature_to_backend_row via
/// [`UnifiedData::subset_features`] and the returned weight vector
/// in lockstep, so downstream sampler/collapse builds see a single
/// compact axis.
fn maybe_cap_features(
    unified: &mut UnifiedData,
    feat_weights: Vec<f32>,
    max_features: usize,
) -> Vec<f32> {
    if max_features == 0 {
        return feat_weights;
    }
    let n = unified.n_features();
    if n <= max_features {
        return feat_weights;
    }
    // Argsort descending by Fisher, then take top-N. Ties broken by
    // ascending gene id for determinism across runs / re-cached weights.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        feat_weights[b]
            .partial_cmp(&feat_weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut selected: Vec<usize> = order.into_iter().take(max_features).collect();
    selected.sort_unstable(); // subset_features remaps in old-axis order
    info!(
        "Feature cap: {} → {} genes (top by NB-Fisher)",
        n,
        selected.len()
    );
    let new_weights: Vec<f32> = selected.iter().map(|&i| feat_weights[i]).collect();
    unified.subset_features(&selected);
    new_weights
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

/// Composite-objective gbe fit.
///
/// Builds one shared `(E_feat, b_feat)` and one head per axis:
/// - **Per-cell axis** — fine-resolution triplets with optional
///   cell-cell NCE term.
/// - **Per pseudobulk level** (coarsest..finest from
///   `collapse_columns_multilevel_vec`) — pseudobulk-feature triplets
///   from `mu_adjusted` / `mu_observed` of that level.
///
/// All heads share `E_feat` / `b_feat` Vars in a single `VarMap`. Each
/// minibatch step samples one positive batch from every axis,
/// accumulates `Σ_k λ_k · L_NCE_k` (+ optional cell-cell term on the
/// per-cell axis), and takes a single AdamW step. The shared feature
/// embedding receives gradients from every axis simultaneously, so
/// coarse and fine constraints regularize each other in lockstep
/// instead of via curriculum hand-offs.
pub fn fit(unified: &mut UnifiedData, mut config: FitConfig) -> anyhow::Result<FitOutput> {
    let n_cells = unified.n_cells();
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
        unified.per_file_data[0].project_columns_weighted(
            config.sketch_dim,
            config.block_size,
            batch_arg,
            w,
        )?
    } else {
        unified.per_file_data[0].project_columns_with_batch_correction(
            config.sketch_dim,
            config.block_size,
            batch_arg,
        )?
    };

    let feat_weights = load_or_compute_fisher_weights(
        unified,
        config.block_size,
        config.fisher_weights_cache.as_deref(),
    )?;

    // ---- Optional hard cap on feature count ----
    // Picks the top-`max_features` genes by Fisher (ties broken by id
    // for determinism), prunes `unified` + `feat_weights` to that axis.
    // Runs after Fisher so the ranking is final; runs before the
    // multilevel collapse so every downstream pass sees the smaller
    // axis (smaller pb matrices, smaller sampler prefix sums, smaller
    // E_feat).
    let feat_weights = maybe_cap_features(unified, feat_weights, config.max_features);

    // ---- Multilevel collapse → batch-corrected pseudobulks ----
    info!(
        "Multilevel collapse (target {} super-cells, {} levels)...",
        config.super_cells, config.num_coarsen_seeds
    );
    let collapse_out = collapse_columns_multilevel_with_hierarchy(
        &mut unified.per_file_data[0],
        &proj_out.proj,
        &batch_labels,
        &MultilevelParams {
            knn_super_cells: 10,
            num_levels: config.num_coarsen_seeds.max(1),
            sort_dim: config.sketch_dim,
            num_opt_iter: 100,
            refine: config.refine.clone(),
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
    for collapsed in collapsed_levels.iter() {
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
    let cell_model = JointEmbedModel::new_with_init(
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
                var_prefix: &format!("pb_l{}", level_idx),
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
    )?;
    info!(
        "Composite axis cell ({} cells × {} features, strat-cell α={}, {} active batch(es))",
        n_cells,
        n_features,
        DEFAULT_STRATIFY_ALPHA_CELL,
        cell_samplers.len()
    );

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
                "pb_l{}: stratified sampler build failed (no positives or empty feature pool)",
                level_idx
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

    // Cell-cell loss prep (attaches only to the per-cell axis).
    let cell_cell_built =
        build_cell_cell_training(unified, n_cells, alpha_neg, config.cell_cell.take());

    let mut smoother = build_smoother(config.feature_network.take(), n_features, h)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            ..Default::default()
        },
    )?;

    let cell_cell_training = cell_cell_built.as_ref().map(|p| CellCellTraining {
        samplers: &p.samplers,
        edges: &p.edges,
        lambda: p.lambda,
        n_negatives: p.n_negatives,
    });

    let mut axes: Vec<CompositeAxis> = Vec::with_capacity(num_levels + 1);
    axes.push(CompositeAxis {
        model: &cell_model,
        unified,
        cell_axis: &cell_axis_coarsening,
        sampler: AxisSampler::PerBatchStratified(&cell_samplers),
        lambda: DEFAULT_AXIS_LAMBDA,
        cell_cell: cell_cell_training,
        label: "cell",
    });
    for (i, model) in level_models.iter().enumerate() {
        let (axis, stratified) = &level_axes_data[i];
        axes.push(CompositeAxis {
            model,
            unified: &pb_blobs[i],
            cell_axis: axis,
            sampler: AxisSampler::Stratified(stratified),
            lambda: DEFAULT_AXIS_LAMBDA,
            cell_cell: None,
            label: "pb",
        });
    }

    info!(
        "Composite training: {} axes (1 cell + {} pseudobulk levels), {} epochs",
        axes.len(),
        num_levels,
        config.epochs
    );

    let train_params = stage_params(&config);
    train_composite(
        &CompositeTrainContext {
            axes: &axes,
            feat_weights: &feat_weights,
            dev: &config.device,
            stop: &stop,
            cell_to_pb_per_level: Some(&cell_to_pb_per_level),
        },
        &mut opt,
        &train_params,
        smoother.as_mut(),
    )?;

    Ok(FitOutput {
        model: cell_model,
        varmap,
    })
}

struct CellCellPrepared {
    samplers: Vec<Option<crate::loss::PerBatchCellSampler>>,
    edges: Vec<(u32, u32)>,
    lambda: f32,
    n_negatives: usize,
}

fn build_cell_cell_training(
    unified: &UnifiedData,
    n_cells: usize,
    alpha_neg: f32,
    cell_cell: Option<CellCellConfig>,
) -> Option<CellCellPrepared> {
    let cc = match cell_cell {
        Some(cc) if cc.lambda > 0.0 && !cc.edges.is_empty() => cc,
        _ => return None,
    };
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
        return None;
    }
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

/// Build the stratified per-batch cell samplers and filter out empty
/// batches. Mirrors the previous `build_active_samplers` (flat) but
/// uses the two-stage `cell_picker` → `per_cell` draw — every cell in
/// a batch gets coverage proportional to `degree^alpha_cell` instead
/// of being drowned by deeply sequenced cells.
fn build_active_samplers(
    unified: &UnifiedData,
    feat_weights: &[f32],
    alpha_cell: f32,
    alpha_neg: f32,
) -> anyhow::Result<Vec<PerBatchStratifiedCellSampler>> {
    let samplers_all = build_per_batch_stratified_cell_samplers(
        &unified.triplets,
        &unified.batch_membership,
        unified.n_batches(),
        unified.n_features(),
        feat_weights,
        alpha_cell,
        alpha_neg,
    );
    let mut empty: Vec<&str> = Vec::new();
    let active: Vec<PerBatchStratifiedCellSampler> = samplers_all
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
        composite_mode: config.composite_mode,
    }
}
