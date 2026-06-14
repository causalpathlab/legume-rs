//! Public entry point for `graph-embedding`. Callers translate their
//! own CLI args into a [`FitConfig`] and pass already-loaded
//! [`UnifiedData`] (so this crate stays free of file/path concerns).

use crate::coarsen::{identity_axis, AxisCoarsenings};
use crate::data::UnifiedData;
use crate::feature_network::FeatureNetworkSmoother;
use crate::loss::{
    build_per_batch_cell_samplers, build_stratified_sampler, CellFeatureSampler,
    PerBatchStratifiedCellSampler, StratifiedSampler,
};
use crate::model::{JointEmbedModel, ModelArgs, ModelInit, ShareFeaturesArgs};
use crate::progress::new_progress_bar;
use crate::stop::setup_stop_handler;
use crate::training::{
    train_composite, AxisSampler, CellCellPbChainTraining, CellCellTraining, CompositeAxis,
    CompositeMode, CompositeTrainContext, TrainingParams,
};
use auxiliary_data::feature_names::FeatureNameKind;
use candle_util::candle_core::Device;
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_util::frozen_features::trainable_vars;
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
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
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
    /// Number of multilevel-collapse levels (coarse → fine). Maps
    /// directly to [`MultilevelParams::num_levels`].
    pub num_levels: usize,
    /// Binary-tree partition depth at the finest level — at most
    /// `2^sort_dim + 1` pseudobulk leaves. Maps to
    /// [`MultilevelParams::sort_dim`].
    pub sort_dim: usize,
    /// In-batch k-NN used when merging cells into pseudobulk samples.
    /// Maps to [`MultilevelParams::knn_pb_samples`].
    pub knn_pb_samples: usize,
    /// Coordinate-descent iterations for the per-batch δ correction
    /// inside the collapse. Maps to [`MultilevelParams::num_opt_iter`].
    pub num_opt_iter: usize,
    /// Target rank of the random-projection sketch that seeds batch
    /// correction and the multilevel collapse.
    pub proj_dim: usize,
    pub epochs: usize,
    /// `None` = auto: one weighted pass per epoch over the largest axis.
    /// `Some(n)` = fixed step budget.
    pub batches_per_epoch: Option<usize>,
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
    /// Explicit L2 penalty `λ · ‖E_feat‖_F²` on the shared feature
    /// embedding, added to the composite loss before backward. `0.0`
    /// disables.
    pub feature_embedding_l2: f32,
    /// Number of latent programs `K` for the per-condition feature gate
    /// (`e_feat(f|s) = E_feat[f] ⊙ exp(Σ_k z[f,k]·δ[k,s,:])`). The gate is
    /// identity at init and on the reference condition, so `K` only adds
    /// capacity when conditions actually diverge.
    pub num_programs: usize,
    /// L2 penalty on the gate program loadings `z` (mean-normalized).
    pub z_l2: f32,
    /// L2 penalty on the gate deviation directions `δ` (mean-normalized).
    pub delta_l2: f32,
    /// AdamW decoupled weight decay applied uniformly to every parameter
    /// (the shared `E_feat`, `b_feat`, and every per-axis head). Post-
    /// step shrinkage; doesn't enter the backward graph. `0.0` disables.
    pub weight_decay: f64,
    /// Global-norm gradient clip per AdamW step (`0.0` = off). Bounds the
    /// update magnitude so embeddings don't inflate on NCE loss spikes.
    pub max_grad_norm: f32,
    /// Optional per-cell multiplier on the cell-axis sampling weight
    /// (length = `n_cells`, indexed by global cell id). Folded into the
    /// `degree^α` cell picker so up-weighted cells are sampled more often.
    /// Used by `--multiome` to up-weight matched (bridge) cells so they
    /// anchor the cross-modal alignment. `None` = every cell weight ×1.
    pub cell_weight_mult: Option<Vec<f32>>,
    /// Phase-1 cell-axis mode (`k`). Controls only what shapes `E_feat` in
    /// phase 1; phase 2 always analytically projects *every* cell against the
    /// fixed feature side, so the full per-cell embedding is unaffected.
    /// - `k == 0`: suppress the cell axis entirely (pure-pb — `E_feat` shaped
    ///   by pb aggregates only; fastest). This is the default.
    /// - `1 ≤ k < n_cells`: keep ≤`k` cells per pb-sample at EVERY collapse
    ///   level (union), shrinking the phase-1 step budget
    ///   (`Σ active_cells / batch_size`) while keeping rare/shallow cells
    ///   visible to the shared feature dictionary.
    /// - `k ≥ n_cells`: no pb-sample exceeds `k`, so subsampling is a no-op —
    ///   every cell shapes `E_feat` (legacy all-cells behaviour; slowest).
    pub phase1_cells_per_pb: usize,
}

/// Cell-cell NCE configuration. Positives = neighbor pairs from a
/// caller-provided graph, gated by pb co-membership at every chain
/// level; per-chain-position sibling negatives drive the loss across
/// pb-tree resolutions. Set `lambda = 0` to disable the cell-cell
/// term entirely.
#[derive(Clone)]
pub struct CellCellConfig {
    /// Positive cell pairs as global cell ids (canonical (i, j) with i < j).
    pub edges: Vec<(u32, u32)>,
    /// Loss mixing weight λ. 0.0 → cell-cell term skipped; 1.0 → equal
    /// weight with the bipartite loss; > 1 emphasizes cell-cell.
    pub lambda: f32,
    /// Negative cells per positive pair.
    pub n_negatives: usize,
    /// Which collapse levels to chain over. `None` = every level
    /// produced by the multilevel collapse inside [`fit`]. Indices
    /// refer to the coarsest-first `cell_to_pb_per_level`, so `0` =
    /// coarsest, `last` = finest. Out-of-range entries are dropped
    /// with a warning. Levels whose pb count is close to `n_cells`
    /// (per-cell partitions, no useful classification signal) are
    /// auto-pruned by [`resolve_pb_chain`].
    pub pb_levels: Option<Vec<usize>>,
    /// Per-level λ; same length as the resolved `pb_levels`. `None` =
    /// uniform 1.0. User-supplied values pass through as-is — they
    /// are not normalized by the number of levels, so adjust the
    /// global `CellCellConfig::lambda` accordingly when chaining many
    /// levels.
    pub lambda_per_level: Option<Vec<f32>>,
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
/// `senna gbe` reaches this so the resolution + zero-edge guard live
/// here to keep error messages consistent.
pub struct FeatureNetworkArgs<'a> {
    pub path: &'a str,
    pub feature_names: &'a [Box<str>],
    pub prefix_match: bool,
    pub delim: Option<char>,
    pub k_hops: usize,
    pub alpha: f32,
    pub refresh_epochs: usize,
    /// Canonicalizer applied to both the axis names and each edge endpoint
    /// before matching — the same `FeatureNameKind` used to load the data,
    /// so an edge file with raw gene / `chrX:start-end` names resolves
    /// against the canonicalized unified axis (e.g. multiome cis-links).
    pub feature_kind: FeatureNameKind,
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
    let full_weights = compute_nb_fisher_weights(unified.count_backend(), block_size)?;
    info!(
        "  {} features, mean Fisher weight {:.3}",
        full_weights.len(),
        full_weights.iter().sum::<f32>() / full_weights.len().max(1) as f32
    );
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
    let kind = args.feature_kind.clone();
    let graph = FeaturePairGraph::from_edge_list_canon(
        args.path,
        args.feature_names.to_vec(),
        args.prefix_match,
        args.delim,
        &|s| kind.canonicalize(s),
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
/// save checkpoints or re-run inference; the current caller (`senna
/// gbe`) only consumes `model`, so it sits unused but kept alive.
pub struct FitOutput {
    pub model: JointEmbedModel,
    pub varmap: VarMap,
    /// Un-normalized baseline MAP per-cell projection norm from phase 2 (`0`
    /// for cells with no observed features / when phase 2 was skipped). The
    /// empty-droplet cell QC reads this: empties solve to ≈0, real cells far
    /// above. The stored latent (`model.e_cell`) is the L2 *direction*; this
    /// norm is the un-normalized magnitude it was divided by.
    pub cell_nrms: Vec<f32>,
    /// Depth-free, gene-co-scaled cell embedding `[n_cells × H]` row-major —
    /// the same direction as the latent but with a depth-normalized magnitude
    /// gauged onto the gene-dictionary scale. For a joint cell–gene biplot, not
    /// clustering (empty when phase 2 was skipped). See `project_cells_phase2`.
    pub cell_embedding_scaled: Vec<f32>,
}

/// Composite-objective gbe fit — trained in **two phases**.
///
/// The bilinear score is `E_feat[f]·E_cell[c] + b_feat[f]` (no per-cell
/// bias — `b_cell` is dropped: it carries no cell-type signal and only
/// siphons the sparse per-cell gradient).
///
/// **Phase 1 — features + pseudobulks.** Train only the pseudobulk axes
/// (coarsest..finest from `collapse_columns_multilevel_vec`, pseudobulk-
/// feature triplets) with `Sum`. They share — and learn — `E_feat /
/// b_feat / z / δ` and per-level pb cell-side embeddings.
///
/// **Phase 2 — dense per-cell embedding.** Freeze the entire feature side
/// and fit ONLY `E_cell` against it. With a single axis the objective is
/// separable per cell — each row's gradient depends only on that cell's
/// own edges (embarrassingly parallel) — and the auto per-epoch budget
/// (sized by `n_units` = n_cells) sweeps every cell ~once per epoch.
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

    // ---- Optional hard cap on feature count ----
    // Picks the top-`max_features` genes by Fisher (ties broken by id
    // for determinism), prunes `unified` + `feat_weights` to that axis.
    // Runs after Fisher so the ranking is final; runs before the
    // multilevel collapse so every downstream pass sees the smaller
    // axis (smaller pb matrices, smaller sampler prefix sums, smaller
    // E_feat).
    let feat_weights = maybe_cap_features(unified, feat_weights, config.max_features);

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
    let n_conditions = unified.n_conditions();
    let mut cell_model = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features,
            n_cells,
            embedding_dim: h,
            n_conditions,
            num_programs: config.num_programs,
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
                shared_z: cell_model.z.clone(),
                shared_delta: cell_model.delta.clone(),
                num_programs: cell_model.num_programs,
                n_conditions: cell_model.n_conditions,
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
    let phase1_cell_samplers: &[PerBatchStratifiedCellSampler] = match &phase1_cell_samplers_owned {
        Some(sub) => {
            let kept: usize = sub.iter().map(|s| s.active_cells.len()).sum();
            info!(
                "Phase-1 cell subsampling: ≤{} cells per pb-sample (all {} levels) → \
                 {} of {} cells shape E_feat (phase 2 still projects all {})",
                config.phase1_cells_per_pb, num_levels, kept, n_cells, n_cells
            );
            sub
        }
        None => {
            // k == 0 → cell axis suppressed (logged); k ≥ n_cells → legacy all-cells.
            if !use_cell_axis {
                info!(
                    "Phase-1 cell axis SUPPRESSED (pure-pb): E_feat shaped by pb aggregates \
                     only; phase 2 still projects all {} cells",
                    n_cells
                );
            }
            &cell_samplers
        }
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
    let cell_cell_built = build_cell_cell_training(
        unified,
        n_cells,
        alpha_neg,
        config.cell_cell.take(),
        &cell_to_pb_per_level,
    );

    let mut smoother = build_smoother(config.feature_network.take(), n_features, h)?;

    // Note on biases: the per-CELL bias `b_cell` is dropped (see `fit()`
    // doc / `model.rs`) — never trained, never written. The per-PB biases
    // (`pb_l*_b_cell`) DO train in phase 1: a per-pseudobulk bias absorbs
    // that pb's depth so the shared `E_feat` captures composition, not
    // library size.

    let cell_cell_training = cell_cell_built.as_ref().map(|p| CellCellTraining {
        samplers: &p.samplers,
        edges: &p.edges,
        lambda: p.lambda,
        n_negatives: p.n_negatives,
        pb_chain: CellCellPbChainTraining {
            levels: &p.chain.levels,
            lambdas: &p.chain.lambdas,
            cell_to_pb_per_level: &cell_to_pb_per_level,
        },
    });

    // Two-phase training (always — `ge::fit` is the bge driver only); see
    // the `fit()` doc for the rationale. Shared AdamW hyperparameters:
    let adamw_params = || ParamsAdamW {
        lr: config.learning_rate,
        weight_decay: config.weight_decay,
        ..Default::default()
    };

    // Cell axis (per-cell embedding). Trained jointly in phase 1 (to shape
    // `E_feat`) and recalibrated in phase 2 against the fixed feature side.
    // The optional cell-cell NCE term attaches here.
    let cell_axis = CompositeAxis {
        model: &cell_model,
        unified,
        cell_axis: &cell_axis_coarsening,
        sampler: AxisSampler::PerBatchStratified(phase1_cell_samplers),
        lambda: DEFAULT_AXIS_LAMBDA,
        cell_cell: cell_cell_training,
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
            cell_cell: None,
            label: "pb",
        });
    }

    /////////////////////////////
    // Phase 1: joint training //
    /////////////////////////////

    // The cell axis is trained HERE (e_cell trainable; base `b_cell` dropped,
    // pb `pb_l*_b_cell` stay trainable) so the per-cell stratified sampler —
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
        let mut opt1 = AdamW::new(trainable_vars(&varmap, &["b_cell"]), adamw_params())?;
        let mut p1 = stage_params(&config);
        p1.composite_mode = CompositeMode::Sum;
        info!(
            "Phase 1 (joint) — features + {}{} pb level(s) [Sum], {} epochs",
            if use_cell_axis { "cell + " } else { "" },
            joint_axes.len() - use_cell_axis as usize,
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
    // directly (Poisson MAP, ridge prior) in parallel. bge scores without a
    // per-cell bias, so the intercept is dropped. See `crate::cell_projection`.
    let mut cell_nrms: Vec<f32> = Vec::new();
    let mut cell_embedding_scaled: Vec<f32> = Vec::new();
    if !stop.load(std::sync::atomic::Ordering::Relaxed) {
        info!(
            "Phase 2 — analytical per-cell projection ({} cells, feature side fixed, ridge λ={})",
            n_cells, PHASE2_RIDGE
        );
        let (nrms, scaled) = project_cells_phase2(
            &mut cell_model,
            &varmap,
            &cell_samplers,
            &unified.condition_membership,
            n_cells,
            PHASE2_RIDGE as f64,
            &config.device,
        )?;
        cell_nrms = nrms;
        cell_embedding_scaled = scaled;
    }

    Ok(FitOutput {
        model: cell_model,
        varmap,
        cell_nrms,
        cell_embedding_scaled,
    })
}

/// Ridge prior strength λ on `e_cell` in the analytical phase-2 projection.
/// The Poisson MAP fits each cell's observed features and this Gaussian
/// prior stands in for the (infeasible) all-feature softmax partition.
const PHASE2_RIDGE: f32 = 1.0;

/// Phase 2 — project every cell onto the fixed feature dictionary, in
/// parallel, and overwrite the `e_cell` var. The gated feature embedding is
/// condition-dependent (`E_feat ⊙ exp(z·δ̄_s)`), so cells are grouped by
/// condition and each condition's gated table is built once. The per-cell
/// bias is fitted (to absorb library size) but discarded — bge scores
/// without it.
///
/// The stored latent (`model.e_cell`) is the **L2 direction** of the baseline
/// Poisson-MAP embedding — depth-robust and best for clustering/UMAP. (Storing
/// the magnitude instead blurs cell types: the magnitude axis ≈ profile
/// specialization, roughly orthogonal to cell-type identity, so Euclidean
/// clustering mixes it in — a measured ~7–11pt purity loss.)
///
/// Separately returns a **biplot side embedding** (not the latent): the same
/// direction but with a *depth-free* magnitude — counts normalized to a common
/// total before the solve, so sequencing depth drops out (the raw-count MAP
/// magnitude tracks depth ≈0.8; normalized ≈0) — then co-scaled onto the gene
/// dictionary by one global gauge `median‖e_feat‖/median‖e_cell‖`. This puts
/// cells and genes on one scale for a joint cell–gene plot; it is not used for
/// clustering.
///
/// Returns `(cell_nrms, scaled)`. `cell_nrms` is the **un-normalized** baseline
/// MAP norm — the empty-droplet QC keys on it (empties solve to ≈0 only on raw
/// counts; normalizing to the common total would upscale and hide them).
/// `scaled` is the `[n_cells × H]` row-major biplot side embedding.
fn project_cells_phase2(
    model: &mut JointEmbedModel,
    varmap: &VarMap,
    cell_samplers: &[PerBatchStratifiedCellSampler],
    condition_membership: &[u32],
    n_cells: usize,
    lambda: f64,
    dev: &Device,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    use crate::cell_projection::solve_one_cell;
    use anyhow::Context;
    use candle_util::candle_core::{IndexOp, Tensor};
    use rayon::prelude::*;

    let h = model.embedding_dim;
    let n_conditions = model.n_conditions.max(1);

    let b_feat: Vec<f32> = model.b_feat.to_vec1()?;
    let delta_centered = model.delta.broadcast_sub(&model.delta.mean_keepdim(1)?)?;
    let mut e_out: Vec<f32> = model.e_cell.flatten_all()?.to_vec1()?;
    let mut cell_nrms = vec![0f32; n_cells];
    let mut scaled = vec![0f32; n_cells * h]; // biplot side embedding (returned)

    let mut cells: Vec<(u32, &[u32], &[f32])> = Vec::new();
    for s in cell_samplers {
        for (i, &cell) in s.active_cells.iter().enumerate() {
            let cf = &s.per_cell[i];
            cells.push((cell, &cf.features, &cf.counts));
        }
    }
    let mut by_cond: Vec<Vec<usize>> = vec![Vec::new(); n_conditions];
    for (i, &(cell, _, _)) in cells.iter().enumerate() {
        let s = (condition_membership[cell as usize] as usize).min(n_conditions - 1);
        by_cond[s].push(i);
    }

    // Option-1 target total = median cell total (counts rescaled to this).
    let mut totals: Vec<f32> = cells
        .iter()
        .map(|(_, _, c)| c.iter().sum::<f32>())
        .collect();
    totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let target_total = if totals.is_empty() {
        1.0
    } else {
        totals[totals.len() / 2].max(1.0)
    };

    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();

    for (s, idxs) in by_cond.iter().enumerate() {
        if idxs.is_empty() {
            continue;
        }
        let delta_s = delta_centered.i((.., s, ..))?.contiguous()?; // [K, H]
        let logdev = model.z.matmul(&delta_s)?; // [F, H]
        let gated = model.e_feat.broadcast_mul(&logdev.exp()?)?;
        let gated_flat: Vec<f32> = gated.flatten_all()?.to_vec1()?;

        let solved: Vec<(usize, Vec<f32>, Vec<f32>, f32)> = idxs
            .par_iter()
            .map(|&i| {
                let (cell, feats, counts) = cells[i];
                // Baseline MAP on RAW counts → L2 direction (the latent) + the
                // depth-coupled norm the empty-droplet QC keys on.
                let edges: Vec<(u32, f32)> =
                    feats.iter().zip(counts).map(|(&f, &c)| (f, c)).collect();
                let (e_map, _) = solve_one_cell(&edges, &gated_flat, &b_feat, h, lambda);
                let nrm_map = norm(&e_map);
                let dir: Vec<f32> = if nrm_map > 1e-8 {
                    e_map.iter().map(|x| x / nrm_map).collect()
                } else {
                    e_map.clone()
                };
                // Depth-normalized solve (counts → common total) → the biplot
                // side embedding: depth-free magnitude, same direction.
                let tot: f32 = counts.iter().sum::<f32>().max(1e-6);
                let sc = target_total / tot;
                let edges_n: Vec<(u32, f32)> = feats
                    .iter()
                    .zip(counts)
                    .map(|(&f, &c)| (f, c * sc))
                    .collect();
                let (e_n, _) = solve_one_cell(&edges_n, &gated_flat, &b_feat, h, lambda);
                (cell as usize, dir, e_n, nrm_map)
            })
            .collect();
        for (cell, dir, e_n, nrm_map) in solved {
            e_out[cell * h..(cell + 1) * h].copy_from_slice(&dir);
            scaled[cell * h..(cell + 1) * h].copy_from_slice(&e_n);
            cell_nrms[cell] = nrm_map;
        }
    }

    // Co-scale the biplot side embedding onto the gene dictionary's scale (one
    // global scalar — no per-cell tuning). The clustering latent (`e_out`) is
    // left as unit directions; only `scaled` is gauged.
    let feat_flat = model.e_feat.flatten_all()?.to_vec1()?;
    let feat_scale = median_row_norm(&feat_flat, h);
    let mut cell_scales: Vec<f32> = scaled
        .chunks_exact(h)
        .map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt())
        .filter(|&n| n > 1e-8)
        .collect();
    let cell_scale = median_of(&mut cell_scales);
    let gauge = if cell_scale > 1e-8 {
        feat_scale / cell_scale
    } else {
        1.0
    };
    info!(
        "Phase 2 — latent = L2 direction; biplot side embedding gauged ‖e_feat‖/‖e_cell‖ = {:.3}/{:.3} = {:.3}",
        feat_scale, cell_scale, gauge
    );
    if (gauge - 1.0).abs() > 1e-6 {
        for x in scaled.iter_mut() {
            *x *= gauge;
        }
    }

    let e_t = Tensor::from_vec(e_out, (n_cells, h), dev)?;
    varmap
        .data()
        .lock()
        .unwrap()
        .get("e_cell")
        .context("e_cell var missing")?
        .set(&e_t)?;
    model.e_cell = e_t;
    Ok((cell_nrms, scaled))
}

/// Median of a slice (sorts in place; upper-median for even length). Diagnostic.
fn median_of(xs: &mut [f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// Median L2 norm over the rows of a row-major `[n × h]` matrix. Diagnostic.
fn median_row_norm(flat: &[f32], h: usize) -> f32 {
    if h == 0 || flat.is_empty() {
        return 0.0;
    }
    let mut norms: Vec<f32> = flat
        .chunks_exact(h)
        .map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();
    median_of(&mut norms)
}

struct CellCellPrepared {
    samplers: Vec<Option<crate::loss::PerBatchCellSampler>>,
    edges: Vec<(u32, u32)>,
    lambda: f32,
    n_negatives: usize,
    chain: CellCellPreparedChain,
}

struct CellCellPreparedChain {
    /// Indexes into `cell_to_pb_per_level` (coarsest-first); kept owned
    /// so [`CellCellTraining`] can hand out `&[usize]`.
    levels: Vec<usize>,
    /// Same length as `levels`; user-supplied or uniform 1.0.
    lambdas: Vec<f32>,
}

fn build_cell_cell_training(
    unified: &UnifiedData,
    n_cells: usize,
    alpha_neg: f32,
    cell_cell: Option<CellCellConfig>,
    cell_to_pb_per_level: &[Vec<usize>],
) -> Option<CellCellPrepared> {
    let cc = match cell_cell {
        Some(cc) if cc.lambda > 0.0 && !cc.edges.is_empty() => cc,
        _ => return None,
    };

    let chain = resolve_pb_chain(
        cc.pb_levels.as_deref(),
        cc.lambda_per_level.as_deref(),
        cell_to_pb_per_level,
        n_cells,
    );

    let pb_filter = crate::loss::PbChainFilter {
        cell_to_pb_per_level,
        levels: &chain.levels,
    };

    let (samplers, stats) = build_per_batch_cell_samplers(
        &cc.edges,
        &unified.batch_membership,
        unified.n_batches(),
        n_cells,
        alpha_neg,
        Some(pb_filter),
    );
    let n_active = samplers.iter().filter(|s| s.is_some()).count();
    if stats.cross_batch_dropped > 0 {
        info!(
            "Cell-cell loss: dropped {} cross-batch edges; {} batch(es) have within-batch edges",
            stats.cross_batch_dropped, n_active
        );
    }
    if stats.pb_mismatch_dropped > 0 {
        info!(
            "Cell-cell loss: dropped {} edges whose endpoints disagree on pb at one of the chain levels",
            stats.pb_mismatch_dropped
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
        "Cell-cell chain enabled: λ={}, K={}, levels={:?}, λ_per_level={:?}, {} active batch(es), {} edges total",
        cc.lambda,
        cc.n_negatives,
        chain.levels,
        chain.lambdas,
        n_active,
        cc.edges.len(),
    );
    Some(CellCellPrepared {
        samplers,
        edges: cc.edges,
        lambda: cc.lambda,
        n_negatives: cc.n_negatives,
        chain,
    })
}

/// Resolve caller-facing `(pb_levels, lambda_per_level)` into owned
/// `CellCellPreparedChain`. `pb_levels: None` expands to every
/// available level (coarsest-first, matching `cell_to_pb_per_level`).
/// Out-of-range indices are dropped with a warning.
///
/// Levels that are effectively per-cell partitions (pb count >
/// `DEGENERATE_PB_RATIO * n_cells`) are also dropped — at those levels
/// `pb(u) == pb(v)` implies `u == v`, so requiring positives to share
/// pb wipes out the entire edge set and yields no useful signal.
fn resolve_pb_chain(
    user_levels: Option<&[usize]>,
    user_lambdas: Option<&[f32]>,
    cell_to_pb_per_level: &[Vec<usize>],
    n_cells: usize,
) -> CellCellPreparedChain {
    let n_levels = cell_to_pb_per_level.len();
    let raw_levels: Vec<usize> = match user_levels {
        Some(ls) => ls.to_vec(),
        None => (0..n_levels).collect(),
    };

    let mut levels: Vec<usize> = Vec::with_capacity(raw_levels.len());
    let mut keep_mask: Vec<bool> = Vec::with_capacity(raw_levels.len());
    let mut out_of_range: Vec<usize> = Vec::new();
    let mut degenerate: Vec<(usize, usize)> = Vec::new();

    for l in raw_levels {
        if l >= n_levels {
            out_of_range.push(l);
            keep_mask.push(false);
            continue;
        }
        let n_pbs = pb_count(&cell_to_pb_per_level[l]);
        if (n_pbs as f32) > DEGENERATE_PB_RATIO * (n_cells.max(1) as f32) {
            degenerate.push((l, n_pbs));
            keep_mask.push(false);
            continue;
        }
        levels.push(l);
        keep_mask.push(true);
    }
    if !out_of_range.is_empty() {
        warn!(
            "Cell-cell chain: dropping out-of-range pb-level indices {:?} (have {} levels)",
            out_of_range, n_levels
        );
    }
    if !degenerate.is_empty() {
        warn!(
            "Cell-cell chain: dropping degenerate pb levels (pb count > {:.0}% of {} cells, \
             so pb membership ≈ identity): {:?}. Lower --pb-samples to produce chunkier pb's, \
             or pick specific coarser levels via --cell-cell-pb-levels.",
            DEGENERATE_PB_RATIO * 100.0,
            n_cells,
            degenerate
        );
    }

    let lambdas: Vec<f32> = match user_lambdas {
        Some(ls) if ls.len() == keep_mask.len() => ls
            .iter()
            .zip(keep_mask.iter())
            .filter_map(|(&l, &k)| k.then_some(l))
            .collect(),
        Some(ls) => {
            warn!(
                "Cell-cell chain: lambda_per_level length {} doesn't match input levels {} — using uniform 1.0",
                ls.len(),
                keep_mask.len()
            );
            vec![1.0; levels.len()]
        }
        None => vec![1.0; levels.len()],
    };
    CellCellPreparedChain { levels, lambdas }
}

/// Distinct pb count in a compacted `cell_to_pb` map. The multilevel
/// collapse runs `compact_labels` so ids are dense `0..k`; we still
/// take the `max + 1` rather than trust the contract.
fn pb_count(cell_to_pb: &[usize]) -> usize {
    cell_to_pb.iter().copied().max().map(|m| m + 1).unwrap_or(0)
}

/// Threshold above which a pb level is treated as a per-cell partition
/// and pruned from the chain. `n_pbs / n_cells > 0.5` means avg pb size
/// < 2 — even if a few edges survive the pb-mismatch filter at that
/// level, the loss carries near-zero training signal.
const DEGENERATE_PB_RATIO: f32 = 0.5;

/// Derive a phase-1-only subsampled view of the per-batch stratified cell
/// samplers: keep at most `k` cells per pb-sample at EVERY collapse level
/// (`cell_to_pb_per_level`, coarsest..finest), unioned across levels. The
/// returned samplers cover only the kept cells; each batch's `cell_picker` is
/// rebuilt from the kept cells' recomputed degree weights (`degree^alpha_cell ·
/// mult`, degree recovered from `CellFeatureSampler.counts`), while the
/// negative marginal (`neg` / `feature_pool`) is cloned unchanged so negatives
/// stay drawn from the full per-batch feature pool.
///
/// Keeping ≤k per pb at *every* level (not just the finest) lets each level's
/// partition contribute diverse representatives — robust even when refinement
/// breaks strict nesting between adjacent levels. The finest level is batch-
/// aware, so every non-empty batch keeps ≥1 cell; empty batches are dropped
/// (the cell axis re-indexes at sample time and ignores the original batch id).
fn subsample_cell_samplers_multilevel(
    full: &[PerBatchStratifiedCellSampler],
    cell_to_pb_per_level: &[Vec<usize>],
    k: usize,
    alpha_cell: f32,
    cell_weight_mult: Option<&[f32]>,
    seed: u64,
) -> Vec<PerBatchStratifiedCellSampler> {
    let n_cells = cell_to_pb_per_level.first().map_or(0, |v| v.len());
    // Global keep bitmap: ≤k cells per pb-sample, per level, unioned.
    let mut keep = vec![false; n_cells];
    for (level, c2pb) in cell_to_pb_per_level.iter().enumerate() {
        let n_pb = c2pb.iter().copied().max().map_or(0, |m| m + 1);
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_pb];
        for (cell, &pb) in c2pb.iter().enumerate() {
            buckets[pb].push(cell as u32);
        }
        // Per-level seed so the K kept cells differ across levels (more union
        // diversity) yet stay reproducible across runs.
        let mut rng =
            StdRng::seed_from_u64(seed ^ (level as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for b in buckets.iter_mut() {
            // `partial_shuffle` does only k swaps (vs a full O(bucket) shuffle)
            // and hands back the k-element random subset directly.
            let chosen: &[u32] = if b.len() > k {
                b.partial_shuffle(&mut rng, k).0
            } else {
                &b[..]
            };
            for &c in chosen {
                keep[c as usize] = true;
            }
        }
    }

    // Filter each batch sampler to the kept cells; rebuild `cell_picker`.
    full.iter()
        .filter_map(|s| {
            let cap = s.active_cells.len();
            let mut active_cells: Vec<u32> = Vec::with_capacity(cap);
            let mut per_cell: Vec<CellFeatureSampler> = Vec::with_capacity(cap);
            let mut cell_w: Vec<f32> = Vec::with_capacity(cap);
            for (i, &c) in s.active_cells.iter().enumerate() {
                if !keep[c as usize] {
                    continue;
                }
                let cf = &s.per_cell[i];
                let degree: f32 = cf.counts.iter().sum();
                let mult = cell_weight_mult.map_or(1.0, |m| m[c as usize]);
                cell_w.push(degree.max(1e-8).powf(alpha_cell) * mult);
                per_cell.push(cf.clone());
                active_cells.push(c);
            }
            if active_cells.is_empty() {
                return None;
            }
            let cell_picker =
                WeightedIndex::new(cell_w).expect("non-empty subsampled cell weights");
            Some(PerBatchStratifiedCellSampler {
                cell_picker,
                active_cells,
                per_cell,
                neg: s.neg.clone(),
                feature_pool: s.feature_pool.clone(),
            })
        })
        .collect()
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
    cell_weight_mult: Option<&[f32]>,
) -> anyhow::Result<Vec<PerBatchStratifiedCellSampler>> {
    // Build the per-batch stratified-cell samplers by **streaming columns**
    // from the backend, never materializing the flat cell↔feature edge list.
    // The strat-cell sampler groups edges by cell (`per_cell`), which is
    // exactly a column read — so the 5 GB flat triplet list (which only the
    // unused flat `PerBatch` path ever read) is skipped entirely. The HVG /
    // frozen subset is honored via `feature_to_backend_row`.
    let data = unified.count_backend();
    let n_cells = data.num_columns();
    let n_features = unified.n_features();
    let n_batches = unified.n_batches();
    let batch_membership = &unified.batch_membership;

    // backend compact row → unified id (u32::MAX ⇒ dropped by a subset).
    let backend_rows = data.num_rows();
    let mut backend_to_unified = vec![u32::MAX; backend_rows];
    for (uid, &brow) in unified.feature_to_backend_row.iter().enumerate() {
        if brow < backend_rows {
            backend_to_unified[brow] = uid as u32;
        }
    }

    // Per-batch accumulators, filled as cells stream in.
    let mut active_cells: Vec<Vec<u32>> = vec![Vec::new(); n_batches];
    let mut per_cell: Vec<Vec<CellFeatureSampler>> = (0..n_batches).map(|_| Vec::new()).collect();
    let mut cell_w: Vec<Vec<f32>> = vec![Vec::new(); n_batches];
    let mut feat_count: Vec<Vec<f32>> = (0..n_batches).map(|_| vec![0f32; n_features]).collect();

    // Slab width targets ~8M edges/slab. When nnz can't be reported
    // (num_non_zeros errs → 0) fall back to a fixed cell-count slab rather
    // than the whole matrix, so the streaming memory bound always holds.
    let chunk = match data.num_non_zeros() {
        Ok(nnz) if nnz > 0 => {
            let avg_per_col = (nnz / n_cells.max(1)).max(1);
            (8_000_000 / avg_per_col).clamp(1, n_cells.max(1))
        }
        _ => (1usize << 14).min(n_cells.max(1)),
    };
    let pb_bar = new_progress_bar(n_cells as u64);
    pb_bar.set_message("strat-cell sampler (streaming columns)");

    let mut start = 0usize;
    while start < n_cells {
        let end = (start + chunk).min(n_cells);
        let slab = end - start;
        // Group this slab's nonzeros by local column (cell). `for_each_triplet`
        // emits out_col relative to the passed `start..end`, i.e. 0..slab.
        let mut col_feats: Vec<Vec<u32>> = vec![Vec::new(); slab];
        let mut col_counts: Vec<Vec<f32>> = vec![Vec::new(); slab];
        let mut col_wts: Vec<Vec<f32>> = vec![Vec::new(); slab];
        let mut col_deg: Vec<f32> = vec![0.0; slab];
        data.for_each_triplet(start..end, slab, |brow, local_col, v| {
            if v == 0.0 {
                return;
            }
            let uid = backend_to_unified[brow as usize];
            if uid == u32::MAX {
                return;
            }
            let lc = local_col as usize;
            let cell = start + lc;
            let b = batch_membership[cell] as usize;
            col_feats[lc].push(uid);
            col_counts[lc].push(v);
            col_wts[lc].push((v * feat_weights[uid as usize]).max(1e-8));
            col_deg[lc] += v;
            feat_count[b][uid as usize] += v;
        })?;
        for lc in 0..slab {
            if col_feats[lc].is_empty() {
                continue;
            }
            let cell = (start + lc) as u32;
            let b = batch_membership[cell as usize] as usize;
            let picker =
                WeightedIndex::new(std::mem::take(&mut col_wts[lc])).expect("cell-feature weights");
            per_cell[b].push(CellFeatureSampler {
                features: std::mem::take(&mut col_feats[lc]),
                counts: std::mem::take(&mut col_counts[lc]),
                picker,
            });
            active_cells[b].push(cell);
            let mult = cell_weight_mult.map_or(1.0, |m| m[cell as usize]);
            cell_w[b].push(col_deg[lc].max(1e-8).powf(alpha_cell) * mult);
        }
        pb_bar.inc(slab as u64);
        start = end;
    }
    pb_bar.finish_and_clear();

    // Finalize one sampler per non-empty batch (re-indexed; the original batch
    // id isn't used at sample time).
    let mut empty: Vec<&str> = Vec::new();
    let mut active: Vec<PerBatchStratifiedCellSampler> = Vec::new();
    for b in 0..n_batches {
        if active_cells[b].is_empty() {
            empty.push(unified.batch_names[b].as_ref());
            continue;
        }
        let cell_picker = WeightedIndex::new(std::mem::take(&mut cell_w[b])).expect("cell weights");
        let fc = &feat_count[b];
        let feature_pool: Vec<u32> = (0..n_features as u32)
            .filter(|&f| fc[f as usize] > 0.0)
            .collect();
        let neg_w: Vec<f32> = feature_pool
            .iter()
            .map(|&f| fc[f as usize].powf(alpha_neg))
            .collect();
        let neg = WeightedIndex::new(neg_w).expect("batch feature pool");
        active.push(PerBatchStratifiedCellSampler {
            cell_picker,
            active_cells: std::mem::take(&mut active_cells[b]),
            per_cell: std::mem::take(&mut per_cell[b]),
            neg,
            feature_pool,
        });
    }
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
        // bge is two-phase: phase 1 (pb axes, no cell axis) and phase 2
        // (single cell axis) both require `Sum`. Each phase sets its own
        // mode explicitly; this default just makes the value well-formed.
        composite_mode: CompositeMode::Sum,
        feature_embedding_l2: config.feature_embedding_l2,
        z_l2: config.z_l2,
        delta_l2: config.delta_l2,
        max_grad_norm: config.max_grad_norm,
    }
}
