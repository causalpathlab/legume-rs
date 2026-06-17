use crate::data::UnifiedData;
use crate::model::JointEmbedModel;
use crate::training::{CompositeMode, TrainingParams};
use auxiliary_data::feature_names::FeatureNameKind;
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use data_beans_alg::gene_weighting::{
    compute_nb_fisher_weights, load_per_gene_weights, save_per_gene_weights,
};
use data_beans_alg::refine_multilevel::RefineParams;
use log::{info, warn};
use matrix_util::pair_graph::FeaturePairGraph;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Per-axis mixing weight in the composite loss. Defaults to 1.0 for
/// every axis (uniform); callers can override by passing a different
/// `lambda_per_axis` shape via [`FitConfig`] in the future.
pub(crate) const DEFAULT_AXIS_LAMBDA: f32 = 1.0;

/// Stratification exponent for pb-axis positive sampling: `q(p) ∝
/// pb_size(p)^alpha`. `0` is uniform (every pb equal coverage); `1`
/// is count-proportional (matches the old flat sampler). `0.5`
/// (sublinear, mirrors the `count^0.75` we use for negatives) gives
/// rare cell types meaningful coverage without starving the dominant
/// strata.
pub(crate) const DEFAULT_STRATIFY_ALPHA_PB: f32 = 0.5;

/// Stratification exponent for cell-axis positive sampling: outer pick
/// is `q(c) ∝ degree(c)^alpha_cell` within each batch. Same shape as
/// `alpha_pb`. `0.5` gives rare/shallow cells real coverage without
/// starving deeply sequenced cells.
pub(crate) const DEFAULT_STRATIFY_ALPHA_CELL: f32 = 0.5;

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
    /// Optional per-row HVG weights for the random projection (length =
    /// full feature axis). When `Some(w)`, the RP uses
    /// `project_columns_weighted` with these weights — uninformative
    /// genes are down-weighted but still contribute to the sketch and
    /// every downstream pass. When `None`, falls back to plain batch-
    /// corrected RP (every gene weight = 1).
    pub hvg_weights: Option<Vec<f32>>,
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
    /// `AdamW` decoupled weight decay applied uniformly to every parameter
    /// (the shared `E_feat`, `b_feat`, and every per-axis head). Post-
    /// step shrinkage; doesn't enter the backward graph. `0.0` disables.
    pub weight_decay: f64,
    /// Global-norm gradient clip per `AdamW` step (`0.0` = off). Bounds the
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
pub(crate) fn load_or_compute_fisher_weights(
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
                    info!("Reusing cached NB-Fisher weights from {path}");
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
                Err(e) => warn!("Failed to load Fisher cache {path} ({e}); recomputing"),
            }
        }
    }

    info!("Computing NB-Fisher weights (block_size={block_size:?})...");
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
            warn!("Failed to save Fisher cache to {path}: {e}");
        } else {
            info!("Saved NB-Fisher weights to {path}");
        }
    }

    Ok(feat_weights)
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

pub(crate) fn stage_params(config: &FitConfig) -> TrainingParams {
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
        max_grad_norm: config.max_grad_norm,
    }
}
