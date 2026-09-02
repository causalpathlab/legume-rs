use super::lift::{CellLineage, LineageQc};
use super::projection::PbLevelVelocity;
use crate::model::JointEmbedModel;
use crate::training::{ModuleTrainParams, TrainingParams};
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use data_beans_alg::refine_multilevel::RefineParams;

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

/// Fraction of `epochs` the lineage warm-up (phase 1) gets when `--lineage-dag`
/// is on; the DAG refine takes the remainder, so the two passes **share one**
/// `epochs` budget instead of each taking the full count (which doubled the
/// training). The warm-up must be long enough that the pb velocity readout can
/// orient the DAG — that is exactly what this fraction trades. Off the lineage
/// path phase 1 keeps the whole budget and the run is byte-identical.
pub(crate) const LINEAGE_WARMUP_FRAC: f64 = 0.5;

/// Fraction of `epochs` the module membership is held at its warm start when
/// [`GeneModuleConfig::warmup_epochs`] is not given.
pub(crate) const MODULE_WARMUP_FRAC: f64 = 0.25;

/// Hyperparameter / configuration bundle for [`fit`]. Constructed by
/// each caller from its own CLI arguments — this crate doesn't import
/// `clap`.
pub struct FitConfig {
    pub embedding_dim: usize,
    /// Batch labels to anchor the cross-batch counterfactual on — a prior
    /// run's carried pseudobulks. Maps to [`MultilevelParams::anchor_batches`]
    /// (greedy batch correction: new batches corrected toward the anchor
    /// frame, the frame never re-adjusted).
    pub anchor_batches: Option<Vec<Box<str>>>,
    /// Batch labels whose columns are mixtures over cell states — maps to
    /// [`MultilevelParams::bulk_batches`]. Greedy: they are corrected toward
    /// the non-bulk (cell) frame and never serve as its counterfactual.
    pub bulk_batches: Option<Vec<Box<str>>>,
    /// Carry the finest collapse level (posterior + cell → pb membership) out
    /// on [`FitOutput::finest_collapse`], retaining its sufficient statistics
    /// even under the memory-lean calibration. `senna bge --emit-pb-reference`
    /// serializes it as the next round's carried reference.
    pub emit_finest_collapse: bool,
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
    /// See [`crate::training::TrainingParams::gpu_mem_fraction`]:
    /// `Some(frac)` lets a CUDA run shrink `batch_size` to fit memory.
    pub gpu_mem_fraction: Option<f32>,
    pub num_negatives: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub device: Device,
    /// Streaming block size for column-block I/O. `None` falls back to
    /// `matrix_util::utils::default_block_size(n_features)` which
    /// clamps to 100 for large feature counts — that's tiny on
    /// rotational disks. Pass `Some(1024)` or higher when you have
    /// the RAM, especially without `--preload-data`.
    pub block_size: Option<usize>,
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
    /// L2 (ridge) penalty on the per-gene splice offset `δ_g` (factored β-sharing
    /// splice models only). `0.0` = plain β-sharing (no `δ_g`); `> 0` allocates a
    /// ridge-shrunk `δ_g` so unspliced rows embed as `β_g + δ_g`.
    /// See [`crate::model::FeatFactor`].
    pub delta_l2: f32,
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
    /// Optional per-gene β-sharing feature parameterization. When `Some`, the
    /// feature side is built as [`crate::model::FeatFactor`] (every feature row
    /// reuses its gene's `β_g`) instead of a free `E_feat` table, phase-2 identity
    /// is resolved on the spliced edges (raw `θ`), and the same pass emits the raw
    /// velocity increment `δ` (see [`FitOutput::cell_velocity`]).
    /// `None` = the standard free embedding (bge / Stage 0).
    pub feat_factor: Option<FeatFactorSpec>,
    /// Lineage-DAG path (gem β-sharing only). When `true`, [`fit`] runs the
    /// analytic **pseudobulk** velocity readout after phase 1 (identity `θ_pb` +
    /// velocity `δ_pb` per pb node per level) and returns it in
    /// [`FitOutput::pb_velocity`]; `δ_pb` orients the pb-DAG structure term. A
    /// no-op (and a warning) when `feat_factor` is `None`. `false` = current
    /// behaviour (bge and plain gem), byte-identical output.
    pub lineage_dag: bool,
    /// Smooth + confidence-gate the pb velocity readout `δ_pb` before it orients the
    /// lineage graph / SEM drift / cell-lift (see
    /// [`crate::fit::lineage::smooth_pb_velocity`]). Denoises `sign(δ_pb)` via θ-space
    /// neighbour averaging — neutral on clean data, robustness on noisy real velocity.
    /// Ignored when `lineage_dag` is `false`.
    pub lineage_smooth: bool,
    /// Within the lineage refine, build the pb structure as a **minimum spanning tree**
    /// oriented into a DAG ([`crate::fit::lineage::build_pb_lineage`] `mst`) instead of the
    /// dense velocity-KNN — a sparse single-tree lineage. Ignored when `lineage_dag` is
    /// `false`.
    pub lineage_mst: bool,
    /// Phase-2 velocity mode. When `true`, the per-cell identity `θ` and velocity `δ` are
    /// estimated **jointly** in one SGD (θ pulled by both spliced and unspliced tracks)
    /// instead of the default sequential θ-then-δ-with-θ-fixed. Only meaningful on the
    /// β-sharing (splice) path.
    pub joint_velocity: bool,
    /// Sample phase 1 instead of stopping at its SGD point estimate: a two-sided
    /// blocked Gibbs over the **pseudobulk** model, warm-started from that MAP
    /// (see [`crate::posterior::pb_gibbs`]). Runs between phase 1 and
    /// `materialize_e_feat`, and writes its posterior means back into the Vars,
    /// so everything downstream — phase 2, the dictionary, the co-embed — reads a
    /// refined fit rather than a second set of tables.
    ///
    /// `None` = SGD only, and the run is byte-identical to one built before this
    /// existed.
    pub pb_posterior: Option<crate::posterior::pb_gibbs::PbGibbsConfig>,
    /// On the β-sharing (splice) model, allow `z_δ = 1` only where `z_β = 1` —
    /// velocity is a deviation from the identity loading, so a gene should not
    /// move along a dim its identity does not load. Also breaks the symmetry that
    /// lets two independent gates split inclusion mass on a gene where only
    /// `β + δ` is identified. Ignored without `feat_factor`.
    pub pb_posterior_nested_delta: bool,
    /// NCE objective for the feature side ([`crate::loss::NceObjective`]). Defaults to
    /// `Softmax` (InfoNCE). Every CLI that exposes it — `senna gem`, `senna bge` and
    /// `pinto cage`, all as `--nce-objective` — also defaults to `Softmax`; `Logistic`
    /// is opt-in and is the historical bge loss, kept byte-identical when chosen.
    pub nce_objective: crate::loss::NceObjective,
    /// Optional per-gene spike-and-slab gate over the embedding dimensions (Bernoulli
    /// inclusion + Gaussian effect prior = graceful feature selection). `Some` enables
    /// it for both the free (`e_feat`) and factored (`β`) feature sides; `None`
    /// (default) = ungated. Its `σ(S)` output is the same estimand
    /// `posterior::dim_block` samples as a PIP. See [`FeatureGateConfig`] and
    /// [`crate::model::FeatureGateSpec`].
    pub feature_gate: Option<FeatureGateConfig>,
    /// Learned gene modules in front of the feature embedding
    /// ([`crate::model::FeatModules`]): `ρ_g = Σ_m π_gm μ_m + r_g` with a learned
    /// mixed membership, an exact cell–module softmax term, within-module NCE
    /// negatives and gene dropout at pooling time. `None` (default) = the free
    /// embedding, byte-identical to a build before this existed. Mutually
    /// exclusive with `feat_factor` and `pb_posterior`.
    pub gene_modules: Option<GeneModuleConfig>,
}

/// Caller-facing configuration of the learned gene modules.
#[derive(Clone, Debug)]
pub struct GeneModuleConfig {
    /// Number of modules `M`.
    pub n_modules: usize,
    /// Epochs the warm-start membership is held before it trains. `None` = a
    /// quarter of the epochs, at least one.
    pub warmup_epochs: Option<usize>,
    /// Per-step probability that a feature is hidden when the module counts are
    /// pooled (`0` = off).
    pub gene_dropout: f32,
    /// Weight of the exact cell–module term relative to the NCE.
    pub lambda_module: f32,
    /// Weight of the load-balance prior `KL(π̄ ‖ Uniform)`.
    pub lambda_balance: f32,
    /// Weight of the row-entropy penalty on the membership (`0` = off).
    pub lambda_entropy: f32,
    /// Ridge on the per-feature residual `r_g` — the module model's only per-row
    /// table, so this replaces `feature_embedding_l2`.
    pub residual_l2: f32,
    /// Units (cells or pseudobulks) pooled for the exact term per step per axis.
    pub units_per_step: usize,
    /// Share of a feature's warm-start membership on its k-means module.
    pub init_own_mass: f32,
}

impl Default for GeneModuleConfig {
    fn default() -> Self {
        Self {
            n_modules: 128,
            warmup_epochs: None,
            gene_dropout: 0.2,
            lambda_module: 1.0,
            lambda_balance: 1.0,
            lambda_entropy: 0.0,
            residual_l2: 0.1,
            units_per_step: 64,
            init_own_mass: 0.9,
        }
    }
}

/// Caller-provided spec for the per-gene β-sharing feature factorization. Lengths
/// of `row_to_gene` / `unspliced_rows` equal the unified feature count; the gene
/// count is derived as `max(row_to_gene) + 1` (dense ids).
pub struct FeatFactorSpec {
    /// row → gene index (length = n_features).
    pub row_to_gene: Vec<u32>,
    /// per-row modality flag — true for the unspliced rows. The feature side
    /// ignores it (spliced & unspliced both embed as `β_g`); phase 2 uses it to
    /// split each cell's edges for the dual axis-δ projection.
    pub unspliced_rows: Vec<bool>,
}

/// Caller-facing name for the gate spec. One type, not a parallel copy: an earlier
/// duplicate meant the model's doc was updated when the gate changed and the
/// config's was not, so a caller reading the public API got the deleted design.
pub use crate::model::FeatureGateSpec as FeatureGateConfig;

/// Trained model + its `VarMap`. The varmap is exposed so callers can
/// save checkpoints or re-run inference; the current caller (`senna
/// gbe`) only consumes `model`, so it sits unused but kept alive.
pub struct FitOutput {
    pub model: JointEmbedModel,
    /// The finest collapse level and its cell → pb membership, present iff
    /// [`FitConfig::emit_finest_collapse`] was set. The membership indexes
    /// the global cell ids of the `UnifiedData` the fit ran on.
    pub finest_collapse: Option<(data_beans_alg::collapse_data::CollapsedOut, Vec<usize>)>,
    pub varmap: VarMap,
    /// Un-normalized baseline MAP per-cell projection norm from phase 2 (`0`
    /// for cells with no observed features / when phase 2 was skipped). The
    /// empty-droplet cell QC reads this: empties solve to ≈0, real cells far
    /// above. The stored latent (`model.e_cell`) is the L2 *direction*; this
    /// norm is the un-normalized magnitude it was divided by.
    pub cell_nrms: Vec<f32>,
    /// Per-cell **raw velocity increment** `δ` from phase 2, present only when
    /// `feat_factor` was set (β-sharing spliced/unspliced model). Flattened
    /// `[n_cells × H]` row-major in global cell-id order. `δ` is the analytic
    /// Poisson-MAP shift explaining the cell's unspliced edges with the identity `θ`
    /// held fixed — magnitude = speed, direction = velocity (no normalization). The
    /// nascent state is `θ + δ` = `latent + velocity`. `0` for a cell missing either
    /// modality; `None` for a free (non-factored) model.
    pub cell_velocity: Option<Vec<f32>>,
    /// Per-level pseudobulk velocity readout (identity `θ_pb` + velocity `δ_pb`),
    /// present only when `lineage_dag` was set on a β-sharing model. One entry per
    /// collapse level (coarsest→finest). Consumed by the lineage-DAG structure
    /// term and the phase-2 cell lift. `None` otherwise.
    pub pb_velocity: Option<Vec<PbLevelVelocity>>,
    /// Phase-2 cell-lineage lift (cell-lift): per-cell pseudotime `τ_c` + fate + ambiguity,
    /// evaluated (no training) from the finest-level pb trajectory. `Some` only on the
    /// lineage-DAG path with a non-empty pb velocity readout; `None` otherwise.
    pub cell_lineage: Option<CellLineage>,
    /// Unsupervised per-run QC diagnostics + `underfit` hygiene floor (decisiveness,
    /// velocity coherence, fate count, ambiguity, likelihood, flag). For an agent to reject
    /// broken runs and inspect structure — NOT a validated quality ranker. Written as
    /// `{out}.lineage_qc.json`. `Some` alongside `cell_lineage`; `None` otherwise.
    pub lineage_qc: Option<LineageQc>,
    /// Phase-1 posterior: per-`(feature, dim)` inclusion probability and both
    /// sides' posterior-mean loadings. `Some` iff [`FitConfig::pb_posterior`] was
    /// set and the run reached the stage.
    ///
    /// The loadings here are also already written back into the model, so this is
    /// for the uncertainty — the PIP table and the per-dim hypers — not for
    /// re-deriving the fit.
    pub pb_posterior: Option<crate::posterior::pb_gibbs::PbGibbsResult>,
    /// Both gates' posteriors on the β-sharing (splice) model. `Some` in place of
    /// [`Self::pb_posterior`] when `feat_factor` was set.
    pub splice_posterior: Option<crate::posterior::pb_gibbs::SpliceGibbsResult>,
}

pub(crate) fn stage_params(config: &FitConfig) -> TrainingParams {
    TrainingParams {
        epochs: config.epochs,
        batches_per_epoch: config.batches_per_epoch,
        batch_size: config.batch_size,
        gpu_mem_fraction: config.gpu_mem_fraction,
        num_negatives: config.num_negatives,
        seed: config.seed,
        // bge is two-phase: phase 1 (pb axes, no cell axis) and phase 2
        // (single cell axis) both require `Sum`. Each phase sets its own
        // mode explicitly; this default just makes the value well-formed.
        objective: config.nce_objective,
        feature_embedding_l2: config.feature_embedding_l2,
        max_grad_norm: config.max_grad_norm,
        delta_l2: config.delta_l2,
        module: config.gene_modules.as_ref().map(|g| ModuleTrainParams {
            warmup_epochs: g
                .warmup_epochs
                .unwrap_or_else(|| ((config.epochs as f64) * MODULE_WARMUP_FRAC).ceil() as usize)
                .clamp(usize::from(config.epochs > 0), config.epochs),
            gene_dropout: g.gene_dropout,
            units_per_step: g.units_per_step,
            lambda_module: g.lambda_module,
            lambda_balance: g.lambda_balance,
            lambda_entropy: g.lambda_entropy,
            residual_l2: g.residual_l2,
        }),
    }
}
