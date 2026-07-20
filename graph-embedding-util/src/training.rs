//! Composite multi-axis count-NCE training loop.
//!
//! Each minibatch step samples one positive batch from *every* axis
//! (the per-cell axis plus every pseudobulk-level axis), computes the
//! NCE loss on each, and sums them with per-axis weights `λ_k`. A
//! single `AdamW` step then updates the shared `E_feat` / `b_feat` Vars
//! (gradients accumulate across all axes' losses naturally — they
//! reference the same tensors) plus each axis's own cell-side Vars.
//!
//! Cell-cell NCE is an additional positive-pair term that attaches
//! only to the per-cell axis (it operates on real `E_cell`, not on
//! pseudobulk embeddings).
//!
//! Polls `stop` at minibatch boundaries so SIGINT cleanly returns to
//! the caller for output finalization.

use crate::coarsen::AxisCoarsenings;
use crate::data::UnifiedData;
use crate::feature_network::FeatureNetworkSmoother;
use crate::fit::lineage::PbLineageLevel;
use crate::fit::projection::PbLevelVelocity;
use crate::loss::{
    nce_loss, nce_loss_chain, nce_loss_identity, sample_chain_batch, sample_edge_batch,
    sample_per_batch_stratified_edge_batch, sample_stratified_edge_batch, ChainAxis,
    ChainBatchArgs, ChainSampler, EdgeBatch, EdgeBatchArgs, PerBatchSampler,
    PerBatchStratifiedCellSampler, PerBatchStratifiedEdgeBatchArgs, StratifiedEdgeBatchArgs,
    StratifiedSampler,
};
use crate::model::JointEmbedModel;
use crate::progress::new_progress_bar;
use candle_util::candle_core::{DType, Var};
use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_nn::AdamW;
use candle_util::candle_nn::VarMap;
use log::info;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// How `train_composite` mixes per-axis NCE losses each step.
///
/// - [`Sum`]: every step computes one minibatch per axis and sums the
///   losses (with per-axis `λ` weights). Lower-variance gradient per
///   step but `O(n_axes)` work per step.
/// - [`Sample`]: every step picks a single axis with probability
///   `λ_k / Σλ`, computes its NCE on one minibatch, and scales by
///   `Σλ`. Same expected gradient as `Sum`, higher variance per step,
///   `O(1)` work per step.
/// - [`Chain`]: every step samples a coordinated bottom-up chain — a
///   real `(cell, feature)` triplet whose pb ancestors at each level
///   are derived via the `cell→pb_per_level` map. All axes share the
///   same positive feature and negatives per chain; cell-side indices
///   differ per axis. One feature-side gather + one backward per step,
///   with coherent across-level gradients on `E_feat`. Lowest variance
///   per step, comparable per-step compute to `Sum`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompositeMode {
    #[default]
    Sum,
    Sample,
    Chain,
}

/// One axis in the composite training objective. `model` shares its
/// `e_feat` / `b_feat` Tensors with every other axis (same Var under
/// the hood); `e_cell` / `b_cell` are unique per axis.
pub struct CompositeAxis<'a> {
    pub model: &'a JointEmbedModel,
    pub unified: &'a UnifiedData,
    pub cell_axis: &'a AxisCoarsenings,
    pub sampler: AxisSampler<'a>,
    /// Mixing weight in the summed objective. Defaults to 1.0; tune
    /// down for axes that should have less influence on `E_feat`.
    pub lambda: f32,
    /// Short label for log lines (e.g. "cell", "`pb_l0`"). Cosmetic.
    pub label: &'a str,
}

/// Bipartite sampler attached to a composite axis. Three variants:
/// - `PerBatch`: flat per-batch positive draw weighted by `count`.
/// - `PerBatchStratified`: per-batch two-stage draw — cell by
///   `degree^α_cell`, feature within cell by `count`. Guarantees
///   per-cell coverage so rare/shallow cells aren't drowned by deeply
///   sequenced ones. Used by the cell axis by default.
/// - `Stratified`: single-sampler two-stage draw over pb's — pb by
///   `pb_size^α_pb`, feature within pb by `count`. Used by the
///   pb axes (one synthetic batch each).
pub enum AxisSampler<'a> {
    PerBatch(&'a [PerBatchSampler]),
    PerBatchStratified(&'a [PerBatchStratifiedCellSampler]),
    Stratified(&'a StratifiedSampler),
}

impl AxisSampler<'_> {
    fn is_empty(&self) -> bool {
        match self {
            Self::PerBatch(s) => s.is_empty(),
            Self::PerBatchStratified(s) => s.is_empty(),
            Self::Stratified(_) => false,
        }
    }

    /// Number of "draw units" on this axis — used by auto
    /// `--batches-per-epoch` (one weighted pass = `n_units / batch_size`).
    /// A cell axis exposes one unit per *cell* (summed across its per-batch
    /// samplers) so a "pass" sweeps every cell once; pb axes expose
    /// `active_pbs.len()`. (The cell axis previously reported the number of
    /// batches here, which starved per-cell training — the cell axis was
    /// invisible to the budget and got the same ~1 step/epoch as the pb
    /// axes despite having orders of magnitude more units.)
    #[must_use]
    pub fn n_units(&self) -> usize {
        match self {
            Self::PerBatch(s) => s.len(),
            Self::PerBatchStratified(s) => s.iter().map(|x| x.active_cells.len()).sum(),
            Self::Stratified(s) => s.active_pbs.len(),
        }
    }
}

pub struct TrainingParams {
    pub epochs: usize,
    /// `None` = auto: one weighted pass over the largest axis
    /// (`ceil(max_axis_units / batch_size)`). `Some(n)` = fixed budget.
    pub batches_per_epoch: Option<usize>,
    pub batch_size: usize,
    pub num_negatives: usize,
    pub seed: u64,
    pub composite_mode: CompositeMode,
    /// Which NCE objective the feature side trains with ([`crate::loss::NceObjective`]).
    /// Defaults to `Softmax` (InfoNCE, `faba gem`); `senna bge` / `pinto cage` set
    /// `Logistic`.
    pub objective: crate::loss::NceObjective,
    /// Explicit L2 penalty `λ · ‖E_feat‖_F²` on the shared feature
    /// embedding, added to the per-step composite loss before backward.
    /// `0.0` disables. Equivalent to a zero-mean Gaussian prior on
    /// `E_feat` with precision `2 · λ`.
    pub feature_embedding_l2: f32,
    /// Global-norm gradient clip per `AdamW` step (`0.0` = off). Bounds the
    /// update magnitude so embeddings don't inflate on NCE loss spikes.
    pub max_grad_norm: f32,
    /// L2 (ridge) penalty `λ · mean(δ_g²)` on the per-gene splice offset (factored
    /// β-sharing only). Shrinks `δ_g` toward 0 so the splice signal is explained
    /// on the cell axis unless a gene's nascent deviation genuinely lowers the
    /// loss — a dense prior that fits the (dense) per-gene γ structure and is
    /// well-behaved under AdamW. `0.0` disables (plain β-sharing, no `δ_g`).
    pub delta_l2: f32,
}

pub struct CompositeTrainContext<'a> {
    pub axes: &'a [CompositeAxis<'a>],
    pub dev: &'a Device,
    pub stop: &'a Arc<AtomicBool>,
    /// Per-level cell→pb mappings (coarsest-first; length = number of
    /// pb axes = `axes.len() - 1`). Required for `CompositeMode::Chain`;
    /// ignored otherwise. Comes from
    /// `MultilevelCollapseOut::cell_to_pb_per_level`.
    pub cell_to_pb_per_level: Option<&'a [Vec<usize>]>,
    /// Optional per-axis velocity-drift SEM term (lineage-DAG refine pass, fixed velocity-KNN
    /// structure). Aligned 1:1 with `axes`: `None` for axes with no lineage structure
    /// (the cell axis, and any pb level with no oriented edges), `Some` otherwise.
    /// When set, each `Some` term's penalty is added to the per-step loss so its
    /// pb embedding is pulled toward the velocity-consistent geometry.
    pub lineage_sem: Option<&'a [Option<PbSemTerm>]>,
    /// Optional SECOND per-axis SEM term: the θ-pseudotime DAG (same `PbSemTerm` form,
    /// but the drift is the θ-manifold pseudotime gradient, not velocity). Added
    /// alongside `lineage_sem`/`lineage_dag` so the embedding is shaped by both the
    /// velocity flow AND the dense identity manifold — robust where δ is sparse (a
    /// δ-less pb is dropped by the velocity graph but kept by this one). Aligned 1:1
    /// with `axes`; `None` (default) is a no-op.
    pub lineage_sem_theta: Option<&'a [Option<PbSemTerm>]>,
    /// Optional per-axis learnable pb-DAG term (lineage-DAG refine pass, learned). Aligned
    /// 1:1 with `axes` like `lineage_sem`, but its `W` is a learned adjacency
    /// co-optimized with the embedding. Mutually exclusive with `lineage_sem`.
    pub lineage_dag: Option<&'a [Option<PbDagTerm>]>,
    /// Apply the `lineage_dag` structure loss only every `lineage_dag_stride` steps
    /// (NCE still runs every step). The structure is a light prior on a warm-started
    /// `W`; hammering it every step over-shapes the embedding, so we sample the DATA
    /// side densely and the STRUCTURE side sparsely. `1` = every step; ignored unless
    /// `lineage_dag` is set.
    pub lineage_dag_stride: usize,
}

/// Device-side velocity-drift SEM term for one pb axis (lineage-DAG, fixed velocity-KNN
/// structure). Precomputes the per-edge source/target index tensors, weights, and
/// the constant drift `s·v̂_i`, so each training step is two `index_select`s plus
/// elementwise ops. Penalizes `Σ_{i→j} w_ij ‖e_j − e_i − s·v̂_i‖² · λ / Σw`, i.e.
/// a pb node should sit one velocity-step ahead of its parent along the flow.
pub struct PbSemTerm {
    /// `[E]` parent (source) node id per edge.
    src: Tensor,
    /// `[E]` child (target) node id per edge.
    dst: Tensor,
    /// `[E]` edge weight.
    w: Tensor,
    /// `[E, H]` constant drift `s·v̂_i` (unit velocity of the parent, scaled).
    drift: Tensor,
    /// `λ_sem / Σw` — the weight, folded with the `Σw` normalizer.
    scale: f64,
}

impl PbSemTerm {
    /// Build the device term for one level, or `None` when the level has no
    /// oriented edges (nothing to penalize). `step` is `s`, `weight` is `λ_sem`.
    pub fn new(
        level: &PbLineageLevel,
        h: usize,
        step: f32,
        weight: f32,
        dev: &Device,
    ) -> anyhow::Result<Option<Self>> {
        if level.edges.is_empty() {
            return Ok(None);
        }
        let e = level.edges.len();
        let mut src = Vec::with_capacity(e);
        let mut dst = Vec::with_capacity(e);
        let mut w = Vec::with_capacity(e);
        let mut drift = Vec::with_capacity(e * h);
        let mut wsum = 0f32;
        for &(i, j, wij) in &level.edges {
            src.push(i);
            dst.push(j);
            w.push(wij);
            wsum += wij;
            let vi = &level.velocity[i as usize * h..(i as usize + 1) * h];
            for &vk in vi {
                drift.push(step * vk);
            }
        }
        Ok(Some(Self {
            src: Tensor::from_vec(src, e, dev)?,
            dst: Tensor::from_vec(dst, e, dev)?,
            w: Tensor::from_vec(w, e, dev)?,
            drift: Tensor::from_vec(drift, (e, h), dev)?,
            scale: f64::from(weight) / f64::from(wsum.max(1e-8)),
        }))
    }
}

/// `tr(A) = Σ_i A_ii`, via the elementwise product with the identity `eye`.
fn trace(a: &Tensor, eye: &Tensor) -> anyhow::Result<Tensor> {
    Ok(a.mul(eye)?.sum_all()?)
}

/// DAGMA/NOTEARS-style acyclicity surrogate `Σ_{k=1}^K tr((W∘W)^k)/k! / P` — candle
/// has no differentiable log-det, so this truncated trace-of-powers of `W∘W` stands
/// in for `tr(e^{W∘W}) − P`: it is `≈ 0` iff `W` is acyclic (a nilpotent adjacency
/// has zero-trace powers) and strictly positive when `W` carries a cycle.
fn acyclicity_series(
    w_eff: &Tensor,
    eye: &Tensor,
    order: usize,
    p: usize,
) -> anyhow::Result<Tensor> {
    let a = w_eff.sqr()?; // W ∘ W, [P, P]
    let mut ak = a.clone();
    let mut h = trace(&ak, eye)?; // k = 1
    let mut fact = 1.0f64;
    for k in 2..=order {
        ak = ak.matmul(&a)?;
        fact *= k as f64;
        h = (h + trace(&ak, eye)?.affine(1.0 / fact, 0.0)?)?;
    }
    Ok(h.affine(1.0 / p as f64, 0.0)?)
}

/// Velocity-drift SEM penalty on one axis's pb embedding `e_cell`, differentiable
/// in `e_cell`: `λ · (Σ_ij w_ij ‖e_j − e_i − s·v̂_i‖²) / Σw`.
fn sem_penalty(e_cell: &Tensor, term: &PbSemTerm) -> anyhow::Result<Tensor> {
    let e_src = e_cell.index_select(&term.src, 0)?;
    let e_dst = e_cell.index_select(&term.dst, 0)?;
    let resid = e_dst.sub(&e_src)?.sub(&term.drift)?; // [E, H]
    let sq = resid.sqr()?.sum(1)?; // [E]
    let weighted = sq.mul(&term.w)?.sum_all()?; // scalar
    Ok(weighted.affine(term.scale, 0.0)?)
}

/// Hyperparameters for the learnable pb-DAG term (lineage-DAG, learned).
#[derive(Clone, Copy)]
pub struct PbDagParams {
    /// Velocity-drift step `s` in the SEM residual `e_j − Σ_i W_ij(e_i + s·v̂_i)`.
    pub step: f32,
    /// Weight on the SEM reconstruction residual.
    pub sem_weight: f32,
    /// Weight on the L1 sparsity penalty `‖W‖_1`.
    pub l1_weight: f32,
    /// Weight on the DAGMA-style acyclicity penalty.
    pub acyc_weight: f32,
    /// Truncation order `K` of the trace-of-powers acyclicity surrogate
    /// `Σ_{k=1}^K tr((W∘W)^k)/k!` (≈ `tr(e^{W∘W}) − P`, zero iff acyclic).
    pub acyc_order: usize,
}

impl Default for PbDagParams {
    fn default() -> Self {
        Self {
            step: 2.0,
            sem_weight: 1.0,
            l1_weight: 0.01,
            acyc_weight: 0.3,
            acyc_order: 4,
        }
    }
}

/// **Unified** learnable directed pb-DAG term for one level (lineage-DAG, learned):
/// one adjacency `W [P×P]` explaining BOTH the δ (velocity) and θ (identity) structures.
/// θ and δ aren't competing graphs — θ is the *topology* (which pbs are adjacent) and δ
/// the *orientation* (which way the arrow points). So a single `W` carries both:
/// - **δ → orientation:** the drift `s·û` and the forward mask (velocity-forward, with a
///   θ-pseudotime fallback where δ is sparse) set the direction.
/// - **θ → topology:** the L1 is θ-distance-weighted, so `W` is cheap only on θ-proximal
///   edges and settles onto the identity manifold.
///
/// [`Self::dag_loss`] is `sem·‖E − W_effᵀ(E + s·û)‖² + l1·Σ|W_eff|·d_θ + acyc·h(W_eff)`,
/// differentiable in both `E` and `W`. `W_eff = W ∘ fwd_mask`.
pub struct PbDagTerm {
    /// Learnable adjacency `[P, P]` (Var tensor).
    w: Tensor,
    /// Constant drift `s·û` `[P, H]`: parent's velocity step where δ is measured, θ-pseudotime
    /// gradient where it isn't (see [`Self::new`]).
    drift: Tensor,
    /// Identity `[P, P]` — used for the trace via `(A∘I).sum`.
    eye: Tensor,
    /// Forward candidate mask `[P, P]`: `1` where edge `i→j` is velocity-forward
    /// (`⟨θ_j − θ_i, v̂_i⟩ > 0`) OR a τ-forward θ-KNN edge; `0` otherwise. The θ-KNN union
    /// keeps δ-sparse pbs — which the velocity mask alone drops — in the DAG.
    fwd_mask: Tensor,
    /// θ-distance weights `[P, P]`: `d_θ(i,j) = ‖θ_i − θ_j‖`, weighting the L1 so `W`
    /// is cheap only on θ-proximal edges — this is what makes the single DAG explain
    /// the θ topology (δ handles orientation via `drift`/`fwd_mask`).
    dtheta: Tensor,
    params: PbDagParams,
    p: usize,
}

/// Inputs to [`PbDagTerm::new`] for one pb level: the warm-up structures the term
/// is built from (`vel`, `theta_dag`), the hyperparameters, and where to register
/// the learnable `W` (`var_name` in `varmap`, on `dev`).
pub struct PbDagTermSpec<'a> {
    /// Phase-2 warm-up readout for this level — `theta` fixes the topology and
    /// `delta` the orientation.
    pub vel: &'a PbLevelVelocity,
    /// θ-pseudotime lineage DAG for the same level: unions in the forward
    /// candidates the velocity mask misses, and supplies the fallback drift
    /// gradient where δ is absent.
    pub theta_dag: &'a crate::fit::lineage::PbLineageLevel,
    /// Embedding dimension `H`.
    pub h: usize,
    /// Step size and the SEM / L1 / acyclicity weights.
    pub params: PbDagParams,
    /// Name the learnable `W` Var is registered under in `varmap`.
    pub var_name: &'a str,
    pub varmap: &'a VarMap,
    pub dev: &'a Device,
    /// Warm-start for `W` (`[p×p]` row-major); `None` zero-initializes.
    pub w_init: Option<&'a [f32]>,
}

impl PbDagTerm {
    /// Build the learnable term for one level, registering the `W` Var under
    /// `spec.var_name`. Returns `None` when the level has fewer than two nodes. The
    /// forward-orientation mask and drift are fixed from the warm-up velocity/identity
    /// (`vel.theta` / `vel.delta`). `w_init` warm-starts `W` from a fixed structure
    /// (the velocity-oriented KNN, `[p×p]` row-major) so SGD refines from a correctly-
    /// oriented start; `None` zero-initializes (the DAGMA-clean but unstable start).
    pub fn new(spec: PbDagTermSpec<'_>) -> anyhow::Result<Option<Self>> {
        let PbDagTermSpec {
            vel,
            theta_dag,
            h,
            params,
            var_name,
            varmap,
            dev,
            w_init,
        } = spec;
        let p = vel.n_pb;
        if p < 2 {
            return Ok(None);
        }
        // Unit velocity v̂ per node (zero when ‖δ‖ ≈ 0), shared with the lineage graph.
        let (vhat, has_vel) = crate::fit::lineage::unit_velocity(vel, h);
        // Confidence-gated drift s·û: trust the velocity where it is measured, fall back
        // to the θ-pseudotime gradient ĝ (`theta_dag.velocity`) where δ is absent — so a
        // δ-sparse pb still drifts along the identity manifold rather than being dropped.
        let ghat = &theta_dag.velocity;
        let mut drift = vec![0f32; p * h];
        for i in 0..p {
            let src = if has_vel[i] {
                &vhat[i * h..(i + 1) * h]
            } else {
                &ghat[i * h..(i + 1) * h]
            };
            for c in 0..h {
                drift[i * h + c] = params.step * src[c];
            }
        }
        // Forward candidate mask: keep W_ij only where j is velocity-forward of i
        // (⟨θ_j − θ_i, v̂_i⟩ > 0). Undefined-velocity rows and the diagonal stay 0, so
        // the learned DAG is forward-oriented by construction. θ is the warm-up identity.
        let theta = &vel.theta;
        let mut fwd = vec![0f32; p * p];
        for i in 0..p {
            if !has_vel[i] {
                continue;
            }
            let ti = &theta[i * h..(i + 1) * h];
            let vi = &vhat[i * h..(i + 1) * h];
            for j in 0..p {
                if j == i {
                    continue;
                }
                let tj = &theta[j * h..(j + 1) * h];
                let f: f32 = (0..h).map(|c| (tj[c] - ti[c]) * vi[c]).sum();
                if f > 0.0 {
                    fwd[i * p + j] = 1.0;
                }
            }
        }
        // Union the θ-pseudotime DAG edges (τ-forward θ-KNN): defined wherever θ is, so
        // δ-sparse pbs — dropped by the velocity mask above — still get forward candidates
        // from the dense identity manifold. This is the topology δ can't supply.
        for &(i, j, _) in &theta_dag.edges {
            fwd[i as usize * p + j as usize] = 1.0;
        }
        // θ-distance weights `d_θ(i,j) = ‖θ_i − θ_j‖` for the topology-shaping L1.
        let mut dtheta = vec![0f32; p * p];
        for i in 0..p {
            let ti = &theta[i * h..(i + 1) * h];
            for j in 0..p {
                let tj = &theta[j * h..(j + 1) * h];
                dtheta[i * p + j] = (0..h).map(|c| (ti[c] - tj[c]).powi(2)).sum::<f32>().sqrt();
            }
        }
        // Normalize by the typical θ-KNN edge length so a θ-PROXIMAL edge costs ≈1 —
        // preserving the plain-L1 scale — and only θ-FAR edges are penalized more. Without
        // this the raw ‖θ‖ (O(1–10)) inflates the L1 ~mean(d_θ)×, over-sparsifying W and
        // collapsing the lineage branches. The weighting is relative, not absolute.
        let scale = {
            let (mut sum, mut n) = (0f32, 0u32);
            for &(i, j, _) in &theta_dag.edges {
                sum += dtheta[i as usize * p + j as usize];
                n += 1;
            }
            (sum / n.max(1) as f32).max(1e-6)
        };
        for d in dtheta.iter_mut() {
            *d /= scale;
        }
        let eye: Vec<f32> = (0..p * p)
            .map(|idx| f32::from(idx / p == idx % p))
            .collect();

        let w0 = match w_init {
            Some(init) => {
                anyhow::ensure!(
                    init.len() == p * p,
                    "w_init length {} != p·p {}",
                    init.len(),
                    p * p
                );
                Tensor::from_vec(init.to_vec(), (p, p), dev)?
            }
            None => Tensor::zeros((p, p), DType::F32, dev)?,
        };
        let w_var = Var::from_tensor(&w0)?;
        varmap
            .data()
            .lock()
            .unwrap()
            .insert(var_name.to_string(), w_var.clone());

        Ok(Some(Self {
            w: w_var.as_tensor().clone(),
            drift: Tensor::from_vec(drift, (p, h), dev)?,
            eye: Tensor::from_vec(eye, (p, p), dev)?,
            fwd_mask: Tensor::from_vec(fwd, (p, p), dev)?,
            dtheta: Tensor::from_vec(dtheta, (p, p), dev)?,
            params,
            p,
        }))
    }

    /// Full learnable-DAG loss for this level, differentiable in `e_cell` and `W`.
    pub fn dag_loss(&self, e_cell: &Tensor) -> anyhow::Result<Tensor> {
        let w_eff = self.w.mul(&self.fwd_mask)?; // forward-only (also zeros self-loops)
                                                 // SEM reconstruction: E − W_effᵀ · (E + s·v̂). `fwd_mask[i,j]=1` marks j as
                                                 // velocity-forward of i, so W_eff[i,j] is the edge i→j (i earlier, j later). We
                                                 // must reconstruct the CHILD j from its PARENT i (`e_j ≈ e_i + s·v̂_i`), i.e.
                                                 // `recon[j] = Σ_i W_eff[i,j]·(e_i + drift_i)` — that is `W_effᵀ·(E+drift)`, NOT
                                                 // `W_eff·(...)`. Without the transpose each node is reconstructed from its
                                                 // successors and the whole velocity-drift convention runs backward (the velocity-KNN SEM,
                                                 // `e_j − e_i − s·v̂_i`, is the correct forward form we match here). Normalize PER
                                                 // NODE (sum over H, mean over P) — `mean_all` would divide the W-gradient by P·H.
        let parent = e_cell.add(&self.drift)?; // [P, H]
        let recon = w_eff.t()?.matmul(&parent)?; // [P, H] — Wᵀ: child from parent
        let sem = e_cell.sub(&recon)?.sqr()?.sum(1)?.mean(0)?;
        // θ-weighted L1: Σ_ij |W_eff_ij|·d_θ(i,j). Plain L1 is scale-free over edges;
        // weighting by θ-distance makes a long-range (θ-far) edge cost more than a
        // θ-proximal one, so the learned DAG settles onto the identity manifold — this
        // is the single `W` explaining the θ topology while drift/mask carry δ.
        let l1 = w_eff.abs()?.mul(&self.dtheta)?.sum(1)?.mean(0)?;
        // Acyclicity surrogate.
        let acyc = self.acyclicity(&w_eff)?;
        let loss = sem.affine(f64::from(self.params.sem_weight), 0.0)?;
        let loss = (loss + l1.affine(f64::from(self.params.l1_weight), 0.0)?)?;
        let loss = (loss + acyc.affine(f64::from(self.params.acyc_weight), 0.0)?)?;
        Ok(loss)
    }

    /// DAGMA/NOTEARS-style acyclicity surrogate (see [`acyclicity_series`]).
    fn acyclicity(&self, w_eff: &Tensor) -> anyhow::Result<Tensor> {
        acyclicity_series(w_eff, &self.eye, self.params.acyc_order, self.p)
    }

    /// The current effective forward adjacency `W_eff = W ∘ fwd_mask` as a host
    /// `[P, P]` row-major buffer — consumed by the phase-2 cell lift (cell-lift).
    pub fn w_dense(&self) -> anyhow::Result<Vec<f32>> {
        Ok(self.w.mul(&self.fwd_mask)?.flatten_all()?.to_vec1()?)
    }
}

/// Returns the final epoch's mean composite loss (an NCE ≈ neg-log-likelihood proxy),
/// used as a fit-hygiene signal by the lineage QC.
pub fn train_composite(
    ctx: &CompositeTrainContext,
    opt: &mut AdamW,
    params: &TrainingParams,
    smoother: Option<&mut FeatureNetworkSmoother>,
) -> anyhow::Result<f32> {
    assert!(!ctx.axes.is_empty(), "composite training needs >= 1 axis");

    // Shared style — consistent with every other faba/senna progress bar
    // (`[elapsed] bar pos/len (eta) msg`).
    let prog_bar = new_progress_bar(params.epochs as u64);

    let mut rng = StdRng::seed_from_u64(params.seed);
    let refresh_every = smoother.as_ref().map_or(0, |s| s.refresh_epochs);
    let mut smoother = smoother;
    // Global step index (across epochs) — gates the sparse structure-side update.
    let mut global_step = 0usize;
    let dag_stride = ctx.lineage_dag_stride.max(1);
    let mut last_avg = 0f32; // final-epoch mean loss, returned as the fit-hygiene signal

    // Smoother refreshes against the *shared* E_feat — pull it from the
    // first axis (every axis points at the same tensor).
    let shared_e_feat = ctx.axes[0].model.e_feat.clone();
    // Shared per-gene splice offset δ_g (factored splice models), for the L2 (ridge)
    // penalty below. `None` for free / plain-β-sharing models.
    let shared_delta = ctx.axes[0]
        .model
        .factor
        .as_ref()
        .and_then(|f| f.splice_delta.as_ref().map(|(delta, _)| delta.clone()));
    // Pre-build the axis sampler for `Sample` mode. Reused every step;
    // weights = `λ_k`, so picking axis `k` happens with probability
    // `λ_k / Σλ`. The `Σλ` scale gets applied to the chosen axis's loss
    // so `E_k[L_step] = Σ_k λ_k · L_k` matches `Sum` mode in expectation.
    let lambda_sum: f32 = ctx.axes.iter().map(|a| a.lambda).sum();
    let axis_picker: Option<WeightedIndex<f32>> = if params.composite_mode == CompositeMode::Sample
    {
        let weights: Vec<f32> = ctx.axes.iter().map(|a| a.lambda.max(1e-8)).collect();
        Some(WeightedIndex::new(weights).expect("non-empty axis weights"))
    } else {
        None
    };

    // Resolve `batches_per_epoch`: explicit override, or auto = one
    // weighted pass over the largest axis. `n_units` is per-cell for the
    // cell axis and `active_pbs.len()` for the pb axes.
    let max_axis_units = ctx
        .axes
        .iter()
        .map(|a| a.sampler.n_units())
        .max()
        .unwrap_or(0);
    let batches_per_epoch = params.batches_per_epoch.unwrap_or_else(|| {
        let bs = params.batch_size.max(1);
        max_axis_units.div_ceil(bs).max(1)
    });
    log::info!(
        "train_composite: {} epochs × {} batches (auto={}, max_axis_units={})",
        params.epochs,
        batches_per_epoch,
        params.batches_per_epoch.is_none(),
        max_axis_units,
    );

    for epoch in 0..params.epochs {
        if let Some(sm) = smoother.as_deref_mut() {
            if refresh_every > 0 && epoch % refresh_every == 0 {
                sm.refresh(&shared_e_feat, ctx.dev)?;
            }
        }

        // Loss kept **on-device** and synced to a scalar once per epoch (not
        // per minibatch) — `detach()` keeps the running sum off the autograd
        // graph so each step's forward graph is still freed immediately,
        // while avoiding a per-step GPU→CPU stall. Mirrors faba gem.
        let mut loss_acc: Option<Tensor> = None;
        let mut n_steps = 0usize;

        for _ in 0..batches_per_epoch {
            let loss = match params.composite_mode {
                CompositeMode::Sum => sum_step(ctx, &mut rng, params, smoother.as_deref())?,
                CompositeMode::Sample => sample_step(
                    ctx,
                    &mut rng,
                    params,
                    smoother.as_deref(),
                    axis_picker.as_ref().unwrap(),
                    lambda_sum,
                )?,
                CompositeMode::Chain => chain_step(ctx, &mut rng, params, smoother.as_deref())?,
            };
            let Some(mut loss) = loss else { continue };
            if params.feature_embedding_l2 > 0.0 {
                // `mean_all` keeps λ scale-invariant across `D · H`.
                let l2 = shared_e_feat
                    .sqr()?
                    .mean_all()?
                    .affine(f64::from(params.feature_embedding_l2), 0.0)?;
                loss = (loss + l2)?;
            }
            // L2 (ridge) shrinkage on the per-gene splice offset δ_g (factored
            // models with a splice split). `mean(δ_g²)` keeps λ scale-invariant
            // across `G · H` (mirrors the feature-embedding L2 above).
            if let (Some(delta), l2) = (&shared_delta, params.delta_l2) {
                if l2 > 0.0 {
                    let pen = delta.sqr()?.mean_all()?.affine(f64::from(l2), 0.0)?;
                    loss = (loss + pen)?;
                }
            }
            // Velocity-drift SEM residual (lineage-DAG refine pass). One penalty per
            // pb axis with oriented edges, pulling its pb embedding toward the
            // velocity-consistent geometry; gradients reach `E_feat` through the NCE
            // coupling. `None` (default) is a no-op → byte-identical training.
            if let Some(terms) = ctx.lineage_sem {
                for (axis, term) in ctx.axes.iter().zip(terms) {
                    if let Some(t) = term {
                        loss = (loss + sem_penalty(&axis.model.e_cell, t)?)?;
                    }
                }
            }
            // θ-pseudotime DAG: the SECOND lineage term, same `sem_penalty` form but
            // drifting along the θ-manifold pseudotime instead of velocity. Added every
            // step alongside the velocity term (fixed structure, like `lineage_sem`).
            if let Some(terms) = ctx.lineage_sem_theta {
                for (axis, term) in ctx.axes.iter().zip(terms) {
                    if let Some(t) = term {
                        loss = (loss + sem_penalty(&axis.model.e_cell, t)?)?;
                    }
                }
            }
            // Learnable pb-DAG (learned-DAG): the SEM/acyclicity/sparsity/orientation loss on
            // the learned `W`. Applied only every `dag_stride` steps — the DATA side
            // (NCE) trains every step, the STRUCTURE side (`W`) only occasionally, so a
            // warm-started `W` is nudged rather than hammered (over-shaping collapse).
            if let Some(terms) = ctx.lineage_dag {
                if global_step.is_multiple_of(dag_stride) {
                    for (axis, term) in ctx.axes.iter().zip(terms) {
                        if let Some(t) = term {
                            loss = (loss + t.dag_loss(&axis.model.e_cell)?)?;
                        }
                    }
                }
            }
            // Backward + optional global-norm gradient clip + step.
            candle_util::grad_clip::clipped_backward_step(
                opt,
                &loss,
                f64::from(params.max_grad_norm),
            )?;
            let ld = loss.detach();
            loss_acc = Some(match loss_acc.take() {
                None => ld,
                Some(a) => (a + ld)?,
            });
            n_steps += 1;
            global_step += 1;

            if ctx.stop.load(Ordering::Relaxed) {
                break;
            }
        }

        // Single GPU→CPU sync per epoch.
        let avg = match &loss_acc {
            Some(t) => t.to_scalar::<f32>()? / n_steps.max(1) as f32,
            None => 0f32,
        };
        last_avg = avg;
        prog_bar.set_message(format!("loss={avg:.3}"));
        prog_bar.inc(1);
        // Every-epoch info; senna/pinto's `--verbose` flag raises the
        // log level to `info`, so this is gated by the user's choice
        // there. Quiet runs (warn level) suppress it.
        info!(
            "epoch {}/{}: composite loss={:.3}",
            epoch + 1,
            params.epochs,
            avg
        );

        if ctx.stop.load(Ordering::SeqCst) {
            prog_bar.finish_and_clear();
            info!(
                "Stopping early at epoch {}/{} — finalizing outputs",
                epoch + 1,
                params.epochs
            );
            return Ok(last_avg);
        }
    }
    prog_bar.finish_and_clear();

    Ok(last_avg)
}

/// One step of `CompositeMode::Sum` — sample a minibatch from every
/// axis, compute each axis's NCE loss, return the λ-weighted sum.
fn sum_step(
    ctx: &CompositeTrainContext,
    rng: &mut StdRng,
    params: &TrainingParams,
    smoother: Option<&FeatureNetworkSmoother>,
) -> anyhow::Result<Option<Tensor>> {
    let mut total_loss: Option<Tensor> = None;
    for axis in ctx.axes {
        let Some(loss) = single_axis_step(axis, rng, params, smoother, ctx.dev)? else {
            continue;
        };
        let scaled = (loss * f64::from(axis.lambda))?;
        total_loss = Some(match total_loss {
            Some(prev) => (prev + scaled)?,
            None => scaled,
        });
    }
    Ok(total_loss)
}

/// One step of `CompositeMode::Sample` — pick a single axis weighted
/// by λ and run its NCE forward. Multiplied by `Σλ` so the per-step
/// gradient is unbiased for the same multi-task objective as `Sum`.
fn sample_step(
    ctx: &CompositeTrainContext,
    rng: &mut StdRng,
    params: &TrainingParams,
    smoother: Option<&FeatureNetworkSmoother>,
    axis_picker: &WeightedIndex<f32>,
    lambda_sum: f32,
) -> anyhow::Result<Option<Tensor>> {
    let axis_idx = axis_picker.sample(rng);
    let axis = &ctx.axes[axis_idx];
    let Some(loss) = single_axis_step(axis, rng, params, smoother, ctx.dev)? else {
        return Ok(None);
    };
    Ok(Some((loss * f64::from(lambda_sum))?))
}

/// One step of `CompositeMode::Chain` — sample a coordinated chain
/// batch (real cell-axis triplets, with pb ancestors derived from the
/// stored cell→pb maps), then score every axis (cell + each pb level)
/// with the same shared positive feature and shared negatives. One
/// `nce_loss_chain` call returns the λ-weighted sum across all axes.
fn chain_step(
    ctx: &CompositeTrainContext,
    rng: &mut StdRng,
    params: &TrainingParams,
    smoother: Option<&FeatureNetworkSmoother>,
) -> anyhow::Result<Option<Tensor>> {
    let cell_to_pb = ctx.cell_to_pb_per_level.ok_or_else(|| {
        anyhow::anyhow!(
            "CompositeMode::Chain requires CompositeTrainContext.cell_to_pb_per_level = Some(..)"
        )
    })?;
    let cell_axis = ctx
        .axes
        .first()
        .ok_or_else(|| anyhow::anyhow!("CompositeMode::Chain needs the cell axis as axes[0]"))?;
    let pb_axes = &ctx.axes[1..];
    anyhow::ensure!(
        pb_axes.len() == cell_to_pb.len(),
        "Chain mode: {} pb axes vs {} cell→pb levels — must match",
        pb_axes.len(),
        cell_to_pb.len()
    );

    // Chain mode accepts either flat per-batch or stratified per-batch
    // cell samplers — both produce real `(cell, feature)` positives that
    // can be walked up the pb tree. Stratified is the recommended default.
    let chain = match cell_axis.sampler {
        AxisSampler::PerBatch(samplers) => {
            if samplers.is_empty() {
                return Ok(None);
            }
            anyhow::ensure!(
                !cell_axis.unified.triplets.is_empty(),
                "flat PerBatch cell sampler needs a materialized edge list, but \
                 unified.triplets is empty (the streaming PerBatchStratified path \
                 leaves it empty) — call materialize_cell_triplets() to revive the \
                 flat path"
            );
            let id = rng.random_range(0..samplers.len());
            let bs = &samplers[id];
            let chain_sampler = ChainSampler {
                batch_sampler: bs,
                cell_to_pb_per_level: cell_to_pb,
            };
            sample_chain_batch(
                ChainBatchArgs {
                    triplets: &cell_axis.unified.triplets,
                    sampler: &chain_sampler,
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                rng,
            )
        }
        AxisSampler::PerBatchStratified(samplers) => {
            if samplers.is_empty() {
                return Ok(None);
            }
            let id = rng.random_range(0..samplers.len());
            let bs = &samplers[id];
            sample_chain_batch_stratified(bs, params.batch_size, params.num_negatives, rng)
        }
        AxisSampler::Stratified(_) => {
            anyhow::bail!(
                "Chain mode: cell axis must use PerBatch or PerBatchStratified \
                 (need real triplets). Got Stratified (pb-only)."
            );
        }
    };
    let crate::loss::ChainBatch { leaf_cells, feats } = chain;
    let b = leaf_cells.len();

    // Build all per-axis cell-side index tensors up front. The cell
    // axis uses leaf_cells directly; each pb axis derives its indices
    // by mapping leaf_cells through that level's cell→pb on host
    // *before* moving leaf_cells into a tensor (avoids a `to_vec1`
    // round-trip back from device). One Vec→Tensor allocation per
    // axis per step instead of two (the loss function used to clone
    // again to call `Tensor::from_vec` itself).
    let mut idx_tensors: Vec<Tensor> = Vec::with_capacity(ctx.axes.len());
    for c2p in cell_to_pb {
        let pb_ids: Vec<u32> = leaf_cells.iter().map(|&c| c2p[c as usize] as u32).collect();
        idx_tensors.push(Tensor::from_vec(pb_ids, b, ctx.dev)?);
    }
    let cell_idx_tensor = Tensor::from_vec(leaf_cells, b, ctx.dev)?;

    let mut chain_axes: Vec<ChainAxis> = Vec::with_capacity(ctx.axes.len());
    chain_axes.push(ChainAxis {
        e_cell: &cell_axis.model.e_cell,
        b_cell: &cell_axis.model.b_cell,
        indices: &cell_idx_tensor,
        lambda: cell_axis.lambda,
        label: cell_axis.label,
    });
    for (i, axis) in pb_axes.iter().enumerate() {
        chain_axes.push(ChainAxis {
            e_cell: &axis.model.e_cell,
            b_cell: &axis.model.b_cell,
            indices: &idx_tensors[i],
            lambda: axis.lambda,
            label: axis.label,
        });
    }

    let chain_loss = nce_loss_chain(
        &cell_axis.model.e_feat,
        &cell_axis.model.b_feat,
        feats,
        &chain_axes,
        smoother,
        params.objective,
        ctx.dev,
    )?;
    Ok(Some(chain_loss))
}

/// Chain-batch sampler for the stratified per-batch cell sampler.
/// Mirrors `loss::sample_chain_batch` but draws each leaf
/// `(cell, feature)` via the two-stage `cell_picker` → `per_cell` path
/// instead of a flat triplet pick. Shared negatives come from the same
/// per-batch feature pool, so downstream `nce_loss_chain` consumes the
/// same `ChainBatch` shape regardless of which sampler produced it.
fn sample_chain_batch_stratified(
    bs: &PerBatchStratifiedCellSampler,
    batch_size: usize,
    n_negatives: usize,
    rng: &mut StdRng,
) -> crate::loss::ChainBatch {
    let mut leaf_cells = Vec::with_capacity(batch_size);
    let mut fine_feats = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let lc = bs.cell_picker.sample(rng);
        let c = bs.active_cells[lc];
        let pf = &bs.per_cell[lc];
        let lf = pf.picker.sample(rng);
        let f = pf.features[lf];
        leaf_cells.push(c);
        fine_feats.push(f);
    }

    let mut neg_feats = Vec::with_capacity(batch_size * n_negatives);
    for _ in 0..(batch_size * n_negatives) {
        let local = bs.neg.sample(rng);
        neg_feats.push(bs.feature_pool[local]);
    }

    crate::loss::ChainBatch {
        leaf_cells,
        feats: crate::loss::ChainFeatureSide {
            fine_feats,
            neg_feats,
            n_negatives,
        },
    }
}

/// Sample a minibatch from a single axis and compute its bipartite NCE
/// loss (taking the identity fast path when the axis has identity
/// coarsening). Returns `None` when the axis has no positives to sample.
fn single_axis_step(
    axis: &CompositeAxis,
    rng: &mut StdRng,
    params: &TrainingParams,
    smoother: Option<&FeatureNetworkSmoother>,
    dev: &Device,
) -> anyhow::Result<Option<Tensor>> {
    if axis.sampler.is_empty() {
        return Ok(None);
    }
    let n_seeds = axis.cell_axis.coarsenings.len();
    if n_seeds == 0 {
        return Ok(None);
    }
    let seed_k = if n_seeds == 1 {
        0
    } else {
        rng.random_range(0..n_seeds)
    };
    let cc = &axis.cell_axis.coarsenings[seed_k];

    let batch: EdgeBatch = match axis.sampler {
        AxisSampler::PerBatch(samplers) => {
            anyhow::ensure!(
                !axis.unified.triplets.is_empty(),
                "flat PerBatch sampler needs a materialized edge list, but \
                 unified.triplets is empty (streaming PerBatchStratified leaves it \
                 empty) — call materialize_cell_triplets() to revive the flat path"
            );
            let id = rng.random_range(0..samplers.len());
            let bs = &samplers[id];
            sample_edge_batch(
                EdgeBatchArgs {
                    triplets: &axis.unified.triplets,
                    batch_sampler: bs,
                    cell_coarsening: cc,
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                rng,
            )
        }
        AxisSampler::PerBatchStratified(samplers) => {
            let id = rng.random_range(0..samplers.len());
            let bs = &samplers[id];
            sample_per_batch_stratified_edge_batch(
                PerBatchStratifiedEdgeBatchArgs {
                    sampler: bs,
                    cell_coarsening: cc,
                    batch_size: params.batch_size,
                    n_negatives: params.num_negatives,
                },
                rng,
            )
        }
        AxisSampler::Stratified(s) => sample_stratified_edge_batch(
            StratifiedEdgeBatchArgs {
                sampler: s,
                batch_size: params.batch_size,
                n_negatives: params.num_negatives,
            },
            rng,
        ),
    };

    let bip_loss = if axis.cell_axis.is_identity {
        nce_loss_identity(axis.model, batch, smoother, params.objective, dev)?
    } else {
        nce_loss(
            axis.model,
            batch,
            &cc.coarse_to_fine,
            smoother,
            params.objective,
            dev,
        )?
    };
    Ok(Some(bip_loss))
}

#[cfg(test)]
mod tests;
