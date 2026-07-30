//! Joint multiome embedding tables + bias terms + bilinear scoring.
//!
//! Two free embedding tables (`E_feat` over the unified feature axis,
//! `E_cell`) plus two bias vectors (`b_feat`, `b_cell`). Score for a
//! `(feature, cell)` edge under a Poisson rate model:
//!
//!   `score(f, c) = E_feat[f] · E_cell[c] + b_feat[f] + b_cell[c]`
//!
//! All callers (bge, gem, pinto) use the full score: the per-cell bias
//! `b_cell` absorbs library size, so it is trained in phase 1, re-fitted
//! analytically in phase 2, and written out.
//!
//! Features are addressed at fine resolution. The cell axis is
//! coarsened: cell embeddings are mean-pooled (per the batch's chosen
//! seed coarsening) over the fine children of each touched pb-sample.

use candle_util::candle_core::{DType, Device, Result, Tensor};
use candle_util::candle_nn::{self, VarBuilder, VarMap};
use matrix_util::rand_util::name_seed;
use matrix_util::traits::SampleOps;
use std::sync::{Arc, Mutex};

/// stdev of the embedding-table randn init (matches the former
/// `candle_nn::Init::Randn { stdev: 0.1 }`).
const INIT_STDEV: f32 = 0.1;

/// Bernoulli spike-and-slab feature gate — an **independent inclusion probability
/// per (gene, dim)**.
///
/// Each gene's feature loading is `Ẽ_g = γ_g ⊙ β_g`: a SELECTION weight `γ_g`
/// times a Gaussian EFFECT `β_g` (the loading magnitude). Variationally,
///
///   `Ẽ_{g,h} = σ(S_{g,h}/τ) · (E_{g,h} + σ_{g,h}·ε)`,  ε ~ N(0,1),
///
/// where `S` is the selection logit table (`s_feat` free / `s_beta` factored,
/// `[D × H]`), `E_g` is the effect posterior MEAN, and `σ_g` (= `exp` of the
/// log-std table) its posterior std. Training samples via the reparameterization;
/// output/materialize use the mean (`σ = 0`).
///
/// This is mean-field spike-and-slab: `q(z_{g,h}) = Bern(α_{g,h})`,
/// `q(β_{g,h}) = N(E,σ²)`, and the forward carries `E[z·β] = α·E`. No Gumbel or
/// hard-concrete relaxation is involved — the expectation IS differentiable.
///
/// # `α` is the same estimand the sampler reports as a PIP
///
/// That is the point of the Bernoulli form. `posterior::dim_block` samples
/// `z_{g,h}` and reports `P(z=1 | data)`; this gate optimizes `q(z=1)` for the same
/// coordinate. So `feature_selection.parquet` and `feature_pip.parquet` are one
/// quantity estimated two ways, and can be COMPARED — which the previous softmax
/// gate made impossible.
///
/// The predecessor normalized over GENES within a dim, giving every dim one unit of
/// selection mass. That made "the markers of dim `h`" a distribution, but it is a
/// simplex (`D−1` free parameters per dim) where a PIP table is a product of
/// Bernoullis (`D`), so no rescaling connects them: measured on one fit that both
/// trained and sampled, `Spearman(normalized gate, PIP) = 0.021`.
///
/// # Per-dim mass is now SOFT, via a learned `π_h`
///
/// The softmax pinned each dim's mass at exactly 1.000 by construction. A sigmoid
/// constrains nothing, so mass control moves into the prior: `KL(Bern(α) ‖ Bern(π_h))`
/// pulls each dim's expected mass `Σ_g α_{g,h}` toward `D·π_h`, with `π_h` learned
/// per dim exactly as `posterior::hyper::sample_pi0` learns it for the sampler.
///
/// This is a real weakening and the failure mode is on record: when per-dim mass last
/// went unconstrained, column sums on a 12k BMMC fit ran 167–536, a 3.2× spread, with
/// the top two dims taking 25.7% of all selected mass against 12.5% uniform. Report
/// the spread on any new fit rather than assuming `π_h` handles it.
///
/// # KL loss
///
/// Always added, at the fixed [`GATE_KL_WEIGHT`], amortized `1/B` over the
/// minibatch:
///
///   `loss += mean_{g,h}[ KL(Bern(α_{g,h}) ‖ Bern(π_h)) + α_{g,h}·KL(N(E,σ²) ‖ N(0,σ₀²)) ]`
///          `+ Σ_h −log Beta(π_h; a, b)`
///
/// with `σ₀² =` [`GATE_EFFECT_PRIOR_VAR`] and `(a,b) =`
/// [`GATE_PI_BETA_A`]/[`GATE_PI_BETA_B`]. Both mean-reduced terms are `O(1)` per
/// entry, so unlike the softmax form neither needs a `ln D` or `D` correction to
/// stay commensurate with the other — see [`JointEmbedModel::single_gate_kl`] for
/// why that matters and what it cost last time.
/// One gate's frozen inclusion table and its shared per-epoch draw cell.
/// `(pip, mask)`; either may be absent when that gate is not posterior-gated.
type GateTables<'a> = (Option<&'a Tensor>, Option<&'a Arc<Mutex<Option<Tensor>>>>);

/// Which gate a weight lookup is for.
///
/// A factored model carries TWO gates — the identity `β_g` and the velocity `δ_g` —
/// and they are different objects with their own inclusion probabilities (which is why
/// `posterior::pb_gibbs` samples them with separate `σ₀h²`/`π₀h`). The gate weight
/// functions take a `Tensor` of logits, and a `Tensor` cannot be identified, so the
/// caller must say which gate it is asking about. Inferring it would silently hand
/// `δ` the `β` mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateKind {
    Identity,
    Velocity,
}

#[derive(Clone, Copy, Debug)]
pub struct FeatureGateSpec {
    /// Gate temperature `τ` (`1.0` = plain sigmoid; `< 1` sharpens toward 0/1).
    pub temperature: f32,
}

/// Effect-prior variance `σ₀²` for the Gaussian KL `KL(N(μ,σ²) ‖ N(0,σ₀²))`.
pub const GATE_EFFECT_PRIOR_VAR: f64 = 1.0;
/// Fixed weight `λ` on the spike-and-slab KL. The gate is always variational;
/// the KL is applied amortized `λ/B` over the minibatch, so `1.0` gives the `1/B`
/// scale. A single source of truth rather than a per-run CLI knob.
pub const GATE_KL_WEIGHT: f64 = 1.0;
/// `Beta(a, b)` hyperprior on each dim's inclusion rate `π_h`.
///
/// **Mind the convention.** `posterior::hyper::sample_pi0` draws `π₀ = P(OFF)` under
/// `Beta(9, 1)`, centred on 0.9 OFF. `π_h` here is `P(ON)`, so the matching prior is
/// the mirror image, `Beta(1, 9)`, centred on 0.1 — the same 10% inclusion the
/// sampler assumes. With `a = 1` the `(a−1)·ln π` term vanishes and only
/// `(b−1)·ln(1−π)` survives, which is bounded as `π → 0`; that is deliberate, since
/// the `Dirichlet(α<1)` analogue the softmax gate rejected ran away at the boundary.
pub const GATE_PI_BETA_A: f64 = 1.0;
/// See [`GATE_PI_BETA_A`] — `b = 9` centres `π_h` on `a/(a+b) = 0.1`.
pub const GATE_PI_BETA_B: f64 = 9.0;
/// Keeps `π_h` and `α` off the 0/1 boundary so `ln π`, `ln(1−π)` stay finite.
/// Mirrors `posterior::hyper::PI0_EPS`, which does the same job for the sampler.
const GATE_PI_EPS: f64 = 1e-6;
/// Initial gate logit. `σ(4) ≈ 0.982`, so an untrained gate is ~the identity and a
/// fresh model's `Ẽ ≈ β` — the same "inert rather than biased" property the softmax
/// gate got from a zero init (`softmax(0)·D = 1` exactly).
///
/// **Do NOT init at 0 here, and note the asymmetry with the sampler.**
/// `σ(0) = 0.5` would halve the entire dictionary, which propagates into the NCE
/// scale and the phase-2 projection. And where `dim_block` cold-starts `z` all-OFF
/// so the data has to earn each dim, SGD cannot: a zeroed gate kills the gradient
/// reaching `β` through `α·β`, so the fit starts dead. Gibbs can turn a coordinate
/// back on from a likelihood ratio; gradient descent cannot recover from a zeroed
/// multiplier.
const GATE_LOGIT_INIT: f32 = 4.0;
/// Initial effect log-std (`σ_init = e^{-4.6} ≈ 0.01`, a near-deterministic start).
const GATE_LOGSTD_INIT: f32 = -4.6;
/// Clamp bound on the effect log-std in the forward — keeps `σ = exp(logstd)` and
/// `log σ²` finite (no overflow/underflow), so the reparam noise and the Gaussian KL
/// stay well-behaved. Applied at read time (gradient saturates past the bound).
const GATE_LOGSTD_CLAMP: f64 = 8.0;

/// Shape of the embedding tables.
pub struct ModelArgs {
    pub n_features: usize,
    pub n_cells: usize,
    pub embedding_dim: usize,
    /// Base seed for the reproducible randn init of any `None` embedding.
    pub seed: u64,
}

/// Initial values for [`JointEmbedModel::new_with_init`]. `None` for
/// either embedding falls back to randn; bias slices must be
/// dimensionally consistent with [`ModelArgs`].
pub struct ModelInit<'a> {
    pub e_feat: Option<&'a nalgebra::DMatrix<f32>>,
    pub e_cell: Option<&'a nalgebra::DMatrix<f32>>,
    pub b_feat: &'a [f32],
    pub b_cell: &'a [f32],
}

/// Inputs for [`JointEmbedModel::new_sharing_features`]. The feature
/// side (`e_feat` / `b_feat`) is provided pre-allocated and registered
/// in the shared `VarMap` so multiple heads can co-train it. Only the
/// cell side gets new Vars, namespaced by `var_prefix` so multiple
/// heads can coexist in one `VarMap` (e.g. `pb_l0`, `pb_l1`, ..., `cell`).
pub struct ShareFeaturesArgs<'a> {
    pub n_cells: usize,
    pub embedding_dim: usize,
    pub shared_e_feat: Tensor,
    pub shared_b_feat: Tensor,
    pub e_cell_init: Option<&'a nalgebra::DMatrix<f32>>,
    pub b_cell_init: &'a [f32],
    pub var_prefix: &'a str,
    /// Base seed for the reproducible randn init of the cell side when
    /// `e_cell_init` is `None`.
    pub seed: u64,
    /// Shared softmax-gate logits (free model, `[n_features, H]`).
    /// `None` when the gate is off. Every head references the SAME Var so AdamW
    /// updates it once (see [`FeatureGateSpec`]).
    pub shared_s_feat: Option<Tensor>,
    /// Shared raw feature Var kept reachable for the gated training gather (so a
    /// post-phase-1 materialize can bake the gate into `e_feat` without clobbering
    /// the source). `None` when the gate is off.
    pub shared_e_feat_raw: Option<Tensor>,
    /// Shared free-model effect log-std (`[n_features, H]`). `Some` only for a KL gate.
    pub shared_e_feat_logstd: Option<Tensor>,
    /// Shared per-dim inclusion rate `π_h` (`[H]`). Must be handed to every head:
    /// `train_composite` computes [`JointEmbedModel::gate_kl`] on `ctx.axes[0]`, which
    /// is a pb head — not the cell model — whenever `--phase1-cells-per-pb` is 0 (the
    /// default in both CLIs). Leaving it `None` makes `gate_kl` return `None` and the
    /// gate regularisation disappear from the loss with the build green.
    pub shared_gate_pi_logit: Option<Tensor>,
    /// Gate configuration shared across heads. `None` when the gate is off.
    pub gate: Option<FeatureGateSpec>,
}

/// Inputs for [`JointEmbedModel::new_factored`] — a per-gene β-sharing feature
/// parameterization (see [`FeatFactor`]). `row_to_gene[r]` is the gene index of
/// feature row `r` (length `n_features`); rows sharing a gene reuse one `β_g`.
pub struct FactoredInit<'a> {
    pub n_features: usize,
    pub n_cells: usize,
    pub embedding_dim: usize,
    pub n_genes: usize,
    pub row_to_gene: &'a [u32],
    pub b_feat: &'a [f32],
    pub b_cell: &'a [f32],
    /// Base seed for the reproducible randn init of `β` and the cell side.
    pub seed: u64,
    /// Per-row unspliced flag (`len == n_features`). When `Some`, a ridge-shrunk
    /// per-gene `δ_g` Var is allocated and added to the unspliced rows
    /// (spliced identity + nascent offset); `None` = plain β-sharing.
    pub unspliced_rows: Option<&'a [bool]>,
}

/// Optional per-gene β-sharing feature factorization (used by `faba gem`'s
/// spliced/unspliced model). Instead of a free `e_feat` row per feature, every
/// feature row reuses a per-GENE base embedding `β [G, H]`:
///
///   `e_feat[row] = β[gene(row)]`
///
/// so a gene's spliced rows embed as `β_g`. **Optionally** a per-gene splice
/// offset `δ_g` is carried for the unspliced rows:
///
///   `e_feat[row] = β_g + [row is unspliced] · δ_g`
///
/// so spliced = current-state identity `β_g` and unspliced = nascent `β_g + δ_g`.
/// `δ_g` is **L2 (ridge) shrunk** (phase-1 penalty), which resolves the otherwise-
/// ambiguous split against an equal-and-opposite cell-axis shift: the shrunk
/// gene-side `δ_g` absorbs the (dense) static per-gene nascent structure (the
/// "γ"), and the residual dynamics stay on the CELL axis as the phase-2 velocity
/// increment `δ_cell` (a raw Poisson-MAP shift with θ held fixed; see
/// `crate::fit::project_cells_phase2`). With
/// `delta = None` this reduces to plain β-sharing (spliced ≡ unspliced ≡ `β_g`).
/// `β` / `δ_g` are learnable `Var`s; `row_to_gene` / the unspliced mask are fixed.
/// The score/loss path composes the row→gene→(β,δ) gathers directly (no
/// full-table materialization per step); output/co-embed readers use the
/// `e_feat` field after [`JointEmbedModel::materialize_e_feat`].
#[derive(Clone)]
pub struct FeatFactor {
    /// Per-gene base embedding `[G, H]` (Var).
    pub beta: Tensor,
    /// `[n_features]` u32 (device): row → gene index.
    pub row_to_gene: Tensor,
    /// Optional per-gene splice offset, present as `(δ_g [G, H] Var, mask [n_features,
    /// 1])` together (they always co-exist). `δ_g` is added to the **unspliced** rows
    /// (the `mask` = 1/0 selector); L2-ridge in phase-1. `None` = plain β-sharing
    /// (spliced ≡ unspliced ≡ `β_g`).
    pub splice_delta: Option<(Tensor, Tensor)>,
    /// Optional per-gene softmax-gate logits `[G, H]` (see [`FeatureGateSpec`]),
    /// the IDENTITY gate on `β_g`. Gathered by `row_to_gene` alongside `β`/`δ`; `None`
    /// = ungated. Cloned with the factor so composite heads share it.
    pub s_beta: Option<Tensor>,
    /// Optional per-gene Gaussian-effect log-std `[G, H]` (the `β` gate's variational
    /// single-effect posterior std; `σ = exp`). `Some` iff the gate is on.
    pub beta_logstd: Option<Tensor>,
    /// Optional per-gene VELOCITY-gate logits `[G, H]` — the independent
    /// spike-and-slab gate on `δ_g` (the motion), mirroring `s_beta`. `Some`
    /// only when the gate is on AND `splice_delta` exists (velocity present); a gene
    /// with no motion has `σ(s_delta) → 0` → `δ̃_g ≈ 0` (not a driver).
    /// `σ(s_delta)` per feature row = the `velocity_selection` output.
    pub s_delta: Option<Tensor>,
    /// Optional per-gene velocity Gaussian-effect log-std `[G, H]` (the `δ` gate's
    /// variational posterior std). `Some` iff `s_delta` is.
    pub delta_logstd: Option<Tensor>,
    /// Frozen VELOCITY inclusion probabilities `[G, H]` from a `--posterior` run —
    /// the `δ` counterpart of [`JointEmbedModel::gate_pip`]. Separate table because
    /// `β`'s inclusion and `δ`'s are different quantities; sharing one would mask the
    /// motion by the identity's selection.
    pub delta_gate_pip: Option<Tensor>,
    /// This epoch's `z ~ Bern(delta_gate_pip)`, shared across axes like the identity
    /// mask. `None` ⇒ use the mean.
    pub delta_gate_mask: Option<Arc<Mutex<Option<Tensor>>>>,
    /// Per-dim inclusion rate logits `[H]` for the VELOCITY gate — `π_h = σ(·)`.
    /// Separate from the identity gate's, because a loading and a deviation from it
    /// are not on one scale. `Some` iff `s_delta` is.
    pub delta_pi_logit: Option<Tensor>,
}

impl FeatFactor {
    /// Materialize the full feature embedding `[n_features, H]` from `β` (plus
    /// `δ_g` on the unspliced rows). Stays in the autograd graph so gradients
    /// flow back to the `β` / `δ_g` Vars.
    fn e_feat(&self) -> Result<Tensor> {
        let base = self.beta.index_select(&self.row_to_gene, 0)?;
        match &self.splice_delta {
            Some((delta, mask)) => {
                let d = delta.index_select(&self.row_to_gene, 0)?; // [n_features, H]
                base.add(&d.broadcast_mul(mask)?) // + mask ⊙ δ_g on unspliced rows
            }
            None => Ok(base),
        }
    }
}

pub struct JointEmbedModel {
    /// Unified feature embedding (genes ∪ peaks). When `factor` is `Some`, this
    /// is a materialized snapshot of the per-gene `β` gathered to feature rows —
    /// refreshed by [`Self::materialize_e_feat`] after phase 1 so phase-2 /
    /// outputs read a fixed dictionary; the training loss never reads this field
    /// for a factored model — it gathers each batch's rows straight from `β`.
    pub e_feat: Tensor,
    pub e_cell: Tensor,
    pub b_feat: Tensor,
    pub b_cell: Tensor,
    /// Optional per-gene β-sharing feature parameterization (`None` = free `e_feat`).
    pub factor: Option<FeatFactor>,
    #[allow(dead_code)]
    pub embedding_dim: usize,
    /// Free-model softmax-gate logits `[n_features, H]` (see [`FeatureGateSpec`]).
    /// `None` for an ungated model, or for a factored one (its gate lives in
    /// `factor.s_beta`).
    pub s_feat: Option<Tensor>,
    /// Raw `e_feat` Var kept reachable for the gated training gather of a FREE model,
    /// so [`Self::materialize_e_feat`] can overwrite `e_feat` with the gated snapshot
    /// without corrupting the source the gather reads. `None` unless free + gated.
    pub e_feat_raw: Option<Tensor>,
    /// Free-model Gaussian-effect log-std `[n_features, H]` (variational single-effect
    /// posterior std; `σ = exp`). `Some` only when the KL gate is on (free model).
    pub e_feat_logstd: Option<Tensor>,
    /// **Jitter (posterior-informed dropout).** Frozen `[rows, H]` inclusion
    /// probabilities taken from a completed `--posterior` run — `P(z=1 | data)`, not a
    /// learned parameter. `Some` puts the model in jitter mode: [`Self::gate_weights`]
    /// stops consulting the learned logits entirely and returns a per-epoch Bernoulli
    /// draw from this table during training, or this table itself (the mean `E[z]`) at
    /// output.
    ///
    /// This is Hinton's dropout with an INFERRED keep-probability. Classic dropout picks
    /// one global rate `p` and scales weights by `p` at test time; here the rate is
    /// per-`(gene, dim)` and comes from a posterior, and the same rule applies —
    /// `z ~ Bern(pip)` while training, `pip ⊙ β` at output. The regularization story
    /// carries over unchanged (it breaks co-adaptation between genes); what differs is
    /// that the rates are estimated rather than chosen.
    ///
    /// This exists because the LEARNED Bernoulli gate does not train. Measured on BM1:
    /// the KL that has to drive selection is ~70x under-weighted against the true ELBO,
    /// and initializing the logits anywhere the gate is inert (`σ(4) ≈ 0.98`) puts them
    /// where `∂α/∂S = α(1−α)` passes 1/14 of the available gradient — so `α` never
    /// leaves its initialization, and for the 37% of genes never drawn as NCE positives
    /// it cannot, receiving exactly zero gradient. Sampling `z` is not subject to any of
    /// that: the selection is an INPUT, computed by a method that demonstrably works.
    pub gate_pip: Option<Tensor>,
    /// The current epoch's `z ~ Bern(gate_pip)` draw, `[rows, H]` of 0/1 in f32.
    ///
    /// Redrawn once per EPOCH, never per minibatch: `z` is a latent for the DATASET, so
    /// a per-batch draw would model it as if each minibatch had its own inclusion state
    /// and would add gradient variance for nothing. One draw per epoch makes each epoch
    /// a coherent sub-model.
    ///
    /// This is NOT Monte-Carlo EM, despite the resemblance. `gate_pip` is estimated
    /// once and frozen; a real MCEM would re-estimate `z`'s distribution against the
    /// updated `β` on every outer round. Here the selection never learns anything from
    /// the loading that SGD goes on to fit.
    ///
    /// SHARED across axes and interior-mutable on purpose. The composite fit runs
    /// several axes that share Vars but are separate `JointEmbedModel` values, and the
    /// training loop holds them behind `&`. One `Arc` cell means a single redraw is
    /// seen by every axis at once — so the axes cannot drift onto different sub-models
    /// within an epoch, which a per-model field would have allowed.
    pub gate_mask: Arc<Mutex<Option<Tensor>>>,
    /// Per-dim inclusion rate logits `[H]` for the IDENTITY gate — `π_h = σ(·)`, the
    /// SGD analogue of `posterior::hyper::sample_pi0`'s per-dim `π₀h`. Lives on the
    /// model rather than beside the logit table because it is indexed by dim, not by
    /// row, so free and factored models share one home for it. `Some` iff gated.
    ///
    /// Only the axis that computes [`Self::gate_kl`] needs it (training calls that on
    /// `axes[0]`), so it is deliberately NOT plumbed through [`ShareFeaturesArgs`] the
    /// way `s_feat` is.
    pub gate_pi_logit: Option<Tensor>,
    /// Gate configuration (`None` = ungated). Presence is the single "is gated" flag
    /// for both free (`s_feat`) and factored (`factor.s_beta`) models.
    pub gate: Option<FeatureGateSpec>,
}

impl JointEmbedModel {
    /// Construct with optional warm-start values for either embedding.
    /// Used by stage 1 across the multi-level curriculum so each level
    /// inherits `E_feat` from the previous level instead of restarting
    /// from randn.
    pub fn new_with_init(
        args: ModelArgs,
        init: &ModelInit,
        varmap: &VarMap,
        dev: &Device,
    ) -> Result<Self> {
        let e_feat = match init.e_feat {
            Some(m) => register_var_from_mat(varmap, dev, "e_feat", m)?,
            None => register_randn_seeded(
                varmap,
                dev,
                "e_feat",
                args.n_features,
                args.embedding_dim,
                args.seed,
            )?,
        };
        let e_cell = match init.e_cell {
            Some(m) => register_var_from_mat(varmap, dev, "e_cell", m)?,
            None => register_randn_seeded(
                varmap,
                dev,
                "e_cell",
                args.n_cells,
                args.embedding_dim,
                args.seed,
            )?,
        };
        let b_feat = register_var_from_slice(varmap, dev, "b_feat", init.b_feat)?;
        let b_cell = register_var_from_slice(varmap, dev, "b_cell", init.b_cell)?;

        Ok(Self {
            e_feat,
            e_cell,
            b_feat,
            b_cell,
            factor: None,
            embedding_dim: args.embedding_dim,
            s_feat: None,
            e_feat_raw: None,
            e_feat_logstd: None,
            gate_pip: None,
            gate_mask: Arc::new(Mutex::new(None)),
            gate_pi_logit: None,
            gate: None,
        })
    }

    /// Composite-training constructor: reuse pre-existing
    /// `shared_e_feat` / `shared_b_feat` Tensors (already registered as
    /// Vars in `varmap` by an earlier call to `new_with_init`) and
    /// allocate fresh cell-side Vars under `args.var_prefix` so multiple
    /// heads coexist in one `VarMap`. `AdamW` over `varmap.all_vars()` then
    /// updates the shared feature side once and each head's cell side
    /// independently.
    pub fn new_sharing_features(
        args: ShareFeaturesArgs,
        varmap: &VarMap,
        dev: &Device,
    ) -> Result<Self> {
        let ShareFeaturesArgs {
            n_cells,
            embedding_dim,
            shared_e_feat,
            shared_b_feat,
            e_cell_init,
            b_cell_init,
            var_prefix,
            seed,
            shared_s_feat,
            shared_e_feat_raw,
            shared_e_feat_logstd,
            shared_gate_pi_logit,
            gate,
        } = args;
        let e_name = format!("{var_prefix}_e_cell");
        let b_name = format!("{var_prefix}_b_cell");
        let e_cell = if let Some(m) = e_cell_init {
            register_var_from_mat(varmap, dev, &e_name, m)?
        } else {
            register_randn_seeded(varmap, dev, &e_name, n_cells, embedding_dim, seed)?
        };
        let b_cell = register_var_from_slice(varmap, dev, &b_name, b_cell_init)?;
        Ok(Self {
            e_feat: shared_e_feat,
            e_cell,
            b_feat: shared_b_feat,
            b_cell,
            factor: None,
            embedding_dim,
            s_feat: shared_s_feat,
            e_feat_raw: shared_e_feat_raw,
            e_feat_logstd: shared_e_feat_logstd,
            gate_pip: None,
            gate_mask: Arc::new(Mutex::new(None)),
            gate_pi_logit: shared_gate_pi_logit,
            gate,
        })
    }

    /// β-sharing factored constructor: allocate a per-gene `β` Var (randn) plus a
    /// fresh cell side, and register the fixed `row_to_gene` index tensor. The
    /// `e_feat` field is seeded with the materialized `β` (gathered to feature
    /// rows) and refreshed after phase 1 via [`Self::materialize_e_feat`].
    pub fn new_factored(
        args: FactoredInit,
        varmap: &VarMap,
        vs: VarBuilder,
        dev: &Device,
    ) -> Result<Self> {
        let beta = register_randn_seeded(
            varmap,
            dev,
            "beta",
            args.n_genes,
            args.embedding_dim,
            args.seed,
        )?;
        let e_cell = register_randn_seeded(
            varmap,
            dev,
            "e_cell",
            args.n_cells,
            args.embedding_dim,
            args.seed,
        )?;
        let b_feat = register_var_from_slice(varmap, dev, "b_feat", args.b_feat)?;
        let b_cell = register_var_from_slice(varmap, dev, "b_cell", args.b_cell)?;

        // Optional per-gene splice offset δ_g, zero-initialized (so training
        // starts exactly at β-sharing and δ_g grows only where the data + L2
        // tradeoff justifies it).
        let delta = match args.unspliced_rows {
            Some(_) => Some(vs.get_with_hints(
                (args.n_genes, args.embedding_dim),
                "delta",
                candle_nn::Init::Const(0.0),
            )?),
            None => None,
        };
        let factor = build_feat_factor(&beta, args.row_to_gene, delta, args.unspliced_rows, dev)?;
        let e_feat = factor.e_feat()?.detach();
        Ok(Self {
            e_feat,
            e_cell,
            b_feat,
            b_cell,
            factor: Some(factor),
            embedding_dim: args.embedding_dim,
            s_feat: None,
            e_feat_raw: None,
            e_feat_logstd: None,
            gate_pip: None,
            gate_mask: Arc::new(Mutex::new(None)),
            gate_pi_logit: None,
            gate: None,
        })
    }

    /// Composite-training constructor for a factored model: share this model's
    /// `β` / `b_feat` + factor index tensor (so every level trains the SAME
    /// feature side) and allocate a fresh cell side under `var_prefix`. Delegates
    /// the cell-var allocation to [`Self::new_sharing_features`] and re-attaches
    /// the shared [`FeatFactor`].
    pub fn new_sharing_factor(
        &self,
        n_cells: usize,
        var_prefix: &str,
        varmap: &VarMap,
        dev: &Device,
        seed: u64,
    ) -> Result<Self> {
        let factor = self
            .factor
            .as_ref()
            .expect("new_sharing_factor requires a factored parent model");
        let mut model = Self::new_sharing_features(
            ShareFeaturesArgs {
                n_cells,
                embedding_dim: self.embedding_dim,
                shared_e_feat: self.e_feat.clone(),
                shared_b_feat: self.b_feat.clone(),
                e_cell_init: None,
                b_cell_init: &vec![0f32; n_cells],
                var_prefix,
                seed,
                // The factored gate rides on the cloned `factor` (`s_beta`); the
                // free-model gate fields stay empty. Copy the gate spec so the head
                // knows to apply it, and `π_h`, which lives on the model rather than
                // the factor and is what `gate_kl` needs.
                shared_s_feat: None,
                shared_e_feat_raw: None,
                shared_e_feat_logstd: None,
                shared_gate_pi_logit: self.gate_pi_logit.clone(),
                gate: self.gate,
            },
            varmap,
            dev,
        )?;
        model.factor = Some(factor.clone());
        Ok(model)
    }

    /// Snapshot the current `β` (gathered to feature rows) into the `e_feat`
    /// field (detached), so the phase-2 projection and all output/co-embed
    /// readers see a fixed dictionary. No-op for a free (non-factored) model.
    /// Call after phase 1.
    pub fn materialize_e_feat(&mut self) -> Result<()> {
        // Compute the frozen dictionary first (borrows self immutably), then assign.
        // Uses effect MEANS (no reparam sampling) and bakes the gate(s) in.
        let gated = if let Some(f) = &self.factor {
            // Factored: β̃ + mask·δ̃, each side gated separately (see `factored_feat_rows`).
            let mask = f.splice_delta.as_ref().map(|(_, m)| m.clone());
            Some(
                self.factored_feat_rows(f, &f.row_to_gene, mask.as_ref(), false)?
                    .detach(),
            )
        } else if let (Some(raw), Some(s_feat)) = (&self.e_feat_raw, &self.s_feat) {
            // Free + gated: bake the gate into `e_feat` from the raw Var (no smoother
            // at materialize — SGC smoothing is a training-time device; means, no sample).
            Some(
                self.gated_rows(
                    raw,
                    self.e_feat_logstd.as_ref(),
                    Some(&self.gate_weights(GateKind::Identity, s_feat)?),
                    false,
                )?
                .detach(),
            )
        } else {
            // Free + ungated: `e_feat` already IS the trained Var — leave it (no-op,
            // byte-identical to the pre-gate behaviour).
            None
        };
        if let Some(g) = gated {
            self.e_feat = g;
        }
        Ok(())
    }

    /// Enable the variational spike-and-slab gate on the feature side (see
    /// [`FeatureGateSpec`]). Allocates, as Vars in `varmap`: the selection logits
    /// (`s_feat [n_features, H]` free / `s_beta [G, H]` factored), the effect
    /// log-std (`e_feat_logstd` / `beta_logstd [·, H]`), and the per-dim inclusion
    /// rate (`gate_pi_logit [H]`); for a factored model WITH velocity
    /// (`splice_delta`), also the INDEPENDENT δ gate (`s_delta`, `delta_logstd`,
    /// `delta_pi_logit`). Logits start at [`GATE_LOGIT_INIT`], so `σ ≈ 0.98` and an
    /// untrained gate is inert rather than biased — see that constant for why zero
    /// would be wrong here even though it was right for the softmax. Call ONCE on the
    /// primary model, BEFORE building sharing heads (which carry the shared gate via
    /// [`ShareFeaturesArgs`] / the cloned [`FeatFactor`]).
    pub fn enable_feature_gate(
        &mut self,
        spec: FeatureGateSpec,
        varmap: &VarMap,
        dev: &Device,
    ) -> Result<()> {
        let h = self.embedding_dim;
        let register = |name: &str, t: Tensor| -> Result<Tensor> {
            let var = candle_util::candle_core::Var::from_tensor(&t)?;
            varmap
                .data()
                .lock()
                .unwrap()
                .insert(name.to_string(), var.clone());
            Ok(var.as_tensor().clone())
        };
        // Selection logits `[rows, H]` at `GATE_LOGIT_INIT` — `σ ≈ 0.98`, so the gate
        // starts as ~the identity and the data turns coordinates OFF rather than on.
        // The reverse (init at 0 ⇒ `σ = 0.5`) would halve the dictionary AND leave
        // `β`'s gradient scaled by the gate it is trying to earn; see the constant.
        let init_gate = |name: &str, rows: usize| -> Result<Tensor> {
            register(
                name,
                Tensor::from_vec(vec![GATE_LOGIT_INIT; rows * h], (rows, h), dev)?,
            )
        };
        // Per-dim inclusion rate `[H]`, init at the Beta prior's mean so an untrained
        // `π_h` states the prior rather than an arbitrary one.
        let init_pi = |name: &str| -> Result<Tensor> {
            let mean = GATE_PI_BETA_A / (GATE_PI_BETA_A + GATE_PI_BETA_B);
            let logit = (mean / (1.0 - mean)).ln() as f32;
            register(name, Tensor::from_vec(vec![logit; h], (h,), dev)?)
        };
        // Effect log-std `[rows, H]`, init `GATE_LOGSTD_INIT` (near-deterministic start).
        let init_logstd = |name: &str, rows: usize| -> Result<Tensor> {
            register(
                name,
                Tensor::from_vec(vec![GATE_LOGSTD_INIT; rows * h], (rows, h), dev)?,
            )
        };
        self.gate_pi_logit = Some(init_pi("gate_pi_logit")?);
        match &mut self.factor {
            Some(f) => {
                let n_genes = f.beta.dim(0)?;
                f.s_beta = Some(init_gate("s_beta", n_genes)?);
                f.beta_logstd = Some(init_logstd("beta_logstd", n_genes)?);
                // Independent velocity gate on δ_g, only when velocity is present.
                if f.splice_delta.is_some() {
                    f.s_delta = Some(init_gate("s_delta", n_genes)?);
                    f.delta_logstd = Some(init_logstd("delta_logstd", n_genes)?);
                    f.delta_pi_logit = Some(init_pi("delta_pi_logit")?);
                }
            }
            None => {
                let n_features = self.e_feat.dim(0)?;
                // Keep the raw Var reachable so the gather reads it while
                // `materialize_e_feat` overwrites `e_feat` with the gated snapshot.
                self.e_feat_raw = Some(self.e_feat.clone());
                self.s_feat = Some(init_gate("s_feat", n_features)?);
                self.e_feat_logstd = Some(init_logstd("e_feat_logstd", n_features)?);
            }
        }
        self.gate = Some(spec);
        Ok(())
    }

    /// The identity-gate effect log-std table (`e_feat_logstd` free / `beta_logstd`
    /// factored), or `None` if ungated.
    fn effect_logstd(&self) -> Option<&Tensor> {
        match &self.factor {
            Some(f) => f.beta_logstd.as_ref(),
            None => self.e_feat_logstd.as_ref(),
        }
    }

    /// Scale gate logits by the softmax temperature `τ` (`1/τ` on the logits); an
    /// Arc-cheap clone at `τ = 1`. Shared by the gather and the KL.
    fn apply_temperature(&self, logits: &Tensor) -> Result<Tensor> {
        let tau = self
            .gate
            .expect("apply_temperature called on an ungated model")
            .temperature;
        if (tau - 1.0).abs() > f32::EPSILON {
            logits.affine(1.0 / tau as f64, 0.0)
        } else {
            Ok(logits.clone())
        }
    }

    /// The gate's multiplier table `[rows, H]` — `α = σ(S/τ)`, the variational
    /// inclusion probability `q(z=1)` for each `(gene, dim)`.
    ///
    /// **Elementwise, and that is load-bearing.** The predecessor was a softmax down
    /// the GENE axis, so handing it a gathered minibatch silently renormalized over
    /// whatever subset arrived — including rows duplicated by negative sampling, or a
    /// factored model's per-splice-track repeats. It compiled, ran, and trained
    /// garbage, which is why [`Self::gathered_gate_weights`] exists to pin the order.
    /// A sigmoid couples nothing, so that whole hazard is gone: gather-then-gate and
    /// gate-then-gather now agree, and `gate_order_is_invariant_under_a_sigmoid`
    /// asserts it.
    ///
    /// No `D` rescale either. It existed to stop a `1/D`-scale softmax collapsing the
    /// score by three orders of magnitude at `D = 20k`; `σ` is already `O(1)`.
    pub(crate) fn gate_weights(&self, kind: GateKind, full_logits: &Tensor) -> Result<Tensor> {
        let (pip, mask) = self.gate_tables(kind);
        if pip.is_some() {
            if let Some(cell) = mask {
                if let Some(m) = cell.lock().expect("gate mask poisoned").as_ref() {
                    return Ok(m.clone());
                }
            }
        }
        if let Some(p) = pip {
            return Ok(p.clone());
        }
        candle_nn::ops::sigmoid(&self.apply_temperature(full_logits)?)
    }

    /// Install a frozen posterior inclusion table and switch the model to jitter mode.
    /// `pip` is `[rows, H]` row-major, on the gate's own axis (feature rows for a free
    /// model, genes for a factored one). Call [`Self::resample_gate_mask`] once per
    /// epoch afterwards; until then the model uses the mean.
    pub fn set_gate_pip(
        &mut self,
        kind: GateKind,
        pip: &[f32],
        rows: usize,
        dev: &Device,
    ) -> Result<()> {
        let h = self.embedding_dim;
        assert_eq!(pip.len(), rows * h, "gate pip must be [rows, H]");
        let t = Tensor::from_slice(pip, (rows, h), &Device::Cpu)?.to_device(dev)?;
        match kind {
            GateKind::Identity => {
                self.gate_pip = Some(t);
                *self.gate_mask.lock().expect("gate mask poisoned") = None;
            }
            GateKind::Velocity => {
                if let Some(f) = self.factor.as_mut() {
                    f.delta_gate_pip = Some(t);
                    f.delta_gate_mask = Some(Arc::new(Mutex::new(None)));
                }
            }
        }
        Ok(())
    }

    /// Point this model's mask at an existing shared cell, so every composite axis
    /// redraws together. Call after [`Self::set_gate_pip`] on each axis.
    pub fn share_gate_mask(&mut self, cell: &Arc<Mutex<Option<Tensor>>>) {
        self.gate_mask = Arc::clone(cell);
    }

    /// The shared mask cell, for handing to the other axes.
    #[must_use]
    pub fn gate_mask_cell(&self) -> Arc<Mutex<Option<Tensor>>> {
        Arc::clone(&self.gate_mask)
    }

    /// Draw this epoch's `z ~ Bern(gate_pip)` and RETURN it. `None` when jitter is off.
    ///
    /// Returns rather than stores because the composite fit runs several axes that
    /// share Vars but are separate `JointEmbedModel` values — `gate_mask` is a plain
    /// field, not a Var, so it does not propagate. Every axis must be handed the SAME
    /// draw via [`Self::set_gate_mask`], or the axes would train against different
    /// sub-models within one epoch and the mask would stop meaning "this epoch's `z`".
    ///
    /// Uses the device RNG, so a jittered fit is not bit-reproducible across runs —
    /// already true of the gate's reparameterization noise, so this adds no new class
    /// of irreproducibility.
    pub fn resample_gate_mask(&self) -> Result<()> {
        let Some(pip) = &self.gate_pip else {
            return Ok(());
        };
        let u = Tensor::rand(0f32, 1f32, pip.shape(), pip.device())?;
        // `z = 1` with probability `pip`.
        let z = u.lt(pip)?.to_dtype(DType::F32)?;
        *self.gate_mask.lock().expect("gate mask poisoned") = Some(z);
        // The velocity gate draws its OWN `z` from its OWN `pip` — a shared draw would
        // tie a gene's motion to its identity selection, which is precisely the
        // conflation this whole `GateKind` split exists to prevent.
        if let Some(f) = &self.factor {
            if let (Some(dp), Some(cell)) = (&f.delta_gate_pip, &f.delta_gate_mask) {
                let du = Tensor::rand(0f32, 1f32, dp.shape(), dp.device())?;
                *cell.lock().expect("delta gate mask poisoned") =
                    Some(du.lt(dp)?.to_dtype(DType::F32)?);
            }
        }
        Ok(())
    }

    /// Drop the epoch mask so the model reverts to the mean `E[z] = pip` — what output
    /// and `materialize_e_feat` must use, since training averaged over draws.
    pub fn clear_gate_mask(&self) {
        *self.gate_mask.lock().expect("gate mask poisoned") = None;
        if let Some(f) = &self.factor {
            if let Some(cell) = &f.delta_gate_mask {
                *cell.lock().expect("delta gate mask poisoned") = None;
            }
        }
    }

    /// Is this model in jitter (posterior-informed dropout) mode?
    #[must_use]
    pub fn is_jittered(&self) -> bool {
        self.gate_pip.is_some()
    }

    /// Gate the full logit table, then gather the requested rows.
    ///
    /// Order no longer matters — [`Self::gate_weights`] is elementwise — so this is
    /// now a convenience, not a guard. Kept because every call site already uses it
    /// and the shape is right; do not read it as evidence that the ordering is still
    /// dangerous.
    pub(crate) fn gathered_gate_weights(
        &self,
        kind: GateKind,
        logits: Option<&Tensor>,
        rows: &Tensor,
    ) -> Result<Option<Tensor>> {
        // GATHER FIRST. The predecessor was a softmax down the gene axis and had to see
        // the whole table; `σ` is elementwise (`gate_order_is_invariant_under_a_sigmoid`
        // pins it), so selecting rows before transforming is exact and avoids
        // materialising `[G, H]` to keep `[batch, H]` — 33x waste per positive gather at
        // G = 34k, on every step, per axis, and paid again on backward.
        let (pip, mask) = self.gate_tables(kind);
        if pip.is_some() {
            if let Some(cell) = mask {
                if let Some(m) = cell.lock().expect("gate mask poisoned").as_ref() {
                    return Ok(Some(m.index_select(rows, 0)?));
                }
            }
            return pip.map(|p| p.index_select(rows, 0)).transpose();
        }
        logits
            .map(|x| {
                let g = x.index_select(rows, 0)?;
                candle_nn::ops::sigmoid(&self.apply_temperature(&g)?)
            })
            .transpose()
    }

    /// The `(pip, mask cell)` pair for one gate — the branch `gate_weights` and
    /// `gathered_gate_weights` share.
    fn gate_tables(&self, kind: GateKind) -> GateTables<'_> {
        match kind {
            GateKind::Identity => (self.gate_pip.as_ref(), Some(&self.gate_mask)),
            GateKind::Velocity => match &self.factor {
                Some(f) => (f.delta_gate_pip.as_ref(), f.delta_gate_mask.as_ref()),
                None => (None, None),
            },
        }
    }

    /// One gated single-effect: reparam-sample the Gaussian effect (`μ + σ·ε`) when
    /// `sample` and a log-std is present (else use the mean `μ`), then apply the softmax
    /// gate when selection logits are present. `mu` / `logstd` are `[N, H]`, `s` is
    /// `[N, H+1]`. The shared per-row primitive for both the `β` and `δ` sides (and the
    /// free `e_feat`). With no logits/logstd it is the plain ungated gather.
    pub(crate) fn gated_rows(
        &self,
        mu: &Tensor,
        logstd: Option<&Tensor>,
        w: Option<&Tensor>,
        sample: bool,
    ) -> Result<Tensor> {
        let eff = match (sample, logstd) {
            (true, Some(ls)) => self.sample_effect(mu, ls)?,
            _ => mu.clone(),
        };
        match w {
            Some(w) => eff.mul(w),
            None => Ok(eff),
        }
    }

    /// Compose the effective factored feature rows: `β̃ + mask·δ̃`, where each side is
    /// its own gated effect ([`Self::gated_rows`]) — the IDENTITY gate on `β_g`
    /// and the INDEPENDENT velocity gate on `δ_g`. `genes` gathers the per-gene tables
    /// (`row_to_gene[idx]` in training, the full `row_to_gene` at materialize);
    /// `mask_rows` is the `[N,1]` unspliced selector (already gathered), `None` when
    /// there is no velocity. `sample` reparam-samples (training) vs uses means (output).
    pub(crate) fn factored_feat_rows(
        &self,
        f: &FeatFactor,
        genes: &Tensor,
        mask_rows: Option<&Tensor>,
        sample: bool,
    ) -> Result<Tensor> {
        let gather = |t: &Option<Tensor>| -> Result<Option<Tensor>> {
            t.as_ref().map(|x| x.index_select(genes, 0)).transpose()
        };
        let (beta_ls, beta_s) = (
            gather(&f.beta_logstd)?,
            self.gathered_gate_weights(GateKind::Identity, f.s_beta.as_ref(), genes)?,
        );
        let beta = self.gated_rows(
            &f.beta.index_select(genes, 0)?,
            beta_ls.as_ref(),
            beta_s.as_ref(),
            sample,
        )?;
        match (&f.splice_delta, mask_rows) {
            (Some((delta, _)), Some(m)) => {
                let (delta_ls, delta_s) = (
                    gather(&f.delta_logstd)?,
                    self.gathered_gate_weights(GateKind::Velocity, f.s_delta.as_ref(), genes)?,
                );
                let delta = self.gated_rows(
                    &delta.index_select(genes, 0)?,
                    delta_ls.as_ref(),
                    delta_s.as_ref(),
                    sample,
                )?;
                beta.add(&delta.broadcast_mul(m)?) // + mask ⊙ δ̃ on unspliced rows
            }
            _ => Ok(beta),
        }
    }

    /// softmax selection `[n_features, H]` for a per-gene/per-row logit table, gathering
    /// a factored per-gene table to feature rows via `row_to_gene`. `None` if `logits`
    /// is `None`. The excluded null mass per row is `1 − rowsum` (a near-zero row = a
    /// deselected gene).
    fn selection_from(&self, kind: GateKind, logits: Option<&Tensor>) -> Result<Option<Tensor>> {
        let Some(logits) = logits else {
            return Ok(None);
        };
        // Normalize on the FULL table first, then expand per-gene → per-row. Doing
        // it the other way round would normalize over duplicated rows on the `Gene`
        // axis (a factored model repeats each gene once per splice track), halving
        // every value while leaving the β-sharing equality test passing.
        let w = self.gate_weights(kind, logits)?;
        let rows = match &self.factor {
            Some(f) => w.index_select(&f.row_to_gene, 0)?,
            None => w,
        };
        Ok(Some(rows.detach()))
    }

    /// Per-feature-row IDENTITY selection `softmax(s_beta/s_feat)` `[n_features, H]`, for
    /// interpretability; `None` for an ungated model. Rows align with `e_feat` / the
    /// dictionary output.
    pub fn feature_selection(&self) -> Result<Option<Tensor>> {
        self.selection_from(GateKind::Identity, self.gate_logits())
    }

    /// Per-feature-row VELOCITY selection `softmax(s_delta)` `[n_features, H]` — the
    /// per-gene motion gate (driver genes); `None` unless the factored δ gate is on.
    pub fn velocity_selection(&self) -> Result<Option<Tensor>> {
        self.selection_from(
            GateKind::Velocity,
            self.factor.as_ref().and_then(|f| f.s_delta.as_ref()),
        )
    }

    /// The identity-gate logit table (`s_feat` free / `s_beta` factored), or `None` if
    /// ungated.
    fn gate_logits(&self) -> Option<&Tensor> {
        match &self.factor {
            Some(f) => f.s_beta.as_ref(),
            None => self.s_feat.as_ref(),
        }
    }

    /// Reparameterize the Gaussian effect for the variational gate: `μ + σ·ε`, with
    /// `σ = exp(logstd_rows)` and a fresh `ε ~ N(0,1)` each call. `mu_rows` /
    /// `logstd_rows` are `[N, H]`. Used in the TRAINING gather so the posterior
    /// variance feeds the likelihood; output/materialize use the mean (`σ=0`).
    pub fn sample_effect(&self, mu_rows: &Tensor, logstd_rows: &Tensor) -> Result<Tensor> {
        let eps = Tensor::randn(0f32, 1f32, mu_rows.shape(), mu_rows.device())?;
        let sigma = logstd_rows
            .clamp(-GATE_LOGSTD_CLAMP, GATE_LOGSTD_CLAMP)?
            .exp()?;
        mu_rows.add(&sigma.mul(&eps)?)
    }

    /// The inclusion-weighted Gaussian effect KL, per entry
    /// `½[(σ²+μ²)/σ₀² − 1 − 2·logstd + ln σ₀²]` against `σ₀² =`
    /// [`GATE_EFFECT_PRIOR_VAR`], weighted by `w` and meaned over every entry.
    ///
    /// `w` must be the weight the LIKELIHOOD actually applies to this effect, so that a
    /// coordinate the gate has turned off is not also asked to pay a prior for a
    /// loading nothing reads. Under a learned gate that is `α = σ(S/τ)`; under jitter it
    /// is the frozen `pip`, since training averages over `z ~ Bern(pip)` and `E[z] = pip`.
    fn effect_kl(w: &Tensor, logstd: &Tensor, mu: &Tensor) -> Result<Tensor> {
        let s0 = GATE_EFFECT_PRIOR_VAR;
        let logstd = logstd.clamp(-GATE_LOGSTD_CLAMP, GATE_LOGSTD_CLAMP)?; // finite σ, log σ²
        let two_logstd = logstd.affine(2.0, 0.0)?; // 2·logstd = log σ²
        let a = (two_logstd.exp()? + mu.sqr()?)?.affine(1.0 / s0, 0.0)?; // (σ²+μ²)/σ₀²
        let per_entry = (a - &two_logstd)?.affine(0.5, 0.5 * (s0.ln() - 1.0))?; // ½[…]
        w.mul(&per_entry)?.mean_all()
    }

    /// One gate's spike-and-slab KL: the Bernoulli inclusion KL against the dim's
    /// learned rate `π_h`, plus the inclusion-weighted Gaussian effect KL
    /// (`σ₀² =` [`GATE_EFFECT_PRIOR_VAR`]), plus `π_h`'s own Beta hyperprior. Shared
    /// by the identity (`β`/`e_feat`) and velocity (`δ`) gates. In the autograd graph.
    ///
    /// # The three terms must stay commensurate
    ///
    /// They are summed under one [`GATE_KL_WEIGHT`], so if their scales diverge the
    /// largest silently sets the loss and the others stop existing. That is not
    /// hypothetical — it is what happened when the gate's softmax moved from the dim
    /// axis to the gene axis: `α` went from `≈1/(H+1)` to `≈1/D`, the forward pass
    /// compensated via a `D` rescale and this function did not, and at
    /// `D = 20287, H = 16` the effect prior came out **~49,000×** smaller than the
    /// concentration term — numerically switched off.
    ///
    /// The Bernoulli form is better behaved: `α = σ(·)` is `O(1)` per entry with no
    /// `D` in it, so the first two terms are commensurate by construction once both
    /// are meaned over entries — no `ln D` normalization needed. The surviving rule
    /// is the one that fixed it last time, and it is unchanged in spirit:
    ///
    /// * the Gaussian KL is weighted by the weight the LIKELIHOOD actually applies —
    ///   now plain `α`, since that is what [`Self::gate_weights`] returns — and meaned
    ///   over every entry;
    /// * the Beta hyperprior is `O(H)` against the others' `O(D·H)`, so it is divided
    ///   by `D·H` to preserve the ratio the true (summed) ELBO would have. With
    ///   `D ≈ 20k` that makes it a genuinely weak prior which the data dominates —
    ///   the same regime `posterior::hyper::sample_pi0`'s `Beta(9,1)` sits in.
    ///
    /// `gate_terms_are_commensurate` pins the ratio so a future reduction change
    /// cannot re-open this quietly.
    fn single_gate_kl(
        &self,
        logits: &Tensor,
        pi_logit: &Tensor,
        logstd: &Tensor,
        mu: &Tensor,
    ) -> Result<Tensor> {
        let logits = self.apply_temperature(logits)?;
        // `α = q(z=1)` per (gene, dim) — the same table `gate_weights` feeds the
        // likelihood. Clamped off 0/1 so the logs below stay finite.
        let alpha = candle_nn::ops::sigmoid(&logits)?.clamp(GATE_PI_EPS, 1.0 - GATE_PI_EPS)?;
        let n_genes = alpha.dim(0)? as f64;
        let n_dims = alpha.dim(1)? as f64;

        // Per-dim inclusion rate `π_h`, broadcast over genes.
        let pi = candle_nn::ops::sigmoid(pi_logit)?.clamp(GATE_PI_EPS, 1.0 - GATE_PI_EPS)?;
        let pi_row = pi.reshape((1, alpha.dim(1)?))?;

        // KL(Bern(α) ‖ Bern(π)) = α·[ln α − ln π] + (1−α)·[ln(1−α) − ln(1−π)]
        let one_m_alpha = alpha.affine(-1.0, 1.0)?;
        let one_m_pi = pi_row.affine(-1.0, 1.0)?;
        let kl_bern = (alpha.mul(&alpha.log()?.broadcast_sub(&pi_row.log()?)?)?
            + one_m_alpha.mul(&one_m_alpha.log()?.broadcast_sub(&one_m_pi.log()?)?)?)?
        .mean_all()?;

        let kl_gauss = Self::effect_kl(&alpha, logstd, mu)?;

        // −log Beta(π_h; a, b) up to a constant = −[(a−1)·ln π + (b−1)·ln(1−π)],
        // summed over dims and put on the same per-entry footing as the two above.
        let beta_nlp = (pi.log()?.affine(-(GATE_PI_BETA_A - 1.0), 0.0)?.sub(
            &pi.affine(-1.0, 1.0)?
                .log()?
                .affine(GATE_PI_BETA_B - 1.0, 0.0)?,
        )?)
        .sum_all()?
        .affine(1.0 / (n_genes * n_dims), 0.0)?;

        kl_bern + kl_gauss + beta_nlp
    }

    /// The gate's total KL: the identity gate's plus, for a factored model with
    /// velocity, the INDEPENDENT δ gate's. `None` for an ungated model. Kept in the
    /// autograd graph.
    ///
    /// Each gate carries its OWN `π_h` table. They are different objects — an identity
    /// loading and a deviation from it are not on one scale — so a shared inclusion
    /// rate would force them to agree, which is the same reason
    /// `posterior::pb_gibbs` gives every gate its own `σ₀h²` and `π₀h`.
    ///
    /// # Two regimes
    ///
    /// **Learned gate** — the full spike-and-slab KL ([`Self::single_gate_kl`]).
    ///
    /// **Jitter** (a `pip` is installed) — the SELECTION is an input, not a parameter:
    /// `gate_weights` never consults the logits, so the Bernoulli and Beta terms would
    /// only move Vars nothing reads, and the inclusion prior already lives in the
    /// sampler that produced the `pip`. The Gaussian effect KL is a different matter —
    /// `β` is very much still a trained parameter under the mask — so that term stays,
    /// weighted by the `pip` (what the likelihood applies in expectation). Dropping it
    /// too, as an earlier version did, silently removed the ONLY shrinkage on the
    /// loading for the whole jitter fit.
    pub fn gate_kl(&self) -> Result<Option<Tensor>> {
        if self.gate.is_none() {
            return Ok(None);
        }
        let mu = match &self.factor {
            Some(f) => &f.beta,
            None => self.e_feat_raw.as_ref().unwrap_or(&self.e_feat),
        };
        let mut kl = self.one_gate_kl(GateKind::Identity, self.gate_logits(), mu)?;
        // Independent velocity gate on δ_g (factored + velocity present).
        if let Some(f) = &self.factor {
            if let Some((delta, _)) = &f.splice_delta {
                let dkl = self.one_gate_kl(GateKind::Velocity, f.s_delta.as_ref(), delta)?;
                kl = match (kl, dkl) {
                    (Some(a), Some(b)) => Some((a + b)?),
                    (a, b) => a.or(b),
                };
            }
        }
        Ok(kl)
    }

    /// One gate's contribution to [`Self::gate_kl`], picking the regime off whether a
    /// `pip` is installed for that [`GateKind`]. `None` when the gate has no effect
    /// log-std (a deterministic effect has no Gaussian KL to pay) or, on the learned
    /// path, no logits / no `π_h`.
    fn one_gate_kl(
        &self,
        kind: GateKind,
        logits: Option<&Tensor>,
        mu: &Tensor,
    ) -> Result<Option<Tensor>> {
        let (logstd, pi_logit) = match kind {
            GateKind::Identity => (self.effect_logstd(), self.gate_pi_logit.as_ref()),
            GateKind::Velocity => match &self.factor {
                Some(f) => (f.delta_logstd.as_ref(), f.delta_pi_logit.as_ref()),
                None => (None, None),
            },
        };
        let Some(logstd) = logstd else {
            return Ok(None);
        };
        let (pip, _) = self.gate_tables(kind);
        if let Some(pip) = pip {
            return Ok(Some(Self::effect_kl(pip, logstd, mu)?));
        }
        let (Some(logits), Some(pi)) = (logits, pi_logit) else {
            return Ok(None);
        };
        Ok(Some(self.single_gate_kl(logits, pi, logstd, mu)?))
    }

    /// Mean-pool the cell embedding table over the fine children of a
    /// list of coarse-block indices. Output `[n_blocks, H]` plus a
    /// matching `[n_blocks]` bias vector.
    pub fn pool_cells(
        &self,
        coarse_blocks: &[u32],
        coarse_to_fine: &[Vec<usize>],
        dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        pool_axis(
            &self.e_cell,
            &self.b_cell,
            coarse_blocks,
            coarse_to_fine,
            dev,
        )
    }

    /// Bilinear score with bias terms.
    ///
    /// `e_f`: `[B, H]` pooled feature embeddings (one row per positive's
    /// feature block).
    /// `e_c`: `[B, H]` pooled cell embeddings (one row per positive's
    /// cell block).
    /// `b_f`, `b_c`: `[B]` bias scalars per row.
    /// Returns `[B]` scores.
    pub fn score_diag(e_f: &Tensor, e_c: &Tensor, b_f: &Tensor, b_c: &Tensor) -> Result<Tensor> {
        let dot = (e_f * e_c)?.sum(1)?;
        (dot + b_f)? + b_c
    }

    /// Bilinear score for negatives: score positive cells against
    /// alternative feature blocks. `e_f_neg`: `[B, K, H]`,
    /// `e_c`: `[B, H]`, `b_f_neg`: `[B, K]`, `b_c`: `[B]`. Returns `[B, K]`.
    pub fn score_negatives(
        e_f_neg: &Tensor,
        e_c: &Tensor,
        b_f_neg: &Tensor,
        b_c: &Tensor,
    ) -> Result<Tensor> {
        let b = e_f_neg.dim(0)?;
        let k = e_f_neg.dim(1)?;
        let h = e_f_neg.dim(2)?;
        let e_c_expanded = e_c.unsqueeze(1)?.broadcast_as((b, k, h))?;
        let dot = (e_f_neg * e_c_expanded)?.sum(2)?;
        let b_c_b = b_c.unsqueeze(1)?.broadcast_as((b, k))?;
        (dot + b_f_neg)? + b_c_b
    }

    /// Gene-modulated diagonal score for cell-cell positive pairs. The
    /// gene defines a direction in the shared cell-embedding space; the
    /// score is the product of u's and v's projections along that
    /// direction, plus the cell biases:
    ///
    /// `score(u, v, g) = (e_gene · e_cell_u)(e_gene · e_cell_v)
    ///                 + b_cell_u + b_cell_v`
    ///
    /// `e_gene`, `e_cell_l`, `e_cell_r` all share shape `[B, H]` (each row
    /// is a `(gene, positive)` lookup pre-gathered by the caller).
    /// `b_cell_l`, `b_cell_r` are `[B]`. Returns `[B]`.
    pub fn score_cellcell_gated(
        e_gene: &Tensor,
        e_cell_l: &Tensor,
        e_cell_r: &Tensor,
        b_cell_l: &Tensor,
        b_cell_r: &Tensor,
    ) -> Result<Tensor> {
        let proj_l = (e_gene * e_cell_l)?.sum(1)?; // [B]
        let proj_r = (e_gene * e_cell_r)?.sum(1)?; // [B]
        let pair = (proj_l * proj_r)?;
        (pair + b_cell_l)? + b_cell_r
    }

    /// Gene-modulated score for chain negatives. `e_gene`, `e_cell_anchor`
    /// are `[B, H]` (one row per `(gene, positive)`); `e_cell_neg` is
    /// `[B, K, H]` (K sibling negatives per positive); `b_cell_anchor`
    /// is `[B]`; `b_cell_neg` is `[B, K]`. Returns `[B, K]`.
    ///
    /// Per-row score: `(e_gene · e_cell_anchor) · (e_gene · e_cell_neg[k])
    ///               + b_cell_anchor + b_cell_neg[k]`.
    /// The gene-direction projection of the anchor is computed once and
    /// broadcast across K negatives.
    pub fn score_cellcell_gated_neg(
        e_gene: &Tensor,
        e_cell_anchor: &Tensor,
        e_cell_neg: &Tensor,
        b_cell_anchor: &Tensor,
        b_cell_neg: &Tensor,
    ) -> Result<Tensor> {
        let b = e_cell_neg.dim(0)?;
        let k = e_cell_neg.dim(1)?;
        let h = e_cell_neg.dim(2)?;
        let proj_anchor = (e_gene * e_cell_anchor)?.sum(1)?; // [B]
        let e_gene_3d = e_gene.unsqueeze(1)?.broadcast_as((b, k, h))?;
        let proj_neg = (e_cell_neg * e_gene_3d)?.sum(2)?; // [B, K]
        let proj_anchor_2d = proj_anchor.unsqueeze(1)?.broadcast_as((b, k))?;
        let pair = (proj_anchor_2d * proj_neg)?; // [B, K]
        let b_anchor_2d = b_cell_anchor.unsqueeze(1)?.broadcast_as((b, k))?;
        (pair + b_anchor_2d)? + b_cell_neg
    }
}

/// Build a [`FeatFactor`] from the shared `β` Var and the host-side row→gene
/// map: materializes the fixed `row_to_gene` (u32) index tensor on `dev`.
fn build_feat_factor(
    beta: &Tensor,
    row_to_gene: &[u32],
    delta: Option<Tensor>,
    unspliced_rows: Option<&[bool]>,
    dev: &Device,
) -> Result<FeatFactor> {
    let d = row_to_gene.len();
    let row_to_gene_t = Tensor::from_vec(row_to_gene.to_vec(), d, dev)?;
    // `δ_g` and its `[n_features, 1]` 0/1 unspliced-row mask co-exist: both are built
    // exactly when a `δ_g` Var was allocated (`delta_l2 > 0`).
    let splice_delta = match (delta, unspliced_rows) {
        (Some(delta), Some(u)) => {
            let m: Vec<f32> = u.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            Some((delta, Tensor::from_vec(m, (d, 1), dev)?))
        }
        _ => None,
    };
    Ok(FeatFactor {
        beta: beta.clone(),
        row_to_gene: row_to_gene_t,
        splice_delta,
        s_beta: None,
        beta_logstd: None,
        s_delta: None,
        delta_logstd: None,
        delta_gate_pip: None,
        delta_gate_mask: None,
        delta_pi_logit: None,
    })
}

/// Register a `[rows, cols]` learnable parameter initialized with **seeded,
/// reproducible** `N(0, INIT_STDEV)` values, and return the underlying tensor.
///
/// Replaces `vs.get_with_hints(..., Init::Randn)` / `Tensor::randn`: candle's
/// device randn is unseedable on the CPU backend (`Device::set_seed` errors
/// out, `rand_normal` reads OS entropy), so identical-config runs would
/// otherwise draw a fresh init every time. The seeded `Tensor` sampler draws
/// it host-side instead, keyed by `name` so each table (`e_feat`, `e_cell`,
/// `beta`, per-level `{prefix}_e_cell`) gets an independent stream off one
/// `base_seed` with no hand-assigned salts.
fn register_randn_seeded(
    varmap: &VarMap,
    dev: &Device,
    name: &str,
    rows: usize,
    cols: usize,
    base_seed: u64,
) -> Result<Tensor> {
    // `rnorm_seeded` (a host `from_vec`) and `affine` are both contiguous, and
    // `to_device` preserves that; the explicit `contiguous()` is a cheap no-op
    // guard so the registered Var is always contiguous for CUDA matmul kernels.
    let t = Tensor::rnorm_seeded(rows, cols, name_seed(base_seed, name))
        .affine(INIT_STDEV as f64, 0.0)?
        .to_device(dev)?
        .contiguous()?;
    let var = candle_util::candle_core::Var::from_tensor(&t)?;
    varmap
        .data()
        .lock()
        .unwrap()
        .insert(name.to_string(), var.clone());
    Ok(var.as_tensor().clone())
}

/// Register a 1D learnable parameter initialized from a slice and
/// return the underlying tensor (kept in autograd via `VarMap`).
fn register_var_from_slice(
    varmap: &VarMap,
    dev: &Device,
    name: &str,
    values: &[f32],
) -> Result<Tensor> {
    let var = candle_util::candle_core::Var::from_slice(values, values.len(), dev)?;
    {
        let mut data = varmap.data().lock().unwrap();
        data.insert(name.to_string(), var.clone());
    }
    Ok(var.as_tensor().clone())
}

/// Register a 2D learnable parameter initialized from a host matrix
/// (row-major flatten). `nalgebra::DMatrix` is column-major, so we
/// emit row-by-row; the resulting tensor matches candle's `[rows, cols]`
/// row-major layout.
fn register_var_from_mat(
    varmap: &VarMap,
    dev: &Device,
    name: &str,
    mat: &nalgebra::DMatrix<f32>,
) -> Result<Tensor> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    let mut row_major = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            row_major.push(mat[(i, j)]);
        }
    }
    let var = candle_util::candle_core::Var::from_tensor(&Tensor::from_vec(
        row_major,
        (rows, cols),
        dev,
    )?)?;
    {
        let mut data = varmap.data().lock().unwrap();
        data.insert(name.to_string(), var.clone());
    }
    Ok(var.as_tensor().clone())
}

/// Mean-pool `[D, H]` table over the fine children of `coarse_blocks`.
/// Returns `(pooled_emb [n_blocks, H], pooled_bias [n_blocks])`. Both
/// outputs stay in the autograd graph.
///
/// Two ops total in the autograd path: one flat `index_select` gathers
/// every fine row in block order, then `index_add` scatters them into
/// per-block sums in a `[n_blocks, H]` accumulator. Empty blocks get
/// `count = 1` so the all-zero accumulator divides cleanly to zero
/// (matching the loop's zero-pad behavior).
fn pool_axis(
    table: &Tensor,
    bias: &Tensor,
    coarse_blocks: &[u32],
    coarse_to_fine: &[Vec<usize>],
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let h = table.dim(1)?;
    let n_blocks = coarse_blocks.len();

    let total_fine: usize = coarse_blocks
        .iter()
        .map(|&b| coarse_to_fine[b as usize].len())
        .sum();
    let mut flat_fine: Vec<u32> = Vec::with_capacity(total_fine);
    let mut owner: Vec<u32> = Vec::with_capacity(total_fine);
    let mut counts: Vec<f32> = Vec::with_capacity(n_blocks);
    for (b_idx, &block) in coarse_blocks.iter().enumerate() {
        let fine = &coarse_to_fine[block as usize];
        for &f in fine {
            flat_fine.push(f as u32);
            owner.push(b_idx as u32);
        }
        counts.push(fine.len().max(1) as f32);
    }

    if total_fine == 0 {
        // No fine rows at all — every block is empty. Return zeros directly.
        let emb_zeros = Tensor::zeros((n_blocks, h), table.dtype(), dev)?;
        let bias_zeros = Tensor::zeros(n_blocks, bias.dtype(), dev)?;
        return Ok((emb_zeros, bias_zeros));
    }

    let flat_fine_t = Tensor::from_vec(flat_fine, total_fine, dev)?;
    let owner_t = Tensor::from_vec(owner, total_fine, dev)?;
    let counts_2d = Tensor::from_vec(counts.clone(), (n_blocks, 1), dev)?;
    let counts_1d = Tensor::from_vec(counts, n_blocks, dev)?;

    let gathered_emb = table.index_select(&flat_fine_t, 0)?; // [n_fine, H]
    let zeros_emb = Tensor::zeros((n_blocks, h), table.dtype(), dev)?;
    let summed_emb = zeros_emb.index_add(&owner_t, &gathered_emb, 0)?; // [n_blocks, H]
    let pooled_emb = summed_emb.broadcast_div(&counts_2d)?;

    let gathered_bias = bias.index_select(&flat_fine_t, 0)?; // [n_fine]
    let zeros_bias = Tensor::zeros(n_blocks, bias.dtype(), dev)?;
    let summed_bias = zeros_bias.index_add(&owner_t, &gathered_bias, 0)?; // [n_blocks]
    let pooled_bias = (summed_bias / counts_1d)?;

    Ok((pooled_emb, pooled_bias))
}

/// Reference implementation kept for the parity test only — see
/// `tests::pool_axis_index_add_matches_loop`. Identical semantics to
/// the previous version of [`pool_axis`].
#[cfg(test)]
fn pool_axis_loop(
    table: &Tensor,
    bias: &Tensor,
    coarse_blocks: &[u32],
    coarse_to_fine: &[Vec<usize>],
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let h = table.dim(1)?;
    let mut emb_rows: Vec<Tensor> = Vec::with_capacity(coarse_blocks.len());
    let mut bias_rows: Vec<Tensor> = Vec::with_capacity(coarse_blocks.len());

    for &block in coarse_blocks {
        let fine = &coarse_to_fine[block as usize];
        if fine.is_empty() {
            emb_rows.push(Tensor::zeros((h,), table.dtype(), dev)?);
            bias_rows.push(Tensor::zeros((), bias.dtype(), dev)?);
            continue;
        }
        let idx: Vec<u32> = fine.iter().map(|&i| i as u32).collect();
        let idx_t = Tensor::from_vec(idx, fine.len(), dev)?;
        let gathered = table.index_select(&idx_t, 0)?;
        let pooled = gathered.mean(0)?;
        emb_rows.push(pooled);

        let bias_g = bias.index_select(&idx_t, 0)?;
        let mean_b = bias_g.mean(0)?;
        bias_rows.push(mean_b);
    }

    let emb_stack = Tensor::stack(&emb_rows, 0)?;
    let bias_stack = Tensor::stack(&bias_rows, 0)?;
    Ok((emb_stack, bias_stack))
}

#[allow(unused)]
fn dummy_dtype_check() -> DType {
    DType::F32
}

#[cfg(test)]
mod tests;
