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

use candle_util::candle_core::{Device, Result, Tensor};
use candle_util::candle_nn::{self, VarBuilder, VarMap};
use std::sync::{Arc, Mutex};

mod gate;
mod score;
mod vars;

pub use gate::{
    gate_kl_step_weight, ibp_alpha_for_drop, ibp_gate_logit_bias, FeatureGateSpec, GateKind,
    GATE_EFFECT_PRIOR_VAR, GATE_IBP_LOGIT_DROP, GATE_KL_REF_UNITS, GATE_KL_STEP_WEIGHT,
    GATE_KL_WEIGHT,
};
use vars::{
    build_feat_factor, register_randn_seeded, register_var_from_mat, register_var_from_slice,
};
// Reached only through `tests`' `use super::*` — the live `pool_axis` caller is in
// `score`, and `DType` now belongs to `vars`.
#[cfg(test)]
use candle_util::candle_core::DType;
#[cfg(test)]
use vars::{pool_axis, pool_axis_loop};

/// stdev of the embedding-table randn init (matches the former
/// `candle_nn::Init::Randn { stdev: 0.1 }`).
const INIT_STDEV: f32 = 0.1;

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
    pub shared_gate_ibp_bias: Option<Tensor>,
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
    pub embedding_dim: usize,
    /// Free-model gate logits `[n_features, H]` (see [`FeatureGateSpec`]). `None` for
    /// an ungated model, or for a factored one (its gate lives in `factor.s_beta`).
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
    /// It MUST be plumbed through [`ShareFeaturesArgs`] like `s_feat`, and this doc once
    /// said the opposite — that only the axis computing [`Self::gate_kl`] needs it, so
    /// sharing it was unnecessary. That reasoning is inverted: training calls `gate_kl`
    /// on `axes[0]`, which is a pb HEAD, not the model the gate was enabled on, whenever
    /// `--phase1-cells-per-pb` is 0 — the default in both CLIs. A head without it makes
    /// `gate_kl` return `None`, and the entire gate regularisation leaves the loss with
    /// the build green. Do not un-share it.
    ///
    /// The IBP ladder inherits that hazard in a nastier form: a head with
    /// `gate_ibp_bias: None` still gates and still trains, it just silently drops
    /// the per-dim prior — no `None`, no error, just a fit with no ordering on its
    /// dims. `every_shared_head_carries_the_ibp_ladder` pins it.
    ///
    /// A constant, not a `Var`: `α` is chosen, so this has no gradient and is not
    /// a checkpointed parameter. Both gates share ONE ladder — it is a function of
    /// `α` and `H` alone, so the identity and velocity gates cannot disagree about
    /// it the way they legitimately disagree about a learned rate.
    pub gate_ibp_bias: Option<Tensor>,
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
            gate_ibp_bias: None,
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
            shared_gate_ibp_bias,
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
            gate_ibp_bias: shared_gate_ibp_bias,
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
            gate_ibp_bias: None,
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
                shared_gate_ibp_bias: self.gate_ibp_bias.clone(),
                gate: self.gate,
            },
            varmap,
            dev,
        )?;
        model.factor = Some(factor.clone());
        Ok(model)
    }
}

#[cfg(test)]
mod tests;
