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

/// stdev of the embedding-table randn init (matches the former
/// `candle_nn::Init::Randn { stdev: 0.1 }`).
const INIT_STDEV: f32 = 0.1;

/// Softmax feature gate — a **per-dimension distribution over genes**.
///
/// Each gene's feature loading is `Ẽ_g = γ_g ⊙ β_g`: a SELECTION weight `γ_g`
/// times a Gaussian EFFECT `β_g` (the loading magnitude). Variationally,
///
///   `Ẽ_{g,h} = D · softmax_g(S_{·,h}/τ)[g] · (E_{g,h} + σ_{g,h}·ε)`,  ε ~ N(0,1),
///
/// where `S` is the selection logit table (`s_feat` free / `s_beta` factored,
/// `[D × H]`), `E_g` is the effect posterior MEAN, and `σ_g` (= `exp` of the
/// log-std table) its posterior std. Training samples via the reparameterization;
/// output/materialize use the mean (`σ = 0`).
///
/// # The softmax runs over GENES, within a dim
///
/// So every **dim** carries one unit of selection mass to distribute over genes —
/// the topic-model convention (`β_h` is a distribution over genes), and what makes
/// "the markers of dim `h`" a well-posed question.
///
/// The alternative — normalizing over dims within a gene — was implemented,
/// measured and removed. It gives each *gene* a budget and constrains no dim, so
/// dims are free to accumulate mass unequally: measured column sums on a real 12k
/// BMMC fit ran 167–536, a 3.2× spread, with the top two dims taking 25.7% of all
/// selected mass against 12.5% uniform. Under this axis that spread is **1.000 by
/// construction**. It also made the gate a SuSiE single-effect (one unit per gene
/// *is* a single effect) and forced a null "load-nothing" column to exist, since a
/// gene with no signal still had to spend its unit somewhere.
///
/// **There is no null column here.** With no per-gene budget, a gene belonging
/// nowhere simply takes small mass in every column, so an absorber has nothing to
/// absorb. Weights are rescaled by the gene count `D` so a uniform gate is exactly
/// the identity — a raw softmax over 20k genes averages 5e-5 and would collapse
/// the score by three orders of magnitude.
///
/// # KL loss
///
/// Always added, at the fixed [`GATE_KL_WEIGHT`], amortized `1/B` over the
/// minibatch:
///
///   `loss += Σ_h H(A_{·,h})  +  Σ_h A_{g,h}·KL(N(E_{g,h},σ_{g,h}²) ‖ N(0,σ₀²))`
///
/// The second term is the standard selection-weighted Gaussian KL with
/// `σ₀² =` [`GATE_EFFECT_PRIOR_VAR`]. The first is the column's Shannon **entropy**
/// — a concentration regularizer, and deliberately not dressed up as a KL. A KL to
/// a *uniform* prior over genes would be minimized by a dense dim, which is
/// backwards; the Bayesian alternative, a Dirichlet(α<1) log-prior
/// `(α−1)·Σ_g log A_{g,h}`, is unbounded as `A → 0` and runs away. Entropy is the
/// bounded equivalent (in `[0, ln D]`), minimized exactly when a dim concentrates
/// on few genes.
#[derive(Clone, Copy, Debug)]
pub struct SoftmaxGateSpec {
    /// Softmax temperature `τ` (`1.0` = plain softmax; `< 1` sharpens).
    pub temperature: f32,
}

/// Effect-prior variance `σ₀²` for the Gaussian KL `KL(N(μ,σ²) ‖ N(0,σ₀²))`.
pub const GATE_EFFECT_PRIOR_VAR: f64 = 1.0;
/// Fixed weight `λ` on the SuSiE single-effect KL. The gate is always variational;
/// the KL is applied amortized `λ/B` over the minibatch, so `1.0` gives the `1/B`
/// scale. A single source of truth rather than a per-run CLI knob.
pub const GATE_KL_WEIGHT: f64 = 1.0;
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
    /// updates it once (see [`SoftmaxGateSpec`]).
    pub shared_s_feat: Option<Tensor>,
    /// Shared raw feature Var kept reachable for the gated training gather (so a
    /// post-phase-1 materialize can bake the gate into `e_feat` without clobbering
    /// the source). `None` when the gate is off.
    pub shared_e_feat_raw: Option<Tensor>,
    /// Shared free-model effect log-std (`[n_features, H]`). `Some` only for a KL gate.
    pub shared_e_feat_logstd: Option<Tensor>,
    /// Gate configuration shared across heads. `None` when the gate is off.
    pub gate: Option<SoftmaxGateSpec>,
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
    /// Optional per-gene softmax-gate logits `[G, H]` (see [`SoftmaxGateSpec`]),
    /// the IDENTITY gate on `β_g`. Gathered by `row_to_gene` alongside `β`/`δ`; `None`
    /// = ungated. Cloned with the factor so composite heads share it.
    pub s_beta: Option<Tensor>,
    /// Optional per-gene Gaussian-effect log-std `[G, H]` (the `β` gate's variational
    /// single-effect posterior std; `σ = exp`). `Some` iff the gate is on.
    pub beta_logstd: Option<Tensor>,
    /// Optional per-gene VELOCITY-gate logits `[G, H]` — the independent
    /// single-effect softmax gate on `δ_g` (the motion), mirroring `s_beta`. `Some`
    /// only when the gate is on AND `splice_delta` exists (velocity present); a gene
    /// with no motion sends its `δ` mass to null → `δ̃_g ≈ 0` (not a driver).
    /// `softmax(s_delta)` per feature row = the `velocity_selection` output.
    pub s_delta: Option<Tensor>,
    /// Optional per-gene velocity Gaussian-effect log-std `[G, H]` (the `δ` gate's
    /// variational posterior std). `Some` iff `s_delta` is.
    pub delta_logstd: Option<Tensor>,
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
    /// Free-model softmax-gate logits `[n_features, H]` (see [`SoftmaxGateSpec`]).
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
    /// Gate configuration (`None` = ungated). Presence is the single "is gated" flag
    /// for both free (`s_feat`) and factored (`factor.s_beta`) models.
    pub gate: Option<SoftmaxGateSpec>,
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
                // knows to apply it.
                shared_s_feat: None,
                shared_e_feat_raw: None,
                shared_e_feat_logstd: None,
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
                    Some(&self.gate_weights(s_feat)?),
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

    /// Enable the variational softmax gate on the feature side (see
    /// [`SoftmaxGateSpec`]). Allocates, as Vars in `varmap`: the selection logits
    /// (`s_feat [n_features, H]` free / `s_beta [G, H]` factored) and the effect
    /// log-std (`e_feat_logstd` / `beta_logstd [·, H]`); for a factored model WITH
    /// velocity (`splice_delta`), also the INDEPENDENT δ gate (`s_delta`,
    /// `delta_logstd`). Logits start at **zero**, which the `D` rescale makes exactly
    /// the identity — an untrained gate is inert rather than biased. Call ONCE on the
    /// primary model, BEFORE building sharing heads (which carry the shared gate via
    /// [`ShareFeaturesArgs`] / the cloned [`FeatFactor`]).
    pub fn enable_softmax_gate(
        &mut self,
        spec: SoftmaxGateSpec,
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
        // Selection logits `[rows, H]`, all zero — a uniform gate, which the `D`
        // rescale makes exactly the identity, so training starts from "no selection"
        // rather than from an arbitrary bias. (No null column: the budget is on the
        // dims, so a gene belonging nowhere just takes small mass in every column.)
        let init_gate =
            |name: &str, rows: usize| register(name, Tensor::zeros((rows, h), DType::F32, dev)?);
        // Effect log-std `[rows, H]`, init `GATE_LOGSTD_INIT` (near-deterministic start).
        let init_logstd = |name: &str, rows: usize| -> Result<Tensor> {
            register(
                name,
                Tensor::from_vec(vec![GATE_LOGSTD_INIT; rows * h], (rows, h), dev)?,
            )
        };
        match &mut self.factor {
            Some(f) => {
                let n_genes = f.beta.dim(0)?;
                f.s_beta = Some(init_gate("s_beta", n_genes)?);
                f.beta_logstd = Some(init_logstd("beta_logstd", n_genes)?);
                // Independent velocity gate on δ_g, only when velocity is present.
                if f.splice_delta.is_some() {
                    f.s_delta = Some(init_gate("s_delta", n_genes)?);
                    f.delta_logstd = Some(init_logstd("delta_logstd", n_genes)?);
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

    /// The gate's multiplier table `[rows, H]` for a **whole** logit table.
    ///
    /// # This must be handed the FULL table, never a gathered minibatch
    ///
    /// The normalizer runs down the GENE axis, so a
    /// gathered subset would silently normalize over the minibatch — and over
    /// duplicated rows, since negative sampling repeats genes. It would compile,
    /// run, and train garbage. Callers therefore compute this once on the full
    /// table and `index_select` **afterwards**; [`Self::gated_rows`] takes the
    /// already-gathered weights, not logits, so the wrong order is not expressible.
    ///
    /// `Gene` weights are scaled by the row count: a raw softmax over `D` genes
    /// averages `1/D`, which would shrink every score by three orders of magnitude
    /// at `D = 20k`. Scaling by `D` makes a uniform gate exactly the identity, so
    /// the two axes are on the same footing and `temperature` means the same thing.
    pub(crate) fn gate_weights(&self, full_logits: &Tensor) -> Result<Tensor> {
        let logits = self.apply_temperature(full_logits)?;
        let n_genes = logits.dim(0)?;
        candle_nn::ops::softmax(&logits, 0)?.affine(n_genes as f64, 0.0)
    }

    /// The ONLY way to produce a `w` argument for [`Self::gated_rows`]: normalize the
    /// full logit table, then gather the requested rows.
    ///
    /// Both steps live here so the ordering cannot be got wrong at a call site. It
    /// matters because the softmax reduces down the GENE axis: gathering first would
    /// normalize over whatever subset was gathered — a minibatch (with rows negative
    /// sampling duplicates), or a factored model's feature rows (which repeat each
    /// gene once per splice track). Either compiles, runs, and trains garbage.
    pub(crate) fn gathered_gate_weights(
        &self,
        logits: Option<&Tensor>,
        rows: &Tensor,
    ) -> Result<Option<Tensor>> {
        logits
            .map(|x| self.gate_weights(x)?.index_select(rows, 0))
            .transpose()
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
            self.gathered_gate_weights(f.s_beta.as_ref(), genes)?,
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
                    self.gathered_gate_weights(f.s_delta.as_ref(), genes)?,
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
    fn selection_from(&self, logits: Option<&Tensor>) -> Result<Option<Tensor>> {
        let Some(logits) = logits else {
            return Ok(None);
        };
        // Normalize on the FULL table first, then expand per-gene → per-row. Doing
        // it the other way round would normalize over duplicated rows on the `Gene`
        // axis (a factored model repeats each gene once per splice track), halving
        // every value while leaving the β-sharing equality test passing.
        let w = self.gate_weights(logits)?;
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
        self.selection_from(self.gate_logits())
    }

    /// Per-feature-row VELOCITY selection `softmax(s_delta)` `[n_features, H]` — the
    /// per-gene motion gate (driver genes); `None` unless the factored δ gate is on.
    pub fn velocity_selection(&self) -> Result<Option<Tensor>> {
        self.selection_from(self.factor.as_ref().and_then(|f| f.s_delta.as_ref()))
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

    /// One gate's regularization term: the per-dim concentration penalty plus the
    /// selection-weighted Gaussian effect KL (`σ₀² =` [`GATE_EFFECT_PRIOR_VAR`]).
    /// Shared by the identity (`β`/`e_feat`) and velocity (`δ`) gates. In the
    /// autograd graph.
    ///
    /// # Both terms are deliberately kept `O(1)` in `D` and `H`
    ///
    /// They are summed under one [`GATE_KL_WEIGHT`], so if their scales diverge the
    /// larger silently sets the loss and the smaller stops existing. That is not
    /// hypothetical — it is what happened when the gate's softmax moved from the dim
    /// axis to the gene axis: `α` went from `≈1/(H+1)` to `≈1/D`, the forward pass
    /// compensated via the `D` rescale in [`Self::gate_weights`] but this function
    /// did not, and at `D = 20287, H = 16` the effect prior came out **~49,000×**
    /// smaller than the concentration term — numerically switched off. So:
    ///
    /// * the entropy is normalized by its own `ln D` maximum and MEANED over dims,
    ///   putting it in `[0, 1]` whatever the gene count or embedding width;
    /// * the Gaussian KL is weighted by the weights the LIKELIHOOD actually applies
    ///   (`D·α`, i.e. what [`Self::gate_weights`] returns), not raw `α`, and meaned
    ///   over every entry.
    ///
    /// `gate_terms_are_commensurate` pins the ratio so a future reduction change
    /// cannot re-open this quietly.
    fn single_gate_kl(&self, logits: &Tensor, logstd: &Tensor, mu: &Tensor) -> Result<Tensor> {
        let logits = self.apply_temperature(logits)?;
        // Normalize down the GENE axis — the same reduction `gate_weights` does.
        let log_alpha = candle_nn::ops::log_softmax(&logits, 0)?;
        let alpha = log_alpha.exp()?;
        let n_genes = alpha.dim(0)? as f64;

        // Concentration: each dim's Shannon entropy `H(A_{·,h}) = −Σ_g A log A`,
        // scaled by its `ln D` maximum and averaged over dims → `[0, 1]`, where 1 is
        // a perfectly spread dim and 0 a dim concentrated on one gene.
        // `sum(0)` then mean-over-dims is `sum_all / H`; written as `sum_all` because
        // a dim-0 reduce misses candle's vectorized path (it only fast-paths
        // reductions over the LAST dims) while `sum_all` takes it.
        let n_dims = alpha.dim(1)? as f64;
        let kl_cat = alpha
            .mul(&log_alpha)?
            .sum_all()?
            .affine(-1.0 / (n_genes.ln() * n_dims), 0.0)?;

        // Gaussian KL per entry = ½[(σ²+μ²)/σ₀² − 1 − 2·logstd + ln σ₀²], σ² = e^{2 logstd}.
        let s0 = GATE_EFFECT_PRIOR_VAR;
        let logstd = logstd.clamp(-GATE_LOGSTD_CLAMP, GATE_LOGSTD_CLAMP)?; // finite σ, log σ²
        let two_logstd = logstd.affine(2.0, 0.0)?; // 2·logstd = log σ²
        let a = (two_logstd.exp()? + mu.sqr()?)?.affine(1.0 / s0, 0.0)?; // (σ²+μ²)/σ₀²
        let kl_gauss_dim = (a - &two_logstd)?.affine(0.5, 0.5 * (s0.ln() - 1.0))?; // ½[…]
                                                                                   // Weighted by the gate the likelihood sees, then averaged over all entries.
        let kl_gauss = alpha.affine(n_genes, 0.0)?.mul(&kl_gauss_dim)?.mean_all()?;

        kl_cat + kl_gauss
    }

    /// The gate's total SuSiE KL loss: the identity-gate KL plus, for a factored model
    /// with velocity, the INDEPENDENT δ-gate KL. `None` for an ungated model. Kept in
    /// the autograd graph.
    pub fn gate_kl(&self) -> Result<Option<Tensor>> {
        if self.gate.is_none() {
            return Ok(None);
        }
        // Identity gate (β factored / e_feat free) — always present when gated.
        let (Some(logits), Some(logstd)) = (self.gate_logits(), self.effect_logstd()) else {
            return Ok(None);
        };
        let mu = match &self.factor {
            Some(f) => &f.beta,
            None => self.e_feat_raw.as_ref().unwrap_or(&self.e_feat),
        };
        let mut kl = self.single_gate_kl(logits, logstd, mu)?;
        // Independent velocity gate on δ_g (factored + velocity present).
        if let Some(f) = &self.factor {
            if let (Some(s_delta), Some(delta_logstd), Some((delta, _))) =
                (&f.s_delta, &f.delta_logstd, &f.splice_delta)
            {
                kl = (kl + self.single_gate_kl(s_delta, delta_logstd, delta)?)?;
            }
        }
        Ok(Some(kl))
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
