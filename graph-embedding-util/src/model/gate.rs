//! The feature gate: the spike-and-slab selection that multiplies every feature
//! loading, plus the constants it is calibrated by and the KL it pays for.
//!
//! One closed system, so one module. The logit / effect-log-std / `π_h` tables,
//! the `α = σ(S/τ)` lookup, the frozen-`pip` jitter path and the KL all read each
//! other's private helpers, and nothing outside the gate reads any of them.

use candle_util::candle_core::{DType, Device, Result, Tensor};
use candle_util::candle_nn::{self, VarMap};
use std::sync::{Arc, Mutex};

use super::JointEmbedModel;

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
/// Always added, at the fixed [`GATE_KL_WEIGHT`], scaled by
/// [`gate_kl_step_weight`] so the share of the objective it occupies does not
/// move when a throughput knob does:
///
///   `loss += mean_{g,h}[ KL(Bern(α_{g,h}) ‖ Bern(π_h)) + α_{g,h}·KL(N(E,σ²) ‖ N(0,σ₀²)) ]`
///          `+ Σ_h −log Beta(π_h; a, b)`
///
/// with `σ₀² =` [`GATE_EFFECT_PRIOR_VAR`] and `(a,b) =`
/// [`GATE_PI_BETA_A`]/[`GATE_PI_BETA_B`]. Both mean-reduced terms are `O(1)` per
/// entry, so unlike the softmax form neither needs a `ln D` or `D` correction to
/// stay commensurate with the other — see [`JointEmbedModel::single_gate_kl`] for
/// why that matters and what it cost last time.
#[derive(Clone, Copy, Debug)]
pub struct FeatureGateSpec {
    /// Gate temperature `τ` (`1.0` = plain sigmoid; `< 1` sharpens toward 0/1).
    pub temperature: f32,
}

/// Effect-prior variance `σ₀²` for the Gaussian KL `KL(N(μ,σ²) ‖ N(0,σ₀²))`.
pub const GATE_EFFECT_PRIOR_VAR: f64 = 1.0;
/// Fixed weight `λ` on the spike-and-slab KL. The gate is always variational.
/// A single source of truth rather than a per-run CLI knob; see
/// [`gate_kl_step_weight`] for how it reaches a step's loss.
pub const GATE_KL_WEIGHT: f64 = 1.0;

/// Calibration reference for [`gate_kl_step_weight`], in data-term units.
///
/// `1024` is deliberately the historical default `--batch-size` of both
/// `senna bge` and `faba gem`, whose data term contributes exactly one
/// per-example mean per axis. At that default the weight is numerically
/// identical to the `λ/batch_size` this replaced, so pinning the reference here
/// is what makes the change a correctness fix rather than a re-tune.
pub const GATE_KL_REF_UNITS: f64 = 1024.0;

/// How much of a step's loss the gate KL should be.
///
/// [`JointEmbedModel::gate_kl`] is a prior over GLOBAL parameters, not a
/// per-example term: it is mean-reduced per `(feature, dim)` entry and is `O(1)`
/// whatever the runtime knobs are. What matters is the share of the objective it
/// occupies, `w / u`, where `u` = `data_units` is the number of `O(1)`
/// per-example means the data term contributes THAT STEP. Holding `w/u` fixed
/// makes the fitted sparsity independent of batching. The step count cancels: it
/// multiplies the KL total and the data total over an epoch alike, so
/// "per-epoch KL total" is the wrong thing to hold constant.
///
/// `data_units` is a property of the caller's reduction: a MEAN data term is
/// intensive so `u` is its term count, a SUM data term is extensive so `u` is
/// the mass it summed.
///
/// The one caller today is geu's `train_composite`, via
/// `crate::training::gate_kl_weight_for`, and it passes `1` rather than its
/// axis count — deliberately, to reproduce the historical `λ/batch_size` level
/// exactly at the default `--batch-size`. `pinto cage` does NOT call this: it
/// applies its own Fisher-weighted mass ratio, on a different reference, so the
/// two are not calibrated to a common scale. Re-levelling them onto one is a
/// behavioural change and is deferred; until then, changing
/// [`GATE_KL_REF_UNITS`] moves geu's consumers only.
///
/// Before this existed geu used `λ/batch_size`, which made the share `∝ 1/B`:
/// `--batch-size 4096` cut the prior to a quarter and `64` raised it 16×, so a
/// throughput flag silently retuned how sparse the feature sets came out.
pub fn gate_kl_step_weight(kl_weight: f64, data_units: usize) -> f64 {
    kl_weight * data_units.max(1) as f64 / GATE_KL_REF_UNITS
}
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
pub(super) const GATE_LOGSTD_CLAMP: f64 = 8.0;

impl JointEmbedModel {
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
        } else if let Some(w) = self.free_feature_multiplier()? {
            // Free + gated: bake whatever multiplies the loading into `e_feat`, reading
            // the RAW Var so a second call cannot gate twice (no smoother at
            // materialize — SGC smoothing is a training-time device; means, no sample).
            //
            // "Whatever multiplies" is the point. It used to require the LEARNED gate's
            // Vars to be present, so an ungated model carrying only an installed `pip`
            // trained under the mask — `gather_feature_rows` takes the pip branch
            // regardless of the gate — and then shipped the dictionary unmasked,
            // because this arm did not fire. Phase 2 projected against a dictionary the
            // fit never used.
            let raw = self.e_feat_raw.as_ref().unwrap_or(&self.e_feat);
            Some(
                self.gated_rows(raw, self.e_feat_logstd.as_ref(), Some(&w), false)?
                    .detach(),
            )
        } else {
            // Free + ungated + no pip: `e_feat` already IS the trained Var — leave it
            // (no-op, byte-identical to the pre-gate behaviour).
            None
        };
        if let Some(g) = gated {
            self.e_feat = g;
        }
        Ok(())
    }

    /// What currently multiplies a FREE model's feature loading: the installed `pip`
    /// (or this epoch's draw from it), else the learned gate's `α = σ(S/τ)`, else
    /// `None` for a model with neither. Factored models compose their own via
    /// [`Self::factored_feat_rows`].
    ///
    /// The two cases must both be here, and in this order. `gate_weights` ignores its
    /// logits argument once a `pip` is installed, but it panics on an ungated model —
    /// `apply_temperature` reads `self.gate` — so the learned branch may only be
    /// reached when `s_feat` exists.
    fn free_feature_multiplier(&self) -> Result<Option<Tensor>> {
        if self.gate_pip.is_some() {
            let raw = self.e_feat_raw.as_ref().unwrap_or(&self.e_feat);
            return Ok(Some(self.gate_weights(GateKind::Identity, raw)?));
        }
        match self.s_feat.as_ref() {
            Some(s) => Ok(Some(self.gate_weights(GateKind::Identity, s)?)),
            None => Ok(None),
        }
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
    /// [`ShareFeaturesArgs`](super::ShareFeaturesArgs) / the cloned
    /// [`FeatFactor`](super::FeatFactor)).
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

    /// Scale gate logits by the gate temperature `τ` (`1/τ` on the logits); an
    /// Arc-cheap clone at `τ = 1`, which is what both CLIs pass. Shared by the gather
    /// and the KL.
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
    ///
    /// `pip` is `[rows, H]` on the gate's own axis — feature rows for a free model,
    /// genes for a factored one — and ALREADY ON THE DEVICE. Call
    /// [`Self::resample_gate_mask`] once per epoch afterwards; until then the model
    /// uses the mean.
    ///
    /// Takes a `Tensor` rather than a slice because the composite fit installs the SAME
    /// table on the primary model and on every pb head. A slice form would re-upload it
    /// per call — `[34k genes, 128 dims]` is 17 MB a copy, paid once per level on top
    /// of the one that was needed — whereas a `Tensor` clone shares storage, so the
    /// caller uploads once. (There WAS a slice form; it ended up with no caller outside
    /// its own tests once `install_selection` started uploading once, so it went.)
    pub fn install_gate_pip(&mut self, kind: GateKind, pip: &Tensor) -> Result<()> {
        // The row count is the caller's business — it differs per gate axis — but the
        // dim count is this model's, and a mismatch would otherwise surface much later
        // as a broadcast failure inside the training gather.
        assert_eq!(
            pip.dim(1)?,
            self.embedding_dim,
            "gate pip must be [rows, H]"
        );
        let t = pip.clone();
        match kind {
            GateKind::Identity => {
                self.gate_pip = Some(t);
                *self.gate_mask.lock().expect("gate mask poisoned") = None;
                // A free model with no LEARNED gate has no `e_feat_raw`, so `e_feat` is
                // both the trained Var and the field `materialize_e_feat` writes.
                // Pin the Var here (the clone aliases the same storage, so it keeps
                // tracking training) and materialize then reads a source it never
                // writes — otherwise a second materialize would gate an already-gated
                // dictionary. Factored models are unaffected: they rebuild from `beta`.
                if self.factor.is_none() && self.e_feat_raw.is_none() {
                    self.e_feat_raw = Some(self.e_feat.clone());
                }
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

    /// Point BOTH of this model's mask cells at existing shared ones, so every
    /// composite axis redraws together. Call after [`Self::set_gate_pip`] on each axis.
    ///
    /// Takes both gates at once ON PURPOSE. [`Self::set_gate_pip`] mints a fresh `Arc`
    /// for the velocity cell every time it is called, so a caller that shares only the
    /// identity cell leaves each axis holding its own δ mask — and since
    /// [`Self::resample_gate_mask`] runs on `axes[0]` alone, every OTHER axis then gates
    /// δ by the mean while `axes[0]` gates it by a draw. The axes train against
    /// different sub-models within one epoch, silently. A single entry point that
    /// cannot be half-called is the fix.
    pub fn share_gate_masks(
        &mut self,
        identity: &Arc<Mutex<Option<Tensor>>>,
        velocity: Option<&Arc<Mutex<Option<Tensor>>>>,
    ) {
        self.gate_mask = Arc::clone(identity);
        if let (Some(f), Some(cell)) = (self.factor.as_mut(), velocity) {
            if f.delta_gate_pip.is_some() {
                f.delta_gate_mask = Some(Arc::clone(cell));
            }
        }
    }

    /// The shared identity mask cell, for handing to the other axes.
    #[must_use]
    pub fn gate_mask_cell(&self) -> Arc<Mutex<Option<Tensor>>> {
        Arc::clone(&self.gate_mask)
    }

    /// The shared VELOCITY mask cell; `None` unless a δ pip is installed.
    #[must_use]
    pub fn velocity_mask_cell(&self) -> Option<Arc<Mutex<Option<Tensor>>>> {
        self.factor
            .as_ref()?
            .delta_gate_mask
            .as_ref()
            .map(Arc::clone)
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
    /// The SAFE external entry point for applying the gate.
    ///
    /// Gathers rows first, then transforms — exact because the gate is
    /// elementwise, and it avoids materializing `[G, H]` to keep `[batch, H]`.
    /// [`Self::gate_weights`] stays crate-private so out-of-crate callers cannot
    /// re-open the gather-vs-gate ordering hazard this form exists to prevent.
    pub fn gathered_gate_weights(
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

    /// The LEARNED gate's per-row inclusion table `α = σ(S/τ)` `[n_features, H]`,
    /// gathering a factored per-gene table to feature rows via `row_to_gene`. `None` if
    /// `logits` is `None`, or if a `pip` has taken over the selection (see below).
    ///
    /// Every entry is an INDEPENDENT probability in `(0,1)`. There is no null slot and
    /// no per-row budget — a deselected gene simply has `α → 0` on every dim, rather
    /// than sending mass somewhere. The predecessor was a per-dim softmax over genes,
    /// where a row summed to one and `1 − rowsum` WAS the excluded mass; reading these
    /// values against that convention inverts them.
    fn selection_from(&self, kind: GateKind, logits: Option<&Tensor>) -> Result<Option<Tensor>> {
        let Some(logits) = logits else {
            return Ok(None);
        };
        // A LEARNED readout only. Once a selection pass installs a `pip`, training
        // never consults these logits again, so `α = σ(S/τ)` is frozen at its init and
        // says nothing about the fit — while `gate_weights` would hand back the `pip`
        // itself, making this table a byte copy of `feature_pip.parquet` that the docs
        // then invite the reader to compare against it. Emit nothing instead.
        let (pip, _) = self.gate_tables(kind);
        if pip.is_some() {
            return Ok(None);
        }
        // Compute on the FULL table first, then expand per-gene → per-row: a factored
        // model repeats each gene once per splice track, so the other order would
        // transform over duplicated rows on the `Gene` axis.
        let w = self.gate_weights(kind, logits)?;
        let rows = match &self.factor {
            Some(f) => w.index_select(&f.row_to_gene, 0)?,
            None => w,
        };
        Ok(Some(rows.detach()))
    }

    /// Per-feature-row IDENTITY inclusion `σ(s_beta/s_feat)` `[n_features, H]`, for
    /// interpretability; `None` for an ungated model, and `None` under a `pip` (read
    /// `feature_pip.parquet` there). Rows align with `e_feat` / the dictionary output.
    pub fn feature_selection(&self) -> Result<Option<Tensor>> {
        self.selection_from(GateKind::Identity, self.gate_logits())
    }

    /// Per-feature-row VELOCITY inclusion `σ(s_delta)` `[n_features, H]` — the per-gene
    /// motion gate (driver genes); `None` unless the factored δ gate is on, and `None`
    /// under a `pip` (read `delta_pip.parquet` there).
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
        let sum_sq = (two_logstd.exp()? + mu.sqr()?)?; // σ² + μ²
                                                       // `(σ²+μ²)/σ₀²`. Skipped outright at `σ₀² = 1` — candle's `affine` only
                                                       // short-circuits on an EMPTY tensor, so a `×1 +0` still allocates a full
                                                       // `[rows, H]` buffer and adds a backward node. This runs once per minibatch per
                                                       // gate, on the whole feature table rather than the batch, so a no-op pass at
                                                       // 34k × 128 is ~17 MB of traffic for nothing.
        let a = if (s0 - 1.0).abs() > f64::EPSILON {
            sum_sq.affine(1.0 / s0, 0.0)?
        } else {
            sum_sq
        };
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
        // ONE walk of `GateKind`. Splitting the pip out into a second `gate_tables`
        // call meant dispatching twice and discarding the mask half of the result.
        let (logstd, pi_logit, pip) = match kind {
            GateKind::Identity => (
                self.effect_logstd(),
                self.gate_pi_logit.as_ref(),
                self.gate_pip.as_ref(),
            ),
            GateKind::Velocity => match &self.factor {
                Some(f) => (
                    f.delta_logstd.as_ref(),
                    f.delta_pi_logit.as_ref(),
                    f.delta_gate_pip.as_ref(),
                ),
                None => (None, None, None),
            },
        };
        let Some(logstd) = logstd else {
            return Ok(None);
        };
        if let Some(pip) = pip {
            return Ok(Some(Self::effect_kl(pip, logstd, mu)?));
        }
        let (Some(logits), Some(pi)) = (logits, pi_logit) else {
            return Ok(None);
        };
        Ok(Some(self.single_gate_kl(logits, pi, logstd, mu)?))
    }
}
