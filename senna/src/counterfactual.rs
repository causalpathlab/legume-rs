//! Counterfactual axes **benefit** / **forgetting** for `senna probe --counterfactual`.
//!
//! `probe`'s default score is the potential outcome `Y(0)` — does the frozen model
//! explain these cells. This module estimates the **effect of updating on them**, to
//! first order:
//!
//! ```text
//!   g  = ∇ℓ(Q_fit) − ∇ℓ(C_ctrl)          the contrast direction   [K,H]
//!
//!   benefit    =  ⟨ ∇ℓ(Q_eval), g ⟩       efficacy  (> 0 ⇒ the batch teaches)
//!   forgetting = −⟨ ∇ℓ(C_eval), g ⟩       toxicity  (> 0 ⇒ the batch damages)
//! ```
//!
//! These are the first-order terms of a refit: `ℓ(eval; α + η∇ℓ(fit)) − ℓ(eval; α₀)`
//! expands to `η⟨∇ℓ(eval), ∇ℓ(fit)⟩`, and differencing treatment against control leaves
//! `η⟨∇ℓ(eval), g⟩`. The step size `η` is a common scale, so it is dropped — it cancels in
//! every comparison between datasets. **Do not normalize `g`**; see below.
//!
//! Read it as: *step `α` in the direction this batch pulls, rather than where ordinary
//! reference data pulls; how does held-out fit move on each side?* `forgetting` is negated
//! so "larger is worse" on both axes; the price is that the size-weighted net gain
//! **subtracts** — `n_query · benefit − n_reference · forgetting` — so the minus sign
//! lives here and nowhere else.
//!
//! The `forgetting` axis is the reason this module exists. A goodness-of-fit score cannot
//! see it: a contaminated but in-distribution batch reconstructs perfectly and still drags
//! the dictionary off.
//!
//! **The control arm is enacted, not subtracted.** Even a null batch improves fit when you
//! train on it, so the estimand is *"the effect of adding **this** batch rather than
//! ordinary in-distribution data"*. That is the `− ∇ℓ(C_ctrl)` term, and dropping it would
//! measure "does training help at all", which is always yes.
//!
//! **`forgetting > 0` does not mean "refuse".** A batch carrying a topic the model
//! lacks *must* distort the existing ones to make room, so genuine novelty and genuine
//! contamination both forget — they are not separable on this axis alone. Nor is
//! forgetting unconditionally a cost: knowledge is worth protecting in proportion to how
//! likely it is to be needed again, which is a property of the dataset stream and is not
//! observable from one probe call.
//!
//! **Intervention (stated, not implied).** The encoder is frozen, so cell representations
//! `θ_j` are fixed and only the dictionary `α` moves. This is "update the dictionary", not
//! a joint retrain — a real one would also move the encoder, hence `θ`.
//!
//! **Scale and direction are reported separately, because they answer different
//! questions.** `‖g‖` is *how hard* this batch pulls; the two cosines are *where*. Both
//! effects are projections of the same `g`, so they are **correlated by construction** —
//! a merely divergent dataset inflates `‖g‖` and hence both at once. Coupled benefit and
//! forgetting is geometry, never evidence about the data.
//!
//! ⚠️ **Never divide the effects by `‖g‖`.** It looks like a harmless standardization and
//! it inverts the null: a query drawn from the reference has `g ≈ 0` — pure sampling noise
//! — and rescaling that to unit length collides it with the parent's own misfit direction,
//! which is large because the parent is not a stationary point of the scored objective. An
//! in-distribution arm then outscores one carrying a held-out cell type. The cosines exist
//! to carry the scale-free reading without contaminating the effects.
//!
//! Direction and evaluation use disjoint cells, so neither effect carries in-sample
//! optimism, and roles come from a **seeded shuffle**, not file position — see `splits`.
//!
//! # ❌ Do not reintroduce the refit (removed 2026-08-07)
//!
//! Earlier versions ran two 100-step full-batch AdamW refits of `α` — treatment on
//! `C_base ∪ Q_fit`, control on `C_base ∪ C₂` — and differenced held-out fit. That put
//! `--cf-steps`, `--cf-lr` and an inherited `weight_decay: 0.01` **inside the estimand**:
//! "the effect of absorbing this dataset" silently meant "under this optimizer
//! configuration", and no two runs at different settings were comparable.
//!
//! The first-order effect needs none of that — no step count, no learning rate, no
//! shrinkage prior, and no stationarity assumption either. **That last point is what
//! distinguishes this from the deleted influence screen** (`influence.rs`, 0.8.6): that one
//! needed `H⁻¹` and a quadratic approximation around a mode, which the parent's non-zero
//! training gradient invalidated. A directional derivative is valid wherever it is taken.
//!
//! What the refit could see and this cannot: **higher-order terms**. Over many steps topics
//! can reorganize in ways no derivative at the parent predicts. If that turns out to
//! matter, report the curvature along `g` — do not restore a procedure whose
//! hyperparameters are indistinguishable from its findings.
//!
//! # ❌ Do not reintroduce the label-permutation null (removed 2026-08-07)
//!
//! Earlier versions permuted the treatment/control label over the pooled fit cells
//! `Q_fit ∪ C₂` and reported upper-tail p-values. The test was *valid* and it was
//! *useless*, for a structural reason:
//!
//! `Q_eval` is pure query and fixed across permutations. Under a permuted split each arm
//! receives about half of `Q_fit`, so both refits explain `Q_eval` about equally well and
//! the permuted `benefit` sits near zero. The observed arrangement — all query cells in
//! treatment, none in control — is the extreme point of the arrangement space by
//! construction. The same argument applies to `forgetting` with `C_eval`.
//!
//! So the sharp null being tested was *"the query cells are exchangeable with reference
//! cells"* — i.e. the new dataset is drawn from the reference distribution. That is false
//! for every dataset anyone would probe, so the test had near-total power against it and
//! `p` collapsed to the `1/(N+1)` floor mechanically, identically for arms differing in
//! biology, in platform, and in both. Only a query genuinely drawn from the reference
//! failed to reject — the consistency check confirming the diagnosis, not a defence.
//!
//! A p-value that fires on any distributional difference answers the fit score's question,
//! not this module's. **The effect sizes carry the information; the p-values did not.**
//! What would supply a real scale is a *reference* arm — the same contrast with a held-out
//! slice of the reference playing the query role — which is how the fit score is already
//! calibrated. Until that lands, this module reports magnitudes and no verdict.

use crate::embed_common::*;
use crate::topic::eval::GeneRemap;
use crate::topic::eval_indexed::{csc_to_indexed, PerGeneContext};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use candle_util::decoder::{EmbeddedNbTopicDecoder, MaskedNbTarget};
use data_beans::sparse_io_vector::SparseIoVec;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A model rebuilt from disk: the varmap, the **finest** level's decoder (the only one
/// ever scored), and the name of its `α` var.
struct RebuiltModel {
    pub parameters: VarMap,
    pub decoder: EmbeddedNbTopicDecoder,
    /// `dec_{num_levels-1}.topic.embeddings` — derived, not stored in the metadata.
    pub alpha_name: String,
}

/// Rebuild the encoder (for the shared `ρ` handle and the `enc.*` vars the safetensors
/// file expects) plus **all** `num_levels` decoders, then load the trained weights.
///
/// The coarser levels are constructed purely to **register** their vars and are then
/// dropped: `EmbeddedNbTopicDecoder::new` registers through `VarBuilder::get_with_hints`,
/// so the `Var`s outlive the struct. `VarMap::save` writes exactly what is registered, so a
/// caller that ever persists this varmap would otherwise drop `dec_0..dec_{lvl-1}` and produce
/// a checkpoint `masked-topic --init-from` cannot load.
///
/// `ρ` is **detached** before the decoders take it. `Arm::grad` differentiates w.r.t. `α`
/// alone, but `backward()` walks every `is_variable()` node it reaches — and `ρ` is the
/// encoder's `Var`, shared by construction. Left attached it costs, per gradient, a
/// `[H,K]×[K,D]` matmul backward plus an index_add scattering `[N·C, H]` into a zeroed
/// `[D,H]`, both discarded. Detaching shares the same storage and only clears the variable
/// flag, so values still round-trip through `VarMap::save` — pinning `ρ` is the intent
/// anyway.
fn rebuild_model(
    model: &str,
    metadata: &crate::topic::model_metadata::TopicModelMetadata,
    dev: &Device,
) -> anyhow::Result<RebuiltModel> {
    use candle_util::encoder::{IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs};

    let embedding_dim = metadata
        .embedding_dim
        .ok_or_else(|| anyhow::anyhow!("counterfactual: metadata missing embedding_dim"))?;
    let mut parameters = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&parameters, DType::F32, dev);

    let encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: metadata.n_features_full,
            n_topics: metadata.n_topics,
            embedding_dim,
            layers: &metadata.encoder_hidden,
            use_gcn: false,
            attn_pool: true,
            // Must match the checkpoint: M widens the first FC layer, and `VarMap::load`
            // errors on a shape mismatch.
            n_gene_modules: metadata.n_gene_modules.unwrap_or(0),
        },
        &parameters,
        vb.pp("enc"),
    )?;

    let rho = encoder.feature_embeddings().detach();
    let num_levels = metadata.num_levels.max(1);
    let finest = num_levels - 1;
    for i in 0..finest {
        EmbeddedNbTopicDecoder::new(metadata.n_topics, rho.clone(), vb.pp(format!("dec_{i}")))?;
    }
    let decoder =
        EmbeddedNbTopicDecoder::new(metadata.n_topics, rho, vb.pp(format!("dec_{finest}")))?;

    parameters.load(format!("{model}.safetensors"))?;

    Ok(RebuiltModel {
        alpha_name: format!("dec_{finest}.topic.embeddings"),
        parameters,
        decoder,
    })
}

/// One side (calibration or query) of the cell bank.
struct BankSource<'a> {
    pub data_vec: &'a SparseIoVec,
    pub z_nk: &'a Mat,
    pub gene_remap: Option<&'a GeneRemap>,
}

struct BankArgs<'a> {
    pub calib: BankSource<'a>,
    pub query: BankSource<'a>,
    pub context_size: usize,
    pub feature_mean: &'a [f32],
    pub shortlist_weights: &'a [f32],
    pub dev: &'a Device,
}

/// Calibration cells `[0, n_calib)` then query cells `[n_calib, n_calib+n_query)`,
/// packed once. Subsets are `index_select`, so permutations cost no I/O.
pub(crate) struct CellBank {
    indices: Tensor, // [N, C] u32
    values: Tensor,  // [N, C] f32
    log_z: Tensor,   // [N, K] f32 (log θ; the encoder is frozen)
    pub n_calib: usize,
    pub n_query: usize,
    dev: Device,
}

/// A storage-independent copy. `Tensor::clone` is shallow, and `Var::set` refuses a
/// source derived from the var's own value.
fn detached_copy(t: &Tensor) -> anyhow::Result<Tensor> {
    let dims = t.dims().to_vec();
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    Ok(Tensor::from_vec(v, dims, t.device())?)
}

fn mat_to_tensor(m: &Mat, dev: &Device) -> anyhow::Result<Tensor> {
    let (n, k) = (m.nrows(), m.ncols());
    let mut v = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k {
            v.push(m[(i, j)]);
        }
    }
    Ok(Tensor::from_vec(v, (n, k), dev)?)
}

fn pack_side(src: &BankSource<'_>, a: &BankArgs<'_>) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let ntot = src.data_vec.num_columns();
    let csc = src.data_vec.read_columns_csc(0..ntot)?;
    let pack = csc_to_indexed(
        &csc,
        a.context_size,
        a.shortlist_weights,
        src.gene_remap.map(|r| r.new_to_train.as_slice()),
        PerGeneContext {
            feature_mean: Some(a.feature_mean),
        },
        a.dev,
    )?;
    let log_z = mat_to_tensor(src.z_nk, a.dev)?;
    Ok((pack.indices, pack.values, log_z))
}

impl CellBank {
    /// Pack a calibration/query pair from an opened model plus its two scored sides.
    ///
    /// `probe` and `update` both need exactly this wiring, and the model's per-gene mean
    /// / shortlist / context size all come from the already-open [`MaskedModel`], so
    /// nothing is re-read from disk here.
    pub fn from_scored(
        model: &crate::topic::masked_artifact::MaskedModel<'_>,
        cal: &crate::predict::MaskedScored,
        query: &crate::predict::MaskedScored,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let context_size = model
            .metadata
            .enc_context_size
            .ok_or_else(|| anyhow::anyhow!("metadata missing enc_context_size"))?;
        Self::build(BankArgs {
            calib: BankSource {
                data_vec: &cal.data_vec,
                z_nk: &cal.z_nk,
                gene_remap: cal.gene_remap.as_ref(),
            },
            query: BankSource {
                data_vec: &query.data_vec,
                z_nk: &query.z_nk,
                gene_remap: query.gene_remap.as_ref(),
            },
            context_size,
            feature_mean: &model.feature_mean,
            shortlist_weights: &model.shortlist,
            dev,
        })
    }

    fn build(a: BankArgs<'_>) -> anyhow::Result<Self> {
        let (ci, cv, cz) = pack_side(&a.calib, &a)?;
        let (qi, qv, qz) = pack_side(&a.query, &a)?;
        let n_calib = ci.dim(0)?;
        let n_query = qi.dim(0)?;
        Ok(Self {
            indices: Tensor::cat(&[&ci, &qi], 0)?,
            values: Tensor::cat(&[&cv, &qv], 0)?,
            log_z: Tensor::cat(&[&cz, &qz], 0)?,
            n_calib,
            n_query,
            dev: a.dev.clone(),
        })
    }

    /// `(indices, values, log_z)` for the given cell ids.
    fn subset(&self, ids: &[usize]) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let idx = Tensor::from_vec(
            ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            ids.len(),
            &self.dev,
        )?;
        Ok((
            self.indices.index_select(&idx, 0)?,
            self.values.index_select(&idx, 0)?,
            self.log_z.index_select(&idx, 0)?,
        ))
    }
}

/// Negative-binomial log-likelihood of every observed gene, `θ` held fixed.
///
/// `impute_masked_nb` scores the *masked* positions; masking every real position
/// (`value > 0`) turns the training loss into a deterministic full-cell score.
fn nb_llik(
    decoder: &EmbeddedNbTopicDecoder,
    indices: &Tensor,
    values: &Tensor,
    log_z: &Tensor,
) -> anyhow::Result<(Tensor, Tensor)> {
    let full_kd = decoder.full_logits_kd()?;
    let logz_11k = EmbeddedNbTopicDecoder::log_partition_from_logits(&full_kd)?;
    let lib_n1 = (values.sum_keepdim(1)? + 1.0)?;
    let mask = values.gt(0.0)?.to_dtype(DType::F32)?;
    let target = MaskedNbTarget {
        indices,
        residual: None,
        values,
        lib: &lib_n1,
        mask: &mask,
    };
    let llik = decoder.impute_masked_nb(log_z, &target, &logz_11k)?;
    Ok((llik, mask))
}

/// Half-width of the central difference, as a fraction of `‖α‖`. Small enough that the
/// linear term dominates, large enough that `f32` cancellation does not: per-cell fit is
/// `O(1)` nats, so the differenced signal is `~2ε` against an `f32` relative precision of
/// `~1e-7`. `Counterfactual::fd_warning` checks the choice against autograd on every call
/// rather than trusting it.
const FD_EPS_REL: f64 = 1e-3;

/// Warn above this relative gap between a finite-difference effect and its autograd
/// counterpart. Leaves room for `f32` noise while still catching a mis-scaled step. Lives
/// beside `FD_EPS_REL` because it is a judgement about *that* constant, and the caller
/// cannot see it.
const FD_CHECK_TOL: f64 = 0.01;

/// Per-cell mean log-likelihood over each cell's scored genes — `[N]`, differentiable.
///
/// The single definition of "fit" in this module. Both the gradients that build the
/// contrast direction and the values read out on the evaluation sets come from here, so
/// "the direction and the outcome are the same functional" is enforced by construction
/// rather than by two call sites agreeing.
fn per_cell_fit(
    decoder: &EmbeddedNbTopicDecoder,
    bank: &CellBank,
    ids: &[usize],
) -> anyhow::Result<Tensor> {
    let (idx, val, lz) = bank.subset(ids)?;
    // `impute_masked_nb` already sums over the cell's scored genes -> `[N]`.
    let (llik, mask) = nb_llik(decoder, &idx, &val, &lz)?;
    let den = mask.sum(1)?.clamp(1.0f32, 1e30f32)?;
    Ok(llik.div(&den)?)
}

/// A loaded model plus the handles needed to read gradients and directional derivatives
/// off it. Bundled rather than passed as four repeated arguments to every helper.
struct Arm<'a> {
    decoder: &'a EmbeddedNbTopicDecoder,
    alpha: candle_core::Var,
    /// The trained `α`, detached. Every perturbation departs from and returns to it.
    alpha0: Tensor,
    bank: &'a CellBank,
}

impl Arm<'_> {
    /// `∇_α` of the mean per-cell fit over `ids`, at the current `α`.
    fn grad(&self, ids: &[usize]) -> anyhow::Result<Tensor> {
        let grads = per_cell_fit(self.decoder, self.bank, ids)?
            .mean_all()?
            .backward()?;
        grads
            .get(self.alpha.as_tensor())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("counterfactual: α received no gradient"))
    }

    /// Per-cell `⟨∇ℓᵢ, dir⟩` by central difference — **unnormalized**, so the caller gets
    /// the projection onto `dir` itself rather than onto `dir/‖dir‖`.
    ///
    /// Finite differences rather than autograd because the **per-cell** decomposition is
    /// what the influence-function variance needs, and per-cell autograd would cost one
    /// backward pass per cell. Two forward passes give all of them.
    ///
    /// The walk is along the *unit* direction for numerical conditioning, with `‖dir‖`
    /// folded back into the divisor — no second pass over the result.
    fn ddir(&self, ids: &[usize], dir: &Tensor, eps: f64) -> anyhow::Result<Vec<f32>> {
        let scale = tensor_norm(dir)?;
        anyhow::ensure!(scale > 0.0, "counterfactual: zero direction");
        let step = (dir * (eps / scale))?;

        // `α` is restored unconditionally, so an error inside the walk cannot leave the
        // shared `Var` perturbed for the gradients read afterwards.
        let walk = || -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            self.alpha.set(&(&self.alpha0 + &step)?)?;
            let plus = per_cell_fit(self.decoder, self.bank, ids)?.to_vec1::<f32>()?;
            self.alpha.set(&(&self.alpha0 - &step)?)?;
            let minus = per_cell_fit(self.decoder, self.bank, ids)?.to_vec1::<f32>()?;
            Ok((plus, minus))
        };
        let walked = walk();
        self.alpha.set(&self.alpha0)?;
        let (plus, minus) = walked?;

        let inv = (scale / (2.0 * eps)) as f32;
        Ok(plus
            .iter()
            .zip(&minus)
            .map(|(p, m)| (p - m) * inv)
            .collect())
    }
}

pub(crate) struct Counterfactual {
    /// First-order fit **gained** on held-out query cells along `g`. `> 0` ⇒ the batch
    /// teaches. Efficacy. Computed by autograd, so it is exact rather than differenced.
    pub benefit: f64,
    /// First-order fit **lost** on held-out reference cells along `g`, sign-flipped so
    /// larger is worse. `> 0` ⇒ forgetting. Toxicity.
    ///
    /// Both read the *same* direction `g` against two disjoint evaluation sets. Because
    /// `forgetting` is negated, the size-weighted net gain **subtracts**:
    /// `n_query · benefit − n_reference · forgetting`. Do not add them.
    ///
    /// ⚠️ They are two projections of one vector, so they are **correlated by
    /// construction** — near-orthogonal `∇ℓ(Q_eval)` and `∇ℓ(C_eval)` is the only way they
    /// come apart. Coupled benefit and forgetting is geometry, never evidence about the
    /// data.
    pub forgetting: f64,
    /// Standard errors from the **infinitesimal jackknife** — the delta-method limit of the
    /// bootstrap, in closed form, no resampling.
    ///
    /// The SE is stored rather than an interval so the confidence level is the reporter's
    /// choice and never has to be inferred back out of a pair of bounds.
    ///
    /// Each effect is a bilinear form `⟨â, b̂⟩` in two independent sample means, so its
    /// variance is the sum of one term per sampled set, each the sample variance of a
    /// per-cell projection over that set's size:
    ///
    /// ```text
    ///   Var(benefit) = Var_k⟨∇ℓ_k, g⟩/n_qeval        which eval cells were drawn
    ///                + Var_i⟨∇ℓ_i, ∇ℓ(Q_eval)⟩/n_qfit    which cells set the direction
    ///                + Var_j⟨∇ℓ_j, ∇ℓ(Q_eval)⟩/n_cctrl   which cells set the control
    /// ```
    ///
    /// The second and third terms are the reason this replaced a bootstrap over eval cells:
    /// that form could only ever see the first, treating the one realized direction as
    /// fixed when *which cells defined it* is the larger source of variation. Every
    /// projection is a directional derivative, so all three come from passes already
    /// being made.
    ///
    /// ⚠️ Still not reachable: `Q_fit` and `Q_eval` are halves of the **same** dataset, so
    /// no term here sees between-dataset variation. Intervals derived from these are
    /// symmetric, the delta method being first-order.
    pub benefit_se: f64,
    pub forgetting_se: f64,
    /// `‖g‖` — how hard this batch pulls `α` beyond what ordinary reference data does.
    ///
    /// The scale/direction split matters: `‖g‖` is **how hard** the batch pulls and the
    /// two cosines are **where** it pulls. A merely divergent dataset inflates `‖g‖`, hence
    /// both effects at once; the cosines are what distinguish a pull that generalizes
    /// within the batch from one that opposes the reference.
    pub pull_norm: f64,
    /// `cos(∇ℓ(Q_eval), g)` — does the pull generalize to *held-out cells of the same
    /// batch*? This is the axis `benefit` is weakest on: any reproducible structure,
    /// artifact included, scores positive.
    pub benefit_cos: f64,
    /// `−cos(∇ℓ(C_eval), g)` — does the pull oppose what the reference wants?
    pub forgetting_cos: f64,
    /// `‖g_k‖` per topic — which topics this batch pulls on, beyond ordinary data.
    pub pull_norm_per_topic: Vec<f64>,
    /// Largest relative gap between a differenced effect and its autograd counterpart.
    /// Read it through `fd_warning`, which owns the tolerance.
    fd_check: f64,
    pub n_fit: usize,
    pub n_eval_query: usize,
    pub n_eval_calib: usize,
}

impl Counterfactual {
    /// `Some(message)` when the central-difference step is mis-scaled for this model.
    ///
    /// Only the *intervals* are affected: the effects themselves come from autograd, so a
    /// warning here never impugns them. Owned by this module because it is a judgement
    /// about `FD_EPS_REL`, which the caller cannot see.
    pub fn fd_warning(&self) -> Option<String> {
        (self.fd_check > FD_CHECK_TOL).then(|| {
            format!(
                "counterfactual: differenced effects disagree with autograd by {:.1}% — the \
                 central-difference step is mis-scaled for this model, so the intervals are \
                 unreliable (the effects themselves are exact and unaffected)",
                100.0 * self.fd_check
            )
        })
    }
}

fn mean(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64
}

/// Variance of the sample mean: `Var(x) / n`, one infinitesimal-jackknife component.
fn var_of_mean(x: &[f32]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(x);
    let ss: f64 = x.iter().map(|&v| (f64::from(v) - m).powi(2)).sum();
    ss / ((n - 1) * n) as f64
}

/// Two-sided 99% normal critical value. Exposed so the reporter and the struct agree on
/// what an interval built from a stored SE means.
pub(crate) const Z_99: f64 = 2.575_829_303_548_9;

/// Standard error of a sum of independent infinitesimal-jackknife variance components.
fn ij_se(components: [f64; 3]) -> f64 {
    components.iter().sum::<f64>().max(0.0).sqrt()
}

/// Disjoint cell roles. Two former requirements are absent because a gradient contrast
/// does not need them — the shared replay base cancels exactly (a gradient of a mean over
/// a union is the size-weighted average of the group means, so a base common to both arms
/// contributes identically to each), and the control need not be size-matched (it is a
/// sample mean, unbiased at any size). Two consequences:
///
/// - **At first order replay damps, it does not redirect.** It scales the step without
///   changing where it points, so a coreset's *composition* is irrelevant here and only its
///   size matters — a statement about this readout, not about a real multi-step update.
/// - **`c_ctrl` takes half the calibration cells** rather than exactly `|Q_fit|`: strictly
///   more precise, and it removes the "calibration too small for a control arm" failure.
struct Splits {
    /// Supplies the pull direction `∇ℓ(Q_fit)`.
    q_fit: Vec<usize>,
    /// Held out from the direction; reads `benefit`.
    q_eval: Vec<usize>,
    /// The enacted control: what an *ordinary* reference batch pulls toward.
    c_ctrl: Vec<usize>,
    /// Held out from the control; reads `forgetting`.
    c_eval: Vec<usize>,
}

/// Assign the four roles by a **seeded shuffle**, never by file position.
///
/// Contiguous slices would be a real defect, not a stylistic one: single-cell matrices
/// arrive ordered far more often than not — by cell type from an annotated source, by
/// dataset from a concatenation, by barcode grouping. Under file order, `c_ctrl` (which is
/// supposed to be an *ordinary* reference sample, the whole point of the enacted control)
/// could be a single cell type, `c_eval` could be another, and `q_fit` / `q_eval` could
/// differ in composition — so a novel type might land entirely in one half, meaning either
/// the refit never sees it or the evaluation never does. All three break the contrast in
/// ways that depend on nothing but row order.
fn splits(n_calib: usize, n_query: usize, rng: &mut StdRng) -> anyhow::Result<Splits> {
    let mut q: Vec<usize> = (n_calib..n_calib + n_query).collect();
    q.shuffle(rng);
    let (q_fit, q_eval) = q.split_at(q.len() / 2);
    anyhow::ensure!(!q_fit.is_empty(), "counterfactual: query needs ≥ 2 cells");

    let mut c: Vec<usize> = (0..n_calib).collect();
    c.shuffle(rng);
    let (c_ctrl, c_eval) = c.split_at(c.len() / 2);
    anyhow::ensure!(
        !c_ctrl.is_empty() && !c_eval.is_empty(),
        "counterfactual: calibration needs ≥ 2 cells"
    );

    Ok(Splits {
        q_fit: q_fit.to_vec(),
        q_eval: q_eval.to_vec(),
        c_ctrl: c_ctrl.to_vec(),
        c_eval: c_eval.to_vec(),
    })
}

/// Row-wise `‖·‖` for a `[K, H]` tensor — one norm per topic.
fn row_norms(t: &Tensor) -> anyhow::Result<Vec<f64>> {
    let d = t.sqr()?.sum(1)?.sqrt()?.to_vec1::<f32>()?;
    Ok(d.into_iter().map(f64::from).collect())
}

/// `⟨a, b⟩` over all entries of two same-shaped tensors. `Tensor::dot` is 1-D only and
/// every operand here is `[K, H]`.
fn dot(a: &Tensor, b: &Tensor) -> anyhow::Result<f64> {
    Ok(f64::from((a * b)?.sum_all()?.to_scalar::<f32>()?))
}

/// Frobenius norm as a scalar — `Tensor::norm` returns a tensor.
fn tensor_norm(t: &Tensor) -> anyhow::Result<f64> {
    Ok(f64::from(t.norm()?.to_scalar::<f32>()?))
}

/// Fetch the `α` `Var` out of a loaded varmap by name.
fn alpha_var(parameters: &VarMap, alpha_name: &str) -> anyhow::Result<candle_core::Var> {
    parameters
        .data()
        .lock()
        .expect("varmap poisoned")
        .get(alpha_name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("var `{alpha_name}` not found in the model"))
}

pub(crate) struct CfArgs<'a> {
    pub model: &'a str,
    pub metadata: &'a crate::topic::model_metadata::TopicModelMetadata,
    pub dev: &'a Device,
    pub bank: &'a CellBank,
    /// Seeds the role shuffle in `splits`, so a run is reproducible without depending on
    /// the row order of either backend.
    pub seed: u64,
}

/// Two gradients for the contrast direction, two held-out directional reads, and the
/// influence-function variance of each.
///
/// Returns **magnitudes and no verdict.** The label-permutation null and the AdamW refit
/// that used to live here were both removed — see the module docs for the structural
/// reasons, which are worth reading before adding anything back.
pub(crate) fn counterfactual(a: CfArgs<'_>) -> anyhow::Result<Counterfactual> {
    let rebuilt = rebuild_model(a.model, a.metadata, a.dev)?;
    let alpha = alpha_var(&rebuilt.parameters, &rebuilt.alpha_name)?;
    let arm = Arm {
        decoder: &rebuilt.decoder,
        alpha0: detached_copy(alpha.as_tensor())?,
        alpha,
        bank: a.bank,
    };

    let mut split_rng = StdRng::seed_from_u64(a.seed);
    let s = splits(a.bank.n_calib, a.bank.n_query, &mut split_rng)?;

    // The contrast direction: where this batch pulls `α` beyond where ordinary reference
    // data pulls it. Everything downstream is this one vector read two ways.
    let g = (arm.grad(&s.q_fit)? - arm.grad(&s.c_ctrl)?)?;
    let pull_norm = tensor_norm(&g)?;
    anyhow::ensure!(
        pull_norm.is_finite() && pull_norm > 0.0,
        "counterfactual: query and control pull `α` identically (‖g‖ = {pull_norm}); \
         the effects are undefined"
    );

    // Effects by autograd — exact, and free given that the same gradients supply the
    // cosine denominators. Differencing would give the same quantity less precisely.
    let (gq, gc) = (arm.grad(&s.q_eval)?, arm.grad(&s.c_eval)?);
    let (benefit, forgetting) = (dot(&gq, &g)?, -dot(&gc, &g)?);

    // Infinitesimal-jackknife variance: one component per sampled set. Each is the sample
    // variance of a per-cell projection, and each projection is a directional derivative,
    // so all of them come from three walks. Evaluation and direction sets are concatenated
    // per walk — same direction, same perturbation, disjoint rows.
    let eps = FD_EPS_REL * tensor_norm(&arm.alpha0)?.max(f64::MIN_POSITIVE);
    let evals = [&s.q_eval[..], &s.c_eval[..]].concat();
    let dirs = [&s.q_fit[..], &s.c_ctrl[..]].concat();

    let along_g = arm.ddir(&evals, &g, eps)?;
    let (dq, dc) = along_g.split_at(s.q_eval.len());
    let along_gq = arm.ddir(&dirs, &gq, eps)?;
    let along_gc = arm.ddir(&dirs, &gc, eps)?;
    let (bq, bc) = along_gq.split_at(s.q_fit.len());
    let (fq, fc) = along_gc.split_at(s.q_fit.len());

    let benefit_se = ij_se([var_of_mean(dq), var_of_mean(bq), var_of_mean(bc)]);
    let forgetting_se = ij_se([var_of_mean(dc), var_of_mean(fq), var_of_mean(fc)]);

    // The differenced effects must reproduce the autograd ones; a gap means `FD_EPS_REL`
    // is mis-scaled, which impugns the variances above but never the effects themselves.
    let rel = |fd: f64, exact: f64| (fd - exact).abs() / exact.abs().max(1e-12);
    let fd_check = rel(mean(dq), benefit).max(rel(-mean(dc), forgetting));

    // Scale-free companion: the effect divided by both norms. Separates *where* the batch
    // pulls from *how hard*, which the raw effects deliberately conflate.
    let cos = |g_eval: &Tensor, v: f64| -> anyhow::Result<f64> {
        let n = tensor_norm(g_eval)? * pull_norm;
        Ok(if n > 0.0 { v / n } else { 0.0 })
    };

    Ok(Counterfactual {
        benefit,
        forgetting,
        benefit_se,
        forgetting_se,
        pull_norm,
        benefit_cos: cos(&gq, benefit)?,
        forgetting_cos: cos(&gc, forgetting)?,
        pull_norm_per_topic: row_norms(&g)?,
        fd_check,
        n_fit: s.q_fit.len() + s.c_ctrl.len(),
        n_eval_query: s.q_eval.len(),
        n_eval_calib: s.c_eval.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn split_with_seed(n_calib: usize, n_query: usize, seed: u64) -> anyhow::Result<Splits> {
        splits(n_calib, n_query, &mut StdRng::seed_from_u64(seed))
    }

    #[test]
    fn splits_are_disjoint_and_cover_every_cell() {
        let (n_calib, n_query) = (300usize, 40usize);
        let s = split_with_seed(n_calib, n_query, 42).expect("splits");

        let all: Vec<usize> = [&s.q_fit, &s.q_eval, &s.c_ctrl, &s.c_eval]
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let uniq: HashSet<usize> = all.iter().copied().collect();
        assert_eq!(all.len(), uniq.len(), "cell roles must be disjoint");
        assert_eq!(all.len(), n_calib + n_query, "every cell has a role");

        // The direction-defining sets must not overlap the sets they are read on, or
        // both effects would carry in-sample optimism.
        assert!(s.q_fit.iter().all(|&i| i >= n_calib));
        assert!(s.q_eval.iter().all(|&i| i >= n_calib));
        assert!(s.c_ctrl.iter().all(|&i| i < n_calib));
        assert!(s.c_eval.iter().all(|&i| i < n_calib));
    }

    /// The gradient contrast needs no size matching — `∇ℓ(C_ctrl)` is a sample mean,
    /// unbiased at any size — so a calibration set that the old refit rejected for being
    /// "too small for a control arm" is now perfectly usable.
    #[test]
    fn splits_accept_a_calibration_smaller_than_the_query() {
        let s = split_with_seed(12, 20, 42).expect("splits");
        assert_eq!(s.c_ctrl.len() + s.c_eval.len(), 12);
        assert!(s.c_ctrl.len() < s.q_fit.len());
    }

    #[test]
    fn splits_reject_a_calibration_of_one_cell() {
        assert!(split_with_seed(1, 20, 42).is_err());
    }

    /// Roles must come from a shuffle, never from row order. A file ordered by cell type
    /// or by source dataset would otherwise make `c_ctrl` a non-representative control and
    /// `c_eval` a non-representative reference — silently, and differently per input.
    #[test]
    fn splits_do_not_follow_file_order() {
        let (n_calib, n_query) = (300usize, 40usize);
        let s = split_with_seed(n_calib, n_query, 42).expect("splits");

        let contiguous_prefix: Vec<usize> = (0..s.c_eval.len()).collect();
        assert_ne!(
            s.c_eval, contiguous_prefix,
            "c_eval must not be the leading block of the calibration file"
        );
        let query_prefix: Vec<usize> = (n_calib..n_calib + s.q_fit.len()).collect();
        assert_ne!(
            s.q_fit, query_prefix,
            "q_fit must not be the leading block of the query file"
        );

        // …and the assignment must actually depend on the seed, or it is not randomized.
        let other = split_with_seed(n_calib, n_query, 7).expect("splits");
        let a: HashSet<usize> = s.c_eval.iter().copied().collect();
        let b: HashSet<usize> = other.c_eval.iter().copied().collect();
        assert_ne!(a, b, "role assignment must depend on the seed");
    }
}
