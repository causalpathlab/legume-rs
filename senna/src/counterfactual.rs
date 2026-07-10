//! Counterfactual axes **benefit** / **forgetting** for `senna probe --counterfactual`.
//!
//! `probe`'s default score is the potential outcome `Y(0)` — does the frozen model
//! explain these cells. This module estimates the **effect of updating on them**:
//!
//! ```text
//!   benefit    =  E[Y_query(1)     − Y_query(0)]        efficacy  (> 0 ⇒ the batch teaches)
//!   forgetting = −E[Y_reference(1) − Y_reference(0)]    toxicity  (> 0 ⇒ the batch damages)
//! ```
//!
//! One causal effect, two disjoint evaluation sets. `forgetting` is negated so that
//! "larger is worse" on both axes; the price is that the size-weighted net gain
//! **subtracts** — `n_query · benefit − n_reference · forgetting` — so the minus sign
//! lives here and nowhere else.
//!
//! The `forgetting` axis is the reason this module exists. A goodness-of-fit score
//! cannot see it: a contaminated but in-distribution batch reconstructs perfectly and
//! still drags the dictionary off. That is the *covered + risky* quadrant, and only an
//! estimate of the training effect reaches it.
//!
//! **`forgetting > 0` does not mean "refuse".** A batch carrying a topic the model
//! lacks *must* distort the existing ones to make room, so genuine novelty and genuine
//! contamination both forget. They separate on the other axis: high benefit with high
//! forgetting means grow the model; near-zero benefit with high forgetting means refuse.
//!
//! **Intervention (stated, not implied).** With the encoder frozen, cell
//! representations `θ_j` are held fixed and only the dictionary `α` is refit. This is
//! "update the dictionary", not a joint retrain — a real one would also move the
//! encoder, hence `θ`. The permutation null below is exact for *this* intervention.
//! `α` is only `K×H` (tens of numbers), so refitting it directly is cheaper than the
//! influence-function surrogate it replaces (see the design doc for why that was dropped).
//!
//! **The control arm is enacted, not subtracted.** Even a null batch improves fit when
//! you train on it, so the estimand is *"the effect of adding **this** batch rather
//! than an equally-sized in-distribution batch"*:
//!
//! ```text
//!   treatment: refit α on  C_base ∪ Q_fit   -> α₁
//!   control:   refit α on  C_base ∪ C₂      -> α₀      (C₂ = same-sized reference sample)
//!   benefit    =  mean_{Q_eval} [ ℓ(α₁) − ℓ(α₀) ]
//!   forgetting = −mean_{C_eval} [ ℓ(α₁) − ℓ(α₀) ]
//! ```
//!
//! Contrast this with the continual-learning convention, where backward transfer is
//! measured against the *pre-update* model: that baseline confounds the effect of this
//! batch with the effect of spending another gradient budget at all. The matched
//! control also earns the inference — under H₀ the treatment/control assignment among
//! the pooled fit cells carries no information, so **permuting that assignment gives an
//! exact finite-sample null**. No χ², no Fisher, no efficient influence function.
//!
//! Fitting and evaluation use disjoint cells, so neither effect carries in-sample optimism.

use crate::embed_common::*;
use crate::topic::eval::GeneRemap;
use crate::topic::eval_indexed::{csc_to_indexed, PerGeneContext};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarMap};
use candle_util::decoder::{EmbeddedNbTopicDecoder, MaskedNbTarget};
use data_beans::sparse_io_vector::SparseIoVec;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Rebuild the encoder (only to obtain the shared `ρ` handle and to create the
/// `enc.*` vars the safetensors file expects) plus the **finest** decoder, then load
/// the trained weights. Returns the varmap, the decoder, and the name of the `α` var.
pub(crate) fn rebuild_decoder(
    model: &str,
    metadata: &crate::topic::model_metadata::TopicModelMetadata,
    dev: &Device,
) -> anyhow::Result<(VarMap, EmbeddedNbTopicDecoder, String)> {
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
        },
        &parameters,
        vb.pp("enc"),
    )?;

    // Decoders share the encoder's ρ; predict/probe only use the finest level.
    let lvl = metadata.num_levels.saturating_sub(1);
    let decoder = EmbeddedNbTopicDecoder::new(
        metadata.n_topics,
        encoder.feature_embeddings().clone(),
        vb.pp(format!("dec_{lvl}")),
    )?;
    parameters.load(format!("{model}.safetensors"))?;

    Ok((parameters, decoder, format!("dec_{lvl}.topic.embeddings")))
}

/// One side (calibration or query) of the cell bank.
pub(crate) struct BankSource<'a> {
    pub data_vec: &'a SparseIoVec,
    pub z_nk: &'a Mat,
    pub gene_remap: Option<&'a GeneRemap>,
}

pub(crate) struct BankArgs<'a> {
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
    pub fn build(a: BankArgs<'_>) -> anyhow::Result<Self> {
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

pub(crate) struct RefitCfg {
    pub steps: usize,
    pub lr: f64,
}

/// Refit `α` (only) on `fit_ids`, starting from wherever `α` currently sits —
/// the caller resets it between arms.
fn refit_alpha(
    decoder: &EmbeddedNbTopicDecoder,
    alpha: &candle_core::Var,
    bank: &CellBank,
    fit_ids: &[usize],
    cfg: &RefitCfg,
) -> anyhow::Result<()> {
    let (idx, val, lz) = bank.subset(fit_ids)?;
    let mut adam = AdamW::new(
        vec![alpha.clone()],
        candle_nn::ParamsAdamW {
            lr: cfg.lr,
            ..Default::default()
        },
    )?;
    for _ in 0..cfg.steps {
        let (llik, _) = nb_llik(decoder, &idx, &val, &lz)?;
        let loss = llik.mean_all()?.neg()?;
        adam.backward_step(&loss)?;
    }
    Ok(())
}

/// Per-cell mean log-likelihood over that cell's scored genes.
fn score_cells(
    decoder: &EmbeddedNbTopicDecoder,
    bank: &CellBank,
    ids: &[usize],
) -> anyhow::Result<Vec<f32>> {
    let (idx, val, lz) = bank.subset(ids)?;
    // `impute_masked_nb` already sums over the cell's scored genes -> `[N]`.
    let (llik, mask) = nb_llik(decoder, &idx, &val, &lz)?;
    let den = mask.sum(1)?.clamp(1.0f32, 1e30f32)?;
    Ok(llik.div(&den)?.to_vec1::<f32>()?)
}

pub(crate) struct Counterfactual {
    /// Fit the update *gains* on held-out **query** cells. `> 0` ⇒ the batch teaches
    /// the model something. Efficacy.
    pub benefit: f64,
    /// Fit the update *costs* on held-out **reference** cells, sign-flipped so that
    /// larger is worse. `> 0` ⇒ forgetting. Toxicity.
    ///
    /// Both are the same causal effect — refit the dictionary including this batch
    /// rather than an equally sized reference batch — read out on two disjoint
    /// evaluation sets. Because `forgetting` is negated, the size-weighted net gain
    /// **subtracts**: `n_query · benefit − n_reference · forgetting`. Do not add them.
    pub forgetting: f64,
    /// Upper-tail permutation p-values: `P(benefit ≥ observed)` and
    /// `P(forgetting ≥ observed)` under the exchangeable treatment/control labelling.
    pub p_benefit: f64,
    pub p_forgetting: f64,
    /// 95% Bayesian-bootstrap intervals (Rubin 1981; Dirichlet(1,…,1) reweighting of the
    /// per-cell differences) — `(lo, hi)` for each effect. Distribution-free, so no
    /// normality assumption. These condition on the fit split (eval-cell variance only);
    /// for the fit-cell component too, bootstrap the input backends and re-invoke `probe`.
    /// `benefit`/`forgetting` themselves stay the plain sample means.
    pub benefit_ci: (f64, f64),
    pub forgetting_ci: (f64, f64),
    /// `‖α₁_k − α₀_k‖` — how far the query moved topic `k` *beyond* what an equally
    /// sized reference batch moved it. The enacted analog of the influence step's
    /// per-topic `‖Δ_k‖`, and unlike it, the move actually taken.
    pub delta_norm_per_topic: Vec<f64>,
    pub n_perm: usize,
    pub n_fit: usize,
    pub n_eval_query: usize,
    pub n_eval_calib: usize,
}

fn mean(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64
}

/// Replicate count for the Bayesian-bootstrap intervals. Free (reweights an in-memory
/// vector, no refits), so we can afford enough for stable 2.5/97.5 tail quantiles.
const CI_BOOT_REPS: usize = 2000;

/// 95% Bayesian bootstrap interval (Rubin 1981) for `sign · mean(x)`, using Dirichlet(1,…,1)
/// weights (normalized Exp(1) draws). Distribution-free — no CLT/normality assumption, and
/// the interval is asymmetric where the per-cell differences are skewed. Uses its **own**
/// RNG so the permutation stream, and hence the reported p-values, are untouched.
fn bayes_ci(x: &[f32], sign: f64, rng: &mut StdRng) -> (f64, f64) {
    let point = sign * mean(x);
    if x.len() < 2 {
        return (point, point);
    }
    let mut reps = Vec::with_capacity(CI_BOOT_REPS);
    for _ in 0..CI_BOOT_REPS {
        let (mut num, mut den) = (0f64, 0f64);
        for &v in x {
            let w = -rng.random::<f64>().max(f64::MIN_POSITIVE).ln(); // Exp(1) = Gamma(1,1)
            num += w * f64::from(v);
            den += w;
        }
        reps.push(sign * num / den);
    }
    reps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| reps[((p * (reps.len() - 1) as f64).round() as usize).min(reps.len() - 1)];
    (q(0.025), q(0.975))
}

/// Which cells train each arm, and which cells the two arms are compared on.
struct ArmSpec<'a> {
    /// Shared reference cells present in *both* arms.
    base: &'a [usize],
    /// Added in the treatment arm (the query batch, or a permuted stand-in).
    treat_extra: &'a [usize],
    /// Added in the control arm (a same-sized reference batch, or its stand-in).
    ctrl_extra: &'a [usize],
    eval_q: &'a [usize],
    eval_c: &'a [usize],
}

/// Disjoint cell roles. Query splits half to fit, half to evaluate; calibration
/// splits into an evaluation set, the same-sized control batch `c2`, and the shared
/// base both arms train on.
struct Splits {
    q_fit: Vec<usize>,
    q_eval: Vec<usize>,
    c_eval: Vec<usize>,
    c2: Vec<usize>,
    c_base: Vec<usize>,
}

fn splits(n_calib: usize, n_query: usize) -> anyhow::Result<Splits> {
    let q: Vec<usize> = (n_calib..n_calib + n_query).collect();
    let (q_fit, q_eval) = q.split_at(q.len() / 2);
    anyhow::ensure!(!q_fit.is_empty(), "counterfactual: query needs ≥ 2 cells");

    let c: Vec<usize> = (0..n_calib).collect();
    let n_c_eval = (n_calib / 3).max(1);
    let (c_eval, c_rest) = c.split_at(n_c_eval.min(n_calib));
    anyhow::ensure!(
        c_rest.len() > q_fit.len(),
        "counterfactual: calibration too small ({} usable) for a control batch of {} cells",
        c_rest.len(),
        q_fit.len()
    );
    let (c2, c_base) = c_rest.split_at(q_fit.len());

    Ok(Splits {
        q_fit: q_fit.to_vec(),
        q_eval: q_eval.to_vec(),
        c_eval: c_eval.to_vec(),
        c2: c2.to_vec(),
        c_base: c_base.to_vec(),
    })
}

/// Row-wise `‖a − b‖` for two `[K, H]` tensors.
fn row_norms_of_diff(a: &Tensor, b: &Tensor) -> anyhow::Result<Vec<f64>> {
    let d = (a - b)?.sqr()?.sum(1)?.sqrt()?.to_vec1::<f32>()?;
    Ok(d.into_iter().map(f64::from).collect())
}

/// Outcome of one treatment/control pair: the two scalar effects, the raw per-cell
/// difference vectors they average (kept so the observed pair can be bootstrapped), and
/// the per-topic ‖α₁ − α₀‖. `forgetting` is the *negated* reference-cell effect, so both
/// scalar effects are "larger is more extreme".
struct ArmOutcome {
    benefit: f64,
    forgetting: f64,
    dq: Vec<f32>,
    dc: Vec<f32>,
    delta_norm_per_topic: Vec<f64>,
}

/// Run the treatment and control arms from a common starting `α`.
fn two_arms(
    decoder: &EmbeddedNbTopicDecoder,
    alpha: &candle_core::Var,
    alpha0: &Tensor,
    bank: &CellBank,
    spec: &ArmSpec<'_>,
    cfg: &RefitCfg,
) -> anyhow::Result<ArmOutcome> {
    // One arm: restart from the shared `α₀`, refit on `base ∪ extra`, then read both
    // evaluation sets and copy out the resulting `α`. Treatment and control differ only
    // in `extra`, so they share this path.
    let run_arm = |extra: &[usize]| -> anyhow::Result<(Vec<f32>, Vec<f32>, Tensor)> {
        let mut fit = spec.base.to_vec();
        fit.extend_from_slice(extra);
        alpha.set(alpha0)?;
        refit_alpha(decoder, alpha, bank, &fit, cfg)?;
        let q = score_cells(decoder, bank, spec.eval_q)?;
        let c = score_cells(decoder, bank, spec.eval_c)?;
        Ok((q, c, detached_copy(alpha.as_tensor())?))
    };

    let (tq, tc, a_treat) = run_arm(spec.treat_extra)?;
    let (cq, cc, a_ctrl) = run_arm(spec.ctrl_extra)?;

    let dq: Vec<f32> = tq.iter().zip(&cq).map(|(a, b)| a - b).collect();
    let dc: Vec<f32> = tc.iter().zip(&cc).map(|(a, b)| a - b).collect();
    // Negate the reference effect: `forgetting` is a loss, so larger is worse.
    Ok(ArmOutcome {
        benefit: mean(&dq),
        forgetting: -mean(&dc),
        dq,
        dc,
        delta_norm_per_topic: row_norms_of_diff(&a_treat, &a_ctrl)?,
    })
}

/// A self-contained decoder plus its **own** `α` (and `α₀`), so one rayon worker can refit
/// without touching another's parameters. Holds the `VarMap` only to keep `α` alive.
struct Worker {
    _params: VarMap,
    decoder: EmbeddedNbTopicDecoder,
    alpha: candle_core::Var,
    alpha0: Tensor,
}

/// Rebuild an independent decoder from the model file and detach its loaded `α` as `α₀`.
/// Called once for the observed pair and once per rayon worker (via `map_init`).
fn build_worker(
    model: &str,
    metadata: &crate::topic::model_metadata::TopicModelMetadata,
    dev: &Device,
) -> anyhow::Result<Worker> {
    let (params, decoder, alpha_name) = rebuild_decoder(model, metadata, dev)?;
    let alpha = params
        .data()
        .lock()
        .expect("varmap poisoned")
        .get(&alpha_name)
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!("counterfactual: var `{alpha_name}` not found in the model")
        })?;
    let alpha0 = detached_copy(alpha.as_tensor())?;
    Ok(Worker {
        _params: params,
        decoder,
        alpha,
        alpha0,
    })
}

pub(crate) struct CfArgs<'a> {
    pub model: &'a str,
    pub metadata: &'a crate::topic::model_metadata::TopicModelMetadata,
    pub dev: &'a Device,
    pub bank: &'a CellBank,
    pub cfg: &'a RefitCfg,
    pub n_perm: usize,
    pub seed: u64,
}

/// Enacted control arm + label-permutation null.
///
/// The permutations are generated **serially** (same RNG sequence as a plain loop, so the
/// set of shuffles — hence the p-values — is identical), then evaluated **in parallel**:
/// each refit is deterministic and independent, so the only shared state is the read-only
/// `CellBank`. Every rayon worker rebuilds its own decoder/`α`, so parallel refits never
/// collide, and the result is bit-identical to the serial version.
pub(crate) fn counterfactual(a: CfArgs<'_>) -> anyhow::Result<Counterfactual> {
    let (bank, cfg) = (a.bank, a.cfg);
    let main = build_worker(a.model, a.metadata, a.dev)?;

    let s = splits(bank.n_calib, bank.n_query)?;
    let spec = ArmSpec {
        base: &s.c_base,
        treat_extra: &s.q_fit,
        ctrl_extra: &s.c2,
        eval_q: &s.q_eval,
        eval_c: &s.c_eval,
    };
    let obs = two_arms(&main.decoder, &main.alpha, &main.alpha0, bank, &spec, cfg)?;
    let (benefit, forgetting) = (obs.benefit, obs.forgetting);

    // Pre-generate the permutations serially — identical RNG sequence to a plain loop, so
    // the null set and its p-values are unchanged; only the (deterministic) evaluation runs
    // in parallel. Both effects are signed "larger is more extreme", so both tests upper-tail.
    let mut pool: Vec<usize> = s.q_fit.clone();
    pool.extend_from_slice(&s.c2);
    let qn = s.q_fit.len();
    let mut rng = StdRng::seed_from_u64(a.seed);
    let perms: Vec<(Vec<usize>, Vec<usize>)> = (0..a.n_perm)
        .map(|_| {
            pool.shuffle(&mut rng);
            let (g1, g2) = pool.split_at(qn);
            (g1.to_vec(), g2.to_vec())
        })
        .collect();

    // Evaluate in parallel: each worker rebuilds its own decoder/α (once, via `map_init`) so
    // refits never collide. ~1.4× wall-clock here — modest because candle and this loop share
    // rayon's global pool, so the refits and this fan-out contend. It stays worthwhile for a
    // single call; across many concurrent probe calls (a sweep or an external bootstrap),
    // single-core-per-call would compose better.
    let (ge_benefit, ge_forgetting) = perms
        .par_iter()
        .map_init(
            || build_worker(a.model, a.metadata, a.dev),
            |worker, (g1, g2)| -> anyhow::Result<(usize, usize)> {
                let w = worker
                    .as_mut()
                    .map_err(|e| anyhow::anyhow!("counterfactual worker init: {e}"))?;
                let perm = ArmSpec {
                    base: &s.c_base,
                    treat_extra: g1,
                    ctrl_extra: g2,
                    eval_q: &s.q_eval,
                    eval_c: &s.c_eval,
                };
                let o = two_arms(&w.decoder, &w.alpha, &w.alpha0, bank, &perm, cfg)?;
                Ok((
                    usize::from(o.benefit >= benefit),
                    usize::from(o.forgetting >= forgetting),
                ))
            },
        )
        .try_reduce(
            || (0usize, 0usize),
            |(a1, b1), (a2, b2)| Ok((a1 + a2, b1 + b2)),
        )?;

    // Effect-size intervals from the observed pair. A *separate* RNG stream keeps the
    // permutation p-values above bit-identical regardless of the bootstrap.
    let mut boot_rng = StdRng::seed_from_u64(a.seed ^ 0x9E37_79B9_7F4A_7C15);
    let benefit_ci = bayes_ci(&obs.dq, 1.0, &mut boot_rng);
    let forgetting_ci = bayes_ci(&obs.dc, -1.0, &mut boot_rng);

    let denom = (a.n_perm + 1) as f64;
    Ok(Counterfactual {
        benefit,
        forgetting,
        p_benefit: (ge_benefit + 1) as f64 / denom,
        p_forgetting: (ge_forgetting + 1) as f64 / denom,
        benefit_ci,
        forgetting_ci,
        delta_norm_per_topic: obs.delta_norm_per_topic,
        n_perm: a.n_perm,
        n_fit: s.c_base.len() + s.q_fit.len(),
        n_eval_query: s.q_eval.len(),
        n_eval_calib: s.c_eval.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn splits_are_disjoint_and_control_matches_treatment_size() {
        let (n_calib, n_query) = (300usize, 40usize);
        let s = splits(n_calib, n_query).expect("splits");

        // The control batch must be the same size as the treated batch, or the
        // arms differ in gradient budget and τ is not an effect of *this* batch.
        assert_eq!(s.c2.len(), s.q_fit.len());

        let all: Vec<usize> = [&s.q_fit, &s.q_eval, &s.c_eval, &s.c2, &s.c_base]
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let uniq: HashSet<usize> = all.iter().copied().collect();
        assert_eq!(all.len(), uniq.len(), "cell roles must be disjoint");
        assert_eq!(all.len(), n_calib + n_query, "every cell has a role");

        assert!(s.q_fit.iter().all(|&i| i >= n_calib));
        assert!(s.q_eval.iter().all(|&i| i >= n_calib));
        assert!(s.c_eval.iter().all(|&i| i < n_calib));
        assert!(s.c2.iter().all(|&i| i < n_calib));
        assert!(s.c_base.iter().all(|&i| i < n_calib));
    }

    #[test]
    fn splits_reject_a_calibration_too_small_for_a_control_arm() {
        // 20 query -> 10 fit; 12 calib -> 4 eval, 8 rest: 8 is not > 10.
        assert!(splits(12, 20).is_err());
    }
}
