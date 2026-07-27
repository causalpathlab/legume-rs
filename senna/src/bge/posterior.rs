//! `--posterior` / `--mcmc` — the exact MCMC posterior over the fitted gate,
//! run after the SGD MAP.
//!
//! The trained model selects features with a per-gene softmax gate: a SuSiE
//! single-effect (categorical selection over the `H` embedding dims × a Gaussian
//! effect) fit **variationally** during training, and read out as the point
//! estimate `JointEmbedModel::feature_selection()`. This step replaces that point
//! estimate with the **exact posterior** of the same model —
//! `graph_embedding_util::posterior` samples each gene's gate against the frozen
//! MAP cell side, giving per-dim PIP, a credible set over the dims a gene loads,
//! and the effect posterior.
//!
//! Two products, selected by [`PosteriorMode`]:
//!
//! * **gate** — the per-gene selection posterior (the headline: calibrated
//!   selection where the variational gate gives a confident point).
//! * **hyper** — the Tier-1 hierarchical hyperparameters the gate hard-codes:
//!   the slab variance `σ₀²` (`GATE_EFFECT_PRIOR_VAR = 1.0`) and the sparsity
//!   `π₀` (`GATE_NULL_PRIOR = 0.9`), sampled from the data instead of guessed.
//!
//! `both` is more than their union: the hypers run **first**, and the learned
//! `σ₀²` is fed into the gate's slab prior — the actual hierarchical payoff, and
//! the reason to prefer it. That adoption is guarded (see [`SIGMA2_MIN_ESS`]).
//!
//! bge is the plainest home for this: `feat_factor: None`, so a feature row *is*
//! an anchor and the sampler runs with no β-sharing or velocity gate to confound
//! it. (`faba gem` adds both.)

use crate::embed_common::*;
use anyhow::Context;
use ge::posterior::{build_gene_index, gate_posterior, hyper_ss, GateConfig, HyperSsConfig};
use graph_embedding_util as ge;

use super::BgeArgs;

/// Default retained draws per chain when `--posterior` is given without `--mcmc`.
pub const DEFAULT_SAMPLES: usize = 200;

/// Default frozen negative-slate size — the number of cells summed in the Poisson
/// rate normalizer. It is the Monte-Carlo resolution of that normalizer, not a
/// subsample of the data: every observed count still enters exactly.
pub const DEFAULT_PARTITION: usize = 1024;

/// Minimum `σ₀²`-chain ESS before the learned slab variance is allowed to
/// override the model's `GATE_EFFECT_PRIOR_VAR`.
///
/// The variance and the effects it scales are strongly dependent (Neal's funnel),
/// and the centered draw used here mixes badly at large `σ₀²` — a measured ESS of
/// 3 on multi-donor HCA BM. A funnel-degenerate scale is worse than the fixed
/// prior, so a chain below this bar is reported and discarded rather than used.
const SIGMA2_MIN_ESS: f32 = 10.0;

/// Which posterior to run after the MAP fit.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum PosteriorMode {
    /// No posterior — the run is byte-identical to one built before this existed.
    Off,
    /// Per-gene gate posterior only (PIP + credible sets + effect posterior).
    Gate,
    /// Tier-1 hyperparameters only (`σ₀²`, `π₀`, per-gene inclusion probability).
    Hyper,
    /// Both, with the learned `σ₀²` feeding the gate's slab prior.
    Both,
}

impl PosteriorMode {
    fn runs_gate(self) -> bool {
        matches!(self, Self::Gate | Self::Both)
    }
    fn runs_hyper(self) -> bool {
        matches!(self, Self::Hyper | Self::Both)
    }
}

/// The resolved posterior request — what `--posterior` and `--mcmc` agree on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PosteriorPlan {
    pub mode: PosteriorMode,
    /// Retained draws per chain; warmup is `n_samples / 2` on top.
    pub n_samples: usize,
    /// Anchors to sample: `0` = every feature, else the top-N by observed count.
    pub n_genes: usize,
    /// Frozen negative-slate size, clamped to `n_cells` at build time.
    pub n_partition: usize,
}

/// Reconcile `--posterior` (which posterior) with `--mcmc` / `--jitter` (how long).
///
/// `--mcmc N` alone is the headline spelling — it means "run the sampler for `N`
/// draws", so it turns the posterior on at [`PosteriorMode::Both`], the full-Bayes
/// answer. `--posterior` narrows that when less is wanted. Asking for both `off`
/// and a draw count is a genuine contradiction, not something to resolve silently,
/// so it errors — the same treatment `GemArgs::genes` gives its own flag clash.
pub fn resolve(args: &BgeArgs) -> anyhow::Result<PosteriorPlan> {
    let mode = match (args.posterior, args.mcmc) {
        (Some(PosteriorMode::Off), Some(n)) => anyhow::bail!(
            "--posterior off contradicts --mcmc {n} (one turns the posterior off, \
             the other on) — pass one or the other"
        ),
        (Some(mode), _) => mode,
        (None, Some(_)) => PosteriorMode::Both,
        (None, None) => PosteriorMode::Off,
    };
    let n_samples = args.mcmc.unwrap_or(DEFAULT_SAMPLES);
    anyhow::ensure!(
        mode == PosteriorMode::Off || n_samples > 0,
        "--mcmc must be > 0 (got {n_samples})"
    );
    Ok(PosteriorPlan {
        mode,
        n_samples,
        n_genes: args.posterior_genes,
        n_partition: args.posterior_partition,
    })
}

/// Sample the posterior and write its tables. Called only for a complete
/// (non-interrupted) fit, after every normal output is on disk.
pub fn run_posterior(
    args: &BgeArgs,
    plan: PosteriorPlan,
    unified: &ge::data::UnifiedData,
    model: &ge::JointEmbedModel,
) -> anyhow::Result<()> {
    let cpu = candle_core::Device::Cpu;
    let to_vec = |t: &candle_core::Tensor| -> anyhow::Result<Vec<f32>> {
        Ok(t.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?)
    };
    let h = model.e_cell.dim(1)?;
    let (e_cell, b_cell, b_feat) = (
        to_vec(&model.e_cell)?,
        to_vec(&model.b_cell)?,
        to_vec(&model.b_feat)?,
    );

    // The frozen contrastive index: observed (gene, cell) counts + one negative
    // slate drawn ONCE and shared across genes. The slate must not move between
    // sweeps or the chain has no fixed target, so it is built here, outside every
    // sampler call.
    let n_partition = plan.n_partition.min(unified.n_cells());
    let mut idx = build_gene_index(
        unified,
        &e_cell,
        &b_cell,
        &b_feat,
        h,
        n_partition,
        POST_SEED,
    )?;

    // Re-fit the per-feature intercepts to the Poisson rate. The model that
    // produced `b_feat` was trained by NCE, not by Poisson likelihood, so its
    // intercepts are not log-rates; left as-is they make the rate term
    // negligible and the samplers collapse every gene onto the count-weighted
    // mean cell direction. See `ContrastiveIndex::calibrate_anchor_bias`.
    let cal = idx.calibrate_anchor_bias();
    info!(
        "posterior: recalibrated feature intercepts to the Poisson rate \
         (median |Δb| = {:.2} nats, max {:.2}, {} feature(s) with no counts)",
        cal.median_abs_shift, cal.max_abs_shift, cal.n_empty
    );

    // Anchors: all features, or the top-N carrying the most observed signal. A
    // gene with no counts has nothing for the likelihood to move, so ranking by
    // total count puts the budget where the posterior is actually informative.
    let picked = pick_anchors(&idx, plan.n_genes);
    let all_nodes = idx.node_terms();
    let nodes: Vec<_> = picked.iter().map(|&g| all_nodes[g]).collect();
    let biases: Vec<f32> = picked.iter().map(|&g| idx.anchor_b[g]).collect();
    let side = idx.frozen_side();
    let names: Vec<Box<str>> = picked
        .iter()
        .map(|&g| unified.feature_names[g].clone())
        .collect();
    info!(
        "posterior ({:?}): {} of {} features, {} draws/chain (+{} warmup), \
         {n_partition}-cell partition — Ctrl+C returns partial results",
        plan.mode,
        picked.len(),
        unified.n_features(),
        plan.n_samples,
        plan.n_samples / 2,
    );

    // ---- Tier-1 hypers first: σ₀² feeds the gate's slab prior below ----
    let hyper = plan.mode.runs_hyper().then(|| {
        let t = std::time::Instant::now();
        let res = hyper_ss(
            &nodes,
            &biases,
            &side,
            &HyperSsConfig::new(plan.n_samples, plan.n_samples / 2, POST_SEED ^ 0x1),
        );
        info!(
            "  hypers: σ₀² = {:.4} (ESS {:.0}), π₀ = {:.4} (ESS {:.0}) in {:.1}s",
            res.sigma2_mean,
            res.sigma_diag.min_ess,
            res.pi0_mean,
            res.pi0_diag.min_ess,
            t.elapsed().as_secs_f32()
        );
        res
    });

    // Adopt the learned slab variance only from a chain that mixed. The funnel
    // makes σ₀²'s MAGNITUDE the unreliable part, so a stuck chain would hand the
    // gate a worse prior than the fixed one it replaces.
    //
    // The warning is phrased by what is actually at stake in this mode. Without a
    // gate to feed there is no "prior" to have refused, and saying so anyway sends
    // the reader looking for a gate that never ran — but the magnitude is still
    // untrustworthy and must be said.
    let learned_sigma2 = hyper.as_ref().and_then(|res| {
        let ok = res.sigma_diag.min_ess >= SIGMA2_MIN_ESS
            && res.sigma2_mean.is_finite()
            && res.sigma2_mean > 0.0;
        if !ok {
            let consequence = if plan.mode.runs_gate() {
                "not adopted for the gate prior — keeping the model's fixed slab variance"
            } else {
                "report it as a scale, not a value"
            };
            log::warn!(
                "  σ₀² = {:.4} is not trustworthy (ESS {:.0} < {SIGMA2_MIN_ESS}, Neal's \
                 funnel): {consequence}",
                res.sigma2_mean,
                res.sigma_diag.min_ess,
            );
        }
        ok.then_some(res.sigma2_mean as f32)
    });

    // ---- the gate posterior ----
    let gate = plan.mode.runs_gate().then(|| {
        let mut cfg = GateConfig::new(plan.n_samples, plan.n_samples / 2, POST_SEED ^ 0x2);
        if let Some(s2) = learned_sigma2 {
            info!("  gate slab variance σ₀² ← {s2:.4} (learned, replacing the fixed prior)");
            cfg.effect_var = s2;
        }
        let t = std::time::Instant::now();
        let res = gate_posterior(&nodes, &biases, &side, &cfg);
        let done = res.iter().filter(|g| g.is_sampled()).count();
        info!(
            "  gate: {done} of {} genes sampled in {:.1}s",
            res.len(),
            t.elapsed().as_secs_f32()
        );
        if done < res.len() {
            log::warn!(
                "  interrupted — {} gene(s) were not sampled and are written as NaN",
                res.len() - done
            );
        }
        res
    });

    // ---- outputs ----
    if let Some(gate) = &gate {
        write_pip(&args.out, gate, &names, h)?;
        write_summary(&args.out, gate, hyper.as_ref(), &picked, &names)?;
    }
    write_hyper_json(
        &args.out,
        &plan,
        hyper.as_ref(),
        learned_sigma2,
        picked.len(),
    )?;
    Ok(())
}

/// Base seed for the posterior chains. bge pins its own sampling RNG to a
/// constant (`FitConfig::seed = 1`, no `--seed` knob), so the posterior follows
/// suit and stays reproducible; the two samplers get distinct streams by XOR.
const POST_SEED: u64 = 1;

/// Feature indices to sample: all of them, or the `n` with the most observed
/// count mass.
fn pick_anchors(idx: &ge::posterior::ContrastiveIndex, n: usize) -> Vec<usize> {
    let n_features = idx.n_anchors();
    if n == 0 || n >= n_features {
        return (0..n_features).collect();
    }
    let total: Vec<f32> = idx
        .pos
        .iter()
        .map(|p| p.iter().map(|&(_, c)| c).sum())
        .collect();
    let mut order: Vec<usize> = (0..n_features).collect();
    order.sort_unstable_by(|&a, &b| {
        total[b]
            .partial_cmp(&total[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(n);
    order.sort_unstable(); // keep the table in feature order
    order
}

/// `{out}.feature_pip.parquet` — `[D_sel × H]` posterior inclusion probabilities,
/// the calibrated analogue of the variational `feature_selection()` and the table
/// a selection-aware consumer reads. A gene skipped by SIGINT is a NaN row.
fn write_pip(
    out: &str,
    gate: &[ge::posterior::GenePosterior],
    names: &[Box<str>],
    h: usize,
) -> anyhow::Result<()> {
    let mut flat = Vec::with_capacity(gate.len() * h);
    for g in gate {
        match g.is_sampled() {
            true => flat.extend_from_slice(&g.pip),
            false => flat.extend(std::iter::repeat_n(f32::NAN, h)),
        }
    }
    let t = candle_core::Tensor::from_vec(flat, (gate.len(), h), &candle_core::Device::Cpu)?;
    let path = format!("{out}.feature_pip.parquet");
    ge::save_embedding(&path, &t, names, "feature")?;
    info!("wrote {path} (per-dim posterior inclusion probability)");
    Ok(())
}

/// `{out}.feature_posterior.parquet` — the per-gene summary a human reads:
/// how concentrated the selection is, how wide the credible set had to be, and
/// how large the effect is.
fn write_summary(
    out: &str,
    gate: &[ge::posterior::GenePosterior],
    hyper: Option<&ge::posterior::HyperSsResult>,
    picked: &[usize],
    names: &[Box<str>],
) -> anyhow::Result<()> {
    let cols: Vec<Box<str>> = [
        "max_pip",
        "argmax_dim",
        "cs_size",
        "cs_coverage",
        "beta_norm",
        "beta_sd",
        "inclusion_prob",
    ]
    .iter()
    .map(|s| Box::<str>::from(*s))
    .collect();
    let n_col = cols.len();

    let mut flat = Vec::with_capacity(gate.len() * n_col);
    for (i, g) in gate.iter().enumerate() {
        // `inclusion_prob` is indexed by ANCHOR position, like `gate` itself —
        // `hyper_ss` ran on the same picked subset, not on the full feature axis.
        let incl = hyper.map_or(f32::NAN, |hr| hr.inclusion_prob[i]);
        if !g.is_sampled() {
            flat.extend(std::iter::repeat_n(f32::NAN, n_col - 1));
            flat.push(incl); // the hypers may have finished even if the gate didn't
            continue;
        }
        let (argmax, max_pip) = g.pip.iter().enumerate().fold(
            (0usize, f32::MIN),
            |acc, (k, &p)| {
                if p > acc.1 {
                    (k, p)
                } else {
                    acc
                }
            },
        );
        let l2 = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mean_sd = g.std_beta.iter().sum::<f32>() / g.std_beta.len().max(1) as f32;
        flat.extend_from_slice(&[
            max_pip,
            argmax as f32,
            g.credible_set.len() as f32,
            g.cs_coverage,
            l2(&g.mean_beta),
            mean_sd,
            incl,
        ]);
    }
    debug_assert_eq!(picked.len(), gate.len());
    let t = candle_core::Tensor::from_vec(flat, (gate.len(), n_col), &candle_core::Device::Cpu)?;
    let path = format!("{out}.feature_posterior.parquet");
    t.to_parquet_with_names(&path, (Some(names), Some("feature")), Some(&cols))?;
    info!("wrote {path} (per-gene selection posterior summary)");
    Ok(())
}

/// `{out}.posterior_hyper.json` — the run's global calibration read, plus the
/// diagnostics that say whether to believe it.
///
/// Through `serde_json` rather than a format string: these are floats that can
/// come back non-finite from a stalled chain, and a bare `NaN` is not JSON.
/// serde emits `null`, which parses. (Same reasoning as `lineage_qc.json`.)
fn write_hyper_json(
    out: &str,
    plan: &PosteriorPlan,
    hyper: Option<&ge::posterior::HyperSsResult>,
    adopted: Option<f32>,
    n_genes: usize,
) -> anyhow::Result<()> {
    let interrupted = ge::stop_flag().load(std::sync::atomic::Ordering::Relaxed);
    let json = serde_json::json!({
        "mode": format!("{:?}", plan.mode).to_lowercase(),
        "n_genes": n_genes,
        "n_samples": plan.n_samples,
        "warmup": plan.n_samples / 2,
        "n_partition": plan.n_partition,
        "sigma2_mean": hyper.map(|h| h.sigma2_mean),
        "sigma2_min_ess": hyper.map(|h| h.sigma_diag.min_ess),
        "sigma2_stuck_fraction": hyper.map(|h| h.sigma_diag.stuck_fraction),
        "pi0_mean": hyper.map(|h| h.pi0_mean),
        "pi0_min_ess": hyper.map(|h| h.pi0_diag.min_ess),
        "pi0_stuck_fraction": hyper.map(|h| h.pi0_diag.stuck_fraction),
        // Did the learned σ₀² actually reach the gate, or was it discarded as
        // funnel-degenerate? Without this the gate's prior is unrecoverable.
        "adopted_sigma2": adopted,
        "interrupted": interrupted,
    });
    let path = format!("{out}.posterior_hyper.json");
    std::fs::write(&path, format!("{}\n", serde_json::to_string_pretty(&json)?))
        .with_context(|| format!("writing {path}"))?;
    info!("wrote {path}");
    Ok(())
}

#[cfg(test)]
#[path = "posterior_tests.rs"]
mod posterior_tests;
