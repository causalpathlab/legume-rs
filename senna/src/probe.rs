//! `senna probe` — read-only drift probe for a trained masked model.
//!
//! Scores query cells' per-cell predictive fit under a trained masked model
//! (`masked-topic` / `masked-vae` / `masked-sbp`), calibrates a null from an
//! in-distribution `--calibration` backend, flags query cells whose fit falls
//! below the null tail, and emits a batch-level **covered vs novel** verdict.
//!
//! The default verdict rests on a **goodness-of-fit / reconstruction residual**: the
//! predictive log-likelihood of each cell under the *frozen* model — the potential
//! outcome `Y(0)` ("does the current model explain this cell"). That is a novelty
//! **proxy**, not the counterfactual `benefit = E[Y(1) − Y(0)]` ("would updating
//! help"). Scoring reconstruction error against an in-distribution null to call
//! unseen cells is the CAMLU strategy (Li et al. 2022); the query-to-reference
//! setting is that of scArches (Lotfollahi et al. 2021) and the open-world,
//! uncertainty-aware scPoli (De Donno et al. 2023).
//!
//! `--counterfactual N` estimates the two treatment effects instead: refit the
//! dictionary by SGD against an *enacted* control arm and measure the result on
//! held-out cells, calibrated by an `N`-permutation null (see `counterfactual`).
//! It reports **benefit** (fit gained on the query) and **forgetting** (fit lost on the
//! reference), and answers a question the fit score structurally cannot — what updating
//! would *cost* — at the price of a refit per permutation. Together they place the batch
//! on the efficacy-toxicity plane: certify / absorb / expand / refuse.
//!
//! # References
//! - Li et al. (2022) *A machine learning-based method for automatically identifying
//!   novel cells in annotating single-cell RNA-seq data* (CAMLU). Bioinformatics 38:4885.
//! - Lotfollahi et al. (2021) *Mapping single-cell data to reference atlases by transfer
//!   learning* (scArches). Nat. Biotechnol. 39:1436.
//! - De Donno et al. (2023) *Population-level integration of single-cell datasets enables
//!   multi-scale analysis across samples* (scPoli). Nat. Methods 20:1683.

use crate::counterfactual::{counterfactual, CellBank, CfArgs, Counterfactual, RefitCfg};
use crate::embed_common::*;
use crate::predict::MaskedScored;
use crate::topic::masked_artifact::MaskedModel;
use log::info;
use std::f64::consts::SQRT_2;

#[derive(Args, Debug)]
pub struct ProbeArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Query data files to probe (.zarr or .h5)"
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        help = "Trained masked model prefix (output of `senna masked-topic/-vae/-sbp` -o)"
    )]
    model: Box<str>,

    #[arg(
        long,
        required = true,
        help = "In-distribution calibration backend that defines the null",
        long_help = "A backend of cells the model already explains.\n\
                     Held-out training-distribution cells are the usual choice.\n\
                     Its per-cell fit distribution sets the null;\n\
                     the query is flagged relative to its lower tail."
    )]
    calibration: Box<str>,

    #[arg(short, long, required = true, help = "Output file prefix")]
    out: Box<str>,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Null tail probability = per-cell false-positive rate"
    )]
    alpha: f64,

    #[arg(long, default_value_t = 500, help = "Evaluation minibatch size")]
    minibatch_size: usize,

    #[arg(
        long,
        help = "Load all columns into memory before scoring",
        hide = true
    )]
    preload_data: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Estimate benefit / forgetting by SGD refit;\n\
                value = #permutations (0 = off)",
        long_help = "The fit score above is the potential outcome Y(0).\n\
                     This instead estimates the effect of updating:\n\
                     refit the topic embeddings α, with the encoder frozen,\n\
                     and measure the result on held-out cells.\n\
                     \n\
                     Treatment refits α on (reference base + query);\n\
                     control refits on (reference base + an equally-sized reference batch),\n\
                     so the effect is that of adding *this* batch rather than ordinary data.\n\
                     `benefit` is the fit gained on held-out query cells;\n\
                     `forgetting` is the fit lost on held-out reference cells.\n\
                     Both are signed so larger is more extreme.\n\
                     Permute the treatment/control label of the pooled fit cells.\n\
                     That gives an exact finite-sample null.\n\
                     No χ², no Fisher, no EIF.\n\
                     \n\
                     Cost is 2 refits per permutation, and p bottoms out at 1/(N+1).\n\
                     Reaches `forgetting`, which the fit score cannot:\n\
                     an in-distribution but contaminated batch reconstructs well,\n\
                     and still degrades the dictionary."
    )]
    counterfactual: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "SGD steps per refit (--counterfactual)"
    )]
    cf_steps: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "AdamW learning rate for --counterfactual refits"
    )]
    cf_lr: f64,

    #[arg(
        long,
        default_value_t = 42,
        help = "Permutation seed for --counterfactual"
    )]
    cf_seed: u64,
}

/// Significance level for the two permutation tests. Distinct from `--alpha`, which is
/// the *per-cell* false-positive rate of the fit score.
const CF_ALPHA: f64 = 0.05;

/// Read the counterfactual pair as one of the four efficacy-toxicity quadrants
/// (phase-I/II dose-finding). Significance comes from the permutation p-values, so this
/// is calibrated rather than a sign heuristic.
///
/// `forgetting > 0` on its own is never a refusal. A batch carrying a topic the model
/// lacks *must* distort the existing topics to make room, so genuine novelty and genuine
/// contamination both forget; they separate on the benefit axis. Refusal is the one
/// quadrant where the model gains nothing and pays anyway.
fn cf_reading(r: &Counterfactual, alpha: f64) -> &'static str {
    match (r.p_benefit < alpha, r.p_forgetting < alpha) {
        (false, false) => "covered + safe — certify; the model already explains this batch",
        (true, false) => "new + safe — absorb; pure benefit",
        (true, true) => "new + risky — expand capacity; the batch carries a topic the model lacks",
        (false, true) => "covered + risky — REFUSE; damage without benefit",
    }
}

/// Standard normal upper tail `P(Z > z)` via the Abramowitz-Stegun erf.
fn norm_sf(z: f64) -> f64 {
    fn erf(x: f64) -> f64 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        sign * y
    }
    0.5 * (1.0 - erf(z / SQRT_2))
}

/// Lower-tail quantile of a slice (nearest-rank).
fn quantile(xs: &[f32], q: f64) -> f32 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((q * (v.len().max(1) - 1) as f64).round() as usize).min(v.len().saturating_sub(1));
    v.get(idx).copied().unwrap_or(f32::NEG_INFINITY)
}

/// Estimate the benefit/forgetting axes for the query batch and return the JSON fragment
/// to splice into the probe report (leading comma included). Enacts the control arm, logs
/// the reading, and warns when the permutation floor cannot reach `CF_ALPHA`.
fn counterfactual_json(
    args: &ProbeArgs,
    model: &MaskedModel<'_>,
    cal: &MaskedScored,
    query: &MaskedScored,
) -> anyhow::Result<String> {
    let dev = candle_core::Device::Cpu;
    let bank = CellBank::from_scored(model, cal, query, &dev)?;

    let r = counterfactual(CfArgs {
        model: model.prefix,
        metadata: &model.metadata,
        dev: &dev,
        bank: &bank,
        cfg: &RefitCfg {
            steps: args.cf_steps,
            lr: args.cf_lr,
        },
        n_perm: args.counterfactual,
        seed: args.cf_seed,
    })?;

    info!(
        "counterfactual: refit α on {} cells ({} steps, lr {}); eval {} query / {} calib; \
         {} permutations",
        r.n_fit, args.cf_steps, args.cf_lr, r.n_eval_query, r.n_eval_calib, r.n_perm
    );
    info!(
        "counterfactual: benefit={:+.4e} (95% CI [{:+.3e}, {:+.3e}], perm p={:.4})   \
         forgetting={:+.4e} (95% CI [{:+.3e}, {:+.3e}], perm p={:.4})",
        r.benefit,
        r.benefit_ci.0,
        r.benefit_ci.1,
        r.p_benefit,
        r.forgetting,
        r.forgetting_ci.0,
        r.forgetting_ci.1,
        r.p_forgetting
    );
    let perm_floor = 1.0 / (r.n_perm + 1) as f64;
    if perm_floor >= CF_ALPHA {
        log::warn!(
            "--counterfactual {} cannot reach p < {CF_ALPHA}: the permutation floor is \
             1/(N+1) = {:.3}. Raise N to at least {}.",
            r.n_perm,
            perm_floor,
            (1.0 / CF_ALPHA).ceil() as usize
        );
    }
    info!("counterfactual: {}", cf_reading(&r, CF_ALPHA));
    let per_topic: Vec<String> = r
        .delta_norm_per_topic
        .iter()
        .map(|v| format!("{v:.3e}"))
        .collect();
    info!("per-topic ||α₁_k − α₀_k||: [{}]", per_topic.join(", "));

    Ok(format!(
        ",\"cf_perms\":{},\"benefit\":{:.6e},\"p_benefit\":{:.4},\
         \"benefit_ci_lo\":{:.6e},\"benefit_ci_hi\":{:.6e},\
         \"forgetting\":{:.6e},\"p_forgetting\":{:.4},\
         \"forgetting_ci_lo\":{:.6e},\"forgetting_ci_hi\":{:.6e},\
         \"cf_steps\":{},\"cf_lr\":{},\"delta_norm_per_topic\":[{}]",
        r.n_perm,
        r.benefit,
        r.p_benefit,
        r.benefit_ci.0,
        r.benefit_ci.1,
        r.forgetting,
        r.p_forgetting,
        r.forgetting_ci.0,
        r.forgetting_ci.1,
        args.cf_steps,
        args.cf_lr,
        r.delta_norm_per_topic
            .iter()
            .map(|v| format!("{v:.6e}"))
            .collect::<Vec<_>>()
            .join(",")
    ))
}

pub fn run_probe(args: &ProbeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let model = MaskedModel::open(&args.model)?;
    let scored =
        |files: &[Box<str>]| model.score(files, args.preload_data, args.minibatch_size, true);

    // Per-cell fit = predictive log-likelihood / count (depth-invariant).
    fn per_cell_fit(s: &MaskedScored) -> Vec<f32> {
        s.llik
            .iter()
            .zip(&s.total)
            .map(|(&l, &t)| if t > 0.0 { l / t } else { 0.0 })
            .collect()
    }

    let cal = scored(std::slice::from_ref(&args.calibration))?;
    let thr = quantile(&per_cell_fit(&cal), args.alpha);

    let query = scored(&args.data_files)?;
    let q_fit = per_cell_fit(&query);
    let q_names = query.data_vec.column_names()?;
    let n = q_fit.len();
    anyhow::ensure!(n > 0, "probe: query has no cells");
    let n_flag = q_fit.iter().filter(|&&f| f < thr).count();
    let rate = n_flag as f64 / n as f64;

    // One-sided test: are more query cells flagged than the null FPR α?
    let se = (args.alpha * (1.0 - args.alpha) / n as f64)
        .sqrt()
        .max(1e-12);
    let pval = norm_sf((rate - args.alpha) / se);
    let novel = rate > args.alpha && pval < 0.05;
    let verdict = if novel {
        "NOVEL — update warranted"
    } else {
        "COVERED — certify"
    };

    // Counterfactual axes: enact the control arm, calibrate with a permutation null.
    let cf_json = if args.counterfactual > 0 {
        counterfactual_json(args, &model, &cal, &query)?
    } else {
        String::new()
    };

    let mut tsv = String::from("cell\tfit\tflag\n");
    for (nm, &f) in q_names.iter().zip(&q_fit) {
        tsv.push_str(&format!("{nm}\t{f:.6}\t{}\n", u8::from(f < thr)));
    }
    std::fs::write(format!("{}.probe.tsv", args.out), tsv)?;
    std::fs::write(
        format!("{}.probe.json", args.out),
        format!(
            "{{\"n_query\":{n},\"alpha\":{},\"threshold\":{thr:.6},\"n_flagged\":{n_flag},\
             \"novelty_rate\":{rate:.4},\"p_value\":{pval:.3e},\"verdict\":\"{verdict}\"\
             {cf_json}}}\n",
            args.alpha
        ),
    )?;

    info!(
        "probe [{}]: null p{:.0} fit ≤ {:.4}; flagged {}/{} query cells ({:.1}%), p = {:.2e}",
        model.metadata.model_type,
        args.alpha * 100.0,
        thr,
        n_flag,
        n,
        rate * 100.0,
        pval
    );
    info!("probe verdict: {verdict}");
    info!("Wrote {}.probe.{{tsv,json}}", args.out);
    Ok(())
}
