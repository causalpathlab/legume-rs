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

/// Per-cell fit = predictive log-likelihood / count (depth-invariant).
fn per_cell_fit(llik: &[f32], total: &[f32]) -> Vec<f32> {
    llik.iter()
        .zip(total)
        .map(|(&l, &t)| if t > 0.0 { l / t } else { 0.0 })
        .collect()
}

/// Everything the verdict needs, once a family-specific scorer has produced it.
///
/// The verdict itself is model-agnostic — a quantile of the calibration fits, a flag rate,
/// and a one-sided binomial test — so it lives here once rather than in each branch. Only
/// the *scoring* differs by family, and only masked models get `cf_json`.
struct Verdict<'a> {
    args: &'a ProbeArgs,
    model_type: &'a str,
    cal_fit: Vec<f32>,
    q_fit: Vec<f32>,
    q_names: Vec<Box<str>>,
    cf_json: String,
}

pub fn run_probe(args: &ProbeArgs) -> anyhow::Result<()> {
    use crate::topic::model_metadata::masked_head_from_model_type;

    mkdir_parent(&args.out)?;
    // `senna bge` writes no `model.json` — it has no checkpoint at all — so it is identified
    // by the absence of one plus a `RunKind::Bge` manifest, not by a `model_type` string.
    if !std::path::Path::new(&format!("{}.model.json", args.model)).exists() {
        return probe_bge(args);
    }
    let metadata = crate::topic::model_metadata::TopicModelMetadata::load(&args.model)?;
    if masked_head_from_model_type(&metadata.model_type).is_some() {
        probe_masked(args)
    } else {
        probe_fit_only(args, &metadata)
    }
}

/// Fit score for a `senna bge` run.
///
/// No refit is possible here for the same reason as the dense families, only more so: bge's
/// only learnable object on the gene side is ρ itself (`[D,H]`, ~2.5M numbers at D=20k),
/// which a few hundred query cells cannot identify. Chain
/// `bge --skip-etm` → `masked-topic --freeze-feature-embedding` → `probe --counterfactual`
/// when the counterfactual is what you want.
fn probe_bge(args: &ProbeArgs) -> anyhow::Result<()> {
    use crate::bge_artifact::BgeModel;

    anyhow::ensure!(
        args.counterfactual == 0,
        "--counterfactual is available for masked models only; {} is a `senna bge` run, whose \
         gene-side parameter is ρ itself (~2.5M numbers at D=20k) rather than a K×H block. See \
         the `probe_bge` docs for the chaining route.",
        args.model
    );

    let model = BgeModel::open(&args.model)?;
    let cal = model.score(
        std::slice::from_ref(&args.calibration),
        args.preload_data,
        args.minibatch_size,
    )?;
    let query = model.score(&args.data_files, args.preload_data, args.minibatch_size)?;

    write_verdict(Verdict {
        args,
        model_type: "bge",
        cal_fit: per_cell_fit(&cal.llik, &cal.total),
        q_fit: per_cell_fit(&query.llik, &query.total),
        q_names: query.data_vec.column_names()?,
        cf_json: String::new(),
    })
}

fn probe_masked(args: &ProbeArgs) -> anyhow::Result<()> {
    let model = MaskedModel::open(&args.model)?;
    let scored =
        |files: &[Box<str>]| model.score(files, args.preload_data, args.minibatch_size, true);

    let cal = scored(std::slice::from_ref(&args.calibration))?;
    let query = scored(&args.data_files)?;

    // Counterfactual axes: enact the control arm, calibrate with a permutation null.
    let cf_json = if args.counterfactual > 0 {
        counterfactual_json(args, &model, &cal, &query)?
    } else {
        String::new()
    };

    write_verdict(Verdict {
        args,
        model_type: &model.metadata.model_type,
        cal_fit: per_cell_fit(&cal.llik, &cal.total),
        q_fit: per_cell_fit(&query.llik, &query.total),
        q_names: query.data_vec.column_names()?,
        cf_json,
    })
}

/// Fit score only, for the families that have no small identified block to refit.
///
/// **Why no counterfactual here.** The masked refit moves `α [K,H]` — ~640 numbers. The dense
/// families have no such factorization: `topic`'s block is `dictionary.logits [K,D]` (~400k at
/// K=20, D=20k) and `vae`'s is `gauss_decoder.weight [D,K]`. Refitting either from a few hundred
/// query cells is underpowered, and `CellBank`'s indexed top-K packing has no dense equivalent —
/// it needs `enc_context_size` and `shortlist_weights.parquet`, which dense artifacts do not have.
/// For a dense counterfactual today, chain `bge --skip-etm` → `masked-topic
/// --freeze-feature-embedding` → `probe --counterfactual`.
fn probe_fit_only(
    args: &ProbeArgs,
    metadata: &crate::topic::model_metadata::TopicModelMetadata,
) -> anyhow::Result<()> {
    use crate::masked_topic::FeatureNameKindArg;
    use crate::predict::{score_dense_backend, score_vae_backend, DenseScoreArgs, VaeScoreArgs};
    use crate::topic::eval::QueryNameOpts;
    use crate::topic::model_metadata::{MODEL_TYPE_TOPIC, MODEL_TYPE_VAE};
    use crate::topic::predict_common::LatentMode;
    use candle_util::topic_refinement::TopicRefinementConfig;

    anyhow::ensure!(
        args.counterfactual == 0,
        "--counterfactual is available for masked models only ({} is a '{}' model); see the \
         `probe_fit_only` docs for why, and for the chaining route that gets you one.",
        args.model,
        metadata.model_type
    );

    let qopts = QueryNameOpts {
        kind: FeatureNameKindArg::Exact.resolve_or_gene(),
        suffix_delim: None,
        keep_suffix: None,
    };

    let (cal_fit, q_fit, q_names) = match metadata.model_type.as_ref() {
        MODEL_TYPE_VAE => {
            let score = |files: &[Box<str>]| {
                score_vae_backend(VaeScoreArgs {
                    model: &args.model,
                    data_files: files,
                    batch_files: None,
                    preload: args.preload_data,
                    minibatch_size: args.minibatch_size,
                    query_name_opts: &qopts,
                    metadata,
                    need_llik: true,
                })
            };
            let cal = score(std::slice::from_ref(&args.calibration))?;
            let query = score(&args.data_files)?;
            (
                per_cell_fit(&cal.llik, &cal.total),
                per_cell_fit(&query.llik, &query.total),
                query.data_vec.column_names()?,
            )
        }
        MODEL_TYPE_TOPIC => {
            // `LatentMode::Encoder` mirrors the masked path (encoder-only, no per-cell
            // refinement), so `refine_config` is never read.
            let refine = TopicRefinementConfig {
                num_steps: 0,
                learning_rate: 0.0,
                regularization: 0.0,
            };
            // ⚠️ `delta_iters: 0` keeps the legacy single-pass plug-in δ rather than the TMLE
            // refinement `predict` defaults to. δ is estimated **from the query**, so refining it
            // would let a genuinely novel batch explain itself away as a batch effect — the
            // aliasing failure, on exactly the query this tool exists to flag. The plug-in is the
            // conservative end of that trade; the masked path has no δ at all.
            let score = |files: &[Box<str>]| {
                score_dense_backend(DenseScoreArgs {
                    model: &args.model,
                    data_files: files,
                    batch_files: None,
                    preload: args.preload_data,
                    minibatch_size: args.minibatch_size,
                    block_size: None,
                    delta_iters: 0,
                    query_name_opts: &qopts,
                    metadata,
                    mode: LatentMode::Encoder,
                    refine_config: &refine,
                })
            };
            let cal = score(std::slice::from_ref(&args.calibration))?;
            let query = score(&args.data_files)?;
            (
                per_cell_fit(&cal.llik, &cal.total),
                per_cell_fit(&query.llik, &query.total),
                query.data_vec.column_names()?,
            )
        }
        other => anyhow::bail!(
            "probe does not support '{other}' models; masked-topic/-vae/-sbp, topic and vae only"
        ),
    };

    write_verdict(Verdict {
        args,
        model_type: &metadata.model_type,
        cal_fit,
        q_fit,
        q_names,
        cf_json: String::new(),
    })
}

fn write_verdict(v: Verdict<'_>) -> anyhow::Result<()> {
    let Verdict {
        args,
        model_type,
        cal_fit,
        q_fit,
        q_names,
        cf_json,
    } = v;

    let thr = quantile(&cal_fit, args.alpha);
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
        model_type,
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
