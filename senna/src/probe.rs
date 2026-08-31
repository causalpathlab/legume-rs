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
//! `--counterfactual` estimates the two treatment effects instead: build the direction in
//! which this batch pulls `α` *beyond* where ordinary reference data pulls it, and read
//! held-out fit along it (see `counterfactual`). It reports **benefit** (fit gained on the
//! query) and **forgetting** (fit lost on the reference), and answers a question the fit
//! score structurally cannot — what updating would *cost*.
//!
//! **`probe` never writes a model and never trains one.** The whole call is four gradients
//! and four forward passes; `α` is perturbed in memory to read a directional derivative and
//! restored. There is no optimizer and no hyperparameter to match across runs — earlier
//! versions refit `α` with AdamW, which put the step count and learning rate inside the
//! answer.
//!
//! It reports magnitudes and **no verdict**. The efficacy-toxicity quadrant reading was
//! removed together with the permutation p-values it thresholded — see `counterfactual`'s
//! module docs for why that null was unsound, and do not restore a sign-based rule in its
//! place (sign rules on these axes were tested and are κ-fragile).
//!
//! # References
//! - Li et al. (2022) *A machine learning-based method for automatically identifying
//!   novel cells in annotating single-cell RNA-seq data* (CAMLU). Bioinformatics 38:4885.
//! - Lotfollahi et al. (2021) *Mapping single-cell data to reference atlases by transfer
//!   learning* (scArches). Nat. Biotechnol. 39:1436.
//! - De Donno et al. (2023) *Population-level integration of single-cell datasets enables
//!   multi-scale analysis across samples* (scPoli). Nat. Methods 20:1683.

use crate::counterfactual::{counterfactual, CellBank, CfArgs, Z_99};
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
        help = "Estimate first-order benefit / forgetting (no training)",
        long_help = "The fit score above is the potential outcome Y(0).\n\
                     This instead estimates the effect of updating,\n\
                     to first order, with the encoder frozen.\n\
                     \n\
                     Build g = grad(query) - grad(reference control):\n\
                     the direction this batch pulls the dictionary\n\
                     beyond where ordinary reference data pulls it.\n\
                     `benefit` is the fit gained along g on held-out query cells;\n\
                     `forgetting` is the fit lost along g on held-out reference cells.\n\
                     Both are signed so larger is more extreme.\n\
                     \n\
                     Reaches `forgetting`, which the fit score cannot:\n\
                     an in-distribution but contaminated batch reconstructs well,\n\
                     and still degrades the dictionary.\n\
                     \n\
                     No optimizer, so no step count or learning rate\n\
                     enters the answer. Reports magnitudes only —\n\
                     there is no calibrated decision rule on either axis."
    )]
    counterfactual: bool,

    #[arg(
        long,
        default_value_t = 42,
        help = "Seed for the --counterfactual role assignment"
    )]
    cf_seed: u64,
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

/// Estimate and log the benefit/forgetting axes for the query batch, including what the
/// numbers do and do not support.
fn report_counterfactual(
    args: &ProbeArgs,
    model: &MaskedModel<'_>,
    cal: &MaskedScored,
    query: &MaskedScored,
) -> anyhow::Result<()> {
    let dev = candle_core::Device::Cpu;
    let bank = CellBank::from_scored(model, cal, query, &dev)?;

    let r = counterfactual(CfArgs {
        model: model.prefix,
        metadata: &model.metadata,
        dev: &dev,
        bank: &bank,
        seed: args.cf_seed,
    })?;

    let (b, f, bcos, fcos, pull) = (
        r.benefit,
        r.forgetting,
        r.benefit_cos,
        r.forgetting_cos,
        r.pull_norm,
    );
    let (b_se, f_se) = (r.benefit_se, r.forgetting_se);
    let (b_lo, b_hi) = (b - Z_99 * b_se, b + Z_99 * b_se);
    let (f_lo, f_hi) = (f - Z_99 * f_se, f + Z_99 * f_se);

    info!(
        "counterfactual: direction from {} cells; eval {} query / {} calib",
        r.n_fit, r.n_eval_query, r.n_eval_calib
    );
    info!(
        "counterfactual: benefit={b:+.4e} (SE {b_se:.3e}, 99% CI [{b_lo:+.3e}, {b_hi:+.3e}], \
         cos {bcos:+.3})   forgetting={f:+.4e} (SE {f_se:.3e}, 99% CI [{f_lo:+.3e}, \
         {f_hi:+.3e}], cos {fcos:+.3})   ||g||={pull:.4e}"
    );
    info!(
        "counterfactual: first-order magnitudes — there is no calibrated threshold on \
         either axis, and no interval here sees between-dataset variation"
    );
    if let Some(w) = r.fd_warning() {
        log::warn!("{w}");
    }
    let per_topic: Vec<String> = r
        .pull_norm_per_topic
        .iter()
        .map(|v| format!("{v:.3e}"))
        .collect();
    info!("per-topic ||g_k||: [{}]", per_topic.join(", "));

    Ok(())
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
/// the *scoring* differs by family.
struct Verdict<'a> {
    args: &'a ProbeArgs,
    model_type: &'a str,
    cal_fit: Vec<f32>,
    q_fit: Vec<f32>,
    q_names: Vec<Box<str>>,
}

pub fn run_probe(args: &ProbeArgs) -> anyhow::Result<()> {
    use crate::run_manifest::RunKind;
    use crate::topic::model_metadata::resolve_run_kind;

    mkdir_parent(&args.out)?;
    // `resolve_run_kind` says what the run *is*; the match below is probe's own
    // support policy. `update` asks the same question and answers it differently
    // (it supports svd, which has no scorer here), so the two must not share a
    // pre-filtered "family" enum — only the resolution.
    //
    // Head selection *within* the masked family stays with `MaskedModel::open`,
    // which reads `model_type` anyway: `masked-topic` and `masked-sbp` share
    // `RunKind::Itopic` and are told apart only there.
    let kind = resolve_run_kind(&args.model)?;
    match kind {
        k if k.is_masked_family() => probe_masked(args),
        RunKind::Bge => probe_bge(args),
        RunKind::Topic | RunKind::Vae => probe_fit_only(args, kind),
        other => anyhow::bail!(
            "probe does not support a '{}' run. Supported: masked-topic / masked-sbp / \
             masked-vae, topic, vae, bge.",
            other.as_str()
        ),
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
    use crate::bge::score::BgeEmbedding;

    anyhow::ensure!(
        !args.counterfactual,
        "--counterfactual is available for masked models only; {} is a `senna bge` run, whose \
         gene-side parameter is ρ itself (~2.5M numbers at D=20k) rather than a K×H block. See \
         the `probe_bge` docs for the chaining route.",
        args.model
    );

    let model = BgeEmbedding::open(&args.model)?;
    // No coverage floor: a thin panel is exactly what probe exists to score.
    let qopts = crate::topic::eval::QueryNameOpts::default();
    let cal = model.score(
        std::slice::from_ref(&args.calibration),
        args.preload_data,
        args.minibatch_size,
        &qopts,
    )?;
    let query = model.score(
        &args.data_files,
        args.preload_data,
        args.minibatch_size,
        &qopts,
    )?;

    write_verdict(Verdict {
        args,
        model_type: "bge",
        cal_fit: per_cell_fit(&cal.llik, &cal.total),
        q_fit: per_cell_fit(&query.llik, &query.total),
        q_names: query.data_vec.column_names()?,
    })
}

fn probe_masked(args: &ProbeArgs) -> anyhow::Result<()> {
    let model = MaskedModel::open(&args.model)?;
    let scored =
        |files: &[Box<str>]| model.score(files, args.preload_data, args.minibatch_size, true);

    let cal = scored(std::slice::from_ref(&args.calibration))?;
    let query = scored(&args.data_files)?;

    // Counterfactual axes: contrast this batch against the reference control arm.
    if args.counterfactual {
        report_counterfactual(args, &model, &cal, &query)?;
    }

    write_verdict(Verdict {
        args,
        model_type: &model.metadata.model_type,
        cal_fit: per_cell_fit(&cal.llik, &cal.total),
        q_fit: per_cell_fit(&query.llik, &query.total),
        q_names: query.data_vec.column_names()?,
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
/// The three fields the verdict needs, from whichever family produced them.
///
/// `MaskedScored`, `DenseScored`, `VaeScored` and `BgeFit` all carry exactly these; without the
/// trait, "score the calibration set, score the query, reduce both" was written out four times
/// and the definition of *per-cell fit* lived in four places.
trait Scored {
    fn parts(&self) -> (&[f32], &[f32], &data_beans::sparse_io_vector::SparseIoVec);
}

macro_rules! impl_scored {
    ($($t:ty),+) => { $(impl Scored for $t {
        fn parts(&self) -> (&[f32], &[f32], &data_beans::sparse_io_vector::SparseIoVec) {
            (&self.llik, &self.total, &self.data_vec)
        }
    })+ };
}
impl_scored!(
    MaskedScored,
    crate::predict::DenseScored,
    crate::predict::VaeScored,
    crate::bge::score::BgeFit
);

/// `(calibration fits, query fits, query cell names)` — what the verdict consumes.
type FitPair = (Vec<f32>, Vec<f32>, Vec<Box<str>>);

/// Score calibration then query, and reduce both to per-cell fits.
///
/// The calibration score is scoped so its backend — a whole `SparseIoVec`, the full matrix
/// under `--preload-data` — drops *before* the query one is opened. Only its two `f32` vectors
/// are needed afterwards, and `update` already takes this care explicitly.
fn cal_and_query<S: Scored>(
    args: &ProbeArgs,
    score: impl Fn(&[Box<str>]) -> anyhow::Result<S>,
) -> anyhow::Result<FitPair> {
    let cal_fit = {
        let cal = score(std::slice::from_ref(&args.calibration))?;
        let (llik, total, _) = cal.parts();
        per_cell_fit(llik, total)
    };
    let query = score(&args.data_files)?;
    let (llik, total, data_vec) = query.parts();
    Ok((cal_fit, per_cell_fit(llik, total), data_vec.column_names()?))
}

fn probe_fit_only(args: &ProbeArgs, kind: crate::run_manifest::RunKind) -> anyhow::Result<()> {
    use crate::predict::{score_dense_backend, score_vae_backend, DenseScoreArgs, VaeScoreArgs};
    use crate::run_manifest::RunKind;
    use crate::topic::eval::QueryNameOpts;
    use crate::topic::model_metadata::TopicModelMetadata;
    use crate::topic::predict_common::LatentMode;
    use candle_util::topic_refinement::TopicRefinementConfig;

    let metadata = TopicModelMetadata::load(&args.model)?;
    anyhow::ensure!(
        !args.counterfactual,
        "--counterfactual is available for masked models only ({} is a '{}' model); see the \
         `probe_fit_only` docs for why, and for the chaining route that gets you one.",
        args.model,
        metadata.model_type
    );

    let qopts = QueryNameOpts::default();
    let (cal_fit, q_fit, q_names) = match kind {
        RunKind::Vae => cal_and_query(args, |files| {
            score_vae_backend(VaeScoreArgs {
                model: &args.model,
                data_files: files,
                batch_files: None,
                preload: args.preload_data,
                minibatch_size: args.minibatch_size,
                query_name_opts: &qopts,
                metadata: &metadata,
                need_llik: true,
            })
        })?,
        // `LatentMode::Encoder` mirrors the masked path (encoder-only, no per-cell refinement),
        // so `refine_config` is never read.
        //
        // ⚠️ `delta_iters: 0` keeps the legacy single-pass plug-in δ rather than the TMLE
        // refinement `predict` defaults to. δ is estimated **from the query**, so refining it
        // would let a genuinely novel batch explain itself away as a batch effect — the aliasing
        // failure, on exactly the query this tool exists to flag. The plug-in is the conservative
        // end of that trade; the masked path has no δ at all.
        RunKind::Topic => cal_and_query(args, |files| {
            score_dense_backend(DenseScoreArgs {
                model: &args.model,
                data_files: files,
                batch_files: None,
                preload: args.preload_data,
                minibatch_size: args.minibatch_size,
                block_size: None,
                delta_iters: 0,
                query_name_opts: &qopts,
                metadata: &metadata,
                mode: LatentMode::Encoder,
                refine_config: &TopicRefinementConfig::default(),
            })
        })?,
        // `run_probe` routes every other kind elsewhere or rejects it.
        other => unreachable!("{other} has its own path"),
    };

    write_verdict(Verdict {
        args,
        model_type: &metadata.model_type,
        cal_fit,
        q_fit,
        q_names,
    })
}

fn write_verdict(v: Verdict<'_>) -> anyhow::Result<()> {
    let Verdict {
        args,
        model_type,
        cal_fit,
        q_fit,
        q_names,
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
        "NOVEL — the model does not explain this batch"
    } else {
        "COVERED — certify"
    };

    // Per-cell fit and flag — the only output not already in the log, and the only place
    // *which* cells were flagged is recorded. A summary `.probe.json` used to be written
    // alongside; it duplicated the log lines below field for field and nothing read it.
    let mut tsv = String::from("cell\tfit\tflag\n");
    for (nm, &f) in q_names.iter().zip(&q_fit) {
        tsv.push_str(&format!("{nm}\t{f:.6}\t{}\n", u8::from(f < thr)));
    }
    std::fs::write(format!("{}.probe.tsv", args.out), tsv)?;

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
    info!("Wrote {}.probe.tsv", args.out);
    Ok(())
}
