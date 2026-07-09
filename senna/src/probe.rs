//! `senna probe` — read-only drift probe for a trained masked model.
//!
//! Scores query cells' per-cell predictive fit under a trained masked model
//! (`masked-topic` / `masked-vae` / `masked-sbp`), calibrates a null from an
//! in-distribution `--calibration` backend, flags query cells whose fit falls
//! below the null tail, and emits a batch-level **covered vs novel** verdict.
//!
//! What this computes is a **goodness-of-fit / reconstruction residual**: the
//! predictive log-likelihood of each cell under the *frozen* model — the
//! potential outcome `Y(0)` ("does the current model explain this cell"). That
//! is a novelty **proxy**, not the counterfactual `τ_new = E[Y(1) − Y(0)]`
//! ("would updating help"), which needs the gradient `g_new` and the Fisher `H`.
//! The counterfactual axes (`τ_new`, `τ_old`) and the 2×2 verdict are a later
//! rung and are not computed here.

use crate::embed_common::*;
use crate::influence::{counterfactual, topic_gradient_stats};
use crate::masked_topic::FeatureNameKindArg;
use crate::predict::{score_masked_backend, MaskedScoreArgs, MaskedScored};
use crate::topic::eval::QueryNameOpts;
use crate::topic::model_metadata::{
    load_dictionary, load_feature_embedding, masked_head_from_model_type, TopicModelMetadata,
};
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
        long_help = "A backend of cells the model already explains (e.g. held-out\n\
                     training-distribution cells). Its per-cell fit distribution sets the\n\
                     null; the query is flagged relative to its lower tail."
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

    #[arg(long, help = "Load all columns into memory before scoring")]
    preload_data: bool,

    #[arg(
        long,
        help = "Also estimate the counterfactual axes τ_new / τ_old (influence core)",
        long_help = "Differentiate the fit objective w.r.t. the decoder's topic embeddings on\n\
                     both the calibration and query cells, take their empirical Fisher\n\
                     diagonals, and form the EWC-regularized one-step update\n\
                     Δ = ḡ_new / (κ·F_old + F_new). Reports τ_new (predicted fit gain on the\n\
                     query if we updated) and τ_old (predicted change on calibration data;\n\
                     negative means forgetting), plus per-topic ‖Δ_k‖."
    )]
    influence: bool,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "EWC prior strength κ: how hard the old model resists moving",
        long_help = "Weights the old model's Fisher in Δ = ḡ_Δ / (κ·F_cal + F_new). Both Fishers\n\
                     are per-cell, so the principled value is κ ≈ n_train / n_query — the old\n\
                     model should resist in proportion to how much data it was fit on.\n\
                     The default of 1 holds it with the weight of a single cell, which pushes\n\
                     the second-order expansion outside its trust region: use it for ranking\n\
                     only. τ_new and τ_old scale like 1/κ; the sign of τ_old and the ratio\n\
                     τ_new/|τ_old| are the κ-stable readouts."
    )]
    prior_strength: f64,
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

pub fn run_probe(args: &ProbeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let metadata = TopicModelMetadata::load(&args.model)?;
    let head = masked_head_from_model_type(&metadata.model_type).ok_or_else(|| {
        anyhow::anyhow!(
            "probe supports masked models only (masked-topic/-vae/-sbp); got '{}'",
            metadata.model_type
        )
    })?;
    let qopts = QueryNameOpts {
        kind: FeatureNameKindArg::Exact.resolve_or_gene(),
        suffix_delim: None,
        keep_suffix: None,
    };

    let scored = |files: &[Box<str>]| -> anyhow::Result<MaskedScored> {
        score_masked_backend(MaskedScoreArgs {
            model: &args.model,
            data_files: files,
            batch_files: None,
            preload: args.preload_data,
            minibatch_size: args.minibatch_size,
            query_name_opts: &qopts,
            metadata: &metadata,
            head,
        })
    };

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

    // Counterfactual axes: what would a joint retrain actually do?
    let infl_json = if args.influence {
        let (_, beta_dk) = load_dictionary(&args.model)?;
        let (_, rho_dh) = load_feature_embedding(&args.model)?;
        let l2 = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();

        let old = topic_gradient_stats(
            &cal.data_vec,
            &cal.z_nk,
            &beta_dk,
            &rho_dh,
            cal.gene_remap.as_ref(),
            args.minibatch_size,
        )?;
        let new = topic_gradient_stats(
            &query.data_vec,
            &query.z_nk,
            &beta_dk,
            &rho_dh,
            query.gene_remap.as_ref(),
            args.minibatch_size,
        )?;
        let cf = counterfactual(&new, &old, args.prior_strength, beta_dk.ncols());
        let (gn, go) = (l2(&new.g_mean), l2(&old.g_mean));

        info!(
            "influence: n_query={} n_calib={}  ||g_query||={gn:.3e}  ||g_calib||={go:.3e}  \
             ||g_query−g_calib||={:.3e}  (κ={})",
            new.n_cells, old.n_cells, cf.g_delta_norm, args.prior_strength
        );
        info!(
            "counterfactual: τ_new={:+.4e} (query-specific benefit)  τ_old={:+.4e} ({})",
            cf.tau_new,
            cf.tau_old,
            if cf.tau_old < 0.0 {
                "forgetting"
            } else {
                "no forgetting"
            }
        );
        let per_topic: Vec<String> = cf
            .delta_norm_per_topic
            .iter()
            .map(|v| format!("{v:.3e}"))
            .collect();
        info!("per-topic ||Δ_k||: [{}]", per_topic.join(", "));

        format!(
            ",\"tau_new\":{:.6e},\"tau_old\":{:.6e},\"g_query_norm\":{gn:.6e},\
             \"g_calib_norm\":{go:.6e},\"g_delta_norm\":{:.6e},\"kappa\":{},\
             \"delta_norm_per_topic\":[{}]",
            cf.tau_new,
            cf.tau_old,
            cf.g_delta_norm,
            args.prior_strength,
            cf.delta_norm_per_topic
                .iter()
                .map(|v| format!("{v:.6e}"))
                .collect::<Vec<_>>()
                .join(",")
        )
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
             \"novelty_rate\":{rate:.4},\"p_value\":{pval:.3e},\"verdict\":\"{verdict}\"{infl_json}}}\n",
            args.alpha
        ),
    )?;

    info!(
        "probe [{}]: null p{:.0} fit ≤ {:.4}; flagged {}/{} query cells ({:.1}%), p = {:.2e}",
        metadata.model_type,
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
