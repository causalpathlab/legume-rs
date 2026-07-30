//! Emitting `{out}.posterior_hyper.json`: read the frozen pb side back out of the fitted
//! `VarMap`, run the collinearity and effective-rank diagnostics both `--posterior` help
//! texts tell the reader to check before trusting a PIP, and serialize. Two entry points
//! because the gate layout differs — bge reports one `per_dim` block, gem's splice path two.

use super::{PartitionGeometry, PbGibbsResult, SpliceGibbsResult};
use crate::posterior::lnpdf::FrozenSide;
use log::info;

/// Write `{out}.posterior_hyper.json` and fire the collinearity warning.
///
/// Both `--posterior` help texts tell the reader to judge the inclusion
/// probabilities against the effective rank before trusting them, so the numbers
/// that support that judgement have to be produced. The frozen side reported here
/// is the **pseudobulk** side the gene block actually conditioned on — not the
/// full-cell side an earlier post-hoc pass used.
/// Caller-facing wrapper: rebuild the pb frozen side from the fitted `VarMap` (the
/// posterior already wrote its means there) and emit the diagnostics.
pub fn write_posterior_hyper_from_model(
    out_prefix: &str,
    res: &PbGibbsResult,
    varmap: &candle_util::candle_nn::VarMap,
    seed: u64,
) -> anyhow::Result<()> {
    let vars = varmap.data().lock().expect("varmap poisoned");
    let mut e: Vec<f32> = Vec::new();
    let mut b: Vec<f32> = Vec::new();
    for level in 0.. {
        let Some(v) = vars.get(&format!("pb_l{level}_e_cell")) else {
            break;
        };
        e.extend(v.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        if let Some(bv) = vars.get(&format!("pb_l{level}_b_cell")) {
            b.extend(bv.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        }
    }
    drop(vars);
    anyhow::ensure!(
        !e.is_empty() && e.len() == b.len() * res.h,
        "posterior diagnostics: pb side is {} loadings against {} biases at h={}",
        e.len(),
        b.len(),
        res.h
    );
    let side = FrozenSide {
        e: &e,
        b: &b,
        h: res.h,
    };
    write_posterior_hyper(out_prefix, res, &side, seed)
}

/// Splice-model twin of [`write_posterior_hyper_from_model`], with a `per_dim`
/// block per gate. gem's `--posterior` help points at the same effective-rank
/// caveat as bge's, so it needs the same numbers written.
pub fn write_splice_posterior_hyper(
    out_prefix: &str,
    res: &SpliceGibbsResult,
    varmap: &candle_util::candle_nn::VarMap,
    seed: u64,
) -> anyhow::Result<()> {
    let (e, b) = stacked_pb_from_varmap(varmap, res.h)?;
    let side = FrozenSide {
        e: &e,
        b: &b,
        h: res.h,
    };
    let gates = serde_json::json!({
        "beta": {
            "sigma2": res.beta_sigma2,
            "pi0": res.beta_pi0,
            "sigma2_min_ess": res.beta_sigma_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "sigma2_rhat": res.beta_sigma_diag.iter().map(|d| d.rhat).collect::<Vec<_>>(),
            "pi0_min_ess": res.beta_pi0_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "pi0_rhat": res.beta_pi0_diag.iter().map(|d| d.rhat).collect::<Vec<_>>(),
        },
        "delta": {
            "sigma2": res.delta_sigma2,
            "pi0": res.delta_pi0,
            "sigma2_min_ess": res.delta_sigma_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "sigma2_rhat": res.delta_sigma_diag.iter().map(|d| d.rhat).collect::<Vec<_>>(),
            "pi0_min_ess": res.delta_pi0_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "pi0_rhat": res.delta_pi0_diag.iter().map(|d| d.rhat).collect::<Vec<_>>(),
            "n_unidentified": res.delta_identified.iter().filter(|&&x| !x).count(),
        },
    });
    emit_hyper_json(
        out_prefix,
        res.h,
        res.n_kept,
        res.partition,
        seed,
        &side,
        gates,
    )
}

/// The stacked pb side, read back out of the fitted `VarMap`.
fn stacked_pb_from_varmap(
    varmap: &candle_util::candle_nn::VarMap,
    h: usize,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let vars = varmap.data().lock().expect("varmap poisoned");
    let (mut e, mut b): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    for level in 0.. {
        let Some(v) = vars.get(&format!("pb_l{level}_e_cell")) else {
            break;
        };
        e.extend(v.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        if let Some(bv) = vars.get(&format!("pb_l{level}_b_cell")) {
            b.extend(bv.as_tensor().flatten_all()?.to_vec1::<f32>()?);
        }
    }
    anyhow::ensure!(
        !e.is_empty() && e.len() == b.len() * h,
        "posterior diagnostics: pb side is {} loadings against {} biases at h={h}",
        e.len(),
        b.len()
    );
    Ok((e, b))
}

/// Shared body: the frozen-side geometry, its two warnings, and the JSON.
fn emit_hyper_json(
    out_prefix: &str,
    h: usize,
    n_kept: usize,
    partition: PartitionGeometry,
    seed: u64,
    frozen_pb: &FrozenSide<'_>,
    per_gate: serde_json::Value,
) -> anyhow::Result<()> {
    let geom = crate::posterior::frozen_diag::frozen_side_diag(frozen_pb);
    info!(
        "posterior frozen pb side: common-mode cos {:.3}, effective rank {:.1} raw / {:.1} \
         centered of {h}, max |corr| {:.3}, max VIF {:.2}",
        geom.common_mode_cos,
        geom.eff_rank_raw,
        geom.eff_rank_centered,
        geom.max_abs_corr,
        geom.max_vif,
    );
    if geom.max_vif >= 5.0 {
        log::warn!(
            "frozen dims are collinear (max VIF {:.1} ≥ 5) — per-dim inclusion \
             probabilities split mass between the correlated dims and can read \
             confidently wrong on both; read a gene's row as a profile, not a winner",
            geom.max_vif,
        );
    }
    if geom.eff_rank_raw < 0.5 * h as f32 {
        log::warn!(
            "the embedding uses ~{:.1} of its {h} dims, so the likelihood carries no \
             information about the rest and their inclusion indicators fall back to the \
             prior — expect PIPs that do not discriminate between genes",
            geom.eff_rank_raw,
        );
    }
    let json = serde_json::json!({
        "n_sweeps": n_kept,
        // What the normalizers ACTUALLY summed over. `scale == 1.0` marks an axis summed
        // exactly, hence free of the Jensen bias a sampled log-normalizer carries; the
        // config cap is not reported because since the slate became data-dependent it no
        // longer says what a run did.
        "partition": {
            "pb_entries": partition.pb_entries,
            "pb_scale": partition.pb_scale,
            "pb_exact": partition.pb_scale == 1.0,
            "feature_entries": partition.feat_entries,
            "feature_scale": partition.feat_scale,
            "feature_exact": partition.feat_scale == 1.0,
        },
        "seed": seed,
        "interrupted": crate::stop::stop_flag().load(std::sync::atomic::Ordering::Relaxed),
        "gates": per_gate,
        "frozen_side": geom,
    });
    let path = format!("{out_prefix}.posterior_hyper.json");
    std::fs::write(&path, format!("{}\n", serde_json::to_string_pretty(&json)?))?;
    info!("wrote {path}");
    Ok(())
}

fn write_posterior_hyper(
    out_prefix: &str,
    res: &PbGibbsResult,
    frozen_pb: &FrozenSide<'_>,
    seed: u64,
) -> anyhow::Result<()> {
    let geom = crate::posterior::frozen_diag::frozen_side_diag(frozen_pb);
    info!(
        "posterior frozen pb side: common-mode cos {:.3}, effective rank {:.1} raw / {:.1} \
         centered of {}, max |corr| {:.3}, max VIF {:.2}",
        geom.common_mode_cos,
        geom.eff_rank_raw,
        geom.eff_rank_centered,
        res.h,
        geom.max_abs_corr,
        geom.max_vif,
    );
    if geom.max_vif >= 5.0 {
        log::warn!(
            "frozen dims are collinear (max VIF {:.1} ≥ 5) — per-dim inclusion \
             probabilities split mass between the correlated dims and can read \
             confidently wrong on both; read a gene's row as a profile, not a winner",
            geom.max_vif,
        );
    }
    if geom.eff_rank_raw < 0.5 * res.h as f32 {
        log::warn!(
            "the embedding uses ~{:.1} of its {} dims, so the likelihood carries no \
             information about the rest and their inclusion indicators fall back to the \
             prior — expect PIPs that do not discriminate between genes",
            geom.eff_rank_raw,
            res.h,
        );
    }
    let json = serde_json::json!({
        "n_sweeps": res.n_kept,
        // What the normalizers ACTUALLY summed over. `scale == 1.0` marks an axis summed
        // exactly, hence free of the Jensen bias a sampled log-normalizer carries; the
        // config cap is not reported because since the slate became data-dependent it no
        // longer says what a run did.
        "partition": {
            "pb_entries": res.partition.pb_entries,
            "pb_scale": res.partition.pb_scale,
            "pb_exact": res.partition.pb_scale == 1.0,
            "feature_entries": res.partition.feat_entries,
            "feature_scale": res.partition.feat_scale,
            "feature_exact": res.partition.feat_scale == 1.0,
        },
        "seed": seed,
        "interrupted": crate::stop::stop_flag().load(std::sync::atomic::Ordering::Relaxed),
        "per_dim": {
            "sigma2": res.sigma2,
            "pi0": res.pi0,
            "sigma2_min_ess": res.sigma_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "sigma2_rhat": res.sigma_diag.iter().map(|d| d.rhat).collect::<Vec<_>>(),
            "pi0_min_ess": res.pi0_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "pi0_rhat": res.pi0_diag.iter().map(|d| d.rhat).collect::<Vec<_>>(),
        },
        "frozen_side": geom,
    });
    let path = format!("{out_prefix}.posterior_hyper.json");
    // Through `serde_json` rather than a format string: a stalled chain can leave
    // a non-finite diagnostic, and a bare `NaN` is not JSON. serde emits `null`.
    std::fs::write(&path, format!("{}\n", serde_json::to_string_pretty(&json)?))?;
    info!("wrote {path}");
    Ok(())
}
