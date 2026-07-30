//! Two-sided blocked Gibbs over the phase-1 pseudobulk model.
//!
//! Phase 1 fits a bilinear Poisson model on pseudobulk counts: a shared feature
//! side (`e_feat`, `b_feat`, plus the softmax gate) and one pb head per collapse
//! level (`pb_l{k}_e_cell`, `pb_l{k}_b_cell`). It fits them by NCE-SGD and reports
//! a point estimate. This samples the same model instead, warm-started from that
//! point estimate, alternating:
//!
//! ```text
//!   genes | {pb_l0 … pb_lK}     per-dim spike-and-slab
//!   pb    | genes               per-dim Gaussian
//! ```
//!
//! ## All levels at once, and why that needs no special code
//!
//! Phase 1's objective is `CompositeMode::Sum` with `λ = 1` per level, i.e. the
//! log-likelihoods of the levels add. [`StackedPb`] already concatenates every
//! level into one pb axis with `offsets[l]` marking where each starts, so a
//! single Poisson likelihood over that concatenated axis **is** the sum over
//! levels. There is no per-level loop here and there should not be one.
//!
//! ## Selection lives on the feature side only
//!
//! The gene block carries the spike-and-slab `z` — that is what the trained gate
//! is, and its posterior inclusion probability is the output downstream consumes.
//! A pseudobulk has no selection story: it is a location in the latent space, not
//! a thing that is present or absent. So the pb block is a plain per-dim Gaussian
//! update, which is [`dim_block`] with the null mass pinned at zero.
//!
//! ## Cost
//!
//! The frozen side is constant *within* a block, which is what makes each block
//! embarrassingly parallel over its anchors — the same property the one-sided
//! samplers rely on. Between blocks the other side has moved, so each block
//! rebuilds its [`AnchorMoment`]s. That rebuild is `O(edges · h)` once per block
//! per sweep, against `O(anchors · h · evals · |partition|)` for the dim scan
//! itself, so it does not show up in the profile.

use super::diagnostics::scalar_diagnostics;
use super::dim_block::{dim_block, dim_block_multi, DimBlockConfig, DimBlockResult};
use super::diagnostics::ChainDiag;
use super::lnpdf::{AnchorMoment, FrozenSide, NodeTerm};
use super::pb_index::{build_pb_index_pair, AnchorMap, FeatureSide};
use crate::fit::feature_projection::StackedPb;
use crate::progress::new_progress_bar;
use log::info;

/// Configuration for [`pb_gibbs`].
pub struct PbGibbsConfig {
    /// Outer sweeps retained. Each is one gene block + one pb block.
    pub n_sweeps: usize,
    /// Warm-up sweeps discarded before accumulation, on top of `n_sweeps`.
    pub burnin: usize,
    pub seed: u64,
    /// Cap on each side's frozen negative slate; `0` sums both exactly. The pb
    /// axis is usually under this and the feature axis over it.
    pub n_partition: usize,
    /// Inner ESS transitions per `(anchor, dim)` within a block.
    pub transitions_per_dim: usize,
}

impl PbGibbsConfig {
    #[must_use]
    pub fn new(n_sweeps: usize, burnin: usize, seed: u64) -> Self {
        Self {
            n_sweeps,
            burnin,
            seed,
            n_partition: super::run::DEFAULT_PARTITION,
            transitions_per_dim: 1,
        }
    }
}

/// What the sweep leaves behind: the gene side's selection posterior, both sides'
/// posterior-mean loadings, and the per-dim hypers the gene block learned.
pub struct PbGibbsResult {
    /// `[n_anchors × h]` row-major `P(z = 1)` per (gene, dim).
    pub pip: Vec<f32>,
    /// `[n_anchors × h]` row-major `E[z · β]` — the **effective** loading, i.e.
    /// what the model actually uses, with excluded draws contributing zero.
    pub mean_beta: Vec<f32>,
    /// `[n_pb_total × h]` row-major posterior-mean pb loading, levels
    /// concatenated in [`StackedPb`] order.
    pub mean_pb: Vec<f32>,
    /// `[n_anchors]` intercepts implied by [`Self::mean_beta`], and `[n_pb_total]`
    /// implied by [`Self::mean_pb`].
    ///
    /// `multinomial_ll` maximises the anchor intercept out, so a sampled loading
    /// is only identified together with `b_a* = ln(T_a / A(θ))`. Shipping the
    /// loading without it leaves the model pairing a sampled embedding with a bias
    /// fitted under a different objective at a different scale, and every rate the
    /// model predicts downstream is then off by a per-anchor factor that no shape
    /// check can catch.
    pub mean_b_feat: Vec<f32>,
    pub mean_b_pb: Vec<f32>,
    /// Posterior-mean slab variance per dim `[h]`, from the gene block.
    pub sigma2: Vec<f64>,
    /// Posterior-mean null mass per dim `[h]`, from the gene block.
    pub pi0: Vec<f64>,
    pub sigma_diag: Vec<ChainDiag>,
    pub pi0_diag: Vec<ChainDiag>,
    pub h: usize,
    /// Outer sweeps actually retained — below `n_sweeps` only if SIGINT hit.
    pub n_kept: usize,
}

/// Run the alternating sweep. `e_feat` / `b_feat` are the warm start for the gene
/// side (the phase-1 MAP); [`StackedPb::theta`] is the warm start for the pb side.
pub(crate) fn pb_gibbs(
    pb: &StackedPb<'_>,
    feat: &FeatureSide<'_>,
    anchors: Option<&AnchorMap<'_>>,
    h: usize,
    cfg: &PbGibbsConfig,
) -> anyhow::Result<PbGibbsResult> {
    let pair = build_pb_index_pair(pb, feat, anchors, h, cfg.n_partition, cfg.seed)?;
    let n_anchors = pair.by_feature.pos.len();
    let n_pb = pair.by_pb.pos.len();
    info!(
        "pb Gibbs: {n_anchors} gene anchor(s) × {n_pb} pseudobulk(s) over {} edge(s), \
         {} sweeps (+{} warmup); partitions {} pb / {} feature (scale {:.2} / {:.2})",
        pair.n_edges,
        cfg.n_sweeps,
        cfg.burnin,
        pair.by_feature.partition.len(),
        pair.by_pb.partition.len(),
        pair.by_feature.partition_scale,
        pair.by_pb.partition_scale,
    );

    // Live state. The gene side starts at the phase-1 MAP restricted to the
    // anchors this run samples; the pb side starts at its own MAP.
    let mut e_gene = warm_start_genes(feat, anchors, n_anchors, h);
    let mut e_pb = pb.theta.clone();
    // Intercepts are LIVE, not snapshotted. Each block profiles out its own anchor's
    // intercept analytically, so after a block the just-sampled side's biases are known
    // exactly — and the OTHER side then has to score against those, not against a
    // value fitted before any sampling happened. Under a warm start the difference is
    // small because the snapshot was an SGD fit; with no SGD to snapshot there is
    // nothing there but the initialization, which is why this cannot stay frozen.
    //
    // Note these are PROFILE steps, not draws: the conditional maximizer plugged in,
    // i.e. Bayes-EM inside the Gibbs sweep. The write-back has always done exactly this
    // once at the end; carrying it into the loop is the same operation, not a new one.
    let mut b_pb_live = pair.by_feature.other_b.clone();
    let mut b_row_live = pair.by_pb.other_b.clone();
    // Scratch for the row-indexed view of the gene side, rebuilt each sweep.
    let mut e_rows = feat.e_feat.to_vec();

    let mut pip_acc = vec![0.0f64; n_anchors * h];
    let mut beta_acc = vec![0.0f64; n_anchors * h];
    let mut pb_acc = vec![0.0f64; n_pb * h];
    // Per-OUTER-sweep hyper chains. Each inner block runs a single sweep, so its
    // own chains hold one sample and `scalar_diagnostics` short-circuits to
    // ESS = 1; the chain that actually mixes is this one, across outer sweeps.
    let mut sigma2_chain: Vec<Vec<f64>> = vec![Vec::new(); h];
    let mut pi0_chain: Vec<Vec<f64>> = vec![Vec::new(); h];
    // Inclusion state carried BETWEEN outer sweeps. Without this every block
    // restarts cold, the ESS branch is unreachable, and every β is a prior draw.
    let mut z_gene: Option<Vec<bool>> = None;
    let mut n_kept = 0usize;

    let stop = crate::stop::stop_flag();
    let total = cfg.n_sweeps + cfg.burnin;
    let bar = new_progress_bar(total as u64).with_message("pb gibbs sweeps");

    for sweep in 0..total {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let keep = sweep >= cfg.burnin;

        // ---- genes | pb -------------------------------------------------
        // One inner sweep per outer sweep: the block is a valid Gibbs move on
        // its own, and running a whole sub-chain here would just discard the
        // other side's current state that many times over.
        let side_pb = FrozenSide {
            e: &e_pb,
            b: &b_pb_live,
            h,
        };
        let moments: Vec<AnchorMoment> = pair
            .by_feature
            .pos
            .iter()
            .map(|p| AnchorMoment::new(p, &side_pb))
            .collect();
        let nodes: Vec<NodeTerm> = pair
            .by_feature
            .pos
            .iter()
            .zip(&moments)
            .map(|(p, m)| {
                NodeTerm::new(p, &pair.by_feature.partition, pair.by_feature.partition_scale)
                    .with_moment(m)
            })
            .collect();
        let mut gene_cfg = DimBlockConfig::new(1, 0, block_seed(cfg.seed, 0x9E37, sweep))
            .with_init_beta(e_gene.clone())
            .with_label("genes|pb")
            .quiet();
        if let Some(z) = z_gene.clone() {
            gene_cfg = gene_cfg.with_init_z(z);
        }
        gene_cfg.transitions_per_dim = cfg.transitions_per_dim;
        let g = dim_block(&nodes, &side_pb, &gene_cfg);

        // The gene side hands the pb block its EFFECTIVE loading `z·β`: a dim the
        // gate turned off must contribute nothing to what the pseudobulks are
        // fit against, or the pb side would be conditioning on a model the gene
        // side has already rejected. `z` is carried alongside it so the next
        // sweep resumes the chain rather than restarting cold.
        e_gene.copy_from_slice(&g.mean_beta);
        z_gene = Some(g.final_z.clone());
        // One term on this path, so `b_profiled` is one intercept per anchor. Scatter
        // it to rows the same way the loadings are: rows this run does not sample keep
        // their phase-1 value rather than being dropped.
        scatter_to_rows(&g.b_profiled, anchors, feat.b_feat, 1, &mut b_row_live);

        if keep {
            for (acc, v) in pip_acc.iter_mut().zip(&g.pip) {
                *acc += f64::from(*v);
            }
            for (acc, v) in beta_acc.iter_mut().zip(&g.mean_beta) {
                *acc += f64::from(*v);
            }
            for d in 0..h {
                sigma2_chain[d].push(g.sigma2[d]);
                pi0_chain[d].push(g.pi0[d]);
            }
        }

        // ---- pb | genes -------------------------------------------------
        // The pb block's other side is indexed by feature ROW, not by anchor:
        // `pos_pb` holds unified row ids, because a pseudobulk is scored against
        // the whole feature axis. Under a grouping those differ, so scatter the
        // sampled anchor loadings back out — and leave rows this run does not
        // sample (gem's other splice track) at their phase-1 MAP rather than
        // dropping them, which would misspecify the pb likelihood.
        scatter_to_rows(&e_gene, anchors, feat.e_feat, h, &mut e_rows);

        // Every dim is on: a pseudobulk is a location, not a selection.
        let side_gene = FrozenSide {
            e: &e_rows,
            b: &b_row_live,
            h,
        };
        let pb_moments: Vec<AnchorMoment> = pair
            .by_pb
            .pos
            .iter()
            .map(|p| AnchorMoment::new(p, &side_gene))
            .collect();
        let pb_nodes: Vec<NodeTerm> = pair
            .by_pb
            .pos
            .iter()
            .zip(&pb_moments)
            .map(|(p, m)| {
                NodeTerm::new(p, &pair.by_pb.partition, pair.by_pb.partition_scale).with_moment(m)
            })
            .collect();
        let mut pb_cfg = DimBlockConfig::new(1, 0, block_seed(cfg.seed, 0x85EB, sweep))
            .with_init_beta(e_pb.clone())
            .with_label("pb|genes")
            .without_selection()
            .quiet();
        pb_cfg.transitions_per_dim = cfg.transitions_per_dim;
        let p = dim_block(&pb_nodes, &side_gene, &pb_cfg);
        e_pb.copy_from_slice(&p.mean_beta);
        // The pb intercepts absorb the exposure: `T_p` is a pseudobulk's total COUNT
        // (edges already carry `rate · size_p`), so the profiled value is the whole
        // `b_pb + ln size_p` the score needs, not just the free part. That is why this
        // can replace `other_b` outright without re-adding `ln size_p`.
        b_pb_live.copy_from_slice(&p.b_profiled);

        if keep {
            for (acc, v) in pb_acc.iter_mut().zip(&p.mean_beta) {
                *acc += f64::from(*v);
            }
            n_kept += 1;
        }
        bar.inc(1);
    }
    bar.finish_and_clear();

    // A run cut short during burn-in has accumulated nothing. Returning `Ok` with
    // all-zero means would look like a completed posterior to the write-back,
    // which validates only shapes — and the model would silently ship a zeroed
    // dictionary.
    anyhow::ensure!(
        n_kept > 0,
        "pb Gibbs retained no sweeps ({} of {} burn-in completed before the run was \
         interrupted) — the model is left at its SGD fit rather than overwritten",
        total.min(cfg.burnin),
        cfg.burnin
    );
    let inv = 1.0 / n_kept as f64;
    let mean_chain = |c: &[Vec<f64>]| -> Vec<f64> {
        c.iter()
            .map(|v| v.iter().sum::<f64>() / v.len().max(1) as f64)
            .collect()
    };
    let mean_beta: Vec<f32> = beta_acc.iter().map(|&a| (a * inv) as f32).collect();
    let mean_pb: Vec<f32> = pb_acc.iter().map(|&a| (a * inv) as f32).collect();

    // Recover the intercepts the profile likelihood maximised out, against the
    // POSTERIOR-MEAN sides — the loadings and the biases have to describe the same
    // model, or every downstream rate is off by a per-anchor factor.
    //
    // The other side's biases come from the LIVE buffers, i.e. where the chain ended,
    // not from the pre-sampling snapshot. Profiling against the snapshot would pair a
    // sampled embedding with an intercept fitted before any sampling ran, which is the
    // mismatch this whole profiling scheme exists to avoid.
    let side_pb_final = FrozenSide {
        e: &mean_pb,
        b: &b_pb_live,
        h,
    };
    let mean_b_feat: Vec<f32> = (0..n_anchors)
        .map(|a| {
            let total: f64 = pair.by_feature.pos[a].iter().map(|&(_, n)| f64::from(n)).sum();
            profiled_bias(
                total,
                &mean_beta[a * h..(a + 1) * h],
                None,
                &pair.by_feature.partition,
                pair.by_feature.partition_scale,
                &side_pb_final,
            )
        })
        .collect();
    scatter_to_rows(&mean_beta, anchors, feat.e_feat, h, &mut e_rows);
    let side_rows_final = FrozenSide {
        e: &e_rows,
        b: &b_row_live,
        h,
    };
    let mean_b_pb: Vec<f32> = (0..n_pb)
        .map(|p| {
            let total: f64 = pair.by_pb.pos[p].iter().map(|&(_, n)| f64::from(n)).sum();
            profiled_bias(
                total,
                &mean_pb[p * h..(p + 1) * h],
                None,
                &pair.by_pb.partition,
                pair.by_pb.partition_scale,
                &side_rows_final,
            )
        })
        .collect();

    Ok(PbGibbsResult {
        pip: pip_acc.iter().map(|&a| (a * inv) as f32).collect(),
        mean_beta,
        mean_pb,
        mean_b_feat,
        mean_b_pb,
        sigma2: mean_chain(&sigma2_chain),
        pi0: mean_chain(&pi0_chain),
        sigma_diag: sigma2_chain.iter().map(|c| scalar_diagnostics(c)).collect(),
        pi0_diag: pi0_chain.iter().map(|c| scalar_diagnostics(c)).collect(),
        h,
        n_kept,
    })
}

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
    n_partition: usize,
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
    write_posterior_hyper(out_prefix, res, &side, n_partition, seed)
}

/// Splice-model twin of [`write_posterior_hyper_from_model`], with a `per_dim`
/// block per gate. gem's `--posterior` help points at the same effective-rank
/// caveat as bge's, so it needs the same numbers written.
pub fn write_splice_posterior_hyper(
    out_prefix: &str,
    res: &SpliceGibbsResult,
    varmap: &candle_util::candle_nn::VarMap,
    n_partition: usize,
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
            "pi0_min_ess": res.beta_pi0_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
        },
        "delta": {
            "sigma2": res.delta_sigma2,
            "pi0": res.delta_pi0,
            "sigma2_min_ess": res.delta_sigma_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "pi0_min_ess": res.delta_pi0_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "n_unidentified": res.delta_identified.iter().filter(|&&x| !x).count(),
        },
    });
    emit_hyper_json(out_prefix, res.h, res.n_kept, n_partition, seed, &side, gates)
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
    n_partition: usize,
    seed: u64,
    frozen_pb: &FrozenSide<'_>,
    per_gate: serde_json::Value,
) -> anyhow::Result<()> {
    let geom = super::frozen_diag::frozen_side_diag(frozen_pb);
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
        "n_partition": n_partition,
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
    n_partition: usize,
    seed: u64,
) -> anyhow::Result<()> {
    let geom = super::frozen_diag::frozen_side_diag(frozen_pb);
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
        "n_partition": n_partition,
        "seed": seed,
        "interrupted": crate::stop::stop_flag().load(std::sync::atomic::Ordering::Relaxed),
        "per_dim": {
            "sigma2": res.sigma2,
            "pi0": res.pi0,
            "sigma2_min_ess": res.sigma_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
            "pi0_min_ess": res.pi0_diag.iter().map(|d| d.min_ess).collect::<Vec<_>>(),
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

/// The intercept the profile likelihood maximised out, recovered for a finished
/// loading: `b_a* = ln(T_a) − ln(scale · Σ_o exp(⟨e_a + off, e_o⟩ + b_o))`.
///
/// Max-shifted and accumulated in `f64` — the exponents span the whole bias range,
/// which is exactly where a naive sum loses its low bits. An anchor with no counts
/// has no rate to match and is parked at `-SCORE_CLAMP` (rate ≈ 0), matching
/// [`super::index::ContrastiveIndex::calibrate_anchor_bias`].
fn profiled_bias(
    total: f64,
    e_a: &[f32],
    off: Option<&[f32]>,
    partition: &[u32],
    scale: f64,
    side: &FrozenSide<'_>,
) -> f32 {
    if total <= 0.0 {
        return -(crate::cell_projection::SCORE_CLAMP as f32);
    }
    let h = side.h;
    let sc = |o: u32| -> f64 {
        let e_o = &side.e[o as usize * h..(o as usize + 1) * h];
        let dot: f64 = match off {
            None => e_a
                .iter()
                .zip(e_o)
                .map(|(a, v)| f64::from(*a) * f64::from(*v))
                .sum(),
            Some(f) => e_a
                .iter()
                .zip(f)
                .zip(e_o)
                .map(|((a, b), v)| (f64::from(*a) + f64::from(*b)) * f64::from(*v))
                .sum(),
        };
        dot + f64::from(side.b[o as usize])
    };
    let m = partition.iter().map(|&o| sc(o)).fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return 0.0;
    }
    let s: f64 = partition.iter().map(|&o| (sc(o) - m).exp()).sum();
    let log_part = m + (scale * s).max(f64::MIN_POSITIVE).ln();
    (total.ln() - log_part).clamp(
        -(crate::cell_projection::SCORE_CLAMP),
        crate::cell_projection::SCORE_CLAMP,
    ) as f32
}

/// Per-block RNG seed that stays distinct at every sweep, including sweep 0.
///
/// The obvious `seed ^ (sweep * salt)` collapses to `seed` for every salt when
/// `sweep == 0`, so on the sweep that seeds the whole chain the β, δ and pb blocks
/// would draw the identical stream of normals and uniforms — the two gates'
/// proposals perfectly correlated exactly where it matters most.
fn block_seed(seed: u64, salt: u64, sweep: usize) -> u64 {
    seed.rotate_left(17)
        ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (sweep as u64).wrapping_add(1).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

/// Gene-side warm start `[n_anchors × h]`, pooled from the phase-1 feature rows.
///
/// Under a grouping several rows share an anchor (gem's spliced + unspliced both
/// carry `β_g`), so the anchor takes the FIRST mapped row rather than a sum —
/// they are copies of one parameter, and adding them would double it.
fn warm_start_genes(
    feat: &FeatureSide<'_>,
    anchors: Option<&AnchorMap<'_>>,
    n_anchors: usize,
    h: usize,
) -> Vec<f32> {
    match anchors {
        None => feat.e_feat.to_vec(),
        Some(a) => {
            let mut out = vec![0.0f32; n_anchors * h];
            let mut seen = vec![false; n_anchors];
            for (uid, &an) in a.row_to_anchor.iter().enumerate() {
                if an == u32::MAX || seen[an as usize] {
                    continue;
                }
                let (s, d) = (an as usize * h, uid * h);
                out[s..s + h].copy_from_slice(&feat.e_feat[d..d + h]);
                seen[an as usize] = true;
            }
            out
        }
    }
}

/////////////////////////////////////////
// Two-gate (splice) variant — faba gem //
/////////////////////////////////////////

/// `faba gem`'s β-sharing feature side: one `β_g` per gene, plus a velocity
/// deviation `δ_g` that applies only to the gene's **unspliced** rows.
pub struct SpliceTracks<'a> {
    /// Feature row → gene, length `n_features`; `u32::MAX` drops the row.
    pub row_to_gene: &'a [u32],
    /// Which rows carry `δ_g` (an unspliced row scores `⟨β_g + δ_g, e_p⟩`).
    pub unspliced_rows: &'a [bool],
    pub n_genes: usize,
    /// Allow `z_δ = 1` only where `z_β = 1`. See [`DimBlockConfig::z_allowed`] for
    /// why this is the default.
    pub nested: bool,
}

/// Both gates' posteriors, plus which genes could identify `δ` at all.
pub struct SpliceGibbsResult {
    pub beta_pip: Vec<f32>,
    pub beta_mean: Vec<f32>,
    pub delta_pip: Vec<f32>,
    pub delta_mean: Vec<f32>,
    /// Per gene: does it have SPLICED counts? `δ_g` is identified by the
    /// unspliced-minus-spliced contrast, so a gene observed only in the unspliced
    /// track pins `β + δ` but neither separately — its `δ` row is a prior draw,
    /// not a measurement, and must be reported as such rather than as a number.
    pub delta_identified: Vec<bool>,
    pub mean_pb: Vec<f32>,
    /// Per-gene and per-pseudobulk intercepts implied by the posterior-mean
    /// loadings — see [`PbGibbsResult::mean_b_feat`] for why they must travel
    /// together with them.
    pub mean_b_feat: Vec<f32>,
    pub mean_b_pb: Vec<f32>,
    pub beta_sigma2: Vec<f64>,
    pub beta_pi0: Vec<f64>,
    pub delta_sigma2: Vec<f64>,
    pub delta_pi0: Vec<f64>,
    /// Mixing diagnostics per dim, over the OUTER sweeps — see `SpliceAcc`.
    pub beta_sigma_diag: Vec<ChainDiag>,
    pub beta_pi0_diag: Vec<ChainDiag>,
    pub delta_sigma_diag: Vec<ChainDiag>,
    pub delta_pi0_diag: Vec<ChainDiag>,
    pub h: usize,
    pub n_kept: usize,
}

/// Alternating sweep over gem's three blocks: `β | δ, pb`, then `δ | β, pb`, then
/// `pb | β, δ`.
///
/// **`β` reads the spliced rows only.** A [`NodeTerm`] carries one offset for all
/// of an anchor's edges, so "β with δ added on just the unspliced edges" is not
/// expressible as a single node; splitting β's likelihood across the two tracks
/// would need a per-edge offset. Reading β off the spliced rows is also the
/// cleaner estimand — those rows score exactly `⟨β_g, e_p⟩` with nothing to
/// disentangle — and it is what gem's previous posterior did.
///
/// `δ` then reads the unspliced rows with `β_g` as its per-anchor offset,
/// **refreshed every sweep**. Holding it at a one-time MAP snapshot, as the
/// previous implementation did, makes the two conditionals describe no single
/// joint.
pub(crate) fn pb_gibbs_splice(
    pb: &StackedPb<'_>,
    feat: &FeatureSide<'_>,
    tracks: &SpliceTracks<'_>,
    h: usize,
    cfg: &PbGibbsConfig,
) -> anyhow::Result<SpliceGibbsResult> {
    let n_genes = tracks.n_genes;
    anyhow::ensure!(
        tracks.row_to_gene.len() == feat.b_feat.len()
            && tracks.unspliced_rows.len() == feat.b_feat.len(),
        "splice tracks disagree with the {}-row feature axis",
        feat.b_feat.len()
    );

    // One anchor map per track: a row belongs to a track's gene only if it is on
    // that track. Everything else is dropped from THAT gene side (never from the
    // pb side — `build_pb_index_pair` keeps every row there by construction).
    let track_map = |want_unspliced: bool| -> Vec<u32> {
        tracks
            .row_to_gene
            .iter()
            .zip(tracks.unspliced_rows)
            .map(|(&g, &u)| if u == want_unspliced { g } else { u32::MAX })
            .collect()
    };
    let spliced_map = track_map(false);
    let unspliced_map = track_map(true);

    let beta_anchors = AnchorMap {
        row_to_anchor: &spliced_map,
        n_anchors: n_genes,
    };
    let delta_anchors = AnchorMap {
        row_to_anchor: &unspliced_map,
        n_anchors: n_genes,
    };
    let beta_pair = build_pb_index_pair(pb, feat, Some(&beta_anchors), h, cfg.n_partition, cfg.seed)?;
    let delta_pair =
        build_pb_index_pair(pb, feat, Some(&delta_anchors), h, cfg.n_partition, cfg.seed)?;

    // Identifiability, decided up front and reported, not inferred later.
    //
    // δ needs counts on BOTH tracks. No spliced counts ⇒ only β+δ is pinned. No
    // UNSPLICED counts ⇒ δ appears in no likelihood term at all: `multinomial_ll`
    // returns 0 for every argument via its `total == 0` early return, so `z` is
    // drawn straight from the prior odds and β_δ from the slab. Testing only the
    // spliced side would ship that second case as an apparently-measured number.
    let delta_identified: Vec<bool> = beta_pair
        .by_feature
        .pos
        .iter()
        .zip(&delta_pair.by_feature.pos)
        .map(|(spliced, unspliced)| !spliced.is_empty() && !unspliced.is_empty())
        .collect();
    let n_unidentified = delta_identified.iter().filter(|&&x| !x).count();
    if n_unidentified > 0 {
        log::warn!(
            "{n_unidentified} of {n_genes} gene(s) lack counts on one of the two splice \
             tracks, so their δ is not identified — with no spliced counts only β+δ is \
             pinned, and with no unspliced counts δ enters no likelihood term at all. \
             Both cases are flagged in `delta_identified` and written as NaN."
        );
    }
    // Per-TRACK edge counts, from the anchor-side lists. `PbIndexPair::n_edges` is
    // the whole matrix's observed edges — it is incremented once per nonzero
    // regardless of the anchor map — so reporting it here would print the same
    // number twice and claim it was the split.
    let n_spliced: usize = beta_pair.by_feature.pos.iter().map(Vec::len).sum();
    let n_unspliced: usize = delta_pair.by_feature.pos.iter().map(Vec::len).sum();
    info!(
        "gem splice Gibbs: {n_genes} gene(s), {} spliced / {} unspliced edge(s), {} gate ({} sweeps)",
        n_spliced,
        n_unspliced,
        if tracks.nested { "nested z_δ ⊆ z_β" } else { "independent" },
        cfg.n_sweeps,
    );

    let mut e_beta = warm_start_genes(feat, Some(&beta_anchors), n_genes, h);
    let mut e_delta = vec![0f32; n_genes * h];
    let mut e_pb = pb.theta.clone();
    let mut e_rows = feat.e_feat.to_vec();
    // Intercepts are LIVE here for the same reason as on the plain path: each block
    // profiles its own anchor's intercept out exactly, so the other side must score
    // against that rather than against a pre-sampling snapshot. See the plain path's
    // note on why this is a profile step and not a draw.
    let mut b_pb_live = beta_pair.by_feature.other_b.clone();
    let mut b_row_live = beta_pair.by_pb.other_b.clone();
    // Inclusion state carried between outer sweeps, one per gate — without it the
    // ESS branch is unreachable and every draw comes from the prior.
    let (mut z_beta, mut z_delta): (Option<Vec<bool>>, Option<Vec<bool>>) = (None, None);

    let mut acc = SpliceAcc::new(n_genes, pb.n_pb_total(), h);
    let stop = crate::stop::stop_flag();
    let total = cfg.n_sweeps + cfg.burnin;
    let bar = new_progress_bar(total as u64).with_message("gem splice sweeps");

    for sweep in 0..total {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let keep = sweep >= cfg.burnin;
        let side_pb = FrozenSide {
            e: &e_pb,
            b: &b_pb_live,
            h,
        };

        // β | δ, pb — over BOTH tracks. β appears in the unspliced rows too
        // (they score `⟨β + δ, e_p⟩`), so a spliced-only kernel would not be the
        // conditional `p(β | δ, ·)` and the three blocks would stop being
        // conditionals of one joint. The unspliced term carries δ as its offset;
        // the two tracks have independent per-row biases, so each term's
        // intercept profiles out separately and the sum is the joint's profile.
        let beta_moments_s: Vec<AnchorMoment> = beta_pair
            .by_feature
            .pos
            .iter()
            .map(|p| AnchorMoment::new(p, &side_pb))
            .collect();
        let beta_moments_u: Vec<AnchorMoment> = delta_pair
            .by_feature
            .pos
            .iter()
            .map(|p| AnchorMoment::new(p, &side_pb))
            .collect();
        let beta_terms: Vec<Vec<NodeTerm>> = (0..n_genes)
            .map(|a| {
                vec![
                    NodeTerm::new(
                        &beta_pair.by_feature.pos[a],
                        &beta_pair.by_feature.partition,
                        beta_pair.by_feature.partition_scale,
                    )
                    .with_moment(&beta_moments_s[a]),
                    {
                        let mut t = NodeTerm::new(
                            &delta_pair.by_feature.pos[a],
                            &delta_pair.by_feature.partition,
                            delta_pair.by_feature.partition_scale,
                        )
                        .with_moment(&beta_moments_u[a]);
                        t.offset = Some(&e_delta[a * h..(a + 1) * h]);
                        t
                    },
                ]
            })
            .collect();
        let mut beta_cfg = DimBlockConfig::new(1, 0, block_seed(cfg.seed, 0xB37A, sweep))
            .with_init_beta(e_beta.clone())
            .with_label("beta|delta,pb")
            .quiet();
        if let Some(z) = z_beta.clone() {
            beta_cfg = beta_cfg.with_init_z(z);
        }
        beta_cfg.transitions_per_dim = cfg.transitions_per_dim;
        let b = dim_block_multi(&beta_terms, &side_pb, &beta_cfg);
        drop(beta_terms);
        e_beta.copy_from_slice(&b.mean_beta);
        z_beta = Some(b.final_z.clone());

        // δ | β, pb, on the unspliced rows, with β carried as the per-anchor
        // offset so the sampler explores a deviation rather than an absolute
        // loading — refreshed from THIS sweep's β.
        let d = run_block(
            &delta_pair.by_feature.pos,
            &delta_pair.by_feature.partition,
            delta_pair.by_feature.partition_scale,
            &side_pb,
            &e_delta,
            Some(&e_beta),
            tracks.nested.then(|| b.final_z.clone()),
            z_delta.clone(),
            cfg,
            sweep,
            0xD317,
            "delta|beta",
        );
        e_delta.copy_from_slice(&d.mean_beta);
        z_delta = Some(d.final_z.clone());
        // Row intercepts, each from the block that last moved its track. The β block
        // carries two terms (spliced at 0, unspliced at 1), but δ ran after it, so an
        // unspliced row's current intercept is δ's; spliced rows take β's term 0.
        scatter_splice_bias_to_rows(&b.b_profiled, &d.b_profiled, tracks, feat.b_feat, &mut b_row_live);

        // pb | β, δ. The pb side sees the EFFECTIVE per-row loading: β on spliced
        // rows, β+δ on unspliced ones — which is what the model scores.
        scatter_splice_to_rows(&e_beta, &e_delta, tracks, feat.e_feat, h, &mut e_rows);
        let side_rows = FrozenSide {
            e: &e_rows,
            b: &b_row_live,
            h,
        };
        let mut pb_cfg = DimBlockConfig::new(1, 0, block_seed(cfg.seed, 0x85EB, sweep))
            .with_init_beta(e_pb.clone())
            .with_label("pb|beta,delta")
            .without_selection()
            .quiet();
        pb_cfg.transitions_per_dim = cfg.transitions_per_dim;
        let pbres = {
            let moments: Vec<AnchorMoment> = beta_pair
                .by_pb
                .pos
                .iter()
                .map(|p| AnchorMoment::new(p, &side_rows))
                .collect();
            let nodes: Vec<NodeTerm> = beta_pair
                .by_pb
                .pos
                .iter()
                .zip(&moments)
                .map(|(p, m)| {
                    NodeTerm::new(p, &beta_pair.by_pb.partition, beta_pair.by_pb.partition_scale)
                        .with_moment(m)
                })
                .collect();
            dim_block(&nodes, &side_rows, &pb_cfg)
        };
        e_pb.copy_from_slice(&pbres.mean_beta);
        b_pb_live.copy_from_slice(&pbres.b_profiled);

        if keep {
            acc.add(&b, &d, &pbres);
        }
        bar.inc(1);
    }
    bar.finish_and_clear();
    let mut out = acc.finish(h, delta_identified)?;

    // Intercepts against the posterior-mean sides, same reasoning as the bge path.
    let side_pb_final = FrozenSide {
        e: &out.mean_pb,
        b: &b_pb_live,
        h,
    };
    out.mean_b_feat = (0..n_genes)
        .map(|g| {
            let total: f64 = beta_pair.by_feature.pos[g]
                .iter()
                .chain(&delta_pair.by_feature.pos[g])
                .map(|&(_, n)| f64::from(n))
                .sum();
            profiled_bias(
                total,
                &out.beta_mean[g * h..(g + 1) * h],
                None,
                &beta_pair.by_feature.partition,
                beta_pair.by_feature.partition_scale,
                &side_pb_final,
            )
        })
        .collect();
    scatter_splice_to_rows(&out.beta_mean, &out.delta_mean, tracks, feat.e_feat, h, &mut e_rows);
    let side_rows_final = FrozenSide {
        e: &e_rows,
        b: &b_row_live,
        h,
    };
    out.mean_b_pb = (0..pb.n_pb_total())
        .map(|p| {
            let total: f64 = beta_pair.by_pb.pos[p].iter().map(|&(_, n)| f64::from(n)).sum();
            profiled_bias(
                total,
                &out.mean_pb[p * h..(p + 1) * h],
                None,
                &beta_pair.by_pb.partition,
                beta_pair.by_pb.partition_scale,
                &side_rows_final,
            )
        })
        .collect();
    Ok(out)
}

/// One gene-side block: build moments against the current frozen side, run a
/// single `dim_block` sweep, hand back the draw.
#[allow(clippy::too_many_arguments)]
fn run_block(
    pos: &[Vec<(u32, f32)>],
    partition: &[u32],
    partition_scale: f64,
    side: &FrozenSide<'_>,
    warm: &[f32],
    offset: Option<&[f32]>,
    z_allowed: Option<Vec<bool>>,
    init_z: Option<Vec<bool>>,
    cfg: &PbGibbsConfig,
    sweep: usize,
    salt: u64,
    label: &str,
) -> DimBlockResult {
    let h = side.h;
    let moments: Vec<AnchorMoment> = pos.iter().map(|p| AnchorMoment::new(p, side)).collect();
    let nodes: Vec<NodeTerm> = pos
        .iter()
        .zip(&moments)
        .enumerate()
        .map(|(a, (p, m))| {
            let mut n = NodeTerm::new(p, partition, partition_scale).with_moment(m);
            n.offset = offset.map(|o| &o[a * h..(a + 1) * h]);
            n
        })
        .collect();
    let mut bcfg = DimBlockConfig::new(1, 0, block_seed(cfg.seed, salt, sweep))
        .with_init_beta(warm.to_vec())
        .with_label(label)
        .quiet();
    bcfg.transitions_per_dim = cfg.transitions_per_dim;
    if let Some(mask) = z_allowed {
        bcfg = bcfg.with_z_allowed(mask);
    }
    if let Some(z) = init_z {
        bcfg = bcfg.with_init_z(z);
    }
    dim_block(&nodes, side, &bcfg)
}

/// Running sums for the splice sweep; kept in one place so the three blocks'
/// accumulators cannot drift out of step.
struct SpliceAcc {
    bp: Vec<f64>,
    bm: Vec<f64>,
    dp: Vec<f64>,
    dm: Vec<f64>,
    pbm: Vec<f64>,
    /// Per-OUTER-sweep hyper chains, one vec per dim. Sums would be enough for
    /// the posterior means, but not for `scalar_diagnostics` — and the inner
    /// blocks run a single sweep each, so their own chains hold one sample and
    /// report ESS = 1 unconditionally.
    bs2: Vec<Vec<f64>>,
    bpi: Vec<Vec<f64>>,
    ds2: Vec<Vec<f64>>,
    dpi: Vec<Vec<f64>>,
    n: usize,
}

impl SpliceAcc {
    fn new(n_genes: usize, n_pb: usize, h: usize) -> Self {
        Self {
            bp: vec![0.0; n_genes * h],
            bm: vec![0.0; n_genes * h],
            dp: vec![0.0; n_genes * h],
            dm: vec![0.0; n_genes * h],
            pbm: vec![0.0; n_pb * h],
            bs2: vec![Vec::new(); h],
            bpi: vec![Vec::new(); h],
            ds2: vec![Vec::new(); h],
            dpi: vec![Vec::new(); h],
            n: 0,
        }
    }

    fn add(&mut self, b: &DimBlockResult, d: &DimBlockResult, pb: &DimBlockResult) {
        let acc = |dst: &mut [f64], src: &[f32]| {
            for (a, v) in dst.iter_mut().zip(src) {
                *a += f64::from(*v);
            }
        };
        let push64 = |dst: &mut [Vec<f64>], src: &[f64]| {
            for (a, v) in dst.iter_mut().zip(src) {
                a.push(*v);
            }
        };
        acc(&mut self.bp, &b.pip);
        acc(&mut self.bm, &b.mean_beta);
        acc(&mut self.dp, &d.pip);
        acc(&mut self.dm, &d.mean_beta);
        acc(&mut self.pbm, &pb.mean_beta);
        push64(&mut self.bs2, &b.sigma2);
        push64(&mut self.bpi, &b.pi0);
        push64(&mut self.ds2, &d.sigma2);
        push64(&mut self.dpi, &d.pi0);
        self.n += 1;
    }

    fn finish(self, h: usize, delta_identified: Vec<bool>) -> anyhow::Result<SpliceGibbsResult> {
        anyhow::ensure!(self.n > 0, "gem splice Gibbs retained zero sweeps");
        let inv = 1.0 / self.n as f64;
        let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|&a| (a * inv) as f32).collect() };
        let f64s = |c: &[Vec<f64>]| -> Vec<f64> {
            c.iter().map(|v| v.iter().sum::<f64>() / v.len().max(1) as f64).collect()
        };
        let diag = |c: &[Vec<f64>]| -> Vec<ChainDiag> {
            c.iter().map(|v| scalar_diagnostics(v)).collect()
        };
        Ok(SpliceGibbsResult {
            beta_pip: f32s(&self.bp),
            beta_mean: f32s(&self.bm),
            delta_pip: f32s(&self.dp),
            delta_mean: f32s(&self.dm),
            delta_identified,
            mean_pb: f32s(&self.pbm),
            // Filled by the caller once the posterior means exist — they are
            // computed against those means, so they cannot be accumulated here.
            mean_b_feat: Vec::new(),
            mean_b_pb: Vec::new(),
            beta_sigma2: f64s(&self.bs2),
            beta_pi0: f64s(&self.bpi),
            delta_sigma2: f64s(&self.ds2),
            delta_pi0: f64s(&self.dpi),
            beta_sigma_diag: diag(&self.bs2),
            beta_pi0_diag: diag(&self.bpi),
            delta_sigma_diag: diag(&self.ds2),
            delta_pi0_diag: diag(&self.dpi),
            h,
            n_kept: self.n,
        })
    }
}

/// Effective per-row loading for the pb block: `β_g` on a spliced row, `β_g + δ_g`
/// on an unspliced one, and the phase-1 MAP for any row no gene claims.
fn scatter_splice_to_rows(
    e_beta: &[f32],
    e_delta: &[f32],
    tracks: &SpliceTracks<'_>,
    map_fallback: &[f32],
    h: usize,
    out: &mut [f32],
) {
    out.copy_from_slice(map_fallback);
    for (uid, (&g, &unspliced)) in tracks
        .row_to_gene
        .iter()
        .zip(tracks.unspliced_rows)
        .enumerate()
    {
        if g == u32::MAX {
            continue;
        }
        let (s, dst) = (g as usize * h, uid * h);
        for k in 0..h {
            out[dst + k] = e_beta[s + k] + if unspliced { e_delta[s + k] } else { 0.0 };
        }
    }
}

/// Scatter anchor loadings out to the feature-row axis, which is what the pb
/// block scores against.
///
/// Without a grouping the two axes are the same and this is a copy. With one,
/// every row mapped to an anchor takes that anchor's current draw, and every
/// dropped row keeps `map_fallback` — its phase-1 MAP. Dropped rows are not
/// sampled but they ARE observed, so they belong in the pb conditional.
/// Scatter each gate's profiled intercept onto the rows it describes.
///
/// `beta_b` is `[n_genes × 2]` — the β block runs over both tracks, spliced term first —
/// and `delta_b` is `[n_genes × 1]` from the δ block. A row takes δ's value when it is
/// unspliced and β's spliced term otherwise, because those are the blocks that last
/// moved each track. Rows no gene claims keep `fallback`, matching how
/// [`scatter_splice_to_rows`] treats loadings: dropping them would misspecify the pb
/// likelihood rather than merely omit them.
fn scatter_splice_bias_to_rows(
    beta_b: &[f32],
    delta_b: &[f32],
    tracks: &SpliceTracks<'_>,
    fallback: &[f32],
    out: &mut [f32],
) {
    out.copy_from_slice(fallback);
    for (row, &g) in tracks.row_to_gene.iter().enumerate() {
        let g = g as usize;
        out[row] = if tracks.unspliced_rows[row] {
            delta_b[g]
        } else {
            beta_b[g * 2]
        };
    }
}

fn scatter_to_rows(
    e_anchor: &[f32],
    anchors: Option<&AnchorMap<'_>>,
    map_fallback: &[f32],
    h: usize,
    out: &mut [f32],
) {
    match anchors {
        None => out.copy_from_slice(e_anchor),
        Some(a) => {
            out.copy_from_slice(map_fallback);
            for (uid, &an) in a.row_to_anchor.iter().enumerate() {
                if an == u32::MAX {
                    continue;
                }
                let (s, d) = (an as usize * h, uid * h);
                out[d..d + h].copy_from_slice(&e_anchor[s..s + h]);
            }
        }
    }
}

#[cfg(test)]
#[path = "pb_gibbs_tests.rs"]
mod pb_gibbs_tests;
