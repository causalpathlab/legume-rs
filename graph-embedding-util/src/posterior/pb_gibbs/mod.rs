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
//! Phase 1's objective is a SUM over axes with `λ = 1` per level, i.e. the
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
//! samplers rely on. Between blocks the other side has moved, so the per-anchor moment
//! statistics have to be rebuilt; that happens inside the column pass, per tile and in
//! parallel, so this file does not build them. It used to, and they were never read —
//! the only consumer of `NodeTerm::moment` is `multinomial_ll`, and the column path's
//! one caller of that deliberately walks the edges instead.

mod hyper_io;
mod shared;
mod splice;

use super::diagnostics::scalar_diagnostics;
use super::diagnostics::ChainDiag;
use super::dim_block::{dim_block, DimBlockConfig};
use super::lnpdf::{FrozenSide, NodeTerm};
use super::pb_index::{build_pb_index_pair, AnchorMap, FeatureSide};
use crate::fit::stacked_pb::StackedPb;
use crate::progress::new_progress_bar;
use log::info;
use shared::{block_seed, profiled_bias, report_convergence, scatter_to_rows, warm_start_genes};

pub use hyper_io::{write_posterior_hyper_from_model, write_splice_posterior_hyper};
pub(crate) use splice::pb_gibbs_splice;
pub use splice::{SpliceGibbsResult, SpliceTracks};

/// Which likelihood term of gem's β block describes which splice track.
///
/// The β block is deliberately built over BOTH tracks — β appears in the unspliced rows
/// too, so a spliced-only kernel would not be the conditional `p(β | δ, ·)`. That makes
/// "term 0" and "term 1" a contract between where the terms are constructed and everything
/// that reads the block's per-term output. Naming it keeps the contract in one place
/// instead of as a bare stride at each use site.
struct BetaTerm;

impl BetaTerm {
    /// The gene's spliced rows, scoring `⟨β, e_p⟩`.
    const SPLICED: usize = 0;
    /// The gene's unspliced rows, scoring `⟨β + δ, e_p⟩` with δ carried as the offset.
    const UNSPLICED: usize = 1;
    /// How many terms the β block carries. The δ block carries one.
    const COUNT: usize = 2;
}

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
    /// `Some(α)` puts the per-dim inclusion rates under a TRUNCATED IBP
    /// (stick-breaking) instead of an independent `Beta(a,b)` per dim. See
    /// [`super::dim_block::DimBlockConfig::stick_alpha`].
    pub stick_alpha: Option<f64>,
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
            stick_alpha: Some(super::dim_block::DEFAULT_STICK_ALPHA),
        }
    }
}

/// What each block's Poisson rate normalizer actually summed over.
///
/// Reported rather than inferred from the config, because since the slate became
/// data-dependent — exact where an axis is cheap relative to the cap, sampled where it is
/// not, and drawn from the EXPRESSED axis either way — the cap no longer says what a run
/// did. A `scale` of `1.0` means that axis was summed exactly and so carries no
/// Monte-Carlo error in its log-normalizer; anything above it does.
#[derive(Clone, Copy, Debug)]
pub struct PartitionGeometry {
    /// Entries summed in the gene block's normalizer (over the pb axis), and the factor
    /// folding them up to the expressed pb count.
    pub pb_entries: usize,
    pub pb_scale: f64,
    /// Entries summed in the pb block's normalizer (over the feature axis), and its
    /// fold-up factor.
    pub feat_entries: usize,
    pub feat_scale: f64,
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
    /// What the normalizers actually summed over — see [`PartitionGeometry`].
    pub partition: PartitionGeometry,
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
    // Summed over both blocks and every sweep: a fallback anywhere is a coordinate
    // that did not move, and the ratio is what says whether the run sampled at all.
    let (mut fallbacks, mut n_transitions) = (0usize, 0usize);

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
        let nodes: Vec<NodeTerm> = pair
            .by_feature
            .pos
            .iter()
            .map(|p| {
                NodeTerm::new(
                    p,
                    &pair.by_feature.partition,
                    pair.by_feature.partition_scale,
                )
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
        // The GENE block is the one that selects, so it is the one the inclusion prior
        // reaches. (The pb block below runs `without_selection`, so a prior there would
        // have nothing to act on.)
        gene_cfg.stick_alpha = cfg.stick_alpha;
        let g = dim_block(&nodes, &side_pb, &gene_cfg);

        // The gene side hands the pb block its EFFECTIVE loading `z·β`: a dim the
        // gate turned off must contribute nothing to what the pseudobulks are
        // fit against, or the pb side would be conditioning on a model the gene
        // side has already rejected. `z` is carried alongside it so the next
        // sweep resumes the chain rather than restarting cold.
        e_gene.copy_from_slice(&g.mean_beta);
        z_gene = Some(g.final_z.clone());
        fallbacks += g.fallbacks;
        n_transitions += g.n_transitions;
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
        let pb_nodes: Vec<NodeTerm> = pair
            .by_pb
            .pos
            .iter()
            .map(|p| NodeTerm::new(p, &pair.by_pb.partition, pair.by_pb.partition_scale))
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
        fallbacks += p.fallbacks;
        n_transitions += p.n_transitions;

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
    let sigma_diag: Vec<ChainDiag> = sigma2_chain.iter().map(|c| scalar_diagnostics(c)).collect();
    let pi0_diag: Vec<ChainDiag> = pi0_chain.iter().map(|c| scalar_diagnostics(c)).collect();
    report_convergence(
        "pb Gibbs",
        &sigma_diag,
        &pi0_diag,
        Some((fallbacks, n_transitions)),
    );

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
            let total: f64 = pair.by_feature.pos[a]
                .iter()
                .map(|&(_, n)| f64::from(n))
                .sum();
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
        sigma_diag,
        pi0_diag,
        partition: PartitionGeometry {
            pb_entries: pair.by_feature.partition.len(),
            pb_scale: pair.by_feature.partition_scale,
            feat_entries: pair.by_pb.partition.len(),
            feat_scale: pair.by_pb.partition_scale,
        },
        h,
        n_kept,
    })
}

#[cfg(test)]
mod pb_gibbs_tests;
