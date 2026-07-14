//! The marker-panel **stability bootstrap** for the raw-count enrichment path.
//!
//! Resample every celltype's marker panel with replacement, re-walk the enrichment score, re-call
//! the FDR, and ship the consensus — so a cluster's celltype call carries the fraction of
//! resamples that agreed on it, and a call that cannot hold up across them abstains rather than
//! being printed. The embedding-path twin is
//! `graph_embedding_util::type_annotation::marker_bootstrap`; the vote-tally half
//! ([`crate::consensus`]) is literally the same code.
//!
//! # What the support means here — read this before using the number
//!
//! On the enrichment path `cell_membership_nk` is **one-hot**, so
//! `posterior = cell_membership · Q` gives every cell in cluster *k* the identical row `Q[k, ·]`.
//! **A cell's label is exactly its cluster's label.** There is no per-cell decision to bootstrap,
//! so the statistic here is **per-cluster**, broadcast to the cells. It is written out as
//! `cluster_label_support`, and it is *not* the per-cell `label_support` the embedding path
//! reports.
//!
//! The embedding path additionally re-derives its clustering inside every draw (holding the
//! partition fixed costs it AUC 0.93 → 0.69 at separating spurious calls). We cannot: a fresh
//! partition changes `cluster_labels`, which changes the pseudobulk profile, which means
//! re-streaming the raw counts off the backend — 200 disk sweeps is not a bootstrap, it is 200
//! re-runs. So this bootstrap sees the variance contributed by the **panel** and not by the
//! **partition**, and its support is correspondingly optimistic. It still does the job it is here
//! for: it kills the cross-celltype winner's curse within each cluster, replaces a softmaxed test
//! statistic with a resampling frequency, and abstains on instability.
//!
//! # The one piece of real statistics: matching the null to the draw
//!
//! `build_q_matrix` picks a cluster's winner by `argmax_c es_std[k, c]`, so the **restandardization
//! is the decision variable**, and it is the only thing making celltypes with different panel sizes
//! comparable. Resampling with replacement changes two things every draw:
//!
//! * the **effective size shrinks** — a with-replacement draw of `m` items covers only
//!   `≈ 0.632·m` distinct genes, so the miss step changes; and
//! * the **weight multiset disperses** — multiplicities 0, 1, 2, 3… concentrate the walk on fewer
//!   genes, and that dispersion scales as `1/√|M_c|`, so it hits *small panels hardest*.
//!
//! and a third holds even before any resampling:
//!
//! * the **abundance profile** — a uniform draw over all genes is ~30% undetected genes, which sort
//!   to the bottom of every ranking and can never be enriched, so it is trivially easy to beat, and
//!   easy to beat by an amount that differs per celltype. See [`crate::gene_strata`].
//!
//! If the null tracked none of the three, small panels and well-expressed panels would get
//! systematically inflated `es_std` and the bootstrap would **manufacture the very winner's curse
//! it exists to remove**. So [`null_moments`] scatters *the draw's own weight multiset* onto a
//! random gene set of *the draw's own size* drawn *within the draw's own abundance strata*. The
//! `the_null_tracks_*` tests are the regression guards, and they are the reason that function
//! exists.
//!
//! # The one approximation
//!
//! The sample-permutation null is **frozen**, not redrawn per bootstrap draw: redrawing it would
//! be `b_perm × B × K × C` ES walks (~10¹² ops), which is not affordable. Its pooled entries are
//! already self-standardized, so their location and scale carry no panel dependence — only the
//! pool's *shape* does, and a bootstrap resamples the same genes, so the shape moves little.
//! The exact escape hatch is real and cheap: with `num_sample_perm == 0` the p-value comes from
//! row-randomization counts, which this module recomputes **exactly** per draw at no extra cost
//! (the restandardization walks already produce them). So `--num-perm 0` is a fully exact
//! configuration.

use crate::consensus::{
    keyed_rng, summarize_consensus, top_two, Abstain, AbstainConfig, Consensus, MIN_LIVE_MARKERS,
    UNASSIGNED,
};
use crate::es::weighted_ks_es;
use crate::fdr::bh_fdr;
use crate::gene_strata::GeneStrata;
use crate::q_matrix::build_q_matrix;
use crate::Mat;
use anyhow::Result;
use matrix_util::stop::par_replicates;
use rand::RngExt;

#[cfg(test)]
mod tests;

/// Domain separators, so the panel resample and its null never walk the same stream as each other
/// or as the row-randomization (`1_000_003`), row-rand-p (`3_000_003`), and sample-permutation
/// (`2_000_003`) streams already driven by `--seed` in [`crate::orchestrate`].
const BOOT_PANEL_STREAM: u64 = 0xB007_5748_9C0D_5748;
const BOOT_NULL_STREAM: u64 = 0x4E11_B007_5748_0F17;

/// Tunables for [`run_cluster_bootstrap`].
#[derive(Clone, Debug)]
pub struct EnrichmentBootstrapConfig {
    /// Bootstrap resamples of the marker panel.
    pub n_boot: usize,
    /// When a cluster's call is allowed to stand.
    pub abstain: Abstain,
    /// Coverage of the reported credible set (see [`Consensus::label_set`]).
    pub set_coverage: f32,
    /// Largest credible set worth printing.
    pub max_set_size: usize,
    /// Random gene sets drawn **per bootstrap draw, per celltype** for the restandardization
    /// moments. This is the cost centre of the whole routine (`n_boot × C × boot_num_draws × K` ES
    /// walks), and the moments only need ~10% relative accuracy on the SD, so it defaults to 100
    /// rather than the observed path's 1000.
    pub boot_num_draws: usize,
}

impl Default for EnrichmentBootstrapConfig {
    fn default() -> Self {
        Self {
            n_boot: 200,
            abstain: Abstain::Support(0.5),
            set_coverage: 0.8,
            max_set_size: 3,
            boot_num_draws: 100,
        }
    }
}

/// What the bootstrap learned, indexed by **cluster** (not by cell).
#[derive(Debug)]
pub struct ClusterBootstrap {
    /// Replicates that actually completed. **Not** `n_boot` — an interrupt leaves this smaller,
    /// and it is the denominator of every share below.
    pub n_draws: usize,
    /// Number of celltypes (so the `unassigned` column of `consensus.post` is at index `c`).
    pub c: usize,
    /// The per-cluster consensus: label, credible set, support, entropy.
    pub consensus: Consensus,
    /// `K × C` SD of `es_std` across the draws — the jitter in the decision variable itself.
    pub es_std_sd: Mat,
    /// Per cluster: the median `top − runner-up` gap in `es_std` across draws. Compare it against
    /// the corresponding `es_std_sd`: a cluster whose leaders are separated by less than the noise
    /// in the score is being called by chance.
    pub decision_gap: Vec<f32>,
    /// Per celltype: markers with weight > 0 (i.e. after any empirical-specificity reweighting,
    /// which can zero a marker outright).
    pub n_live: Vec<usize>,
    /// Per celltype: `n_live >= MIN_LIVE_MARKERS`. An unusable celltype never competes.
    pub usable: Vec<bool>,
}

/// One replicate's verdict.
struct Draw {
    /// Per cluster: the winning celltype, or [`UNASSIGNED`] when no celltype survived the FDR.
    winner: Vec<usize>,
    /// `K × C` restandardized ES, row-major.
    es_std: Vec<f32>,
}

/// Resample the marker panels, re-score, and ship the consensus.
///
/// * `ranked_per_k` — the observed per-cluster gene ranking. **Panel-independent** (it is a
///   function of the cluster profile alone), so it is computed once by the caller and reused across
///   every draw. That is what makes the ES resampling itself nearly free.
/// * `pooled_null` — per celltype, the frozen sample-permutation pool of self-standardized z's,
///   **sorted ascending**. Pass an empty slice to use exact row-randomization counts instead (the
///   `num_sample_perm == 0` path).
#[allow(clippy::too_many_arguments)]
pub fn run_cluster_bootstrap(
    ranked_per_k: &[Vec<u32>],
    markers_gc: &Mat,
    strata: &GeneStrata,
    pooled_null: &[Vec<f32>],
    g: usize,
    fdr_alpha: f32,
    q_temperature: f32,
    seed: u64,
    cfg: &EnrichmentBootstrapConfig,
) -> Result<ClusterBootstrap> {
    let k = ranked_per_k.len();
    let c = markers_gc.ncols();
    anyhow::ensure!(markers_gc.nrows() == g, "markers G dim mismatch");
    anyhow::ensure!(cfg.n_boot > 0, "bootstrap requires n_boot > 0");

    // The live panel: markers that still carry weight. A marker zeroed by the empirical-specificity
    // reweighting is not evidence, so it must not be resampled as though it were.
    let live: Vec<Vec<(u32, f32)>> = (0..c)
        .map(|cc| {
            (0..g)
                .filter(|&gi| markers_gc[(gi, cc)] > 0.0)
                .map(|gi| (gi as u32, markers_gc[(gi, cc)]))
                .collect()
        })
        .collect();
    let n_live: Vec<usize> = live.iter().map(Vec::len).collect();
    let usable: Vec<bool> = n_live.iter().map(|&m| m >= MIN_LIVE_MARKERS).collect();

    for (cc, &ok) in usable.iter().enumerate() {
        if !ok {
            log::warn!(
                "celltype {cc} has {} live marker(s) (< {MIN_LIVE_MARKERS}): it cannot be \
                 bootstrapped and will not compete. Resampling a one-element panel always returns \
                 that same element, so such a type would come back perfectly stable — the opposite \
                 of the truth.",
                n_live[cc]
            );
        }
    }
    anyhow::ensure!(
        usable.iter().any(|&u| u),
        "no celltype has {MIN_LIVE_MARKERS} or more live markers — every panel is empty or a \
         singleton after reweighting. Nothing can be annotated; check that the marker file's gene \
         names match the data's."
    );

    let draws: Vec<Draw> = par_replicates(cfg.n_boot, "marker bootstrap", |b| {
        one_draw(
            b,
            ranked_per_k,
            &live,
            &usable,
            strata,
            pooled_null,
            g,
            k,
            c,
            fdr_alpha,
            q_temperature,
            seed,
            cfg,
        )
    })?;
    let n_draws = draws.len();

    // Tally the winners: `[k × (c+1)]`, last column `unassigned`.
    let mut post = vec![0f32; k * (c + 1)];
    for d in &draws {
        for (kk, &w) in d.winner.iter().enumerate() {
            let col = if w == UNASSIGNED { c } else { w };
            post[kk * (c + 1) + col] += 1.0;
        }
    }
    let consensus = summarize_consensus(
        post,
        k,
        c + 1,
        n_draws,
        &AbstainConfig {
            abstain: cfg.abstain,
            set_coverage: cfg.set_coverage,
            max_set_size: cfg.max_set_size,
        },
    );

    // Jitter in the decision variable, and how far the winner actually cleared the runner-up.
    let mut es_std_sd = Mat::zeros(k, c);
    let nb = n_draws.max(1) as f32;
    for kk in 0..k {
        for cc in 0..c {
            let idx = kk * c + cc;
            let mean = draws.iter().map(|d| d.es_std[idx]).sum::<f32>() / nb;
            let var = draws
                .iter()
                .map(|d| (d.es_std[idx] - mean).powi(2))
                .sum::<f32>()
                / nb.max(1.0);
            es_std_sd[(kk, cc)] = var.sqrt();
        }
    }
    let decision_gap: Vec<f32> = (0..k)
        .map(|kk| {
            let mut gaps: Vec<f32> = draws
                .iter()
                .map(|d| {
                    let row = &d.es_std[kk * c..(kk + 1) * c];
                    let (_, p1, p2) = top_two(row);
                    p1 - p2
                })
                .collect();
            gaps.sort_by(f32::total_cmp);
            gaps.get(gaps.len() / 2).copied().unwrap_or(0.0)
        })
        .collect();

    Ok(ClusterBootstrap {
        n_draws,
        c,
        consensus,
        es_std_sd,
        decision_gap,
        n_live,
        usable,
    })
}

/// One bootstrap replicate: resample every panel, re-score, re-call, and report each cluster's
/// winner.
#[allow(clippy::too_many_arguments)]
fn one_draw(
    b: usize,
    ranked_per_k: &[Vec<u32>],
    live: &[Vec<(u32, f32)>],
    usable: &[bool],
    strata: &GeneStrata,
    pooled_null: &[Vec<f32>],
    g: usize,
    k: usize,
    c: usize,
    fdr_alpha: f32,
    q_temperature: f32,
    seed: u64,
    cfg: &EnrichmentBootstrapConfig,
) -> Result<Draw> {
    let mut es_std = Mat::zeros(k, c);
    let mut pvalue = Mat::zeros(k, c);

    // Scratch, reused across celltypes within this replicate.
    let mut hit_buf = vec![0f32; g];
    let mut scratch = strata.scratch();
    let mut drawn: Vec<(u32, f32)> = Vec::new();

    for cc in 0..c {
        if !usable[cc] {
            // Never competes: a zero ES cannot clear `build_q_matrix`'s `es_std > 0` gate, and a
            // p of 1 cannot clear the FDR. Belt and braces, deliberately.
            for kk in 0..k {
                es_std[(kk, cc)] = 0.0;
                pvalue[(kk, cc)] = 1.0;
            }
            continue;
        }

        // --- resample this panel with replacement -------------------------------------------
        let lm = &live[cc];
        let m = lm.len();
        let mut rng = keyed_rng(seed ^ BOOT_PANEL_STREAM, b, cc as u64);
        let mut touched: Vec<u32> = Vec::with_capacity(m);
        for _ in 0..m {
            let (gi, w) = lm[rng.random_range(0..m)];
            if hit_buf[gi as usize] == 0.0 {
                touched.push(gi);
            }
            // A gene drawn twice takes a double-size hit step. That IS the bootstrap weight —
            // `weighted_ks_es` needs no change to handle it.
            hit_buf[gi as usize] += w;
        }
        // The draw's own (gene, summed weight) multiset, in the draw's own effective size.
        let resampled: Vec<(u32, f32)> = touched
            .iter()
            .map(|&gi| (gi, hit_buf[gi as usize]))
            .collect();

        // --- observed ES for this draw --------------------------------------------------------
        let es_col: Vec<f32> = (0..k)
            .map(|kk| weighted_ks_es(&ranked_per_k[kk], &hit_buf))
            .collect();

        // Reset only what we touched — `hit_buf.fill(0)` over G would dominate the loop.
        for &gi in &touched {
            hit_buf[gi as usize] = 0.0;
        }

        // --- the null, matched to THIS draw's size, weights AND abundance profile --------------
        let profile = strata.profile_of(&resampled);
        let mut null_rng = keyed_rng(seed ^ BOOT_NULL_STREAM, b, cc as u64);
        let (mean, sd, count_ge) = null_moments(
            ranked_per_k,
            strata,
            &profile,
            resampled.len(),
            &es_col,
            cfg.boot_num_draws,
            &mut hit_buf,
            &mut scratch,
            &mut drawn,
            &mut null_rng,
        );

        for kk in 0..k {
            es_std[(kk, cc)] = (es_col[kk] - mean[kk]) / sd[kk];
            pvalue[(kk, cc)] = match pooled_null.get(cc) {
                // Frozen sample-permutation pool (sorted): the tail above this draw's z.
                Some(p) if !p.is_empty() => {
                    let n_ge = p.len() - lower_bound(p, es_std[(kk, cc)]);
                    (n_ge as f32 + 1.0) / (p.len() as f32 + 1.0)
                }
                // Exact row-randomization p — free, the null walks already counted it.
                _ => (count_ge[kk] as f32 + 1.0) / (cfg.boot_num_draws as f32 + 1.0),
            };
        }
    }

    // BH per cluster row, then the same FDR-sparse Q the observed path builds.
    let mut qvalue = Mat::zeros(k, c);
    for kk in 0..k {
        let row_p: Vec<f32> = (0..c).map(|cc| pvalue[(kk, cc)]).collect();
        let row_q = bh_fdr(&row_p);
        for cc in 0..c {
            qvalue[(kk, cc)] = row_q[cc];
        }
    }
    let q_mat = build_q_matrix(&es_std, &qvalue, fdr_alpha, q_temperature);

    // A cluster's verdict: the top surviving celltype, or no call at all.
    let winner: Vec<usize> = (0..k)
        .map(|kk| {
            let mut best = UNASSIGNED;
            let mut top = 0f32;
            for cc in 0..c {
                if q_mat[(kk, cc)] > top {
                    top = q_mat[(kk, cc)];
                    best = cc;
                }
            }
            best
        })
        .collect();

    Ok(Draw {
        winner,
        es_std: (0..k)
            .flat_map(|kk| (0..c).map(move |cc| (kk, cc)))
            .map(|(kk, cc)| es_std[(kk, cc)])
            .collect(),
    })
}

/// Efron–Tibshirani restandardization moments, **matched on the draw's effective size, its weight
/// multiset, and its abundance profile** — see the module doc for why the first two may not be
/// dropped, and [`crate::gene_strata`] for the third.
///
/// `profile` is the resampled draw's own abundance profile (per stratum, the weights it put there),
/// so the null reproduces all three at once: same number of genes, same weights, same abundance
/// distribution. Walks the ES against every cluster's ranking `b_rand` times, and counts for free
/// how often the null beat `es_obs` — the exact row-randomization p-value when there is no
/// permutation pool to score against.
///
/// `hit_buf` and `scratch` are caller-owned (`hit_buf` must arrive all-zero and leaves all-zero) so
/// a 200-draw bootstrap does not allocate `G` floats a million times.
#[allow(clippy::too_many_arguments)]
fn null_moments(
    ranked_per_k: &[Vec<u32>],
    strata: &GeneStrata,
    profile: &[Vec<f32>],
    n_hit: usize,
    es_obs: &[f32],
    b_rand: usize,
    hit_buf: &mut [f32],
    scratch: &mut [Vec<u32>],
    drawn: &mut Vec<(u32, f32)>,
    rng: &mut impl RngExt,
) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    let k = ranked_per_k.len();
    let mut mean = vec![0f32; k];
    let mut m2 = vec![0f32; k];
    let mut count_ge = vec![0u32; k];

    if n_hit == 0 || b_rand == 0 {
        // Nothing to standardize against; an SD of 1 leaves the raw ES untouched rather than
        // dividing by zero.
        return (mean, vec![1.0; k], count_ge);
    }

    for draw in 0..b_rand {
        strata.draw_matched(profile, scratch, drawn, rng);
        for &(gi, w) in drawn.iter() {
            hit_buf[gi as usize] = w;
        }
        for kk in 0..k {
            let es = weighted_ks_es(&ranked_per_k[kk], hit_buf);
            if es >= es_obs[kk] {
                count_ge[kk] += 1;
            }
            let prev = mean[kk];
            let new = prev + (es - prev) / (draw as f32 + 1.0);
            m2[kk] += (es - prev) * (es - new);
            mean[kk] = new;
        }
        for &(gi, _) in drawn.iter() {
            hit_buf[gi as usize] = 0.0;
        }
    }

    let bf = b_rand as f32;
    let sd: Vec<f32> = m2
        .iter()
        .map(|&v| {
            let var = if bf > 1.0 { v / (bf - 1.0) } else { 0.0 };
            var.sqrt().max(1e-8)
        })
        .collect();
    (mean, sd, count_ge)
}

/// First index of `sorted` whose value is `>= x` (the pool is ascending).
fn lower_bound(sorted: &[f32], x: f32) -> usize {
    sorted.partition_point(|&v| v < x)
}
