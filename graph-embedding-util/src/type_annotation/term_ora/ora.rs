//! Step 5 of the firm pipeline: the cluster × term over-representation test and its
//! calibration.
//!
//! The hypergeometric survival tables, the label-permutation null they are calibrated
//! against (pooled across clusters per term, capped by [`MAX_NULL_POOL`]), the BH-FDR call,
//! and the diagnostics that say whether the machinery itself is unbiased. All of it is
//! keyed off the fixed margins `(N, m_T, n_K)`, which is what lets one table serve the
//! observed count and every permuted one.

use super::TermOraConfig;
use crate::type_annotation::score::argmax_rows;
use crate::type_annotation::UNASSIGNED;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

/// Cluster × term over-representation result. All `[n_comm × c]` matrices are
/// row-major (`[k*c + t]`).
pub(super) struct OraResult {
    /// `−ln P(X≥a)` analytic hypergeometric statistic (larger = more enriched).
    pub(super) stat: Vec<f32>,
    /// Permutation-calibrated p (pooled per term across clusters).
    pub(super) p_perm: Vec<f32>,
    /// BH q of `p_perm`, per cluster row.
    pub(super) q: Vec<f32>,
    /// FDR-sparse row-softmax Q over significant terms (confidence weights). Only the reported
    /// run needs it — see [`Want`].
    pub(super) q_soft: Vec<f32>,
    /// Calibration diagnostics. `None` for a bootstrap replicate, which never reads them.
    pub(super) cal: Option<Calibration>,
}

pub(super) struct Calibration {
    pub(super) n_perm: usize,
    pub(super) median_logratio: f64,
    pub(super) frac_analytic_anticons: f64,
    pub(super) lambda_perm: f64,
    pub(super) ks_perm: f64,
    pub(super) degenerate_frac: f64,
}

/// How much of the ORA the caller actually intends to read.
///
/// A bootstrap replicate wants a *label*, and reads only `stat` and `q` on its way to one
/// ([`cluster_calls`]). It never looks at `q_soft` or `cal` — but `cal` is the most expensive
/// thing here by a wide margin (two sorts and an inverse-normal CDF over every one of the
/// `n_perm × n_comm × c` pooled null values), so computing it 200 times and discarding it 200
/// times costs more than the permutation it is meant to be calibrating.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum Want {
    /// Everything: the reported run, which writes `null_calibration.tsv` and the Q matrix.
    Report,
    /// The label only: one bootstrap replicate.
    CallOnly,
}

/// Cap on the pooled null per term. The permutation statistic is pooled **across clusters** (it
/// is relabeling-invariant), so the pool is `n_perm × n_comm` — and a runaway partition can make
/// that enormous for no gain: `--resolution 8` has produced 1,713 communities on 15k cells, where
/// the full `--num-perm 500` would build and sort 20M f32 per term. This is a **cost ceiling**,
/// not a precision target: at any resolution worth using (tens of clusters) it does nothing and
/// the user's `--num-perm` stands.
pub(super) const MAX_NULL_POOL: usize = 100_000;

/// Draws actually taken. Only ever *reduces* the caller's request; `null_calibration.tsv` records
/// what was used (`n_perm`), and [`super::annotate_inner`] says so on the console when it bites.
pub(super) fn capped_n_perm(requested: usize, n_comm: usize) -> usize {
    if requested == 0 || n_comm == 0 {
        return requested;
    }
    MAX_NULL_POOL.div_ceil(n_comm).clamp(1, requested)
}

/// `lnfact` must cover `0..=n` for the cell count `n` (it is reused across bootstrap replicates,
/// whose `n_tot` is always ≤ `n`).
pub(super) fn cluster_term_ora(
    assign: &[usize],
    community: &[usize],
    n_comm: usize,
    c: usize,
    lnfact: &[f64],
    want: Want,
    cfg: &TermOraConfig,
) -> OraResult {
    let n = assign.len();

    // Assigned cells only (post-QC) feed the contingency: an unassigned cell is out of the
    // hypergeometric population entirely, not a zero in it.
    let assigned: Vec<usize> = (0..n).filter(|&i| assign[i] != UNASSIGNED).collect();
    let labels: Vec<usize> = assigned.iter().map(|&i| assign[i]).collect();
    let comms: Vec<usize> = assigned.iter().map(|&i| community[i]).collect();
    let n_tot = assigned.len();

    let count = contingency(&comms, &labels, n_comm, c);
    let n_k: Vec<usize> = count
        .chunks(c)
        .map(|row| row.iter().map(|&v| v as usize).sum())
        .collect();
    let mut m_t = vec![0usize; c];
    for &t in &labels {
        m_t[t] += 1;
    }

    // Hypergeometric SF tables. The margins `(n_tot, m_t, n_k)` are fixed under permutation, so
    // each table serves the observed count and every permuted one.
    //
    // Tables are keyed by the margin *pair*, not by `(k, t)`: clusters that happen to share a
    // size share a table. At the resolutions worth using the sizes are all distinct and this
    // saves nothing, but it is what keeps a runaway partition (1,713 clusters, most of them
    // tiny and identically-sized) from building thousands of copies of the same table. `slot`
    // resolves the sharing once, so the lookup itself stays a plain index.
    let mut degrees: Vec<usize> = n_k.clone();
    degrees.sort_unstable();
    degrees.dedup();
    let slot: Vec<usize> = n_k
        .iter()
        .map(|nk| degrees.binary_search(nk).expect("n_k came from degrees"))
        .collect();
    let sf_tables: Vec<Vec<f64>> = (0..degrees.len() * c)
        .into_par_iter()
        .map(|st| hypergeom_sf_table(n_tot, m_t[st % c], degrees[st / c], lnfact))
        .collect();
    let sf_at = |k: usize, t: usize, a: usize| -> f64 {
        let tbl = &sf_tables[slot[k] * c + t];
        tbl.get(a)
            .copied()
            .unwrap_or(if tbl.is_empty() { 1.0 } else { 0.0 })
    };

    let mut p_analytic = vec![1f32; n_comm * c];
    let mut stat = vec![0f32; n_comm * c];
    for k in 0..n_comm {
        for t in 0..c {
            let p = sf_at(k, t, count[k * c + t] as usize).clamp(1e-12, 1.0);
            p_analytic[k * c + t] = p as f32;
            stat[k * c + t] = (-p.ln()) as f32;
        }
    }

    //////////////////////////////////////////////////////////
    // permutation null: pool stat across clusters per term //
    //////////////////////////////////////////////////////////
    // Serial by design. This whole function already runs inside a 200-way rayon fan-out over
    // bootstrap replicates (`marker_bootstrap::run_marker_bootstrap`), so the cores are spoken
    // for; a nested fan-out here would only contend. It also keeps the RNG a single chain, which
    // is what makes `--seed` mean something.
    let b = capped_n_perm(cfg.n_perm, n_comm);
    let mut null_pool: Vec<Vec<f32>> = vec![Vec::with_capacity(b * n_comm); c];
    if b > 0 && n_tot >= 2 {
        let mut perm = labels.clone();
        let mut rng = SmallRng::seed_from_u64(cfg.seed ^ 0x5eed_0a4a);
        for _ in 0..b {
            perm.shuffle(&mut rng);
            let cnt = contingency(&comms, &perm, n_comm, c);
            for k in 0..n_comm {
                for t in 0..c {
                    let p = sf_at(k, t, cnt[k * c + t] as usize).clamp(1e-12, 1.0);
                    null_pool[t].push((-p.ln()) as f32);
                }
            }
        }
    }
    null_pool.par_iter_mut().for_each(|pool| {
        pool.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    });

    // permutation p per (K,T): fraction of the term's pooled null ≥ observed.
    let mut p_perm = vec![1f32; n_comm * c];
    for kt in 0..n_comm * c {
        let pool = &null_pool[kt % c];
        p_perm[kt] = if pool.is_empty() {
            p_analytic[kt]
        } else {
            let ge = pool.len() - lower_bound(pool, stat[kt]);
            (ge as f32 + 1.0) / (pool.len() as f32 + 1.0)
        };
    }

    // BH q per cluster row on the permutation p.
    let mut q = vec![1f32; n_comm * c];
    for k in 0..n_comm {
        let row_q = matrix_util::hypothesis::benjamini_hochberg(&p_perm[k * c..(k + 1) * c]);
        q[k * c..(k + 1) * c].copy_from_slice(&row_q);
    }

    // The rest is for the reported run only — a replicate reads `stat` and `q` and stops.
    let (q_soft, cal) = match want {
        Want::CallOnly => (Vec::new(), None),
        Want::Report => (
            // FDR-sparse row-softmax Q (confidence weights): softmax of stat over terms
            // with q < α; zero elsewhere; uniform fallback if a row has no significant term.
            sparse_row_softmax(&stat, &q, n_comm, c, cfg.fdr_alpha, cfg.q_temperature),
            Some(calibrate(&p_analytic, &p_perm, &null_pool, n_comm, c, b)),
        ),
    };

    OraResult {
        stat,
        p_perm,
        q,
        q_soft,
        cal,
    }
}

/// Each cluster's FDR-gated call: its top over-represented term, kept only if significant.
/// [`UNASSIGNED`] when nothing survives `fdr_alpha`.
pub(super) fn cluster_calls(ora: &OraResult, n_comm: usize, c: usize, alpha: f32) -> Vec<usize> {
    let top = argmax_rows(&ora.stat, n_comm, c);
    (0..n_comm)
        .map(|k| {
            let best = top[k];
            if ora.q[k * c + best] < alpha {
                best
            } else {
                UNASSIGNED
            }
        })
        .collect()
}

/// `[n_comm × c]` row-major contingency counts over the **assigned cells only**: `comms` and
/// `labels` are the compacted, parallel per-assigned-cell arrays, so there is no sentinel to
/// filter and the walk is over `n_tot`, not `n`.
fn contingency(comms: &[usize], labels: &[usize], n_comm: usize, c: usize) -> Vec<u32> {
    let mut count = vec![0u32; n_comm * c];
    for (&k, &t) in comms.iter().zip(labels) {
        if k < n_comm && t < c {
            count[k * c + t] += 1;
        }
    }
    count
}

/// `ln(i!)` for `i ∈ 0..=n` (i.e. `ln_gamma(i+1)`), precomputed once so the
/// per-(cluster,term) SF tables share the factorials rather than recomputing
/// `ln_gamma` for every binomial coefficient.
pub(super) fn ln_factorials(n: usize) -> Vec<f64> {
    use statrs::function::gamma::ln_gamma;
    (0..=n).map(|i| ln_gamma(i as f64 + 1.0)).collect()
}

/// Upper-tail hypergeometric SF table: `sf[a] = P(X ≥ a)` for a draw of `draws`
/// from a population of `pop` with `succ` successes, `a ∈ 0..=min(succ,draws)`.
/// Log-space PMF for numerical stability. `lnfact[i] = ln(i!)` must cover
/// `0..=pop`. Empty when `pop==0`.
fn hypergeom_sf_table(pop: usize, succ: usize, draws: usize, lnfact: &[f64]) -> Vec<f64> {
    if pop == 0 || succ == 0 || draws == 0 {
        // No successes or no draws ⇒ a is always 0; P(X≥0)=1, P(X≥1)=0.
        return vec![1.0];
    }
    let lnc = |a: usize, b: usize| -> f64 {
        if b > a {
            return f64::NEG_INFINITY;
        }
        lnfact[a] - lnfact[b] - lnfact[a - b]
    };
    let x_hi = succ.min(draws);
    let x_lo = (draws + succ).saturating_sub(pop);
    let ln_den = lnc(pop, draws);
    let mut pmf = vec![0f64; x_hi + 1];
    for (x, p) in pmf.iter_mut().enumerate().take(x_hi + 1).skip(x_lo) {
        *p = (lnc(succ, x) + lnc(pop - succ, draws - x) - ln_den).exp();
    }
    let mut sf = vec![0f64; x_hi + 1];
    let mut acc = 0f64;
    for a in (0..=x_hi).rev() {
        acc += pmf[a];
        sf[a] = acc.min(1.0);
    }
    sf
}

/// Index of the first element ≥ `x` in a sorted slice (count of strictly-smaller).
fn lower_bound(sorted: &[f32], x: f32) -> usize {
    sorted.partition_point(|&v| v < x)
}

/// Index of the first element > `x` in a sorted slice (count of ≤ `x`).
fn upper_bound(sorted: &[f32], x: f32) -> usize {
    sorted.partition_point(|&v| v <= x)
}

/// Per cluster row: softmax of `stat/τ` over terms with `q < α`, zero elsewhere.
/// Rows with no significant term get a uniform distribution (so the argmax
/// confidence is still defined, but small).
fn sparse_row_softmax(
    stat: &[f32],
    q: &[f32],
    n_comm: usize,
    c: usize,
    alpha: f32,
    temperature: f32,
) -> Vec<f32> {
    let tau = temperature.max(1e-6);
    let mut out = vec![0f32; n_comm * c];
    for k in 0..n_comm {
        let sig: Vec<usize> = (0..c).filter(|&t| q[k * c + t] < alpha).collect();
        if sig.is_empty() {
            let u = 1.0 / c as f32;
            for t in 0..c {
                out[k * c + t] = u;
            }
            continue;
        }
        let mx = sig
            .iter()
            .map(|&t| stat[k * c + t])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0f32;
        for &t in &sig {
            let e = ((stat[k * c + t] - mx) / tau).exp();
            out[k * c + t] = e;
            s += e;
        }
        let s = s.max(1e-12);
        for &t in &sig {
            out[k * c + t] /= s;
        }
    }
    out
}

/// Discreteness-robust calibration of the analytic hypergeometric vs the
/// permutation null. `median_logratio = median log10(p_perm/p_analytic)` (≈0
/// calibrated; >0 ⇒ analytic anticonservative); `frac_analytic_anticons` =
/// share with `p_analytic < ½·p_perm`. Machinery sanity: `lambda_perm` /
/// `ks_perm` on leave-one-out null p (≈1 / small when unbiased).
fn calibrate(
    p_analytic: &[f32],
    p_perm: &[f32],
    null_pool: &[Vec<f32>],
    n_comm: usize,
    c: usize,
    b: usize,
) -> Calibration {
    // analytic-vs-perm agreement over observed (K,T).
    let mut logratios: Vec<f64> = Vec::with_capacity(n_comm * c);
    let mut anticons = 0usize;
    for kt in 0..n_comm * c {
        let pa = p_analytic[kt].max(1e-12) as f64;
        let pp = p_perm[kt].max(1e-12) as f64;
        logratios.push((pp / pa).log10());
        if (pa) < 0.5 * pp {
            anticons += 1;
        }
    }
    let median_logratio = {
        let mut v = logratios.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() {
            0.0
        } else {
            v[v.len() / 2]
        }
    };
    let frac_analytic_anticons = anticons as f64 / (n_comm * c).max(1) as f64;

    // Degenerate fraction: terms whose pooled null has no spread.
    let degenerate_terms = null_pool
        .iter()
        .filter(|pool| pool.is_empty() || pool.first() == pool.last())
        .count();
    let degenerate_frac = degenerate_terms as f64 / c.max(1) as f64;

    // Machinery sanity: leave-one-out empirical p of every pooled null value vs its term's pool
    // (uniform under an unbiased permutation null).
    //
    // **mid-p, not the plain tail count.** The statistic is discrete and the pool is full of
    // ties at the floor — a (group, term) with no enriched cells scores exactly 0, and once the
    // assignment is sparse (as it is after the bootstrap abstains on half the cells) that is
    // most of the pool. The plain tail `#{≥x}/m` hands every one of those p = 1, which the
    // clamp turns into z = Φ⁻¹(1e-12) = −7.03 and λ = 49.48/0.4549 = **108.77** — a number that
    // is a property of the clamp, not of the null, and that used to fire a "raise --num-perm"
    // warning no amount of permutation could ever fix. Splitting the ties (`#{>x} + ½#{=x}`)
    // makes λ mean what it claims to again, and leaves the continuous case unchanged.
    let mut loo: Vec<f64> = Vec::new();
    for pool in null_pool {
        let m = pool.len();
        if m < 2 {
            continue;
        }
        for &x in pool {
            let lo = lower_bound(pool, x); // strictly below
            let hi = upper_bound(pool, x); // ≤ x
            let p = (m as f64 - 0.5 * (lo + hi) as f64) / m as f64;
            loo.push(p.clamp(1e-12, 1.0));
        }
    }
    let (lambda_perm, ks_perm) = if loo.len() >= 8 {
        (lambda_from_p(&loo), ks_uniform(&loo))
    } else {
        (f64::NAN, f64::NAN)
    };

    Calibration {
        n_perm: b,
        median_logratio,
        frac_analytic_anticons,
        lambda_perm,
        ks_perm,
        degenerate_frac,
    }
}

/// Genomic-inflation-style λ for one-sided p-values: `median(z²)/0.4549`,
/// `z = Φ⁻¹(1−p)`. ≈1 when p ~ Uniform.
fn lambda_from_p(ps: &[f64]) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let std = Normal::new(0.0, 1.0).unwrap();
    let mut zsq: Vec<f64> = ps
        .iter()
        .map(|&p| {
            let z = std.inverse_cdf((1.0 - p).clamp(1e-12, 1.0 - 1e-12));
            z * z
        })
        .collect();
    zsq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = zsq[zsq.len() / 2];
    med / 0.4549364
}

/// Kolmogorov–Smirnov distance of `ps` from Uniform(0,1).
fn ks_uniform(ps: &[f64]) -> f64 {
    let mut v = ps.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nf = v.len() as f64;
    let mut d = 0f64;
    for (i, &p) in v.iter().enumerate() {
        let lo = (i as f64) / nf;
        let hi = (i as f64 + 1.0) / nf;
        d = d.max((p - lo).abs()).max((hi - p).abs());
    }
    d
}
