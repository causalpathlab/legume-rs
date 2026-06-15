//! Empirical-Bayes "null vs signal" call for embedding-based routines.
//!
//! A model that fits embeddings (gem β_g / e_cell, bge feature/cell vectors,
//! topic loadings, …) leaves *null* items at their random init: an item the
//! model never moved has `v ~ N(0, σ²I)`, so `‖v‖²` follows a scaled χ². The
//! old per-item χ² QC asserted σ at the init value *and* the nominal dof `h`;
//! both are wrong in practice. AdamW + weight decay shrink the null items
//! below init (so σ must be **estimated**), and the `h` embedding coordinates
//! are **correlated**, so a null item's `‖v‖²` is over-dispersed relative to
//! `χ²_h` — it behaves like `σ²·χ²_ν` with an *effective* dof `ν ≪ h`
//! (Satterthwaite: `ν = (Σλ_i)² / Σλ_i²` for the coordinate-covariance
//! eigenvalues `λ_i`). Forcing `ν = h` makes the upper-tail p-values
//! anti-conservative and over-calls "live".
//!
//! So we fit the null `σ²·χ²_ν` — **both** the scale `σ̂²` and the effective
//! dof `ν̂` — from the *lower* quantiles of the statistic (signal only inflates
//! the upper tail, so the lower tail is null-pure), decoupled from the
//! significance call to avoid a feedback loop. Then each item gets a `χ²_ν`
//! upper-tail p-value, a Storey π̂₀, a BH q-value, and is called **live** (kept)
//! when `q ≤ fdr` — demonstrable signal above the estimated null — else
//! **null** (dropped). This is the ashr spirit (flexible null, point mass at
//! 0, π̂₀ from the data) made cheap for the collapsed-norm statistic, where the
//! per-coordinate standard errors ashr would use are folded into `ν̂`.
//!
//! Two entry points: [`chi2_null_call`] on a precomputed scaled-χ² statistic
//! vector (the reusable core), and [`embedding_null_call`] for the common case
//! where the statistic is the squared norm of each row of a flat `[n × h]`
//! embedding. Per-item and deterministic (no clustering, no RNG).

use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

/// Result of a null call: a live/null flag per item plus the fitted null.
pub struct NullCall {
    /// Per-item: `true` = live (signal above null, keep), `false` = null (drop).
    pub live: Vec<bool>,
    /// Estimated null per-coordinate variance σ̂².
    pub sigma2: f64,
    /// Estimated null effective dof ν̂ (≤ the nominal dof; smaller ⇒ more
    /// correlated coordinates ⇒ more over-dispersed null).
    pub eff_dof: f64,
    /// Estimated null proportion π̂₀ (Storey).
    pub pi0: f64,
    /// Number of live items.
    pub n_live: usize,
}

/// Squared-norm null call on a flat row-major embedding `rows` (`[n × h]`):
/// computes `s_i = ‖row_i‖²` and defers to [`chi2_null_call`] with `dof = h`.
pub fn embedding_null_call(rows: &[f32], n: usize, h: usize, fdr: f32) -> NullCall {
    let s: Vec<f64> = (0..n)
        .map(|i| {
            rows[i * h..(i + 1) * h]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum()
        })
        .collect();
    chi2_null_call(&s, h, fdr)
}

/// Empirical-Bayes null call on scaled-χ² statistics `s` (each `s_i` assumed
/// `~ σ²·χ²_ν` under the null, with `ν` an *effective* dof `≤ dof`) at target
/// false-discovery rate `fdr`. `dof` is the nominal dof (the upper bound /
/// independent-coordinate count). Fits the null `(σ̂², ν̂)` from the lower
/// quantiles of `s` — null-pure because signal only inflates the upper tail,
/// and decoupled from the significance call so there is no σ̂²-shrinks-the-call
/// feedback loop — then keeps items significant above that null (Storey π̂₀ +
/// BH q ≤ fdr).
pub fn chi2_null_call(s: &[f64], dof: usize, fdr: f32) -> NullCall {
    let n = s.len();
    if n == 0 || dof == 0 {
        return NullCall {
            live: vec![false; n],
            sigma2: 0.0,
            eff_dof: dof as f64,
            pi0: 1.0,
            n_live: 0,
        };
    }
    let dof_max = dof as f64;

    let mut sorted = s.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = |p: f64| -> f64 {
        let idx = ((p * n as f64) as usize).min(n - 1);
        sorted[idx]
    };

    // Pick two *low* probabilities that sit inside the null bulk. Signal only
    // inflates `s`, so the lower tail is null-dominated whenever π₀ exceeds the
    // upper probability. A rough Storey π̂₀ (nominal `χ²_dof`, lower-quartile
    // scale) places the fit quantiles below the null mass; cap at 0.40 so they
    // stay clear of the signal even when π₀ is moderate.
    let nominal = ChiSquared::new(dof_max).expect("chi-squared dof");
    let s2_rough = (quantile(0.25) / nominal.inverse_cdf(0.25).max(1e-9)).max(1e-12);
    let lambda = 0.5;
    let m = n as f64;
    let storey_pi0 = |s2: f64, chi: &ChiSquared| -> f64 {
        let n_above = s
            .iter()
            .filter(|&&si| 1.0 - chi.cdf(si / s2) > lambda)
            .count() as f64;
        (n_above / ((1.0 - lambda) * m)).clamp(1e-3, 1.0)
    };
    let mut pi0 = storey_pi0(s2_rough, &nominal);
    let p_hi = (0.5 * pi0).clamp(0.15, 0.40);
    let p_lo = 0.5 * p_hi;
    let (q_lo, q_hi) = (quantile(p_lo).max(1e-12), quantile(p_hi).max(1e-12));
    let r = q_hi / q_lo;

    // Fit the null `(σ̂², ν̂)` from the two null-bulk quantiles. The p-th *mixture*
    // quantile is the `(p/π₀)`-th *null* quantile, so map both probabilities
    // into null-space before matching: ν̂ from the (scale-free) quantile ratio,
    // σ̂² from the lower quantile. π₀ feeds the map and is itself refined from
    // the fitted null — a few iterations on the global proportion (not the
    // per-item call) converge without the σ̂²-shrinks-the-call feedback loop.
    let mut eff_dof = dof_max;
    let mut sigma2 = s2_rough;
    let mut chi = ChiSquared::new(eff_dof).expect("effective dof");
    for _ in 0..4 {
        let a_lo = (p_lo / pi0).min(0.95);
        let a_hi = (p_hi / pi0).min(0.98).max(a_lo + 1e-3);
        eff_dof = solve_eff_dof(r, a_lo, a_hi, dof_max);
        chi = ChiSquared::new(eff_dof).expect("effective dof");
        sigma2 = (q_lo / chi.inverse_cdf(a_lo).max(1e-9)).max(1e-12);
        pi0 = storey_pi0(sigma2, &chi);
    }

    let _ = pi0; // the fit's π₀ fed the quantile map; finish_call re-derives it
    finish_call(s, &chi, sigma2, eff_dof, fdr)
}

/// Result of an [`embedding_lower_tail_call`]: which items are the lower-tail
/// "empty / collapsed" minority (to drop) plus the fitted dominant-mode null.
pub struct LowerTailCall {
    /// Per-item: `true` = lower-tail outlier (empty/ambient — drop).
    pub drop: Vec<bool>,
    /// Null (dominant-mode) location on the `log` statistic.
    pub mu: f64,
    /// Null (dominant-mode) scale on the `log` statistic (robust MAD).
    pub sigma: f64,
    /// Number of dropped (empty) items.
    pub n_drop: usize,
}

/// EB "is this an empty / collapsed item?" call on a per-item embedding **norm**
/// (e.g. faba gem's pre-L2 `cell_nrms`). Mirror-image of [`chi2_null_call`]:
/// there the null is the dominant *low* bulk and signal is the *upper* tail;
/// here the **dominant population is the real/kept mode** and the *empties are
/// the lower tail* — their MAP projection collapsed to ≈0, so `log(norm)` sits
/// far below the real-cell mode.
///
/// The null is the real mode on `x = log(norm)`, fit with **median μ̂ +
/// 1.4826·MAD σ̂** — both 50%-breakdown, so the empty minority *and* any
/// heavy upper (doublet/large) tail cannot bias the null scale (the failure
/// mode of a peak-curvature or upper-half fit, which underestimate the real
/// mode's true spread and over-drop). Each item gets a **lower-tail** p-value
/// `Φ((x−μ̂)/σ̂)`, conservative BH q-values (`π₀ = 1` — Storey is degenerate for
/// a lower-tail test against a symmetric null, and conservative suits the
/// asymmetric cost of dropping a real item), and is **dropped** when `q ≤ fdr`
/// *and* it lies below the mode. Robust, automatic (no bounds), deterministic.
pub fn embedding_lower_tail_call(nrm: &[f32], fdr: f32) -> LowerTailCall {
    let n = nrm.len();
    if n == 0 {
        return LowerTailCall {
            drop: vec![],
            mu: 0.0,
            sigma: 0.0,
            n_drop: 0,
        };
    }
    let median = |v: &[f64]| -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = s.len();
        if m % 2 == 1 {
            s[m / 2]
        } else {
            0.5 * (s[m / 2 - 1] + s[m / 2])
        }
    };
    // log statistic (floor tiny/zero norms so the log is finite).
    let x: Vec<f64> = nrm.iter().map(|&v| (v.max(1e-6) as f64).ln()).collect();
    let mu = median(&x);
    let dev: Vec<f64> = x.iter().map(|&v| (v - mu).abs()).collect();
    let sigma = (1.4826 * median(&dev)).max(1e-9);
    let normal = Normal::new(mu, sigma).expect("normal");
    // Lower-tail p under the real-mode null: small for empties (far below μ̂).
    let p: Vec<f64> = x.iter().map(|&xi| normal.cdf(xi).clamp(0.0, 1.0)).collect();
    // Conservative BH (π₀ = 1) on the lower-tail p-values.
    let m = n as f64;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut q = vec![1.0_f64; n];
    let mut running = 1.0_f64;
    for rank in (0..n).rev() {
        let gi = order[rank];
        let raw = (m * p[gi] / (rank as f64 + 1.0)).min(1.0);
        running = running.min(raw);
        q[gi] = running;
    }
    // Drop = empty: significant on the lower side only.
    let drop: Vec<bool> = (0..n).map(|i| q[i] <= fdr as f64 && x[i] < mu).collect();
    let n_drop = drop.iter().filter(|&&v| v).count();
    LowerTailCall {
        drop,
        mu,
        sigma,
        n_drop,
    }
}

/// Given a fixed null `σ²·χ²_ν` (via `chi`), compute upper-tail p-values on
/// `s`, a Storey π̂₀, BH step-up q-values, and the live/null flags at `fdr`.
/// Used by [`chi2_null_call`] (null fitted from `s`).
fn finish_call(s: &[f64], chi: &ChiSquared, sigma2: f64, eff_dof: f64, fdr: f32) -> NullCall {
    let n = s.len();
    let m = n as f64;
    let lambda = 0.5;
    // Upper-tail χ²_ν p-value per item (large stat ⇒ small p ⇒ non-null).
    let p: Vec<f64> = s
        .iter()
        .map(|&si| (1.0 - chi.cdf(si / sigma2)).clamp(0.0, 1.0))
        .collect();
    // Storey π̂₀ from the null-flat upper tail of the p-distribution.
    let n_above = p.iter().filter(|&&x| x > lambda).count() as f64;
    let pi0 = (n_above / ((1.0 - lambda) * m)).clamp(1e-3, 1.0);
    // BH step-up q-values, scaled by π̂₀.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut q = vec![1.0_f64; n];
    let mut running = 1.0_f64;
    for rank in (0..n).rev() {
        let gi = order[rank];
        let raw = (pi0 * m * p[gi] / (rank as f64 + 1.0)).min(1.0);
        running = running.min(raw);
        q[gi] = running;
    }
    let live: Vec<bool> = q.iter().map(|&qg| qg <= fdr as f64).collect();
    let n_live = live.iter().filter(|&&v| v).count();
    NullCall {
        live,
        sigma2,
        eff_dof,
        pi0,
        n_live,
    }
}

/// Result of [`embedding_mixture_empty_call`]: which items are empty (drop),
/// plus the fitted mixture and the empty/real boundary.
pub struct MixtureEmptyCall {
    /// Per-item: `true` = empty (ambient / collapsed — drop).
    pub drop: Vec<bool>,
    /// Number of dropped (empty) items.
    pub n_drop: usize,
    /// BIC-selected number of mixture components.
    pub k: usize,
    /// Component-labeling antimode on `log(norm)`: the density valley separating
    /// the empty mode from the most-massive bulk, used to decide which mixture
    /// components count as empty (`NEG_INFINITY` ⇒ no empty mode). NOT the
    /// per-item cut — see `cut`.
    pub boundary: f64,
    /// Effective per-item drop cut: the largest `log(norm)` actually called
    /// empty under the MAP rule (`NEG_INFINITY` when nothing is dropped). This
    /// is the value to report ("items with norm ≤ exp(cut) were dropped") — it
    /// can sit well above `boundary` when the empty mode dominates the mass.
    pub cut: f64,
    /// Mixture mass assigned to the empty component(s).
    pub empty_frac: f64,
}

/// Recommended `k_max` for the empty-droplet QC call. Kept **generous**: a
/// minority empty mode (e.g. ~12% of cells in a converged gem run) only gets its
/// OWN low component when BIC has enough components to also model the broad real
/// distribution — cap it too low (tried 4) and the empties merge with the
/// low-depth real cells into one "lowest mode", so the first valley lands up in
/// the real population and over-drops. The run-to-run instability that a wide
/// sweep used to cause was the mass-degenerate `empty_boundary`, not `k` itself;
/// with the lowest-mode + prominent-valley boundary, over-split ambient
/// components overlap into a smooth hump (no spurious deep valley), so a high
/// cap is safe. BIC still picks the parsimonious `k`.
pub const QC_MIXTURE_K_MAX: usize = 30;

/// EB "empty droplet" call on a per-item embedding **norm** via a BIC-selected
/// 1-D Gaussian **mixture** on `x = log(norm)`.
///
/// [`embedding_lower_tail_call`] models the empties as the lower *tail* of a
/// single median+MAD mode; that works only while the model is undertrained
/// enough that empties collapse to ≈0. On a converged model the empties get a
/// small but non-zero norm and form their **own mode**, not a tail, so the
/// lower-tail test misses them entirely. This instead fits the whole
/// (multimodal) distribution with a Gaussian mixture, picks `k` by BIC over
/// `1..=k_max` (see [`QC_MIXTURE_K_MAX`]; high enough to resolve a minority
/// empty mode from the broad real spread), identifies the **empty mode** as the
/// component(s) below the inter-mode density valley ([`empty_boundary`], which
/// prefers the valley below the dominant mode, so it works for both a converged
/// gem run and ~99%-ambient raw droplets), and assigns each item a posterior
/// `P(empty | x)`. An item is dropped by the **MAP rule** —
/// `P(empty | x) ≥ 0.5`, i.e. the empty mode owns the majority of its posterior
/// (Bayes-optimal cluster assignment). Global cumulative-FDR control is
/// deliberately *not* used: the empties are a substantial *mode*
/// (≈`empty_frac` of all items), not a sparse tail, so a per-item
/// `P(empty) > 1−α` requirement drops nothing once the empty and real modes
/// overlap. `fdr` is the target false-drop rate used only to **warn** when the
/// realized rate (mean `1 − P(empty)` over the drops) exceeds it. No median,
/// no fixed `k`, deterministic.
pub fn embedding_mixture_empty_call(nrm: &[f32], k_max: usize, fdr: f32) -> MixtureEmptyCall {
    let n = nrm.len();
    let none = || MixtureEmptyCall {
        drop: vec![false; n],
        n_drop: 0,
        k: 1,
        boundary: f64::NEG_INFINITY,
        cut: f64::NEG_INFINITY,
        empty_frac: 0.0,
    };
    if n < 2 {
        return none();
    }
    let x: Vec<f64> = nrm.iter().map(|&v| (v.max(1e-6) as f64).ln()).collect();

    // Fit the mixture on a deterministic subsample (a stride over the item order
    // — representative across batches): the mixture params are global, so the
    // posterior below is applied to every item. Keeps the wide k-sweep cheap
    // (EM cost is one `exp` per point × component × iter × k).
    const FIT_MAX: usize = 50_000;
    let xs: Vec<f64> = if n > FIT_MAX {
        let stride = (n / FIT_MAX).max(1);
        (0..n).step_by(stride).map(|i| x[i]).collect()
    } else {
        x.clone()
    };
    let mut sorted_xs = xs.clone();
    sorted_xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // BIC-select k over 1..=k_max (k=1 = no-structure baseline) on the subsample.
    let (mut best_k, mut best_bic, mut best) = (1usize, f64::INFINITY, None);
    for k in 1..=k_max.max(1) {
        let (g, bic) = fit_gmm_1d(&xs, &sorted_xs, k);
        if bic < best_bic {
            best_bic = bic;
            best_k = k;
            best = Some(g);
        }
    }
    let gmm = best.unwrap();
    let k = best_k;
    if k < 2 {
        return none();
    }

    // Empty mode = component(s) below the density antimode (valley) between the
    // lowest component and the most massive (bulk) one.
    let boundary = empty_boundary(&gmm);
    let empty: Vec<usize> = (0..k).filter(|&j| gmm.mu[j] < boundary).collect();
    if empty.is_empty() {
        return MixtureEmptyCall {
            drop: vec![false; n],
            n_drop: 0,
            k,
            boundary,
            cut: f64::NEG_INFINITY,
            empty_frac: 0.0,
        };
    }
    let empty_frac: f64 = empty.iter().map(|&j| gmm.pi[j]).sum();

    // Per-item lfdr = P(real | x) = 1 − posterior-empty.
    let mut lfdr = vec![1.0f64; n];
    for (i, &xi) in x.iter().enumerate() {
        let (mut e, mut tot) = (0.0f64, 0.0f64);
        for j in 0..k {
            let d = gmm.pi[j] * gauss1d(xi, gmm.mu[j], gmm.sg[j]);
            tot += d;
            if gmm.mu[j] < boundary {
                e += d;
            }
        }
        lfdr[i] = (1.0 - e / tot.max(1e-300)).clamp(0.0, 1.0);
    }
    // MAP rule: drop items the empty mode owns by majority posterior
    // (P(empty | x) ≥ 0.5 ⇔ lfdr < 0.5). The empty boundary already guarantees
    // we only do this when a genuine density valley separates an empty mode from
    // the bulk, so a unimodal distribution drops nothing.
    let drop: Vec<bool> = lfdr.iter().map(|&l| l < 0.5).collect();
    let n_drop = drop.iter().filter(|&&v| v).count();
    // Effective cut: the largest log-norm actually dropped. Reflects where the
    // MAP rule fell (at the density valley), which can lie well above `boundary`
    // (the component-labeling antimode) when the empty mode dominates the mass.
    let cut = (0..n)
        .filter(|&i| drop[i])
        .map(|i| x[i])
        .fold(f64::NEG_INFINITY, f64::max);
    // Realized false-drop rate = mean P(real | x) over the dropped set. Warn
    // (don't suppress) if it exceeds the target `fdr`: the cut still removes the
    // empty mode, but the modes overlap enough that some real cells go with it.
    if n_drop > 0 {
        let realized = (0..n).filter(|&i| drop[i]).map(|i| lfdr[i]).sum::<f64>() / n_drop as f64;
        if realized > fdr as f64 {
            log::warn!(
                "mixture empty call: dropped {} items at MAP posterior ≥ 0.5, but the \
                 realized false-drop rate {:.3} exceeds the target {:.3} — the empty and \
                 real modes overlap; inspect the cell_qc report before trusting the cut",
                n_drop,
                realized,
                fdr
            );
        }
    }
    MixtureEmptyCall {
        drop,
        n_drop,
        k,
        boundary,
        cut,
        empty_frac,
    }
}

struct Gmm1d {
    mu: Vec<f64>,
    sg: Vec<f64>,
    pi: Vec<f64>,
}

#[inline]
fn gauss1d(x: f64, mu: f64, sg: f64) -> f64 {
    let z = (x - mu) / sg;
    (-0.5 * z * z).exp() / (sg * (2.0 * std::f64::consts::PI).sqrt())
}

/// EM-fit a 1-D `k`-component Gaussian mixture (sufficient-stats — no `n×k`
/// responsibility matrix), returning the fit (components sorted by mean) and
/// its BIC. `sorted` = `x` sorted ascending, for quantile init.
fn fit_gmm_1d(x: &[f64], sorted: &[f64], k: usize) -> (Gmm1d, f64) {
    let n = x.len();
    let q = |p: f64| sorted[((p * n as f64) as usize).min(n - 1)];
    let mut mu: Vec<f64> = (0..k).map(|j| q((j as f64 + 0.5) / k as f64)).collect();
    let mean = x.iter().sum::<f64>() / n as f64;
    let sd = (x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
    let sd_floor = (sd * 0.05).max(1e-3);
    let mut sg = vec![sd.max(1e-2); k];
    let mut pi = vec![1.0 / k as f64; k];
    let mut d = vec![0.0f64; k];
    let (mut ll, mut ll_prev) = (0.0f64, f64::NEG_INFINITY);
    for _ in 0..200 {
        let (mut nk, mut sx, mut sxx) = (vec![0.0f64; k], vec![0.0f64; k], vec![0.0f64; k]);
        ll = 0.0;
        for &xi in x {
            let mut tot = 0.0;
            for j in 0..k {
                d[j] = pi[j] * gauss1d(xi, mu[j], sg[j]);
                tot += d[j];
            }
            let tot = tot.max(1e-300);
            ll += tot.ln();
            let inv = 1.0 / tot;
            for j in 0..k {
                let r = d[j] * inv;
                nk[j] += r;
                sx[j] += r * xi;
                sxx[j] += r * xi * xi;
            }
        }
        for j in 0..k {
            let nkj = nk[j].max(1e-9);
            pi[j] = nk[j] / n as f64;
            mu[j] = sx[j] / nkj;
            let var = (sxx[j] / nkj - mu[j] * mu[j]).max(sd_floor * sd_floor);
            sg[j] = var.sqrt().max(sd_floor);
        }
        if ll_prev.is_finite() && (ll - ll_prev).abs() <= 1e-6 * ll_prev.abs().max(1.0) {
            break;
        }
        ll_prev = ll;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&a, &b| {
        mu[a]
            .partial_cmp(&mu[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let g = Gmm1d {
        mu: idx.iter().map(|&j| mu[j]).collect(),
        sg: idx.iter().map(|&j| sg[j]).collect(),
        pi: idx.iter().map(|&j| pi[j]).collect(),
    };
    let bic = -2.0 * ll + (3.0 * k as f64 - 1.0) * (n as f64).ln();
    (g, bic)
}

/// Empty/real boundary on log(norm): the inter-mode density valley separating
/// the empty (low-norm) population from the real cells.
///
/// The empties may be a MINORITY shoulder (a converged gem run: ~12% of cells at
/// low norm, the dominant mode is real) or the MAJORITY (raw droplet data: ~99%
/// ambient, the dominant mode is empty). The broad real distribution can also
/// carry its OWN deep valley (a high-depth sub-population), so "the deepest /
/// first valley" picks the wrong one. Since the empties are always the LOWEST
/// cells, **prefer the valley BELOW the dominant mode** (`bulk`) — the gap
/// separating a low empty cloud from a real bulk above it. If there is none (the
/// bulk is itself in the empty cloud), take the valley **ABOVE** the bulk (the
/// gap before the real cells). This needs no mass threshold and, by capping the
/// below-search at the bulk, never mistakes a high-depth real valley for the
/// empty cut. `NEG_INFINITY` when neither range holds a genuine valley (one mode
/// only ⇒ drop nothing), so clean pre-called data is never over-dropped.
fn empty_boundary(gmm: &Gmm1d) -> f64 {
    let k = gmm.mu.len();
    if k < 2 {
        return f64::NEG_INFINITY;
    }
    let bulk = (0..k)
        .max_by(|&a, &b| {
            gmm.pi[a]
                .partial_cmp(&gmm.pi[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    valley_in(gmm, gmm.mu[0], gmm.mu[bulk]) // prefer the empty/real gap below the bulk
        .or_else(|| valley_in(gmm, gmm.mu[bulk], gmm.mu[k - 1])) // else above it
        .unwrap_or(f64::NEG_INFINITY)
}

/// The genuine interior density valley of the mixture over `[lo, hi]`: the
/// global minimum, accepted only when it lies strictly below both endpoints and
/// away from the edges (so a monotone — one-mode — range yields `None`).
fn valley_in(gmm: &Gmm1d, lo: f64, hi: f64) -> Option<f64> {
    // Component means are finite, so `hi <= lo` is the empty/degenerate range.
    if hi <= lo {
        return None;
    }
    let k = gmm.mu.len();
    let dens = |t: f64| -> f64 {
        (0..k)
            .map(|j| gmm.pi[j] * gauss1d(t, gmm.mu[j], gmm.sg[j]))
            .sum()
    };
    let steps = 2000usize;
    let t_at = |i: usize| lo + (hi - lo) * i as f64 / steps as f64;
    let d: Vec<f64> = (0..=steps).map(|s| dens(t_at(s))).collect();
    let j = (0..=steps)
        .min_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    let edge = steps / 50;
    if j < edge || j > steps - edge || !(d[j] < d[0] && d[j] < d[steps]) {
        return None;
    }
    Some(t_at(j))
}

/// Solve `χ²_ν⁻¹(a_hi)/χ²_ν⁻¹(a_lo) = r` for the effective dof `ν` by bisection,
/// where `a_lo < a_hi` are the (null-space) probabilities of the two matched
/// quantiles. The ratio is monotone *decreasing* in `ν` (more dof ⇒ tighter
/// quantiles), so a larger observed ratio `r` (more over-dispersion) maps to a
/// smaller `ν`. Clamped to `[0.2, dof_max]`: `r` at/below the `dof_max` ratio ⇒
/// no detectable over-dispersion, return `dof_max`.
fn solve_eff_dof(r: f64, a_lo: f64, a_hi: f64, dof_max: f64) -> f64 {
    let ratio = |nu: f64| -> f64 {
        let c = ChiSquared::new(nu).expect("chi-squared nu");
        c.inverse_cdf(a_hi) / c.inverse_cdf(a_lo).max(1e-12)
    };
    let (mut lo, mut hi) = (0.2_f64, dof_max.max(0.3));
    // Decreasing in ν: ratio(hi) is the min, ratio(lo) the max achievable.
    if r <= ratio(hi) {
        return hi;
    }
    if r >= ratio(lo) {
        return lo;
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if ratio(mid) > r {
            lo = mid; // need more dof to tighten the ratio
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}
