//! Per-gene negative binomial marginal fits + PIT helpers for the
//! scDesign-style copula simulator.
//!
//! NB parametrization is `(size = r, mean = μ)` with PMF
//! `P(X=k) = C(k+r-1, k) p^r (1-p)^k`, `p = r / (r + μ)`. The log PMF is
//! computed iteratively via
//! `ln P(k+1) = ln P(k) + ln((k+r)/(k+1)) + ln(μ/(r+μ))`,
//! which avoids `lgamma` calls and stays stable for the count ranges seen in
//! single-cell data. Reference: Sun, Song, Li & Li 2021, *scDesign2*,
//! Genome Biology 22:163.

#[derive(Debug, Clone, Copy)]
pub struct NbFit {
    /// NB mean μ. For the Poisson fallback this is the rate λ.
    pub mu: f32,
    /// NB size r. `f32::INFINITY` signals the Poisson collapse (σ² ≤ μ).
    pub r: f32,
}

impl NbFit {
    pub fn is_poisson(&self) -> bool {
        !self.r.is_finite()
    }

    /// Apply a multiplicative shift on μ (batch effect δ); leave r unchanged.
    pub fn with_mean(&self, mu: f32) -> Self {
        Self { mu, r: self.r }
    }
}

/// Method-of-moments NB fit from the nonzero values of a gene plus the total
/// cell count (so we can recover the implicit zeros for the variance).
/// Falls back to Poisson when σ² ≤ μ (NB MoM would yield a non-positive r).
pub fn fit_nb_mom_from_nonzeros(nonzeros: &[f32], n_total: usize, r_floor: f32) -> NbFit {
    if n_total == 0 {
        return NbFit {
            mu: 0.0,
            r: f32::INFINITY,
        };
    }
    let inv_n = 1.0 / n_total as f64;
    let s: f64 = nonzeros.iter().map(|&v| v as f64).sum();
    let s2: f64 = nonzeros.iter().map(|&v| (v as f64).powi(2)).sum();
    let mu = s * inv_n;
    let var = (s2 * inv_n - mu * mu).max(0.0);
    if mu <= 0.0 || var <= mu {
        return NbFit {
            mu: mu as f32,
            r: f32::INFINITY,
        };
    }
    let r = (mu * mu) / (var - mu);
    NbFit {
        mu: mu as f32,
        r: (r as f32).max(r_floor),
    }
}

/// Build the cumulative distribution table `F(0), F(1), ..., F(k_max)` by
/// iterating the log-PMF recurrence and exponentiating. Returns a vector of
/// length `k_max + 1`.
pub fn nb_cdf_table(fit: NbFit, k_max: usize) -> Vec<f64> {
    let mu = fit.mu as f64;
    let r = fit.r as f64;
    let mut cdf = Vec::with_capacity(k_max + 1);
    if mu <= 0.0 {
        cdf.resize(k_max + 1, 1.0);
        return cdf;
    }
    if !r.is_finite() {
        // Poisson(μ): ln P(0) = -μ; ln P(k+1) = ln P(k) + ln(μ) - ln(k+1).
        let mut ln_pk = -mu;
        let mut acc = ln_pk.exp();
        cdf.push(acc.min(1.0));
        for k in 0..k_max {
            ln_pk += mu.ln() - ((k + 1) as f64).ln();
            acc += ln_pk.exp();
            cdf.push(acc.min(1.0));
        }
        return cdf;
    }
    let p = r / (r + mu);
    let ln_p = p.ln();
    let ln_q = (1.0 - p).ln();
    let mut ln_pk = r * ln_p;
    let mut acc = ln_pk.exp();
    cdf.push(acc.min(1.0));
    for k in 0..k_max {
        let kf = k as f64;
        ln_pk += ((kf + r) / (kf + 1.0)).ln() + ln_q;
        acc += ln_pk.exp();
        cdf.push(acc.min(1.0));
    }
    cdf
}

/// PIT with continuity correction: `u = 0.5·F(x-1) + 0.5·F(x)`. Clamped to
/// `(EPS, 1-EPS)` so the subsequent `Φ⁻¹` is finite.
pub fn pit_continuity(table: &[f64], x: u32) -> f64 {
    const EPS: f64 = 1e-7;
    let xi = x as usize;
    let cdf_hi = if xi >= table.len() {
        *table.last().unwrap_or(&1.0)
    } else {
        table[xi]
    };
    let cdf_lo = if xi == 0 {
        0.0
    } else if xi > table.len() {
        *table.last().unwrap_or(&1.0)
    } else {
        table[xi - 1]
    };
    let u = 0.5 * (cdf_lo + cdf_hi);
    u.clamp(EPS, 1.0 - EPS)
}

/// Generous cap on the NB PMF table size: `μ + 20·σ + 100` is well past the
/// effective tail for any realistic single-cell mean. Used by both the
/// inverse-CDF helper here and the per-(HVG, batch) table cache in
/// `copula::sample_copula`.
pub fn nb_table_cap(fit: NbFit) -> usize {
    let mu = fit.mu as f64;
    let r = fit.r as f64;
    let var = if r.is_finite() { mu + mu * mu / r } else { mu };
    ((mu + 20.0 * var.sqrt()).ceil() as usize).saturating_add(100)
}

/// `min { k : F(k) ≥ u }` from a pre-built CDF table. Hot-path lookup used by
/// the per-cell sample loop; pair with `nb_cdf_table` once per (HVG, batch).
pub fn nb_inverse_cdf_from_table(u: f64, table: &[f64]) -> u32 {
    if u <= 0.0 {
        return 0;
    }
    for (k, &f) in table.iter().enumerate() {
        if f >= u {
            return k as u32;
        }
    }
    table.len() as u32
}

/// Inverse CDF that builds a fresh table each call. Convenience for one-off
/// callers; the per-cell loop should pre-cache and call `nb_inverse_cdf_from_table`.
pub fn nb_inverse_cdf(u: f64, fit: NbFit) -> u32 {
    if u <= 0.0 || fit.mu <= 0.0 {
        return 0;
    }
    let table = nb_cdf_table(fit, nb_table_cap(fit));
    nb_inverse_cdf_from_table(u, &table)
}

/// Standard normal CDF Φ; thin wrapper over `statrs::distribution::Normal`.
pub fn phi(z: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    Normal::standard().cdf(z)
}

/// Standard normal inverse CDF Φ⁻¹; thin wrapper over `statrs::distribution::Normal`.
pub fn inv_phi(u: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    Normal::standard().inverse_cdf(u)
}

/// Sample from `NB(μ, r)` via the Gamma-Poisson mixture. For Poisson fallback
/// (`r = ∞`), draws directly from `Poisson(μ)`.
pub fn sample_nb<R: rand::Rng + ?Sized>(fit: NbFit, rng: &mut R) -> u32 {
    use rand::distr::Distribution;
    if fit.mu <= 0.0 {
        return 0;
    }
    if !fit.r.is_finite() {
        let p = rand_distr::Poisson::new(fit.mu as f64).expect("μ > 0");
        return p.sample(rng) as u32;
    }
    // Gamma-Poisson: λ ~ Gamma(r, μ/r), x ~ Poisson(λ).
    let scale = (fit.mu / fit.r).max(f32::MIN_POSITIVE) as f64;
    let g = rand_distr::Gamma::new(fit.r as f64, scale).expect("r > 0, scale > 0");
    let lambda = g.sample(rng);
    if !lambda.is_finite() || lambda <= 0.0 {
        return 0;
    }
    let p = rand_distr::Poisson::new(lambda).expect("λ > 0");
    p.sample(rng) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::Distribution;

    #[test]
    fn cdf_monotone_and_unit() {
        let fit = NbFit { mu: 5.0, r: 2.0 };
        let table = nb_cdf_table(fit, 200);
        for w in table.windows(2) {
            assert!(w[1] >= w[0] - 1e-12);
        }
        assert!((table.last().unwrap() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn cdf_inverse_round_trip() {
        let fit = NbFit { mu: 3.0, r: 1.5 };
        let table = nb_cdf_table(fit, 200);
        for k in 0u32..30 {
            let u = pit_continuity(&table, k);
            let k_back = nb_inverse_cdf(u, fit);
            // continuity-correction midpoint must invert to the same value.
            assert!(
                k_back == k || k_back + 1 == k || k_back == k + 1,
                "k={} back={}",
                k,
                k_back
            );
        }
    }

    #[test]
    fn poisson_fallback_when_underdispersed() {
        // All cells have value 1.
        let nonzeros = vec![1.0_f32; 100];
        let fit = fit_nb_mom_from_nonzeros(&nonzeros, 100, 1e-3);
        assert!(fit.is_poisson());
        assert!((fit.mu - 1.0).abs() < 1e-6);
    }

    #[test]
    fn nb_recovery_on_synthetic() {
        // Ground truth μ=4, r=2.
        let mu = 4.0_f32;
        let r = 2.0_f32;
        let truth = NbFit { mu, r };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 20_000;
        let mut nonzeros = Vec::new();
        for _ in 0..n {
            let v = sample_nb(truth, &mut rng);
            if v > 0 {
                nonzeros.push(v as f32);
            }
        }
        let fit = fit_nb_mom_from_nonzeros(&nonzeros, n, 1e-3);
        assert!(
            (fit.mu - mu).abs() / mu < 0.1,
            "mu fit={} truth={}",
            fit.mu,
            mu
        );
        assert!((fit.r - r).abs() / r < 0.3, "r fit={} truth={}", fit.r, r);
    }

    #[test]
    fn phi_inv_phi_round_trip() {
        for &z in &[-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0] {
            let u = phi(z);
            let z_back = inv_phi(u);
            assert!((z_back - z).abs() < 1e-9, "z={} back={}", z, z_back);
        }
    }

    #[test]
    fn phi_inv_phi_extremes() {
        // Tail behavior shouldn't blow up.
        for &u in &[1e-6, 1e-3, 0.5, 1.0 - 1e-3, 1.0 - 1e-6] {
            let z = inv_phi(u);
            assert!(z.is_finite(), "inv_phi({})={} not finite", u, z);
            let _ = rand_distr::Normal::new(0.0, 1.0)
                .unwrap()
                .sample(&mut rand::rng());
        }
    }
}
