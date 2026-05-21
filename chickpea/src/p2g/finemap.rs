//! Per-gene SuSiE-RSS fine-mapping from embedding-space z-scores and LD.
//!
//! Standard summary-statistics SuSiE (Wang et al. 2020; Zhu & Stephens 2017) in
//! IBSS form, on the cis-peak correlation matrix `R` (cosine of peak embeddings)
//! and marginal z-scores `z` (gene·peak embedding inner products), with residual
//! variance fixed at 1. Each gene is an independent block; callers parallelize
//! over genes. See `docs/peak_to_gene_math.md`.

use crate::common::*;

/// Parameters for per-gene RSS fine-mapping.
pub struct FinemapParams {
    /// Number of single-effect components L (max causal peaks per gene).
    pub num_components: usize,
    /// SuSiE prior effect-size variance (z-score scale).
    pub prior_var: f64,
}

/// Fine-map one gene's cis peaks. `r` is the `[C×C]` LD (PSD, unit diagonal),
/// `z` the `[C]` marginal z-scores. Returns `(pip, effect_mean, effect_std)`.
pub fn finemap_gene(r: &Mat, z: &[f32], params: &FinemapParams) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let c = r.nrows();
    if c == 0 {
        return (vec![], vec![], vec![]);
    }
    if c == 1 {
        return (vec![1.0], vec![z[0]], vec![0.0]);
    }
    susie_rss(r, z, params.num_components.max(1), params.prior_var.max(1e-6), 100)
}

/// SuSiE-RSS IBSS with residual variance fixed at 1.
fn susie_rss(
    r: &Mat,
    z: &[f32],
    l: usize,
    sigma2_0: f64,
    max_iter: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let c = r.nrows();
    let log_prior = -(c as f64).ln();

    // Per-peak posterior variance s2_j = 1/(R_jj + 1/σ²₀) (σ²=1).
    let s2: Vec<f64> = (0..c)
        .map(|j| 1.0 / (r[(j, j)] as f64 + 1.0 / sigma2_0))
        .collect();

    let mut alpha = vec![vec![1.0 / c as f64; c]; l];
    let mut mu = vec![vec![0.0f64; c]; l];
    let mut b = vec![0.0f64; c]; // Σ_l α_l⊙μ_l (posterior mean total effect)
    let mut rb = matvec(r, &b); // R·b

    for _iter in 0..max_iter {
        let mut max_delta = 0.0f64;
        for li in 0..l {
            // Effect of component l, and R·(that effect), to residualize z.
            let bl: Vec<f64> = (0..c).map(|j| alpha[li][j] * mu[li][j]).collect();
            let rbl = matvec(r, &bl);

            let mut logbf = vec![0.0f64; c];
            let mut new_mu = vec![0.0f64; c];
            for j in 0..c {
                // Residual marginal association: z_j − (R·b_{−l})_j.
                let rz = z[j] as f64 - (rb[j] - rbl[j]);
                let m = s2[j] * rz; // posterior mean (σ²=1)
                new_mu[j] = m;
                logbf[j] = 0.5 * (s2[j] / sigma2_0).ln() + 0.5 * m * m / s2[j] + log_prior;
            }

            // α_l = softmax(logbf)
            let maxbf = logbf.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0f64;
            let mut new_alpha = vec![0.0f64; c];
            for j in 0..c {
                new_alpha[j] = (logbf[j] - maxbf).exp();
                sum_exp += new_alpha[j];
            }
            for a in new_alpha.iter_mut() {
                *a /= sum_exp;
            }

            // Update running effect and R·b by the component delta.
            let delta_bl: Vec<f64> = (0..c).map(|j| new_alpha[j] * new_mu[j] - bl[j]).collect();
            let r_delta = matvec(r, &delta_bl);
            for j in 0..c {
                b[j] += delta_bl[j];
                rb[j] += r_delta[j];
                max_delta = max_delta.max((new_alpha[j] - alpha[li][j]).abs());
            }
            alpha[li] = new_alpha;
            mu[li] = new_mu;
        }
        if max_delta < 1e-4 {
            break;
        }
    }

    let mut pip = vec![0f32; c];
    let mut eff_mean = vec![0f32; c];
    let mut eff_std = vec![0f32; c];
    for j in 0..c {
        let mut log_one_minus = 0.0f64;
        let mut bj = 0.0f64;
        let mut second = 0.0f64;
        for li in 0..l {
            log_one_minus += (1.0 - alpha[li][j]).max(1e-15).ln();
            bj += alpha[li][j] * mu[li][j];
            second += alpha[li][j] * (s2[j] + mu[li][j] * mu[li][j]);
        }
        pip[j] = (1.0 - log_one_minus.exp()) as f32;
        eff_mean[j] = bj as f32;
        eff_std[j] = (second - bj * bj).max(0.0).sqrt() as f32;
    }
    (pip, eff_mean, eff_std)
}

/// `R · v` for a symmetric `R` (f32) and an f64 vector.
fn matvec(r: &Mat, v: &[f64]) -> Vec<f64> {
    let c = r.nrows();
    (0..c)
        .map(|i| {
            let mut s = 0.0f64;
            for j in 0..c {
                s += r[(i, j)] as f64 * v[j];
            }
            s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_signal_peak() {
        // Independent peaks (R = I); one peak carries a strong z-score.
        let c = 8;
        let r = Mat::identity(c, c);
        let mut z = vec![0.4f32; c];
        z[3] = 6.0;

        let params = FinemapParams {
            num_components: 5,
            prior_var: 5.0,
        };
        let (pip, _mean, _std) = finemap_gene(&r, &z, &params);

        let argmax = pip
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax, 3, "PIPs: {pip:?}");
        assert!(pip[3] > 0.8, "causal PIP too low: {}", pip[3]);
    }

    #[test]
    fn ld_suppresses_redundant_peak() {
        // Two perfectly collinear peaks (0,1) both with strong z; SuSiE should
        // split inclusion between them rather than give both PIP≈1.
        let c = 3;
        let mut r = Mat::identity(c, c);
        r[(0, 1)] = 0.99;
        r[(1, 0)] = 0.99;
        let z = vec![6.0f32, 6.0, 0.2];
        let params = FinemapParams {
            num_components: 2,
            prior_var: 5.0,
        };
        let (pip, _, _) = finemap_gene(&r, &z, &params);
        assert!(pip[0] + pip[1] > 0.9, "joint PIP of the pair too low: {pip:?}");
        assert!(pip[0] < 0.95 && pip[1] < 0.95, "collinear peaks not split: {pip:?}");
        assert!(pip[2] < 0.3, "null peak PIP too high: {pip:?}");
    }
}
