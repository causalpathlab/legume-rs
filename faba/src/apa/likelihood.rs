use crate::apa::fragment::{FragmentCluster, FragmentRecord};

/// Parameters for the SCAPE likelihood model.
pub struct LikelihoodParams {
    /// Expected mean fragment length (mu_f), default 300
    pub mu_f: f32,
    /// Standard deviation of fragment length (sigma_f), default 50
    pub sigma_f: f32,
    /// Step size for theta enumeration, default 10
    pub theta_step: usize,
    /// Maximum polyA length (LA), default 150
    pub max_polya: f32,
    /// Minimum polyA length, default 20
    pub min_polya: f32,
}

impl Default for LikelihoodParams {
    fn default() -> Self {
        Self {
            mu_f: 300.0,
            sigma_f: 50.0,
            theta_step: 10,
            max_polya: 150.0,
            min_polya: 20.0,
        }
    }
}

/// SCAPE per-fragment likelihood `log p(x_n, l_n, r_n | θ_nk)`. Depends
/// only on `(x, l, r, is_junction)` — `pa_site` is only used in site
/// discovery, not in the per-fragment likelihood, so this signature is
/// shared between `FragmentRecord` and `FragmentCluster` via
/// [`log_lik_features_given_theta`].
#[allow(dead_code)] // kept for tests / external callers; production path uses the cluster variant
pub fn log_lik_fragment_given_theta(
    frag: &FragmentRecord,
    theta: f32,
    utr_length: f32,
    params: &LikelihoodParams,
) -> f32 {
    log_lik_features_given_theta(
        frag.x,
        frag.l,
        frag.r,
        frag.is_junction,
        theta,
        utr_length,
        params,
    )
}

/// Same as [`log_lik_fragment_given_theta`] but for a clustered
/// representative — the cluster's representative `(x, l, r)` mean is
/// used; multiplicity enters later in the EM accumulator.
pub fn log_lik_cluster_given_theta(
    cluster: &FragmentCluster,
    theta: f32,
    utr_length: f32,
    params: &LikelihoodParams,
) -> f32 {
    log_lik_features_given_theta(
        cluster.x,
        cluster.l,
        cluster.r,
        cluster.is_junction,
        theta,
        utr_length,
        params,
    )
}

/// Shared kernel: takes raw features. The original code branched on
/// `frag.is_junction && r > 0.0`; that's preserved here unchanged.
///
/// SCAPE eq 5-10: `p(x,l,r|θ) = Σ_s f(x,l,r,s,θ)` with
/// `f = p(l|x,θ) · p(x|s,θ) · p(r|s) · p(s)`. Junction reads collapse
/// the s marginal; SE reads enumerate s.
#[inline]
pub fn log_lik_features_given_theta(
    x: f32,
    l: f32,
    r: f32,
    is_junction: bool,
    theta: f32,
    utr_length: f32,
    params: &LikelihoodParams,
) -> f32 {
    // Junction read: s is known (s = r), theta = pa_site
    if is_junction && r > 0.0 {
        // For junction reads, set p(x|s,theta) = 1 and p(r|s) = 1
        // Only contribution is from p(l|x,theta)
        let max_l = theta - x + 1.0;
        if l >= 1.0 && l <= max_l && max_l > 0.0 {
            return -(max_l).ln();
        } else {
            return f32::NEG_INFINITY;
        }
    }

    // Non-junction read: marginalize over s
    let s_min = params.min_polya;
    let s_max = params.max_polya;
    let s_step = params.theta_step as f32;

    let mut log_sum = f32::NEG_INFINITY;

    let mut s = s_min;
    while s <= s_max {
        let log_f = compute_log_f(x, l, r, s, theta, utr_length, params);
        log_sum = log_sum_exp(log_sum, log_f);
        s += s_step;
    }

    log_sum
}

/// Compute log f(x, l, r, s, theta) = log[p(l|x,theta) * p(x|s,theta) * p(r|s) * p(s)]
fn compute_log_f(
    x: f32,
    l: f32,
    r: f32,
    s: f32,
    theta: f32,
    _utr_length: f32,
    params: &LikelihoodParams,
) -> f32 {
    // p(s) = 1/(LA - min_polya) for s in [min_polya, LA]
    let s_range = params.max_polya - params.min_polya;
    if s_range <= 0.0 || s < params.min_polya || s > params.max_polya {
        return f32::NEG_INFINITY;
    }
    let log_ps = -s_range.ln();

    // p(r|s) = 1/s for 1 <= r <= s, else 0
    // For non-junction reads with r=0, we treat r as unobserved -> p(r|s) = 1
    let log_pr_s = if r > 0.0 {
        if r >= 1.0 && r <= s {
            -s.ln()
        } else {
            return f32::NEG_INFINITY;
        }
    } else {
        0.0 // r unobserved
    };

    // p(x|s,theta) = N(theta + s + 1 - mu_f, sigma_f^2) evaluated at x
    // i.e., x ~ N(theta + s + 1 - mu_f, sigma_f^2)
    let mean_x = theta + s + 1.0 - params.mu_f;
    let log_px_s_theta = log_normal_pdf(x, mean_x, params.sigma_f);

    // p(l|x,theta) = 1/(theta - x + 1) for 1 <= l <= theta - x + 1
    let max_l = theta - x + 1.0;
    let log_pl_x_theta = if max_l > 0.0 && l >= 1.0 && l <= max_l {
        -max_l.ln()
    } else {
        return f32::NEG_INFINITY;
    };

    log_ps + log_pr_s + log_px_s_theta + log_pl_x_theta
}

/// Log of the noise model likelihood (eq 13): 1/(L^2 * LA)
pub fn log_lik_noise(utr_length: f32, max_polya: f32) -> f32 {
    -2.0 * utr_length.ln() - max_polya.ln()
}

/// Precompute the θ likelihood matrix for a slice of **clusters** in
/// flat row-major layout: entry `[m * n_theta + t]` is
/// `log p(x_m, l_m, r_m | θ_t)` for cluster `m`. Cluster counts (`c_m`)
/// do **not** enter here — they're applied later in the EM accumulator
/// (per-cluster log-likelihood is weighted by `c_m`, the per-cluster
/// row of the matrix is shared across all `c_m` original fragments).
///
/// Returns `(matrix, theta_grid)`. Recover `n_theta` as `theta_grid.len()`.
pub fn precompute_theta_lik_matrix(
    clusters: &[FragmentCluster],
    utr_length: f32,
    params: &LikelihoodParams,
) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;

    let step = params.theta_step;
    let theta_grid: Vec<f32> = (1..=utr_length as usize)
        .step_by(step)
        .map(|t| t as f32)
        .collect();
    let n_theta = theta_grid.len();
    let n_obs = clusters.len();

    let mut matrix = vec![0.0_f32; n_obs * n_theta];

    // Threshold matches the EM E-step's parallel gate: parallelise only
    // when the per-cluster cost has anything to amortise rayon spawn over.
    const PARALLEL_THRESHOLD: usize = 4096;
    if n_obs >= PARALLEL_THRESHOLD {
        matrix
            .par_chunks_mut(n_theta)
            .zip(clusters.par_iter())
            .for_each(|(row, cluster)| {
                for (t, &theta) in theta_grid.iter().enumerate() {
                    row[t] = log_lik_cluster_given_theta(cluster, theta, utr_length, params);
                }
            });
    } else {
        for (m, cluster) in clusters.iter().enumerate() {
            let row = &mut matrix[m * n_theta..(m + 1) * n_theta];
            for (t, &theta) in theta_grid.iter().enumerate() {
                row[t] = log_lik_cluster_given_theta(cluster, theta, utr_length, params);
            }
        }
    }

    (matrix, theta_grid)
}

/// Compute log p(x_n, l_n, r_n | alpha_k, beta_k) by marginalizing theta
/// over the Gaussian N(alpha_k, beta_k^2) weighted by precomputed theta likelihoods.
///
/// log p(x,l,r|alpha,beta) = log sum_theta p(x,l,r|theta) * N(theta|alpha, beta^2)
pub fn log_lik_fragment_given_site(
    frag_theta_liks: &[f32],
    theta_grid: &[f32],
    alpha: f32,
    beta: f32,
) -> f32 {
    let mut log_sum = f32::NEG_INFINITY;

    for (t, &theta) in theta_grid.iter().enumerate() {
        let log_lik = frag_theta_liks[t];
        if log_lik.is_finite() {
            let log_prior = log_normal_pdf(theta, alpha, beta);
            log_sum = log_sum_exp(log_sum, log_lik + log_prior);
        }
    }

    log_sum
}

/// Robust per-site emission: marginalize theta over a heavy-tailed prior
///
///     prior(theta) = (1 - eta) * N(theta | alpha, beta^2)
///                  +      eta  * Uniform(theta | alpha - W, alpha + W)
///
/// where `W = skirt_mult * beta`. The local uniform "skirt" absorbs near-site
/// outliers that would otherwise nudge BIC into picking an extra Gaussian site
/// in the same broad cleavage cluster.
pub fn log_lik_fragment_given_site_robust(
    frag_theta_liks: &[f32],
    theta_grid: &[f32],
    alpha: f32,
    beta: f32,
    eta: f32,
    skirt_mult: f32,
) -> f32 {
    let w = skirt_mult * beta;
    if eta <= 0.0 || skirt_mult <= 0.0 || w <= 0.0 || !w.is_finite() {
        return log_lik_fragment_given_site(frag_theta_liks, theta_grid, alpha, beta);
    }
    let lo = alpha - w;
    let hi = alpha + w;
    let log_uniform_density = -(2.0 * w).ln();
    let log_1m_eta = (1.0 - eta).ln();
    let log_eta = eta.ln();

    let mut log_sum = f32::NEG_INFINITY;
    for (t, &theta) in theta_grid.iter().enumerate() {
        let log_lik = frag_theta_liks[t];
        if !log_lik.is_finite() {
            continue;
        }
        let log_gauss = log_normal_pdf(theta, alpha, beta);
        let log_uniform = if theta >= lo && theta <= hi {
            log_uniform_density
        } else {
            f32::NEG_INFINITY
        };
        let log_prior = log_sum_exp(log_1m_eta + log_gauss, log_eta + log_uniform);
        if log_prior.is_finite() {
            log_sum = log_sum_exp(log_sum, log_lik + log_prior);
        }
    }
    log_sum
}

/// Compute log-normal PDF: log N(x | mu, sigma)
fn log_normal_pdf(x: f32, mu: f32, sigma: f32) -> f32 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * std::f32::consts::TAU.ln()
}

/// Numerically stable log(exp(a) + exp(b))
pub fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests;
