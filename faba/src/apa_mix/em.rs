use crate::apa_mix::likelihood::{log_lik_fragment_given_site, log_lik_noise, log_sum_exp};
use log::info;

/// EM algorithm parameters.
pub struct EmParams {
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Convergence tolerance for log-likelihood change
    pub tol: f64,
    /// Minimum component weight before pruning
    pub min_weight: f64,
}

impl Default for EmParams {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-4,
            min_weight: 0.01,
        }
    }
}

/// Result of EM inference for one UTR.
pub struct EmResult {
    /// Estimated mixing weights pi_0..pi_K (pi_0 = noise)
    pub weights: Vec<f64>,
    /// pA site positions alpha_k (for k=1..K)
    pub alphas: Vec<f64>,
    /// pA site dispersions beta_k (for k=1..K)
    pub betas: Vec<f64>,
    /// Posterior assignment probabilities gamma[n][k] (k=0 is noise)
    pub gamma: Vec<Vec<f64>>,
    /// Final log-likelihood
    pub log_lik: f64,
    /// BIC
    pub bic: f64,
    /// Number of iterations
    pub n_iter: usize,
}

/// Constrained EM: alpha and beta are fixed, only estimate weights pi_k.
/// This is SCAPE's `fixed_inference()`.
///
/// * `alpha_arr` - Fixed pA site positions (UTR-relative)
/// * `beta_arr` - Fixed pA site dispersions
/// * `theta_lik_matrix` - Precomputed [n_frag x n_theta] log-likelihood matrix
/// * `theta_grid` - Theta grid values corresponding to matrix columns
/// * `utr_length` - Length of the UTR
/// * `max_polya` - Maximum polyA length (LA)
/// * `params` - EM parameters
pub fn fixed_inference(
    alpha_arr: &[f64],
    beta_arr: &[f64],
    theta_lik_matrix: &[Vec<f64>],
    theta_grid: &[f64],
    utr_length: f64,
    max_polya: f64,
    params: &EmParams,
) -> EmResult {
    let n_frag = theta_lik_matrix.len();
    let n_components = alpha_arr.len(); // K APA components
    let n_total = n_components + 1; // +1 for noise component (k=0)

    // Precompute log p(x_n, l_n, r_n | alpha_k, beta_k) for each fragment and component
    let mut component_log_liks = vec![vec![0.0; n_total]; n_frag];
    let noise_ll = log_lik_noise(utr_length, max_polya);

    for n in 0..n_frag {
        // k=0: noise component
        component_log_liks[n][0] = noise_ll;

        // k=1..K: APA components
        for k in 0..n_components {
            component_log_liks[n][k + 1] = log_lik_fragment_given_site(
                &theta_lik_matrix[n],
                theta_grid,
                alpha_arr[k],
                beta_arr[k],
            );
        }
    }

    // Initialize weights uniformly
    let mut weights = vec![1.0 / n_total as f64; n_total];
    let mut gamma = vec![vec![0.0; n_total]; n_frag];
    let mut prev_ll = f64::NEG_INFINITY;

    let mut iter = 0;
    loop {
        // E-step: compute gamma_nk = pi_k * p(x,l,r|alpha_k,beta_k) / sum_j(...)
        let mut total_ll = 0.0;

        for n in 0..n_frag {
            // Compute log(pi_k * p(x,l,r|k)) for each component
            let mut log_probs = vec![f64::NEG_INFINITY; n_total];
            for k in 0..n_total {
                if weights[k] > 0.0 {
                    log_probs[k] = weights[k].ln() + component_log_liks[n][k];
                }
            }

            // Log-sum-exp for normalization
            let log_norm = log_probs
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &x| log_sum_exp(acc, x));

            total_ll += log_norm;

            // Normalize to get gamma
            for (k, lp) in log_probs.iter().enumerate().take(n_total) {
                gamma[n][k] = (lp - log_norm).exp();
            }
        }

        iter += 1;

        // Check convergence
        let ll_change = (total_ll - prev_ll).abs();
        if iter > 1 && (ll_change < params.tol || iter >= params.max_iter) {
            info!(
                "EM converged after {} iterations, log-lik = {:.2}, change = {:.6}",
                iter, total_ll, ll_change
            );

            // Compute BIC: -2*logL + (3K+1)*log(N)
            let n_params = 3 * n_components + 1;
            let bic = -2.0 * total_ll + n_params as f64 * (n_frag as f64).ln();

            return EmResult {
                weights,
                alphas: alpha_arr.to_vec(),
                betas: beta_arr.to_vec(),
                gamma,
                log_lik: total_ll,
                bic,
                n_iter: iter,
            };
        }

        prev_ll = total_ll;

        // M-step: update weights (alpha, beta fixed)
        // pi_k = (1/N) * sum_n gamma_nk
        for k in 0..n_total {
            let sum_gamma: f64 = gamma.iter().map(|g| g[k]).sum();
            weights[k] = sum_gamma / n_frag as f64;
        }

        // Prune components with weight below threshold
        for w in weights.iter_mut().take(n_total).skip(1) {
            if *w < params.min_weight {
                *w = 0.0;
            }
        }

        // Renormalize weights
        let w_sum: f64 = weights.iter().sum();
        if w_sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= w_sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apa_mix::likelihood::{precompute_theta_lik_matrix, LikelihoodParams};
    use crate::apa_mix::simulate::{simulate_fragments, ScapeSimParams};

    fn run_em_on_sim(params: &ScapeSimParams) -> EmResult {
        let (fragments, _labels) = simulate_fragments(params);
        let lik_params = LikelihoodParams {
            mu_f: params.mu_f,
            sigma_f: params.sigma_f,
            theta_step: 10,
            max_polya: params.max_polya,
            min_polya: params.min_polya,
        };
        let (theta_lik_matrix, theta_grid) =
            precompute_theta_lik_matrix(&fragments, params.utr_length, &lik_params);

        let em_params = EmParams {
            max_iter: 200,
            tol: 1e-6,
            min_weight: 0.005,
        };

        fixed_inference(
            &params.alphas,
            &params.betas,
            &theta_lik_matrix,
            &theta_grid,
            params.utr_length,
            params.max_polya,
            &em_params,
        )
    }

    #[test]
    fn test_single_site_no_noise() {
        let params = ScapeSimParams {
            utr_length: 2000.0,
            weights: vec![0.0, 1.0], // no noise
            alphas: vec![500.0],
            betas: vec![30.0],
            n_fragments: 2000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 42,
            ..Default::default()
        };
        let result = run_em_on_sim(&params);

        eprintln!(
            "single_site_no_noise: weights={:?}, n_iter={}",
            result.weights, result.n_iter
        );
        assert!(
            result.weights[1] > 0.9,
            "APA weight should be >0.9, got {}",
            result.weights[1]
        );
    }

    #[test]
    fn test_two_sites_balanced() {
        let params = ScapeSimParams {
            utr_length: 3000.0,
            weights: vec![0.0, 0.5, 0.5],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            n_fragments: 3000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 123,
            ..Default::default()
        };
        let result = run_em_on_sim(&params);

        eprintln!(
            "two_sites_balanced: weights={:?}, n_iter={}",
            result.weights, result.n_iter
        );
        // Both APA weights should be in [0.3, 0.7]
        assert!(
            result.weights[1] > 0.3 && result.weights[1] < 0.7,
            "site 1 weight should be ~0.5, got {}",
            result.weights[1]
        );
        assert!(
            result.weights[2] > 0.3 && result.weights[2] < 0.7,
            "site 2 weight should be ~0.5, got {}",
            result.weights[2]
        );
    }

    #[test]
    fn test_two_sites_with_noise() {
        let params = ScapeSimParams {
            utr_length: 3000.0,
            weights: vec![0.1, 0.45, 0.45],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            n_fragments: 5000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 456,
            ..Default::default()
        };
        let result = run_em_on_sim(&params);

        eprintln!(
            "two_sites_with_noise: weights={:?}, n_iter={}",
            result.weights, result.n_iter
        );
        // Noise weight should be detectable but not dominant
        assert!(
            result.weights[0] > 0.01 && result.weights[0] < 0.25,
            "noise weight should be in [0.01, 0.25], got {}",
            result.weights[0]
        );
        assert!(
            result.weights[1] > 0.25,
            "site 1 weight should be >0.25, got {}",
            result.weights[1]
        );
        assert!(
            result.weights[2] > 0.25,
            "site 2 weight should be >0.25, got {}",
            result.weights[2]
        );
    }

    #[test]
    fn test_unbalanced_weights() {
        let params = ScapeSimParams {
            utr_length: 3000.0,
            weights: vec![0.05, 0.7, 0.25],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            n_fragments: 5000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 789,
            ..Default::default()
        };
        let result = run_em_on_sim(&params);

        eprintln!(
            "unbalanced_weights: weights={:?}, n_iter={}",
            result.weights, result.n_iter
        );
        // Site 1 (weight=0.7) should have larger estimated weight than site 2 (weight=0.25)
        assert!(
            result.weights[1] > result.weights[2],
            "site 1 ({}) should have larger weight than site 2 ({})",
            result.weights[1],
            result.weights[2]
        );
    }
}
