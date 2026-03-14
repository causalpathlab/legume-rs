use crate::apa::likelihood::{log_lik_fragment_given_site, log_lik_noise};

/// EM algorithm parameters.
pub struct EmParams {
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Convergence tolerance for log-likelihood change
    pub tol: f32,
    /// Minimum component weight before pruning
    pub min_weight: f32,
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
    pub weights: Vec<f32>,
    /// pA site positions alpha_k (for k=1..K)
    pub alphas: Vec<f32>,
    /// pA site dispersions beta_k (for k=1..K)
    pub betas: Vec<f32>,
    /// Posterior assignment probabilities gamma[n][k] (k=0 is noise)
    pub gamma: Vec<Vec<f32>>,
    /// Final log-likelihood
    pub log_lik: f32,
    /// BIC
    pub bic: f32,
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
    alpha_arr: &[f32],
    beta_arr: &[f32],
    theta_lik_matrix: &[Vec<f32>],
    theta_grid: &[f32],
    utr_length: f32,
    max_polya: f32,
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

    // Use generic fixed EM
    let generic_params = crate::mixture::em::EmParams {
        max_iter: params.max_iter,
        tol: params.tol,
        min_weight: params.min_weight,
    };

    let n_free_params = 3 * n_components + 1;
    let result = crate::mixture::em::fixed_em(&component_log_liks, n_free_params, &generic_params);

    EmResult {
        weights: result.weights,
        alphas: alpha_arr.to_vec(),
        betas: beta_arr.to_vec(),
        gamma: result.gamma,
        log_lik: result.log_lik,
        bic: result.bic,
        n_iter: result.n_iter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apa::likelihood::{precompute_theta_lik_matrix, LikelihoodParams};
    use crate::apa::simulate::{simulate_fragments, ScapeSimParams};

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
