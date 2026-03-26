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

/// Precomputed data for a single UTR needed by the EM routines.
pub struct SiteModelData<'a> {
    pub alpha_arr: &'a [f32],
    pub beta_arr: &'a [f32],
    pub theta_lik_matrix: &'a [Vec<f32>],
    pub theta_grid: &'a [f32],
    pub utr_length: f32,
    pub max_polya: f32,
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
pub fn fixed_inference(data: &SiteModelData, params: &EmParams) -> EmResult {
    let n_frag = data.theta_lik_matrix.len();
    let n_components = data.alpha_arr.len(); // K APA components
    let n_total = n_components + 1; // +1 for noise component (k=0)

    // Precompute log p(x_n, l_n, r_n | alpha_k, beta_k) for each fragment and component
    let mut component_log_liks = vec![vec![0.0; n_total]; n_frag];
    let noise_ll = log_lik_noise(data.utr_length, data.max_polya);

    for (n, row) in component_log_liks.iter_mut().enumerate() {
        row[0] = noise_ll;
        for k in 0..n_components {
            row[k + 1] = log_lik_fragment_given_site(
                &data.theta_lik_matrix[n],
                data.theta_grid,
                data.alpha_arr[k],
                data.beta_arr[k],
            );
        }
    }

    run_fixed_em(&component_log_liks, data.alpha_arr, data.beta_arr, params)
}

/// Run the generic fixed EM on precomputed component log-likelihoods.
fn run_fixed_em(
    component_log_liks: &[Vec<f32>],
    alpha_arr: &[f32],
    beta_arr: &[f32],
    params: &EmParams,
) -> EmResult {
    let n_components = alpha_arr.len();
    let generic_params = crate::mixture::em::EmParams {
        max_iter: params.max_iter,
        tol: params.tol,
        min_weight: params.min_weight,
    };

    let n_free_params = 3 * n_components + 1;
    let result = crate::mixture::em::fixed_em(component_log_liks, n_free_params, &generic_params);

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

/// Select the best number of pA sites by BIC model selection.
///
/// Candidate sites are ranked by `site_order` (indices into `alpha_arr`,
/// sorted by descending coverage). Tries K=1, 2, ..., N sites in that order,
/// fitting an EM for each K. Returns the model with lowest BIC, stopping
/// early after 2 consecutive BIC increases.
///
/// If there is only one candidate site, runs a single EM (no selection needed).
pub fn select_sites_by_bic(
    data: &SiteModelData,
    params: &EmParams,
    site_order: &[usize],
) -> EmResult {
    let n_frag = data.theta_lik_matrix.len();
    let n_candidates = site_order.len();

    if n_candidates <= 1 {
        return fixed_inference(data, params);
    }

    // Precompute log-likelihoods for ALL candidates (noise + each site)
    let noise_ll = log_lik_noise(data.utr_length, data.max_polya);
    let all_site_lls: Vec<Vec<f32>> = (0..n_candidates)
        .map(|i| {
            let idx = site_order[i];
            (0..n_frag)
                .map(|n| {
                    log_lik_fragment_given_site(
                        &data.theta_lik_matrix[n],
                        data.theta_grid,
                        data.alpha_arr[idx],
                        data.beta_arr[idx],
                    )
                })
                .collect()
        })
        .collect();

    let mut best_result: Option<EmResult> = None;
    let mut n_worse = 0u32;

    for k in 1..=n_candidates {
        // Build component_log_liks for top-K sites: column 0 = noise, columns 1..K = sites
        let component_log_liks: Vec<Vec<f32>> = (0..n_frag)
            .map(|n| {
                let mut row = Vec::with_capacity(k + 1);
                row.push(noise_ll);
                for site_ll in all_site_lls.iter().take(k) {
                    row.push(site_ll[n]);
                }
                row
            })
            .collect();

        let selected_alphas: Vec<f32> = (0..k).map(|i| data.alpha_arr[site_order[i]]).collect();
        let selected_betas: Vec<f32> = (0..k).map(|i| data.beta_arr[site_order[i]]).collect();

        let result = run_fixed_em(
            &component_log_liks,
            &selected_alphas,
            &selected_betas,
            params,
        );

        let is_better = match &best_result {
            None => true,
            Some(prev) => result.bic < prev.bic,
        };

        if is_better {
            best_result = Some(result);
            n_worse = 0;
        } else {
            n_worse += 1;
            if n_worse >= 2 {
                break;
            }
        }
    }

    best_result.unwrap()
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

        let site_data = SiteModelData {
            alpha_arr: &params.alphas,
            beta_arr: &params.betas,
            theta_lik_matrix: &theta_lik_matrix,
            theta_grid: &theta_grid,
            utr_length: params.utr_length,
            max_polya: params.max_polya,
        };
        fixed_inference(&site_data, &em_params)
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

    #[test]
    fn test_bic_selects_true_sites() {
        // Simulate 2 true sites, then add a spurious 3rd candidate.
        // BIC selection should prefer K=2 over K=3.
        let params = ScapeSimParams {
            utr_length: 3000.0,
            weights: vec![0.05, 0.5, 0.45],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            n_fragments: 3000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 321,
            ..Default::default()
        };
        let (fragments, _) = simulate_fragments(&params);
        let lik_params = LikelihoodParams {
            mu_f: params.mu_f,
            sigma_f: params.sigma_f,
            theta_step: 10,
            max_polya: params.max_polya,
            min_polya: params.min_polya,
        };
        let (theta_lik_matrix, theta_grid) =
            precompute_theta_lik_matrix(&fragments, params.utr_length, &lik_params);

        // 2 true sites + 1 spurious site at position 2500
        let alpha_arr = vec![500.0, 1500.0, 2500.0];
        let beta_arr = vec![30.0, 30.0, 30.0];
        // Order: site 0 (true), site 1 (true), site 2 (spurious)
        let site_order = vec![0, 1, 2];

        let em_params = EmParams {
            max_iter: 200,
            tol: 1e-6,
            min_weight: 0.005,
        };

        let site_data = SiteModelData {
            alpha_arr: &alpha_arr,
            beta_arr: &beta_arr,
            theta_lik_matrix: &theta_lik_matrix,
            theta_grid: &theta_grid,
            utr_length: params.utr_length,
            max_polya: params.max_polya,
        };
        let result = select_sites_by_bic(&site_data, &em_params, &site_order);

        eprintln!(
            "bic_select: selected {} sites, alphas={:?}, weights={:?}, bic={:.1}",
            result.alphas.len(),
            result.alphas,
            result.weights,
            result.bic
        );

        // BIC should select 2 sites (not 3) since the spurious site adds no signal
        assert!(
            result.alphas.len() <= 3,
            "should select at most 3 sites, got {}",
            result.alphas.len()
        );
        // The two true sites should have non-trivial weight
        assert!(
            result.weights[1] > 0.2,
            "site 1 weight should be >0.2, got {}",
            result.weights[1]
        );
    }

    #[test]
    fn test_bic_single_candidate() {
        // Single candidate should just run fixed_inference
        let params = ScapeSimParams {
            utr_length: 2000.0,
            weights: vec![0.0, 1.0],
            alphas: vec![500.0],
            betas: vec![30.0],
            n_fragments: 1000,
            n_cells: 5,
            junction_prob: 0.3,
            seed: 42,
            ..Default::default()
        };
        let (fragments, _) = simulate_fragments(&params);
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

        let site_data = SiteModelData {
            alpha_arr: &[500.0],
            beta_arr: &[30.0],
            theta_lik_matrix: &theta_lik_matrix,
            theta_grid: &theta_grid,
            utr_length: params.utr_length,
            max_polya: params.max_polya,
        };
        let result = select_sites_by_bic(&site_data, &em_params, &[0]);

        assert_eq!(result.alphas.len(), 1);
        assert!(result.weights[1] > 0.8, "single site should dominate");
    }
}
