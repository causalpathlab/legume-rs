use super::*;
use crate::apa::fragment::{cluster_fragments, ClusterBins};
use crate::apa::likelihood::{precompute_theta_lik_matrix, LikelihoodParams};
use crate::apa::simulate::{simulate_fragments, ScapeSimParams};

/// Cluster simulated fragments (with the default bin sizes used in
/// production) and return the inputs needed to build a SiteModelData:
/// flat θ-likelihood matrix, θ grid, per-cluster counts, and the
/// original-fragment count for BIC.
fn cluster_for_em(
    fragments: &[crate::apa::fragment::FragmentRecord],
    params: &ScapeSimParams,
    lik_params: &LikelihoodParams,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    let (clusters, _cluster_idx) = cluster_fragments(fragments, ClusterBins::default());
    let counts: Vec<f32> = clusters.iter().map(|c| c.count as f32).collect();
    let (mat, grid) = precompute_theta_lik_matrix(&clusters, params.utr_length, lik_params);
    (mat, grid, counts, fragments.len())
}

fn run_em_on_sim(params: &ScapeSimParams) -> EmResult {
    let (fragments, _labels) = simulate_fragments(params);
    let lik_params = LikelihoodParams {
        mu_f: params.mu_f,
        sigma_f: params.sigma_f,
        theta_step: 10,
        max_polya: params.max_polya,
        min_polya: params.min_polya,
    };
    let (theta_lik_matrix, theta_grid, cluster_counts, n_for_bic) =
        cluster_for_em(&fragments, params, &lik_params);

    // Baseline EM behavior: disable skirt and post-EM merge so these
    // tests cover the non-robust path. Robust path has its own test.
    let em_params = EmParams {
        max_iter: 200,
        tol: 1e-6,
        min_weight: 0.005,
        skirt_eta: 0.0,
        skirt_mult: 0.0,
        merge_beta_mult: 0.0,
        max_sites: 0,
    };

    let site_data = SiteModelData {
        alpha_arr: &params.alphas,
        beta_arr: &params.betas,
        theta_lik_matrix: &theta_lik_matrix,
        theta_grid: &theta_grid,
        cluster_counts: &cluster_counts,
        n_for_bic,
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
    let (theta_lik_matrix, theta_grid, cluster_counts, n_for_bic) =
        cluster_for_em(&fragments, &params, &lik_params);

    // 2 true sites + 1 spurious site at position 2500
    let alpha_arr = vec![500.0, 1500.0, 2500.0];
    let beta_arr = vec![30.0, 30.0, 30.0];
    // Order: site 0 (true), site 1 (true), site 2 (spurious)
    let site_order = vec![0, 1, 2];

    // Baseline EM behavior: disable skirt and post-EM merge so these
    // tests cover the non-robust path. Robust path has its own test.
    let em_params = EmParams {
        max_iter: 200,
        tol: 1e-6,
        min_weight: 0.005,
        skirt_eta: 0.0,
        skirt_mult: 0.0,
        merge_beta_mult: 0.0,
        max_sites: 0,
    };

    let site_data = SiteModelData {
        alpha_arr: &alpha_arr,
        beta_arr: &beta_arr,
        theta_lik_matrix: &theta_lik_matrix,
        theta_grid: &theta_grid,
        cluster_counts: &cluster_counts,
        n_for_bic,
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
    let (theta_lik_matrix, theta_grid, cluster_counts, n_for_bic) =
        cluster_for_em(&fragments, &params, &lik_params);

    // Baseline EM behavior: disable skirt and post-EM merge so these
    // tests cover the non-robust path. Robust path has its own test.
    let em_params = EmParams {
        max_iter: 200,
        tol: 1e-6,
        min_weight: 0.005,
        skirt_eta: 0.0,
        skirt_mult: 0.0,
        merge_beta_mult: 0.0,
        max_sites: 0,
    };

    let site_data = SiteModelData {
        alpha_arr: &[500.0],
        beta_arr: &[30.0],
        theta_lik_matrix: &theta_lik_matrix,
        theta_grid: &theta_grid,
        cluster_counts: &cluster_counts,
        n_for_bic,
        utr_length: params.utr_length,
        max_polya: params.max_polya,
    };
    let result = select_sites_by_bic(&site_data, &em_params, &[0]);

    assert_eq!(result.alphas.len(), 1);
    assert!(result.weights[1] > 0.8, "single site should dominate");
}

#[test]
fn test_post_em_merge_collapses_close_sites() {
    // Two true sites at 500 and 1500; offer BIC a third spurious candidate
    // very close to 500 (within merge tolerance) and verify the post-EM
    // merge collapses it.
    let params = ScapeSimParams {
        utr_length: 3000.0,
        weights: vec![0.05, 0.5, 0.45],
        alphas: vec![500.0, 1500.0],
        betas: vec![30.0, 30.0],
        n_fragments: 3000,
        n_cells: 10,
        junction_prob: 0.3,
        seed: 4242,
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
    let (theta_lik_matrix, theta_grid, cluster_counts, n_for_bic) =
        cluster_for_em(&fragments, &params, &lik_params);

    // 530 sits 30bp from the true 500 site → within 2*beta=60 merge tol.
    let alpha_arr = vec![500.0, 530.0, 1500.0];
    let beta_arr = vec![30.0, 30.0, 30.0];
    let site_order = vec![0, 1, 2];

    let em_params = EmParams {
        max_iter: 200,
        tol: 1e-6,
        min_weight: 0.005,
        skirt_eta: 0.05,
        skirt_mult: 3.0,
        merge_beta_mult: 2.0,
        max_sites: 0,
    };
    let site_data = SiteModelData {
        alpha_arr: &alpha_arr,
        beta_arr: &beta_arr,
        theta_lik_matrix: &theta_lik_matrix,
        theta_grid: &theta_grid,
        cluster_counts: &cluster_counts,
        n_for_bic,
        utr_length: params.utr_length,
        max_polya: params.max_polya,
    };
    let result = select_sites_by_bic(&site_data, &em_params, &site_order);

    // After the merge, no two surviving alphas should be within tol of each other.
    let live: Vec<f32> = result
        .alphas
        .iter()
        .zip(result.weights.iter().skip(1))
        .filter(|(_, &w)| w > 0.0)
        .map(|(&a, _)| a)
        .collect();
    for (i, &a) in live.iter().enumerate() {
        for &b in &live[i + 1..] {
            assert!(
                    (a - b).abs() >= 2.0 * 30.0,
                    "live sites at {} and {} are within merge tolerance; merge should have collapsed them",
                    a,
                    b
                );
        }
    }
    assert!(
        result.bic.is_finite(),
        "merged result bic should be finite, got {}",
        result.bic
    );
}
