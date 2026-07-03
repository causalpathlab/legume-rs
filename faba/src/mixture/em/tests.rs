use super::*;

#[test]
fn test_log_sum_exp() {
    let a = 2.0_f32.ln();
    let b = 3.0_f32.ln();
    let result = log_sum_exp(a, b);
    assert!((result - 5.0_f32.ln()).abs() < 1e-5);
}

#[test]
fn test_log_sum_exp_neg_inf() {
    assert_eq!(log_sum_exp(f32::NEG_INFINITY, 1.0), 1.0);
    assert_eq!(log_sum_exp(1.0, f32::NEG_INFINITY), 1.0);
}

#[test]
fn test_gmm_single_cluster() {
    // Generate 100 points around mu=50
    let mut obs = Vec::new();
    let mut rng_state = 42u64;
    for _ in 0..100 {
        // Simple LCG for deterministic test
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (rng_state >> 11) as f32 / (1u64 << 53) as f32;
        // Box-Muller (simplified, using one uniform)
        let x = 50.0 + 5.0 * (2.0 * std::f32::consts::PI * u).cos();
        obs.push(x);
    }

    let params = EmParams {
        max_iter: 100,
        tol: 1e-6,
        min_weight: 0.01,
    };

    let result = gaussian_mixture_em(&obs, &[50.0], 10.0, 100.0, &params);

    assert_eq!(result.mus.len(), 1);
    assert!(
        (result.mus[0] - 50.0).abs() < 10.0,
        "mu should be near 50, got {}",
        result.mus[0]
    );
    assert!(
        result.weights[1] > 0.5,
        "Gaussian weight should dominate, got {}",
        result.weights[1]
    );
}

#[test]
fn test_gmm_two_well_separated_clusters() {
    // Two groups: 50 points around 20, 50 around 80
    let mut obs = Vec::new();
    for i in 0..50 {
        obs.push(20.0 + (i as f32 - 25.0) * 0.5);
    }
    for i in 0..50 {
        obs.push(80.0 + (i as f32 - 25.0) * 0.5);
    }

    let params = EmParams {
        max_iter: 200,
        tol: 1e-6,
        min_weight: 0.01,
    };

    let result = gaussian_mixture_em(&obs, &[20.0, 80.0], 15.0, 100.0, &params);

    assert_eq!(result.mus.len(), 2);
    // Check means are near their true values
    let mut sorted_mus = result.mus.clone();
    sorted_mus.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (sorted_mus[0] - 20.0).abs() < 5.0,
        "first mu should be near 20, got {}",
        sorted_mus[0]
    );
    assert!(
        (sorted_mus[1] - 80.0).abs() < 5.0,
        "second mu should be near 80, got {}",
        sorted_mus[1]
    );
}

#[test]
fn test_fixed_em_basic() {
    // 2 components, 10 observations; flat row-major with stride 2.
    // Component 0 has high likelihood for first 5, component 1 for last 5
    let mut component_log_liks: Vec<f32> = Vec::new();
    for i in 0..10 {
        if i < 5 {
            component_log_liks.extend_from_slice(&[-1.0, -10.0]);
        } else {
            component_log_liks.extend_from_slice(&[-10.0, -1.0]);
        }
    }

    let params = EmParams {
        max_iter: 100,
        tol: 1e-6,
        min_weight: 0.01,
    };

    let result = fixed_em(&component_log_liks, 2, 1, &params);

    assert!(
        (result.weights[0] - 0.5).abs() < 0.1,
        "weight 0 should be ~0.5, got {}",
        result.weights[0]
    );
    assert!(
        (result.weights[1] - 0.5).abs() < 0.1,
        "weight 1 should be ~0.5, got {}",
        result.weights[1]
    );
}
