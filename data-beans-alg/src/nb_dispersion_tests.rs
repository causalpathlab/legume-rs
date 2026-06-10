use super::*;
use rand::rngs::SmallRng;
use rand::RngExt;
use rand::SeedableRng;

/// Synthesize per-gene (mean, variance) observations under NB with a
/// known dispersion `phi_true`. Variance is set exactly to
/// `mu + phi_true * mu^2` with a small multiplicative jitter to mimic
/// empirical variance noise.
fn synth_nb_moments(
    n_genes: usize,
    phi_true: f32,
    mu_range: (f32, f32),
    jitter: f32,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(n_genes);
    let mut vars = Vec::with_capacity(n_genes);
    let log_lo = mu_range.0.ln();
    let log_hi = mu_range.1.ln();
    for _ in 0..n_genes {
        let log_mu = rng.random_range(log_lo..log_hi);
        let mu = log_mu.exp();
        let var = (mu + phi_true * mu * mu) * (1.0 + rng.random_range(-jitter..jitter));
        means.push(mu);
        vars.push(var.max(1e-6));
    }
    (means, vars)
}

#[test]
fn test_trend_recovers_constant_phi() {
    // When `phi(mu)` is actually constant across genes, the fitted slope
    // `b` should be close to zero and `exp(a)` should be close to the
    // true `phi`.
    let (means, vars) = synth_nb_moments(500, 0.1, (0.1, 50.0), 0.02, 1);
    let trend = DispersionTrend::fit(&means, &vars);
    assert!(trend.num_fit > 100, "expected most genes to contribute");
    assert!(
        trend.b.abs() < 0.2,
        "slope should be near zero, got {}",
        trend.b
    );
    let phi_center = trend.phi_at(5.0);
    assert!(
        (phi_center - 0.1).abs() < 0.05,
        "phi at center should be ≈ 0.1, got {}",
        phi_center
    );
}

#[test]
fn test_fisher_weight_bounds_and_poisson_limit() {
    let (means, vars) = synth_nb_moments(200, 0.2, (1.0, 100.0), 0.01, 2);
    let trend = DispersionTrend::fit(&means, &vars);

    // Bounded in (0, 1].
    for &pi in &[1e-5f32, 1e-3, 1e-1, 0.5] {
        for &mu in &[0.5f32, 5.0, 50.0] {
            let w = trend.fisher_weight(pi, 1000.0, mu);
            assert!(w > 0.0 && w <= 1.0, "weight out of (0, 1]: {}", w);
        }
    }

    // Monotone decreasing in pi.
    let mu = 10.0f32;
    let w_low = trend.fisher_weight(1e-4, 1000.0, mu);
    let w_high = trend.fisher_weight(1e-1, 1000.0, mu);
    assert!(
        w_low > w_high,
        "weight should decrease with pi: low={}, high={}",
        w_low,
        w_high
    );

    // Poisson limit: phi = 0 → w = 1.
    let poisson_trend = DispersionTrend {
        a: f32::NEG_INFINITY, // exp(-inf) = 0
        b: 0.0,
        num_fit: 0,
    };
    let w = poisson_trend.fisher_weight(0.01, 1000.0, 10.0);
    assert!(
        (w - 1.0).abs() < 1e-6,
        "Poisson limit w should be 1, got {}",
        w
    );
}

#[test]
fn test_excess_ranks_outliers_on_top() {
    // Build a dataset with a trend phi_true = 0.1 and inject a handful
    // of outlier genes with phi = 1.0 at the same means. The excess
    // score should rank those outliers above the typical genes.
    let (mut means, mut vars) = synth_nb_moments(300, 0.1, (1.0, 20.0), 0.02, 3);
    let mut outlier_indices = Vec::new();
    for _ in 0..5 {
        let mu = 5.0f32;
        let var = mu + 1.0 * mu * mu; // phi = 1.0
        outlier_indices.push(means.len());
        means.push(mu);
        vars.push(var);
    }
    let trend = DispersionTrend::fit(&means, &vars);
    let mut scored: Vec<(usize, f32)> = means
        .iter()
        .zip(vars.iter())
        .enumerate()
        .map(|(i, (&m, &v))| (i, trend.excess(m, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top5: Vec<usize> = scored.iter().take(5).map(|(i, _)| *i).collect();
    for &outlier in &outlier_indices {
        assert!(
            top5.contains(&outlier),
            "outlier {} not in top-5 HVG set {:?}",
            outlier,
            top5
        );
    }
}

#[test]
fn test_degenerate_input_does_not_panic() {
    // Empty input.
    let trend = DispersionTrend::fit(&[], &[]);
    assert_eq!(trend.num_fit, 0);
    assert_eq!(trend.phi_at(1.0), PHI_FLOOR);

    // All underdispersed.
    let means = vec![1.0f32, 2.0, 3.0];
    let vars = vec![0.5f32, 1.0, 1.5]; // all var < mean
    let trend = DispersionTrend::fit(&means, &vars);
    assert_eq!(trend.num_fit, 0);
    assert_eq!(trend.phi_at(2.0), PHI_FLOOR);

    // All zero means — skipped by MIN_MEAN_FOR_FIT.
    let trend = DispersionTrend::fit(&[0.0f32; 10], &[0.0f32; 10]);
    assert_eq!(trend.num_fit, 0);

    // Single point — degenerate regression.
    let trend = DispersionTrend::fit(&[5.0], &[30.0]);
    assert!(trend.num_fit <= 1);
}
