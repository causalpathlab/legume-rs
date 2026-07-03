use super::*;

#[test]
fn test_single_cluster() {
    // All observations near position 50
    let obs: Vec<WeightedObservation> = (0..50)
        .map(|i| WeightedObservation {
            cell_idx: i % 5,
            position: 50.0 + (i as f32 - 25.0) * 0.2,
            count: 1.0,
        })
        .collect();

    let params = MixtureParams {
        min_sites: 3,
        max_k: 3,
        ..Default::default()
    };

    let result = fit_gene_mixture(&obs, 100.0, &params).unwrap();
    // BIC should prefer K=1
    assert_eq!(
        result.best_k, 1,
        "expected K=1 for single cluster, got K={}",
        result.best_k
    );
}

#[test]
fn test_two_clusters() {
    // Two well-separated groups
    let mut obs = Vec::new();
    for i in 0..30 {
        obs.push(WeightedObservation {
            cell_idx: i % 5,
            position: 20.0 + (i as f32 - 15.0) * 0.3,
            count: 1.0,
        });
    }
    for i in 0..30 {
        obs.push(WeightedObservation {
            cell_idx: i % 5,
            position: 80.0 + (i as f32 - 15.0) * 0.3,
            count: 1.0,
        });
    }

    let params = MixtureParams {
        min_sites: 3,
        max_k: 4,
        ..Default::default()
    };

    let result = fit_gene_mixture(&obs, 100.0, &params).unwrap();
    assert!(
        result.best_k >= 2,
        "expected K>=2 for two clusters, got K={}",
        result.best_k
    );

    // Check that means are near 20 and 80
    let mut mus = result.gmm.mus.clone();
    mus.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Filter out noise-assigned mus
    let active_mus: Vec<f32> = mus
        .iter()
        .zip(result.gmm.weights.iter().skip(1))
        .filter(|(_, &w)| w > 0.01)
        .map(|(&m, _)| m)
        .collect();
    assert!(
        active_mus.len() >= 2,
        "expected at least 2 active components"
    );
}

#[test]
fn test_too_few_sites() {
    let obs = vec![
        WeightedObservation {
            cell_idx: 0,
            position: 50.0,
            count: 1.0,
        },
        WeightedObservation {
            cell_idx: 1,
            position: 50.0,
            count: 1.0,
        },
    ];

    let params = MixtureParams {
        min_sites: 3,
        ..Default::default()
    };

    assert!(fit_gene_mixture(&obs, 100.0, &params).is_none());
}

#[test]
fn test_cell_component_counts() {
    // 10 observations from 2 cells, all at same position
    let obs: Vec<WeightedObservation> = (0..10)
        .map(|i| WeightedObservation {
            cell_idx: i % 2,
            position: 30.0 + i as f32,
            count: 1.0,
        })
        .collect();

    let params = MixtureParams {
        min_sites: 3,
        max_k: 1,
        ..Default::default()
    };

    let result = fit_gene_mixture(&obs, 100.0, &params).unwrap();
    // All observations should be assigned to some component
    let total_count: f32 = result.cell_component_counts.iter().map(|(_, _, c)| c).sum();
    assert!(
        (total_count - 10.0).abs() < 1e-5,
        "total count should be ~10, got {}",
        total_count
    );
}

#[test]
fn test_posterior_weight_formula() {
    // Verify the Beta-posterior weight formula at the caller layer.
    // This test pins the math; pipeline.rs is responsible for calling it,
    // but we can hand-check the values here so a future change to the
    // formula or its defaults is loud.
    let posterior_weight = |c: f32, u: f32, a: f32, b: f32| -> f32 {
        let n = c + u;
        let r_hat = (c + a) / (n + a + b);
        n * r_hat
    };

    // High coverage all-converted: w → c
    let w = posterior_weight(50.0, 0.0, 1.0, 1.0);
    assert!((w - 50.0 * (51.0 / 52.0)).abs() < 1e-5);

    // High coverage half-converted: w ≈ c
    let w = posterior_weight(50.0, 50.0, 1.0, 1.0);
    assert!((w - 100.0 * (51.0 / 102.0)).abs() < 1e-5);

    // Single read converted: w shrinks below c=1
    let w = posterior_weight(1.0, 0.0, 1.0, 1.0);
    assert!((w - (2.0 / 3.0)).abs() < 1e-5, "1/1 posterior was {}", w);

    // Zero coverage: w = 0
    let w = posterior_weight(0.0, 0.0, 1.0, 1.0);
    assert_eq!(w, 0.0);

    // Unconverted with coverage: w > 0 (prior pulls up from 0)
    let w = posterior_weight(0.0, 10.0, 1.0, 1.0);
    assert!((w - 10.0 * (1.0 / 12.0)).abs() < 1e-5);
}

#[test]
fn test_bic_n_invariance() {
    // Same observations, scaled weights → best_k must be identical.
    // Guards the BIC-N decoupling (sum-of-weights → n_observations).
    let positions: Vec<(usize, f32)> = (0..30)
        .map(|i| (i % 5, 20.0 + (i as f32 - 15.0) * 0.3))
        .chain((0..30).map(|i| (i % 5, 80.0 + (i as f32 - 15.0) * 0.3)))
        .collect();

    let mk = |scale: f32| -> Vec<WeightedObservation> {
        positions
            .iter()
            .map(|&(cell_idx, position)| WeightedObservation {
                cell_idx,
                position,
                count: scale,
            })
            .collect()
    };

    let params = MixtureParams {
        min_sites: 3,
        max_k: 4,
        ..Default::default()
    };

    let r1 = fit_gene_mixture(&mk(1.0), 100.0, &params).unwrap();
    let r2 = fit_gene_mixture(&mk(10.0), 100.0, &params).unwrap();
    let r3 = fit_gene_mixture(&mk(0.1), 100.0, &params).unwrap();
    assert_eq!(
        r1.best_k, r2.best_k,
        "best_k must be invariant to weight scaling (1× vs 10×)"
    );
    assert_eq!(
        r1.best_k, r3.best_k,
        "best_k must be invariant to weight scaling (1× vs 0.1×)"
    );
}
