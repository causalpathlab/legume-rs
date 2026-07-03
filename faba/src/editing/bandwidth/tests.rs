use super::*;

#[test]
fn test_empty_falls_back_to_floor() {
    let est = estimate_bandwidth(&[], &BandwidthParams::m6a());
    assert_eq!(est.n_gaps, 0);
    assert_eq!(est.bandwidth, BandwidthParams::m6a().min_bandwidth);
}

#[test]
fn test_recovers_small_mode_for_clustered_sites() {
    // Two genes, each a tight cluster: sites every 20nt. Median gap = 20.
    let gene_a: Vec<(f32, f32)> = (0..5).map(|i| (1000.0 + i as f32 * 20.0, 10.0)).collect();
    let gene_b: Vec<(f32, f32)> = (0..5).map(|i| (5000.0 + i as f32 * 20.0, 10.0)).collect();
    let per_gene = vec![gene_a, gene_b];

    // m6a: scale 1 → ~20, clamped up to floor 10 (20 > 10, stays 20).
    let m6a = estimate_bandwidth(&per_gene, &BandwidthParams::m6a());
    assert!(
        (m6a.bandwidth - 20.0).abs() < 1e-3,
        "m6a bw {}",
        m6a.bandwidth
    );

    // atoi: scale 3 → ~60 (cluster-aware, wider).
    let atoi = estimate_bandwidth(&per_gene, &BandwidthParams::atoi());
    assert!(
        (atoi.bandwidth - 60.0).abs() < 1e-3,
        "atoi bw {}",
        atoi.bandwidth
    );
}

#[test]
fn test_marginal_sites_downweighted() {
    // A dense well-covered cluster (gap 10, high weight) plus one far,
    // low-weight outlier creating a huge gap. The weighted median should
    // track the dense cluster, not the outlier gap.
    let mut gene: Vec<(f32, f32)> = (0..6).map(|i| (100.0 + i as f32 * 10.0, 50.0)).collect();
    gene.push((100_000.0, 0.01)); // far, near-zero signal
    let est = estimate_bandwidth(&[gene], &BandwidthParams::m6a());
    // Median gap should be ~10 (clamped to floor 10), not dominated by the
    // ~99_850 outlier gap.
    assert!(
        est.bandwidth <= 12.0,
        "bw should track dense cluster, got {}",
        est.bandwidth
    );
}
