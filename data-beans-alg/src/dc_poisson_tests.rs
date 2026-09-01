use super::*;
use rand::SeedableRng;

fn toy_gene_sums(
    num_entities: usize,
    num_features: usize,
    labels: &[usize],
    rng_seed: u64,
) -> Vec<Vec<(usize, f32)>> {
    let mut rng = SmallRng::seed_from_u64(rng_seed);
    let num_blocks = *labels.iter().max().unwrap() + 1;
    let per_block = num_features / num_blocks;
    (0..num_entities)
        .map(|e| {
            let c = labels[e];
            let start = c * per_block;
            let end = ((c + 1) * per_block).min(num_features);
            let mut row: Vec<(usize, f32)> = (start..end)
                .map(|g| (g, 5.0 + rng.random_range(0.0..3.0_f32)))
                .collect();
            for _ in 0..3 {
                let g: usize = rng.random_range(0..num_features);
                row.push((g, rng.random_range(0.0..1.0_f32)));
            }
            row.sort_unstable_by_key(|&(g, _)| g);
            row.dedup_by_key(|&mut (g, _)| g);
            row
        })
        .collect()
}

fn make_profiles(gene_sums: &[Vec<(usize, f32)>], num_features: usize) -> Profiles {
    Profiles::from_gene_sums(gene_sums, num_features)
}

#[test]
fn test_log_probs_match_after_delta_moves() {
    let n = 24;
    let m = 16;
    let labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
    let gs = toy_gene_sums(n, m, &labels, 1);
    let profiles = make_profiles(&gs, m);

    let mut stats = DcPoissonStats::from_profiles(&profiles, 4, &labels);
    let mut rng = SmallRng::seed_from_u64(7);
    for _ in 0..50 {
        let e: usize = rng.random_range(0..n);
        let to: usize = rng.random_range(0..4);
        let from = stats.membership[e];
        stats.delta_move(e, from, to, &profiles);
    }

    let kept_mem = stats.membership.clone();
    let mut fresh = DcPoissonStats::from_profiles(&profiles, 4, &kept_mem);
    fresh.recompute(&profiles);

    for i in 0..stats.gene_sum.len() {
        assert!((stats.gene_sum[i] - fresh.gene_sum[i]).abs() < 1e-6);
        assert!((stats.log_gene[i] - fresh.log_gene[i]).abs() < 1e-6);
    }
    for i in 0..stats.size_sum.len() {
        assert!((stats.size_sum[i] - fresh.size_sum[i]).abs() < 1e-6);
        assert!((stats.log_size_offset[i] - fresh.log_size_offset[i]).abs() < 1e-6);
    }
}

#[test]
fn test_restricted_matches_unrestricted_on_allowed() {
    let n = 20;
    let m = 12;
    let labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
    let gs = toy_gene_sums(n, m, &labels, 2);
    let profiles = make_profiles(&gs, m);
    let stats = DcPoissonStats::from_profiles(&profiles, 4, &labels);

    let mut full = vec![f64::NEG_INFINITY; 4];
    let mut restricted = vec![f64::NEG_INFINITY; 4];
    compute_log_probs(3, &stats, &profiles, &mut full);

    let allowed = vec![1usize, 3usize];
    compute_log_probs_restricted(3, &stats, &profiles, &allowed, &mut restricted);
    for &k in &allowed {
        assert!((full[k] - restricted[k]).abs() < 1e-9);
    }
    for (k, &lp) in restricted.iter().enumerate() {
        if !allowed.contains(&k) {
            assert!(lp.is_infinite() && lp < 0.0);
        }
    }
}

#[test]
fn test_empty_block_finite() {
    let gs = vec![vec![(0usize, 3.0f32)], vec![(1usize, 4.0f32)]];
    let profiles = Profiles::from_gene_sums(&gs, 2);
    let stats = DcPoissonStats::from_profiles(&profiles, 3, &[1, 2]);
    assert!(stats.log_size_offset[0].is_finite());
    assert_eq!(stats.size_sum[0], 0.0);
}

#[test]
fn test_delta_move_drift_keeps_logs_finite() {
    // Large-magnitude masses make the incremental f64 accumulator's residues
    // exceed LOG_EPS when a block's mass is drained back to zero; without
    // clamping, `ln(negative)` poisons the log caches with NaN.
    let n = 64;
    let m = 8;
    let k = 4;
    let mut rng = SmallRng::seed_from_u64(11);
    // Wide magnitude spread maximizes rounding in the running sums: adding a
    // tiny value to a huge accumulator loses low bits, so draining a block
    // back to (true) zero leaves a signed residue.
    let gs: Vec<Vec<(usize, f32)>> = (0..n)
        .map(|e| {
            let scale = if e % 3 == 0 { 1.0e-3 } else { 1.0e7 };
            (0..m)
                .map(|g| (g, scale * (1.0 + rng.random_range(0.0..1.0_f32))))
                .collect()
        })
        .collect();
    let profiles = make_profiles(&gs, m);
    let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
    let mut stats = DcPoissonStats::from_profiles(&profiles, k, &labels);
    let mut order: Vec<usize> = (0..n).collect();
    // Churn: repeatedly collapse every entity into one block and scatter
    // back, visiting entities in a fresh random order each round.
    for round in 0..100 {
        order.shuffle(&mut rng);
        for &e in &order {
            let from = stats.membership[e];
            let to = if round % 2 == 0 { round % k } else { e % k };
            stats.delta_move(e, from, to, &profiles);
        }
        for &s in &stats.gene_sum {
            assert!(s >= 0.0, "gene_sum went negative: {s}");
        }
        for &s in &stats.size_sum {
            assert!(s >= 0.0, "size_sum went negative: {s}");
        }
        for &l in &stats.log_gene {
            assert!(l.is_finite(), "log_gene not finite: {l}");
        }
        for &l in &stats.log_size_offset {
            assert!(l.is_finite(), "log_size_offset not finite: {l}");
        }
    }
}

#[test]
fn test_score_is_leave_one_out_for_current_block() {
    // The score of e's current block must equal the score of the same block
    // computed against stats built with e removed — otherwise staying put
    // earns a self-inclusion bonus over every candidate destination.
    let n = 12;
    let m = 10;
    let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();
    let gs = toy_gene_sums(n, m, &labels, 5);
    let profiles = make_profiles(&gs, m);
    let stats = DcPoissonStats::from_profiles(&profiles, 3, &labels);

    let e = 4;
    let k_cur = labels[e];
    let mut lp = vec![f64::NEG_INFINITY; 3];
    compute_log_probs_restricted(e, &stats, &profiles, &[0, 1, 2], &mut lp);

    // Rebuild stats with e parked in a scratch block so k_cur excludes it;
    // scoring e -> k_cur then goes through the ordinary cached path.
    let mut labels_wo = labels.clone();
    labels_wo[e] = 3;
    let stats_wo = DcPoissonStats::from_profiles(&profiles, 4, &labels_wo);
    let mut lp_wo = vec![f64::NEG_INFINITY; 4];
    compute_log_probs_restricted(e, &stats_wo, &profiles, &[k_cur], &mut lp_wo);

    assert!(
        (lp[k_cur] - lp_wo[k_cur]).abs() < 1e-3,
        "current-block score {} differs from leave-one-out score {}",
        lp[k_cur],
        lp_wo[k_cur]
    );
}

fn planted_partition_recovery(parallel: bool) {
    let n = 80;
    let m = 40;
    let k = 4;
    let planted: Vec<usize> = (0..n).map(|i| i % k).collect();
    let gs = toy_gene_sums(n, m, &planted, 9);
    let profiles = make_profiles(&gs, m);

    let mut labels = planted.clone();
    let mut rng = SmallRng::seed_from_u64(3);
    for (e, l) in labels.iter_mut().enumerate() {
        if e % 4 == 0 {
            *l = rng.random_range(0..k);
        }
    }
    let candidates: Vec<Vec<usize>> = vec![(0..k).collect(); n];
    let params = RefineParams {
        parallel,
        ..Default::default()
    };
    let ctx = RefineContext {
        profiles: &profiles,
        k,
        params: &params,
        level_label: "test",
    };
    let mut sweep_rng = SmallRng::seed_from_u64(4);
    refine_with_candidates(&mut labels, &candidates, &mut sweep_rng, &ctx);
    let acc = labels
        .iter()
        .zip(planted.iter())
        .filter(|(a, b)| a == b)
        .count();
    assert!(
        acc as f64 >= 0.95 * n as f64,
        "recovered only {acc}/{n} planted labels (parallel={parallel})"
    );
}

#[test]
fn test_refine_recovers_planted_partition_sequential() {
    planted_partition_recovery(false);
}

#[test]
fn test_refine_recovers_planted_partition_jacobi() {
    planted_partition_recovery(true);
}

#[test]
fn test_compact_labels() {
    let (c, k) = compact_labels(&[5, 5, 2, 7, 2, 7, 5]);
    assert_eq!(k, 3);
    assert_eq!(c, vec![0, 0, 1, 2, 1, 2, 0]);
}

#[test]
fn test_compute_sibling_sets_at_coarsest_gives_all() {
    let refined = vec![
        vec![0usize, 1, 2, 3], // finest
        vec![0usize, 0, 1, 1], // coarsest
    ];
    let top = refined.len() - 1;
    let sibs = compute_sibling_sets(&refined, top, 2);
    assert!(sibs.iter().all(|s| s == &vec![0usize, 1]));
}

#[test]
fn test_compute_sibling_sets_respects_parent() {
    let refined = vec![
        vec![0usize, 1, 2, 3], // finest: 4 groups
        vec![0usize, 0, 1, 1], // parent: {0,1}→0, {2,3}→1
    ];
    let sibs = compute_sibling_sets(&refined, 0, 4);
    assert_eq!(sibs[0], vec![0, 1]);
    assert_eq!(sibs[1], vec![0, 1]);
    assert_eq!(sibs[2], vec![2, 3]);
    assert_eq!(sibs[3], vec![2, 3]);
}
