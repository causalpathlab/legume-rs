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

    let mut full = vec![0f64; 4];
    let mut restricted = vec![0f64; 4];
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
