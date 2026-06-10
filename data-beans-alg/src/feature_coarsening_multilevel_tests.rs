use super::*;
use matrix_util::traits::SampleOps;
use rand::RngExt;
use rand::SeedableRng;

fn planted_sketch(d_per_block: usize, s_per_block: usize, k: usize, seed: u64) -> DMatrix<f32> {
    let d = d_per_block * k;
    let s = s_per_block * k;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut x = DMatrix::<f32>::rnorm(d, s);
    x *= 0.2;
    for block in 0..k {
        let row_lo = block * d_per_block;
        let row_hi = (block + 1) * d_per_block;
        let col_lo = block * s_per_block;
        let col_hi = (block + 1) * s_per_block;
        for f in row_lo..row_hi {
            for j in col_lo..col_hi {
                x[(f, j)] = 5.0 + rng.random_range(0.0..2.0_f32);
            }
        }
    }
    x
}

fn assert_partitions_features(fc: &FeatureCoarsening, d: usize) {
    assert_eq!(fc.fine_to_coarse.len(), d);
    let mut seen = vec![false; d];
    for group in &fc.coarse_to_fine {
        for &f in group {
            assert!(!seen[f], "feature {f} appears in two groups");
            seen[f] = true;
        }
    }
    assert!(seen.iter().all(|&x| x));
}

fn assert_nesting(ml: &MultilevelFeatureCoarsening) {
    for level in 1..ml.levels.len() {
        let parent = ml
            .parent_at(level)
            .expect("nesting check")
            .expect("non-coarsest level must have parent");
        let fine = &ml.levels[level];
        let coarse = &ml.levels[level - 1];
        assert_eq!(parent.len(), fine.num_coarse);
        for (f, &fine_grp) in fine.fine_to_coarse.iter().enumerate() {
            assert_eq!(coarse.fine_to_coarse[f], parent[fine_grp]);
        }
    }
    assert!(ml.parent_at(0).unwrap().is_none());
}

#[test]
fn rough_init_preserves_d_and_nests() {
    let sketch = planted_sketch(40, 8, 5, 7);
    let d = sketch.nrows();
    let targets = vec![5usize, 12, 30];
    let knn = FeatureKnnContext::from_sketch(&sketch, 12).unwrap();
    let ml = compute_multilevel_feature_coarsening(&sketch, &targets, &knn).unwrap();
    assert_eq!(ml.levels.len(), targets.len());
    for fc in &ml.levels {
        assert_partitions_features(fc, d);
    }
    for w in ml.levels.windows(2) {
        assert!(w[0].num_coarse <= w[1].num_coarse);
    }
    assert!(ml.levels.last().unwrap().num_coarse >= 5);
    assert!(ml.levels.last().unwrap().num_coarse <= d);
    assert_nesting(&ml);
}

#[test]
fn refinement_preserves_d_and_nesting() {
    let sketch = planted_sketch(30, 10, 4, 13);
    let d = sketch.nrows();
    let targets = vec![4usize, 10, 20];
    let knn = FeatureKnnContext::from_sketch(&sketch, 8).unwrap();
    let init = compute_multilevel_feature_coarsening(&sketch, &targets, &knn).unwrap();
    let mut params = MultilevelRefineParams::default();
    params.dc_poisson.num_gibbs = 3;
    params.dc_poisson.num_greedy = 2;
    params.dc_poisson.gibbs_stagnation = 0.0;
    let refined = refine_multilevel_feature_coarsening(&sketch, init, &knn, &params).unwrap();
    assert_eq!(refined.levels.len(), targets.len());
    for fc in &refined.levels {
        assert_partitions_features(fc, d);
    }
    assert_nesting(&refined);
    for w in refined.levels.windows(2) {
        assert!(w[0].num_coarse <= w[1].num_coarse);
    }
}

#[test]
fn duplicate_level_targets_share_one_snapshot() {
    // Targets [10, 10, 30] used to panic — both 10s competed for the
    // same group_count threshold and one stayed unsnapshotted.
    let sketch = planted_sketch(20, 8, 5, 31);
    let targets = vec![10usize, 10, 30];
    let knn = FeatureKnnContext::from_sketch(&sketch, 8).unwrap();
    let ml = compute_multilevel_feature_coarsening(&sketch, &targets, &knn).unwrap();
    assert_eq!(ml.levels.len(), 3);
    assert_eq!(ml.levels[0].num_coarse, ml.levels[1].num_coarse);
}
