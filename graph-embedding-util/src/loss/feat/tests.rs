//! The in-place weight refresh a posterior-jitter round applies to a pseudobulk
//! sampler: support fixed, weights redrawn, every picker rebuilt.

use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// 3 pseudobulks × 4 features, every entry nonzero and distinct.
fn count_of(pb: u32, f: u32) -> f32 {
    1.0 + (pb * 4 + f) as f32
}

fn triplets() -> Vec<Triplet> {
    let mut t = Vec::new();
    for pb in 0..3u32 {
        for f in 0..4u32 {
            t.push(Triplet {
                cell: pb,
                feature: f,
                count: count_of(pb, f),
            });
        }
    }
    t
}

fn build() -> StratifiedSampler {
    build_stratified_sampler(&triplets(), 3, 4, 0.5, None).expect("sampler")
}

/// Draw `n` positives with a fixed RNG. The `(pb, feature)` stream is what the
/// pickers encode, so two samplers with equal weights must emit equal streams.
fn positives(s: &StratifiedSampler, n: usize) -> Vec<(u32, u32)> {
    let mut rng = StdRng::seed_from_u64(7);
    let b = sample_stratified_edge_batch(
        StratifiedEdgeBatchArgs {
            sampler: s,
            batch_size: n,
            n_negatives: 2,
            module_pools: None,
        },
        &mut rng,
    );
    b.coarse_cells.into_iter().zip(b.fine_feats).collect()
}

fn degree_negatives(s: &StratifiedSampler, n: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(3);
    (0..n)
        .map(|_| s.feature_pool[s.neg_by_degree.sample(&mut rng)])
        .collect()
}

fn counts(s: &StratifiedSampler) -> Vec<Vec<f32>> {
    s.per_pb.iter().map(|p| p.counts.clone()).collect()
}

/// Redrawing exactly the weights the sampler was built from must change
/// nothing: not the stored weights, not a single draw from any picker.
#[test]
fn rejitter_with_the_mean_is_the_identity() {
    let mut s = build();
    let before_counts = counts(&s);
    let before_pos = positives(&s, 200);
    let before_neg = degree_negatives(&s, 200);
    let change = s.rejitter(4, &|pb, f| count_of(pb, f));
    assert!(
        change.abs() < 1e-7,
        "redrawing the mean itself must change nothing: {change}"
    );
    assert_eq!(before_counts, counts(&s));
    assert_eq!(before_pos, positives(&s, 200));
    assert_eq!(before_neg, degree_negatives(&s, 200));
}

#[test]
fn rejitter_keeps_the_support_and_moves_the_weights() {
    let mut s = build();
    let support: Vec<Vec<u32>> = s.per_pb.iter().map(|p| p.features.clone()).collect();
    let pool = s.feature_pool.clone();
    let active = s.active_pbs.clone();
    // Double feature 0 everywhere, leave the rest.
    let change = s.rejitter(4, &|pb, f| {
        if f == 0 {
            2.0 * count_of(pb, f)
        } else {
            count_of(pb, f)
        }
    });
    assert!(change > 0.0);
    let support_after: Vec<Vec<u32>> = s.per_pb.iter().map(|p| p.features.clone()).collect();
    assert_eq!(support, support_after);
    assert_eq!(pool, s.feature_pool);
    assert_eq!(active, s.active_pbs);
    for (i, p) in s.per_pb.iter().enumerate() {
        let pb = s.active_pbs[i];
        for (j, &f) in p.features.iter().enumerate() {
            let want = if f == 0 {
                2.0 * count_of(pb, f)
            } else {
                count_of(pb, f)
            };
            assert!(
                (p.counts[j] - want).abs() < 1e-6,
                "pb {pb} feature {f}: {} vs {want}",
                p.counts[j]
            );
        }
    }
    // The reported change is `Σ|new − old| / Σ old` over every weight.
    let total_old: f32 = (0..3)
        .flat_map(|pb| (0..4).map(move |f| count_of(pb, f)))
        .sum();
    let delta: f32 = (0..3).map(|pb| count_of(pb, 0)).sum();
    assert!(
        (change - f64::from(delta / total_old)).abs() < 1e-6,
        "change {change}"
    );
}

/// A feature whose draw collapses to zero stays in the negative POOL (support is
/// fixed) but the degree-proportional negatives stop drawing it; the uniform
/// half still does.
#[test]
fn rejitter_refreshes_the_degree_negatives() {
    let mut s = build();
    s.rejitter(4, &|pb, f| if f == 3 { 0.0 } else { count_of(pb, f) });
    assert!(s.feature_pool.contains(&3), "support is fixed");
    let drawn = degree_negatives(&s, 5000);
    assert_eq!(drawn.iter().filter(|&&f| f == 3).count(), 0);
    let mut rng = StdRng::seed_from_u64(1);
    let uniform: Vec<u32> = (0..5000)
        .map(|_| s.feature_pool[s.neg.sample(&mut rng)])
        .collect();
    assert!(uniform.iter().filter(|&&f| f == 3).count() > 500);
}

/// Gene-paired mode weights a gene by the sum of its two rows, at build time
/// and after a refresh alike.
#[test]
fn rejitter_sums_the_paired_rows() {
    // Rows: 0 = gene 0 spliced, 1 = gene 0 unspliced, 2 = gene 1 spliced.
    let row_to_gene = vec![0u32, 0, 1];
    let unspliced = vec![false, true, false];
    let t: Vec<Triplet> = (0..3u32)
        .map(|f| Triplet {
            cell: 0,
            feature: f,
            count: 10.0 * (f + 1) as f32,
        })
        .collect();
    let fp = FeatPairing {
        row_to_gene: &row_to_gene,
        unspliced_rows: &unspliced,
    };
    let mut s = build_stratified_sampler(&t, 1, 3, 0.5, Some(&fp)).expect("sampler");
    assert_eq!(s.per_pb[0].counts, vec![30.0, 30.0]);
    s.rejitter(3, &|_, f| match f {
        0 => 5.0,
        1 => 7.0,
        2 => 11.0,
        _ => unreachable!(),
    });
    assert_eq!(s.per_pb[0].counts, vec![12.0, 11.0]);
}
