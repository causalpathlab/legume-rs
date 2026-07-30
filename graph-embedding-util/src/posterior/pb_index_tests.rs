use super::*;
use nalgebra::DMatrix;

/// Two levels: 2 pb columns then 3, over 4 backend rows. Values are deliberately
/// a mix of zero and nonzero so the sparse pass has something to drop.
fn fixture() -> (Vec<DMatrix<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    // level 0: 4 rows × 2 cols
    let l0 = DMatrix::from_row_slice(
        4,
        2,
        &[
            1.0, 0.0, //
            2.0, 3.0, //
            0.0, 0.0, //
            4.0, 5.0,
        ],
    );
    // level 1: 4 rows × 3 cols
    let l1 = DMatrix::from_row_slice(
        4,
        3,
        &[
            0.0, 1.0, 0.0, //
            0.0, 0.0, 2.0, //
            3.0, 0.0, 0.0, //
            0.0, 4.0, 0.0,
        ],
    );
    let sizes = vec![vec![10.0, 20.0], vec![2.0, 4.0, 8.0]];
    let offsets = vec![0, 2];
    (vec![l0, l1], sizes, offsets)
}

fn stacked<'a>(
    counts: &'a [DMatrix<f32>],
    sizes: &[Vec<f32>],
    offsets: &[usize],
    h: usize,
) -> StackedPb<'a> {
    let n_pb: usize = counts.iter().map(nalgebra::Matrix::ncols).sum();
    StackedPb {
        theta: vec![0.5f32; n_pb * h],
        // Stand-in for `b_pb + ln(size_p)`; the builder only copies it through.
        bias: sizes.iter().flatten().map(|s: &f32| s.ln()).collect(),
        counts: counts.iter().collect(),
        sizes: sizes.to_vec(),
        offsets: offsets.to_vec(),
    }
}

fn feature_side(n_features: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    (
        vec![0.25f32; n_features * h],
        (0..n_features).map(|i| i as f32 * 0.1).collect(),
        (0..n_features).collect(),
    )
}

/// The plan's orientation-agreement check: the same edges, bucketed two ways.
#[test]
fn both_orientations_hold_the_same_edges() {
    let (counts, sizes, offsets) = fixture();
    let (h, n_features) = (3usize, 4usize);
    let pb = stacked(&counts, &sizes, &offsets, h);
    let (e_feat, b_feat, map) = feature_side(n_features, h);
    let feat = FeatureSide {
        e_feat: &e_feat,
        b_feat: &b_feat,
        feature_to_backend_row: &map,
    };

    let pair = build_pb_index_pair(&pb, &feat, None, h, 0, 7).unwrap();

    let by_feature: usize = pair.by_feature.pos.iter().map(Vec::len).sum();
    let by_pb: usize = pair.by_pb.pos.iter().map(Vec::len).sum();
    assert_eq!(
        by_feature, by_pb,
        "the two bucketings must hold the same edges"
    );
    assert_eq!(by_feature, pair.n_edges);

    // 5 nonzeros in level 0 (rows 0,1,1,3,3) + 4 in level 1 (one per row).
    assert_eq!(pair.n_edges, 9);

    // Total mass agrees too, not just the count.
    let mass = |idx: &ContrastiveIndex| -> f64 {
        idx.pos
            .iter()
            .flat_map(|p| p.iter())
            .map(|&(_, v)| f64::from(v))
            .sum()
    };
    let (a, b) = (mass(&pair.by_feature), mass(&pair.by_pb));
    assert!((a - b).abs() < 1e-6, "edge mass disagrees: {a} vs {b}");
}

/// Exposure: an edge must be on the COUNT scale, `rate · size_p`. Getting this
/// wrong is what drove `σ̂² → 0` for the LRT null gate — see the module doc.
#[test]
fn edges_carry_the_size_p_exposure() {
    let (counts, sizes, offsets) = fixture();
    let (h, n_features) = (3usize, 4usize);
    let pb = stacked(&counts, &sizes, &offsets, h);
    let (e_feat, b_feat, map) = feature_side(n_features, h);
    let feat = FeatureSide {
        e_feat: &e_feat,
        b_feat: &b_feat,
        feature_to_backend_row: &map,
    };

    let pair = build_pb_index_pair(&pb, &feat, None, h, 0, 7).unwrap();

    // Row 3 has level-0 rates [4, 5] against sizes [10, 20] → counts [40, 100],
    // and a level-1 rate 4 at column 1 (global 3) against size 4 → 16.
    let mut row3: Vec<(u32, f32)> = pair.by_feature.pos[3].clone();
    row3.sort_by_key(|&(o, _)| o);
    assert_eq!(row3, vec![(0, 40.0), (1, 100.0), (3, 16.0)]);

    // Same three edges show up transposed, under the matching pb anchors.
    assert!(pair.by_pb.pos[0].contains(&(3, 40.0)));
    assert!(pair.by_pb.pos[1].contains(&(3, 100.0)));
    assert!(pair.by_pb.pos[3].contains(&(3, 16.0)));
}

/// An axis wide relative to the cap must slate, and the scale must fold the slate back up
/// to the pool it was drawn from — not to the raw axis length, since the normalizer runs
/// over the EXPRESSED axis.
///
/// `n_partition = 1` against a 5-pb axis puts it past `EXACT_FACTOR × n_partition = 4`.
/// The 4-row feature axis sits exactly at the threshold and so stays exact, which
/// incidentally checks that the two sides are decided independently.
#[test]
fn a_wide_axis_slates_and_scales_to_its_pool() {
    let (counts, sizes, offsets) = fixture();
    let (h, n_features) = (3usize, 4usize);
    let pb = stacked(&counts, &sizes, &offsets, h);
    let (e_feat, b_feat, map) = feature_side(n_features, h);
    let feat = FeatureSide {
        e_feat: &e_feat,
        b_feat: &b_feat,
        feature_to_backend_row: &map,
    };

    let pair = build_pb_index_pair(&pb, &feat, None, h, 1, 7).unwrap();

    assert_eq!(
        pair.by_feature.partition.len(),
        1,
        "pb side slated to the cap"
    );
    assert!(
        (pair.by_feature.partition_scale - 5.0).abs() < 1e-12,
        "scale folds the 5-pb pool up from a 1-entry slate, got {}",
        pair.by_feature.partition_scale
    );
    // 4 expressed rows <= 4 x 1, so the feature side is summed outright.
    assert_eq!(
        pair.by_pb.partition.len(),
        4,
        "feature side should be exact here"
    );
    assert!((pair.by_pb.partition_scale - 1.0).abs() < 1e-12);
}

/// An axis cheap relative to the cap is summed EXACTLY, which is what removes the Jensen
/// bias in the profiled log-normalizer: `ln` of a sampled sum underestimates `ln` of the
/// true sum, so an exact axis has no such gap to correct.
#[test]
fn a_cheap_axis_is_summed_exactly() {
    let (counts, sizes, offsets) = fixture();
    let (h, n_features) = (3usize, 4usize);
    let pb = stacked(&counts, &sizes, &offsets, h);
    let (e_feat, b_feat, map) = feature_side(n_features, h);
    let feat = FeatureSide {
        e_feat: &e_feat,
        b_feat: &b_feat,
        feature_to_backend_row: &map,
    };

    // A cap of 1024 dwarfs both axes, so neither slates.
    let pair = build_pb_index_pair(&pb, &feat, None, h, 1024, 7).unwrap();
    assert!((pair.by_feature.partition_scale - 1.0).abs() < 1e-12);
    assert!((pair.by_pb.partition_scale - 1.0).abs() < 1e-12);
    assert_eq!(pair.by_feature.partition.len(), 5);
    assert_eq!(pair.by_pb.partition.len(), 4);
}

/// The slate pool is the EXPRESSED axis, matching the trainer, whose negatives are drawn
/// uniformly over its expressed `feature_pool`. A row observed nowhere has a prior-driven
/// embedding, so what it contributes to a normalizer is arbitrary — noise entering every
/// anchor's score with nothing behind it.
#[test]
fn the_slate_pool_excludes_rows_observed_nowhere() {
    // Row 2 is already zero in level 0; blank it in level 1 too so it is unobserved.
    let (mut counts, sizes, offsets) = fixture();
    for m in &mut counts {
        for s in 0..m.ncols() {
            m[(2, s)] = 0.0;
        }
    }
    let (h, n_features) = (3usize, 4usize);
    let pb = stacked(&counts, &sizes, &offsets, h);
    let (e_feat, b_feat, map) = feature_side(n_features, h);
    let feat = FeatureSide {
        e_feat: &e_feat,
        b_feat: &b_feat,
        feature_to_backend_row: &map,
    };

    let pair = build_pb_index_pair(&pb, &feat, None, h, 1024, 7).unwrap();
    assert!(
        !pair.by_pb.partition.contains(&2),
        "an unobserved row must not be in the normalizer pool: {:?}",
        pair.by_pb.partition
    );
    assert_eq!(
        pair.by_pb.partition.len(),
        3,
        "3 of 4 rows are expressed, so the exact sum is over 3"
    );
    // The other three survive, so this is not a blanket exclusion.
    for r in [0u32, 1, 3] {
        assert!(pair.by_pb.partition.contains(&r), "row {r} is expressed");
    }
}

/// A row dropped by the anchor map leaves the GENE side but must stay in every
/// pb's edge list — a pseudobulk is scored against the whole feature axis.
#[test]
fn dropped_rows_leave_the_gene_side_but_not_the_pb_side() {
    let (counts, sizes, offsets) = fixture();
    let (h, n_features) = (3usize, 4usize);
    let pb = stacked(&counts, &sizes, &offsets, h);
    let (e_feat, b_feat, map) = feature_side(n_features, h);
    let feat = FeatureSide {
        e_feat: &e_feat,
        b_feat: &b_feat,
        feature_to_backend_row: &map,
    };

    // Rows 0 and 1 pool into gene 0; rows 2 and 3 are dropped from the gene side.
    let row_to_anchor = vec![0u32, 0, u32::MAX, u32::MAX];
    let anchors = AnchorMap {
        row_to_anchor: &row_to_anchor,
        n_anchors: 1,
    };
    let pair = build_pb_index_pair(&pb, &feat, Some(&anchors), h, 0, 7).unwrap();

    // Rows 0+1: level-0 nonzeros (1,0),(2,3) → 3 edges; level-1 (·,1,·),(·,·,2) → 2.
    assert_eq!(pair.by_feature.pos.len(), 1);
    assert_eq!(pair.by_feature.pos[0].len(), 5);

    // The pb side still holds all 9.
    let by_pb: usize = pair.by_pb.pos.iter().map(Vec::len).sum();
    assert_eq!(by_pb, 9);
    assert_eq!(
        pair.n_edges, 9,
        "n_edges counts observed data, not gene-side keeps"
    );

    // Pooling adds: gene 0 gets rows 0 and 1 at pb column 0 as two separate edges.
    let at_pb0: Vec<f32> = pair.by_feature.pos[0]
        .iter()
        .filter(|&&(o, _)| o == 0)
        .map(|&(_, v)| v)
        .collect();
    assert_eq!(at_pb0, vec![10.0, 20.0], "rate 1×10 and rate 2×10");
}

/// The slate must be a random sample, not the low-index prefix. `partial_shuffle`
/// puts its sample at the END of the slice, so discarding the return value and
/// truncating keeps exactly the wrong half — and the symptom is invisible in any
/// shape or count check.
#[test]
fn the_negative_slate_is_a_random_sample_not_the_head() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let n = 4000usize;
    let k = 200usize;
    let mut rng = StdRng::seed_from_u64(9);
    let pool: Vec<u32> = (0..n as u32).collect();
    let slate = super::sample_slate_from(&pool, k, &mut rng);

    assert_eq!(slate.len(), k);
    let mut sorted = slate.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), k, "indices must be distinct");

    // The broken form returned ~all of 0..k. A uniform sample of 200 from 4000
    // puts ~10 below 200; anything near k is the prefix bug.
    let in_head = slate.iter().filter(|&&i| (i as usize) < k).count();
    assert!(
        in_head < k / 4,
        "{in_head} of {k} slate entries are in the first {k} indices — this is the \
         truncate-the-wrong-half bug, not a sample"
    );
    // And it should actually reach the far end of the axis.
    assert!(
        slate.iter().any(|&i| (i as usize) > n / 2),
        "slate never reaches the upper half of the axis"
    );
}
