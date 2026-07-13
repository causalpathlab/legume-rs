use super::knn_distinct_pbsamples_in_batch;
use matrix_util::knn_match::ColumnDict;
use nalgebra::DMatrix;
use std::collections::HashSet;

/// Build a 1-D batch where the nearest cells collapse into only 3
/// pb-samples, but 12 more pb-samples sit farther out. The adaptive search
/// must still recover `knn` distinct pb-samples — the old fixed `4·knn+1`
/// (41-cell) single shot would have seen only 3.
#[test]
fn adaptive_recovers_knn_distinct_pbsamples() {
    let knn = 10usize;
    let mut feats: Vec<f32> = Vec::new();
    let mut cell_to_pbsamp: Vec<usize> = Vec::new();
    // pb 0,1,2: 20 cells each at x=0,1,2  → 60 dense near cells / 3 pb-samples
    for pb in 0..3 {
        for _ in 0..20 {
            feats.push(pb as f32);
            cell_to_pbsamp.push(pb);
        }
    }
    // pb 3..=14: 3 cells each at x=3..=14 → 12 sparse far pb-samples
    for pb in 3..15 {
        for _ in 0..3 {
            feats.push(pb as f32);
            cell_to_pbsamp.push(pb);
        }
    }
    let n = feats.len();
    let names: Vec<usize> = (0..n).collect(); // name == global cell index
    let mat = DMatrix::<f32>::from_row_slice(1, n, &feats);
    let bknn = ColumnDict::<usize>::from_dmatrix(mat, names);

    let query = vec![0.0f32];
    let hits = knn_distinct_pbsamples_in_batch(&bknn, &query, knn, &cell_to_pbsamp, usize::MAX - 1)
        .unwrap();

    let distinct: HashSet<usize> = hits.iter().map(|&(p, _)| p).collect();
    assert_eq!(
        distinct.len(),
        knn,
        "should recover exactly knn distinct pb-samples, got {distinct:?}"
    );
    // Growth happened: at least one recovered pb-sample lies past the dense
    // near cluster (pb id ≥ 3), unreachable by a single 41-cell fetch.
    assert!(
        distinct.iter().any(|&p| p >= 3),
        "must reach pb-samples beyond the dense near cluster"
    );
    // Distances are sorted ascending and truncated to knn (closest set).
    assert!(hits.windows(2).all(|w| w[0].1 <= w[1].1));
}

/// Fewer pb-samples than `knn` → return all of them (exhaustion), no hang.
#[test]
fn adaptive_returns_all_when_fewer_than_knn() {
    let knn = 10usize;
    let mut feats: Vec<f32> = Vec::new();
    let mut cell_to_pbsamp: Vec<usize> = Vec::new();
    for pb in 0..3 {
        for _ in 0..5 {
            feats.push(pb as f32);
            cell_to_pbsamp.push(pb);
        }
    }
    let n = feats.len();
    let names: Vec<usize> = (0..n).collect();
    let mat = DMatrix::<f32>::from_row_slice(1, n, &feats);
    let bknn = ColumnDict::<usize>::from_dmatrix(mat, names);
    let query = vec![0.0f32];
    let hits = knn_distinct_pbsamples_in_batch(&bknn, &query, knn, &cell_to_pbsamp, usize::MAX - 1)
        .unwrap();
    let distinct: HashSet<usize> = hits.iter().map(|&(p, _)| p).collect();
    assert_eq!(distinct.len(), 3);
}
