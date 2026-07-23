use super::*;
use crate::traits::RandomizedAlgs;

/// Three blobs of simplex rows (topic proportions) in `d` dimensions: each
/// blob concentrates its mass on a different pair of coordinates, then every
/// row is renormalized to sum to 1. Deterministic — the jitter is a hash of
/// the row index, so there is no RNG dependence in the fixtures.
fn simplex_blobs(per_blob: usize, d: usize) -> Mat {
    let n = 3 * per_blob;
    let mut m = Mat::zeros(n, d);
    for b in 0..3 {
        for p in 0..per_blob {
            let i = b * per_blob + p;
            let jit = |s: usize| ((i * 7 + s * 13) % 11) as f32 / 11.0;
            for j in 0..d {
                m[(i, j)] = 0.05 + 0.02 * jit(j);
            }
            // Blob-specific mass on two coordinates.
            m[(i, b % d)] += 1.0 + 0.3 * jit(1);
            m[(i, (b + 1) % d)] += 0.6 + 0.3 * jit(2);
            let s: f32 = (0..d).map(|j| m[(i, j)]).sum();
            for j in 0..d {
                m[(i, j)] /= s;
            }
        }
    }
    m
}

/// Pearson correlation of two equal-length vectors.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut num, mut sa, mut sb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        num += (x - ma) * (y - mb);
        sa += (x - ma) * (x - ma);
        sb += (y - mb) * (y - mb);
    }
    num / (sa.sqrt() * sb.sqrt()).max(1e-12)
}

/// The premise behind `skip = 1`: on nonnegative (simplex) rows the leading
/// right singular vector is the *mean profile* — every entry the same sign,
/// and near-perfectly aligned with the column mean. That is what makes
/// dropping it a mean-removal step rather than a loss of structure.
#[test]
fn leading_component_is_the_mean_axis() {
    let data = simplex_blobs(40, 8);
    let (_u, _s, v) = data.rsvd(3).expect("rsvd");

    let v0: Vec<f32> = v.column(0).iter().copied().collect();
    let all_same_sign = v0.iter().all(|&x| x > 0.0) || v0.iter().all(|&x| x < 0.0);
    assert!(all_same_sign, "leading loadings change sign: {v0:?}");

    let col_mean: Vec<f32> = (0..data.ncols())
        .map(|j| data.column(j).mean())
        .collect();
    let cos = {
        let dot: f32 = v0.iter().zip(&col_mean).map(|(a, b)| a * b).sum();
        let nv: f32 = v0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nm: f32 = col_mean.iter().map(|x| x * x).sum::<f32>().sqrt();
        (dot / (nv * nm)).abs()
    };
    assert!(cos > 0.99, "leading component is not the mean axis: cos={cos:.4}");
}

/// Dropping the leading uncentered component recovers what an explicit
/// mean-centering pass would: the two representations' scores agree up to
/// sign on both retained components.
#[test]
fn drop_leading_matches_centered_pca() {
    let data = simplex_blobs(40, 8);
    let dropped = pc_scores(&data, 2, 1).expect("uncentered, skip 1");

    let mut centered = data.clone();
    for j in 0..data.ncols() {
        let mu = data.column(j).mean();
        centered.column_mut(j).add_scalar_mut(-mu);
    }
    let explicit = pc_scores(&centered, 2, 0).expect("centered, skip 0");

    for c in 0..2 {
        let a: Vec<f32> = dropped.column(c).iter().copied().collect();
        let b: Vec<f32> = explicit.column(c).iter().copied().collect();
        let r = pearson(&a, &b).abs();
        assert!(r > 0.95, "component {c} disagrees with centered PCA: |r|={r:.4}");
    }
}

/// The whole point of the SVD init: blobs start out *separated*, so SGD only
/// has to refine local structure. A random draw would interleave them, and
/// the assertion below (max within-blob radius < min between-centroid gap)
/// is exactly what it would fail.
#[test]
fn pc_init_separates_blobs() {
    let per = 40;
    let data = simplex_blobs(per, 8);
    let n = data.nrows();
    let (scores, init) = pc_layout_init(&data, 10, 1, 42);
    assert!(scores.is_some(), "simplex blobs must yield PC scores");
    assert_eq!(init.len(), n * 2);
    assert!(init.iter().all(|v| v.is_finite()), "init must be finite");

    let centroid = |b: usize| {
        let (mut x, mut y) = (0.0f32, 0.0f32);
        for p in 0..per {
            let i = b * per + p;
            x += init[i * 2];
            y += init[i * 2 + 1];
        }
        (x / per as f32, y / per as f32)
    };
    let cens: Vec<(f32, f32)> = (0..3).map(centroid).collect();

    let mut max_within = 0f32;
    for (b, &c) in cens.iter().enumerate() {
        for p in 0..per {
            let i = b * per + p;
            let d = ((init[i * 2] - c.0).powi(2) + (init[i * 2 + 1] - c.1).powi(2)).sqrt();
            max_within = max_within.max(d);
        }
    }
    let mut min_between = f32::INFINITY;
    for a in 0..3 {
        for b in (a + 1)..3 {
            let d = ((cens[a].0 - cens[b].0).powi(2) + (cens[a].1 - cens[b].1).powi(2)).sqrt();
            min_between = min_between.min(d);
        }
    }
    assert!(
        min_between > max_within,
        "PC init did not separate the blobs: within={max_within:.3} between={min_between:.3}"
    );
}

/// The init fills `[-INIT_SCALE, INIT_SCALE]` — it touches the bound (so SGD
/// starts at UMAP's reference scale) without exceeding it by more than the
/// jitter.
#[test]
fn init_is_scaled_to_the_reference_range() {
    let data = simplex_blobs(30, 6);
    let scores = pc_scores(&data, 2, 1).expect("scores");
    let init = init_2d_from_scores(&scores, 7);
    let max_abs = init.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let jitter = INIT_SCALE * JITTER_FRAC;
    assert!(
        (max_abs - INIT_SCALE).abs() <= 2.0 * jitter,
        "init not scaled to ±{INIT_SCALE}: max|coord|={max_abs}"
    );
}

/// Two points that would otherwise coincide are separated by the jitter —
/// coincident points have zero UMAP gradient and could never come apart.
#[test]
fn identical_rows_are_jittered_apart() {
    let mut data = simplex_blobs(20, 6);
    let row0 = data.row(0).into_owned();
    data.row_mut(1).copy_from(&row0);
    let scores = pc_scores(&data, 2, 1).expect("scores");
    let init = init_2d_from_scores(&scores, 11);
    let d = ((init[0] - init[2]).powi(2) + (init[1] - init[3]).powi(2)).sqrt();
    assert!(d > 0.0, "coincident rows left at zero distance");
}

/// Too few features to spare one for the mean: 2 columns minus the leading
/// component leaves a single score, which cannot seed a 2D layout. The
/// fallback reports `None` — the signal a caller uses to keep its kNN graph
/// on the raw features — and still returns a usable random init.
#[test]
fn falls_back_when_no_components_remain() {
    let data = simplex_blobs(20, 2);
    let n = data.nrows();
    let (scores, init) = pc_layout_init(&data, 2, 1, 3);
    assert!(scores.is_none(), "2 features minus the mean axis is not 2D");
    assert_eq!(init.len(), n * 2);
    assert!(init.iter().all(|v| v.abs() <= INIT_SCALE && v.is_finite()));
}

/// `skip` past everything the matrix has is an error, not a silent empty
/// matrix a caller would feed to a kNN graph.
#[test]
fn skipping_every_component_errors() {
    let data = simplex_blobs(10, 3);
    assert!(pc_scores(&data, 2, 3).is_err());
    assert!(pc_scores(&data, 0, 1).is_err());
}
