//! Seeded k-means on the rows of a dense matrix.
//!
//! k-means++ initialisation from a seeded stream, then Lloyd iterations whose
//! assignment step is chunked over rows and whose per-cluster sums are reduced
//! in chunk order — so the fit depends on the seed and the data, never on how
//! many threads rayon happens to have. Euclidean, or spherical: rows and
//! centroids kept on the unit sphere, where the nearest centroid by Euclidean
//! distance is the nearest by cosine, and a diffuse cluster cannot win rows
//! merely by having a short centroid.
//!
//! No `n × k` distance matrix is ever formed: each row's distances to the `k`
//! centroids are taken into a per-thread scratch of `k` floats — one streaming
//! pass per dimension over a dimension-major centroid table — and only the
//! argmin survives.

use crate::knn::metric::{l2_sq, sqdist_soa_range};
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Rows per assignment task. Fixed rather than derived from the thread count so
/// the chunk partition — and with it every float sum — is the same on any
/// machine.
const CHUNK_ROWS: usize = 4096;

/// A centroid whose mean falls below this norm under the spherical metric has
/// no direction and is treated as empty.
const MIN_DIRECTION_NORM: f64 = 1e-12;

/////////////
// Options //
/////////////

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KmeansMetric {
    Euclidean,
    /// Rows are L2-normalised on entry and centroids after every update.
    Cosine,
}

#[derive(Debug, Clone)]
pub struct KmeansRowsOpts {
    pub k: usize,
    pub max_iter: usize,
    pub seed: u64,
    pub metric: KmeansMetric,
    /// Stop once fewer than this fraction of rows changed label in an
    /// iteration; `0` stops only when no row changed.
    pub min_changed_frac: f64,
    /// Rows the k-means++ seeding draws from: a seeded subsample of this size,
    /// or every row when `0`.
    pub init_sample: usize,
}

pub struct KmeansRowsFit {
    /// `[k × d]`.
    pub centroids: DMatrix<f32>,
    /// One label per row, in `0..k`.
    pub labels: Vec<usize>,
    /// Lloyd iterations run.
    pub n_iter: usize,
}

/// Per-chunk sufficient statistics of one assignment pass.
struct Partial {
    /// `[k × d]` sums of the rows assigned to each centroid.
    sums: Vec<f64>,
    counts: Vec<usize>,
    /// Rows whose label changed.
    changed: usize,
}

///////////
// Entry //
///////////

/// k-means on the rows of `z` (`n × d`). Returns `k` centroids, a label per
/// row and the iteration count. `k ≤ 1`, no rows, or no columns yield one
/// centroid at the column mean (`k.max(1)` rows) and all-zero labels; `k > n`
/// leaves the surplus centroids as duplicates of rows.
pub fn kmeans_rows_seeded(z: &DMatrix<f32>, opts: &KmeansRowsOpts) -> KmeansRowsFit {
    let (n, d, k) = (z.nrows(), z.ncols(), opts.k);
    if k <= 1 || n == 0 || d == 0 {
        return single_centroid(z, k);
    }
    let cosine = opts.metric == KmeansMetric::Cosine;

    // The transpose's column-major storage is exactly row-major `z`: one copy,
    // and every row is a contiguous slice.
    let mut zt = z.transpose();
    if cosine {
        zt.as_mut_slice().par_chunks_mut(d).for_each(normalise_row);
    }
    let rows: &[f32] = zt.as_slice();

    let mut cents = kmeans_pp_init(rows, n, d, k, opts.seed, opts.init_sample);
    // Every row "changes" in the first pass, so the loop always runs once.
    let mut labels = vec![usize::MAX; n];
    let mut nearest = vec![0f32; n];
    let mut n_iter = 0usize;

    for _ in 0..opts.max_iter.max(1) {
        n_iter += 1;

        ////////////////
        // Assignment //
        ////////////////
        // Fixed-size chunks collected in order: the reduction below sums the
        // partials as they appear, so the result never depends on thread count.
        let cents_soa = to_soa(&cents, k, d);
        let Partial {
            mut sums,
            mut counts,
            changed,
        } = assign_all(rows, d, &cents_soa, k, cosine, &mut labels, &mut nearest);

        ///////////////////
        // Centroid step //
        ///////////////////
        let mut next = vec![0f32; k * d];
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            let inv = 1.0 / counts[c] as f64;
            let mean = &mut sums[c * d..(c + 1) * d];
            for m in mean.iter_mut() {
                *m *= inv;
            }
            if cosine {
                let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm < MIN_DIRECTION_NORM {
                    // No direction to speak of: re-seed it below.
                    counts[c] = 0;
                    continue;
                }
                for m in mean.iter_mut() {
                    *m /= norm;
                }
            }
            for (dst, &m) in next[c * d..(c + 1) * d].iter_mut().zip(mean.iter()) {
                *dst = m as f32;
            }
        }
        reseed_empty_clusters(&mut next, &counts, &mut nearest, rows, d);
        cents = next;

        if changed == 0 || (changed as f64) < opts.min_changed_frac * n as f64 {
            break;
        }
    }

    KmeansRowsFit {
        centroids: DMatrix::from_row_slice(k, d, &cents),
        labels,
        n_iter,
    }
}

/// Nearest centroid of every row of `z` under `metric`, for a centroid table
/// fitted elsewhere (say, on a subsample of `z`). Same assignment rule as the
/// Lloyd step: strict `<` from centroid 0, so ties go to the lowest index.
/// Under the spherical metric the rows are normalised before comparing.
pub fn nearest_centroid_rows(
    z: &DMatrix<f32>,
    centroids: &DMatrix<f32>,
    metric: KmeansMetric,
) -> Vec<usize> {
    let (n, d, k) = (z.nrows(), z.ncols(), centroids.nrows());
    assert_eq!(
        centroids.ncols(),
        d,
        "centroids and rows disagree on the dimension"
    );
    if n == 0 || k == 0 || d == 0 {
        return vec![0; n];
    }
    let cosine = metric == KmeansMetric::Cosine;
    let mut zt = z.transpose();
    if cosine {
        zt.as_mut_slice().par_chunks_mut(d).for_each(normalise_row);
    }
    // A column-major `[k × d]` table IS the dimension-major layout the kernel reads.
    let mut labels = vec![usize::MAX; n];
    let mut nearest = vec![0f32; n];
    assign_all(
        zt.as_slice(),
        d,
        centroids.as_slice(),
        k,
        cosine,
        &mut labels,
        &mut nearest,
    );
    labels
}

/// The degenerate return: `k.max(1)` centroid rows with the column mean in
/// row 0, every row labelled 0.
fn single_centroid(z: &DMatrix<f32>, k: usize) -> KmeansRowsFit {
    let (n, d) = (z.nrows(), z.ncols());
    let mut centroids = DMatrix::<f32>::zeros(k.max(1), d);
    if n > 0 {
        for j in 0..d {
            let s: f64 = z.column(j).iter().map(|&v| f64::from(v)).sum();
            centroids[(0, j)] = (s / n as f64) as f32;
        }
    }
    KmeansRowsFit {
        centroids,
        labels: vec![0; n],
        n_iter: 0,
    }
}

fn normalise_row(row: &mut [f32]) {
    let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        row.iter_mut().for_each(|v| *v /= norm);
    }
}

////////////////////
// Initialisation //
////////////////////

/// k-means++ over row-major `rows` (`n × d`): the first centroid uniform, each
/// next one drawn ∝ its squared distance to the nearest centroid so far.
/// Draws come from `SmallRng(seed)`; `init_sample > 0` restricts the candidate
/// pool to a seeded subsample of that many rows. Returns `[k × d]` row-major.
fn kmeans_pp_init(
    rows: &[f32],
    n: usize,
    d: usize,
    k: usize,
    seed: u64,
    init_sample: usize,
) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let pool: Option<Vec<usize>> = (init_sample > 0 && init_sample < n).then(|| {
        let mut ids = rand::seq::index::sample(&mut rng, n, init_sample).into_vec();
        ids.sort_unstable();
        ids
    });
    let m = pool.as_ref().map_or(n, Vec::len);
    let row_of = |p: usize| -> &[f32] {
        let r = pool.as_ref().map_or(p, |ids| ids[p]);
        &rows[r * d..(r + 1) * d]
    };

    let mut cents = vec![0f32; k * d];
    // Squared distance from each candidate to its nearest centroid so far.
    let mut d2 = vec![f32::INFINITY; m];
    let first = rng.random_range(0..m);
    cents[..d].copy_from_slice(row_of(first));
    fold_min_sqdist(&mut d2, &cents[..d], &row_of);

    for c in 1..k {
        let sum: f64 = d2.iter().map(|&x| f64::from(x)).sum();
        let pick = if sum > 0.0 {
            let target = rng.random_range(0.0f64..sum);
            let mut acc = 0f64;
            let mut idx = m - 1;
            for (i, &w) in d2.iter().enumerate() {
                acc += f64::from(w);
                if acc >= target {
                    idx = i;
                    break;
                }
            }
            idx
        } else {
            // Every candidate coincides with a centroid already: any will do.
            rng.random_range(0..m)
        };
        cents[c * d..(c + 1) * d].copy_from_slice(row_of(pick));
        fold_min_sqdist(&mut d2, &cents[c * d..(c + 1) * d], &row_of);
    }
    cents
}

/// Fold a newly added centroid into each candidate's nearest-centroid squared
/// distance. Elementwise, so parallel and serial agree bit for bit.
fn fold_min_sqdist<'a>(
    d2: &mut [f32],
    cent: &[f32],
    row_of: &(impl Fn(usize) -> &'a [f32] + Sync),
) {
    d2.par_iter_mut().enumerate().for_each(|(p, slot)| {
        let dd = l2_sq(row_of(p), cent);
        if dd < *slot {
            *slot = dd;
        }
    });
}

////////////////
// Lloyd step //
////////////////

/// Row-major `[k × d]` to dimension-major `[d × k]`.
fn to_soa(cents: &[f32], k: usize, d: usize) -> Vec<f32> {
    (0..d)
        .flat_map(|dim| (0..k).map(move |c| cents[c * d + dim]))
        .collect()
}

/// One assignment pass over every row: fixed-size chunks in parallel, the
/// partials summed in chunk order, so the result never depends on the thread
/// count. `cents_soa` is the dimension-major `[d × k]` centroid table.
fn assign_all(
    rows: &[f32],
    d: usize,
    cents_soa: &[f32],
    k: usize,
    cosine: bool,
    labels: &mut [usize],
    nearest: &mut [f32],
) -> Partial {
    let partials: Vec<Partial> = rows
        .par_chunks(CHUNK_ROWS * d)
        .zip(labels.par_chunks_mut(CHUNK_ROWS))
        .zip(nearest.par_chunks_mut(CHUNK_ROWS))
        .map(|((xs, ls), ns)| assign_chunk(xs, d, cents_soa, k, cosine, ls, ns))
        .collect();
    let mut total = Partial {
        sums: vec![0f64; k * d],
        counts: vec![0usize; k],
        changed: 0,
    };
    for p in &partials {
        for (s, &v) in total.sums.iter_mut().zip(&p.sums) {
            *s += v;
        }
        for (c, &v) in total.counts.iter_mut().zip(&p.counts) {
            *c += v;
        }
        total.changed += p.changed;
    }
    total
}

/// Assign every row of one chunk to its nearest centroid (strict `<` from
/// centroid 0, so ties go to the lowest index), recording the distance in
/// `nearest` and returning the chunk's sums, counts and change count.
/// `cents_soa` is the dimension-major `[d × k]` centroid table. Under the
/// spherical metric a zero row has no direction and reports distance `0`, so
/// it is never chosen to re-seed an empty cluster.
fn assign_chunk(
    xs: &[f32],
    d: usize,
    cents_soa: &[f32],
    k: usize,
    cosine: bool,
    labels: &mut [usize],
    nearest: &mut [f32],
) -> Partial {
    let mut sums = vec![0f64; k * d];
    let mut counts = vec![0usize; k];
    let mut changed = 0usize;
    let mut dist = vec![0f32; k];
    for (r, x) in xs.chunks_exact(d).enumerate() {
        sqdist_soa_range(cents_soa, k, x, 0, k, &mut dist);
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (c, &dd) in dist.iter().enumerate() {
            if dd < best_d {
                best_d = dd;
                best = c;
            }
        }
        nearest[r] = if cosine && x.iter().all(|&v| v == 0.0) {
            0.0
        } else {
            best_d
        };
        if labels[r] != best {
            changed += 1;
            labels[r] = best;
        }
        counts[best] += 1;
        for (s, &v) in sums[best * d..(best + 1) * d].iter_mut().zip(x) {
            *s += f64::from(v);
        }
    }
    Partial {
        sums,
        counts,
        changed,
    }
}

/// Give every empty cluster the row currently worst served by its centroid,
/// then zero that row's distance so the next empty cluster takes another. The
/// pick is a total order (largest distance, then lowest index), so the parallel
/// reduction is associative and thread-count independent.
fn reseed_empty_clusters(
    next: &mut [f32],
    counts: &[usize],
    nearest: &mut [f32],
    rows: &[f32],
    d: usize,
) {
    for (c, &count) in counts.iter().enumerate() {
        if count > 0 {
            continue;
        }
        let (_, src) = nearest.par_iter().enumerate().map(|(i, &v)| (v, i)).reduce(
            || (f32::NEG_INFINITY, usize::MAX),
            |a, b| {
                if b.0 > a.0 || (b.0 == a.0 && b.1 < a.1) {
                    b
                } else {
                    a
                }
            },
        );
        if src == usize::MAX {
            continue;
        }
        next[c * d..(c + 1) * d].copy_from_slice(&rows[src * d..(src + 1) * d]);
        nearest[src] = 0.0;
    }
}

///////////
// Shims //
///////////

/// Seeded Euclidean k-means on the rows of `z` (cells × D): kmeans++ from a
/// `SmallRng(seed)` then Lloyd iterations, returning `(centroids K×D, labels)`.
/// Reproducible for a given `seed` on any thread count — the substrate
/// `senna lineage --seed` and bootstrap-support scoring rely on. Empty
/// clusters are re-seeded from the point currently worst served by its
/// centroid, so `K` stays non-degenerate; `k ≤ 1` or no rows yields one
/// centroid (the column mean) and all-zero labels.
pub fn kmeans_centroids_seeded(
    z: &DMatrix<f32>,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (DMatrix<f32>, Vec<usize>) {
    let fit = kmeans_rows_seeded(
        z,
        &KmeansRowsOpts {
            k,
            max_iter,
            seed,
            metric: KmeansMetric::Euclidean,
            min_changed_frac: 0.0,
            init_sample: 0,
        },
    );
    (fit.centroids, fit.labels)
}

/// [`kmeans_centroids_seeded`] at a fixed seed, for callers that don't need
/// seed control.
pub fn kmeans_centroids(z: &DMatrix<f32>, k: usize, max_iter: usize) -> (DMatrix<f32>, Vec<usize>) {
    kmeans_centroids_seeded(z, k, max_iter, 42)
}

#[cfg(test)]
mod tests;
