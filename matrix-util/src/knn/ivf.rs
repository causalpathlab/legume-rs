//! Approximate all-rows k-nearest neighbours by an inverted-file search.
//!
//! [`super::all_pairs::knn_rows_l2`] is exact and `O(n²·d)`: the right tool up
//! to tens of thousands of rows, and no tool at all beyond that. Here the rows
//! are first partitioned into cells by seeded k-means (`√n` of them by
//! default), and each row is then searched only among the rows of its
//! `n_probe` nearest cells, by direct differences. A query sees
//! `n_probe · n / n_lists` candidates, so the whole search is `O(n^1.5 · d)`
//! at the default, and it is exact whenever every cell is probed.
//!
//! The rows are copied once into cell order and stored dimension-major, so a
//! probed cell is a contiguous range in every dimension and the distances
//! from one query to a whole cell are `d` streaming passes over that range —
//! no per-row reduction, nothing gathered. At millions of rows that layout,
//! not the arithmetic count, sets the wall clock.
//!
//! Every query is scored on its own against cells fixed by the seeded,
//! thread-independent partition, so the result depends on the seed and the
//! data and not on how rayon schedules the queries. Nearest first, ties by
//! index, self excluded. Rows are expected to be finite.

use super::all_pairs::{by_distance_then_index, keep_smallest, sort_key};
use crate::kmeans::{kmeans_rows_seeded, nearest_centroid_rows, KmeansMetric, KmeansRowsOpts};
use crate::knn::metric::sqdist_soa_range;
use crate::utils::generate_minibatch_intervals;
use log::info;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Cells probed per query when the caller does not say.
pub const DEFAULT_N_PROBE: usize = 16;
/// Lloyd iterations for the partition: a rough tessellation is all the search
/// needs, and the probes cover its seams.
const PARTITION_ITER: usize = 10;
/// Rows per cell the partition is fitted on. The centroids are trained on a
/// seeded subsample of `PARTITION_ROWS_PER_CELL · n_lists` rows and every row
/// is then assigned once; fitting on all rows would cost a full assignment
/// pass per Lloyd iteration for a tessellation the probes smooth over anyway.
const PARTITION_ROWS_PER_CELL: usize = 256;
/// Rows per cell the k-means++ seeding draws its candidates from.
const PARTITION_INIT_PER_CELL: usize = 16;
/// The partition stops once fewer than this fraction of its rows still move.
const PARTITION_MIN_CHANGED: f64 = 1e-3;
/// Consecutive queries searched together, so a probed cell is streamed once
/// for all of them.
const QUERY_BLOCK: usize = 256;

#[derive(Debug, Clone)]
pub struct IvfArgs {
    /// Neighbours per row; clamped to the other rows.
    pub k: usize,
    /// Cells; `0` picks `⌈√n⌉`. Clamped to `n`.
    pub n_lists: usize,
    /// Cells searched per query; clamped to `n_lists`.
    pub n_probe: usize,
    /// Seeds the partition.
    pub seed: u64,
}

/// `k` approximate nearest neighbours of every row of `x` among the other
/// rows: `(indices, Euclidean distances)` per row, nearest first.
pub fn knn_rows_ivf(x: &DMatrix<f32>, args: &IvfArgs) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let (n, d) = (x.nrows(), x.ncols());
    let k = args.k.min(n.saturating_sub(1));
    if n == 0 || k == 0 {
        return (vec![Vec::new(); n], vec![Vec::new(); n]);
    }
    let n_lists = if args.n_lists == 0 {
        (n as f64).sqrt().ceil() as usize
    } else {
        args.n_lists
    }
    .clamp(1, n);

    ///////////////
    // Partition //
    ///////////////
    let t_partition = std::time::Instant::now();
    let train_rows = (PARTITION_ROWS_PER_CELL * n_lists).min(n);
    let opts = KmeansRowsOpts {
        k: n_lists,
        max_iter: PARTITION_ITER,
        seed: args.seed,
        metric: KmeansMetric::Euclidean,
        min_changed_frac: PARTITION_MIN_CHANGED,
        // The k-means++ pick walks a serial prefix sum over its candidates
        // once per centroid; a seeded subsample keeps that off the clock.
        init_sample: PARTITION_INIT_PER_CELL * n_lists,
    };
    let fit = if train_rows < n {
        let mut rng = StdRng::seed_from_u64(args.seed);
        let mut ids = rand::seq::index::sample(&mut rng, n, train_rows).into_vec();
        ids.sort_unstable();
        let mut fit = kmeans_rows_seeded(&x.select_rows(&ids), &opts);
        fit.labels = nearest_centroid_rows(x, &fit.centroids, opts.metric);
        fit
    } else {
        kmeans_rows_seeded(x, &opts)
    };
    let labels = fit.labels;
    let n_probe = args.n_probe.clamp(1, n_lists);

    // Inverted lists: row ids grouped by cell, ascending within a cell.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.par_sort_unstable_by_key(|&i| (labels[i as usize], i));
    let mut offsets = vec![0usize; n_lists + 1];
    for &l in &labels {
        offsets[l + 1] += 1;
    }
    for c in 0..n_lists {
        offsets[c + 1] += offsets[c];
    }
    let widest_cell = (0..n_lists)
        .map(|c| offsets[c + 1] - offsets[c])
        .max()
        .unwrap_or(0);

    // Dimension-major copies in cell order: `soa[dim · n + p]` is coordinate
    // `dim` of the row at position `p`, so a cell is one contiguous range per
    // dimension. `x` is column-major, so each dimension is a gather from one
    // contiguous column.
    let mut soa = vec![0f32; n * d];
    soa.par_chunks_mut(n).enumerate().for_each(|(dim, dst)| {
        let col = x.column(dim);
        for (v, &i) in dst.iter_mut().zip(&order) {
            *v = col[i as usize];
        }
    });
    // A column-major `[n_lists × d]` table is already dimension-major.
    let cents_soa: &[f32] = fit.centroids.as_slice();
    info!(
        "IVF kNN: {n} rows x {d} into {n_lists} cells fitted on {train_rows} rows in {} Lloyd \
         iterations ({:.1} s); probing {n_probe} per query",
        fit.n_iter,
        t_partition.elapsed().as_secs_f64()
    );

    ////////////
    // Search //
    ////////////
    let t_search = std::time::Instant::now();
    let blocks = generate_minibatch_intervals(n, 0, Some(QUERY_BLOCK));
    let bar = crate::progress::new_progress_bar(blocks.len() as u64).with_message(format!(
        "IVF kNN {n} x {d}, k={k}, {n_lists} cells, {n_probe} probed"
    ));
    let index = Index {
        soa: &soa,
        n,
        d,
        cents_soa,
        n_lists,
        order: &order,
        offsets: &offsets,
    };
    // Query blocks walk the cell order, so a block's queries probe mostly the
    // same cells and each cell is streamed once per block rather than once
    // per query; the results are scattered back to row order at the end.
    let found: Vec<(Vec<usize>, Vec<f32>)> = blocks
        .into_par_iter()
        .map_init(
            || Scratch::new(d, widest_cell.max(n_lists), n_probe, k),
            |scratch, (p0, p1)| {
                let out = search_block(&index, p0, p1, n_probe, k, scratch);
                bar.inc(1);
                out
            },
        )
        .flatten()
        .collect();
    bar.finish_and_clear();
    info!(
        "IVF kNN: searched {n} queries in {:.1} s",
        t_search.elapsed().as_secs_f64()
    );
    let mut indices = vec![Vec::new(); n];
    let mut distances = vec![Vec::new(); n];
    for (p, (nb, ds)) in found.into_iter().enumerate() {
        let i = order[p] as usize;
        indices[i] = nb;
        distances[i] = ds;
    }
    (indices, distances)
}

/// The partitioned rows, borrowed for the search.
struct Index<'a> {
    /// `[d × n]` dimension-major in cell order.
    soa: &'a [f32],
    n: usize,
    d: usize,
    /// `[d × n_lists]` dimension-major.
    cents_soa: &'a [f32],
    n_lists: usize,
    /// Row id at each position.
    order: &'a [u32],
    /// Cell `c` occupies positions `offsets[c]..offsets[c + 1]`.
    offsets: &'a [usize],
}

/// Per-thread buffers reused across query blocks.
struct Scratch {
    /// The block's query coordinates, `[B × d]` row-major.
    q: Vec<f32>,
    /// Squared distances to one range of rows (or to every centroid).
    dist: Vec<f32>,
    probes: Vec<(f32, usize)>,
    /// `(cell, query)` for every probe in the block, sorted by cell.
    hits: Vec<(usize, usize)>,
    /// Each query's short list.
    cand: Vec<Vec<(f32, usize)>>,
}

impl Scratch {
    fn new(d: usize, widest: usize, n_probe: usize, k: usize) -> Self {
        Self {
            q: vec![0f32; QUERY_BLOCK * d],
            dist: vec![0f32; widest],
            probes: Vec::with_capacity(n_probe + 1),
            hits: Vec::with_capacity(QUERY_BLOCK * n_probe),
            cand: (0..QUERY_BLOCK)
                .map(|_| Vec::with_capacity(k + 1))
                .collect(),
        }
    }
}

/// The queries at positions `p0..p1`: each one's `n_probe` nearest cells,
/// then every cell any of them probes is streamed once and scored for each
/// query that asked for it. Returns row ids and true distances, in position
/// order.
fn search_block(
    index: &Index<'_>,
    p0: usize,
    p1: usize,
    n_probe: usize,
    k: usize,
    scratch: &mut Scratch,
) -> Vec<(Vec<usize>, Vec<f32>)> {
    let (n, d) = (index.n, index.d);
    let b = p1 - p0;
    let Scratch {
        q,
        dist,
        probes,
        hits,
        cand,
    } = scratch;

    // Coordinates and nearest cells per query, ties by index.
    hits.clear();
    for (i, qi) in q[..b * d].chunks_exact_mut(d).enumerate() {
        for (dim, qd) in qi.iter_mut().enumerate() {
            *qd = index.soa[dim * n + p0 + i];
        }
        sqdist_soa_range(index.cents_soa, index.n_lists, qi, 0, index.n_lists, dist);
        probes.clear();
        for (c, &dd) in dist[..index.n_lists].iter().enumerate() {
            keep_smallest(probes, (sort_key(dd), c), n_probe);
        }
        hits.extend(probes.iter().map(|&(_, c)| (c, i)));
        cand[i].clear();
    }
    // Grouped by cell: each probed cell is streamed once, for the queries
    // that asked for it.
    hits.sort_unstable();
    for group in hits.chunk_by(|a, b| a.0 == b.0) {
        let c = group[0].0;
        let (lo, hi) = (index.offsets[c], index.offsets[c + 1]);
        for &(_, i) in group {
            let p = p0 + i;
            sqdist_soa_range(index.soa, n, &q[i * d..(i + 1) * d], lo, hi, dist);
            let list = &mut cand[i];
            for (j, &dd) in dist[..hi - lo].iter().enumerate() {
                let pos = lo + j;
                // One comparison rejects almost every candidate before the
                // sorted insert is even considered.
                if pos != p && (list.len() < k || dd < list[k - 1].0) {
                    keep_smallest(list, (sort_key(dd), index.order[pos] as usize), k);
                }
            }
        }
    }

    cand[..b]
        .iter_mut()
        .map(|list| {
            list.sort_unstable_by(by_distance_then_index);
            list.iter().map(|&(d2, j)| (j, d2.sqrt())).unzip()
        })
        .collect()
}

#[cfg(test)]
#[path = "ivf_tests.rs"]
mod tests;
