use super::*;

/// Shared context for the per-cell scoring routines ([`type_scores`],
/// [`permutation_zscores`]): the feature embedding, the marker pool the null
/// draws from, the unit-normalized cells, and the run config.
pub(super) struct ScoreCtx<'a> {
    pub(super) feature_emb: &'a [f32],
    pub(super) marker_pool: &'a [u32],
    pub(super) cell_u: &'a [f32],
    pub(super) n_cells: usize,
    pub(super) h: usize,
    pub(super) cfg: &'a AnnotateProjConfig,
}

/// Per-cell, per-type score matrix `[N × n_types]`. With a permutation null
/// (`n_perm > 0`) this is the null-standardized z-score and we also return
/// `p = pnorm(-z)`; otherwise it is the raw cosine and `p` is `None`.
pub(super) fn type_scores(
    ctx: &ScoreCtx<'_>,
    type_markers: &[Vec<(u32, f32)>],
    type_emb: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>) {
    let n_types = type_markers.len();
    if ctx.cfg.n_perm > 0 && n_types > 0 {
        let z = permutation_zscores(ctx, type_markers, type_emb);
        let p: Vec<f32> = z.par_iter().map(|&v| pnorm_upper(v)).collect();
        (z, Some(p))
    } else {
        let (cell_u, h, n_cells) = (ctx.cell_u, ctx.h, ctx.n_cells);
        let mut s = vec![0f32; n_cells * n_types];
        s.par_chunks_mut(n_types.max(1))
            .enumerate()
            .for_each(|(n, row)| {
                let cu = &cell_u[n * h..(n + 1) * h];
                for (t, slot) in row.iter_mut().enumerate() {
                    *slot = dot(cu, &type_emb[t * h..(t + 1) * h]);
                }
            });
        (s, None)
    }
}

/// `[C × H]` L2-normalized weighted centroid of each type's marker feature
/// embeddings (parallel over types). Empty types get a zero row.
pub(super) fn type_signatures(
    feature_emb: &[f32],
    n_features: usize,
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
) -> Vec<f32> {
    let n_types = type_markers.len();
    let mut out = vec![0f32; n_types * h];
    out.par_chunks_mut(h)
        .zip(type_markers.par_iter())
        .for_each(|(row, markers)| {
            for &(gi, w) in markers {
                let gi = gi as usize;
                if gi >= n_features {
                    continue;
                }
                let ef = &feature_emb[gi * h..(gi + 1) * h];
                for (r, &e) in row.iter_mut().zip(ef) {
                    *r += w * e;
                }
            }
            let nrm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if nrm > 0.0 {
                for v in row.iter_mut() {
                    *v /= nrm;
                }
            }
        });
    out
}

/// `[N × C]` null-standardized z-scores under a **label-shuffle** null. For
/// each type we draw `n_perm` random gene sets of the same size from the
/// `marker_pool` (the union of every type's marker genes), keeping the type's
/// own weights; each draw's normalized centroid is a null signature. Sampling
/// from the marker pool — not the whole genome — makes the null "another
/// type's markers of the same size" rather than "random background genes", so
/// it cancels the common-mode shared by markers as a class and tests whether
/// THIS type's markers specifically align with the cell. A cell's observed
/// cosine is standardized against the moments of its `n_perm` null cosines.
pub(super) fn permutation_zscores(
    ctx: &ScoreCtx<'_>,
    type_markers: &[Vec<(u32, f32)>],
    type_emb_ch: &[f32],
) -> Vec<f32> {
    let &ScoreCtx {
        feature_emb,
        marker_pool,
        cell_u,
        n_cells,
        h,
        cfg,
    } = ctx;
    let n_types = type_markers.len();
    let n_perm = cfg.n_perm;
    let pool_n = marker_pool.len();

    // C·n_perm null signatures, built in parallel. Each (type, perm) is a
    // deterministic seeded draw of marker-pool genes; the set is accumulated
    // straight into its centroid row — no intermediate marker-list is built.
    let mut null_emb = vec![0f32; n_types * n_perm * h];
    null_emb
        .par_chunks_mut(h)
        .enumerate()
        .for_each(|(idx, row)| {
            let (t, p) = (idx / n_perm, idx % n_perm);
            let markers = &type_markers[t];
            let m = markers.len().min(pool_n);
            let mut rng = SmallRng::seed_from_u64(
                cfg.seed ^ ((t as u64) << 32) ^ (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            // shuffle labels: m random genes from the marker pool, T's weights
            let drawn = rand::seq::index::sample(&mut rng, pool_n, m);
            for (pool_i, &(_, w)) in drawn.iter().zip(markers.iter()) {
                let gidx = marker_pool[pool_i] as usize;
                let ef = &feature_emb[gidx * h..(gidx + 1) * h];
                for (r, &e) in row.iter_mut().zip(ef) {
                    *r += w * e;
                }
            }
            let nrm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if nrm > 0.0 {
                for v in row.iter_mut() {
                    *v /= nrm;
                }
            }
        });

    // Per cell (parallel): standardize observed cosine against null moments.
    let mut zscore_nc = vec![0f32; n_cells * n_types];
    zscore_nc
        .par_chunks_mut(n_types)
        .enumerate()
        .for_each(|(n, zr)| {
            let cu = &cell_u[n * h..(n + 1) * h];
            for t in 0..n_types {
                let obs = f64::from(dot(cu, &type_emb_ch[t * h..(t + 1) * h]));
                // Online mean/variance over the n_perm null cosines.
                let (mut mean, mut m2) = (0f64, 0f64);
                for p in 0..n_perm {
                    let v = t * n_perm + p;
                    let s = f64::from(dot(cu, &null_emb[v * h..(v + 1) * h]));
                    let delta = s - mean;
                    mean += delta / (p as f64 + 1.0);
                    m2 += delta * (s - mean);
                }
                let sd = (m2 / (n_perm as f64).max(1.0)).sqrt().max(1e-6);
                zr[t] = ((obs - mean) / sd) as f32;
            }
        });
    zscore_nc
}

/// Argmax type per row of an `[n × c]` row-major score matrix.
pub(super) fn argmax_rows(score: &[f32], n: usize, c: usize) -> Vec<usize> {
    if c == 0 {
        return vec![0; n];
    }
    (0..n)
        .map(|i| {
            let row = &score[i * c..(i + 1) * c];
            let mut best = 0;
            for j in 1..c {
                if row[j] > row[best] {
                    best = j;
                }
            }
            best
        })
        .collect()
}

/// kNN-graph smoothing of an `[n × c]` row-major score matrix: replace each
/// cell's row by the self-plus-neighbor weighted mean (self weight 1, neighbor
/// weights from the fuzzy kNN graph). One round of local consensus denoises a
/// non-definitive per-cell argmax by borrowing evidence from its neighbourhood
/// — the fine-resolution analogue of the cluster-level coarse layer. Returns a
/// new `[n × c]` buffer. Parallel over cells; the symmetric weighted adjacency
/// is materialized once so there are no write races.
pub(super) fn smooth_scores_over_graph(
    score: &[f32],
    n: usize,
    c: usize,
    graph: &KnnGraph,
) -> Vec<f32> {
    if c == 0 || n == 0 {
        return score.to_vec();
    }
    let w = graph.fuzzy_kernel_weights(); // parallel to graph.edges
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for (&(i, j), &wij) in graph.edges.iter().zip(&w) {
        adj[i].push((j, wij));
        adj[j].push((i, wij));
    }
    let mut out = vec![0f32; n * c];
    out.par_chunks_mut(c).enumerate().for_each(|(i, row)| {
        let mut wsum = 1.0f32; // self
        row.copy_from_slice(&score[i * c..(i + 1) * c]);
        for &(j, wij) in &adj[i] {
            wsum += wij;
            let nbr = &score[j * c..(j + 1) * c];
            for (r, &v) in row.iter_mut().zip(nbr) {
                *r += wij * v;
            }
        }
        let inv = 1.0 / wsum.max(1e-12);
        for r in row.iter_mut() {
            *r *= inv;
        }
    });
    out
}

/// Per-row definitiveness margin: `top1 − top2` of an `[n × c]` score matrix.
/// `c < 2` ⇒ a single type, always definitive (`+∞`). The argmax is fragile
/// exactly when this margin is small (a near-tie between the two best types).
pub(super) fn top2_margin(score: &[f32], n: usize, c: usize) -> Vec<f32> {
    if c < 2 {
        return vec![f32::INFINITY; n];
    }
    (0..n)
        .map(|i| {
            let row = &score[i * c..(i + 1) * c];
            let (mut m1, mut m2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for &v in row {
                if v > m1 {
                    m2 = m1;
                    m1 = v;
                } else if v > m2 {
                    m2 = v;
                }
            }
            m1 - m2
        })
        .collect()
}

/// Normal upper-tail p-value `pnorm(-z) = ½·erfc(z/√2)`. The permutation
/// null cosines are ~Gaussian in high dimension, so this is well-calibrated
/// with no `1/(n_perm+1)` empirical floor.
pub(super) fn pnorm_upper(z: f32) -> f32 {
    use statrs::function::erf::erfc;
    (0.5 * erfc(z as f64 / std::f64::consts::SQRT_2)) as f32
}

#[inline]
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Subtract the column mean from every row of a row-major `[rows × cols]`
/// buffer in place — i.e. remove the **common mode**, the component every row
/// shares. Returns the share of the total sum-of-squares that was removed, for
/// logging.
///
/// # Why a cosine scorer must do this
///
/// A shared offset `m` is by construction the part of the embedding that says
/// nothing about which row is which, but cosine does not ignore it: with
/// `x_i = m + d_i` and `‖m‖ ≫ ‖d_i‖`, every pair is nearly parallel and the
/// discriminative term is attenuated by `‖d‖²/‖m‖²`. Euclidean distance is
/// immune (the offset cancels in `x_a − x_b`), which is why this is a scorer
/// concern and not an embedding-file one.
///
/// Measured on a `faba gem-topic` fit (3 wt libraries, 8791 cells, 20 topics):
/// the cell embedding was **93.8 %** common mode and the co-embedded gene
/// vectors **99.5 %**, so every type signature was very nearly the same unit
/// vector. Mean pairwise cell cosine was `+0.940`, and `+0.007` once centred.
/// Annotation collapsed accordingly — one type took 5145 of 8791 cells with
/// 2208 unassigned; after centring the same fit and the same panel gave a real
/// lineage spread and 490 unassigned, and the worst "called by noise" fraction
/// fell from 70.7 % to 17.1 %.
///
/// The permutation null does not rescue this on its own. Drawing null marker
/// sets from the marker pool was meant to cancel the common mode shared by
/// markers as a class, and it does cancel the *mean* — but when the gene table
/// is 99.5 % common mode, the spread between a real signature and a null one is
/// down at the resampling noise, so the z-score is standardizing against its own
/// noise floor. That is exactly the condition `type_qc` reports as "moves
/// further under marker resampling than the margin its assignment is decided
/// by".
pub(super) fn remove_common_mode(buf: &mut [f32], rows: usize, cols: usize) -> f32 {
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let mut mean = vec![0f64; cols];
    for r in 0..rows {
        for (j, m) in mean.iter_mut().enumerate() {
            *m += f64::from(buf[r * cols + j]);
        }
    }
    for m in mean.iter_mut() {
        *m /= rows as f64;
    }
    let total: f64 = buf[..rows * cols]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    let removed = rows as f64 * mean.iter().map(|&m| m * m).sum::<f64>();
    buf.par_chunks_mut(cols).take(rows).for_each(|row| {
        for (v, &m) in row.iter_mut().zip(mean.iter()) {
            *v -= m as f32;
        }
    });
    if total > 0.0 {
        (removed / total) as f32
    } else {
        0.0
    }
}

/// [`remove_common_mode`] for a column-major `DMatrix`, returning the centred
/// copy and the share of sum-of-squares removed.
///
/// Same rationale, different caller: the term-ORA path reaches the geometry
/// through `DMatrix` (its cell kNN graph and its per-cluster gene ranking), not
/// through the row-major slices [`annotate_by_projection`] uses.
pub(super) fn remove_common_mode_dmat(m: &DMatrix<f32>) -> (DMatrix<f32>, f32) {
    if m.nrows() == 0 || m.ncols() == 0 {
        return (m.clone(), 0.0);
    }
    let mu = m.row_mean();
    let mut out = m.clone();
    for mut row in out.row_iter_mut() {
        row -= &mu;
    }
    let total: f64 = m.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let removed = m.nrows() as f64 * mu.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
    let share = if total > 0.0 {
        (removed / total) as f32
    } else {
        0.0
    };
    (out, share)
}

/// L2-normalize each row of a row-major `[rows × cols]` buffer in place.
pub(super) fn l2_normalize_rows(buf: &mut [f32], rows: usize, cols: usize) {
    buf.par_chunks_mut(cols.max(1)).take(rows).for_each(|row| {
        let nrm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for v in row.iter_mut() {
                *v /= nrm;
            }
        }
    });
}

/// Flatten a column-major `DMatrix` into a row-major `Vec<f32>` (parallel rows).
pub(super) fn row_major(m: &DMatrix<f32>) -> Vec<f32> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut v = vec![0f32; r * c];
    v.par_chunks_mut(c.max(1)).enumerate().for_each(|(i, row)| {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = m[(i, j)];
        }
    });
    v
}
