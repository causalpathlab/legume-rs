//! Per-pair Bernoulli logistic log-likelihood for cage-mcmc.
//!
//! Mirrors `graph_embedding_util::loss::cell_cell_nce_loss_per_level_batched_gated`
//! up to sign (returns a positive log-likelihood; the cage loss returns
//! the negative). For each sampled positive pair `(u, v)` at gene `g`,
//! chain level `ℓ`, with `K` sibling negatives `w_k`:
//!
//! ```text
//!   gated_gene = softplus_floored(γ[ℓ,:]) ⊙ e_gene[g,:]
//!   proj_u     = e_cell[u,:] · gated_gene
//!   proj_v     = e_cell[v,:] · gated_gene
//!   proj_w_k   = e_cell[w_k,:] · gated_gene
//!   pos_score  = proj_u · proj_v + b_cell[u] + b_cell[v]
//!   neg_score_k = proj_u · proj_w_k + b_cell[u] + b_cell[w_k]
//!   ll_pair    = log σ(pos_score) + Σ_k log σ(-neg_score_k)
//! ```
//!
//! Performance notes:
//!
//! - `DMatrix<f32>` is column-major, so iterating across one *row*'s
//!   columns strides through memory and trashes the cache. We repack
//!   `e_cell` and `e_gene` to row-major `Vec<f32>` once per
//!   `loglik_total` call so the inner dot reads contiguous slices and
//!   the compiler can vectorize.
//! - The `b_cell` ESS block sees the same minibatch and the same
//!   `(e_cell, e_gene, γ)` across all bracket evaluations. The expensive
//!   `proj_u · proj_v` / `proj_u · proj_w_k` interaction terms are then
//!   constant. `BiasCache` precomputes them once; `loglik_with_bias`
//!   reuses the tape so each bracket eval reduces to bias add +
//!   `log_sigmoid`.
//! - Outer gene loop is rayon-parallelized; inner is single-threaded
//!   (matches `feedback_rayon_nesting`).

use graph_embedding_util::loss::CellChainBatch;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use super::model::softplus_floored;

////////////////////////////////////////////////////////////////
//                                                            //
// Row-major repack of column-major nalgebra::DMatrix         //
//                                                            //
////////////////////////////////////////////////////////////////

/// Row-major flattening of a `DMatrix<f32>`. Row `i` is a contiguous
/// `&[f32]` of length `cols`, so the inner dot is cache-friendly and
/// auto-SIMD-able.
pub(super) struct RowMajor {
    data: Vec<f32>,
    pub(super) cols: usize,
}

impl RowMajor {
    pub(super) fn from_dmatrix(m: &DMatrix<f32>) -> Self {
        let rows = m.nrows();
        let cols = m.ncols();
        let mut data = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            for c in 0..cols {
                data[base + c] = m[(r, c)];
            }
        }
        Self { data, cols }
    }
    #[inline]
    pub(super) fn row(&self, i: usize) -> &[f32] {
        let s = i * self.cols;
        &self.data[s..s + self.cols]
    }
}

/// Const-generic dot kernel. Fully unrolled at the call site for the
/// specialized `D`; the compiler emits dense SIMD (NEON / AVX2) with
/// no length dispatch and no chunk loop.
#[inline(always)]
fn dot_d<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut s = 0.0f32;
    for j in 0..D {
        s += a[j] * b[j];
    }
    s
}

/// 8-wide SIMD dot product + scalar tail. Fallback path for embedding
/// dims that don't match a const-generic specialization.
#[inline(always)]
fn dot_wide(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let head = chunks * 8;

    let mut acc = wide::f32x8::ZERO;
    let a_chunks = a[..head].chunks_exact(8);
    let b_chunks = b[..head].chunks_exact(8);
    for (av, bv) in a_chunks.zip(b_chunks) {
        let ax = wide::f32x8::from(<[f32; 8]>::try_from(av).unwrap());
        let bx = wide::f32x8::from(<[f32; 8]>::try_from(bv).unwrap());
        acc = ax.mul_add(bx, acc);
    }
    let mut s = acc.reduce_add();
    for j in head..n {
        s += a[j] * b[j];
    }
    s
}

/// Runtime-length dot. Dispatches to a fully unrolled const-generic
/// kernel for the common `--embedding-dim` values (4, 8, 16, 32),
/// otherwise falls back to the `wide::f32x8` + scalar-tail path.
/// `PINTO_CGM_CONST_DOT=0` forces the wide-only path for A/B testing.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if !CONST_DOT.load(std::sync::atomic::Ordering::Relaxed) {
        return dot_wide(a, b);
    }
    match a.len() {
        4 => dot_d::<4>(a.try_into().unwrap(), b.try_into().unwrap()),
        8 => dot_d::<8>(a.try_into().unwrap(), b.try_into().unwrap()),
        16 => dot_d::<16>(a.try_into().unwrap(), b.try_into().unwrap()),
        32 => dot_d::<32>(a.try_into().unwrap(), b.try_into().unwrap()),
        _ => dot_wide(a, b),
    }
}

static CONST_DOT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Read `PINTO_CGM_CONST_DOT` once and cache the result. Call at the
/// top of `fit_cage_mcmc`. Default is ON.
pub fn init_const_dot_from_env() -> bool {
    let on = std::env::var("PINTO_CGM_CONST_DOT")
        .map(|v| v.trim() != "0")
        .unwrap_or(true);
    CONST_DOT.store(on, std::sync::atomic::Ordering::Relaxed);
    on
}

/// `log σ(s)` with a polynomial-tail fast path.
///
/// - `|s| > 16`: `exp(-|s|) < 1.2e-7` so `softplus(-|s|) ≈ 0` and the
///   identity reduces to `min(s, 0)` exactly.
/// - `|s| ≤ 16` and `exp(-|s|) < 0.2`: replace `ln_1p(exp(-|s|))` with
///   its 5-term Taylor expansion. One `exp`, no `ln_1p`. Max-abs error
///   < 1e-4 across `s ∈ [-20, 20]` (pinned by
///   `fast_log_sigmoid_matches_exact`).
/// - Otherwise: standard `exp().ln_1p()` path.
#[inline]
fn log_sigmoid(s: f32) -> f32 {
    let x = s.abs();
    if x > 16.0 {
        return s.min(0.0);
    }
    let e = (-x).exp();
    let sp_neg_x = if e < 0.2 {
        let e2 = e * e;
        let e3 = e2 * e;
        let e4 = e3 * e;
        let e5 = e4 * e;
        e - 0.5 * e2 + (1.0 / 3.0) * e3 - 0.25 * e4 + 0.2 * e5
    } else {
        e.ln_1p()
    };
    s.min(0.0) - sp_neg_x
}

/// Exact reference `log σ` retained for the accuracy unit test.
#[cfg(test)]
fn log_sigmoid_exact(s: f32) -> f32 {
    s.min(0.0) - (-s.abs()).exp().ln_1p()
}

////////////////////////////////////////////////////////////////
//                                                            //
// Full log-likelihood (used by e_cell / e_gene / γ blocks)   //
//                                                            //
////////////////////////////////////////////////////////////////

fn loglik_one(
    e_cell: &RowMajor,
    e_gene: &RowMajor,
    gated_gamma: &DMatrix<f32>,
    b_cell: &DVector<f32>,
    gene_id: usize,
    cb: &CellChainBatch,
    gated_gene_scratch: &mut [f32],
) -> f32 {
    let d = e_cell.cols;
    debug_assert!(gated_gene_scratch.len() >= d);
    let n_levels = gated_gamma.nrows();
    let pos_count = cb.left_cells.len();
    let k = cb.n_negatives;
    let e_gene_row = e_gene.row(gene_id);

    let mut acc = 0.0f32;
    for lvl in 0..n_levels {
        for j in 0..d {
            gated_gene_scratch[j] = gated_gamma[(lvl, j)] * e_gene_row[j];
        }
        let gg = &gated_gene_scratch[..d];
        let neg_for_lvl = &cb.per_level_neg[lvl];
        for b in 0..pos_count {
            let u = cb.left_cells[b] as usize;
            let v = cb.right_cells[b] as usize;
            let proj_u = dot(e_cell.row(u), gg);
            let proj_v = dot(e_cell.row(v), gg);
            let bu = b_cell[u];
            let pos_score = proj_u * proj_v + bu + b_cell[v];
            acc += log_sigmoid(pos_score);

            let base = b * k;
            for kk in 0..k {
                let w = neg_for_lvl[base + kk] as usize;
                let proj_w = dot(e_cell.row(w), gg);
                let neg_score = proj_u * proj_w + bu + b_cell[w];
                acc += log_sigmoid(-neg_score);
            }
        }
    }
    acc
}

/// Full positive log-likelihood given already-packed row-major
/// embeddings and an already-positivized γ. Use this from per-block
/// scopes that own the packed buffers and want to avoid re-packing
/// the *fixed* blocks on every ESS bracket evaluation.
pub(super) fn loglik_total_packed(
    e_cell_rm: &RowMajor,
    e_gene_rm: &RowMajor,
    gated_gamma: &DMatrix<f32>,
    b_cell: &DVector<f32>,
    chunks: &[Vec<(usize, CellChainBatch)>],
) -> f32 {
    let d = e_cell_rm.cols;
    chunks
        .par_iter()
        .map(|chunk| {
            let mut scratch = vec![0.0f32; d];
            let mut acc = 0.0f32;
            for (gene_id, cb) in chunk {
                acc += loglik_one(
                    e_cell_rm,
                    e_gene_rm,
                    gated_gamma,
                    b_cell,
                    *gene_id,
                    cb,
                    &mut scratch,
                );
            }
            acc
        })
        .sum()
}

/// Full positive log-likelihood. Packs the column-major DMatrix
/// embeddings into row-major buffers once, then rayon-parallelizes
/// over gene chunks.
pub fn loglik_total(
    e_cell: &DMatrix<f32>,
    e_gene: &DMatrix<f32>,
    gamma: &DMatrix<f32>,
    b_cell: &DVector<f32>,
    chunks: &[Vec<(usize, CellChainBatch)>],
) -> f32 {
    let e_cell_rm = RowMajor::from_dmatrix(e_cell);
    let e_gene_rm = RowMajor::from_dmatrix(e_gene);
    let gated_gamma = softplus_floored(gamma);
    loglik_total_packed(&e_cell_rm, &e_gene_rm, &gated_gamma, b_cell, chunks)
}

////////////////////////////////////////////////////////////////
//                                                            //
// Bias-only fast path (used by the b_cell ESS block)         //
//                                                            //
////////////////////////////////////////////////////////////////

/// Precomputed interaction term `proj_u·proj_v` (positive) and
/// `proj_u·proj_w_k` (negative) for one minibatch. Built once with
/// the current `(e_cell, e_gene, γ)`; reused across all bracket
/// evaluations of the `b_cell` ESS step where only the bias changes.
///
/// **Invariant**: `tapes[i]` is in lockstep with `chunks[i]` — walk
/// order is `(item, lvl, b)` and emits one positive interaction
/// followed by `K` negative interactions, identical to [`loglik_one`].
/// [`loglik_with_bias`] relies on this without bounds-checking.
pub struct BiasCache<'a> {
    pub chunks: &'a [Vec<(usize, CellChainBatch)>],
    pub tapes: Vec<Vec<f32>>,
}

pub fn build_bias_cache<'a>(
    e_cell: &DMatrix<f32>,
    e_gene: &DMatrix<f32>,
    gamma: &DMatrix<f32>,
    chunks: &'a [Vec<(usize, CellChainBatch)>],
) -> BiasCache<'a> {
    let e_cell_rm = RowMajor::from_dmatrix(e_cell);
    let e_gene_rm = RowMajor::from_dmatrix(e_gene);
    let gated_gamma = softplus_floored(gamma);
    let d = e_cell_rm.cols;
    let n_levels = gated_gamma.nrows();

    let tapes: Vec<Vec<f32>> = chunks
        .par_iter()
        .map(|chunk| {
            let tape_cap: usize = chunk
                .iter()
                .map(|(_, cb)| cb.left_cells.len() * cb.per_level_neg.len() * (1 + cb.n_negatives))
                .sum();
            let mut tape = Vec::with_capacity(tape_cap);
            let mut gated_gene = vec![0.0f32; d];
            for (gene_id, cb) in chunk {
                let pos_count = cb.left_cells.len();
                let k = cb.n_negatives;
                let e_gene_row = e_gene_rm.row(*gene_id);
                for lvl in 0..n_levels {
                    for j in 0..d {
                        gated_gene[j] = gated_gamma[(lvl, j)] * e_gene_row[j];
                    }
                    let gg = &gated_gene[..d];
                    let neg_for_lvl = &cb.per_level_neg[lvl];
                    for b in 0..pos_count {
                        let u = cb.left_cells[b] as usize;
                        let v = cb.right_cells[b] as usize;
                        let proj_u = dot(e_cell_rm.row(u), gg);
                        let proj_v = dot(e_cell_rm.row(v), gg);
                        tape.push(proj_u * proj_v);
                        let base = b * k;
                        for kk in 0..k {
                            let w = neg_for_lvl[base + kk] as usize;
                            let proj_w = dot(e_cell_rm.row(w), gg);
                            tape.push(proj_u * proj_w);
                        }
                    }
                }
            }
            tape
        })
        .collect();

    BiasCache { chunks, tapes }
}

/// Log-likelihood given a `b_cell` candidate and a prebuilt
/// [`BiasCache`]. Walks the chunk index data and the tape in lock-step,
/// adding only bias terms and `log_sigmoid` — no dot products, no
/// embedding lookups, no softplus.
pub fn loglik_with_bias(cache: &BiasCache, b_cell: &DVector<f32>) -> f32 {
    cache
        .chunks
        .par_iter()
        .zip(cache.tapes.par_iter())
        .map(|(chunk, tape)| {
            let mut acc = 0.0f32;
            let mut idx = 0usize;
            for (_gene_id, cb) in chunk {
                let pos_count = cb.left_cells.len();
                let k = cb.n_negatives;
                let n_levels = cb.per_level_neg.len();
                for lvl in 0..n_levels {
                    let neg_for_lvl = &cb.per_level_neg[lvl];
                    for b in 0..pos_count {
                        let u = cb.left_cells[b] as usize;
                        let v = cb.right_cells[b] as usize;
                        let bu = b_cell[u];
                        let bv = b_cell[v];
                        let pos_score = tape[idx] + bu + bv;
                        idx += 1;
                        acc += log_sigmoid(pos_score);
                        let base = b * k;
                        for kk in 0..k {
                            let w = neg_for_lvl[base + kk] as usize;
                            let neg_score = tape[idx] + bu + b_cell[w];
                            idx += 1;
                            acc += log_sigmoid(-neg_score);
                        }
                    }
                }
            }
            acc
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph_embedding_util::loss::CellChainBatch;
    use nalgebra::{DMatrix, DVector};

    fn fixture() -> (
        DMatrix<f32>,
        DMatrix<f32>,
        DMatrix<f32>,
        DVector<f32>,
        Vec<Vec<(usize, CellChainBatch)>>,
    ) {
        let n = 5;
        let d = 3;
        let l = 2;
        let g = 4;
        let e_cell = DMatrix::from_fn(n, d, |i, j| (i + j) as f32 * 0.1);
        let e_gene = DMatrix::from_fn(g, d, |i, j| (i as f32 - j as f32 * 0.5) * 0.3);
        let gamma = DMatrix::from_fn(l, d, |i, j| (i as f32 - j as f32) * 0.2);
        let b_cell = DVector::from_fn(n, |i, _| i as f32 * 0.01);
        let cb = CellChainBatch {
            left_cells: vec![0, 1, 2],
            right_cells: vec![1, 2, 3],
            per_level_neg: vec![vec![4, 0, 1, 2, 3, 4], vec![3, 4, 0, 1, 2, 0]],
            n_negatives: 2,
        };
        let chunks = vec![vec![(2, cb)]];
        (e_cell, e_gene, gamma, b_cell, chunks)
    }

    /// Constant-gene invariant from PLAN_V3 Phase A: identical gene rows
    /// → score independent of gene id.
    #[test]
    fn constant_gene_invariant() {
        let (e_cell, _, gamma, b_cell, mut chunks) = fixture();
        let e_gene = DMatrix::from_element(4, 3, 0.5);
        let cb_clone = CellChainBatch {
            left_cells: chunks[0][0].1.left_cells.clone(),
            right_cells: chunks[0][0].1.right_cells.clone(),
            per_level_neg: chunks[0][0].1.per_level_neg.clone(),
            n_negatives: chunks[0][0].1.n_negatives,
        };
        let ll0 = loglik_total(&e_cell, &e_gene, &gamma, &b_cell, &chunks);
        chunks[0][0] = (0, cb_clone);
        let ll1 = loglik_total(&e_cell, &e_gene, &gamma, &b_cell, &chunks);
        assert!(
            (ll0 - ll1).abs() < 1e-4,
            "constant-gene invariant violated: ll0={ll0} ll1={ll1}"
        );
    }

    /// `loglik_with_bias` against a fresh `BiasCache` must match
    /// `loglik_total` exactly (up to float noise).
    #[test]
    fn bias_cache_matches_loglik_total() {
        let (e_cell, e_gene, gamma, b_cell, chunks) = fixture();
        let direct = loglik_total(&e_cell, &e_gene, &gamma, &b_cell, &chunks);
        let cache = build_bias_cache(&e_cell, &e_gene, &gamma, &chunks);
        let cached = loglik_with_bias(&cache, &b_cell);
        assert!(
            (direct - cached).abs() < 1e-4,
            "bias-cache mismatch: direct={direct} cached={cached}"
        );
    }

    /// `log_sigmoid` (the polynomial-tail fast path) must match the
    /// reference exact implementation to ~1e-4 across a representative
    /// score range. Anything larger would distort ESS slice acceptances.
    #[test]
    fn fast_log_sigmoid_matches_exact() {
        let mut max_err = 0.0f32;
        let mut worst_s = 0.0f32;
        let mut s = -20.0f32;
        while s <= 20.0 {
            let a = log_sigmoid_exact(s);
            let b = log_sigmoid(s);
            let err = (a - b).abs();
            if err > max_err {
                max_err = err;
                worst_s = s;
            }
            s += 0.05;
        }
        assert!(
            max_err < 1e-4,
            "fast log_sigmoid worst-case error {max_err} at s={worst_s}"
        );
    }

    /// Mutating `b_cell` only must produce the same delta whether we
    /// recompute the full log-lik or reuse the bias cache.
    #[test]
    fn bias_cache_tracks_b_cell_changes() {
        let (e_cell, e_gene, gamma, b_cell, chunks) = fixture();
        let cache = build_bias_cache(&e_cell, &e_gene, &gamma, &chunks);

        let perturbed = DVector::from_fn(b_cell.nrows(), |i, _| b_cell[i] + 0.3 * (i as f32));
        let direct = loglik_total(&e_cell, &e_gene, &gamma, &perturbed, &chunks);
        let cached = loglik_with_bias(&cache, &perturbed);
        assert!(
            (direct - cached).abs() < 1e-3,
            "bias-cache delta mismatch: direct={direct} cached={cached}"
        );
    }
}
