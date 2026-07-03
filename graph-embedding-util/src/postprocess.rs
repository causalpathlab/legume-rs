//! Post-hoc SIMBA-style feature co-embedding (`si.tl.embed`-equivalent).
//!
//! After bge finishes, cells and features share an H-dim space but sit on
//! *different* manifolds (cells in tight per-topic balls, genes diffuse), so a
//! joint UMAP shows them separated. SIMBA's fix
//! (`simba/tools/_post_training.py`) is purely post-hoc: re-embed each feature
//! as a softmax-over-cells weighted average of the CELL embeddings, so genes
//! land on the cell manifold. Nothing in training changes — the pseudobulk
//! efficiency and the phase-2 per-cell projection are untouched.
//!
//! ```text
//!   S        = e_cell · e_featᵀ                  [N, D]   (raw dot, no unit-norm)
//!   P[:,f]   = softmax_over_cells( S[:,f] / T )    [N, D]   (column-stochastic)
//!   coembed  = Pᵀ · e_cell                         [D, H]   (gene = wtd avg of cells)
//! ```
//!
//! The softmax temperature `T` is auto-calibrated (no user knob): bisected so
//! the mean effective number of cells per gene (`1 / Σ_c P²`) equals the median
//! cell-cluster size — each gene attends to ~one cluster's worth of cells, which
//! is stable across datasets. The cluster size comes from [`cell_clusters`],
//! which the bge driver runs ONCE and shares with ETM resolution. Cells are the
//! reference and are not moved.

use candle_util::candle_core::{Result, Tensor};
use candle_util::candle_nn::ops::softmax;
use log::info;

/// Feature-axis block size for the streaming pass (peak memory `N×FEAT_BLOCK`).
const FEAT_BLOCK: usize = 512;
/// Feature subsample for temperature calibration (`T` is a single global scalar,
/// so a representative strided subsample suffices and keeps the bisection cheap).
const CALIB_FEATS: usize = 1024;
/// Bisection iterations.
const BISECT_ITERS: usize = 30;
/// Temperature search bounds (eff-cells is monotone increasing in `T`).
const T_LO: f64 = 1e-3;
const T_HI: f64 = 1.0;
/// Leiden parameters for [`cell_clusters`].
const LEIDEN_KNN: usize = 30;
const LEIDEN_RES: f64 = 1.0;
const LEIDEN_SEED: u64 = 42;
/// Clamp the eff-cells target into a sane band so a degenerate clustering (one
/// giant cluster / all singletons) can't drive `T` to an extreme.
const MIN_TARGET_EFF: f64 = 20.0;

/// Leiden-cluster the cell embedding once. Returns `(labels, target_eff)` where
/// `labels[i]` is cell `i`'s cluster (dense `0..k`) and `target_eff` is the
/// median cluster size — the eff-cells target for [`feature_coembedding`]. The
/// bge driver calls this a single time and feeds the labels to ETM resolution
/// (topics) and the target to the co-embedding (temperature), so the embedding
/// is clustered only once. `target_clusters` forwards `--num-topics`.
pub fn cell_clusters(
    e_cell: &Tensor,
    target_clusters: Option<usize>,
) -> anyhow::Result<(Vec<usize>, f64)> {
    use matrix_util::traits::ConvertMatOps;
    let n = e_cell.dim(0)?;
    let dm = nalgebra::DMatrix::<f32>::from_tensor(e_cell)?;
    let labels = matrix_util::clustering::leiden_clustering(
        &dm,
        LEIDEN_KNN,
        LEIDEN_RES,
        target_clusters,
        Some(LEIDEN_SEED),
        true, // cosine — cell embeddings are directional
    )?;
    anyhow::ensure!(labels.len() == n, "cell_clusters: label/row count mismatch");
    let k = labels.iter().copied().max().map_or(0, |m| m + 1);
    anyhow::ensure!(k >= 1, "cell_clusters: no clusters");
    let target = target_eff_from_labels(&labels, k);
    info!("co-embedding: {n} cells → {k} Leiden clusters, eff-cells target {target:.0}");
    Ok((labels, target))
}

/// Co-embedding effective-cells temperature target from PRECOMPUTED cluster
/// labels: the median cluster size, clamped to `[MIN_TARGET_EFF, n/2]`. Lets a
/// caller that already clustered (e.g. annotation's coarse communities) reuse
/// those labels instead of re-running Leiden just to derive the target.
#[must_use]
pub fn target_eff_from_labels(labels: &[usize], n_clusters: usize) -> f64 {
    let n = labels.len();
    let k = n_clusters.max(1);
    let mut sizes = vec![0f32; k];
    for &l in labels {
        if l < k {
            sizes[l] += 1.0;
        }
    }
    let upper = (n as f64 / 2.0).max(MIN_TARGET_EFF);
    f64::from(matrix_util::utils::median(&sizes)).clamp(MIN_TARGET_EFF, upper)
}

/// Re-embed every feature onto the cell manifold (SIMBA-style). The attention is
/// keyed on the identity `e_cell` (query = feature `β_f`, keys = cells):
/// `P_f[c] = softmax_c(⟨e_cell[c], β_f⟩ / T)` places each feature over the cells
/// that express it, and the coordinate is the `P_f`-weighted mean of the cell
/// rows. `target_eff` is the eff-cells target from [`cell_clusters`]. Returns
/// `(coembed [D,H], calibrated_T)`; both inputs are 2-D `[*, H]` on the same
/// device and the output is on that device.
pub fn feature_coembedding(
    e_cell: &Tensor,
    e_feat: &Tensor,
    target_eff: f64,
) -> anyhow::Result<(Tensor, f32)> {
    let (n, h) = e_cell.dims2()?;
    let (d, h2) = e_feat.dims2()?;
    anyhow::ensure!(
        h == h2,
        "feature_coembedding: H mismatch (e_cell {h} vs e_feat {h2})"
    );
    anyhow::ensure!(
        n > 1 && d > 0,
        "feature_coembedding: need ≥2 cells and ≥1 feature (got {n}, {d})"
    );
    // `index_select` / `matmul` require contiguous sources; callers that build
    // the inputs from a column-major nalgebra matrix (`Mat::to_tensor`, e.g.
    // `senna rest`) hand us non-contiguous tensors. A no-op when already
    // contiguous (e.g. bge's candle-native embeddings), so it costs nothing
    // there.
    let e_cell = e_cell.contiguous()?;
    let e_feat = e_feat.contiguous()?;

    // Calibrate T on a feature subsample. The score matrix is invariant in T —
    // only the softmax temperature changes — so compute it once and reuse it
    // across the whole bisection.
    let e_feat_sub = subsample_rows(&e_feat, CALIB_FEATS)?;
    let scores_sub = e_cell.matmul(&e_feat_sub.t()?.contiguous()?)?; // [N, B]
    let t = calibrate_t_for_eff(&scores_sub, target_eff)?;
    info!(
        "feature co-embedding: T={:.4} (eff-cells target {:.0}); mean eff-cells/gene={:.0}",
        t,
        target_eff,
        eff_from_scores(&scores_sub, t)?
    );

    // Full pass, blocked over features.
    let mut blocks: Vec<Tensor> = Vec::new();
    let mut start = 0usize;
    while start < d {
        let len = FEAT_BLOCK.min(d - start);
        let ef = e_feat.narrow(0, start, len)?;
        blocks.push(coembed_block(&e_cell, &ef, t)?); // [len, H]
        start += len;
    }
    let coembed = Tensor::cat(blocks.as_slice(), 0)?; // [D, H]
    Ok((coembed, t as f32))
}

/// One feature-block: `softmax_over_cells(e_cell·e_fᵀ / T)ᵀ · e_cell`.
fn coembed_block(e_cell: &Tensor, e_feat_block: &Tensor, t: f64) -> Result<Tensor> {
    let scores = e_cell.matmul(&e_feat_block.t()?.contiguous()?)?; // [N, B]
    let p = softmax(&scores.affine(1.0 / t, 0.0)?, 0)?; // softmax over cells (dim 0)
    p.t()?.contiguous()?.matmul(e_cell) // [B, N]·[N, H] = [B, H]
}

/// Mean over features of the effective number of cells `1 / Σ_c P[c,f]²`, given
/// a precomputed `[N, B]` score matrix (only the temperature varies in `T`).
fn eff_from_scores(scores: &Tensor, t: f64) -> Result<f64> {
    let p = softmax(&scores.affine(1.0 / t, 0.0)?, 0)?;
    let sum_sq = p.sqr()?.sum(0)?; // [B]
    Ok(f64::from(sum_sq.recip()?.mean_all()?.to_scalar::<f32>()?))
}

/// Bisect `T` so mean eff-cells ≈ `target` (eff ↑ in `T`). Clamps to the search
/// bounds when the target is unreachable within them.
fn calibrate_t_for_eff(scores: &Tensor, target: f64) -> Result<f64> {
    if eff_from_scores(scores, T_LO)? >= target {
        return Ok(T_LO);
    }
    if eff_from_scores(scores, T_HI)? <= target {
        return Ok(T_HI);
    }
    let (mut lo, mut hi) = (T_LO.ln(), T_HI.ln());
    for _ in 0..BISECT_ITERS {
        let mid = 0.5 * (lo + hi);
        if eff_from_scores(scores, mid.exp())? < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((0.5 * (lo + hi)).exp())
}

/// Evenly-strided row subsample (≤ `k` rows) of a `[M, H]` matrix.
fn subsample_rows(x: &Tensor, k: usize) -> Result<Tensor> {
    let m = x.dim(0)?;
    if m <= k {
        return x.contiguous();
    }
    let stride = m as f64 / k as f64;
    let idx: Vec<u32> = (0..k)
        .map(|i| (((i as f64) * stride) as usize).min(m - 1) as u32)
        .collect();
    let idx_t = Tensor::from_vec(idx, k, x.device())?;
    x.index_select(&idx_t, 0)
}
