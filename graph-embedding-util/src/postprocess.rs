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
//!   S̃[:,f]   = S[:,f] / sd_c(S[:,f])              [N, D]   (per-FEATURE unit scale)
//!   P[:,f]   = softmax_over_cells( S̃[:,f] / T )    [N, D]   (column-stochastic)
//!   coembed  = Pᵀ · e_cell                         [D, H]   (gene = wtd avg of cells)
//! ```
//!
//! The per-feature rescale is what makes a single `T` well-defined. `‖e_f‖`
//! spans ~38× across genes, so on RAW scores one `T` saturates the long genes
//! onto one cell and leaves the short ones spread over all of them — see
//! [`standardize_columns`].
//!
//! The softmax temperature `T` is auto-calibrated (no user knob): bisected so
//! the mean effective number of cells per gene (`1 / Σ_c P²`) equals the median
//! cell-cluster size, clamped to `[MIN_TARGET_EFF, MAX_TARGET_EFF]`. The clamp is
//! load-bearing, not defensive: an unclamped "one cluster's worth of cells" is
//! ~600 here and scores WORSE than no calibration at all, because averaging that
//! many cell rows pulls the barycentre to the global centroid. The cluster size
//! comes from [`cell_clusters`], which the bge driver runs ONCE and shares with
//! ETM resolution. Cells are the reference and are not moved.
//!
//! Biases are absent by design — `b_f` cancels in the softmax, `b_c` is a
//! shared per-cell depth offset that would drag every gene toward deep cells.
//! See [`coembed_block`].

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
/// Upper clamp on the eff-cells target.
///
/// A gene attending to a WHOLE cluster is past the useful point: averaging that
/// many cell rows shrinks the barycentre toward the global centroid and the
/// coordinate stops tracking the gene's own top cells.
///
/// Numbers below are in the SAME units [`eff_from_scores`] reports — the MEAN
/// over features of inverse-Simpson `1/Σ_c p²` — because that is what the
/// bisection actually drives. (An earlier revision of this comment quoted median
/// `exp(entropy)` instead, which is ~8× smaller here and made the constant look
/// mis-set.) Measured on 12k BMMC (`H=16`), recall of the model's own top-100
/// cells against the gene-specific score `⟨e_f, e_c⟩`:
///
/// | mean eff | 16 | 49 | 178 | 488 | 1045 |
/// |---|---|---|---|---|---|
/// | recall@100 | 0.468 | 0.464 | 0.436 | 0.385 | 0.315 |
///
/// The old behaviour (median cluster size, ~600 cells) sat at ~0.37, below the
/// 0.366 the uncalibrated path already achieved. 100 lands at ~0.45: off the peak
/// on purpose, since tuning to the ~20 that maximizes recall would be fitting a
/// proxy that structurally rewards near-hard assignment (a gene parked on its top
/// cell inherits that cell's neighbours).
///
/// REPLICATED over 25 fits spanning 3 datasets (6270 / 12602 / 42826 cells) and
/// `H` ∈ {8, 16}: the per-fit optimum is median 22, range 7–61, and 100 captures
/// **97.2%** of each fit's achievable optimum (worst 94.1%) against **74.0%**
/// (worst 50.5%) for the old ~600 target. 100 beat 600 on 25/25.
///
/// Why a CONSTANT and not a fraction of `n`: the optimum does not track cell
/// count — median optimum was 29 at N=6270, 21 at N=12602, 42 at N=42826. A gene
/// attends to a fixed-size neighbourhood, not a fixed share of the dataset, so
/// scaling this with `n` would be wrong in the large-`n` limit exactly where the
/// old cluster-size target already failed.
const MAX_TARGET_EFF: f64 = 100.0;
/// Floor on a feature's score SD, so a zero-embedding feature (constant score
/// column) divides by a positive number instead of producing `NaN`.
const SD_FLOOR: f64 = 1e-6;

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
    let upper = (n as f64 / 2.0).clamp(MIN_TARGET_EFF, MAX_TARGET_EFF);
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
    // Standardized here so `T` is calibrated in the SAME units `coembed_block`
    // applies it in. Calibrating on raw scores and spending the result on
    // standardized ones is shape-valid and silently wrong, which is why the
    // rescale lives in one named helper both paths call.
    let scores_sub = standardize_columns(&e_cell.matmul(&e_feat_sub.t()?.contiguous()?)?)?; // [N, B]
    let t = calibrate_t_for_eff(&scores_sub, target_eff)?;
    info!(
        "feature co-embedding: T={:.4} (eff-cells target {:.0}); mean eff-cells/gene={:.0}",
        t,
        target_eff,
        eff_from_scores(&scores_sub, t)?
    );

    // Full pass, blocked over features. This is the heavy post-training step
    // (a `genes × cells` softmax-weighted average); a progress bar over the
    // feature blocks keeps it from looking hung (canonical workspace style).
    let prog = crate::progress::new_progress_bar(d.div_ceil(FEAT_BLOCK) as u64)
        .with_message("co-embedding features");
    let mut blocks: Vec<Tensor> = Vec::new();
    let mut start = 0usize;
    while start < d {
        let len = FEAT_BLOCK.min(d - start);
        let ef = e_feat.narrow(0, start, len)?;
        blocks.push(coembed_block(&e_cell, &ef, t)?); // [len, H]
        start += len;
        prog.inc(1);
    }
    prog.finish_and_clear();
    let coembed = Tensor::cat(blocks.as_slice(), 0)?; // [D, H]
    Ok((coembed, t as f32))
}

/// SIMBA's own `si.tl.embed`: the same softmax-over-cells average as
/// [`feature_coembedding`], but on the RAW dot scores at a FIXED temperature
/// (`T = 0.5` in SIMBA) — no per-feature rescale, no calibration, no biases.
/// `senna simba` writes this as its `feature_embedding`, so the file holds
/// SIMBA's co-embedding rather than bge's calibrated one.
pub fn feature_coembedding_fixed_t(
    e_cell: &Tensor,
    e_feat: &Tensor,
    t: f64,
) -> anyhow::Result<Tensor> {
    let (n, h) = e_cell.dims2()?;
    let (d, h2) = e_feat.dims2()?;
    anyhow::ensure!(
        h == h2,
        "feature_coembedding_fixed_t: H mismatch (e_cell {h} vs e_feat {h2})"
    );
    anyhow::ensure!(
        n > 0 && d > 0,
        "feature_coembedding_fixed_t: need ≥1 cell and ≥1 feature (got {n}, {d})"
    );
    anyhow::ensure!(
        t > 0.0,
        "feature_coembedding_fixed_t: T must be positive (got {t})"
    );
    let e_cell = e_cell.contiguous()?;
    let e_feat = e_feat.contiguous()?;
    let prog = crate::progress::new_progress_bar(d.div_ceil(FEAT_BLOCK) as u64)
        .with_message("co-embedding features (fixed T)");
    let mut blocks: Vec<Tensor> = Vec::new();
    let mut start = 0usize;
    while start < d {
        let len = FEAT_BLOCK.min(d - start);
        let ef = e_feat.narrow(0, start, len)?;
        let scores = e_cell.matmul(&ef.t()?.contiguous()?)?; // [N, len], raw
        blocks.push(coembed_from_scores(&scores, &e_cell, t)?); // [len, H]
        start += len;
        prog.inc(1);
    }
    prog.finish_and_clear();
    Ok(Tensor::cat(blocks.as_slice(), 0)?)
}

/// Rescale each FEATURE's score column to unit SD over cells.
///
/// Without this, one global `T` meets a score scale that varies with `‖e_f‖` —
/// measured spread 38× between the median and the largest gene on 12k BMMC — so
/// the same `T` saturates the long genes onto a single cell while leaving the
/// short ones nearly uniform over all cells. The result is bimodal rather than
/// merely variable: at the calibrated `T`, 34% of genes sat on <10 cells and 22%
/// on >5000, and the mean the bisection targets (1812) described almost no gene
/// (median 47). Dividing by the column SD makes `T` mean the same thing for every
/// feature, which is also what makes [`calibrate_t_for_eff`]'s mean-based target
/// representative instead of an average over two degenerate populations.
///
/// No centering: softmax is shift-invariant along the axis it normalizes, so only
/// the scale matters, and skipping it avoids materializing a second `[N, B]`.
///
/// This narrows the spread but does NOT make it uniform: at the operating point
/// the eff-cells distribution is still right-skewed (mean 16 vs median 2), down
/// from ~39× before. So [`calibrate_t_for_eff`]'s mean-based target remains
/// coarser than the median gene's experience — a known limitation, not a claim
/// this helper fixes.
fn standardize_columns(scores: &Tensor) -> Result<Tensor> {
    let mean = scores.mean_keepdim(0)?; // [1, B]
    let var = scores.sqr()?.mean_keepdim(0)?.sub(&mean.sqr()?)?;
    // `E[x²]−E[x]²` can land slightly negative by cancellation; the floor doubles
    // as the guard for that and for a genuinely constant column.
    let sd = var.maximum(SD_FLOOR * SD_FLOOR)?.sqrt()?;
    scores.broadcast_div(&sd)
}

/// One feature-block: `softmax_over_cells(standardize(e_cell·e_fᵀ) / T)ᵀ · e_cell`.
///
/// Biases are deliberately absent. `b_f` is constant down a column and cancels in
/// the softmax; `b_c` does NOT cancel, but it is a per-cell depth offset shared by
/// every gene, so including it drags all genes toward high-depth cells — measured,
/// it halves fidelity (recall@100 0.387 → 0.162) and makes 43.6% of any gene's
/// top-100 the same cells. The co-embedding wants the gene-SPECIFIC part of the
/// score only.
fn coembed_block(e_cell: &Tensor, e_feat_block: &Tensor, t: f64) -> Result<Tensor> {
    let scores = standardize_columns(&e_cell.matmul(&e_feat_block.t()?.contiguous()?)?)?; // [N, B]
    coembed_from_scores(&scores, e_cell, t)
}

/// `softmax_over_cells(scores / T)ᵀ · e_cell` for one `[N, B]` score block:
/// the step both the calibrated path ([`coembed_block`], standardized scores)
/// and SIMBA's fixed-T path ([`feature_coembedding_fixed_t`], raw scores) share.
fn coembed_from_scores(scores: &Tensor, e_cell: &Tensor, t: f64) -> Result<Tensor> {
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

#[cfg(test)]
#[path = "postprocess_tests.rs"]
mod postprocess_tests;
