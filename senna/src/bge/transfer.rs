//! Gene-axis alignment for `senna predict` on a bge model: the pure pieces
//! between the projector and the writer, so each is testable without a backend.
//!
//! The flow in `BgeEmbedding::score`:
//! 1. pass 1 — project every cell on the MATCHED genes (as before);
//! 2. cluster the pass-1 latents into pseudobulks and accumulate the new data's
//!    per-gene profiles over them ([`profiles_by_cluster`]);
//! 3. align the gene axes ([`graph_embedding_util::transfer::align_gene_axis`]):
//!    unseen genes get a membership-initialized row, then a moment-matched bias;
//! 4. optional pass 2 — re-project with the initialized genes as observations
//!    ([`union_remap`] maps the new rows onto the union axis);
//! 5. score the initialized genes on their OWN column ([`score_initialized`]):
//!    a prior's score is never mixed into the comparable per-gene score.

use crate::embed_common::Mat;
use nalgebra::DMatrix;
use std::collections::HashSet;

/// Rows of the NEW data that matched no training gene by name, excluding the
/// rows deliberately withheld by `--ablate-features` — those are model genes
/// being tested, not unseen ones. `new_to_train` must be the remap BEFORE the
/// hide pass zeroed the hidden rows.
pub(crate) fn unseen_rows(
    new_to_train: &[Option<usize>],
    hidden_rows: &HashSet<usize>,
) -> Vec<usize> {
    new_to_train
        .iter()
        .enumerate()
        .filter(|(n, t)| t.is_none() && !hidden_rows.contains(n))
        .map(|(n, _)| n)
        .collect()
}

/// Per-gene count profiles over pseudobulks defined by a cell clustering:
/// `profiles[g, s] = Σ_{c: labels[c] = s} x_cg`, `[n_genes × n_clusters]`.
/// `cells` yields `(cell, feature rows, counts)` on the NEW data's row axis.
pub(crate) fn profiles_by_cluster<'a>(
    n_genes: usize,
    n_clusters: usize,
    labels: &[usize],
    cells: impl Iterator<Item = (usize, &'a [u32], &'a [f32])>,
) -> DMatrix<f32> {
    let mut p = DMatrix::<f32>::zeros(n_genes, n_clusters);
    for (c, feats, counts) in cells {
        let s = labels[c];
        for (&f, &x) in feats.iter().zip(counts) {
            p[(f as usize, s)] += x;
        }
    }
    p
}

/// New-data row → union-axis index: a matched row keeps its training position,
/// an unseen row `unseen[i]` becomes `n_train + i`, and any other row (hidden,
/// or dropped) maps to `None`.
pub(crate) fn union_remap(
    new_to_train: &[Option<usize>],
    unseen: &[usize],
    n_train: usize,
) -> Vec<Option<usize>> {
    let mut out: Vec<Option<usize>> = new_to_train.to_vec();
    for (i, &row) in unseen.iter().enumerate() {
        out[row] = Some(n_train + i);
    }
    out
}

/// Per-cell score of the INITIALIZED genes, in the same multinomial-per-count
/// form as the comparable score, kept in its own column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InitScore {
    /// Observed counts on the initialized genes.
    pub count: f32,
    /// `Σ_g x_g log p_g / Σ_g x_g` with `p` the model's composition over the
    /// initialized genes; `NaN` when the cell has no counts on them.
    pub llik_per_count: f32,
    /// The same under `null_comp`.
    pub null_llik_per_count: f32,
}

/// `rows` / `bias` are the initialized genes' `[U × H]` / `[U]`; `theta` `[N × H]`
/// and `b_cell` `[N]` are the latents the rates are evaluated at; `obs[c]` lists
/// `(local initialized index, count)` for cell `c`; `null_comp` is a composition
/// over the `U` genes.
pub(crate) fn score_initialized(
    rows: &DMatrix<f32>,
    bias: &[f32],
    theta: &Mat,
    b_cell: &[f32],
    obs: &[Vec<(u32, f32)>],
    null_comp: &[f32],
) -> Vec<InitScore> {
    let (u, h) = (rows.nrows(), rows.ncols());
    let floor = f64::from(matrix_util::agreement::PROB_FLOOR);
    let log_null: Vec<f64> = null_comp
        .iter()
        .map(|&q| f64::from(q).max(floor).ln())
        .collect();
    (0..theta.nrows())
        .map(|c| {
            let count: f32 = obs[c].iter().map(|&(_, x)| x).sum();
            if count <= 0.0 {
                return InitScore {
                    count: 0.0,
                    llik_per_count: f32::NAN,
                    null_llik_per_count: f32::NAN,
                };
            }
            // Composition of the model's rates over the initialized genes; the
            // cell bias is a common factor and cancels.
            let logits: Vec<f64> = (0..u)
                .map(|g| {
                    let s: f32 = (0..h).map(|j| rows[(g, j)] * theta[(c, j)]).sum::<f32>()
                        + bias[g]
                        + b_cell[c];
                    f64::from(s)
                })
                .collect();
            let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln() + m;
            let (mut ll, mut nl) = (0f64, 0f64);
            for &(g, x) in &obs[c] {
                let x = f64::from(x);
                ll += x * (logits[g as usize] - lse).max(floor.ln());
                nl += x * log_null[g as usize];
            }
            InitScore {
                count,
                llik_per_count: (ll / f64::from(count)) as f32,
                null_llik_per_count: (nl / f64::from(count)) as f32,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
