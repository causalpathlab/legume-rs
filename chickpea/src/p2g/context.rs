//! Per-cluster link tables.
//!
//! For gene `g`, cis peak `p` and cell cluster `k`,
//!
//! ```text
//! c_gpk = w_gp λ_pk / Σ_{q ∈ cis(g)} w_gq λ_qk
//! ```
//!
//! the share of `g`'s regulatory input from `p` in `k`. With `w` the learned
//! attention shares this is the context-specific link; with `w` the fixed ABC
//! contact it is ABC per cluster, the baseline. A gene whose candidates are all
//! closed in a cluster gets zeros there.

use super::cis::CisPairs;
use nalgebra::DMatrix;

/// `[pairs × clusters]` shares; `weight` has one entry per pair and `lambda`
/// is `[peaks × clusters]`.
pub fn context_shares(
    pairs: &CisPairs,
    weight: &[f32],
    lambda: &DMatrix<f32>,
) -> anyhow::Result<DMatrix<f32>> {
    anyhow::ensure!(
        weight.len() == pairs.n_pairs(),
        "{} weights for {} pairs",
        weight.len(),
        pairs.n_pairs()
    );
    let n_clusters = lambda.ncols();
    let mut out = DMatrix::zeros(pairs.n_pairs(), n_clusters);
    for g in 0..pairs.n_genes() {
        let r = pairs.gene(g);
        for j in 0..n_clusters {
            let total: f32 = r
                .clone()
                .map(|k| weight[k] * lambda[(pairs.peak[k] as usize, j)])
                .sum();
            if total > 0.0 {
                for k in r.clone() {
                    out[(k, j)] = weight[k] * lambda[(pairs.peak[k] as usize, j)] / total;
                }
            }
        }
    }
    Ok(out)
}
