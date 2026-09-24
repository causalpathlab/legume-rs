//! Per-cluster link tables.
//!
//! For gene `g`, cis peak `p` and cell cluster `k`,
//!
//! ```text
//! c_gpk = w_gp λ_pk / Σ_{q ∈ cis(g)} w_gq λ_qk
//! ```
//!
//! with `w` the trained gate weights, and again with the fixed ABC contact as
//! the baseline. Built gene-by-gene so we never materialise a dense
//! `[pairs × clusters]` matrix (that OOMs on genome-scale cis tables).

use super::cis::CisPairs;
use nalgebra::DMatrix;

/// Call `f(pair, cluster, gate_share, abc_share)` for every `(pair, cluster)`
/// where either share is positive, under the per-pair `gate_w`, the pairs'
/// ABC weights and the peak × cluster accessibility `lambda`. A gene whose
/// candidates are all closed in a cluster gets zeros there.
pub fn for_each_context_share(
    pairs: &CisPairs,
    gate_w: &[f32],
    lambda: &DMatrix<f32>,
    mut f: impl FnMut(usize, usize, f32, f32),
) -> anyhow::Result<()> {
    anyhow::ensure!(
        gate_w.len() == pairs.n_pairs(),
        "{} gate weights for {} pairs",
        gate_w.len(),
        pairs.n_pairs()
    );
    let n_clusters = lambda.ncols();
    let n_peaks = lambda.nrows();
    let share = |w: f32, total: f32| if total > 0.0 { w / total } else { 0.0 };
    for g in 0..pairs.n_genes() {
        let r = pairs.gene(g);
        for j in 0..n_clusters {
            let (mut tot_g, mut tot_a) = (0f32, 0f32);
            for k in r.clone() {
                let p = pairs.peak[k] as usize;
                anyhow::ensure!(p < n_peaks, "peak {p} out of range ({n_peaks})");
                let lam = lambda[(p, j)];
                tot_g += gate_w[k] * lam;
                tot_a += pairs.weight[k] * lam;
            }
            if tot_g <= 0.0 && tot_a <= 0.0 {
                continue;
            }
            for k in r.clone() {
                let lam = lambda[(pairs.peak[k] as usize, j)];
                let a = share(gate_w[k] * lam, tot_g);
                let b = share(pairs.weight[k] * lam, tot_a);
                if a > 0.0 || b > 0.0 {
                    f(k, j, a, b);
                }
            }
        }
    }
    Ok(())
}
