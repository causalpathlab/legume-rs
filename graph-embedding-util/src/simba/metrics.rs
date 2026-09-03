//! `si.tl.compare_entities`: per-gene marker metrics from the cell × gene
//! scores `X = E_cell · E_geneᵀ`.
//!
//! ```text
//!   norm    = X − log(mean_c exp X)                 (column-wise)
//!   softmax = exp(X / T) / Σ_c exp(X / T)
//!   max     = mean of the top `n_top_cells` of clip(norm, 0, ∞)
//!   std     = std_c(X)                              (ddof = 1)
//!   gini    = Gini index of the softmax column      (SIMBA's `_gini`)
//!   entropy = −Σ_c p log p of the softmax column
//! ```
//!
//! A gene with a high `max` / `gini` and a low `entropy` scores a small,
//! specific set of cells: SIMBA's marker-gene readout.

use candle_util::candle_core::{Device, Result as CandleResult, Tensor};
use rayon::prelude::*;

/// Genes per dense `[block, N]` score slab.
const GENE_BLOCK: usize = 512;

/// One value per gene, aligned with the rows of `e_gene`.
#[derive(Clone, Debug, Default)]
pub struct EntityMetrics {
    pub max: Vec<f32>,
    pub std: Vec<f32>,
    pub gini: Vec<f32>,
    pub entropy: Vec<f32>,
}

impl EntityMetrics {
    /// Column order of [`Self::to_tensor`].
    pub const COLUMNS: [&'static str; 4] = ["max", "std", "gini", "entropy"];

    /// `[G, 4]` on the CPU in [`Self::COLUMNS`] order.
    pub fn to_tensor(&self) -> CandleResult<Tensor> {
        let g = self.max.len();
        let mut v = Vec::with_capacity(4 * g);
        for i in 0..g {
            v.extend([self.max[i], self.std[i], self.gini[i], self.entropy[i]]);
        }
        Tensor::from_vec(v, (g, 4), &Device::Cpu)
    }
}

/// SIMBA's `compare_entities(adata_ref=cells, adata_query=genes, n_top_cells, T)`.
pub fn compare_entities(
    e_cell: &Tensor,
    e_gene: &Tensor,
    n_top_cells: usize,
    t: f64,
) -> anyhow::Result<EntityMetrics> {
    let (n, h) = e_cell.dims2()?;
    let (g, h2) = e_gene.dims2()?;
    anyhow::ensure!(h == h2, "compare_entities: H mismatch ({h} vs {h2})");
    anyhow::ensure!(n >= 2, "compare_entities: need ≥2 cells (got {n})");
    anyhow::ensure!(t > 0.0, "compare_entities: T must be positive (got {t})");
    let n_top = n_top_cells.clamp(1, n);
    let cell_t = e_cell.t()?.contiguous()?; // [H, N]
    let e_gene = e_gene.contiguous()?;
    let mut out = EntityMetrics::default();
    let mut start = 0usize;
    while start < g {
        let len = GENE_BLOCK.min(g - start);
        // [len, N]: one contiguous row of cell scores per gene.
        let scores = e_gene
            .narrow(0, start, len)?
            .matmul(&cell_t)?
            .to_vec2::<f32>()?;
        let rows: Vec<[f64; 4]> = scores
            .par_iter()
            .map(|x| gene_metrics(x, n_top, t))
            .collect();
        for r in rows {
            out.max.push(r[0] as f32);
            out.std.push(r[1] as f32);
            out.gini.push(r[2] as f32);
            out.entropy.push(r[3] as f32);
        }
        start += len;
    }
    Ok(out)
}

/// `[max, std, gini, entropy]` of one gene's score column, in f64.
fn gene_metrics(x: &[f32], n_top: usize, t: f64) -> [f64; 4] {
    let n = x.len();
    let nf = n as f64;
    let x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // norm = x − log(mean exp x), via a stable log-sum-exp.
    let log_mean_exp = m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln() - nf.ln();
    let mut norm: Vec<f64> = x.iter().map(|v| v - log_mean_exp).collect();
    let k = n - n_top;
    norm.select_nth_unstable_by(k, f64::total_cmp);
    let max = norm[k..].iter().map(|v| v.max(0.0)).sum::<f64>() / n_top as f64;
    let mean = x.iter().sum::<f64>() / nf;
    let std = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
    // softmax over cells at temperature T
    let mt = m / t;
    let z: f64 = x.iter().map(|v| (v / t - mt).exp()).sum();
    let p: Vec<f64> = x.iter().map(|v| (v / t - mt).exp() / z).collect();
    let entropy = -p
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|v| v * v.ln())
        .sum::<f64>();
    // SIMBA `_gini`: shift to ≥ 0, add 1e-7, sort ascending, Σ(2i − n − 1)x_i / (n Σx_i).
    let mut gv = p;
    let mn = gv.iter().copied().fold(f64::INFINITY, f64::min);
    if mn < 0.0 {
        gv.iter_mut().for_each(|v| *v -= mn);
    }
    gv.iter_mut().for_each(|v| *v += 1e-7);
    gv.sort_by(f64::total_cmp);
    let num: f64 = gv
        .iter()
        .enumerate()
        .map(|(i, v)| (2.0 * (i as f64 + 1.0) - nf - 1.0) * v)
        .sum();
    let gini = num / (nf * gv.iter().sum::<f64>());
    [max, std, gini, entropy]
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod metrics_tests;
