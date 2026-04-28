//! Per-cell profile store for flat-K cell community model.
//!
//! Mirrors `link_community::model::LinkProfileStore` but indexed by cell:
//! each row is a length-M profile for one cell, where M is either the
//! gene-module count or a random-projection dimension.

use crate::util::common::*;
use matrix_util::utils::generate_minibatch_intervals;

const LOG_EPS: f64 = 1e-12;

/// Row-major `[n_cells × m]` per-cell profile store.
pub struct CellProfileStore {
    pub profiles: Vec<f32>,
    pub size_factors: Vec<f32>,
    pub n_cells: usize,
    pub m: usize,
}

impl CellProfileStore {
    pub fn new(profiles: Vec<f32>, n_cells: usize, m: usize) -> Self {
        debug_assert_eq!(profiles.len(), n_cells * m);
        let size_factors: Vec<f32> = (0..n_cells)
            .map(|u| profiles[u * m..(u + 1) * m].iter().sum())
            .collect();
        Self {
            profiles,
            size_factors,
            n_cells,
            m,
        }
    }

    #[inline]
    pub fn profile(&self, u: usize) -> &[f32] {
        &self.profiles[u * self.m..(u + 1) * self.m]
    }

    /// Column marginal normalised to a distribution (sum to 1).
    pub fn empirical_marginal(&self) -> Vec<f64> {
        let mut col_sum = vec![0.0f64; self.m];
        for u in 0..self.n_cells {
            for (g, &v) in self.profile(u).iter().enumerate() {
                col_sum[g] += v as f64;
            }
        }
        let total: f64 = col_sum.iter().sum::<f64>().max(1.0);
        for v in col_sum.iter_mut() {
            *v /= total;
        }
        col_sum
    }

    /// In-place IDF reweighting: y'_ug = -ln(bg[g] + ε) · y_ug.
    /// Housekeeping genes (large bg) get small weight; specific genes get large weight.
    pub fn weight_by_idf(&mut self, bg: &[f64]) {
        debug_assert_eq!(bg.len(), self.m);
        let w: Vec<f32> = bg.iter().map(|&p| (-((p + LOG_EPS).ln())) as f32).collect();
        for u in 0..self.n_cells {
            let row = &mut self.profiles[u * self.m..(u + 1) * self.m];
            let mut s = 0.0f32;
            for (y, &wg) in row.iter_mut().zip(w.iter()) {
                *y *= wg;
                s += *y;
            }
            self.size_factors[u] = s;
        }
    }
}

/// Build per-cell projection profiles y_u = W^T x_u, blocked over cells.
///
/// * `data` — sparse expression `[n_genes × n_cells]`
/// * `basis` — projection `[n_genes × proj_dim]`
pub fn build_cell_projection_profiles(
    data: &SparseIoVec,
    basis: &Mat,
    block_size: Option<usize>,
) -> anyhow::Result<CellProfileStore> {
    let n_cells = data.num_columns();
    let n_genes = data.num_rows();
    let m = basis.ncols();
    debug_assert_eq!(basis.nrows(), n_genes);

    let basis_t = basis.transpose();
    let jobs = generate_minibatch_intervals(n_cells, n_genes, block_size);
    let pb = new_progress_bar(
        jobs.len() as u64,
        "Cell profiles {bar:40} {pos}/{len} blocks ({eta})",
    );

    let partials: Vec<(usize, Vec<f32>)> = jobs
        .par_iter()
        .progress_with(pb.clone())
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>)> {
            let x = data.read_columns_csc(lb..ub)?;
            let chunk_size = ub - lb;
            let mut chunk = vec![0.0f32; chunk_size * m];
            let mut temp_g = DVec::zeros(n_genes);
            for col in 0..chunk_size {
                temp_g.fill(0.0);
                let s = x.col(col);
                for (&row, &val) in s.row_indices().iter().zip(s.values().iter()) {
                    temp_g[row] += val;
                }
                let proj = &basis_t * &temp_g;
                let base = col * m;
                for (d, &v) in proj.iter().enumerate() {
                    chunk[base + d] = v.max(0.0);
                }
            }
            Ok((lb, chunk))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    pb.finish_and_clear();

    let mut profiles = vec![0.0f32; n_cells * m];
    for (lb, chunk) in partials {
        let dst = lb * m;
        profiles[dst..dst + chunk.len()].copy_from_slice(&chunk);
    }
    Ok(CellProfileStore::new(profiles, n_cells, m))
}

/// Aggregate fine-cell profiles into super-cell profiles by summation.
/// `super_labels[u]` ∈ [0, n_super) for each fine cell u.
pub fn coarsen_cell_profiles(
    fine: &CellProfileStore,
    super_labels: &[usize],
    n_super: usize,
) -> CellProfileStore {
    debug_assert_eq!(super_labels.len(), fine.n_cells);
    let m = fine.m;
    let mut super_profiles = vec![0.0f32; n_super * m];
    for u in 0..fine.n_cells {
        let s = super_labels[u];
        debug_assert!(s < n_super);
        let src = fine.profile(u);
        let dst_base = s * m;
        for g in 0..m {
            super_profiles[dst_base + g] += src[g];
        }
    }
    CellProfileStore::new(super_profiles, n_super, m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idf_reweights_size_factors() {
        let profiles = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut store = CellProfileStore::new(profiles, 2, 3);
        let bg = vec![0.1, 0.2, 0.7];
        store.weight_by_idf(&bg);
        for u in 0..2 {
            let row = store.profile(u);
            let expected: f32 = row.iter().sum();
            assert!((store.size_factors[u] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_empirical_marginal_sums_to_one() {
        let profiles = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let store = CellProfileStore::new(profiles, 2, 3);
        let dist = store.empirical_marginal();
        let s: f64 = dist.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_coarsen_sums_cells() {
        // 4 cells × 2 dims, super labels [0, 0, 1, 1]
        let profiles = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let store = CellProfileStore::new(profiles, 4, 2);
        let super_labels = vec![0, 0, 1, 1];
        let coarse = coarsen_cell_profiles(&store, &super_labels, 2);
        assert_eq!(coarse.profile(0), &[4.0, 6.0]);
        assert_eq!(coarse.profile(1), &[12.0, 14.0]);
    }
}
