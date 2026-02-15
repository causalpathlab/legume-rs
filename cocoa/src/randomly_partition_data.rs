use crate::common::*;

use data_beans_alg::collapse_data::CollapsingOps;
use data_beans_alg::random_projection::RandProjOps;
use std::collections::HashMap;

pub trait RandPartitionOps {
    fn assign_pseudobulk_individuals<T>(
        &mut self,
        proj_dim: usize,
        block_size: usize,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

    /// Use a known individual-level confounder matrix V (n_indv x n_covar)
    /// instead of random projection. Each cell inherits its individual's row
    /// from V, producing a K x N feature matrix for pseudobulk partitioning.
    fn assign_pseudobulk_with_known_confounders<T>(
        &mut self,
        confounder_v: &Mat,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;
}

impl RandPartitionOps for SparseIoVec {
    fn assign_pseudobulk_individuals<T>(
        &mut self,
        proj_dim: usize,
        block_size: usize,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        // Centred projection for pseudobulk partitioning
        // (needs cells mixed across individuals within each group)
        let centred = self.project_columns_with_batch_correction(
            proj_dim,
            Some(block_size),
            Some(cell_to_indv),
        )?;
        self.partition_columns_to_groups(&centred.proj, None, None)?;

        // Uncentred projection for HNSW matching
        // (preserves individual-level signal V for confounder adjustment)
        let raw = self.project_columns(proj_dim, Some(block_size))?;
        self.build_hnsw_per_batch(&raw.proj, cell_to_indv)?;

        Ok(())
    }

    fn assign_pseudobulk_with_known_confounders<T>(
        &mut self,
        confounder_v: &Mat,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let n_cells = cell_to_indv.len();
        let n_covar = confounder_v.ncols();

        info!(
            "Using known confounders: {} individuals x {} covariates -> {} cells",
            confounder_v.nrows(),
            n_covar,
            n_cells
        );

        // Build individual name -> row index mapping
        let mut indv_to_row: HashMap<String, usize> = HashMap::new();
        for i in 0..confounder_v.nrows() {
            indv_to_row.insert(i.to_string(), i);
        }

        // Expand V from individual-level to cell-level: K x N (features x cells)
        let mut proj_kn = Mat::zeros(n_covar, n_cells);
        for (j, indv) in cell_to_indv.iter().enumerate() {
            let indv_str = indv.to_string();
            if let Some(&row) = indv_to_row.get(&indv_str) {
                for k in 0..n_covar {
                    proj_kn[(k, j)] = confounder_v[(row, k)];
                }
            }
        }

        self.partition_columns_to_groups(&proj_kn, None, None)?;
        self.build_hnsw_per_batch(&proj_kn, cell_to_indv)?;

        Ok(())
    }
}
