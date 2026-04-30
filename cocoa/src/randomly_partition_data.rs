use crate::common::*;

use data_beans_alg::collapse_data::{
    CollapsingOps, MultilevelCollapsingOps, MultilevelParams, DEFAULT_KNN, DEFAULT_OPT_ITER,
};
use data_beans_alg::random_projection::RandProjOps;
use data_beans_alg::refine_multilevel::RefineParams;
use rustc_hash::FxHashMap as HashMap;

/// Opt-in multilevel refinement settings for pseudobulk assignment.
/// When `Some`, cocoa routes pseudobulk construction through
/// `collapse_columns_multilevel_vec` (same path senna uses) so each
/// pseudobulk is DC-Poisson-refined instead of being a raw hash partition.
#[derive(Clone)]
pub struct RefineSettings {
    pub num_levels: usize,
    pub knn_super_cells: usize,
    pub sort_dim: usize,
    pub num_opt_iter: usize,
    pub refine_params: RefineParams,
}

impl RefineSettings {
    pub fn with_proj_dim(proj_dim: usize) -> Self {
        Self {
            num_levels: 2,
            knn_super_cells: DEFAULT_KNN,
            sort_dim: proj_dim.min(12),
            num_opt_iter: DEFAULT_OPT_ITER,
            refine_params: RefineParams::default(),
        }
    }
}

pub trait RandPartitionOps {
    fn assign_pseudobulk_individuals<T>(
        &mut self,
        proj_dim: usize,
        block_size: usize,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

    /// Multilevel DC-Poisson-refined pseudobulk assignment. Mirrors
    /// senna's `collapse_columns_multilevel_vec` pathway so the refined
    /// cell→pseudobulk mapping is DC-Poisson coherent across tools.
    fn assign_pseudobulk_individuals_refined<T>(
        &mut self,
        proj_dim: usize,
        block_size: usize,
        cell_to_indv: &[T],
        refine: &RefineSettings,
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

    /// Use separate SC data (e.g., scRNA-seq) to compute projections for
    /// confounder adjustment, then apply the resulting pseudobulk partitioning
    /// and HNSW index to self. Both datasets must have the same cells.
    fn assign_pseudobulk_from_adjustment_data<T>(
        &mut self,
        adjustment_data: &SparseIoVec,
        proj_dim: usize,
        block_size: usize,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;
}

/// Apply pre-computed projections: centred for pseudobulk partitioning,
/// raw (uncentred) for HNSW matching.
fn apply_projections<T>(
    target: &mut SparseIoVec,
    centred_proj: &Mat,
    raw_proj: &Mat,
    cell_to_indv: &[T],
) -> anyhow::Result<()>
where
    T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
{
    target.partition_columns_to_groups(centred_proj, None, None)?;
    target.build_hnsw_per_batch(raw_proj, cell_to_indv)?;
    Ok(())
}

/// Project `source` data and apply the resulting pseudobulk partitioning
/// and HNSW index to `target`.
fn project_and_partition<T>(
    target: &mut SparseIoVec,
    source: &SparseIoVec,
    proj_dim: usize,
    block_size: usize,
    cell_to_indv: &[T],
) -> anyhow::Result<()>
where
    T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
{
    let centred = source.project_columns_with_batch_correction(
        proj_dim,
        Some(block_size),
        Some(cell_to_indv),
    )?;
    let raw = source.project_columns(proj_dim, Some(block_size))?;
    apply_projections(target, &centred.proj, &raw.proj, cell_to_indv)
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
        // Compute both projections (immutable borrows) before mutating self
        let centred = self.project_columns_with_batch_correction(
            proj_dim,
            Some(block_size),
            Some(cell_to_indv),
        )?;
        let raw = self.project_columns(proj_dim, Some(block_size))?;
        apply_projections(self, &centred.proj, &raw.proj, cell_to_indv)
    }

    fn assign_pseudobulk_individuals_refined<T>(
        &mut self,
        proj_dim: usize,
        block_size: usize,
        cell_to_indv: &[T],
        refine: &RefineSettings,
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let centred = self.project_columns_with_batch_correction(
            proj_dim,
            Some(block_size),
            Some(cell_to_indv),
        )?;

        let params = MultilevelParams {
            knn_super_cells: refine.knn_super_cells,
            num_levels: refine.num_levels.max(1),
            sort_dim: refine.sort_dim,
            num_opt_iter: refine.num_opt_iter,
            refine: Some(refine.refine_params.clone()),
        };

        // collapse_columns_multilevel_vec:
        // - registers batch membership (cell_to_indv)
        // - builds per-batch HNSW (raw proj_kn is the centred one here; cocoa's
        //   KNN matching uses the HNSW index built here)
        // - hash-partitions at the finest level then DC-Poisson refines
        //   sibling-constrained moves
        // - leaves `self.grouped_columns` set to the finest refined level
        // We drop the returned Vec<CollapsedOut> — cocoa doesn't use the
        // per-level collapsed Gamma stats; it only needs the refined
        // cell→group mapping + HNSW.
        let _levels = self.collapse_columns_multilevel_vec(&centred.proj, cell_to_indv, &params)?;
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
        let mut indv_to_row: HashMap<String, usize> = Default::default();
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

    fn assign_pseudobulk_from_adjustment_data<T>(
        &mut self,
        adjustment_data: &SparseIoVec,
        proj_dim: usize,
        block_size: usize,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        info!(
            "Projecting adjustment data ({} features x {} cells) for confounder adjustment",
            adjustment_data.num_rows(),
            adjustment_data.num_columns()
        );
        project_and_partition(self, adjustment_data, proj_dim, block_size, cell_to_indv)
    }
}
