use crate::common::*;

use data_beans_alg::collapse_data::{
    CollapsedOut, CollapsingOps, MultilevelCollapsingOps, MultilevelParams, DEFAULT_KNN,
    DEFAULT_OPT_ITER,
};
use data_beans_alg::random_projection::RandProjOps;
use data_beans_alg::refine_multilevel::RefineParams;
use rustc_hash::FxHashMap as HashMap;

/// Multilevel refinement settings for pseudobulk assignment.
/// Routes through `collapse_columns_multilevel_vec` (senna path).
#[derive(Clone)]
pub struct RefineSettings {
    pub num_levels: usize,
    pub knn_pb_samples: usize,
    pub sort_dim: usize,
    pub num_opt_iter: usize,
    pub refine_params: RefineParams,
}

impl RefineSettings {
    pub fn with_proj_dim(proj_dim: usize) -> Self {
        Self {
            num_levels: 2,
            knn_pb_samples: DEFAULT_KNN,
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

    /// Multilevel DC-Poisson-refined pseudobulk assignment. No exposure
    /// strata: pseudobulks may mix exposures, so τ sees the full
    /// between-individual spread.
    ///
    /// Returns the finest-level [`CollapsedOut`] (includes δ when estimated).
    fn assign_pseudobulk_individuals_refined<T>(
        &mut self,
        proj_dim: usize,
        block_size: usize,
        cell_to_indv: &[T],
        refine: &RefineSettings,
    ) -> anyhow::Result<CollapsedOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

    fn assign_pseudobulk_with_known_confounders<T>(
        &mut self,
        confounder_v: &Mat,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

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
    ) -> anyhow::Result<CollapsedOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let centred = self.project_columns_with_batch_correction(
            proj_dim,
            Some(block_size),
            Some(cell_to_indv),
        )?;

        let params = MultilevelParams {
            knn_pb_samples: refine.knn_pb_samples,
            num_levels: refine.num_levels.max(1),
            sort_dim: refine.sort_dim,
            num_opt_iter: refine.num_opt_iter,
            refine: refine.refine_params.clone(),
            output_calibration: matrix_param::traits::CalibrateTarget::All,
            anchor_batches: None,
            bulk_batches: None,
            observe_panels: true,
            keep_finest_stats: false,
            pb_tree: None,
            strata: None,
        };

        // Levels are finest-first; cocoa needs the finest δ + membership.
        self.collapse_columns_multilevel_vec(&centred.proj, cell_to_indv, &params)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("multilevel collapse returned no levels"))
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

        let mut indv_to_row: HashMap<String, usize> = Default::default();
        for i in 0..confounder_v.nrows() {
            indv_to_row.insert(i.to_string(), i);
        }

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
