use crate::common::*;
use crate::stat::*;
use crate::DiffArgs;

use asap_alg::collapse_data::CollapsingOps;
use asap_alg::random_projection::RandProjOps;

pub trait RandPartitionOps {
    fn assign_pseudobulk_individuals<T>(
        &mut self,
        proj_dim: usize,
        block_size: usize,
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
        // treat an individual as a "batch" so that we can estimate projections
        // comparable across batches
        let proj_out = self.project_columns_with_batch_correction(
            proj_dim,
            Some(block_size),
            Some(&cell_to_indv),
        )?;

        let proj_kn = proj_out.proj;
        self.partition_columns_to_groups(&proj_kn, None, None)?;

        // treat each individual as a batch
        self.build_hnsw_per_batch(&proj_kn, cell_to_indv)?;

        Ok(())
    }
}
