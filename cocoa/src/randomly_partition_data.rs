use crate::common::*;

use data_beans::alg::collapse_data::CollapsingOps;
use data_beans::alg::random_projection::RandProjOps;
use data_beans::alg::refine_mixed::{MixedRefineOps, MixedRefineParams};

/// Pseudobulks by cell state across individuals: project the cells, centre
/// the projection per individual within cell state (so cells group by
/// state, not by donor, exposure or composition),
/// and bin all cells together by the sign pattern of `bits` projection
/// coordinates. Cells of different individuals in the same state share a
/// pseudobulk, which is what identifies the stage-1 baseline.
pub trait RandPartitionOps {
    /// Partition this data by its own projection.
    fn assign_pseudobulk_across_individuals<T>(
        &mut self,
        spec: &PartitionSpec,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

    /// Partition this data by the projection of separate data on the same
    /// cells.
    fn assign_pseudobulk_from_adjustment_data<T>(
        &mut self,
        adjustment_data: &SparseIoVec,
        spec: &PartitionSpec,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;
}

/// Likelihood sweeps moving cells between pseudobulks.
const REFINE_SWEEPS: usize = 10;

/// How cells are binned into pseudobulks.
pub struct PartitionSpec {
    pub proj_dim: usize,
    /// bits of the finest level
    pub bits: usize,
    /// levels a poorly mixed bin may merge up
    pub merge_levels: usize,
    /// individuals a pseudobulk needs
    pub min_individuals: usize,
    pub block_size: usize,
}

/// Bits for the partition: about two cells per individual per pseudobulk on
/// average, so a pseudobulk can hold several individuals, capped by the
/// projection dimension.
pub fn partition_bits(n_cells: usize, n_indv: usize, proj_dim: usize) -> usize {
    let per_pb = (2 * n_indv.max(1)) as f64;
    let bits = (n_cells as f64 / per_pb).log2().floor().max(1.0) as usize;
    bits.min(proj_dim.max(1))
}

/// Bin cells by a projection already centred per individual within cell
/// state, merging poorly mixed bins up the code tree, move cells between
/// bins under a Poisson likelihood with individual offsets so each holds
/// one cell state, and register each cell's individual. The refinement
/// scores the target's own counts, also when the bins come from adjustment
/// data, since those are the counts stage 1 fits.
fn partition_by_projection<T>(
    target: &mut SparseIoVec,
    centred_proj: &Mat,
    spec: &PartitionSpec,
    cell_to_indv: &[T],
) -> anyhow::Result<()>
where
    T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
{
    let refine = MixedRefineParams {
        max_sweeps: REFINE_SWEEPS,
        block_size: Some(spec.block_size),
        ..MixedRefineParams::default()
    };
    target.partition_columns_to_refined_mixed_groups(
        centred_proj,
        Some(spec.bits),
        cell_to_indv,
        spec.min_individuals,
        spec.merge_levels,
        &refine,
    )?;
    // registers each cell's individual (batch); data-beans builds the batch
    // lookup index in the same call, and has no registration-only method
    target.build_hnsw_per_batch(centred_proj, cell_to_indv)?;
    Ok(())
}

impl RandPartitionOps for SparseIoVec {
    fn assign_pseudobulk_across_individuals<T>(
        &mut self,
        spec: &PartitionSpec,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let centred = self.project_columns_with_batch_correction(
            spec.proj_dim,
            Some(spec.block_size),
            Some(cell_to_indv),
        )?;
        partition_by_projection(self, &centred.proj, spec, cell_to_indv)
    }

    fn assign_pseudobulk_from_adjustment_data<T>(
        &mut self,
        adjustment_data: &SparseIoVec,
        spec: &PartitionSpec,
        cell_to_indv: &[T],
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        info!(
            "Projecting adjustment data ({} features x {} cells) for the pseudobulks",
            adjustment_data.num_rows(),
            adjustment_data.num_columns()
        );
        let centred = adjustment_data.project_columns_with_batch_correction(
            spec.proj_dim,
            Some(spec.block_size),
            Some(cell_to_indv),
        )?;
        partition_by_projection(self, &centred.proj, spec, cell_to_indv)
    }
}
