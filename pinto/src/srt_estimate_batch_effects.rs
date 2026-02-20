//! Batch effect estimation via hierarchical pseudobulk collapsing.
//!
//! ## How multilevel collapsing estimates batch effects
//!
//! The batch correction uses a METIS-inspired hierarchical coarsening
//! scheme that avoids expensive cell-level cross-batch comparisons:
//!
//! 1. **Partition cells into pseudobulk groups** via binary sort on
//!    random projection coordinates (at progressively finer `sort_dim`
//!    per level).
//!
//! 2. **Build super-cells**: each (batch, group) intersection becomes
//!    a super-cell with a centroid (mean projection vector) and
//!    aggregated gene sums. This is a small set — typically
//!    O(n_batches * n_groups) entries.
//!
//! 3. **Cross-batch KNN matching** (`--batch-knn`): for each
//!    super-cell, find its `batch_knn` nearest neighbors among
//!    super-cells from *other* batches using HNSW on centroids.
//!    Because we search over coarsened super-cells (not individual
//!    cells), this is fast even with many cells.
//!
//! 4. **Counterfactual imputation**: the matched neighbors provide a
//!    "what would this group look like in another batch?" estimate.
//!    - `imputed_sum[g,s]` = weighted average of matched neighbors'
//!      per-cell gene expression, scaled by cell count
//!    - `residual_sum[g,s]` = observed / imputed ratio
//!
//! 5. **EM-style optimization**: iteratively refine four Gamma
//!    parameters (μ, μ_resid, γ, δ) to decompose observed expression
//!    into true signal (μ), batch effect (δ), and cross-batch
//!    correction terms (γ, μ_resid).
//!
//! 6. **Multilevel refinement**: repeat steps 1-5 at each level with
//!    increasing `sort_dim` (coarse → fine). Coarser levels capture
//!    large-scale batch effects; finer levels refine them.
//!
//! The key insight: because KNN matching operates on super-cell
//! centroids (not individual cells), `batch_knn` remains small
//! (default 10) regardless of dataset size. The hierarchical levels
//! ensure both coarse global corrections and fine-grained local
//! adjustments.

use crate::srt_common::*;
use data_beans_alg::collapse_data::*;
use data_beans_alg::random_projection::*;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::io::ParamIo;
use matrix_param::traits::Inference;

pub struct EstimateBatchArgs {
    pub proj_dim: usize,
    pub sort_dim: usize,
    pub block_size: usize,
    /// KNN for cross-batch super-cell matching during hierarchical
    /// collapsing. Searches are over coarsened super-cell centroids,
    /// not individual cells, so this stays small.
    pub batch_knn: usize,
    pub num_levels: usize,
}

/// Estimate per-gene batch effect multipliers (δ) via hierarchical
/// pseudobulk collapsing with cross-batch KNN matching.
///
/// Returns `None` if fewer than 2 batches are present.
pub fn estimate_batch(
    data_vec: &mut SparseIoVec,
    batch_membership: &[Box<str>],
    args: EstimateBatchArgs,
) -> anyhow::Result<Option<GammaMatrix>> {
    let batch_hash: HashSet<Box<str>> = batch_membership.iter().cloned().collect();
    let nbatch = batch_hash.len();
    if nbatch < 2 {
        return Ok(None);
    }

    let proj_out = data_vec.project_columns_with_batch_correction(
        args.proj_dim,
        Some(args.block_size),
        Some(batch_membership),
    )?;

    let proj_kn = proj_out.proj;
    info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    let collapse_out = data_vec.collapse_columns_multilevel(
        &proj_kn,
        batch_membership,
        &MultilevelParams {
            knn_super_cells: args.batch_knn,
            sort_dim: args.sort_dim,
            num_levels: args.num_levels,
            ..MultilevelParams::new(proj_kn.nrows())
        },
    )?;

    Ok(collapse_out.delta)
}

/// Estimate batch effects and write them to parquet.
///
/// Skips estimation when fewer than 2 batches are present.
/// Returns the posterior mean matrix `[n_genes × n_batches]` when
/// multi-batch, or `None` for single-batch.
pub fn estimate_and_write_batch_effects(
    data_vec: &mut SparseIoVec,
    batch_membership: &[Box<str>],
    args: EstimateBatchArgs,
    out_prefix: &str,
) -> anyhow::Result<Option<Mat>> {
    let uniq_batches: HashSet<&Box<str>> = batch_membership.iter().collect();
    let n_batches = uniq_batches.len();
    drop(uniq_batches);

    if n_batches < 2 {
        return Ok(None);
    }

    info!("Estimating batch effects ({} batches)...", n_batches);
    let batch_effects = estimate_batch(data_vec, batch_membership, args)?;

    if let Some(batch_db) = batch_effects.as_ref() {
        let outfile = out_prefix.to_string() + ".delta.parquet";
        let batch_names = data_vec.batch_names();
        let gene_names = data_vec.row_names()?;
        batch_db.to_parquet_with_names(
            &outfile,
            (Some(&gene_names), Some("gene")),
            batch_names.as_deref(),
        )?;
    }

    Ok(batch_effects.map(|x| x.posterior_mean().clone()))
}
