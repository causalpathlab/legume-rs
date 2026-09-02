//! Per-batch gene fold for the phase-2 projection.
//!
//! The collapse estimates, per gene and batch, the fold `δ_gb` by which that
//! batch's observed counts depart from the batch-free pseudobulk rate `μ`
//! (`E[y_gb] = δ_gb · μ_g`). Phase 1 trains the dictionary on those batch-free
//! pseudobulks, so phase 2 puts each cell in the same frame by **dividing its
//! counts by its batch's fold** before the Poisson-MAP solve.
//!
//! Why divide rather than offset the rate: a low-rank log-linear dictionary
//! cannot reproduce a whole gene profile exactly, and at the optimum the Poisson
//! fit matches the count-weighted centroid in dictionary space. With `log δ` as a
//! rate offset that centroid is weighted by each batch's own `δ`, so two batches
//! solve two different weighted problems and land apart even when `δ` is right.
//! Dividing the counts first hands every batch the same estimating equation, so
//! the solution is batch-invariant regardless of misfit — measured at pseudobulk
//! resolution, where sparsity plays no part, the divide agreed across studies
//! markedly better than the offset.
//!
//! This module turns the collapse's `δ` into that fold on the unified axes:
//! batches matched **by name** (the count backend numbers batches by sorted name,
//! the unified data by first appearance), rows gathered onto the unified feature
//! axis, floored so a gene a batch never measured divides by a small positive
//! number rather than zero.

use super::projection::CellBatchFold;
use anyhow::Context;
use nalgebra::DMatrix;
use rustc_hash::FxHashMap;

/// Floor on a posterior-mean `δ`, so a gene a batch never measured stays a finite
/// positive divisor.
pub const DELTA_FLOOR: f32 = 1e-6;

/// `δ_gb` on the unified feature axis, one row per **unified** batch id.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchGeneFold {
    /// `[n_batches × n_features]` row-major, linear scale, floored.
    pub delta: Vec<f32>,
    pub n_features: usize,
    /// Unified batch names, in row order.
    pub batch_names: Vec<Box<str>>,
}

impl BatchGeneFold {
    pub fn n_batches(&self) -> usize {
        self.batch_names.len()
    }

    /// The fold row for unified batch `b`.
    pub fn row(&self, b: usize) -> &[f32] {
        &self.delta[b * self.n_features..(b + 1) * self.n_features]
    }

    /// The per-cell view phase 2 divides by: this table plus each cell's batch.
    pub(crate) fn cell_fold<'a>(&'a self, cell_to_batch: &'a [u32]) -> CellBatchFold<'a> {
        CellBatchFold {
            fold: self,
            cell_to_batch,
        }
    }
}

/// What [`batch_gene_fold`] reads.
pub(crate) struct FoldSource<'a> {
    /// The collapse's `δ` posterior mean, `[backend_rows × collapse_batches]`.
    pub delta: &'a DMatrix<f32>,
    /// The collapse's batch names, in `delta`'s column order.
    pub collapse_batch_names: &'a [Box<str>],
    /// Unified batch names, in unified batch-id order.
    pub unified_batch_names: &'a [Box<str>],
    pub n_features: usize,
    /// Unified feature → backend row.
    pub feature_to_backend: &'a [usize],
}

/// Build the per-batch fold table, or `None` when there is a single unified
/// batch (nothing to correct against).
pub(crate) fn batch_gene_fold(src: FoldSource) -> anyhow::Result<Option<BatchGeneFold>> {
    let n_batches = src.unified_batch_names.len();
    if n_batches < 2 {
        return Ok(None);
    }
    anyhow::ensure!(
        src.delta.ncols() == src.collapse_batch_names.len(),
        "collapse δ has {} columns but {} batch names",
        src.delta.ncols(),
        src.collapse_batch_names.len()
    );
    let col_of: FxHashMap<&str, usize> = src
        .collapse_batch_names
        .iter()
        .enumerate()
        .map(|(c, name)| (name.as_ref(), c))
        .collect();
    let delta =
        super::setup::gather_to_unified_axis(src.delta, src.n_features, src.feature_to_backend);
    let n_features = src.n_features;
    let mut fold = Vec::with_capacity(n_batches * n_features);
    for name in src.unified_batch_names {
        let &c = col_of.get(name.as_ref()).with_context(|| {
            format!(
                "batch {name:?} has no δ column in the collapse (collapse batches: {})",
                src.collapse_batch_names.join(", ")
            )
        })?;
        fold.extend((0..n_features).map(|f| delta[(f, c)].max(DELTA_FLOOR)));
    }
    Ok(Some(BatchGeneFold {
        delta: fold,
        n_features,
        batch_names: src.unified_batch_names.to_vec(),
    }))
}

#[cfg(test)]
mod tests;
