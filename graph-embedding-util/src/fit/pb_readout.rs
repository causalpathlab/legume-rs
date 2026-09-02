//! The phase-1 pseudobulk embeddings as an output: one table per collapse
//! level, with each pseudobulk's batch. Phase 1 trains the feature side against
//! exactly these, so their geometry by batch is the first place to look when
//! the per-cell embedding separates by batch — it says whether the collapse's
//! adjustment already failed or phase 2 re-introduced the effect.

use nalgebra::DMatrix;

/// One collapse level's trained pseudobulk table and its batch labels.
pub struct PbLevelEmbedding {
    /// `[n_pb × H]`.
    pub e_pb: DMatrix<f32>,
    /// Batch index of each pseudobulk: the batch its member cells belong to
    /// (the majority when a pseudobulk straddles batches; `u32::MAX` for an
    /// empty pseudobulk).
    pub batch: Vec<u32>,
}

/// The majority batch of each pseudobulk from the cell → pseudobulk map and the
/// cells' batches. Empty pseudobulks get `u32::MAX`.
#[must_use]
pub fn majority_batch_per_pb(cell_to_pb: &[usize], batch_of_cell: &[u32], n_pb: usize) -> Vec<u32> {
    let _ = (cell_to_pb, batch_of_cell, n_pb);
    todo!("majority_batch_per_pb")
}

#[cfg(test)]
mod tests;
