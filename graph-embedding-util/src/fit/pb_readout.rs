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
    let n_batches = batch_of_cell
        .iter()
        .copied()
        .max()
        .map_or(0, |b| b as usize + 1);
    let mut counts = vec![vec![0usize; n_batches]; n_pb];
    for (c, &p) in cell_to_pb.iter().enumerate() {
        if p < n_pb {
            counts[p][batch_of_cell[c] as usize] += 1;
        }
    }
    counts
        .iter()
        .map(|row| {
            let best = row.iter().copied().max().unwrap_or(0);
            if best == 0 {
                return u32::MAX;
            }
            row.iter()
                .position(|&n| n == best)
                .expect("max is in the row") as u32
        })
        .collect()
}

#[cfg(test)]
mod tests;
