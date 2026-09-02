//! Gene-axis alignment: carrying a trained gene side onto a new dataset's gene
//! axis with a stated status and provenance per gene.
//!
//! Every gene of the UNION of the model's axis and the new data's axis ends up
//! in exactly one of four states:
//!
//! * `Matched` — in both; the trained row, bias and membership verbatim.
//! * `Missing` — in the model, absent from the data; the trained row is kept
//!   (it stays in the partition and its rate can be predicted), nothing is
//!   observed for it.
//! * `Initialized` — in the data, absent from the model; a row placed by
//!   membership: `π̂_g` is the similarity-weighted mean of the membership rows
//!   of the `k` matched genes whose count profiles are closest, `ρ̂_g = π̂_g μ`,
//!   and the bias is set later by moment matching against pass-1 latents.
//! * `Dropped` — in the data, absent from the model, and no way to place it
//!   (no profiles were given, or the model has neither modules nor a usable
//!   neighbour).
//!
//! Pure functions over matrices: no names, no manifests, no files. The caller
//! owns name matching (`new_to_train`), the profile matrix, and where the
//! alignment is written. An initialized row is a PRIOR, never a measurement,
//! and the provenance carried here is what lets every consumer say so.

use nalgebra::DMatrix;

/// Where a union gene came from and what it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneStatus {
    Matched,
    Missing,
    Initialized,
    Dropped,
}

/// The trained module tables, when the model has them.
pub struct ModuleTables<'a> {
    /// `[D_train × M]` membership.
    pub pi: &'a DMatrix<f32>,
    /// `[M × H]` module dictionary.
    pub mu: &'a DMatrix<f32>,
}

/// Inputs to [`align_gene_axis`].
pub struct AlignInputs<'a> {
    /// Trained composed rows `[D_train × H]`.
    pub rho: &'a DMatrix<f32>,
    /// Trained per-gene bias `[D_train]`.
    pub b_feat: &'a [f32],
    /// Module tables; `None` for a free model, in which case an unseen gene is
    /// placed on the similarity-weighted mean of its neighbours' ROWS instead.
    pub modules: Option<ModuleTables<'a>>,
    /// For each NEW-data gene, the training row it matched by name, or `None`.
    pub new_to_train: &'a [Option<usize>],
    /// Count profiles of the NEW data's genes over its pseudobulks `[G_new × S]`;
    /// `None` makes every unseen gene `Dropped`.
    pub profiles_new: Option<&'a DMatrix<f32>>,
    /// Neighbours used for an initialized gene.
    pub k: usize,
    /// Below this best cosine similarity a gene is placed on the module-average
    /// (or row-average) prior and flagged `diffuse`.
    pub similarity_floor: f32,
}

/// How an `Initialized` gene was placed.
#[derive(Clone, Debug, PartialEq)]
pub struct Provenance {
    /// Training rows of the neighbours used, best first. Empty when diffuse.
    pub neighbours: Vec<usize>,
    /// Cosine similarity of the best neighbour (`0` when none).
    pub best_similarity: f32,
    /// Placed on the average prior because no neighbour reached the floor.
    pub diffuse: bool,
}

/// The aligned gene side on the union axis: training genes first, in training
/// order, then the new-only genes in the new data's order.
pub struct GeneAlignment {
    pub rows: DMatrix<f32>,
    pub bias: Vec<f32>,
    pub status: Vec<GeneStatus>,
    /// `[G_union × M]` when the model has modules: trained rows for `Matched` /
    /// `Missing`, `π̂` for `Initialized`, zeros for `Dropped`.
    pub membership: Option<DMatrix<f32>>,
    /// Per union gene; `None` unless `Initialized`.
    pub provenance: Vec<Option<Provenance>>,
    /// Union gene → training row.
    pub union_to_train: Vec<Option<usize>>,
    /// Union gene → new-data gene.
    pub union_to_new: Vec<Option<usize>>,
}

impl GeneAlignment {
    pub fn n_union(&self) -> usize {
        self.status.len()
    }

    /// Union genes with a given status.
    pub fn with_status(&self, s: GeneStatus) -> Vec<usize> {
        (0..self.n_union())
            .filter(|&g| self.status[g] == s)
            .collect()
    }
}

/// Build the alignment. See the module doc for the four states.
pub fn align_gene_axis(inputs: &AlignInputs) -> GeneAlignment {
    let _ = inputs;
    todo!("align_gene_axis")
}

/// Moment-matched bias for initialized rows: with pass-1 latents `theta [N × H]`
/// and cell biases `b_cell [N]` fixed, the bias that makes each gene's total
/// predicted count equal its total observed count,
///
/// ```text
///   â_g = log Σ_c x_cg − log Σ_c exp(ρ̂_g · θ_c + b_c)
/// ```
///
/// `rows` is `[G × H]` (the initialized rows), `observed_total` is `[G]`. A gene
/// with no observed counts gets `fallback`.
pub fn moment_matched_bias(
    rows: &DMatrix<f32>,
    theta: &DMatrix<f32>,
    b_cell: &[f32],
    observed_total: &[f32],
    fallback: f32,
) -> Vec<f32> {
    let _ = (rows, theta, b_cell, observed_total, fallback);
    todo!("moment_matched_bias")
}

#[cfg(test)]
mod tests;
