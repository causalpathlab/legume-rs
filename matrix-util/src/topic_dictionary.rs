//! Topic–feature dictionary readout from a feature embedding and a set of
//! topic positions in the same space. The topic positions come from the
//! caller (cluster centroids in `senna bge`); nothing here chooses them.

use crate::traits::MatOps;
use nalgebra::DMatrix;

/// Topic–feature dictionary readout `β [D, K]` from feature embeddings
/// `rho [D, H]` and topic embeddings `alpha [K, H]`:
/// `log_softmax_d(ρ · (α − ᾱ)ᵀ)`, each column a simplex over features.
///
/// The topic positions are mean-centered across topics first: the raw loading
/// `ρ·αᵀ` is dominated by a shared "abundance" direction (the mean ᾱ) that
/// ranks the same features top in *every* topic, burying real markers;
/// reading out each topic's deviation from ᾱ surfaces topic-specific
/// features instead.
pub fn topic_dictionary(rho: &DMatrix<f32>, alpha: &DMatrix<f32>) -> DMatrix<f32> {
    // `centre_columns` on α [K, H] subtracts, per embedding dimension, the
    // mean across the K topic rows — i.e. the mean topic position ᾱ.
    (rho * alpha.centre_columns().transpose()).log_softmax_columns()
}

#[cfg(test)]
#[path = "topic_dictionary_tests.rs"]
mod tests;
