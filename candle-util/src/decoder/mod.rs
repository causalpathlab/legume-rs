//! Decoder modules for VAE-style topic / link-community models.
//!
//! - [`topic`]: dense multinomial / NB topic decoders
//! - [`delta_topic`]: delta-parameterized topic decoder
//! - [`joint_topic`]: paired/multi-view topic decoder
//! - [`nb_mixture`]: ambient-RNA NB mixture topic decoder
//! - [`poisson`]: simple Poisson decoder
//! - [`bipartite`]: bipartite link-community decoder
//! - [`dyn_decoder`]: trait-object wrapper for runtime decoder selection

pub mod bipartite;
pub mod delta_topic;
pub mod dyn_decoder;
pub mod gaussian_nb;
pub mod gem_etm;
pub mod joint_topic;
pub mod masked_etm;
pub mod nb_mixture;
pub mod poisson;
pub mod topic;

pub use bipartite::{
    BipartiteDecoder, BipartiteLikelihood, BlockModelMultinomial, GaussianLikelihood, NbLikelihood,
    PoissonLikelihood, SymmetricMultinomial,
};
pub use delta_topic::DeltaTopicDecoder;
pub use dyn_decoder::{create_dyn_decoder, DynDecoderModuleT};
pub use gaussian_nb::GaussianNbDecoder;
pub use gem_etm::{GemEtmDecoder, GemMaskedTarget, Track};
pub use joint_topic::JointTopicDecoder;
pub use masked_etm::{EmbeddedNbTopicDecoder, MaskedNbTarget};
pub use nb_mixture::NbMixtureTopicDecoder;
pub use poisson::PoissonDecoder;
pub use topic::{MultinomTopicDecoder, NbTopicDecoder};

/// `(start, len)` pairs tiling `n_features` in slices of at most `chunk`.
///
/// The gene-slicing loop every chunked `llik_gene_chunked` runs, in one place:
/// four decoders had begun to hand-roll the same offset bookkeeping. Yields
/// nothing for an empty feature axis, so a caller folding over it is left with
/// its own "no slices" case rather than a division by zero.
pub(crate) fn gene_slices(n_features: usize, chunk: usize) -> impl Iterator<Item = (usize, usize)> {
    let chunk = chunk.max(1);
    (0..n_features)
        .step_by(chunk)
        .map(move |start| (start, chunk.min(n_features - start)))
}
