//! Topic-model subcommand: SuSiE-fine-mapped joint RNA + ATAC topic
//! model. Entry point: [`fit::fit_topic_model`].

pub mod cis_mask;
pub mod coarsening;
pub mod decoder;
pub mod encoder;
pub mod eval;
pub mod fit;
pub mod input;
pub mod linkage;
pub mod susie;
pub mod training;

pub use decoder::ChickpeaDecoder;
pub use encoder::ChickpeaEncoder;
pub use susie::SuSiE;
