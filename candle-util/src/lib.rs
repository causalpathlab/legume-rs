// Re-export the upstream candle crates so downstream code can refer to
// `candle_util::candle_core` / `candle_util::candle_nn` without taking a
// direct dep.
pub use candle_core;
pub use candle_nn;

// Primitive layers, model traits, loaders, encoders, decoders, and
// training drivers — grouped by role.
pub mod cli;
pub mod data;
pub mod decoder;
pub mod encoder;
pub mod frozen_features;
pub mod loss;
pub mod mcmc;
pub mod nn;
pub mod sgvb;
pub mod topic_refinement;
pub mod traits;
pub mod value_transform;
pub mod vae;

////////////////////////////////////////////////////////////////////////
// Deprecated flat re-exports
//
// Kept so downstream crates compile while they migrate to the grouped
// paths. Each alias names the new location in its `note`. Remove once
// all callers have switched.
////////////////////////////////////////////////////////////////////////

// nn/*
#[deprecated(note = "use candle_util::nn::linear")]
pub use nn::linear as candle_aux_linear;
#[deprecated(note = "use candle_util::nn::layers")]
pub use nn::layers as candle_aux_layers;
#[deprecated(note = "use candle_util::nn::module")]
pub use nn::module as candle_aux_module;
#[deprecated(note = "use candle_util::nn::gcn")]
pub use nn::gcn as candle_aux_gcn;
#[deprecated(note = "use candle_util::nn::batch_norm")]
pub use nn::batch_norm as candle_batch_norm;

// data/*
#[deprecated(note = "use candle_util::data::loader")]
pub use data::loader as candle_data_loader;
#[deprecated(note = "use candle_util::data::loader_util")]
pub use data::loader_util as candle_data_loader_util;
#[deprecated(note = "use candle_util::data::joint")]
pub use data::joint as candle_joint_data_loader;
#[deprecated(note = "use candle_util::data::cell_grouped")]
pub use data::cell_grouped as candle_cell_grouped_data_loader;
#[deprecated(note = "use candle_util::data::indexed")]
pub use data::indexed as candle_indexed_data_loader;

// encoder/*
#[deprecated(note = "use candle_util::encoder::softmax")]
pub use encoder::softmax as candle_encoder_softmax;
#[deprecated(note = "use candle_util::encoder::softmax_iaf")]
pub use encoder::softmax_iaf as candle_encoder_softmax_iaf;
#[deprecated(note = "use candle_util::encoder::joint_softmax")]
pub use encoder::joint_softmax as candle_encoder_joint_softmax;
#[deprecated(note = "use candle_util::encoder::indexed")]
pub use encoder::indexed as candle_encoder_indexed;
#[deprecated(note = "use candle_util::encoder::cell_embedded")]
pub use encoder::cell_embedded as candle_encoder_cell_embedded;

// decoder/*
#[deprecated(note = "use candle_util::decoder::topic")]
pub use decoder::topic as candle_decoder_topic;
#[deprecated(note = "use candle_util::decoder::embedded_topic")]
pub use decoder::embedded_topic as candle_decoder_embedded_topic;
#[deprecated(note = "use candle_util::decoder::delta_topic")]
pub use decoder::delta_topic as candle_decoder_delta_topic;
#[deprecated(note = "use candle_util::decoder::joint_topic")]
pub use decoder::joint_topic as candle_decoder_joint_topic;
#[deprecated(note = "use candle_util::decoder::nb_mixture")]
pub use decoder::nb_mixture as candle_decoder_nb_mixture;
#[deprecated(note = "use candle_util::decoder::poisson")]
pub use decoder::poisson as candle_decoder_poisson;
#[deprecated(note = "use candle_util::decoder::bipartite")]
pub use decoder::bipartite as candle_bipartite_decoder;
#[deprecated(note = "use candle_util::decoder::dyn_decoder")]
pub use decoder::dyn_decoder as candle_dyn_decoder;

// traits/*
#[deprecated(note = "use candle_util::traits::model")]
pub use traits::model as candle_model_traits;
#[deprecated(note = "use candle_util::traits::indexed")]
pub use traits::indexed as candle_indexed_model_traits;

// vae/*
#[deprecated(note = "use candle_util::vae::core")]
pub use vae::core as candle_vae_inference;
#[deprecated(note = "use candle_util::vae::TrainConfig")]
pub mod candle_inference {
    #[allow(deprecated)]
    pub use crate::vae::TrainConfig;
}

// top-level renames
#[deprecated(note = "use candle_util::loss")]
pub use loss as candle_loss_functions;
#[deprecated(note = "use candle_util::mcmc")]
pub use mcmc as candle_ess;
#[deprecated(note = "use candle_util::value_transform")]
pub use value_transform as candle_value_transform;
#[deprecated(note = "use candle_util::topic_refinement")]
pub use topic_refinement as candle_topic_refinement;
