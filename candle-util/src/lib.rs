pub mod nn;

// Deprecated flat aliases for modules now living under `nn::*`.
// Kept so downstream crates compile during migration; remove once callers
// switch to `candle_util::nn::*`.
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

pub mod candle_bipartite_decoder;
pub mod candle_decoder_delta_topic;
pub mod candle_decoder_embedded_topic;
pub mod candle_decoder_joint_topic;
pub mod candle_decoder_nb_mixture;
pub mod candle_decoder_poisson;
pub mod candle_decoder_topic;
pub mod candle_dyn_decoder;
pub mod candle_ess;
pub mod candle_inference;
pub mod candle_topic_refinement;
pub mod candle_vae_inference;
pub mod candle_value_transform;
pub mod cli;
pub mod frozen_features;
pub mod data;
pub mod encoder;
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
pub mod loss;
#[deprecated(note = "use candle_util::loss")]
pub use loss as candle_loss_functions;
pub mod traits;
#[deprecated(note = "use candle_util::traits::model")]
pub use traits::model as candle_model_traits;
#[deprecated(note = "use candle_util::traits::indexed")]
pub use traits::indexed as candle_indexed_model_traits;
pub mod sgvb;
pub mod vae;

pub use candle_core;
pub use candle_nn;
