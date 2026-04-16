use crate::candle_decoder_topic::{
    BgmMultinomTopicDecoder, BgmNbTopicDecoder, MultinomTopicDecoder, NbTopicDecoder,
};
use crate::candle_decoder_vmf_topic::{BgmVmfTopicDecoder, VmfTopicDecoder};
use crate::candle_model_traits::{DecoderModuleT, EssLlikFn, NewDecoder};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Object-safe decoder trait for dynamic dispatch across decoder types.
///
/// All three topic decoders ignore the generic `llik` closure in
/// `DecoderModuleT::forward_with_llik` and use their own internal likelihood.
/// This trait removes the generic parameter, enabling heterogeneous collections.
pub trait DynDecoderModuleT: Send + Sync {
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor>;
    fn get_dictionary(&self) -> Result<Tensor>;
    fn forward_llik(&self, z_nk: &Tensor, x_nd: &Tensor) -> Result<(Tensor, Tensor)>;
    fn dim_obs(&self) -> usize;
    fn dim_latent(&self) -> usize;
    fn build_ess_llik<'a>(&'a self, x_nd: &'a Tensor, smoothing: f64) -> Result<EssLlikFn<'a>>;
    fn decoder_name(&self) -> &str;
}

fn dummy_llik(_: &Tensor, _: &Tensor) -> Result<Tensor> {
    Err(candle_core::Error::Msg(
        "DynDecoderModuleT: external closure should never be called".into(),
    ))
}

macro_rules! impl_dyn_decoder {
    ($ty:ty, $name:expr) => {
        impl DynDecoderModuleT for $ty {
            fn forward(&self, z_nk: &Tensor) -> Result<Tensor> {
                DecoderModuleT::forward(self, z_nk)
            }
            fn get_dictionary(&self) -> Result<Tensor> {
                DecoderModuleT::get_dictionary(self)
            }
            fn forward_llik(&self, z_nk: &Tensor, x_nd: &Tensor) -> Result<(Tensor, Tensor)> {
                DecoderModuleT::forward_with_llik(self, z_nk, x_nd, &dummy_llik)
            }
            fn dim_obs(&self) -> usize {
                DecoderModuleT::dim_obs(self)
            }
            fn dim_latent(&self) -> usize {
                DecoderModuleT::dim_latent(self)
            }
            fn build_ess_llik<'a>(&'a self, x_nd: &'a Tensor, s: f64) -> Result<EssLlikFn<'a>> {
                DecoderModuleT::build_ess_llik(self, x_nd, s)
            }
            fn decoder_name(&self) -> &str {
                $name
            }
        }
    };
}

impl_dyn_decoder!(MultinomTopicDecoder, "multinom");
impl_dyn_decoder!(NbTopicDecoder, "nb");
impl_dyn_decoder!(VmfTopicDecoder, "vmf");
impl_dyn_decoder!(BgmMultinomTopicDecoder, "bgm");
impl_dyn_decoder!(BgmNbTopicDecoder, "nbbgm");
impl_dyn_decoder!(BgmVmfTopicDecoder, "vmfbgm");

/// Create a boxed dynamic decoder by name.
pub fn create_dyn_decoder(
    name: &str,
    n_features: usize,
    n_topics: usize,
    vs: VarBuilder,
) -> Result<Box<dyn DynDecoderModuleT>> {
    match name {
        "multinom" => Ok(Box::new(MultinomTopicDecoder::new(
            n_features, n_topics, vs,
        )?)),
        "nb" => Ok(Box::new(NbTopicDecoder::new(n_features, n_topics, vs)?)),
        "vmf" => Ok(Box::new(VmfTopicDecoder::new(n_features, n_topics, vs)?)),
        "bgm" => Ok(Box::new(BgmMultinomTopicDecoder::new(
            n_features, n_topics, vs,
        )?)),
        "nbbgm" => Ok(Box::new(BgmNbTopicDecoder::new(n_features, n_topics, vs)?)),
        "vmfbgm" => Ok(Box::new(BgmVmfTopicDecoder::new(n_features, n_topics, vs)?)),
        other => Err(candle_core::Error::Msg(format!(
            "unknown decoder type: {other}"
        ))),
    }
}
