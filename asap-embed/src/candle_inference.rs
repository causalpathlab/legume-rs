use crate::candle_model_decoder::*;
use crate::candle_model_encoder::*;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{ModuleT, VarBuilder, VarMap};

// use candle_nn::{AdamW, ModuleT, Optimizer, VarBuilder, VarMap};

struct VAE<Enc: EncoderModuleT, Dec: ModuleT> {
    pub encoder: Enc,
    pub decoder: Dec,
}

pub fn train_vae_in_memory<Enc, Dec>(
    data: &Tensor,
    enc: Enc,
    dec: Dec,
    dev: Device,
    learning_rate: f32,
) where
    Enc: EncoderModuleT,
    Dec: ModuleT,
{

    let vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&vm, DType::F32, &dev);

    let vae = VAE {
        encoder: enc,
        decoder: dec,
    };

    

    // let vm = VarMap::new();

    // how should i handle data?

    // todo: think about how to cancel out batch effects
}

// /////////////////////////
// // Emedded Topic Model //
// /////////////////////////

// #[allow(dead_code)]
// pub struct ETM {
//     pub encoder: NonNegEncoder,
//     pub decoder: ETMDecoder,
// }

// #[allow(dead_code)]
// impl candle_nn::ModuleT for ETM {
//     fn forward_t(&self, x_nd: &Tensor, train: bool) -> Result<Tensor> {
//         let z_nk = self.encoder.forward_t(x_nd, train)?;
//         let logits_nd = self.decoder.forward_t(&z_nk, train)?;
//         topic_likelihood(x_nd, &logits_nd)
//     }
// }

// #[allow(dead_code)]
// impl ETM {
//     pub fn new(
//         n_features: usize,
//         n_topics: usize,
//         enc_layers: &[usize],
//         vs: VarBuilder,
//     ) -> Result<Self> {
//         let encoder = NonNegEncoder::new(n_features, n_topics, enc_layers, vs.clone())?;
//         let decoder = ETMDecoder::new(n_features, n_topics, vs.clone())?;
//         Ok(Self { encoder, decoder })
//     }
// }
