use crate::srt_common::*;

use candle_util::candle_data_loader::*;
use candle_util::candle_matched_vae_inference::*;

use log::info;

pub fn train_encoder_decoder<Enc, Dec, LLikFn>(
    input_nm: &Mat,
    input_matched_nm: &Mat,
    output_nd: &Mat,
    output_matched_nd: &Mat,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    log_likelihood_func: &LLikFn,
    train_config: &TrainConfig,
) -> anyhow::Result<(Vec<f32>, MatchedEncoderLatent)>
where
    Enc: MatchedEncoderModuleT + MatchedEncoderEvaluateOps + Send + Sync + 'static,
    Dec: MatchedDecoderModuleT + Send + Sync + 'static,
    LLikFn: Fn(&candle_core::Tensor, &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>
        + Sync
        + Send,
{
    let mut data_loader = InMemoryData::new_with_matched_input_and_matched_output(
        input_nm,
        input_matched_nm,
        output_nd,
        output_matched_nd,
    )?;

    let mut vae = MatchedVae::build(encoder, decoder, parameters);

    for var in parameters.all_vars() {
        var.to_device(&train_config.device)?;
    }

    let llik_trace =
        vae.train_encoder_decoder(&mut data_loader, log_likelihood_func, train_config)?;

    info!("Done with training {} epochs", train_config.num_epochs);

    let latent = encoder.evaluate(&data_loader, train_config)?;

    Ok((llik_trace, latent))
}
