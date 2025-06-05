use crate::srt_common::*;

use candle_util::candle_data_loader::*;
use candle_util::candle_matched_vae_inference::*;

use log::info;

pub fn train_left_right_vae<D, Enc, Dec, LLikFn>(
    data: DataLoaderArgs<'_, D>,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    log_likelihood_func: &LLikFn,
    train_config: &TrainConfig,
) -> anyhow::Result<(Vec<f32>, MatchedEncoderLatent)>
where
    D: RowsToTensorVec,
    Enc: MatchedEncoderModuleT + MatchedEncoderEvaluateOps + Send + Sync + 'static,
    Dec: MatchedDecoderModuleT + Send + Sync + 'static,
    LLikFn: Fn(&candle_core::Tensor, &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>
        + Sync
        + Send,
{
    let mut data_loader = InMemoryData::from(data)?;

    let mut vae = MatchedVae::build(encoder, decoder, parameters);

    for var in parameters.all_vars() {
        var.to_device(&train_config.device)?;
    }

    let llik_trace =
        vae.train_encoder_decoder(&mut data_loader, log_likelihood_func, train_config)?;

    info!(
        "Done with training over {} epochs using {} samples",
        train_config.num_epochs,
        data_loader.num_data()
    );

    let latent = encoder.evaluate(&data_loader, train_config)?;

    info!("Evaluated the latent states of the training data");

    Ok((llik_trace, latent))
}
