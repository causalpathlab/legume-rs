use crate::srt_common::*;

use candle_util::candle_data_loader::*;
use candle_util::candle_inference::*;
use candle_util::candle_model_traits::*;
use candle_util::candle_spatial_model_traits::SpatialEncoderModuleT;
use candle_util::candle_spatial_vae_inference::*;

use log::info;

pub fn train_encoder_decoder<Enc, Dec, LLikFn>(
    input_data_nm: &Mat,
    target_data_nd: &Mat,
    spatial_data_nc: Option<&Mat>,
    null_data_nm: Option<&Mat>,
    encoder: &Enc,
    decoder: &Dec,
    parameters: &candle_nn::VarMap,
    log_likelihood_func: &LLikFn,
    train_config: &TrainConfig,
) -> anyhow::Result<Vec<f32>>
where
    Enc: SpatialEncoderModuleT + Send + Sync + 'static,
    Dec: DecoderModule + Send + Sync + 'static,
    LLikFn: Fn(&candle_core::Tensor, &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>
        + Sync
        + Send,
{
    let mut vae = SpatialVae::build(encoder, decoder, parameters);

    for var in parameters.all_vars() {
        var.to_device(&train_config.device)?;
    }

    let mut data_loader = match null_data_nm {
        Some(null_nm) => {
            info!(
                "data loader with [{}, {}] -> [{}]",
                input_data_nm.ncols(),
                null_nm.ncols(),
                target_data_nd.ncols()
            );

            match spatial_data_nc {
                Some(spatial_nc) => InMemoryData::new_with_coord_null_output(
                    input_data_nm,
                    null_nm,
                    spatial_nc,
                    target_data_nd,
                )?,
                _ => InMemoryData::new_with_null_output(input_data_nm, null_nm, target_data_nd)?,
            }
        }
        None => {
            info!(
                "data loader with [{}] -> [{}]",
                input_data_nm.ncols(),
                target_data_nd.ncols()
            );

            match spatial_data_nc {
                Some(spatial_nc) => {
                    InMemoryData::new_with_coord_output(input_data_nm, spatial_nc, target_data_nd)?
                }
                _ => InMemoryData::new_with_output(input_data_nm, target_data_nd)?,
            }
        }
    };

    let llik_trace =
        vae.train_encoder_decoder(&mut data_loader, log_likelihood_func, train_config)?;

    info!("Done with training {} epochs", train_config.num_epochs);

    Ok(llik_trace)
}
