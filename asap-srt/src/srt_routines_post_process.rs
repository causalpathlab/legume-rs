use crate::{srt_cell_pairs::*, srt_common::*};

use candle_util::candle_inference::TrainConfig;
use matrix_util::traits::ConvertMatOps;

pub trait SrtLatentStatePairsOps {
    fn evaluate_latent_states<Enc>(
        &self,
        encoder: &Enc,
        train_config: &TrainConfig,
        block_size: usize,
    ) -> anyhow::Result<MatchedEncoderLatent>
    where
        Enc: MatchedEncoderModuleT + Send + Sync + 'static;
}

impl<'a> SrtLatentStatePairsOps for SrtCellPairs<'a> {
    fn evaluate_latent_states<Enc>(
        &self,
        encoder: &Enc,
        train_config: &TrainConfig,
        block_size: usize,
    ) -> anyhow::Result<MatchedEncoderLatent>
    where
        Enc: MatchedEncoderModuleT + Send + Sync + 'static,
    {
        let njobs = self.num_pairs().div_ceil(block_size);
        let mut latent_vec = Vec::with_capacity(njobs);
        self.visit_pairs_by_block(
            &evaluate_latent_state_visitor,
            &(encoder, train_config),
            &mut latent_vec,
            block_size,
        )?;

        latent_vec.sort_by_key(|&(lb, _)| lb);
        latent_vec
            .into_iter()
            .map(|(_, x)| x)
            .collect::<Vec<_>>()
            .concatenate()
    }
}

fn evaluate_latent_state_visitor<Enc>(
    job: PairsWithBounds,
    data: &SrtCellPairs,
    encoder_config: &(&Enc, &TrainConfig),
    latent_vec: Arc<Mutex<&mut Vec<(usize, MatchedEncoderLatent)>>>,
) -> anyhow::Result<()>
where
    Enc: MatchedEncoderModuleT + Send + Sync + 'static,
{
    let (encoder, config) = *encoder_config;
    let dev = &config.device;

    let left = job.pairs.into_iter().map(|&(i, _)| i);
    let right = job.pairs.into_iter().map(|&(_, j)| j);

    let y_left = data
        .data
        .read_columns_dmatrix(left)?
        .transpose()
        .to_tensor(dev)?;
    let y_right = data
        .data
        .read_columns_dmatrix(right)?
        .transpose()
        .to_tensor(dev)?;

    let latent = encoder.forward_t(&y_left, &y_right, false)?;
    latent_vec
        .lock()
        .expect("latent vec lock")
        .push((job.lb, latent));

    Ok(())
}
