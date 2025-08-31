use crate::{srt_cell_pairs::*, srt_common::*};

use candle_util::candle_inference::TrainConfig;
use matrix_util::traits::ConvertMatOps;

pub trait SrtLatentStatePairsOps {
    fn evaluate_latent_states<Enc>(
        &self,
        encoder: &Enc,
        aggregate_rows: &Mat,
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
        aggregate_features: &Mat,
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
            &(encoder, aggregate_features, train_config),
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
    bound: (usize, usize),
    data: &SrtCellPairs,
    encoder_aggregate_config: &(&Enc, &Mat, &TrainConfig),
    latent_vec: Arc<Mutex<&mut Vec<(usize, MatchedEncoderLatent)>>>,
) -> anyhow::Result<()>
where
    Enc: MatchedEncoderModuleT + Send + Sync + 'static,
{
    let (encoder, aggregate_features, config) = *encoder_aggregate_config;
    let dev = &config.device;

    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let left = pairs.into_iter().map(|pp| pp.left);
    let right = pairs.into_iter().map(|pp| pp.right);

    let y_left = data.data.read_columns_dmatrix(left)?.transpose() * aggregate_features;
    let y_right = data.data.read_columns_dmatrix(right)?.transpose() * aggregate_features;

    // todo: 
    // let pairs_neighbours = &data.pairs_neighbours[lb..ub];

    unimplemented!("");

    // let latent = encoder.forward_t(
    //     MatchedEncoderData {
    //         marginal_left: &y_left.to_tensor(dev)?,
    //         marginal_right: &y_right.to_tensor(dev)?,
    //     },
    //     false,
    // )?;
    // latent_vec
    //     .lock()
    //     .expect("latent vec lock")
    //     .push((lb, latent));

    Ok(())
}
