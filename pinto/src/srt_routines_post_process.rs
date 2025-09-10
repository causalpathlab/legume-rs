use crate::{srt_cell_pairs::*, srt_common::*};

use candle_util::candle_inference::TrainConfig;
use matrix_util::traits::ConvertMatOps;

pub trait SrtLatentTopicOps {
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

impl<'a> SrtLatentTopicOps for SrtCellPairs<'a> {
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

    let y_left = data.data.read_columns_csc(left)?.transpose() * aggregate_features;
    let y_right = data.data.read_columns_csc(right)?.transpose() * aggregate_features;

    ////////////////////////////////////////////////////
    // imputation by neighbours and update statistics //
    ////////////////////////////////////////////////////

    let pairs_neighbours = &data.pairs_neighbours[lb..ub];

    let y_delta_left = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, n)| -> anyhow::Result<Mat> {
            let left = pairs[j].left;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(left))?;
            let y_neigh_dm = data.data.read_columns_csc(n.right_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);
            Ok(y_d1.transpose() * aggregate_features)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let y_delta_right = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, n)| -> anyhow::Result<Mat> {
            let right = pairs[j].right;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(right))?;
            let y_neigh_dm = data.data.read_columns_csc(n.left_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);
            Ok(y_d1.transpose() * aggregate_features)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let delta_left = concatenate_vertical(&y_delta_left)?.to_tensor(dev)?;
    let delta_right = concatenate_vertical(&y_delta_right)?.to_tensor(dev)?;

    let data = MatchedEncoderData {
        marginal_left: &y_left.to_tensor(dev)?,
        marginal_right: &y_right.to_tensor(dev)?,
        delta_left: Some(&delta_left),
        delta_right: Some(&delta_right),
    };

    let latent = encoder.forward_t(data, false)?;

    latent_vec
        .lock()
        .expect("latent vec lock")
        .push((lb, latent));

    Ok(())
}
