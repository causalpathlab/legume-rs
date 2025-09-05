use crate::candle_inference::TrainConfig;
use crate::candle_matched_data_loader::*;
use crate::candle_model_traits::*;

use candle_core::{Result, Tensor};
use candle_nn::AdamW;
use candle_nn::Optimizer;
use indicatif::{ProgressBar, ProgressDrawTarget};
use log::info;
use rayon::prelude::*;

pub struct MatchedVae<'a, Enc, Dec>
where
    Enc: MatchedEncoderModuleT + Send + Sync + 'static,
    Dec: MatchedDecoderModuleT + Send + Sync + 'static,
{
    pub encoder: &'a Enc,
    pub decoder: &'a Dec,
    pub variable_map: &'a candle_nn::VarMap,
}

pub trait DiffVaeT<'a, Enc, Dec>
where
    Enc: MatchedEncoderModuleT + Send + Sync + 'static,
    Dec: MatchedDecoderModuleT + Send + Sync + 'static,
{
    /// Train the VAE model
    /// * `data` - data loader should have `minibatch_data`
    /// * `llik` - log likelihood function
    /// * `train_config` - training configuration
    fn train_encoder_decoder<DataL, LlikFn>(
        &mut self,
        data: &mut DataL,
        llik: &LlikFn,
        train_config: &TrainConfig,
    ) -> anyhow::Result<Vec<f32>>
    where
        DataL: DataLoader,
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor> + Sync + Send;

    /// Build a VAE model
    /// * `encoder` - encoder module
    /// * `decoder` - decoder module
    fn build(encoder: &'a Enc, decoder: &'a Dec, variable_map: &'a candle_nn::VarMap) -> Self;
}

impl<'a, Enc, Dec> DiffVaeT<'a, Enc, Dec> for MatchedVae<'a, Enc, Dec>
where
    Enc: MatchedEncoderModuleT + Send + Sync + 'static,
    Dec: MatchedDecoderModuleT + Send + Sync + 'static,
{
    fn build(encoder: &'a Enc, decoder: &'a Dec, variable_map: &'a candle_nn::VarMap) -> Self {
        assert_eq!(encoder.dim_latent(), decoder.dim_latent());

        Self {
            encoder,
            decoder,
            variable_map,
        }
    }

    fn train_encoder_decoder<DataL, LlikFn>(
        &mut self,
        data: &mut DataL,
        llik_func: &LlikFn,
        train_config: &TrainConfig,
    ) -> anyhow::Result<Vec<f32>>
    where
        DataL: DataLoader,
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor> + Sync + Send,
    {
        let device = &train_config.device;
        let mut adam = AdamW::new_lr(
            self.variable_map.all_vars(),
            train_config.learning_rate.into(),
        )?;

        let pb = ProgressBar::new(train_config.num_epochs as u64);

        if train_config.verbose {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        let mut llik_trace = vec![];

        data.shuffle_minibatch(train_config.batch_size)?;

        let num_minbatches = data.num_minibatch();

        let data_vec = (0..num_minbatches)
            .map(|b| {
                data.minibatch_shuffled(b, device)
                    .expect(format!("failed to preload minibatch #{}", b).as_str())
            })
            .collect::<Vec<_>>();

        if train_config.device.is_cpu() {
            use std::sync::{Arc, Mutex};
            let arc_adam = Arc::new(Mutex::new(&mut adam));
            let arc_llik_tot = Arc::new(Mutex::new(0_f32));

            for _epoch in 0..train_config.num_epochs {
                {
                    let mut llik_tot = arc_llik_tot.lock().expect("llik lock");
                    *llik_tot = 0_f32;
                }
                data_vec
                    .par_iter()
                    .try_for_each(|mb| -> anyhow::Result<()> {
                        let latent = self.encoder.forward_t(
                            MatchedEncoderData {
                                marginal_left: mb.input_marginal_left.as_ref(),
                                marginal_right: mb.input_marginal_right.as_ref(),
                                delta_left: mb.input_delta_left.as_ref(),
                                delta_right: mb.input_delta_right.as_ref(),
                            },
                            true,
                        )?;

                        let kl = &latent.kl_div;

                        let (_, llik) = self.decoder.forward_with_llik(
                            &latent,
                            MatchedDecoderData {
                                marginal_left: mb
                                    .output_marginal_left
                                    .as_ref()
                                    .ok_or(anyhow::anyhow!("need output left"))?,
                                marginal_right: mb
                                    .output_marginal_right
                                    .as_ref()
                                    .ok_or(anyhow::anyhow!("need output right"))?,
                                border_left: mb.output_delta_left.as_ref(),
                                border_right: mb.output_delta_right.as_ref(),
                            },
                            llik_func,
                        )?;

                        let loss = (kl - &llik)?.mean_all()?;
                        let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                        {
                            let mut adam = arc_adam.lock().expect("adam lock");
                            adam.backward_step(&loss)?;
                        }
                        {
                            let mut llik_tot = arc_llik_tot.lock().expect("llik lock");
                            *llik_tot += llik_val;
                        }
                        Ok(())
                    })?;

                {
                    let llik_tot = arc_llik_tot.lock().expect("llik lock");
                    llik_trace.push(*llik_tot / data.num_minibatch() as f32);
                }

                if train_config.verbose {
                    info!(
                        "[{}] log-likelihood: {}",
                        _epoch + 1,
                        llik_trace.last().ok_or(anyhow::anyhow!("llik"))?
                    );
                }
                pb.inc(1);
            }
        } else {
            for _epoch in 0..train_config.num_epochs {
                let mut llik_tot = 0f32;

                for mb in &data_vec {
                    let latent = self.encoder.forward_t(
                        MatchedEncoderData {
                            marginal_left: mb.input_marginal_left.as_ref(),
                            marginal_right: mb.input_marginal_right.as_ref(),
                            delta_left: mb.input_delta_left.as_ref(),
                            delta_right: mb.input_delta_right.as_ref(),
                        },
                        true,
                    )?;
                    let kl = &latent.kl_div;

                    let (_, llik) = self.decoder.forward_with_llik(
                        &latent,
                        MatchedDecoderData {
                            marginal_left: mb
                                .output_marginal_left
                                .as_ref()
                                .ok_or(anyhow::anyhow!("need output left"))?,
                            marginal_right: mb
                                .output_marginal_right
                                .as_ref()
                                .ok_or(anyhow::anyhow!("need output right"))?,
                            border_left: mb.output_delta_left.as_ref(),
                            border_right: mb.output_delta_right.as_ref(),
                        },
                        llik_func,
                    )?;

                    let loss = (kl - &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;
                    let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                    llik_tot += llik_val;
                }
                llik_trace.push(llik_tot / data.num_minibatch() as f32);
                pb.inc(1);
                if train_config.verbose {
                    info!(
                        "[{}] log-likelihood: {}",
                        _epoch + 1,
                        llik_trace.last().ok_or(anyhow::anyhow!("llik"))?
                    );
                }
            }
            pb.finish_and_clear();
        }
        Ok(llik_trace)
    }
}
