#![allow(dead_code)]

use crate::candle_data_loader::*;
use crate::candle_inference::TrainConfig;
use crate::candle_model_traits::{DecoderModuleT, EncoderModuleT};

use candle_core::{Result, Tensor};
use candle_nn::AdamW;
use candle_nn::Optimizer;
use indicatif::{ProgressBar, ProgressDrawTarget};
use log::info;
use rayon::prelude::*;

pub struct Vae<'a, Enc, Dec>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
    Dec: DecoderModuleT + Send + Sync + 'static,
{
    pub encoder: &'a Enc,
    pub decoder: &'a Dec,
    pub variable_map: &'a candle_nn::VarMap,
}

pub trait VaeT<'a, Enc, Dec>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
    Dec: DecoderModuleT + Send + Sync + 'static,
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

    /// Pretrain the encoder module with pseudo latent output
    ///
    /// * `data` - data loader should have `minibatch_data`
    /// * `llik` - log likelihood function
    /// * `train_config` - training configuration
    fn pretrain_encoder<DataL, LlikFn>(
        &mut self,
        data: &mut DataL,
        llik: &LlikFn,
        train_config: &TrainConfig,
    ) -> anyhow::Result<Vec<f32>>
    where
        DataL: DataLoader,
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    /// Build a VAE model
    /// * `encoder` - encoder module
    /// * `decoder` - decoder module
    fn build(encoder: &'a Enc, decoder: &'a Dec, variable_map: &'a candle_nn::VarMap) -> Self;
}

impl<'a, Enc, Dec> VaeT<'a, Enc, Dec> for Vae<'a, Enc, Dec>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
    Dec: DecoderModuleT + Send + Sync + 'static,
{
    fn pretrain_encoder<DataL, LlikFn>(
        &mut self,
        data: &mut DataL,
        llik: &LlikFn,
        train_config: &TrainConfig,
    ) -> anyhow::Result<Vec<f32>>
    where
        DataL: DataLoader,
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let device = &train_config.device;
        let mut adam = AdamW::new_lr(
            self.variable_map.all_vars(),
            train_config.learning_rate.into(),
        )?;

        let pb = ProgressBar::new(train_config.num_pretrain_epochs as u64);

        if train_config.verbose {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        let mut llik_trace = vec![];

        data.shuffle_minibatch(train_config.batch_size)?;

        let num_minbatches = data.num_minibatch();

        let data_aux_out_vec = (0..num_minbatches)
            .map(|b| {
                data.minibatch_shuffled(b, &device)
                    .expect(format!("failed to preload minibatch #{}", b).as_str())
            })
            .collect::<Vec<_>>();

        for _epoch in 0..train_config.num_pretrain_epochs {
            let mut llik_tot = 0f32;
            for b in 0..data.num_minibatch() {
                let x = data_aux_out_vec[b].input.as_ref();
                let x0 = data_aux_out_vec[b].input_null.as_ref();
                let z = data_aux_out_vec[b].output.as_ref();

                if let Some(z_target) = z {
                    let (z_hat, kl) = self.encoder.forward_t(x, x0, true)?;

                    let llik = llik(&z_hat, z_target)?;
                    let loss = (kl - &llik)?.mean_all()?;
                    let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                    llik_tot += llik_val;
                    adam.backward_step(&loss)?;
                }
            }
            pb.inc(1);
            llik_trace.push(llik_tot / data.num_minibatch() as f32);
            if train_config.verbose {
                info!(
                    "[{}] log-likelihood: {}",
                    _epoch + 1,
                    llik_trace.last().ok_or(anyhow::anyhow!("llik"))?
                );
            }
        }
        pb.finish_and_clear();
        Ok(llik_trace)
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

        let data_aux_vec = (0..num_minbatches)
            .map(|b| {
                data.minibatch_shuffled(b, &device)
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
                data_aux_vec.par_iter().for_each(|minibatch_data| {
                    let (x_nd, x0_nd, y_nd) = (
                        minibatch_data.input.as_ref(),
                        minibatch_data.input_null.as_ref(),
                        minibatch_data.output.as_ref(),
                    );

                    let (z_nk, kl) = self
                        .encoder
                        .forward_t(x_nd, x0_nd, true)
                        .expect("enc x and x0");

                    let (_, llik) = match y_nd {
                        Some(y_nd) => self
                            .decoder
                            .forward_with_llik(&z_nk, y_nd, llik_func)
                            .expect("dec vs. y"),
                        None => self
                            .decoder
                            .forward_with_llik(&z_nk, x_nd, llik_func)
                            .expect("dec vs. x"),
                    };

                    let loss = (kl - &llik)
                        .expect("kl - llik")
                        .mean_all()
                        .expect("average");

                    let llik_val = llik
                        .sum_all()
                        .expect("llik sum")
                        .to_scalar::<f32>()
                        .expect("llik to scalar");

                    {
                        let mut adam = arc_adam.lock().expect("adam lock");
                        adam.backward_step(&loss).expect("adam backward");
                    }
                    {
                        let mut llik_tot = arc_llik_tot.lock().expect("llik lock");
                        *llik_tot += llik_val;
                    }
                });
                pb.inc(1);

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
            } // each epoch
        } else {
            for _epoch in 0..train_config.num_epochs {
                let mut llik_tot = 0f32;
                for b in 0..data.num_minibatch() {
                    let minibatch_data = &data_aux_vec[b];

                    let (x_nd, x0_nd, y_nd) = (
                        minibatch_data.input.as_ref(),
                        minibatch_data.input_null.as_ref(),
                        minibatch_data.output.as_ref(),
                    );

                    let (z_nk, kl) = self.encoder.forward_t(x_nd, x0_nd, true)?;

                    let (_, llik) = match y_nd {
                        Some(y_nd) => self.decoder.forward_with_llik(&z_nk, y_nd, llik_func)?,
                        None => self.decoder.forward_with_llik(&z_nk, x_nd, llik_func)?,
                    };
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
            } // each epoch
        }

        pb.finish_and_clear();
        Ok(llik_trace)
    }

    fn build(encoder: &'a Enc, decoder: &'a Dec, variable_map: &'a candle_nn::VarMap) -> Self {
        assert_eq!(encoder.dim_latent(), decoder.dim_latent());

        Self {
            encoder,
            decoder,
            variable_map,
        }
    }
}
