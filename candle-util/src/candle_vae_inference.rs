#![allow(dead_code)]

use crate::candle_data_loader::*;
use crate::candle_model_traits::*;
use candle_core::{Device, Result, Tensor};
use candle_nn::AdamW;
use candle_nn::Optimizer;
use indicatif::{ProgressBar, ProgressDrawTarget};
use log::info;

pub struct Vae<'a, Enc: EncoderModuleT, Dec: DecoderModule> {
    pub encoder: &'a Enc,
    pub decoder: &'a Dec,
    pub variable_map: &'a candle_nn::VarMap,
}

pub trait VaeT<'a, Enc: EncoderModuleT, Dec: DecoderModule> {
    /// Train the VAE model
    /// * `data` - data loader should have `minibatch_data_aux`
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
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    /// Pretrain the encoder module with pseudo latent output
    ///
    /// * `data` - data loader should have `minibatch_data_aux_output`
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

    /// Pretrain the encoder module with pseudo latent input
    /// * `data` - data loader should have `minibatch_data_aux`
    /// * `llik` - log likelihood function
    /// * `train_config` - training configuration
    fn pretrain_decoder<DataL, LlikFn>(
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
    Enc: EncoderModuleT,
    Dec: DecoderModule,
{
    fn pretrain_decoder<DataL, LlikFn>(
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

        if train_config.verbose {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        let mut llik_trace = vec![];

        data.shuffle_minibatch(train_config.batch_size);

        let num_minbatches = data.num_minibatch();

        let x_nd_z_nk_vec = (0..num_minbatches)
            .map(|b| {
                data.minibatch_data_aux(b, &device)
                    .expect(format!("failed to preload minibatch #{}", b).as_str())
            })
            .collect::<Vec<_>>();

        for _epoch in 0..train_config.num_pretrain_epochs {
            let mut llik_tot = 0f32;

            for b in 0..data.num_minibatch() {
                let (x, z) = &x_nd_z_nk_vec[b];
                if let Some(z_target) = z {
                    let (_, llik) = self.decoder.forward_with_llik(z_target, x, llik)?;
                    let loss = (-1. * &llik)?.mean_all()?;
                    adam.backward_step(&loss)?;
                    let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                    llik_tot += llik_val;
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

        data.shuffle_minibatch(train_config.batch_size);

        let num_minbatches = data.num_minibatch();

        let data_aux_out_vec = (0..num_minbatches)
            .map(|b| {
                data.minibatch_data_aux_output(b, &device)
                    .expect(format!("failed to preload minibatch #{}", b).as_str())
            })
            .collect::<Vec<_>>();

        for _epoch in 0..train_config.num_pretrain_epochs {
            let mut llik_tot = 0f32;
            for b in 0..data.num_minibatch() {
                let (x, x0, z) = &data_aux_out_vec[b];
                if let Some(z_target) = z {
                    // let (z_hat, kl) = self.encoder.forward_t(&x, true)?;
                    let (z_hat, kl) = match x0 {
                        Some(x0) => self.encoder.forward_with_null_t(&x, &x0, true)?,
                        None => self.encoder.forward_t(&x, true)?,
                    };
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
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
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

        data.shuffle_minibatch(train_config.batch_size);

        let num_minbatches = data.num_minibatch();

        let data_aux_vec = (0..num_minbatches)
            .map(|b| {
                data.minibatch_data_aux(b, &device)
                    .expect(format!("failed to preload minibatch #{}", b).as_str())
            })
            .collect::<Vec<_>>();

        for _epoch in 0..train_config.num_epochs {
            let mut llik_tot = 0f32;
            for b in 0..data.num_minibatch() {
                let (x_nd, _x0_nd) = &data_aux_vec[b];
                let (z_nk, kl) = match _x0_nd {
                    Some(x0_nd) => self.encoder.forward_with_null_t(&x_nd, &x0_nd, true)?,
                    None => self.encoder.forward_t(&x_nd, true)?,
                };
                let (_, llik) = self.decoder.forward_with_llik(&z_nk, &x_nd, llik_func)?;
                let loss = (kl - &llik)?.mean_all()?;
                adam.backward_step(&loss)?;
                let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                llik_tot += llik_val;
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

    fn build(encoder: &'a Enc, decoder: &'a Dec, variable_map: &'a candle_nn::VarMap) -> Self {
        assert_eq!(encoder.dim_obs(), decoder.dim_obs());
        assert_eq!(encoder.dim_latent(), decoder.dim_latent());

        Self {
            encoder,
            decoder,
            variable_map,
        }
    }
}

pub struct TrainConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub num_pretrain_epochs: usize,
    pub device: Device,
    pub verbose: bool,
}
