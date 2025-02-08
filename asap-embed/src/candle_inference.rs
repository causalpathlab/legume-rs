use crate::candle_data_loader::*;
use crate::candle_model_decoder::DecoderModule;
use crate::candle_model_encoder::EncoderModuleT;
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
    /// * `data` - data loader
    /// * `llik` - log likelihood function
    /// * `train_config` - training configuration
    fn train<DataL, LlikFn>(
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
    fn train<DataL, LlikFn>(
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
        data.shuffle_minibatch(train_config.batch_size);

        let mut adam = AdamW::new_lr(
            self.variable_map.all_vars(),
            train_config.learning_rate.into(),
        )?;

        let pb = ProgressBar::new(train_config.num_epochs as u64);

        if train_config.verbose {
            pb.set_draw_target(ProgressDrawTarget::hidden());
        }

        let mut llik_trace = vec![];

        for _epoch in 0..train_config.num_epochs {
            let mut llik_tot = 0f32;
            for b in 0..data.num_minibatch() {
                let x_nd = data.minibatch(b, &device)?;
                let (z_nk, kl) = self.encoder.forward_t(&x_nd, true)?;
                let (_, llik) = self.decoder.forward_with_llik(&z_nk, &x_nd, llik_func)?;
                let loss = (kl - &llik)?.mean_all()?;
                let llik_val = llik.mean_all()?.to_scalar::<f32>()?;
                llik_tot += llik_val;
                adam.backward_step(&loss)?;
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
    pub device: Device,
    pub verbose: bool,
}
