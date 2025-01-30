use crate::candle_data_loader::*;
use crate::candle_model_decoder::DecoderModule;
use crate::candle_model_encoder::EncoderModuleT;
use candle_core::{Device, Result, Tensor};
use candle_nn::AdamW;
use candle_nn::Optimizer;
use indicatif::*;

pub struct Vae<'a, Enc: EncoderModuleT, Dec: DecoderModule> {
    pub encoder: Enc,
    pub decoder: Dec,
    pub variable_map: &'a candle_nn::VarMap,
}

pub trait VaeT<'a, Enc: EncoderModuleT, Dec: DecoderModule> {
    /// Train the VAE model
    /// * `data` - data loader
    /// * `llik` - log likelihood function
    /// * `train_config` - training configuration
    fn train<DataL, LlikFn>(
        &mut self,
        data: DataL,
        llik: &LlikFn,
        train_config: TrainingConfig,
    ) -> Result<()>
    where
        DataL: DataLoader,
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    /// Build a VAE model
    /// * `encoder` - encoder module
    /// * `decoder` - decoder module
    fn build(encoder: Enc, decoder: Dec, variable_map: &'a candle_nn::VarMap) -> Self;
}

impl<'a, Enc, Dec> VaeT<'a, Enc, Dec> for Vae<'a, Enc, Dec>
where
    Enc: EncoderModuleT,
    Dec: DecoderModule,
{
    fn train<DataL, LlikFn>(
        &mut self,
        data: DataL,
        llik: &LlikFn,
        train_config: TrainingConfig,
    ) -> Result<()>
    where
        DataL: DataLoader,
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let device = train_config.device;

        let mut adam = AdamW::new_lr(
            self.variable_map.all_vars(),
            train_config.learning_rate.into(),
        )?;

        let pb = ProgressBar::new(train_config.num_epochs as u64);
        for _ in 0..train_config.num_epochs {
            for b in 0..data.num_minibatch() {
                let x_nd = data.minibatch(b, &device)?;
                let (z_nk, kl) = self.encoder.forward_t(&x_nd, true)?;
                let (_, llik) = self.decoder.forward_with_llik(&z_nk, &x_nd, llik)?;
                let loss = (kl - llik)?;
                adam.backward_step(&loss)?;
            }
            pb.inc(1);
        }
	pb.finish_and_clear();
        Ok(())
    }

    fn build(encoder: Enc, decoder: Dec, variable_map: &'a candle_nn::VarMap) -> Self {
        assert_eq!(encoder.dim_obs(), decoder.dim_obs());
        assert_eq!(encoder.dim_latent(), decoder.dim_latent());

        Self {
            encoder,
            decoder,
            variable_map,
        }
    }
}

pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub device: Device,
}
