// #![allow(dead_code)]

use crate::candle_aux_layers::StackLayers;
use crate::candle_data_loader::*;
use crate::candle_inference::*;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_model_traits::*;

use candle_core::{Result, Tensor};
use candle_nn::{BatchNorm, Embedding, Linear, ModuleT, VarBuilder, ops};

pub struct MatchedLogSoftmaxEncoder {
    n_features: usize,
    n_topics: usize,
    n_vocab: usize,
    emb_x: Embedding,
    emb_logx: Embedding,
    fc: StackLayers<Linear>,
    fc_left: StackLayers<Linear>,
    fc_right: StackLayers<Linear>,
    bn: BatchNorm,
    bn_left: BatchNorm,
    bn_right: BatchNorm,
    z_average_mean: Linear,
    z_average_lnvar: Linear,
    z_left_mean: Linear,
    z_left_lnvar: Linear,
    z_right_mean: Linear,
    z_right_lnvar: Linear,
}

impl MatchedEncoderModuleT for MatchedLogSoftmaxEncoder {
    ///
    /// Returns:
    /// 1. two types of encoding results: `z_difference` and `z_average` (log-softmax)
    /// 2. a classification result to determine which mode is more feasible
    /// 3. kl-divergence
    ///
    fn forward_t(&self, data: MatchedEncoderData, train: bool) -> Result<MatchedEncoderLatent> {
        let embeded = self.preprocess_pairs(data.left, data.right, train)?;
        let (left, left_lnvar) = self.encoding_left(&embeded.left, train)?;
        let (right, right_lnvar) = self.encoding_right(&embeded.right, train)?;
        let (avg, avg_lnvar) = self.encoding_average(&embeded.average, train)?;

        let kl_div = ((gaussian_kl_loss(&left, &left_lnvar)?
            + gaussian_kl_loss(&right, &right_lnvar)?)?
            + gaussian_kl_loss(&avg, &avg_lnvar)?)?;

        let z_average = self.reparameterize(&avg, &avg_lnvar, train)?;
        let z_left = self.reparameterize(&left, &left_lnvar, train)?;
        let z_right = self.reparameterize(&right, &right_lnvar, train)?;

        let left = ops::log_softmax(&z_left, 1)?;
        let right = ops::log_softmax(&z_right, 1)?;
        let average = ops::log_softmax(&z_average, 1)?;

        Ok(MatchedEncoderLatent {
            left,
            right,
            average,
            kl_div,
        })
    }

    fn dim_latent(&self) -> usize {
        self.n_topics
    }

    fn dim_obs(&self) -> usize {
        self.n_features
    }
}

struct Preprocessed {
    left: Tensor,
    right: Tensor,
    average: Tensor,
}

impl MatchedLogSoftmaxEncoder {
    /// Will create a new non-negative encoder module
    /// with these variables:
    ///
    /// * `nn.embed_x` for intensity embedding
    /// * `nn.enc.fc.{}.weight` where {} is the layer index
    /// * `nn.enc.z.mean.weight`
    /// * `nn.enc.z.lnvar.weight`
    ///
    /// # Arguments
    /// * `n_features` - the number of features
    /// * `n_topics` - the number of topics (latent factors)
    /// * `n_vocab` - the size of intensity vocabulary
    /// * `d_emb` - vocabulary embedding dim
    /// * `layers` - fully connected layers, each with the dim
    /// * `vs` - variable builder
    pub fn new(
        n_features: usize,
        n_topics: usize,
        n_vocab: usize,
        d_emb: usize,
        layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        let bn_config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };

        let emb_x = candle_nn::embedding(n_vocab, d_emb, vs.pp("nn.embed_x"))?;
        let emb_logx = candle_nn::embedding(n_vocab, d_emb, vs.pp("nn.embed_logx"))?;

        // (1) data -> fc
        let mut fc = StackLayers::<Linear>::new();
        let mut fc_left = StackLayers::<Linear>::new();
        let mut fc_right = StackLayers::<Linear>::new();

        let mut prev_dim = n_features;
        for (j, &next_dim) in layers.iter().enumerate() {
            let _name = format!("nn.enc.fc.{}", j);
            fc.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Elu(0.1),
            );
            let _name = format!("nn.enc.fc.left.{}", j);
            fc_left.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Elu(0.1),
            );
            let _name = format!("nn.enc.fc.right.{}", j);
            fc_right.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Elu(0.1),
            );
            prev_dim = next_dim;
        }

        let bn = candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn"))?;
        let bn_left = candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn.left"))?;
        let bn_right = candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn.right"))?;

        // (2) fc -> K
        let z_left_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.left.z.mean"))?;
        let z_left_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.left.z.lnvar"))?;
        let z_right_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.right.z.mean"))?;
        let z_right_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.right.z.lnvar"))?;
        let z_average_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.average.z.mean"))?;
        let z_average_lnvar =
            candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.average.z.lnvar"))?;

        Ok(Self {
            n_features,
            n_topics,
            n_vocab,
            emb_x,
            emb_logx,
            fc,
            fc_left,
            fc_right,
            bn,
            bn_left,
            bn_right,
            z_average_mean,
            z_average_lnvar,
            z_left_mean,
            z_left_lnvar,
            z_right_mean,
            z_right_lnvar,
        })
    }

    ///
    /// Evaluate latent Gaussian parameters: mu and log_var
    /// z ~ (mu(x), log_var(x))
    ///
    fn encoding(
        &self,
        emb_nd: &Tensor,
        fc: &StackLayers<Linear>,
        bn: &BatchNorm,
        z_mean: &Linear,
        z_lnvar: &Linear,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let min_mean = -(self.n_features as f64).sqrt(); // stabilize
        let max_mean = (self.n_features as f64).sqrt(); // mean
        let min_lv = -8.; // and log variance
        let max_lv = 8.; //

        let fc_nl = fc.forward_t(emb_nd, train)?;
        let bn_nl = bn.forward_t(&fc_nl, train)?;

        let z_mean_nk = z_mean.forward_t(&bn_nl, train)?.clamp(min_mean, max_mean)?;
        let z_lnvar_nk = z_lnvar.forward_t(&bn_nl, train)?.clamp(min_lv, max_lv)?;
        Ok((z_mean_nk, z_lnvar_nk))
    }

    fn encoding_left(&self, emb_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        self.encoding(
            emb_nd,
            &self.fc_left,
            &self.bn_left,
            &self.z_left_mean,
            &self.z_left_lnvar,
            train,
        )
    }

    fn encoding_right(&self, emb_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        self.encoding(
            emb_nd,
            &self.fc_right,
            &self.bn_right,
            &self.z_right_mean,
            &self.z_right_lnvar,
            train,
        )
    }

    fn encoding_average(&self, emb_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        self.encoding(
            emb_nd,
            &self.fc,
            &self.bn,
            &self.z_average_mean,
            &self.z_average_lnvar,
            train,
        )
    }

    fn discretize_whitened_tensor(&self, x_nd: &Tensor) -> Result<Tensor> {
        use candle_core::DType::U32;

        let n_vocab = self.n_vocab as f64;
        let d = x_nd.dims().len();
        let min_val = x_nd.min_keepdim(d - 1)?;
        let max_val = x_nd.max_keepdim(d - 1)?;
        let div_val = ((max_val - &min_val)? + 1.)?;

        let x_nd = x_nd.broadcast_sub(&min_val)?.broadcast_div(&div_val)?;

        (x_nd * n_vocab)?.floor()?.to_dtype(U32)
    }

    fn preprocess_pairs(
        &self,
        left_nd: &Tensor,
        right_nd: &Tensor,
        train: bool,
    ) -> Result<Preprocessed> {
        // 1. Discretize data after log1p transformation
        let left_x_nd = self.discretize_whitened_tensor(&left_nd)?;
        let left_logx_nd = self.discretize_whitened_tensor(&(left_nd + 1.)?.log()?)?;

        let right_x_nd = self.discretize_whitened_tensor(&right_nd)?;
        let right_logx_nd = self.discretize_whitened_tensor(&(right_nd + 1.)?.log()?)?;

        let last_dim = 2_usize;

        // 2. embedding and pooling: n x d -> n x d x k
        let left_emb_ndk = (self.emb_x.forward_t(&left_x_nd, train)?
            + self.emb_logx.forward_t(&left_logx_nd, train)?)?;

        let right_emb_ndk = (self.emb_x.forward_t(&right_x_nd, train)?
            + self.emb_logx.forward_t(&right_logx_nd, train)?)?;

        Ok(Preprocessed {
            left: (&left_emb_ndk - &right_emb_ndk)?.sum(last_dim)?,
            right: (&right_emb_ndk - &left_emb_ndk)?.sum(last_dim)?,
            average: ((&left_emb_ndk + &right_emb_ndk)? * 0.5)?.sum(last_dim)?,
        })
    }

    ///
    /// z = mu + sigma * eps
    /// where eps ~ N(0, 1)
    ///
    /// # Arguments
    /// * `z_mean` - mean of Gaussian distribution
    /// * `z_lnvar` - log variance of Gaussian distribution
    fn reparameterize(&self, z_mean: &Tensor, z_lnvar: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            let eps = Tensor::randn_like(z_mean, 0., 1.)?;
            z_mean + (z_lnvar * 0.5)?.exp()? * eps
        } else {
            Ok(z_mean.clone())
        }
    }
}

pub trait MatchedEncoderEvaluateOps {
    fn evaluate<DataL: DataLoader>(
        &self,
        data: &DataL,
        train_config: &TrainConfig,
    ) -> anyhow::Result<MatchedEncoderLatent>;
}

impl MatchedEncoderEvaluateOps for MatchedLogSoftmaxEncoder {
    fn evaluate<DataL: DataLoader>(
        &self,
        data: &DataL,
        train_config: &TrainConfig,
    ) -> anyhow::Result<MatchedEncoderLatent> {
        let device = &train_config.device;
        let ntot = data.num_data();
        let batch_size = train_config.batch_size;
        let jobs = generate_minibatch_intervals(ntot, batch_size);
        let num_jobs = jobs.len();

        let mut ret = Vec::with_capacity(num_jobs);

        for (lb, ub) in jobs {
            let mb = data.minibatch_ordered(lb, ub, device)?;
            let input_nm = mb.input.as_ref();
            let input_matched_nm = mb
                .input_matched
                .as_ref()
                .ok_or(anyhow::anyhow!("need input matched"))?;

            let latent = self.forward_t(
                MatchedEncoderData {
                    left: input_nm,
                    right: input_matched_nm,
                },
                false,
            )?;
            ret.push(latent);
        }
        ret.concatenate()
    }
}

pub trait MatchedEncoderLatentVecOps {
    fn concatenate(&self) -> anyhow::Result<MatchedEncoderLatent>;
}

impl MatchedEncoderLatentVecOps for Vec<MatchedEncoderLatent> {
    fn concatenate(&self) -> anyhow::Result<MatchedEncoderLatent> {
        // Collect references to tensors for each field
        let left: Vec<&Tensor> = self.iter().map(|latent| &latent.left).collect();
        let right: Vec<&Tensor> = self.iter().map(|latent| &latent.right).collect();
        let averages: Vec<&Tensor> = self.iter().map(|latent| &latent.average).collect();

        let kl_divs: Vec<&Tensor> = self.iter().map(|latent| &latent.kl_div).collect();

        // Concatenate tensors along dimension 0
        let left = Tensor::cat(&left, 0)?;
        let right = Tensor::cat(&right, 0)?;
        let average = Tensor::cat(&averages, 0)?;
        let kl_div = Tensor::cat(&kl_divs, 0)?;

        // Return the concatenated MatchedEncoderLatent
        Ok(MatchedEncoderLatent {
            left,
            right,
            average,
            kl_div,
        })
    }
}
