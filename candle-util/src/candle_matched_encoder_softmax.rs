// #![allow(dead_code)]

use crate::candle_aux_layers::StackLayers;
use crate::candle_inference::*;
use crate::candle_loss_functions::gaussian_kl_loss;
use crate::candle_matched_data_loader::*;
use crate::candle_model_traits::*;

use candle_core::{Result, Tensor};
use candle_nn::{BatchNorm, Embedding, Linear, ModuleT, VarBuilder, ops};

pub struct MatchedLogSoftmaxEncoder {
    n_features: usize,
    n_topics: usize,
    n_vocab: usize,
    emb_x: Embedding,
    emb_logx: Embedding,
    fc_left_marginal: StackLayers<Linear>,
    fc_right_marginal: StackLayers<Linear>,
    fc_left_delta: StackLayers<Linear>,
    fc_right_delta: StackLayers<Linear>,
    bn_left_marginal: BatchNorm,
    bn_right_marginal: BatchNorm,
    bn_left_delta: BatchNorm,
    bn_right_delta: BatchNorm,
    z_left_mean: Linear,
    z_left_lnvar: Linear,
    z_right_mean: Linear,
    z_right_lnvar: Linear,
    z_left_delta_mean: Linear,
    z_left_delta_lnvar: Linear,
    z_right_delta_mean: Linear,
    z_right_delta_lnvar: Linear,
}

impl MatchedEncoderModuleT for MatchedLogSoftmaxEncoder {
    ///
    /// Returns:
    /// 1. two types of encoding results: `z_sum` and (`left`, `right`) (log-softmax)
    /// 2. a classification result to determine which mode is more feasible
    /// 3. kl-divergence
    ///
    fn forward_t(&self, data: MatchedEncoderData, train: bool) -> Result<MatchedEncoderLatent> {
        // shared (i,j) encoder -> z shared -> x(i) + x(j)

        // delta x (i) -> z delta(i) -> delta x(i)

        // adjust the marginal with the opposite (border values)
        let xb_l = self.preprocess(data.marginal_left, data.delta_right, train)?;
        let xb_r = self.preprocess(data.marginal_right, data.delta_left, train)?;

        // just the marginal as they are
        let x_l = self.preprocess(data.marginal_left, None, train)?;
        let x_r = self.preprocess(data.marginal_right, None, train)?;

        let (left_mean, left_lnvar) = self.encoding_left(&x_l, train)?;
        let (right_mean, right_lnvar) = self.encoding_right(&x_r, train)?;

        let (left_border_mean, left_border_lnvar) = self.encoding_left_boundary(&xb_l, train)?;
        let (right_border_mean, right_border_lnvar) = self.encoding_right_boundary(&xb_r, train)?;

        let kl_left = gaussian_kl_loss(&left_mean, &left_lnvar)?;
        let kl_right = gaussian_kl_loss(&right_mean, &right_lnvar)?;
        let kl_left_border = gaussian_kl_loss(&left_border_mean, &left_border_lnvar)?;
        let kl_right_border = gaussian_kl_loss(&right_border_mean, &right_border_lnvar)?;

        let kl_div = kl_left_border;
        // let kl_div = (kl_left + kl_right + kl_left_border + kl_right_border)?;

        // let z_left = (self.reparameterize(&left_mean, &left_lnvar, train)? * 0.5)?;
        let z_right = (self.reparameterize(&right_mean, &right_lnvar, train)? * 0.5)?;
        let z_left_border =
            (self.reparameterize(&left_border_mean, &left_border_lnvar, train)? * 0.5)?;
        let z_right_border =
            (self.reparameterize(&right_border_mean, &right_border_lnvar, train)? * 0.5)?;

        // let border = ops::log_softmax(&(z_left_border + z_right_border)?, 1)?;
        let z_left = self.reparameterize(&left_mean, &left_lnvar, train)?;
        let border = ops::log_softmax(&z_left_border, 1)?;
        let marginal = ops::log_softmax(&(z_left + z_right)?, 1)?;

        Ok(MatchedEncoderLatent {
            marginal,
            border,
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
        let mut fc_left_marginal = StackLayers::<Linear>::new();
        let mut fc_right_marginal = StackLayers::<Linear>::new();
        let mut fc_left_border = StackLayers::<Linear>::new();
        let mut fc_right_border = StackLayers::<Linear>::new();

        let mut prev_dim = n_features;
        for (j, &next_dim) in layers.iter().enumerate() {
            let _name = format!("nn.enc.fc.left.marginal.{}", j);
            fc_left_marginal.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Elu(0.01),
            );
            let _name = format!("nn.enc.fc.left.border.{}", j);
            fc_left_border.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Elu(0.01),
            );
            let _name = format!("nn.enc.fc.right.marginal.{}", j);
            fc_right_marginal.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Elu(0.01),
            );
            let _name = format!("nn.enc.fc.right.border.{}", j);
            fc_right_border.push_with_act(
                candle_nn::linear(prev_dim, next_dim, vs.pp(_name))?,
                candle_nn::Activation::Elu(0.01),
            );

            prev_dim = next_dim;
        }

        let bn_left_marginal =
            candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn.left.marginal"))?;
        let bn_left_border =
            candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn.left.border"))?;
        let bn_right_marginal =
            candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn.right.marginal"))?;
        let bn_right_border =
            candle_nn::batch_norm(prev_dim, bn_config, vs.pp("nn.enc.bn.right.border"))?;

        // (2) fc -> K
        let z_left_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.left.z.mean"))?;
        let z_left_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.left.z.lnvar"))?;
        let z_right_mean = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.right.z.mean"))?;
        let z_right_lnvar = candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.right.z.lnvar"))?;

        // (3) fc -> K for boundary encoding
        let z_left_border_mean =
            candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.left.z.border.mean"))?;
        let z_left_border_lnvar =
            candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.left.z.border.lnvar"))?;
        let z_right_border_mean =
            candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.right.z.border.mean"))?;
        let z_right_border_lnvar =
            candle_nn::linear(prev_dim, n_topics, vs.pp("nn.enc.right.z.border.lnvar"))?;

        Ok(Self {
            n_features,
            n_topics,
            n_vocab,
            emb_x,
            emb_logx,
            fc_left_marginal,
            fc_right_marginal,
            fc_left_delta: fc_left_border,
            fc_right_delta: fc_right_border,
            bn_left_marginal,
            bn_right_marginal,
            bn_left_delta: bn_left_border,
            bn_right_delta: bn_right_border,
            z_left_mean,
            z_left_lnvar,
            z_right_mean,
            z_right_lnvar,
            z_left_delta_mean: z_left_border_mean,
            z_left_delta_lnvar: z_left_border_lnvar,
            z_right_delta_mean: z_right_border_mean,
            z_right_delta_lnvar: z_right_border_lnvar,
        })
    }

    ///
    /// Evaluate latent Gaussian parameters: mu and log_var
    /// z ~ (mu(x), log_var(x))
    ///
    fn _encoding(
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
        self._encoding(
            emb_nd,
            &self.fc_left_marginal,
            &self.bn_left_marginal,
            &self.z_left_mean,
            &self.z_left_lnvar,
            train,
        )
    }

    fn encoding_right(&self, emb_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        self._encoding(
            emb_nd,
            &self.fc_right_marginal,
            &self.bn_right_marginal,
            &self.z_right_mean,
            &self.z_right_lnvar,
            train,
        )
    }

    fn encoding_left_boundary(&self, emb_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        self._encoding(
            emb_nd,
            &self.fc_left_delta,
            &self.bn_left_delta,
            &self.z_left_delta_mean,
            &self.z_left_delta_lnvar,
            train,
        )
    }

    fn encoding_right_boundary(&self, emb_nd: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        self._encoding(
            emb_nd,
            &self.fc_right_delta,
            &self.bn_right_delta,
            &self.z_right_delta_mean,
            &self.z_right_delta_lnvar,
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

    fn embedding(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let logx = self.discretize_whitened_tensor(&(x + 1.)?.log()?)?;
        let x = self.discretize_whitened_tensor(x)?;
        self.emb_x.forward_t(&x, train)? + self.emb_logx.forward_t(&logx, train)?
    }

    fn preprocess(&self, x: &Tensor, x_bg: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let last_dim = 2;
        if let Some(x_bg) = x_bg {
            (self.embedding(x, train)? - self.embedding(x_bg, train)?)?.sum(last_dim)
        } else {
            self.embedding(x, train)?.sum(last_dim)
        }
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

pub struct MatchedLogSoftmaxEncoderArgs<'a> {
    pub n_features: usize,
    pub n_topics: usize,
    pub n_modules: usize,
    pub n_vocab: usize,
    pub d_vocab_emb: usize,
    pub layers: &'a [usize],
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

            let latent = self.forward_t(
                MatchedEncoderData {
                    marginal_left: mb.input_marginal_left.as_ref(),
                    marginal_right: mb.input_marginal_right.as_ref(),
                    delta_left: mb.input_delta_left.as_ref(),
                    delta_right: mb.input_delta_right.as_ref(),
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
        let marginal: Vec<&Tensor> = self.iter().map(|latent| &latent.marginal).collect();
        let border: Vec<&Tensor> = self.iter().map(|latent| &latent.border).collect();

        let kl_divs: Vec<&Tensor> = self.iter().map(|latent| &latent.kl_div).collect();

        // Concatenate tensors along dimension 0
        let marginal = Tensor::cat(&marginal, 0)?;
        let border = Tensor::cat(&border, 0)?;
        let kl_div = Tensor::cat(&kl_divs, 0)?;

        // Return the concatenated MatchedEncoderLatent
        Ok(MatchedEncoderLatent {
            marginal,
            border,
            kl_div,
        })
    }
}
