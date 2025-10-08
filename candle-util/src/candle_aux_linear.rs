#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{ops, Module};

/////////////////////////////////////////////
// Linear module with non-negative weights //
/////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct NonNegLinear {
    in_dim: usize,
    out_dim: usize,
    log_weight_dk: Tensor,
    bias_d: Option<Tensor>,
}

impl NonNegLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        log_weight_dk: Tensor,
        bias_d: Option<Tensor>,
    ) -> Self {
        Self {
            in_dim,
            out_dim,
            log_weight_dk,
            bias_d,
        }
    }

    pub fn weight(&self) -> Result<Tensor> {
        Ok(self.log_weight_dk.clone())
    }
}

pub fn non_neg_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<NonNegLinear> {
    // let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
    let bs_d = vb.get_with_hints((1, out_dim), "bias.d", candle_nn::init::ZERO)?;

    Ok(NonNegLinear::new(in_dim, out_dim, ws, Some(bs_d)))
}

impl Module for NonNegLinear {
    fn forward(&self, h_nk: &Tensor) -> Result<Tensor> {
        let log_w_kd = match *h_nk.dims() {
            [b1, b2, _, _] => self.log_weight_dk.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.log_weight_dk.broadcast_left(bsize)?.t()?,
            _ => self.log_weight_dk.t()?,
        };

        let log_w_kd = match &self.bias_d {
            None => log_w_kd,
            Some(bias) => log_w_kd.broadcast_add(bias)?,
        };

        let eps = 1e-4;
        let w_kd = (log_w_kd.relu()? + eps)?;

        h_nk.matmul(&w_kd)
    }
}

///////////////////////
// Linear Aggregator //
///////////////////////

#[derive(Clone, Debug)]
pub struct AggregateLinear {
    weight_dk: Tensor,
}

impl AggregateLinear {
    pub fn membership(&self) -> Result<Tensor> {
        ops::log_softmax(&self.weight_dk, self.weight_dk.rank() - 1)?.exp()
    }
}

impl Module for AggregateLinear {
    fn forward(&self, x_nd: &Tensor) -> Result<Tensor> {
        let c_dk = self.membership()?;
        x_nd.matmul(&c_dk)
    }
}

/// aggregate `X[n,d]` into `Y[n, k] = X[n, d] * C[d, k]` where we
/// hope to have each feature to belong to a small number of modules `{k}`
pub fn aggregate_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<AggregateLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let weight_dk = vb.get_with_hints((in_dim, out_dim), "logits", init_ws)?;
    Ok(AggregateLinear { weight_dk })
}

////////////////////////////////
// Linear module with Softmax //
////////////////////////////////

#[derive(Clone, Debug)]
pub struct SoftmaxLinear {
    weight_dk: Tensor,
    bias_d1: Option<Tensor>,
}

impl SoftmaxLinear {
    pub fn new(weight_dk: Tensor, bias_d1: Option<Tensor>) -> Self {
        Self { weight_dk, bias_d1 }
    }
    pub fn weight(&self) -> Result<Tensor> {
        ops::log_softmax(&self.weight_dk, 0)
    }

    pub fn biased_weight(&self) -> Result<Tensor> {
        match &self.bias_d1 {
            Some(bias) => ops::log_softmax(&self.weight_dk.broadcast_add(bias)?, 0),
            _ => ops::log_softmax(&self.weight_dk, 0),
        }
    }
}

impl Module for SoftmaxLinear {
    fn forward(&self, h_nk: &Tensor) -> Result<Tensor> {
        let log_w_kd = match *h_nk.dims() {
            [b1, b2, _, _] => self.biased_weight()?.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.biased_weight()?.broadcast_left(bsize)?.t()?,
            _ => self.biased_weight()?.t()?,
        };

        h_nk.matmul(&log_w_kd.exp()?)
        // the following is exact... but there is sacrifice in speed
        // candle_nn::ops::log_softmax(&(prob_nd + eps)?.log()?, 1)
        // perhaps, we should consider memory alignment?
        // let prob_dn = self.dictionary.forward(&theta_nk)?.t()?;
        // candle_nn::ops::log_softmax(&(prob_dn + eps)?.log()?, 0)?.t()
    }
}

/// create a softmax linear layer
/// `output_nd <- log( input_nk * t(β_dk) )`
/// where
/// `Σ_j β_jk = 1` for all `k`
pub fn log_softmax_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<SoftmaxLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws_dk = vb.get_with_hints((out_dim, in_dim), "logits", init_ws)?;
    let b_d1 = vb.get_with_hints((out_dim, 1), "logit_bias", candle_nn::init::ZERO)?;
    Ok(SoftmaxLinear::new(ws_dk, Some(b_d1)))
}
