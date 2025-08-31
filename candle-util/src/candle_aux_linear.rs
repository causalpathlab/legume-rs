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

////////////////////////////////
// Linear module with Softmax //
////////////////////////////////

#[derive(Clone, Debug)]
pub struct SoftmaxLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl SoftmaxLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
    pub fn weight(&self) -> Result<Tensor> {
        ops::log_softmax(&self.weight, 0)
    }

    pub fn biased_weight(&self) -> Result<Tensor> {
        match &self.bias {
            Some(bias) => ops::log_softmax(&self.weight.broadcast_add(bias)?, 0),
            _ => ops::log_softmax(&self.weight, 0),
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
    }
}

pub fn softmax_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<SoftmaxLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bs = vb.get_with_hints((out_dim, 1), "bias", candle_nn::init::ZERO)?;

    Ok(SoftmaxLinear::new(ws, Some(bs)))
}
