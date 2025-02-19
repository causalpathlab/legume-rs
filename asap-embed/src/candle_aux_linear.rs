#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{ops, Module};

/////////////////////////////////////////////
// Linear module with non-negative weights //
/////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct NonNegLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl NonNegLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> Result<Tensor> {
        Ok(self.weight.clone())
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

pub fn non_neg_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<NonNegLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bs = vb.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;

    Ok(NonNegLinear::new(ws, Some(bs)))
}

impl Module for NonNegLinear {
    fn forward(&self, x_nk: &Tensor) -> Result<Tensor> {
        let max_log_rate = 10.;
        let min_log_rate = -10.;

        let log_w_kd = match *x_nk.dims() {
            [b1, b2, _, _] => self.weight.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.weight.broadcast_left(bsize)?.t()?,
            _ => self.weight.t()?,
        };
        match &self.bias {
            None => x_nk.matmul(&log_w_kd.clamp(min_log_rate, max_log_rate)?.exp()?),
            Some(bias) => x_nk
                .matmul(&log_w_kd.clamp(min_log_rate, max_log_rate)?.exp()?)?
                .broadcast_add(&bias.clamp(min_log_rate, max_log_rate)?.exp()?),
        }
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
}

impl Module for SoftmaxLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let log_w_kd = match *x.dims() {
            [b1, b2, _, _] => self.weight()?.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.weight()?.broadcast_left(bsize)?.t()?,
            _ => self.weight()?.t()?,
        };
        match &self.bias {
            None => x.matmul(&log_w_kd.exp()?),
            Some(bias) => x.matmul(&log_w_kd.broadcast_add(&bias)?.exp()?),
        }
    }
}

pub fn softmax_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<SoftmaxLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bs = vb.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;

    Ok(SoftmaxLinear::new(ws, Some(bs)))
}
