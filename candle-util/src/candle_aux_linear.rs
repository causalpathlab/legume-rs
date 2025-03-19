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
    bias_k: Option<Tensor>,
}

impl NonNegLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        log_weight_dk: Tensor,
        bias_d: Option<Tensor>,
        bias_k: Option<Tensor>,
    ) -> Self {
        Self {
            in_dim,
            out_dim,
            log_weight_dk,
            bias_d,
            bias_k,
        }
    }

    pub fn weight(&self) -> Result<Tensor> {
        Ok(self.log_weight_dk.clone())
    }

    // pub fn bias(&self) -> Option<&Tensor> {
    //     self.bias_d.as_ref()
    // }
}

pub fn non_neg_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<NonNegLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bs_d = vb.get_with_hints((1, out_dim), "bias.d", candle_nn::init::ZERO)?;
    let bs_k = vb.get_with_hints((in_dim, 1), "bias.k", candle_nn::init::ZERO)?;

    Ok(NonNegLinear::new(
        in_dim,
        out_dim,
        ws,
        Some(bs_d),
        Some(bs_k),
    ))
}

impl Module for NonNegLinear {
    fn forward(&self, h_nk: &Tensor) -> Result<Tensor> {
        let n_max = (self.out_dim as f64).log2().min(10.0);
        let max_log_rate = n_max;
        let min_log_rate = -n_max;

        let log_w_kd = match *h_nk.dims() {
            [b1, b2, _, _] => self.log_weight_dk.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.log_weight_dk.broadcast_left(bsize)?.t()?,
            _ => self.log_weight_dk.t()?,
        };

        let log_w_kd = match &self.bias_d {
            None => log_w_kd,
            Some(bias) => log_w_kd.broadcast_add(&bias)?,
        };

        let log_w_kd = match &self.bias_k {
            None => log_w_kd.clamp(min_log_rate, max_log_rate)?,
            Some(bias) => log_w_kd
                .broadcast_add(&bias)?
                .clamp(min_log_rate, max_log_rate)?,
        };

        let log_w_kd = log_w_kd.clamp(min_log_rate, max_log_rate)?;

        h_nk.matmul(&log_w_kd.exp()?)
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
    fn forward(&self, h_nk: &Tensor) -> Result<Tensor> {
        let log_w_kd = match *h_nk.dims() {
            [b1, b2, _, _] => self.weight()?.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.weight()?.broadcast_left(bsize)?.t()?,
            _ => self.weight()?.t()?,
        };
        match &self.bias {
            None => h_nk.matmul(&log_w_kd.exp()?),
            Some(bias) => h_nk.matmul(&log_w_kd.broadcast_add(&bias)?.exp()?),
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
