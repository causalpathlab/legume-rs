#![allow(dead_code)]

use std::usize;

use candle_core::{Result, Tensor};
use candle_nn::{ops, Module};

#[derive(Clone, Debug)]
pub struct ModularSofmaxLinear {
    logit_topic_tk: Tensor, // topic x embed, where sum_k exp(logit_topic[t,k]) = 1
    log_loading_kk: Tensor, // embed x embed, diagonal loading
    logit_module_kd: Tensor, // embed x data, where sum_k exp(logit_module[k,j]) = 1
    bias_d: Option<Tensor>, // 1 x data
}

impl ModularSofmaxLinear {
    pub fn new(
        logit_topic_tk: Tensor,
        log_loading_kk: Tensor,
        logit_module_kd: Tensor,
        bias_d: Option<Tensor>,
    ) -> Self {
        Self {
            logit_topic_tk,
            log_loading_kk,
            logit_module_kd,
            bias_d,
        }
    }

    /// take composite weights: data x topic
    pub fn weight(&self) -> Result<Tensor> {
        let min_loading = -8.;
        let max_loading = 8.;
        let eps = 1e-6;

        let topic_tk = ops::log_softmax(&self.logit_topic_tk, 1)?.exp()?;
        let loading_kk = &self.log_loading_kk.clamp(min_loading, max_loading)?.exp()?;
        let module_kd = ops::log_softmax(&self.logit_module_kd, 0)?.exp()?;

        (topic_tk.matmul(loading_kk)?.matmul(&module_kd)? + eps)?
            .log()?
            .transpose(0, 1)
    }
}

impl Module for ModularSofmaxLinear {
    fn forward(&self, _h_nt: &Tensor) -> Result<Tensor> {
        todo!("");
    }
}

pub fn module_softmax_linear(
    num_topic: usize,
    num_module: usize,
    data_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<ModularSofmaxLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;

    let logit_topic = vb.get_with_hints((num_topic, num_module), "logit.topic", init_ws)?;

    let log_loading = vb.get_with_hints((num_module, num_module), "module.loading", init_ws)?;

    let logit_module = vb.get_with_hints((num_module, data_dim), "logit.topic", init_ws)?;

    let bias = vb.get_with_hints(data_dim, "bias", candle_nn::init::ZERO)?;

    Ok(ModularSofmaxLinear::new(
        logit_topic,
        log_loading,
        logit_module,
        Some(bias),
    ))
}

#[test]
fn module_forward_test() -> Result<()> {
    Ok(())
}
