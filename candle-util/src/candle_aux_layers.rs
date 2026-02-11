#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{ops, Activation, Linear, Module};

use crate::candle_loss_functions::gaussian_kl_loss;

/// build inverse autoregressive flow layers
pub struct IAFLayers<M, N>
where
    M: Module,
    N: Module,
{
    z_mean: M,
    z_lnvar: M,
    autoregressive_mean: Vec<N>,    // K+L -> K
    autoregressive_sigmoid: Vec<N>, // K+L -> K
}

impl<M, N> IAFLayers<M, N>
where
    M: Module,
    N: Module,
{
    fn iaf_step(&self, h: &Tensor, depth: usize, train: bool) -> Result<(Tensor, Tensor)> {
        if depth == 0 {
            let min_lv = -10.; // clamp log variance
            let max_lv = 10.; //

            let z_mean = self.z_mean.forward(&h)?;
            let z_lnvar = self.z_lnvar.forward(&h)?.clamp(min_lv, max_lv)?;
            let kl = gaussian_kl_loss(&z_mean, &z_lnvar)?;
            let z = if train {
                let eps = Tensor::randn_like(&z_mean, 0., 1.)?;
                (z_mean + (z_lnvar * 0.5)?.exp()?.mul(&eps)?)?
            } else {
                z_mean
            };
            Ok((z, kl))
        } else {
            let (z_prev, kl) = self.iaf_step(h, depth - 1, train)?;
            let zh = Tensor::cat(&[&z_prev, h], h.rank() - 1)?;

            let z_new = self.autoregressive_mean[depth - 1].forward(&zh)?;
            let s = self.autoregressive_sigmoid[depth - 1].forward(&zh)?;

            let eps = 1e-8;
            let p_stay = ((ops::sigmoid(&s)? * (1.0 - 2.0 * eps))? + eps)?;
            let p_explore = ((ops::sigmoid(&s.neg()?)? * (1.0 - 2.0 * eps))? + eps)?;
            let z = (z_new.mul(&p_explore)? + z_prev.mul(&p_stay)?)?;
            let kl = kl.sub(&p_stay.log()?.sum(p_stay.rank() - 1)?)?;

            Ok((z, kl))
        }
    }

    /// Run Inverse Autoregressive Flow and get the latent Z and KL divergence
    ///
    /// * `h` - hidden `n x k` states
    /// * `train` - train or not
    pub fn flow(&self, h: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        self.iaf_step(h, self.autoregressive_mean.len(), train)
    }

    /// Build Inverse Autoregressive Flow layers (Kingma et al. 2016)
    pub fn new(z_mean: M, z_lnvar: M, mean_layers: Vec<N>, sigmoid_layers: Vec<N>) -> Self {
        debug_assert_eq!(mean_layers.len(), sigmoid_layers.len());
        Self {
            z_mean,
            z_lnvar,
            autoregressive_mean: mean_layers,
            autoregressive_sigmoid: sigmoid_layers,
        }
    }
}

/// build IAF layers with STACK layers
pub fn iaf_stack_linear(
    in_dim: usize,
    out_dim: usize,
    n_layers: usize,
    n_stack_layers: &[usize],
    vb: candle_nn::VarBuilder,
) -> Result<IAFLayers<Linear, StackLayers<Linear>>> {
    let z_mean = candle_nn::linear(in_dim, out_dim, vb.pp("iaf.z.mean"))?;
    let z_lnvar = candle_nn::linear(in_dim, out_dim, vb.pp("iaf.z.lnvar"))?;

    let mut mean_layers = vec![];
    let mut sigmoid_layers = vec![];

    for j in 0..n_layers {
        let mean = stack_relu_linear(
            in_dim + out_dim,
            out_dim,
            n_stack_layers,
            vb.pp(format!("iaf.mean.{}", j)),
        )?;
        let sigmoid = stack_relu_linear(
            in_dim + out_dim,
            out_dim,
            n_stack_layers,
            vb.pp(format!("iaf.sigmoid.{}", j)),
        )?;
        mean_layers.push(mean);
        sigmoid_layers.push(sigmoid);
    }

    let iaf_layers = IAFLayers::new(z_mean, z_lnvar, mean_layers, sigmoid_layers);
    Ok(iaf_layers)
}

/// build a stack of alternating `M` and `A` layers
pub struct StackLayers<M>
where
    M: Module,
{
    module_layers: Vec<M>,
    activation_layers: Vec<Option<Activation>>,
}

impl<M> Module for StackLayers<M>
where
    M: Module,
{
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for (module, activation) in self.module_layers.iter().zip(self.activation_layers.iter()) {
            x = module.forward(&x)?;
            if let Some(activation) = activation {
                x = activation.forward(&x)?;
            }
        }
        Ok(x)
    }
}

impl<M> StackLayers<M>
where
    M: Module,
{
    /// build a stack of alternating `M` and `A` layers
    pub fn new() -> Self {
        Self {
            module_layers: Vec::new(),
            activation_layers: Vec::new(),
        }
    }

    /// add a new layer with an activation function
    pub fn push_with_act(&mut self, layer: M, activation: Activation) {
        self.module_layers.push(layer);
        self.activation_layers.push(Some(activation));
    }

    /// add a new layer
    pub fn push(&mut self, layer: M) {
        self.module_layers.push(layer);
        self.activation_layers.push(None);
    }
}

impl<M> Default for StackLayers<M>
where
    M: Module,
{
    fn default() -> Self {
        Self::new()
    }
}

/// create stacked relu linear
/// * `in_dim` - input
/// * `out_dim` - output
/// * `intermediate_dims` - intermediate layers
/// * `vb` - variable builder
/// * `var_header` - variable name header
pub fn stack_relu_linear(
    in_dim: usize,
    out_dim: usize,
    intermediate_dims: &[usize],
    vb: candle_nn::VarBuilder,
) -> Result<StackLayers<Linear>> {
    let mut prev_dim = in_dim;
    let mut ret = StackLayers::<Linear>::new();
    for (k, &next_dim) in intermediate_dims.iter().enumerate() {
        let _name = format!("relu_linear_stack.{}", k);
        ret.push_with_act(
            candle_nn::linear(prev_dim, next_dim, vb.pp(_name))?,
            candle_nn::Activation::Relu,
        );
        prev_dim = next_dim;
    }
    // add the final layer
    let k = intermediate_dims.len();
    let next_dim = out_dim;
    let _name = format!("relu_linear_stack.{}", k);
    ret.push_with_act(
        candle_nn::linear(prev_dim, next_dim, vb.pp(_name))?,
        candle_nn::Activation::Relu,
    );
    Ok(ret)
}

///////////////////////////////////
// Sparsemax activation function //
///////////////////////////////////

/// Sparsemax activation function (Martins & Astudillo, 2016)
///
/// Projects input onto the probability simplex, producing sparse outputs.
/// Unlike softmax, can output exact zeros.
///
/// * `z` - input tensor of shape (batch, dim)
///
/// Returns tensor of same shape with values in [0, 1] summing to 1 along last dim.
///
pub fn sparsemax(z: &Tensor) -> Result<Tensor> {
    let z = z.contiguous()?; // ensure contiguous for sort_last_dim
    let dim = z.rank() - 1;
    let (z_sorted, _indices) = z.sort_last_dim(false)?; // descending order
    let k = z.dim(dim)?;
    let device = z.device();
    let dtype = z.dtype();

    // Compute cumsum of sorted values
    let cumsum = z_sorted.cumsum(dim)?;

    // Compute 1 + i * z_sorted[i] for i = 1..k
    let range = Tensor::arange(1.0, (k + 1) as f64, device)?.to_dtype(dtype)?;
    // Broadcast range to match z shape
    let shape: Vec<usize> = (0..z.rank())
        .map(|i| if i == dim { k } else { 1 })
        .collect();
    let range = range.reshape(shape.as_slice())?;
    let bound = (z_sorted.broadcast_mul(&range)? + 1.0)?;

    // Find support: where bound > cumsum
    let support = bound.gt(&cumsum)?;

    // Count support size per row (sum of True values)
    let support_f = support.to_dtype(dtype)?;
    let support_size = support_f.sum_keepdim(dim)?;

    // tau = (sum(z_sorted * support) - 1) / support_size
    let z_sorted_masked = z_sorted.broadcast_mul(&support_f)?;
    let z_sum_support = z_sorted_masked.sum_keepdim(dim)?;
    let tau = (z_sum_support - 1.0)?.broadcast_div(&support_size.clamp(1.0, f64::INFINITY)?)?;

    // Output: max(z - tau, 0)
    z.broadcast_sub(&tau)?.clamp(0.0, f64::INFINITY)
}
