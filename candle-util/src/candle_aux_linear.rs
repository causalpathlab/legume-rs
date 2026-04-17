#![allow(dead_code)]

use crate::candle_aux_layers::sparsemax;
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
    use_hard_assignment: bool,
}

impl AggregateLinear {
    /// Get soft membership weights (for analysis/visualization)
    pub fn membership(&self) -> Result<Tensor> {
        let eps = 1e-8;
        (ops::sigmoid(&self.weight_dk)? * (1.0 - eps))? + eps
    }

    /// Get hard assignments: which module each feature belongs to
    /// Returns [d] tensor where each value is the module index (0..k)
    pub fn get_assignments(&self) -> Result<Tensor> {
        self.weight_dk.argmax(1) // argmax along module dimension
    }

    /// Fast forward using hard assignments (scatter-based aggregation)
    fn forward_hard(&self, x_nd: &Tensor) -> Result<Tensor> {
        let k = self.weight_dk.dim(1)?; // number of modules

        // Get hard assignments: [d] where each value is module index
        let assignments_d = self.get_assignments()?;

        // Build sparse assignment matrix [d, k] with one-hot encoding
        // Convert to same dtype as input to ensure output dtype consistency
        let mut columns = Vec::with_capacity(k);
        for module_idx in 0..k {
            let mask_d = assignments_d.eq(module_idx as f64)?;
            let mask_float = mask_d.to_dtype(x_nd.dtype())?;
            columns.push(mask_float.unsqueeze(1)?);
        }
        let assignment_matrix_dk = Tensor::cat(&columns, 1)?;

        // Efficient aggregation: x_nk = x_nd @ assignment_dk
        x_nd.matmul(&assignment_matrix_dk)
    }

    /// Soft forward (original dense matmul)
    fn forward_soft(&self, x_nd: &Tensor) -> Result<Tensor> {
        let c_dk = self.membership()?;
        x_nd.matmul(&c_dk)
    }
}

impl Module for AggregateLinear {
    fn forward(&self, x_nd: &Tensor) -> Result<Tensor> {
        if self.use_hard_assignment {
            self.forward_hard(x_nd)
        } else {
            self.forward_soft(x_nd)
        }
    }
}

/// aggregate `X[n,d]` into `Y[n, k] = X[n, d] * C[d, k]` where we
/// hope to have each feature to belong to a small number of modules `{k}`
/// Uses soft assignments (dense matmul)
pub fn aggregate_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<AggregateLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let weight_dk = vb.get_with_hints((in_dim, out_dim), "logits", init_ws)?;
    Ok(AggregateLinear {
        weight_dk,
        use_hard_assignment: false,
    })
}

/// aggregate `X[n,d]` into `Y[n, k] = X[n, d] * C[d, k]` where
/// each feature belongs to exactly one module (hard assignment)
/// Much faster than soft assignment, uses scatter-gather instead of matmul
pub fn aggregate_linear_hard(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<AggregateLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let weight_dk = vb.get_with_hints((in_dim, out_dim), "logits", init_ws)?;
    Ok(AggregateLinear {
        weight_dk,
        use_hard_assignment: true,
    })
}

////////////////////////////////
// Linear module with Softmax //
////////////////////////////////

#[derive(Clone, Debug)]
pub struct SoftmaxLinear {
    weight_kd: Tensor,
    bias_1d: Option<Tensor>,
}

impl SoftmaxLinear {
    pub fn new(weight_kd: Tensor, bias_1d: Option<Tensor>) -> Self {
        Self { weight_kd, bias_1d }
    }

    pub fn weight_dk(&self) -> Result<Tensor> {
        ops::log_softmax(&self.weight_kd, self.weight_kd.rank() - 1)?
            .transpose(0, 1)?
            .contiguous()
    }

    /// Raw logits with optional bias applied (before softmax normalization).
    /// Returns `weight_kd + bias_1d` (or just `weight_kd` if no bias).
    pub(crate) fn raw_biased_logits_kd(&self) -> Result<Tensor> {
        match &self.bias_1d {
            Some(bias) => self.weight_kd.broadcast_add(bias),
            _ => Ok(self.weight_kd.clone()),
        }
    }

    /// Importance-weighted conditional log-softmax over selected features.
    ///
    /// Computes `log_softmax(w_ks - log_q_s)` where `log_q_s` is the log
    /// selection frequency of each feature. This corrects for the sampling
    /// bias of top-K feature selection, approximating full-vocabulary softmax.
    ///
    /// Reference: Jean et al. (2015), "On Using Very Large Target Vocabulary
    /// for Neural Machine Translation", ACL.
    ///
    /// O(K·S) forward and backward — no [K, D] materialization.
    pub(crate) fn biased_weight_ks_conditional(
        &self,
        union_indices: &Tensor,
        log_q_s: &Tensor,
    ) -> Result<Tensor> {
        let w_ks = self.weight_kd.index_select(union_indices, 1)?;
        let w_ks = match &self.bias_1d {
            Some(bias) => {
                let bias_s = bias.index_select(union_indices, 1)?;
                w_ks.broadcast_add(&bias_s)?
            }
            _ => w_ks,
        };
        // Importance correction: subtract log selection frequency before softmax
        let w_ks = w_ks.broadcast_sub(log_q_s)?;
        ops::log_softmax(&w_ks, w_ks.rank() - 1)
    }

    pub(crate) fn biased_weight_kd(&self) -> Result<Tensor> {
        match &self.bias_1d {
            Some(bias) => ops::log_softmax(
                &self.weight_kd.broadcast_add(bias)?,
                self.weight_kd.rank() - 1,
            ),
            _ => ops::log_softmax(&self.weight_kd, self.weight_kd.rank() - 1),
        }?
        .contiguous()
    }

    /// Log-space forward: log(recon_nd) via logsumexp
    ///
    /// log(Σ_k z_nk * β_kd) = logsumexp_k(log(z_nk) + log(β_kd))
    ///
    /// Input `log_h_nk` is already in log-space (from encoder's log_softmax).
    pub fn forward_log(&self, log_h_nk: &Tensor) -> Result<Tensor> {
        logsumexp_forward(log_h_nk, &self.biased_weight_kd()?)
    }
}

impl Module for SoftmaxLinear {
    /// Input `log_h_nk` is log-probabilities from the encoder.
    fn forward(&self, log_h_nk: &Tensor) -> Result<Tensor> {
        self.forward_log(log_h_nk)?.exp()
    }
}

/// Log-space forward: log(Σ_k z_nk * β_kd) = logsumexp_k(log_z_nk + log_β_kd).
/// Handles broadcasting for batched inputs (2D, 3D, 4D).
pub fn logsumexp_forward(log_h_nk: &Tensor, log_w_kd: &Tensor) -> Result<Tensor> {
    let log_w_kd = match *log_h_nk.dims() {
        [b1, b2, _, _] => log_w_kd.broadcast_left((b1, b2))?,
        [bsize, _, _] => log_w_kd.broadcast_left(bsize)?,
        _ => log_w_kd.clone(),
    };
    let log_h = log_h_nk.unsqueeze(2)?;
    let log_w = log_w_kd.unsqueeze(0)?;
    log_h.broadcast_add(&log_w)?.log_sum_exp(1)
}

/// Create a softmax linear layer with a per-output `logit_bias`:
///   `β_kd = softmax_d(logits_kd + bias_d)`
///   `output_nd = log( input_nk · β_kd )`
///
/// The per-output bias is load-bearing: it absorbs baseline gene-frequency
/// offsets so the per-topic logits encode *deviations* from baseline.
/// Empirically, removing it leaves the multinom decoder near-degenerate
/// (≈7× worse train llik on BMMNC). NB-family decoders tolerate `_nobias`
/// because the per-gene dispersion / NbMixture's α absorb baseline
/// instead, but they still benefit slightly from the bias term.
/// Default to `log_softmax_linear` unless you have a specific reason.
pub fn log_softmax_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<SoftmaxLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws_kd = vb.get_with_hints((in_dim, out_dim), "logits", init_ws)?;
    let b_1d = vb.get_with_hints((1, out_dim), "logit_bias", candle_nn::init::ZERO)?;
    Ok(SoftmaxLinear::new(ws_kd, Some(b_1d)))
}

/// Bias-free variant — see [`log_softmax_linear`] for the bias rationale.
/// Currently only used by the indexed-topic decoder where top-K masking
/// already breaks per-gene baseline absorption.
pub fn log_softmax_linear_nobias(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<SoftmaxLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws_kd = vb.get_with_hints((in_dim, out_dim), "logits", init_ws)?;
    Ok(SoftmaxLinear::new(ws_kd, None))
}

/////////////////////////////
// Sparsemax linear layer //
/////////////////////////////

#[derive(Clone, Debug)]
pub struct SparsemaxLinear {
    weight_kd: Tensor,
    bias_1d: Option<Tensor>,
}

impl SparsemaxLinear {
    pub fn new(weight_kd: Tensor, bias_1d: Option<Tensor>) -> Self {
        Self { weight_kd, bias_1d }
    }

    pub fn weight_dk(&self) -> Result<Tensor> {
        sparsemax(&self.weight_kd)?.transpose(0, 1)?.contiguous()
    }

    fn biased_weight_kd(&self) -> Result<Tensor> {
        match &self.bias_1d {
            Some(bias) => sparsemax(&self.weight_kd.broadcast_add(bias)?),
            _ => sparsemax(&self.weight_kd),
        }
    }

    /// Log-space forward: log(recon_nd) = log(z_nk @ sparsemax(W) + eps)
    ///
    /// Input `log_h_nk` is log-probabilities; exp to get probs for matmul.
    /// Sparsemax has exact zeros so we can't use pure log-space logsumexp.
    pub fn forward_log(&self, log_h_nk: &Tensor) -> Result<Tensor> {
        let h_nk = log_h_nk.exp()?;
        let w_kd = match *h_nk.dims() {
            [b1, b2, _, _] => self.biased_weight_kd()?.broadcast_left((b1, b2))?,
            [bsize, _, _] => self.biased_weight_kd()?.broadcast_left(bsize)?,
            _ => self.biased_weight_kd()?,
        };
        let eps = 1e-20;
        (h_nk.matmul(&w_kd)? + eps)?.log()
    }
}

impl Module for SparsemaxLinear {
    /// Input `log_h_nk` is log-probabilities; exp to get probs for matmul.
    fn forward(&self, log_h_nk: &Tensor) -> Result<Tensor> {
        let h_nk = log_h_nk.exp()?;
        let w_kd = match *h_nk.dims() {
            [b1, b2, _, _] => self.biased_weight_kd()?.broadcast_left((b1, b2))?,
            [bsize, _, _] => self.biased_weight_kd()?.broadcast_left(bsize)?,
            _ => self.biased_weight_kd()?,
        };
        h_nk.matmul(&w_kd)
    }
}

/// create a sparsemax linear layer
/// `output_nd <- input_nk * t(β_dk)`
/// where β_dk = sparsemax(weight_kd) is sparse and sums to 1
pub fn sparsemax_linear(
    in_dim: usize,
    out_dim: usize,
    vb: candle_nn::VarBuilder,
) -> Result<SparsemaxLinear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws_kd = vb.get_with_hints((in_dim, out_dim), "logits", init_ws)?;
    let b_1d = vb.get_with_hints((1, out_dim), "logit_bias", candle_nn::init::ZERO)?;
    Ok(SparsemaxLinear::new(ws_kd, Some(b_1d)))
}

/////////////////////////
// modularized softmax //
/////////////////////////
