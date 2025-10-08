#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::Module;

#[derive(Clone, Debug)]
pub struct RankEmbedding {
    nvocab: usize,
    embeddings: Tensor,
    hidden_size: usize,
}

impl RankEmbedding {
    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for RankEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut final_dims = xs.dims().to_vec();
        let last_dim_size = *final_dims.last().unwrap() as f64;
        final_dims.push(self.hidden_size);
        let order = xs
            .arg_sort_last_dim(true)?
            .to_dtype(candle_core::DType::F16)?;

        let ratio = (self.nvocab as f64) / last_dim_size;
        let order = (order * ratio)?
            .floor()?
            .to_dtype(candle_core::DType::U8)?
            .flatten_all()?;
        let values = self.embeddings.index_select(&order, 0)?;
        values.reshape(final_dims)
    }
}

pub fn rank_embedding(
    in_size: usize,
    out_size: usize,
    vb: candle_nn::VarBuilder,
) -> Result<RankEmbedding> {
    let embeddings = vb.get_with_hints(
        (in_size, out_size),
        "weight",
        candle_nn::Init::Uniform { lo: 0.0, up: 1.0 },
    )?;
    Ok(RankEmbedding {
        nvocab: in_size,
        embeddings,
        hidden_size: out_size,
    })
}

/////////////////////////
// Aggregate Embedding //
/////////////////////////

#[derive(Clone, Debug)]
pub struct AggregateEmbedding {
    weight_dh_k: Tensor,      // linearized d x (k*h)
    embeddings: Tensor,       // discretized value to real value
    log1p_embeddings: Tensor, // discretized value to real value
    n_feature: usize,
    n_module: usize,
    n_type: usize,
    n_vocab: usize,
}

impl AggregateEmbedding {
    pub fn membership(&self) -> Result<Tensor> {
        use candle_nn::ops::log_softmax;
        let ret_dhk = log_softmax(&self.weight_dh_k, self.weight_dh_k.rank() - 1)?.reshape(&[
            self.n_feature,
            self.n_type,
            self.n_module,
        ])?;
        ret_dhk.transpose(ret_dhk.rank() - 2, ret_dhk.rank() - 1)
    }

    fn whitened_discretize_flatten(&self, x_n_hk: &Tensor) -> Result<Tensor> {
        let d = x_n_hk.rank() - 1;
        let min_val = x_n_hk.min_keepdim(d)?;
        let max_val = x_n_hk.max_keepdim(d)?;
        let div_val = ((max_val - &min_val)? + 1.0)?;
        let x_n_hk = x_n_hk.broadcast_sub(&min_val)?.broadcast_div(&div_val)?;

        // discretize
        (x_n_hk.flatten_all()? * (self.n_vocab as f64 - 1.0))?
            .floor()?
            .to_dtype(candle_core::DType::U32)
    }
}

impl Module for AggregateEmbedding {
    /// Assume non-negative `x_nd` data
    fn forward(&self, x_nd: &Tensor) -> Result<Tensor> {
        use candle_nn::ops::log_softmax;
        let c_d_hk = log_softmax(&self.weight_dh_k, self.weight_dh_k.rank() - 1)?
            .reshape(&[self.n_feature, self.n_type * self.n_module])?;

        let x_n_hk = x_nd.matmul(&c_d_hk)?;
        let indexes = self.whitened_discretize_flatten(&x_n_hk)?;

        let log1p_x_n_hk = (x_n_hk + 1.0)?.log()?;
        let log1p_indexes = self.whitened_discretize_flatten(&log1p_x_n_hk)?;

        let ret = self
            .embeddings
            .index_select(&indexes, 0)?
            .add(&self.log1p_embeddings.index_select(&log1p_indexes, 0)?)?;

        let nbatch = x_nd.dim(0)?;
        let ret_nhk = ret.reshape(&[nbatch, self.n_type, self.n_module])?;
        let x_nkh = ret_nhk.transpose(ret_nhk.rank() - 2, ret_nhk.rank() - 1)?;
        Ok(x_nkh)
    }
}

/// aggregate `X[n,d]` into `Y[n, k, h] = X[n, d] * C[d, k, h]` where
/// `Σ_k C[d,k,h] = 1` to capture each feature's membership for each d
/// and h independently
pub fn aggregate_embedding(
    n_feature: usize,
    n_module: usize,
    n_type: usize,
    n_vocab: usize,
    vb: candle_nn::VarBuilder,
) -> Result<AggregateEmbedding> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let weight_dh_k = vb.get_with_hints((n_feature * n_type, n_module), "logits", init_ws)?;

    let embeddings = vb.get_with_hints(
        (n_vocab, 1),
        "embedding",
        candle_nn::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;

    let log1p_embeddings = vb.get_with_hints(
        (n_vocab, 1),
        "log1p.embedding",
        candle_nn::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;

    Ok(AggregateEmbedding {
        weight_dh_k,
        embeddings,
        log1p_embeddings,
        n_feature,
        n_module,
        n_type,
        n_vocab,
    })
}
