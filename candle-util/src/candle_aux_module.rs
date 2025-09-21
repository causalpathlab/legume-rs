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
