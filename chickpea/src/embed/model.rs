//! Joint multiome embedding tables + bias terms + bilinear scoring.
//!
//! Two free embedding tables (`E_feat` over genes ∪ peaks, `E_cell`)
//! plus two bias vectors (`b_feat`, `b_cell`). Score for a
//! `(feature, cell)` edge under a Poisson rate model:
//!
//!   `score(f, c) = E_feat[f] · E_cell[c] + b_feat[f] + b_cell[c]`
//!
//! Features are addressed at fine resolution. The cell axis is
//! coarsened: cell embeddings are mean-pooled (per the batch's chosen
//! seed coarsening) over the fine children of each touched super-cell.

use candle_util::candle_core::{DType, Device, Result, Tensor};
use candle_util::candle_nn::{self, VarBuilder, VarMap};

/// Hyperparameters for `JointEmbedModel::new`.
pub struct ModelArgs {
    /// Total feature cardinality (unified across all input files).
    pub n_features: usize,
    pub n_cells: usize,
    pub embedding_dim: usize,
}

/// Initialization vectors for bias terms.
pub struct BiasInit<'a> {
    pub b_feat: &'a [f32], // length n_features
    pub b_cell: &'a [f32], // length n_cells
}

pub struct JointEmbedModel {
    /// Unified feature embedding (genes ∪ peaks).
    pub e_feat: Tensor,
    pub e_cell: Tensor,
    pub b_feat: Tensor,
    pub b_cell: Tensor,
    #[allow(dead_code)]
    pub embedding_dim: usize,
}

impl JointEmbedModel {
    pub fn new(
        args: ModelArgs,
        bias_init: &BiasInit,
        varmap: &VarMap,
        vs: VarBuilder,
        dev: &Device,
    ) -> Result<Self> {
        let init = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        };
        let e_feat = vs.get_with_hints((args.n_features, args.embedding_dim), "e_feat", init)?;
        let e_cell = vs.get_with_hints((args.n_cells, args.embedding_dim), "e_cell", init)?;

        let b_feat = register_var_from_slice(varmap, dev, "b_feat", bias_init.b_feat)?;
        let b_cell = register_var_from_slice(varmap, dev, "b_cell", bias_init.b_cell)?;

        Ok(Self {
            e_feat,
            e_cell,
            b_feat,
            b_cell,
            embedding_dim: args.embedding_dim,
        })
    }

    /// Mean-pool the cell embedding table over the fine children of a
    /// list of coarse-block indices. Output `[n_blocks, H]` plus a
    /// matching `[n_blocks]` bias vector.
    pub fn pool_cells(
        &self,
        coarse_blocks: &[u32],
        coarse_to_fine: &[Vec<usize>],
        dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        pool_axis(
            &self.e_cell,
            &self.b_cell,
            coarse_blocks,
            coarse_to_fine,
            dev,
        )
    }

    /// Bilinear score with bias terms.
    ///
    /// `e_f`: `[B, H]` pooled feature embeddings (one row per positive's
    /// feature block).
    /// `e_c`: `[B, H]` pooled cell embeddings (one row per positive's
    /// cell block).
    /// `b_f`, `b_c`: `[B]` bias scalars per row.
    /// Returns `[B]` scores.
    pub fn score_diag(e_f: &Tensor, e_c: &Tensor, b_f: &Tensor, b_c: &Tensor) -> Result<Tensor> {
        let dot = (e_f * e_c)?.sum(1)?;
        (dot + b_f)? + b_c
    }

    /// Bilinear score for negatives: score positive cells against
    /// alternative feature blocks. `e_f_neg`: `[B, K, H]`,
    /// `e_c`: `[B, H]`, `b_f_neg`: `[B, K]`, `b_c`: `[B]`. Returns `[B, K]`.
    pub fn score_negatives(
        e_f_neg: &Tensor,
        e_c: &Tensor,
        b_f_neg: &Tensor,
        b_c: &Tensor,
    ) -> Result<Tensor> {
        let b = e_f_neg.dim(0)?;
        let k = e_f_neg.dim(1)?;
        let h = e_f_neg.dim(2)?;
        let e_c_expanded = e_c.unsqueeze(1)?.broadcast_as((b, k, h))?;
        let dot = (e_f_neg * e_c_expanded)?.sum(2)?;
        let b_c_b = b_c.unsqueeze(1)?.broadcast_as((b, k))?;
        (dot + b_f_neg)? + b_c_b
    }
}

/// Register a 1D learnable parameter initialized from a slice and
/// return the underlying tensor (kept in autograd via `VarMap`).
fn register_var_from_slice(
    varmap: &VarMap,
    dev: &Device,
    name: &str,
    values: &[f32],
) -> Result<Tensor> {
    let var = candle_util::candle_core::Var::from_slice(values, values.len(), dev)?;
    {
        let mut data = varmap.data().lock().unwrap();
        data.insert(name.to_string(), var.clone());
    }
    Ok(var.as_tensor().clone())
}

/// Mean-pool `[D, H]` table over the fine children of `coarse_blocks`.
/// Returns `(pooled_emb [n_blocks, H], pooled_bias [n_blocks])`. Both
/// outputs stay in the autograd graph.
fn pool_axis(
    table: &Tensor,
    bias: &Tensor,
    coarse_blocks: &[u32],
    coarse_to_fine: &[Vec<usize>],
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let h = table.dim(1)?;
    let mut emb_rows: Vec<Tensor> = Vec::with_capacity(coarse_blocks.len());
    let mut bias_rows: Vec<Tensor> = Vec::with_capacity(coarse_blocks.len());

    for &block in coarse_blocks {
        let fine = &coarse_to_fine[block as usize];
        if fine.is_empty() {
            emb_rows.push(Tensor::zeros((h,), table.dtype(), dev)?);
            bias_rows.push(Tensor::zeros((), bias.dtype(), dev)?);
            continue;
        }
        let idx: Vec<u32> = fine.iter().map(|&i| i as u32).collect();
        let idx_t = Tensor::from_vec(idx, fine.len(), dev)?;
        let gathered = table.index_select(&idx_t, 0)?; // [n_fine, H]
        let pooled = gathered.mean(0)?; // [H]
        emb_rows.push(pooled);

        let bias_g = bias.index_select(&idx_t, 0)?; // [n_fine]
        let mean_b = bias_g.mean(0)?; // scalar tensor (0-dim) — stays in graph
        bias_rows.push(mean_b);
    }

    let emb_stack = Tensor::stack(&emb_rows, 0)?; // [n_blocks, H]
    let bias_stack = Tensor::stack(&bias_rows, 0)?; // [n_blocks]
    Ok((emb_stack, bias_stack))
}

#[allow(unused)]
fn dummy_dtype_check() -> DType {
    DType::F32
}
