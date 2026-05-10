//! Joint multiome embedding tables + bias terms + bilinear scoring.
//!
//! Two free embedding tables (`E_feat` over the unified feature axis,
//! `E_cell`) plus two bias vectors (`b_feat`, `b_cell`). Score for a
//! `(feature, cell)` edge under a Poisson rate model:
//!
//!   `score(f, c) = E_feat[f] · E_cell[c] + b_feat[f] + b_cell[c]`
//!
//! Features are addressed at fine resolution. The cell axis is
//! coarsened: cell embeddings are mean-pooled (per the batch's chosen
//! seed coarsening) over the fine children of each touched super-cell.

use candle_util::candle_core::{DType, Device, Result, Tensor, Var};
use candle_util::candle_nn::{self, VarBuilder, VarMap};

/// Shape of the embedding tables.
pub struct ModelArgs {
    pub n_features: usize,
    pub n_cells: usize,
    pub embedding_dim: usize,
}

/// Initial values for [`JointEmbedModel::new_with_init`]. `None` for
/// either embedding falls back to randn; bias slices must be
/// dimensionally consistent with [`ModelArgs`].
pub struct ModelInit<'a> {
    pub e_feat: Option<&'a nalgebra::DMatrix<f32>>,
    pub e_cell: Option<&'a nalgebra::DMatrix<f32>>,
    pub b_feat: &'a [f32],
    pub b_cell: &'a [f32],
}

/// Inputs for [`JointEmbedModel::new_sharing_features`]. The feature
/// side (`e_feat` / `b_feat`) is provided pre-allocated and registered
/// in the shared `VarMap` so multiple heads can co-train it. Only the
/// cell side gets new Vars, namespaced by `var_prefix` so multiple
/// heads can coexist in one VarMap (e.g. `pb_l0`, `pb_l1`, ..., `cell`).
pub struct ShareFeaturesArgs<'a> {
    pub n_cells: usize,
    pub embedding_dim: usize,
    pub shared_e_feat: Tensor,
    pub shared_b_feat: Tensor,
    pub e_cell_init: Option<&'a nalgebra::DMatrix<f32>>,
    pub b_cell_init: &'a [f32],
    pub var_prefix: &'a str,
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
    /// Construct with optional warm-start values for either embedding.
    /// Used by stage 1 across the multi-level curriculum so each level
    /// inherits `E_feat` from the previous level instead of restarting
    /// from randn.
    pub fn new_with_init(
        args: ModelArgs,
        init: &ModelInit,
        varmap: &VarMap,
        vs: VarBuilder,
        dev: &Device,
    ) -> Result<Self> {
        let randn = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        };
        let e_feat = match init.e_feat {
            Some(m) => register_var_from_mat(varmap, dev, "e_feat", m)?,
            None => vs.get_with_hints((args.n_features, args.embedding_dim), "e_feat", randn)?,
        };
        let e_cell = match init.e_cell {
            Some(m) => register_var_from_mat(varmap, dev, "e_cell", m)?,
            None => vs.get_with_hints((args.n_cells, args.embedding_dim), "e_cell", randn)?,
        };
        let b_feat = register_var_from_slice(varmap, dev, "b_feat", init.b_feat)?;
        let b_cell = register_var_from_slice(varmap, dev, "b_cell", init.b_cell)?;
        Ok(Self {
            e_feat,
            e_cell,
            b_feat,
            b_cell,
            embedding_dim: args.embedding_dim,
        })
    }

    /// Composite-training constructor: reuse pre-existing
    /// `shared_e_feat` / `shared_b_feat` Tensors (already registered as
    /// Vars in `varmap` by an earlier call to `new_with_init`) and
    /// allocate fresh cell-side Vars under `args.var_prefix` so multiple
    /// heads coexist in one VarMap. AdamW over `varmap.all_vars()` then
    /// updates the shared feature side once and each head's cell side
    /// independently.
    pub fn new_sharing_features(
        args: ShareFeaturesArgs,
        varmap: &VarMap,
        dev: &Device,
    ) -> Result<Self> {
        let ShareFeaturesArgs {
            n_cells,
            embedding_dim,
            shared_e_feat,
            shared_b_feat,
            e_cell_init,
            b_cell_init,
            var_prefix,
        } = args;
        let e_name = format!("{}_e_cell", var_prefix);
        let b_name = format!("{}_b_cell", var_prefix);
        let e_cell = match e_cell_init {
            Some(m) => register_var_from_mat(varmap, dev, &e_name, m)?,
            None => {
                let randn = Tensor::randn(0.0_f32, 0.1, (n_cells, embedding_dim), dev)?;
                let var = Var::from_tensor(&randn)?;
                varmap.data().lock().unwrap().insert(e_name, var.clone());
                var.as_tensor().clone()
            }
        };
        let b_cell = register_var_from_slice(varmap, dev, &b_name, b_cell_init)?;
        Ok(Self {
            e_feat: shared_e_feat,
            e_cell,
            b_feat: shared_b_feat,
            b_cell,
            embedding_dim,
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

/// Register a 2D learnable parameter initialized from a host matrix
/// (row-major flatten). `nalgebra::DMatrix` is column-major, so we
/// emit row-by-row; the resulting tensor matches candle's `[rows, cols]`
/// row-major layout.
fn register_var_from_mat(
    varmap: &VarMap,
    dev: &Device,
    name: &str,
    mat: &nalgebra::DMatrix<f32>,
) -> Result<Tensor> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    let mut row_major = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            row_major.push(mat[(i, j)]);
        }
    }
    let var = candle_util::candle_core::Var::from_tensor(&Tensor::from_vec(
        row_major,
        (rows, cols),
        dev,
    )?)?;
    {
        let mut data = varmap.data().lock().unwrap();
        data.insert(name.to_string(), var.clone());
    }
    Ok(var.as_tensor().clone())
}

/// Mean-pool `[D, H]` table over the fine children of `coarse_blocks`.
/// Returns `(pooled_emb [n_blocks, H], pooled_bias [n_blocks])`. Both
/// outputs stay in the autograd graph.
///
/// Two ops total in the autograd path: one flat `index_select` gathers
/// every fine row in block order, then `index_add` scatters them into
/// per-block sums in a `[n_blocks, H]` accumulator. Empty blocks get
/// `count = 1` so the all-zero accumulator divides cleanly to zero
/// (matching the loop's zero-pad behavior).
fn pool_axis(
    table: &Tensor,
    bias: &Tensor,
    coarse_blocks: &[u32],
    coarse_to_fine: &[Vec<usize>],
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let h = table.dim(1)?;
    let n_blocks = coarse_blocks.len();

    let total_fine: usize = coarse_blocks
        .iter()
        .map(|&b| coarse_to_fine[b as usize].len())
        .sum();
    let mut flat_fine: Vec<u32> = Vec::with_capacity(total_fine);
    let mut owner: Vec<u32> = Vec::with_capacity(total_fine);
    let mut counts: Vec<f32> = Vec::with_capacity(n_blocks);
    for (b_idx, &block) in coarse_blocks.iter().enumerate() {
        let fine = &coarse_to_fine[block as usize];
        for &f in fine {
            flat_fine.push(f as u32);
            owner.push(b_idx as u32);
        }
        counts.push(fine.len().max(1) as f32);
    }

    if total_fine == 0 {
        // No fine rows at all — every block is empty. Return zeros directly.
        let emb_zeros = Tensor::zeros((n_blocks, h), table.dtype(), dev)?;
        let bias_zeros = Tensor::zeros(n_blocks, bias.dtype(), dev)?;
        return Ok((emb_zeros, bias_zeros));
    }

    let flat_fine_t = Tensor::from_vec(flat_fine, total_fine, dev)?;
    let owner_t = Tensor::from_vec(owner, total_fine, dev)?;
    let counts_2d = Tensor::from_vec(counts.clone(), (n_blocks, 1), dev)?;
    let counts_1d = Tensor::from_vec(counts, n_blocks, dev)?;

    let gathered_emb = table.index_select(&flat_fine_t, 0)?; // [n_fine, H]
    let zeros_emb = Tensor::zeros((n_blocks, h), table.dtype(), dev)?;
    let summed_emb = zeros_emb.index_add(&owner_t, &gathered_emb, 0)?; // [n_blocks, H]
    let pooled_emb = summed_emb.broadcast_div(&counts_2d)?;

    let gathered_bias = bias.index_select(&flat_fine_t, 0)?; // [n_fine]
    let zeros_bias = Tensor::zeros(n_blocks, bias.dtype(), dev)?;
    let summed_bias = zeros_bias.index_add(&owner_t, &gathered_bias, 0)?; // [n_blocks]
    let pooled_bias = (summed_bias / counts_1d)?;

    Ok((pooled_emb, pooled_bias))
}

/// Reference implementation kept for the parity test only — see
/// `tests::pool_axis_index_add_matches_loop`. Identical semantics to
/// the previous version of [`pool_axis`].
#[cfg(test)]
fn pool_axis_loop(
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
        let gathered = table.index_select(&idx_t, 0)?;
        let pooled = gathered.mean(0)?;
        emb_rows.push(pooled);

        let bias_g = bias.index_select(&idx_t, 0)?;
        let mean_b = bias_g.mean(0)?;
        bias_rows.push(mean_b);
    }

    let emb_stack = Tensor::stack(&emb_rows, 0)?;
    let bias_stack = Tensor::stack(&bias_rows, 0)?;
    Ok((emb_stack, bias_stack))
}

#[allow(unused)]
fn dummy_dtype_check() -> DType {
    DType::F32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::Cpu
    }

    #[test]
    fn pool_axis_index_add_matches_loop() {
        // 8 fine rows × H=3, grouped into 4 coarse blocks (incl. one empty).
        let dev = dev();
        let table =
            Tensor::from_vec((0..24).map(|x| x as f32).collect::<Vec<_>>(), (8, 3), &dev).unwrap();
        let bias = Tensor::from_vec(
            (0..8).map(|x| (x as f32) * 0.1).collect::<Vec<_>>(),
            8,
            &dev,
        )
        .unwrap();

        let coarse_to_fine = vec![
            vec![0, 1, 2],    // block 0
            vec![3],          // block 1
            vec![],           // block 2 (empty)
            vec![4, 5, 6, 7], // block 3
        ];
        let blocks = vec![3u32, 0, 2, 1, 0]; // mixed order, repeats allowed

        let (emb_new, bias_new) = pool_axis(&table, &bias, &blocks, &coarse_to_fine, &dev).unwrap();
        let (emb_ref, bias_ref) =
            pool_axis_loop(&table, &bias, &blocks, &coarse_to_fine, &dev).unwrap();

        let emb_n: Vec<f32> = emb_new.flatten_all().unwrap().to_vec1().unwrap();
        let emb_r: Vec<f32> = emb_ref.flatten_all().unwrap().to_vec1().unwrap();
        let bias_n: Vec<f32> = bias_new.flatten_all().unwrap().to_vec1().unwrap();
        let bias_r: Vec<f32> = bias_ref.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(emb_n.len(), emb_r.len());
        assert_eq!(bias_n.len(), bias_r.len());
        for (a, b) in emb_n.iter().zip(emb_r.iter()) {
            assert!((a - b).abs() < 1e-5, "emb mismatch: {a} vs {b}");
        }
        for (a, b) in bias_n.iter().zip(bias_r.iter()) {
            assert!((a - b).abs() < 1e-5, "bias mismatch: {a} vs {b}");
        }
    }
}
