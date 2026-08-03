//! `VarMap` registration and the coarse-block pooling kernels.
//!
//! Free functions rather than methods: the registration helpers run while a
//! [`JointEmbedModel`](super::JointEmbedModel) is still being built, and
//! [`pool_axis`] takes a bare `[D, H]` table, so none of them can borrow the
//! model. Kept together as the layer between a host-side init and a device Var.

use candle_util::candle_core::{Device, Result, Tensor};
use candle_util::candle_nn::VarMap;
use matrix_util::rand_util::name_seed;
use matrix_util::traits::SampleOps;

use super::{FeatFactor, INIT_STDEV};

/// Build a [`FeatFactor`] from the shared `β` Var and the host-side row→gene
/// map: materializes the fixed `row_to_gene` (u32) index tensor on `dev`.
pub(super) fn build_feat_factor(
    beta: &Tensor,
    row_to_gene: &[u32],
    delta: Option<Tensor>,
    unspliced_rows: Option<&[bool]>,
    dev: &Device,
) -> Result<FeatFactor> {
    let d = row_to_gene.len();
    let row_to_gene_t = Tensor::from_vec(row_to_gene.to_vec(), d, dev)?;
    // `δ_g` and its `[n_features, 1]` 0/1 unspliced-row mask co-exist: both are built
    // exactly when a `δ_g` Var was allocated (`delta_l2 > 0`).
    let splice_delta = match (delta, unspliced_rows) {
        (Some(delta), Some(u)) => {
            let m: Vec<f32> = u.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            Some((delta, Tensor::from_vec(m, (d, 1), dev)?))
        }
        _ => None,
    };
    Ok(FeatFactor {
        beta: beta.clone(),
        row_to_gene: row_to_gene_t,
        splice_delta,
        s_beta: None,
        beta_logstd: None,
        s_delta: None,
        delta_logstd: None,
        delta_gate_pip: None,
        delta_gate_mask: None,
    })
}

/// Register a `[rows, cols]` learnable parameter initialized with **seeded,
/// reproducible** `N(0, INIT_STDEV)` values, and return the underlying tensor.
///
/// Replaces `vs.get_with_hints(..., Init::Randn)` / `Tensor::randn`: candle's
/// device randn is unseedable on the CPU backend (`Device::set_seed` errors
/// out, `rand_normal` reads OS entropy), so identical-config runs would
/// otherwise draw a fresh init every time. The seeded `Tensor` sampler draws
/// it host-side instead, keyed by `name` so each table (`e_feat`, `e_cell`,
/// `beta`, per-level `{prefix}_e_cell`) gets an independent stream off one
/// `base_seed` with no hand-assigned salts.
pub(super) fn register_randn_seeded(
    varmap: &VarMap,
    dev: &Device,
    name: &str,
    rows: usize,
    cols: usize,
    base_seed: u64,
) -> Result<Tensor> {
    // `rnorm_seeded` (a host `from_vec`) and `affine` are both contiguous, and
    // `to_device` preserves that; the explicit `contiguous()` is a cheap no-op
    // guard so the registered Var is always contiguous for CUDA matmul kernels.
    let t = Tensor::rnorm_seeded(rows, cols, name_seed(base_seed, name))
        .affine(INIT_STDEV as f64, 0.0)?
        .to_device(dev)?
        .contiguous()?;
    let var = candle_util::candle_core::Var::from_tensor(&t)?;
    varmap
        .data()
        .lock()
        .unwrap()
        .insert(name.to_string(), var.clone());
    Ok(var.as_tensor().clone())
}

/// Register a 1D learnable parameter initialized from a slice and
/// return the underlying tensor (kept in autograd via `VarMap`).
pub(super) fn register_var_from_slice(
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
pub(super) fn register_var_from_mat(
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
pub(super) fn pool_axis(
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
pub(super) fn pool_axis_loop(
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
