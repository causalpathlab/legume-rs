//! `VarMap` registration.
//!
//! Free functions rather than methods: the registration helpers run while a
//! [`JointEmbedModel`](super::JointEmbedModel) is still being built, so none of
//! them can borrow the model. Kept together as the layer between a host-side
//! init and a device Var.

use legume_numeric::candle::candle_core::{Device, Result, Tensor};
use legume_numeric::candle::candle_nn::VarMap;
use legume_numeric::matrix::rand_util::name_seed;
use legume_numeric::matrix::traits::SampleOps;

use super::INIT_STDEV;

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
    let var = legume_numeric::candle::candle_core::Var::from_tensor(&t)?;
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
    let var = legume_numeric::candle::candle_core::Var::from_slice(values, values.len(), dev)?;
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
    let var = legume_numeric::candle::candle_core::Var::from_tensor(&Tensor::from_vec(
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
