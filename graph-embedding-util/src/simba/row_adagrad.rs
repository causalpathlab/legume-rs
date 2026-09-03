//! PBG's `RowAdagrad`: Adagrad with one accumulator per embedding ROW.
//!
//! ```text
//!   sum_r += mean_d(grad_r²)
//!   p_r   -= lr · grad_r / (sqrt(sum_r) + 1e-10)
//! ```
//!
//! The gradient candle hands back for an embedding table is dense (the
//! backward of `index_select` is a zero table plus `index_add`), with exact
//! zeros on rows the batch never touched, so `mean_d(grad_r²) = 0` there and
//! the row is left bit-identical — the same outcome as PBG's coalesced sparse
//! path. A row gathered twice in one batch receives the SUM of its
//! contributions, as a coalesced sparse gradient would.

use super::ADAGRAD_EPS;
use candle_util::candle_core::{DType, Device, Result, Tensor, Var};

pub struct RowAdagrad {
    acc: Tensor,
    lr: f64,
    eps: f64,
}

impl RowAdagrad {
    pub fn new(n_rows: usize, lr: f64, dev: &Device) -> Result<Self> {
        Ok(Self {
            acc: Tensor::zeros(n_rows, DType::F32, dev)?,
            lr,
            eps: ADAGRAD_EPS,
        })
    }

    /// One update of `var` (`[rows, D]`) from its dense gradient.
    pub fn step(&mut self, var: &Var, grad: &Tensor) -> Result<()> {
        // candle records autograd history on every op, including those that
        // produced `grad`; an accumulator chained through it would keep every
        // step's gradient storage alive for the whole run. Detach at both ends.
        let grad = grad.detach();
        let g2 = grad.sqr()?.mean(1)?; // [rows]
        self.acc = (&self.acc + g2)?.detach();
        // (sqrt(sum) + eps) / lr in one pass, then a broadcast divide.
        let denom = self.acc.sqrt()?.affine(1.0 / self.lr, self.eps / self.lr)?;
        let upd = grad.broadcast_div(&denom.unsqueeze(1)?)?;
        // `Var::set` needs fresh storage; the subtraction allocates it.
        var.set(&var.as_tensor().sub(&upd)?)
    }

    /// Per-row `Σ mean_d(grad²)` so far.
    #[must_use]
    pub fn accumulator(&self) -> &Tensor {
        &self.acc
    }
}

#[cfg(test)]
#[path = "row_adagrad_tests.rs"]
mod row_adagrad_tests;
