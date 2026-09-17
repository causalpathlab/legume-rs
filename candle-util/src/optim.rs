//! PBG's `RowAdagrad`: Adagrad with one accumulator per embedding ROW.
//!
//! ```text
//!   sum_r += mean_d(grad_r²)
//!   p_r   -= lr · grad_r / (sqrt(sum_r) + 1e-10)
//! ```
//!
//! The gradient candle hands back for an embedding table is dense (the
//! backward of a row gather is a zero table plus a row scatter-add), with
//! exact zeros on rows the batch never touched, so `mean_d(grad_r²) = 0`
//! there and the row is left bit-identical — the same outcome as PBG's
//! coalesced sparse path. A row gathered twice in one batch receives the SUM
//! of its contributions, as a coalesced sparse gradient would.
//!
//! [`RowAdagrad::step_with_bias`] is the same rule for a row that carries a
//! scalar bias alongside it: the accumulator sees the mean of `grad²` over the
//! row AND the bias, and both move by the same row step — what a caller would
//! get by concatenating the bias onto the row, without the copy. It also takes
//! the two things a pinned-row trainer needs: a `[n, 1]` gradient mask (a
//! masked row keeps its value while its bias trains, the accumulator seeing
//! the bias alone) and a per-step weight-decay factor applied to the rows the
//! step touched.

use candle_core::{DType, Device, Result, Tensor, Var};

/// PBG `RowAdagrad` denominator floor.
pub const ADAGRAD_EPS: f64 = 1e-10;

pub struct RowAdagrad {
    acc: Tensor,
    lr: f64,
    eps: f64,
    /// `[rows]`, the accumulator's divisor for a row-plus-bias step when no
    /// row is masked: `H + 1` everywhere. Built once, on the first such step.
    full_count: Option<Tensor>,
}

impl RowAdagrad {
    pub fn new(n_rows: usize, lr: f64, dev: &Device) -> Result<Self> {
        Ok(Self {
            acc: Tensor::zeros(n_rows, DType::F32, dev)?,
            lr,
            eps: ADAGRAD_EPS,
            full_count: None,
        })
    }

    /// `lr / (sqrt(acc) + eps)` per row after adding `g2` to the accumulator.
    fn advance(&mut self, g2: &Tensor) -> Result<Tensor> {
        // candle records autograd history on every op, including those that
        // produced the gradient; an accumulator chained through it would keep
        // every step's gradient storage alive for the whole run. Detach.
        self.acc = (&self.acc + g2)?.detach();
        self.acc
            .sqrt()?
            .affine(1.0 / self.lr, self.eps / self.lr)?
            .recip()
    }

    /// One update of `var` (`[rows, D]`) from its dense gradient.
    pub fn step(&mut self, var: &Var, grad: &Tensor) -> Result<()> {
        let row_sq = grad.sqr()?.sum(1)?;
        self.step_with_row_sq(var, grad, &row_sq)
    }

    /// [`Self::step`] for a caller that already holds `Σ_d grad²` per row
    /// (`[rows]`), so the square is taken once.
    pub fn step_with_row_sq(&mut self, var: &Var, grad: &Tensor, row_sq: &Tensor) -> Result<()> {
        let grad = grad.detach();
        let h = var.dims()[1] as f64;
        let step = self.advance(&row_sq.detach().affine(1.0 / h, 0.0)?)?;
        var.set(
            &var.as_tensor()
                .sub(&grad.broadcast_mul(&step.unsqueeze(1)?)?)?,
        )
    }

    /// One update of `(row, bias)` from their gradients (`[n, D]` and `[n]`).
    /// `row_mask` is `[n, 1]` with `0` on pinned rows; `decay` multiplies every
    /// row whose gradient is nonzero before its step (`1.0` for none).
    pub fn step_with_bias(
        &mut self,
        row: &Var,
        bias: &Var,
        g_row: &Tensor,
        g_bias: &Tensor,
        row_mask: Option<&Tensor>,
        decay: f64,
    ) -> Result<()> {
        let h = row.dims()[1] as f64;
        let g_row = match row_mask {
            None => g_row.detach(),
            Some(m) => g_row.detach().broadcast_mul(m)?,
        };
        let g_bias = g_bias.detach();
        let row_sq = g_row.sqr()?.sum(1)?; // [n]
                                           // Over the row and the bias where the row trains, the bias alone where
                                           // it is pinned.
        let count = match row_mask {
            Some(m) => m.squeeze(1)?.affine(h, 1.0)?,
            None => match self.full_count.as_ref() {
                Some(c) => c.clone(),
                None => {
                    let c = Tensor::full((h + 1.0) as f32, row_sq.dims()[0], row_sq.device())?;
                    self.full_count = Some(c.clone());
                    c
                }
            },
        };
        let g2 = (&row_sq + g_bias.sqr()?)?.div(&count)?;
        let step = self.advance(&g2)?;
        let mut new_row = row.as_tensor().clone();
        if decay != 1.0 {
            // `row *= decay` on touched rows only: a row the step never scored
            // keeps its value.
            let touched = row_sq.gt(0f32)?.to_dtype(DType::F32)?;
            let factor = touched.affine(decay - 1.0, 1.0)?; // 1 untouched, decay touched
            new_row = new_row.broadcast_mul(&factor.unsqueeze(1)?)?;
        }
        let new_row = new_row.sub(&g_row.broadcast_mul(&step.unsqueeze(1)?)?)?;
        row.set(&new_row)?;
        bias.set(&bias.as_tensor().sub(&(g_bias * step)?)?)
    }

    /// One update of a bias vector (`[rows]`) from its gradient: the
    /// accumulator takes `g²` itself, as [`Self::step_with_bias`] does for the
    /// bias of a pinned row.
    pub fn step_bias(&mut self, bias: &Var, g_bias: &Tensor) -> Result<()> {
        let g = g_bias.detach();
        let step = self.advance(&g.sqr()?)?;
        bias.set(&bias.as_tensor().sub(&(g * step)?)?)
    }

    /// Per-row `Σ mean_d(grad²)` so far.
    #[must_use]
    pub fn accumulator(&self) -> &Tensor {
        &self.acc
    }
}

#[cfg(test)]
#[path = "optim_tests.rs"]
mod optim_tests;
