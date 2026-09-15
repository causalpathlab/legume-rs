//! A pair code from two cell codes: a gated mixture of linear experts over
//! symmetric pair features.
//!
//! An unordered pair has no first member, so the head reads only features
//! that do not care which code came first — the mean of the two codes and
//! their product — and every expert is one linear map of those:
//!
//! ```text
//! m = (h_u + h_v) / 2            [B, L]   what the two share
//! p = h_u ⊙ h_v                  [B, L]   where they agree, and how strongly
//! f = [m ‖ p ‖ x]                [B, 2L + E]   x: the caller's own symmetric pair features
//! π = softmax(W_g f + c_g)       [B, K]   which expert reads this pair
//! z = Σ_k π_k (W_k f + c_k)      [B, D]
//! ```
//!
//! `K = 1` is one linear map of `f` and registers no gate. With more experts
//! the gate can route a pair whose codes agree (large `p` against `m²`)
//! differently from one straddling two programmes, which is where a single
//! linear map has to compromise. `x` is for whatever the caller can compute
//! exactly for the pair — a sufficient statistic of its likelihood, say —
//! and is the caller's to keep symmetric.

use candle_core::{Result, Tensor};
use candle_nn::{linear, ops, Linear, Module, VarBuilder};

pub struct SymmetricPairHeadArgs {
    /// Width `L` of each cell code.
    pub code_dim: usize,
    /// Width `D` of the pair code.
    pub out_dim: usize,
    /// Experts `K`; `1` is a plain linear head.
    pub n_experts: usize,
    /// Width `E` of the caller's own pair features; `0` for none.
    pub extra_dim: usize,
}

pub struct SymmetricPairHead {
    /// `2L + E → K`; absent for a single expert.
    gate: Option<Linear>,
    /// `2L + E → K·D`, every expert's map in one matrix.
    experts: Linear,
    code_dim: usize,
    out_dim: usize,
    n_experts: usize,
    extra_dim: usize,
}

impl SymmetricPairHead {
    /// Register `experts.{weight,bias}` and, for `n_experts > 1`,
    /// `gate.{weight,bias}` under `vb`.
    pub fn new(args: SymmetricPairHeadArgs, vb: VarBuilder) -> Result<Self> {
        let k = args.n_experts.max(1);
        let in_dim = 2 * args.code_dim + args.extra_dim;
        let gate = if k > 1 {
            Some(linear(in_dim, k, vb.pp("gate"))?)
        } else {
            None
        };
        let experts = linear(in_dim, k * args.out_dim, vb.pp("experts"))?;
        Ok(Self {
            gate,
            experts,
            code_dim: args.code_dim,
            out_dim: args.out_dim,
            n_experts: k,
            extra_dim: args.extra_dim,
        })
    }

    pub fn code_dim(&self) -> usize {
        self.code_dim
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn extra_dim(&self) -> usize {
        self.extra_dim
    }

    /// The pair features `[m ‖ p ‖ x]` → `[B, 2L + E]`; `extra` is required
    /// exactly when the head was built with `extra_dim > 0`.
    pub fn features(&self, h_u: &Tensor, h_v: &Tensor, extra: Option<&Tensor>) -> Result<Tensor> {
        let m = ((h_u + h_v)? * 0.5)?;
        let p = (h_u * h_v)?;
        match (extra, self.extra_dim) {
            (None, 0) => Tensor::cat(&[&m, &p], 1),
            (Some(x), e) if e > 0 => Tensor::cat(&[&m, &p, x], 1),
            (Some(_), _) => {
                candle_core::bail!("pair head: extra features given to a head built without them")
            }
            (None, e) => {
                candle_core::bail!("pair head: built with {e} extra features but none were given")
            }
        }
    }

    /// Expert weights per pair → `[B, K]`, rows summing to one.
    pub fn gate(&self, features: &Tensor) -> Result<Tensor> {
        match &self.gate {
            Some(g) => ops::softmax(&g.forward(features)?, 1),
            None => Tensor::ones((features.dim(0)?, 1), features.dtype(), features.device()),
        }
    }

    /// The pair code → `[B, D]`.
    pub fn forward(&self, h_u: &Tensor, h_v: &Tensor, extra: Option<&Tensor>) -> Result<Tensor> {
        let f = self.features(h_u, h_v, extra)?;
        let b = f.dim(0)?;
        let e = self
            .experts
            .forward(&f)?
            .reshape((b, self.n_experts, self.out_dim))?; // [B, K, D]
        if self.n_experts == 1 {
            return e.squeeze(1);
        }
        let pi = self.gate(&f)?.unsqueeze(2)?; // [B, K, 1]
        e.broadcast_mul(&pi)?.sum(1)
    }
}

#[cfg(test)]
#[path = "pair_head_tests.rs"]
mod tests;
