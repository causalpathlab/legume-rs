//! LoRA on a pinned table: `ρ = base + u · v`.
//!
//! `base` is `[N, H]` and never trained; `u` is `[N, rank]`, one small row per
//! table row, and `v` is `[rank, H]`, shared by every row. `rank` is what
//! makes it a residual rather than a second table: at rank 0 the base is
//! frozen, at rank H nothing is pinned, and in between the rows can only move
//! together, inside the `rank`-dimensional subspace `v` spans.
//!
//! Initialization is LoRA's (Hu et al., 2021): `u ~ N(0, 1/rank)` and `v = 0`,
//! so the residual is exactly nothing at step 0 and the first gradients reach
//! `v`. Training is LoRA+'s (Hayou, Ghosh & Yu, ICML 2024): `v`, which starts
//! at zero and is touched by every step, takes a learning rate `lr_ratio`
//! times the base rate, while the row factor `u` keeps it. A ratio of 1 is
//! plain LoRA.
//!
//! Every read here composes without forming the `[N, H]` residual: rows are
//! gathered from `u` and multiplied by `v`, and a right factor is applied to
//! `v` first. The one dense product is [`residual`], for folding the factors
//! back into the base when training ends ([`fold`]), after which the table is
//! a plain one and no reader needs to know LoRA was involved.

use crate::fast_index::gather_rows;
use candle_core::{Result, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};

/// Registered names of the factors, relative to the table's prefix.
pub const U_VAR_NAME: &str = "lora_u";
pub const V_VAR_NAME: &str = "lora_v";

/// The trained factors of one table.
pub struct LoraFactors {
    /// `[N, rank]`.
    pub u: Tensor,
    /// `[rank, H]`.
    pub v: Tensor,
}

impl LoraFactors {
    /// Register `u` (`N(0, 1/rank)`) and `v` (zero) under `vs`.
    pub fn new(n_rows: usize, dim: usize, rank: usize, vs: VarBuilder) -> Result<Self> {
        if rank == 0 {
            candle_core::bail!("a LoRA residual needs rank ≥ 1");
        }
        Ok(Self {
            u: vs.get_with_hints(
                (n_rows, rank),
                U_VAR_NAME,
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: (1.0 / rank as f64).sqrt(),
                },
            )?,
            v: vs.get_with_hints((rank, dim), V_VAR_NAME, candle_nn::Init::Const(0.0))?,
        })
    }

    /// Factors a caller already holds (the PBG engine keeps its own `Var`s).
    #[must_use]
    pub fn from_parts(u: Tensor, v: Tensor) -> Self {
        Self { u, v }
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.v.dims()[0]
    }

    /// The residual on the rows named by `ids`: `u[ids] · v`, `[ids, H]`.
    pub fn residual_rows(&self, ids: &Tensor) -> Result<Tensor> {
        gather_rows(&self.u, ids)?.matmul(&self.v)
    }

    /// The residual projected through a `[H, C]` right factor without forming
    /// it: `u · (v · v_hc)`.
    pub fn project_dims(&self, v_hc: &Tensor) -> Result<Tensor> {
        self.u.matmul(&self.v.matmul(v_hc)?)
    }

    /// The whole `[N, H]` residual. For output and folding only.
    pub fn residual(&self) -> Result<Tensor> {
        self.u.matmul(&self.v)
    }
}

/// The LoRA+ learning-rate split, for a trainer that builds its optimizers
/// from a `VarMap`: the named `v` leaves the main group and gets its own.
#[derive(Clone, Copy, Debug)]
pub struct LoraPlus<'a> {
    /// The full name of `v` in the map (`"{prefix}.lora_v"`).
    pub v_var: &'a str,
    pub lr_ratio: f32,
}

impl LoraPlus<'_> {
    /// The AdamW group for `v` at `lr_ratio × lr`, no weight decay (the
    /// residual is shrunk by its rank, not by decay). The caller keeps `v_var`
    /// out of its main group.
    pub fn optimizer(&self, varmap: &VarMap, lr: f32) -> anyhow::Result<AdamW> {
        let v = crate::frozen_features::trainable_only(varmap, &[self.v_var]);
        anyhow::ensure!(
            v.len() == 1,
            "LoRA+ names `{}` but the model has no such Var",
            self.v_var
        );
        Ok(AdamW::new(
            v,
            candle_nn::ParamsAdamW {
                lr: f64::from(lr * self.lr_ratio),
                weight_decay: 0.0,
                ..Default::default()
            },
        )?)
    }
}

/// The factor names under a table prefix: `("{prefix}.lora_u", "{prefix}.lora_v")`.
#[must_use]
pub fn factor_names(prefix: &str) -> (String, String) {
    let name = |slot: &str| {
        if prefix.is_empty() {
            slot.to_string()
        } else {
            format!("{prefix}.{slot}")
        }
    };
    (name(U_VAR_NAME), name(V_VAR_NAME))
}

/// Fold the residual into the base and drop the factors from the map: the
/// `Var` at `base_name` becomes `base + u · v`, and `{prefix}.lora_u` /
/// `{prefix}.lora_v` leave. After this the checkpoint is a plain table. A map
/// without the factors (no LoRA was used) is left alone.
pub fn fold(varmap: &VarMap, base_name: &str, prefix: &str) -> Result<()> {
    let (u_name, v_name) = factor_names(prefix);
    let mut tbl = varmap.data().lock().unwrap();
    let (Some(u), Some(v)) = (tbl.remove(&u_name), tbl.remove(&v_name)) else {
        return Ok(());
    };
    let base = tbl.get(base_name).ok_or_else(|| {
        candle_core::Error::Msg(format!("no {base_name} to fold the LoRA residual into"))
    })?;
    let factors = LoraFactors::from_parts(u.as_tensor().clone(), v.as_tensor().clone());
    let folded = (base.as_tensor() + factors.residual()?)?;
    base.set(&folded.detach())
}

#[cfg(test)]
#[path = "lora_tests.rs"]
mod lora_tests;
