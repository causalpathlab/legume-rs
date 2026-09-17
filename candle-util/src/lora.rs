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
use crate::optim::RowAdagrad;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use matrix_util::rand_util::{mix_seed, normal_f32_seeded};

/// The salt the row factor of a pinned table is drawn under, mixed with the
/// run's seed.
const ROW_FACTOR_SALT: u64 = 0x4c4f_5241;

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

    /// `‖u·v‖²_F = Σ_rows ‖u_g·v‖²`, the residual's summed row norm², without
    /// forming the residual: `‖u·v‖²_F = Σ (uᵀu) ⊙ (v vᵀ)` on two `[rank, rank]`
    /// Grams. A sum, not a mean, on purpose: the data gradient on the shared
    /// factor `v` is itself a sum over the rows, so a ridge weight per row is
    /// the one that means the same thing whatever the table's size. Rows of
    /// `u` that are zero contribute nothing. The ridge every engine adds.
    pub fn ridge(&self) -> Result<Tensor> {
        let uu = self.u.t()?.matmul(&self.u)?;
        let vv = self.v.matmul(&self.v.t()?)?;
        (uu * vv)?.sum_all()
    }
}

/// A residual on a table SOME of whose rows are pinned: `u` is drawn on the
/// pinned rows only, `v` is shared and starts at zero, and a free row has a
/// zero `u` row that never moves. Two trainers use it: the row-optimizer
/// engines (the hierarchical phase of bge, the PBG engine of fne and simba)
/// step it with [`Self::step`], which masks `u`'s gradient by hand; a trainer
/// whose optimizer sees every row (AdamW over a `VarMap`) composes through
/// [`Self::residual_rows_masked`], where the mask sits in the graph, so a free
/// row's factor gets an exact-zero gradient and stays at its zero init.
pub struct PinnedLora {
    pub u: Var,
    pub v: Var,
    /// `[N, 1]`, `1` on the pinned rows.
    pub u_mask: Tensor,
    /// How many rows are pinned.
    pub n_pinned: usize,
}

impl PinnedLora {
    /// `u ~ N(0, 1/rank)` on `pinned` (ids into the `N` rows), zero elsewhere;
    /// `v = 0`. The draw is seeded from `seed` under this module's salt; a
    /// second residual of the same run passes a seed mixed with its own salt.
    pub fn new(
        n_rows: usize,
        dim: usize,
        rank: usize,
        pinned: &[u32],
        seed: u64,
        dev: &Device,
    ) -> Result<Self> {
        if rank == 0 {
            candle_core::bail!("a LoRA residual needs rank ≥ 1");
        }
        let draw = normal_f32_seeded(
            pinned.len() * rank,
            (1.0 / rank as f32).sqrt(),
            mix_seed(seed, ROW_FACTOR_SALT),
        );
        let mut u = vec![0f32; n_rows * rank];
        let mut mask = vec![0f32; n_rows];
        for (i, &g) in pinned.iter().enumerate() {
            let g = g as usize;
            if g >= n_rows {
                candle_core::bail!("pinned row {g} is outside the {n_rows}-row table");
            }
            u[g * rank..(g + 1) * rank].copy_from_slice(&draw[i * rank..(i + 1) * rank]);
            mask[g] = 1.0;
        }
        Ok(Self {
            u: Var::from_tensor(&Tensor::from_vec(u, (n_rows, rank), dev)?)?,
            v: Var::zeros((rank, dim), DType::F32, dev)?,
            u_mask: Tensor::from_vec(mask, (n_rows, 1), dev)?,
            n_pinned: pinned.len(),
        })
    }

    #[must_use]
    pub fn factors(&self) -> LoraFactors {
        LoraFactors::from_parts(self.u.as_tensor().clone(), self.v.as_tensor().clone())
    }

    /// The residual on the rows named by `ids` with the mask IN the graph:
    /// `(u[ids] ⊙ mask[ids]) · v`, `[ids, H]`. Gather first, then mask, so
    /// the cost is the chunk's, not the table's.
    pub fn residual_rows_masked(&self, ids: &Tensor) -> Result<Tensor> {
        gather_rows(&self.u, ids)?
            .broadcast_mul(&gather_rows(&self.u_mask, ids)?)?
            .matmul(&self.v)
    }

    /// The whole `[N, H]` residual with the mask in the graph, for a trainer
    /// that materializes the composed table once.
    pub fn residual_masked(&self) -> Result<Tensor> {
        self.u.broadcast_mul(&self.u_mask)?.matmul(&self.v)
    }

    /// The residual on the rows named by `ids`, `[ids, H]`.
    pub fn residual_rows(&self, ids: &Tensor) -> Result<Tensor> {
        self.factors().residual_rows(ids)
    }

    /// The whole `[N, H]` residual, for output and folding.
    pub fn residual(&self) -> Result<Tensor> {
        self.factors().residual()
    }

    /// The residual's summed row norm² over the pinned rows, without forming
    /// it (see [`LoraFactors::ridge`]). No mask: a free row's factor is zero
    /// by construction and stays so under either trainer, so it adds nothing.
    pub fn ridge(&self) -> Result<Tensor> {
        self.factors().ridge()
    }

    /// The row optimizers: `u` at `lr`, `v` at `lr_ratio × lr` (LoRA+).
    pub fn optimizers(&self, lr: f64, lr_ratio: f32, dev: &Device) -> Result<PinnedLoraOpt> {
        Ok(PinnedLoraOpt {
            u: RowAdagrad::new(self.u.dims()[0], lr, dev)?,
            v: RowAdagrad::new(self.v.dims()[0], lr * f64::from(lr_ratio), dev)?,
        })
    }

    /// One step of both factors from `grads`: `u`'s gradient masked to the
    /// pinned rows, `v`'s as it is. A factor the loss never reached is left.
    pub fn step(&self, opt: &mut PinnedLoraOpt, grads: &GradStore) -> Result<()> {
        if let Some(g) = grads.get(&self.u) {
            opt.u.step(&self.u, &g.broadcast_mul(&self.u_mask)?)?;
        }
        if let Some(g) = grads.get(&self.v) {
            opt.v.step(&self.v, g)?;
        }
        Ok(())
    }
}

/// The two row optimizers of a [`PinnedLora`].
pub struct PinnedLoraOpt {
    pub u: RowAdagrad,
    pub v: RowAdagrad,
}

/// The LoRA+ learning-rate split, for a trainer that builds its optimizers
/// from a `VarMap`: the named `v` leaves the main group and gets its own.
#[derive(Clone, Copy, Debug)]
pub struct LoraPlus<'a> {
    /// The full name of `v` in the map (`"{prefix}.lora_v"`).
    pub v_var: &'a str,
    pub lr_ratio: f32,
    /// Per-epoch ridge weight per row on the residual's row norm² (see
    /// [`LoraFactors::ridge`]); the trainer spreads it over the epoch's steps.
    pub ridge: f32,
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

/// `"{prefix}.{slot}"`, or `slot` alone under an empty prefix — candle's own
/// `VarBuilder` path rule.
#[must_use]
pub fn join(prefix: &str, slot: &str) -> String {
    if prefix.is_empty() {
        slot.to_string()
    } else {
        format!("{prefix}.{slot}")
    }
}

/// The factor names under a table prefix: `("{prefix}.lora_u", "{prefix}.lora_v")`.
#[must_use]
pub fn factor_names(prefix: &str) -> (String, String) {
    (join(prefix, U_VAR_NAME), join(prefix, V_VAR_NAME))
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
