//! Joint multiome embedding tables + bias terms + bilinear scoring.
//!
//! Two free embedding tables (`E_feat` over the unified feature axis,
//! `E_cell`) plus two bias vectors (`b_feat`, `b_cell`). Score for a
//! `(feature, cell)` edge under a Poisson rate model:
//!
//!   `score(f, c) = E_feat[f] · E_cell[c] + b_feat[f] + b_cell[c]`
//!
//! All callers (bge, gem, pinto) use the full score: the per-cell bias
//! `b_cell` absorbs library size, so it is trained in phase 1, re-fitted
//! analytically in phase 2, and written out.
//!
//! Features are addressed at fine resolution, and so is the cell axis: every
//! score reads a cell's own row of `e_cell` directly, with no coarse→fine
//! pooling.

use candle_util::candle_core::{Device, Result, Tensor};
use candle_util::candle_nn::VarMap;
use candle_util::fast_index::gather_rows;
use candle_util::lora::PinnedLora;

mod modules;
mod score;
mod vars;

pub use modules::{
    module_logit_for_own_mass, FeatModules, ModuleInit, ModuleWarmStart, MODULE_BIAS_VAR_NAME,
    MODULE_LOGITS_VAR_NAME, MODULE_MU_VAR_NAME, MODULE_RESIDUAL_VAR_NAME,
};
use vars::{register_randn_seeded, register_var_from_mat, register_var_from_slice};

/// stdev of the embedding-table randn init (matches the former
/// `candle_nn::Init::Randn { stdev: 0.1 }`).
const INIT_STDEV: f32 = 0.1;

/// Shape of the embedding tables.
pub struct ModelArgs {
    pub n_features: usize,
    pub n_cells: usize,
    pub embedding_dim: usize,
    /// Base seed for the reproducible randn init of any `None` embedding.
    pub seed: u64,
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
/// heads can coexist in one `VarMap` (e.g. `pb_l0`, `pb_l1`, ..., `cell`).
pub struct ShareFeaturesArgs<'a> {
    pub n_cells: usize,
    pub embedding_dim: usize,
    pub shared_e_feat: Tensor,
    pub shared_b_feat: Tensor,
    pub e_cell_init: Option<&'a nalgebra::DMatrix<f32>>,
    pub b_cell_init: &'a [f32],
    pub var_prefix: &'a str,
    /// Base seed for the reproducible randn init of the cell side when
    /// `e_cell_init` is `None`.
    pub seed: u64,
    /// Shared module tables ([`FeatModules`]) for a module-parameterized primary
    /// model. Every head must carry the SAME clone: a head without it gathers from
    /// `e_feat`, which for this parameterization is a detached snapshot, and trains a
    /// feature side nothing else sees.
    pub shared_modules: Option<FeatModules>,
}

/// Inputs for [`JointEmbedModel::new_adapted`] — a fixed-dictionary
/// adapter feature parameterization (see [`FeatAdapter`]).
/// The registered name of the free feature-embedding Var. Public so a caller
/// that owns the training loop (and therefore the `VarMap`) can fetch the Var
/// without hard-coding a string this crate chose.
pub const E_FEAT_VAR_NAME: &str = "e_feat";

pub struct AdapterInit<'a> {
    pub n_cells: usize,
    /// Output width `H` (must match the cell side).
    pub embedding_dim: usize,
    /// Fixed dictionary `[n_features, h_src]`. Uploaded as a constant tensor,
    /// never registered as a Var. Row `i` must already be aligned to feature
    /// `i` of this model's feature axis; alignment is the caller's job.
    pub rho: &'a nalgebra::DMatrix<f32>,
    pub b_feat: &'a [f32],
    pub b_cell: &'a [f32],
    /// Base seed for the reproducible randn init of `W` and the cell side.
    pub seed: u64,
    /// Allocate the optional per-feature residual (zero-init, so training
    /// starts exactly at `rho . W`).
    pub residual: bool,
}

/// Fixed-dictionary adapter feature side. Instead of a free `e_feat` row per
/// feature, every row is a linear map of its row in a FIXED dictionary `rho`
/// (a pre-trained embedding from another run):
///
///   `e_feat[g] = rho[g] . W (+ residual[g])`
///
/// `W [h_src, H]` is the only mandatory gene-side parameter, so every
/// feature's gradient trains the same shared map; the optional per-feature
/// `residual [n_features, H]` restores row-level freedom where the shared map
/// is not enough (callers ridge it). `rho` is a constant tensor, NOT a Var:
/// it never trains and the optimizer never sees it. The score/loss path
/// composes per-batch gathers directly (no full-table materialization per
/// step); output/co-embed readers use the `e_feat` field after
/// [`JointEmbedModel::materialize_e_feat`].
#[derive(Clone)]
pub struct FeatAdapter {
    /// Fixed dictionary `[n_features, h_src]` (constant tensor).
    pub rho: Tensor,
    /// Learnable map `[h_src, H]` (Var).
    pub w: Tensor,
    /// Optional per-feature residual `[n_features, H]` (Var, zero-init).
    pub residual: Option<Tensor>,
}

/// A feature side whose `e_feat` is a detached COMPOSED snapshot of live
/// parameters — the adapter (`ρ·W + r`) and the module layer (`π μ + r`). The
/// gather, the materialize and the ridge all treat these two the same way, so
/// they dispatch on this trait rather than on each field.
pub trait ComposedFeat {
    /// The full composed table `[n_features, H]`, on the live parameters.
    fn compose(&self) -> Result<Tensor>;
    /// Composed rows for `idx`, `[b, H]`, on the live parameters.
    fn compose_rows(&self, idx: &Tensor) -> Result<Tensor>;
    /// The per-row table that can overfit row by row and takes the ridge, if any.
    fn ridge_table(&self) -> Option<&Tensor>;
}

impl ComposedFeat for FeatAdapter {
    fn compose(&self) -> Result<Tensor> {
        let base = self.rho.matmul(&self.w)?;
        match &self.residual {
            Some(r) => base.add(r),
            None => Ok(base),
        }
    }

    fn compose_rows(&self, idx: &Tensor) -> Result<Tensor> {
        let mut rows = self.rho.index_select(idx, 0)?.matmul(&self.w)?;
        if let Some(r) = &self.residual {
            rows = rows.add(&r.index_select(idx, 0)?)?;
        }
        Ok(rows)
    }

    fn ridge_table(&self) -> Option<&Tensor> {
        self.residual.as_ref()
    }
}

impl FeatAdapter {
    /// The full composed table `[n_features, H]`, on the live parameters.
    pub fn compose(&self) -> Result<Tensor> {
        ComposedFeat::compose(self)
    }
}

/// Anchored feature side: `e_feat[g] = base[g] + u_g · v` on the anchored
/// rows and `base[g]` elsewhere. `base` is the model's own `e_feat` Var (the
/// caller pins its anchored rows, by a post-step restore, and lets the rest
/// train); the factors are a [`PinnedLora`] whose `u` is masked to the
/// anchored rows inside the composition, so a free row's factor never gets
/// gradient. The residual is shrunk by [`Self::ridge`] at the LoRA ridge
/// weight, not by the table ridge, so [`ComposedFeat::ridge_table`] is `None`.
pub struct FeatLora {
    /// The trained table, `[n_features, H]`, shared with the model's `e_feat`
    /// Var at construction (a clone of the same storage).
    pub base: Tensor,
    pub lora: PinnedLora,
}

impl ComposedFeat for FeatLora {
    fn compose(&self) -> Result<Tensor> {
        self.base.add(&self.lora.masked_factors()?.residual()?)
    }

    fn compose_rows(&self, idx: &Tensor) -> Result<Tensor> {
        let rows = gather_rows(&self.base, idx)?;
        rows.add(&self.lora.masked_factors()?.residual_rows(idx)?)
    }

    fn ridge_table(&self) -> Option<&Tensor> {
        None
    }
}

impl FeatLora {
    /// The full composed table `[n_features, H]`, on the live parameters.
    pub fn compose(&self) -> Result<Tensor> {
        ComposedFeat::compose(self)
    }

    /// The residual's summed row norm² over the anchored rows, without forming
    /// it (see [`candle_util::lora::LoraFactors::ridge`]).
    pub fn ridge(&self) -> Result<Tensor> {
        self.lora.masked_factors()?.ridge()
    }
}

pub struct JointEmbedModel {
    /// Unified feature embedding (genes ∪ peaks). When `adapter` or `modules` is
    /// `Some`, this is a materialized snapshot of the composed live parameters —
    /// refreshed by [`Self::materialize_e_feat`] after training so phase-2 /
    /// outputs read a fixed dictionary; the training loss never reads this field
    /// for such a model — it gathers each batch's rows straight from the live
    /// parameters.
    pub e_feat: Tensor,
    /// The "cell" axis is the CALLER'S trained unit: senna bge/gem pass
    /// cells here; pinto cage passes finest-level pseudobulks.
    pub e_cell: Tensor,
    pub b_feat: Tensor,
    pub b_cell: Tensor,
    /// Optional fixed-dictionary adapter parameterization (`None` = free
    /// `e_feat`). Mutually exclusive with `modules` by construction.
    pub adapter: Option<FeatAdapter>,
    /// Optional learned-module parameterization (`None` = free `e_feat`): every
    /// row is `Σ_m π_gm μ_m + r_g`. Mutually exclusive with `adapter`.
    /// The `e_feat` field is a detached composed snapshot, as for the adapter.
    pub modules: Option<FeatModules>,
    /// Optional LoRA residual on an anchored `e_feat` (`None` = no residual):
    /// see [`FeatLora`]. Exclusive with the other two by construction.
    pub lora: Option<FeatLora>,
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
        dev: &Device,
    ) -> Result<Self> {
        let e_feat = match init.e_feat {
            Some(m) => register_var_from_mat(varmap, dev, E_FEAT_VAR_NAME, m)?,
            None => register_randn_seeded(
                varmap,
                dev,
                E_FEAT_VAR_NAME,
                args.n_features,
                args.embedding_dim,
                args.seed,
            )?,
        };
        let e_cell = match init.e_cell {
            Some(m) => register_var_from_mat(varmap, dev, "e_cell", m)?,
            None => register_randn_seeded(
                varmap,
                dev,
                "e_cell",
                args.n_cells,
                args.embedding_dim,
                args.seed,
            )?,
        };
        let b_feat = register_var_from_slice(varmap, dev, "b_feat", init.b_feat)?;
        let b_cell = register_var_from_slice(varmap, dev, "b_cell", init.b_cell)?;

        Ok(Self {
            e_feat,
            e_cell,
            b_feat,
            b_cell,
            adapter: None,
            lora: None,
            modules: None,
            embedding_dim: args.embedding_dim,
        })
    }

    /// The L2 term for whichever gene-side table can overfit row by row under
    /// this parameterization: the free `e_feat` Var, the adapter's or the module
    /// model's per-feature residual, or nothing (an adapter without a residual
    /// trains only the shared map).
    ///
    /// Owning this here keeps a trainer from ridging `e_feat` on a model where
    /// that field is a detached snapshot, which is silently inert.
    pub fn feature_ridge(&self, lam: f64) -> Result<Option<Tensor>> {
        let table = match self.composed() {
            Some(c) => c.ridge_table(),
            None => Some(&self.e_feat),
        };
        match table {
            Some(t) => Ok(Some(crate::loss::embedding_ridge(t, lam)?)),
            None => Ok(None),
        }
    }

    /// Snapshot the composed feature side into the `e_feat` field (detached), so
    /// the phase-2 projection and every output / co-embed reader see a fixed
    /// dictionary. A no-op for a free model, whose `e_feat` already IS the
    /// trained Var. Call after phase 1.
    pub fn materialize_e_feat(&mut self) -> Result<()> {
        let snapshot = match self.composed() {
            Some(c) => Some(c.compose()?.detach()),
            None => None,
        };
        if let Some(s) = snapshot {
            self.e_feat = s;
        }
        Ok(())
    }

    /// The composed feature side, when this model has one — the module layer or
    /// the adapter, mutually exclusive by construction. `None` only for a FREE
    /// model, whose `e_feat` is the trained Var itself.
    ///
    /// Every consumer that has to ask "which table does this parameterization
    /// actually train" goes through here — the gather, the materialize and the
    /// ridge — so a third parameterization is one `impl`, not two new match
    /// arms in three files.
    pub fn composed(&self) -> Option<&dyn ComposedFeat> {
        if let Some(m) = &self.modules {
            return Some(m);
        }
        if let Some(l) = &self.lora {
            return Some(l);
        }
        self.adapter.as_ref().map(|a| a as &dyn ComposedFeat)
    }

    /// Put a rank-`rank` LoRA residual on the free `e_feat` of this model, on
    /// the rows `anchored` (ids into the feature axis): the factors are
    /// registered in `varmap` beside the table under the shared LoRA names
    /// ([`candle_util::lora::factor_names`] of [`E_FEAT_VAR_NAME`]), `u`
    /// drawn on the anchored rows from `seed`, `v` at zero. The caller pins
    /// the anchored rows of `e_feat` itself and gives `v` its LoRA+ group.
    /// Refused on a model whose feature side is already composed.
    pub fn with_lora(
        mut self,
        varmap: &VarMap,
        dev: &Device,
        rank: usize,
        anchored: &[u32],
        seed: u64,
    ) -> Result<Self> {
        if self.composed().is_some() {
            candle_util::candle_core::bail!(
                "with_lora: the feature side is already a composed parameterization"
            );
        }
        let n_features = self.e_feat.dims()[0];
        let lora = PinnedLora::new(
            n_features,
            self.embedding_dim,
            rank,
            anchored,
            1.0,
            seed,
            dev,
        )?;
        let (u_name, v_name) = candle_util::lora::factor_names(E_FEAT_VAR_NAME);
        {
            let mut tbl = varmap.data().lock().unwrap();
            tbl.insert(u_name, lora.u.clone());
            tbl.insert(v_name, lora.v.clone());
        }
        self.lora = Some(FeatLora {
            base: self.e_feat.clone(),
            lora,
        });
        Ok(self)
    }

    /// Fixed-dictionary adapter constructor: upload `rho` as a constant,
    /// allocate the `W [h_src, H]` Var (randn, seeded) and optionally the
    /// zero-init per-feature residual, plus a fresh cell side. The `e_feat`
    /// field is seeded with the composed table and refreshed after phase 1
    /// via [`Self::materialize_e_feat`].
    pub fn new_adapted(args: AdapterInit, varmap: &VarMap, dev: &Device) -> Result<Self> {
        let n_features = args.rho.nrows();
        let h_src = args.rho.ncols();
        if args.b_feat.len() != n_features {
            candle_util::candle_core::bail!(
                "new_adapted: b_feat has {} entries but rho has {} rows",
                args.b_feat.len(),
                n_features
            );
        }
        if args.b_cell.len() != args.n_cells {
            candle_util::candle_core::bail!(
                "new_adapted: b_cell has {} entries but n_cells is {}",
                args.b_cell.len(),
                args.n_cells
            );
        }

        // Constant upload: same `[rows, cols]` layout as `register_var_from_mat`,
        // but deliberately NOT a Var. `to_tensor` returns a transposed view,
        // so make it contiguous for the per-batch index_select/matmul path.
        let rho = matrix_util::traits::ConvertMatOps::to_tensor(args.rho, dev)
            .map_err(|e| candle_util::candle_core::Error::Msg(e.to_string()))?
            .contiguous()?;

        let w = register_randn_seeded(
            varmap,
            dev,
            "adapter_w",
            h_src,
            args.embedding_dim,
            args.seed,
        )?;
        let residual = if args.residual {
            let zeros = nalgebra::DMatrix::<f32>::zeros(n_features, args.embedding_dim);
            Some(register_var_from_mat(
                varmap,
                dev,
                "adapter_residual",
                &zeros,
            )?)
        } else {
            None
        };
        let e_cell = register_randn_seeded(
            varmap,
            dev,
            "e_cell",
            args.n_cells,
            args.embedding_dim,
            args.seed,
        )?;
        let b_feat = register_var_from_slice(varmap, dev, "b_feat", args.b_feat)?;
        let b_cell = register_var_from_slice(varmap, dev, "b_cell", args.b_cell)?;

        let adapter = FeatAdapter { rho, w, residual };
        let e_feat = adapter.compose()?.detach();
        Ok(Self {
            e_feat,
            e_cell,
            b_feat,
            b_cell,
            adapter: Some(adapter),
            modules: None,
            lora: None,
            embedding_dim: args.embedding_dim,
        })
    }

    /// Composite-training constructor: reuse pre-existing
    /// `shared_e_feat` / `shared_b_feat` Tensors (already registered as
    /// Vars in `varmap` by an earlier call to `new_with_init`) and
    /// allocate fresh cell-side Vars under `args.var_prefix` so multiple
    /// heads coexist in one `VarMap`. `AdamW` over `varmap.all_vars()` then
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
            seed,
            shared_modules,
        } = args;
        let e_name = format!("{var_prefix}_e_cell");
        let b_name = format!("{var_prefix}_b_cell");
        let e_cell = if let Some(m) = e_cell_init {
            register_var_from_mat(varmap, dev, &e_name, m)?
        } else {
            register_randn_seeded(varmap, dev, &e_name, n_cells, embedding_dim, seed)?
        };
        let b_cell = register_var_from_slice(varmap, dev, &b_name, b_cell_init)?;
        Ok(Self {
            e_feat: shared_e_feat,
            e_cell,
            b_feat: shared_b_feat,
            b_cell,
            adapter: None,
            lora: None,
            modules: shared_modules,
            embedding_dim,
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod adapter_tests;

#[cfg(test)]
mod lora_tests;

#[cfg(test)]
mod module_tests;
