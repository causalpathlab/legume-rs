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
//! Features are addressed at fine resolution. The cell axis is
//! coarsened: cell embeddings are mean-pooled (per the batch's chosen
//! seed coarsening) over the fine children of each touched pb-sample.

use candle_util::candle_core::{Device, Result, Tensor};
use candle_util::candle_nn::{self, VarBuilder, VarMap};

mod modules;
mod score;
mod vars;

pub use modules::{
    module_logit_for_own_mass, FeatModules, ModuleInit, ModuleWarmStart, MODULE_BIAS_VAR_NAME,
    MODULE_LOGITS_VAR_NAME, MODULE_MU_VAR_NAME, MODULE_RESIDUAL_VAR_NAME,
};
use vars::{
    build_feat_factor, register_randn_seeded, register_var_from_mat, register_var_from_slice,
};
// Reached only through `tests`' `use super::*` — the live `pool_axis` caller is in
// `score`.
#[cfg(test)]
use vars::{pool_axis, pool_axis_loop};

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

/// Inputs for [`JointEmbedModel::new_factored`] — a per-gene β-sharing feature
/// parameterization (see [`FeatFactor`]). `row_to_gene[r]` is the gene index of
/// feature row `r` (length `n_features`); rows sharing a gene reuse one `β_g`.
pub struct FactoredInit<'a> {
    pub n_features: usize,
    pub n_cells: usize,
    pub embedding_dim: usize,
    pub n_genes: usize,
    pub row_to_gene: &'a [u32],
    pub b_feat: &'a [f32],
    pub b_cell: &'a [f32],
    /// Base seed for the reproducible randn init of `β` and the cell side.
    pub seed: u64,
    /// Per-row unspliced flag (`len == n_features`). When `Some`, a ridge-shrunk
    /// per-gene `δ_g` Var is allocated and added to the unspliced rows
    /// (spliced identity + nascent offset); `None` = plain β-sharing.
    pub unspliced_rows: Option<&'a [bool]>,
}

/// Optional per-gene β-sharing feature factorization (used by `senna gem`'s
/// spliced/unspliced model). Instead of a free `e_feat` row per feature, every
/// feature row reuses a per-GENE base embedding `β [G, H]`:
///
///   `e_feat[row] = β[gene(row)]`
///
/// so a gene's spliced rows embed as `β_g`. **Optionally** a per-gene splice
/// offset `δ_g` is carried for the unspliced rows:
///
///   `e_feat[row] = β_g + [row is unspliced] · δ_g`
///
/// so spliced = current-state identity `β_g` and unspliced = nascent `β_g + δ_g`.
/// `δ_g` is **L2 (ridge) shrunk** (phase-1 penalty), which resolves the otherwise-
/// ambiguous split against an equal-and-opposite cell-axis shift: the shrunk
/// gene-side `δ_g` absorbs the (dense) static per-gene nascent structure (the
/// "γ"), and the residual dynamics stay on the CELL axis as the phase-2 velocity
/// increment `δ_cell` (a raw Poisson-MAP shift with θ held fixed; see
/// `crate::fit::project_cells_phase2`). With
/// `delta = None` this reduces to plain β-sharing (spliced ≡ unspliced ≡ `β_g`).
/// `β` / `δ_g` are learnable `Var`s; `row_to_gene` / the unspliced mask are fixed.
/// The score/loss path composes the row→gene→(β,δ) gathers directly (no
/// full-table materialization per step); output/co-embed readers use the
/// `e_feat` field after [`JointEmbedModel::materialize_e_feat`].
#[derive(Clone)]
pub struct FeatFactor {
    /// Per-gene base embedding `[G, H]` (Var).
    pub beta: Tensor,
    /// `[n_features]` u32 (device): row → gene index.
    pub row_to_gene: Tensor,
    /// Optional per-gene splice offset, present as `(δ_g [G, H] Var, mask [n_features,
    /// 1])` together (they always co-exist). `δ_g` is added to the **unspliced** rows
    /// (the `mask` = 1/0 selector); L2-ridge in phase-1. `None` = plain β-sharing
    /// (spliced ≡ unspliced ≡ `β_g`).
    pub splice_delta: Option<(Tensor, Tensor)>,
}

impl FeatFactor {
    /// Materialize the full feature embedding `[n_features, H]` from `β` (plus
    /// `δ_g` on the unspliced rows). Stays in the autograd graph so gradients
    /// flow back to the `β` / `δ_g` Vars.
    fn e_feat(&self) -> Result<Tensor> {
        let base = self.beta.index_select(&self.row_to_gene, 0)?;
        match &self.splice_delta {
            Some((delta, mask)) => {
                let d = delta.index_select(&self.row_to_gene, 0)?; // [n_features, H]
                base.add(&d.broadcast_mul(mask)?) // + mask ⊙ δ_g on unspliced rows
            }
            None => Ok(base),
        }
    }
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

pub struct JointEmbedModel {
    /// Unified feature embedding (genes ∪ peaks). When `factor` is `Some`, this
    /// is a materialized snapshot of the per-gene `β` gathered to feature rows —
    /// refreshed by [`Self::materialize_e_feat`] after phase 1 so phase-2 /
    /// outputs read a fixed dictionary; the training loss never reads this field
    /// for a factored model — it gathers each batch's rows straight from `β`.
    pub e_feat: Tensor,
    /// The "cell" axis is the CALLER'S trained unit: senna bge/gem pass
    /// cells here; pinto cage passes finest-level pseudobulks.
    pub e_cell: Tensor,
    pub b_feat: Tensor,
    pub b_cell: Tensor,
    /// Optional per-gene β-sharing feature parameterization (`None` = free `e_feat`).
    pub factor: Option<FeatFactor>,
    /// Optional fixed-dictionary adapter parameterization (`None` = free
    /// `e_feat`). Mutually exclusive with `factor` by construction.
    pub adapter: Option<FeatAdapter>,
    /// Optional learned-module parameterization (`None` = free `e_feat`): every
    /// row is `Σ_m π_gm μ_m + r_g`. Mutually exclusive with `factor` and `adapter`.
    /// The `e_feat` field is a detached composed snapshot, as for the adapter.
    pub modules: Option<FeatModules>,
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
            factor: None,
            adapter: None,
            modules: None,
            embedding_dim: args.embedding_dim,
        })
    }

    /// The L2 term for whichever gene-side table can overfit row by row under
    /// this parameterization: the free `e_feat` Var, the factored model's per-gene
    /// `β` (its `δ_g` has the trainer's own `delta_l2`), the adapter's or the module
    /// model's per-feature residual, or nothing (an adapter without a residual
    /// trains only the shared map).
    ///
    /// Owning this here keeps a trainer from ridging `e_feat` on a model where
    /// that field is a detached snapshot, which is silently inert.
    pub fn feature_ridge(&self, lam: f64) -> Result<Option<Tensor>> {
        if let Some(c) = self.composed() {
            return match c.ridge_table() {
                Some(r) => Ok(Some(crate::loss::embedding_ridge(r, lam)?)),
                None => Ok(None),
            };
        }
        if let Some(f) = &self.factor {
            return Ok(Some(crate::loss::embedding_ridge(&f.beta, lam)?));
        }
        Ok(Some(crate::loss::embedding_ridge(&self.e_feat, lam)?))
    }

    /// The composed feature side, when this model has one (adapter or modules;
    /// mutually exclusive by construction). `None` for free and factored models.
    pub fn composed(&self) -> Option<&dyn ComposedFeat> {
        if let Some(m) = &self.modules {
            return Some(m);
        }
        self.adapter.as_ref().map(|a| a as &dyn ComposedFeat)
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
            factor: None,
            adapter: Some(adapter),
            modules: None,
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
            factor: None,
            adapter: None,
            modules: shared_modules,
            embedding_dim,
        })
    }

    /// β-sharing factored constructor: allocate a per-gene `β` Var (randn) plus a
    /// fresh cell side, and register the fixed `row_to_gene` index tensor. The
    /// `e_feat` field is seeded with the materialized `β` (gathered to feature
    /// rows) and refreshed after phase 1 via [`Self::materialize_e_feat`].
    pub fn new_factored(
        args: FactoredInit,
        varmap: &VarMap,
        vs: VarBuilder,
        dev: &Device,
    ) -> Result<Self> {
        let beta = register_randn_seeded(
            varmap,
            dev,
            "beta",
            args.n_genes,
            args.embedding_dim,
            args.seed,
        )?;
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

        // Optional per-gene splice offset δ_g, zero-initialized (so training
        // starts exactly at β-sharing and δ_g grows only where the data + L2
        // tradeoff justifies it).
        let delta = match args.unspliced_rows {
            Some(_) => Some(vs.get_with_hints(
                (args.n_genes, args.embedding_dim),
                "delta",
                candle_nn::Init::Const(0.0),
            )?),
            None => None,
        };
        let factor = build_feat_factor(&beta, args.row_to_gene, delta, args.unspliced_rows, dev)?;
        let e_feat = factor.e_feat()?.detach();
        Ok(Self {
            e_feat,
            e_cell,
            b_feat,
            b_cell,
            factor: Some(factor),
            adapter: None,
            modules: None,
            embedding_dim: args.embedding_dim,
        })
    }

    /// Composite-training constructor for a factored model: share this model's
    /// `β` / `b_feat` + factor index tensor (so every level trains the SAME
    /// feature side) and allocate a fresh cell side under `var_prefix`. Delegates
    /// the cell-var allocation to [`Self::new_sharing_features`] and re-attaches
    /// the shared [`FeatFactor`].
    pub fn new_sharing_factor(
        &self,
        n_cells: usize,
        var_prefix: &str,
        varmap: &VarMap,
        dev: &Device,
        seed: u64,
    ) -> Result<Self> {
        let factor = self
            .factor
            .as_ref()
            .expect("new_sharing_factor requires a factored parent model");
        let mut model = Self::new_sharing_features(
            ShareFeaturesArgs {
                n_cells,
                embedding_dim: self.embedding_dim,
                shared_e_feat: self.e_feat.clone(),
                shared_b_feat: self.b_feat.clone(),
                e_cell_init: None,
                b_cell_init: &vec![0f32; n_cells],
                var_prefix,
                seed,
                shared_modules: None,
            },
            varmap,
            dev,
        )?;
        model.factor = Some(factor.clone());
        Ok(model)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod adapter_tests;

#[cfg(test)]
mod module_tests;

impl JointEmbedModel {
    /// Snapshot the composed feature side into the `e_feat` field (detached), so
    /// the phase-2 projection and every output / co-embed reader see a fixed
    /// dictionary: recomposed from the live parameters for an adapter / module
    /// model, `β` (+ `δ` on unspliced rows) gathered to feature rows for a
    /// factored one. A no-op for a free model, whose `e_feat` already IS the
    /// trained Var. Call after phase 1.
    pub fn materialize_e_feat(&mut self) -> Result<()> {
        let snapshot = if let Some(c) = self.composed() {
            Some(c.compose()?.detach())
        } else if let Some(f) = &self.factor {
            let mask = f.splice_delta.as_ref().map(|(_, m)| m.clone());
            Some(
                self.factored_feat_rows(f, &f.row_to_gene, mask.as_ref())?
                    .detach(),
            )
        } else {
            None
        };
        if let Some(s) = snapshot {
            self.e_feat = s;
        }
        Ok(())
    }
}
