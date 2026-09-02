//! Learned gene modules: a mixed-membership mixture of module vectors in front
//! of the feature embedding.
//!
//! A free model trains one row `ρ_g` per feature, and that row receives gradient
//! only on the steps that draw its feature as a positive or a negative. A feature
//! that is rare, or absent from a later dataset, has nothing standing in for it.
//! Here every row is a mixture of `M` shared module vectors plus a small residual:
//!
//!   `ρ_g = Σ_m π_gm · μ_m + r_g`,   `π_g = sparsemax(logits_g)`
//!
//! so an edge on any member of a module updates `μ_m`, and a feature that is never
//! drawn still inherits a trained row through its siblings. `sparsemax` puts each
//! feature on a few modules with exact zeros elsewhere — mixed membership by
//! construction, never a uniform average of every module, which is the one-point
//! degeneracy a soft pool collapses into.
//!
//! The membership is a learned `Var`, warm-started from a clustering of the
//! feature profiles and held fixed for the first epochs (see
//! [`FeatModules::set_frozen`]): a learned partition trained from a cold start is
//! rich-get-richer, and the exact module term (`crate::loss::modules`) rewards
//! putting every feature in one module unless a load-balance prior and the residual
//! ridge push back. The trainer owns those terms; this type owns the tables and the
//! composition.
//!
//! Follows the adapter's contract: `e_feat` on the model is a detached composed
//! snapshot, the per-batch gather composes the live parameters, and
//! [`super::JointEmbedModel::materialize_e_feat`] refreshes the snapshot after
//! training so phase 2 and every output reader see a fixed dictionary.

use candle_util::candle_core::{Device, Result, Tensor};
use candle_util::candle_nn::VarMap;
use candle_util::nn::sparsemax;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::vars::{register_randn_seeded, register_var_from_mat, register_var_from_slice};
use super::JointEmbedModel;

/// Registered names of the module Vars, so a caller that owns the `VarMap` can
/// fetch them without hard-coding strings this crate chose.
pub const MODULE_LOGITS_VAR_NAME: &str = "module_logits";
pub const MODULE_MU_VAR_NAME: &str = "module_mu";
pub const MODULE_RESIDUAL_VAR_NAME: &str = "module_residual";
pub const MODULE_BIAS_VAR_NAME: &str = "module_bias";

/// Inputs for [`JointEmbedModel::new_with_modules`].
pub struct ModuleInit<'a> {
    pub n_features: usize,
    pub n_cells: usize,
    pub embedding_dim: usize,
    pub n_modules: usize,
    /// Warm-start module per feature (`len == n_features`, values `< n_modules`).
    /// `None` seeds every feature uniformly across modules.
    pub init_labels: Option<&'a [u32]>,
    /// Share of a feature's membership placed on its warm-start module; the rest
    /// is spread evenly over the others so every module stays in the sparsemax
    /// support and can still earn the feature. See [`module_logit_for_own_mass`].
    pub init_own_mass: f32,
    /// Explicit membership logits `[n_features, M]`, taking precedence over
    /// `init_labels`. A membership row is a simplex point and sparsemax of a
    /// simplex point is itself, so a parent's `π` passed here is reproduced
    /// exactly — the warm start `update` carries.
    pub init_logits: Option<&'a nalgebra::DMatrix<f32>>,
    /// Explicit module dictionary `[M, H]` instead of the seeded randn.
    pub init_mu: Option<&'a nalgebra::DMatrix<f32>>,
    pub b_feat: &'a [f32],
    pub b_cell: &'a [f32],
    /// Base seed for the reproducible randn init of `μ` and the cell side.
    pub seed: u64,
}

/// The module tables. Cloned into every sharing head so all heads train the
/// SAME Vars; `frozen` is shared through the `Arc` so a warm-up flip is seen by
/// every axis at once.
#[derive(Clone)]
pub struct FeatModules {
    /// Membership logits `[n_features, M]` (Var). `sparsemax` per row.
    pub logits: Tensor,
    /// Module dictionary `[M, H]` (Var).
    pub mu: Tensor,
    /// Per-feature residual `[n_features, H]` (Var, zero-init, ridge-shrunk by the
    /// trainer). What no combination of a feature's modules explains.
    pub residual: Tensor,
    /// Per-module bias `[M]` (Var), the module-level abundance in the exact
    /// module term. Not part of `ρ`.
    pub b_module: Tensor,
    pub n_modules: usize,
    /// While set, the membership is detached in every forward, so the warm-start
    /// partition is held and only `μ` / `r` train.
    pub frozen: Arc<AtomicBool>,
}

/// The logit that puts a share `p` of a feature's sparsemax membership on one
/// module when every other module's logit is zero.
///
/// With `logits = (κ, 0, …, 0)` and `κ < 1` every module is in the sparsemax
/// support, the threshold is `τ = (κ − 1)/M`, the own module gets `κ − τ` and each
/// other gets `−τ`; solving `κ − τ = p` gives `κ = (p·M − 1)/(M − 1)`. Valid for
/// `1/M ≤ p ≤ 1`; `p = 1` (`κ = 1`) is the hard one-hot start. A softmax would
/// have needed `ln(p/(1−p)·(M−1))` here — a flat `+3` gives only a few percent at
/// large `M` — so the number is derived, not chosen.
#[must_use]
pub fn module_logit_for_own_mass(own_mass: f32, n_modules: usize) -> f32 {
    let m = n_modules.max(2) as f32;
    let p = own_mass.clamp(1.0 / m, 1.0);
    (p * m - 1.0) / (m - 1.0)
}

impl FeatModules {
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::Relaxed)
    }

    /// Hold (`true`) or release (`false`) the membership. Shared across heads.
    pub fn set_frozen(&self, frozen: bool) {
        self.frozen.store(frozen, Ordering::Relaxed);
    }

    fn live_logits(&self) -> Tensor {
        if self.is_frozen() {
            self.logits.detach()
        } else {
            self.logits.clone()
        }
    }

    /// The full membership `π [n_features, M]` on the live logits (detached while
    /// frozen). Rows are on the simplex with exact zeros.
    pub fn membership(&self) -> Result<Tensor> {
        sparsemax(&self.live_logits())
    }

    /// Membership rows for `idx` — sparsemax is row-wise, so gathering first is
    /// exact and keeps the per-step work at `[b, M]`.
    pub fn membership_rows(&self, idx: &Tensor) -> Result<Tensor> {
        sparsemax(&self.live_logits().index_select(idx, 0)?)
    }

    /// The full composed table `ρ = π μ + r`, `[n_features, H]`, on the live
    /// parameters.
    pub fn compose(&self) -> Result<Tensor> {
        self.membership()?.matmul(&self.mu)?.add(&self.residual)
    }

    /// Composed rows for `idx`, `[b, H]`, on the live parameters.
    pub fn compose_rows(&self, idx: &Tensor) -> Result<Tensor> {
        self.membership_rows(idx)?
            .matmul(&self.mu)?
            .add(&self.residual.index_select(idx, 0)?)
    }
}

impl JointEmbedModel {
    /// Module constructor: allocate the membership logits (warm-started from
    /// `init_labels`), the randn module dictionary, the zero residual and the
    /// module bias, plus a fresh cell side. The `e_feat` field is seeded with the
    /// composed table and refreshed after phase 1 via [`Self::materialize_e_feat`].
    pub fn new_with_modules(args: ModuleInit, varmap: &VarMap, dev: &Device) -> Result<Self> {
        let (d, m, h) = (args.n_features, args.n_modules, args.embedding_dim);
        if m < 2 {
            candle_util::candle_core::bail!("new_with_modules: need at least 2 modules, got {m}");
        }
        if args.b_feat.len() != d {
            candle_util::candle_core::bail!(
                "new_with_modules: b_feat has {} entries but n_features is {d}",
                args.b_feat.len()
            );
        }
        if args.b_cell.len() != args.n_cells {
            candle_util::candle_core::bail!(
                "new_with_modules: b_cell has {} entries but n_cells is {}",
                args.b_cell.len(),
                args.n_cells
            );
        }
        let kappa = module_logit_for_own_mass(args.init_own_mass, m);
        let mut logits_host = nalgebra::DMatrix::<f32>::zeros(d, m);
        if let Some(l) = args.init_logits {
            if l.nrows() != d || l.ncols() != m {
                candle_util::candle_core::bail!(
                    "new_with_modules: init_logits is {}×{} but the model is {d}×{m}",
                    l.nrows(),
                    l.ncols()
                );
            }
            logits_host.copy_from(l);
        }
        // No warm start leaves every module in the support with equal mass, so
        // the data alone decides the partition. Explicit logits win over labels.
        if let (Some(labels), None) = (args.init_labels, args.init_logits) {
            {
                if labels.len() != d {
                    candle_util::candle_core::bail!(
                        "new_with_modules: {} warm-start labels for {d} features",
                        labels.len()
                    );
                }
                for (g, &lab) in labels.iter().enumerate() {
                    if lab as usize >= m {
                        candle_util::candle_core::bail!(
                            "new_with_modules: warm-start label {lab} for feature {g} is not \
                             below the module count {m}"
                        );
                    }
                    logits_host[(g, lab as usize)] = kappa;
                }
            }
        }
        let logits = register_var_from_mat(varmap, dev, MODULE_LOGITS_VAR_NAME, &logits_host)?;
        let mu = match args.init_mu {
            Some(parent) => {
                if parent.nrows() != m || parent.ncols() != h {
                    candle_util::candle_core::bail!(
                        "new_with_modules: init_mu is {}×{} but the model needs {m}×{h}",
                        parent.nrows(),
                        parent.ncols()
                    );
                }
                register_var_from_mat(varmap, dev, MODULE_MU_VAR_NAME, parent)?
            }
            None => register_randn_seeded(varmap, dev, MODULE_MU_VAR_NAME, m, h, args.seed)?,
        };
        let residual = register_var_from_mat(
            varmap,
            dev,
            MODULE_RESIDUAL_VAR_NAME,
            &nalgebra::DMatrix::<f32>::zeros(d, h),
        )?;
        let b_module = register_var_from_slice(varmap, dev, MODULE_BIAS_VAR_NAME, &vec![0f32; m])?;
        let e_cell = register_randn_seeded(varmap, dev, "e_cell", args.n_cells, h, args.seed)?;
        let b_feat = register_var_from_slice(varmap, dev, "b_feat", args.b_feat)?;
        let b_cell = register_var_from_slice(varmap, dev, "b_cell", args.b_cell)?;

        let modules = FeatModules {
            logits,
            mu,
            residual,
            b_module,
            n_modules: m,
            frozen: Arc::new(AtomicBool::new(false)),
        };
        let e_feat = modules.compose()?.detach();
        Ok(Self {
            e_feat,
            e_cell,
            b_feat,
            b_cell,
            factor: None,
            adapter: None,
            modules: Some(modules),
            embedding_dim: h,
            s_feat: None,
            e_feat_raw: None,
            e_feat_logstd: None,
            gate_pip: None,
            gate_mask: Arc::new(std::sync::Mutex::new(None)),
            gate_ibp_bias: None,
            gate: None,
        })
    }

    /// The membership `π` as a detached `[n_features, M]` table for output — the
    /// mean parameter, never a frozen view. `None` without modules.
    pub fn module_membership(&self) -> Result<Option<Tensor>> {
        self.modules
            .as_ref()
            .map(|m| sparsemax(&m.logits)?.detach().contiguous())
            .transpose()
    }
}
