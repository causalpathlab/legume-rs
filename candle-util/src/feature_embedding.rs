//! The feature side of a model, as a mixture of shared modules.
//!
//! A free `[D, H]` table gives every feature its own row, so a feature receives
//! gradient only on the steps that draw it: a rare one keeps its initialization,
//! and a feature the axis gains later has nothing to inherit. Here a row is a
//! sparse mixture of `M` shared vectors,
//!
//!   `ρ_g = sparsemax(ℓ_g) · μ`,
//!
//! so every draw of any member trains the shared vector, and everything the
//! model has learned lives in objects that are NOT indexed by feature. Growing
//! the feature axis then adds `M` numbers saying which modules a feature belongs
//! to, rather than inventing `H` trained-looking ones.
//!
//! `sparsemax` rather than a softmax, for two reasons that the rest of the
//! design leans on: a feature lands on a few modules with exact zeros elsewhere,
//! so "unused module" is a fact rather than a threshold; and mixed membership
//! survives, so a feature is not forced to pick one.
//!
//! The same property has a sharp edge. Sparsemax's Jacobian is zero outside the
//! support, so a module a feature carries no mass on receives no gradient
//! through that feature, and a module no feature carries mass on receives none
//! at all and can never earn its way in. Capacity is therefore added by
//! SPLITTING a loaded module, never by appending a fresh one.
//!
//! One thing does still move a logit outside the support: AdamW's decoupled
//! weight decay, which shrinks every logit toward zero after the step, whatever
//! the gradient was. Sparsemax is shift-invariant but not scale-invariant, so
//! that flattens a row and can pull a dropped module back into its support. At
//! the default `--weight-decay 0` it does not run, and a support only ever
//! shrinks; a caller that turns it on is choosing the other behaviour.

use crate::fast_index::gather_rows;
use crate::nn::layers::sparsemax;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Spread of the membership logits at initialization.
///
/// Small enough that sparsemax keeps every module in the support and the
/// membership is near-uniform, large enough to break the symmetry that would
/// otherwise stall the dictionary and the topics at step 0.
const INIT_LOGIT_JITTER: f64 = 0.01;

/// Registered names, so a caller owning the `VarMap` can reach them without
/// hard-coding a string this module chose.
pub const FREE_VAR_NAME: &str = "feature.embeddings";
pub const LOGITS_VAR_NAME: &str = "modules.logits";
pub const MU_VAR_NAME: &str = "modules.mu";

/// The feature side of a model.
///
/// `Composed` is the module parameterization above. `Free` is the older
/// `[D, H]` table, kept for `M = 0`, where there are no modules to compose
/// from — every method below behaves the same way for both, so a caller only
/// cares which one it has when it wants the membership.
pub enum FeatureEmbedding {
    Free(Tensor),
    Composed { logits: Tensor, mu: Tensor },
}

impl FeatureEmbedding {
    /// Register both parameters.
    ///
    /// The membership starts near-uniform: every module is inside the support,
    /// so none is preferred and none is excluded, and the data alone decides.
    /// It is near-uniform rather than exactly flat because exactly flat is a
    /// stationary point — every feature would compose the SAME row, the topic
    /// term would cancel out of the normalized dictionary, and both the
    /// dictionary and the topic embeddings would receive identically zero
    /// gradient. The noise is small enough to carry no prior about which
    /// features belong together, and only breaks that symmetry.
    ///
    /// A feature the axis gains LATER is different: the model around it is
    /// already trained, so there is no symmetry left to break, and it is given
    /// the exactly-flat row that starts it at the dictionary's centroid — see
    /// [`crate::grow::new_slab_value`].
    pub fn new(
        n_features: usize,
        n_modules: usize,
        embedding_dim: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        if n_modules == 0 {
            return Ok(Self::Free(vs.get_with_hints(
                (n_features, embedding_dim),
                FREE_VAR_NAME,
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?));
        }
        Ok(Self::Composed {
            logits: vs.get_with_hints(
                (n_features, n_modules),
                LOGITS_VAR_NAME,
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: INIT_LOGIT_JITTER,
                },
            )?,
            mu: vs.get_with_hints(
                (n_modules, embedding_dim),
                MU_VAR_NAME,
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?,
        })
    }

    /// Every feature's membership `[D, M]`, rows on the simplex with exact
    /// zeros off it. `None` for a free table, which has no modules.
    pub fn membership(&self) -> Result<Option<Tensor>> {
        match self {
            Self::Free(_) => Ok(None),
            Self::Composed { logits, .. } => Ok(Some(sparsemax(logits)?)),
        }
    }

    /// Apply a row-LINEAR map to the feature side and return the composed
    /// result, without ever forming the whole `[D, H]` table.
    ///
    /// Every read of this type is this one operation. Because composition is a
    /// matmul on the right, a map that is linear in the rows commutes with it:
    /// `f(π μ) = f(π) μ`. So the map runs on the `[·, M]` membership and the
    /// dictionary is applied once afterwards. Selecting a minibatch's rows,
    /// averaging within a coarse group, or taking the whole table are all the
    /// same call with a different `f`.
    ///
    /// **`f` must be linear in the rows.** Anything else — a normalization, a
    /// softmax — gives a different answer on each variant, silently: on a free
    /// table it applies to composed rows, here it applies to the membership.
    pub fn map_rows_linear(&self, f: impl FnOnce(&Tensor) -> Result<Tensor>) -> Result<Tensor> {
        match self {
            Self::Free(rho) => f(rho),
            Self::Composed { logits, mu } => f(&sparsemax(logits)?)?.matmul(mu),
        }
    }

    /// The rows named by `ids`, composed without forming the whole table.
    ///
    /// The cheap path, and the one training uses: `[n, M]` gathered, projected,
    /// then one matmul against `μ`.
    pub fn gather(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Free(rho) => gather_rows(rho, ids),
            Self::Composed { logits, mu } => sparsemax(&gather_rows(logits, ids)?)?.matmul(mu),
        }
    }

    /// The membership of the rows named by `ids`, `[n, M]`, without forming
    /// the whole table. `None` for a free table.
    pub fn gather_membership(&self, ids: &Tensor) -> Result<Option<Tensor>> {
        match self {
            Self::Free(_) => Ok(None),
            Self::Composed { logits, .. } => Ok(Some(sparsemax(&gather_rows(logits, ids)?)?)),
        }
    }

    /// The whole `[D, H]` table. For export and diagnostics: forming it is the
    /// cost this parameterization exists to avoid, so it does not belong on a
    /// training path.
    pub fn full(&self) -> Result<Tensor> {
        self.map_rows_linear(|rows| Ok(rows.clone()))
    }

    /// Features on the axis.
    #[must_use]
    pub fn n_features(&self) -> usize {
        match self {
            Self::Free(rho) => rho.dims()[0],
            Self::Composed { logits, .. } => logits.dims()[0],
        }
    }

    /// Width of a composed row.
    #[must_use]
    pub fn embedding_dim(&self) -> usize {
        match self {
            Self::Free(rho) => rho.dims()[1],
            Self::Composed { mu, .. } => mu.dims()[1],
        }
    }

    /// A fixed table, for a caller that already has one — tests, benches, and
    /// anything scoring a trained checkpoint rather than fitting it.
    ///
    /// Panics on a table that is not 2-D. Every consumer indexes `dims()[0]`
    /// and `dims()[1]`, so the alternative is an out-of-bounds panic further
    /// away from the mistake.
    #[must_use]
    pub fn fixed(rho: Tensor) -> std::sync::Arc<Self> {
        assert_eq!(
            rho.rank(),
            2,
            "a feature table must be 2-D [D, H], got {:?}",
            rho.dims()
        );
        std::sync::Arc::new(Self::Free(rho))
    }

    /// What a ridge penalty should shrink: the dictionary when there is one,
    /// since penalizing the composed rows would charge every feature for the
    /// same shared vector, and the table itself otherwise.
    #[must_use]
    pub fn ridge_table(&self) -> &Tensor {
        match self {
            Self::Free(rho) => rho,
            Self::Composed { mu, .. } => mu,
        }
    }

    /// Where the parameters live. Taken from the table itself rather than from
    /// the `VarBuilder` a caller happens to hold, so a fixed table built outside
    /// a varmap still places what is derived from it on its own device.
    #[must_use]
    pub fn device(&self) -> &candle_core::Device {
        match self {
            Self::Free(rho) => rho.device(),
            Self::Composed { logits, .. } => logits.device(),
        }
    }

    /// Modules composing each row; 0 for a free table.
    pub fn n_modules(&self) -> usize {
        match self {
            Self::Free(_) => 0,
            Self::Composed { logits, .. } => logits.dims()[1],
        }
    }

    /// The module dictionary `[M, H]`; `None` for a free table.
    #[must_use]
    pub fn dictionary(&self) -> Option<&Tensor> {
        match self {
            Self::Free(_) => None,
            Self::Composed { mu, .. } => Some(mu),
        }
    }
}

#[cfg(test)]
#[path = "feature_embedding_tests.rs"]
mod feature_embedding_tests;
