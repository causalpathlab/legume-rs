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
//! support, so a module no feature carries mass on receives no gradient and can
//! never earn its way in. Capacity is therefore added by SPLITTING a loaded
//! module, never by appending a fresh one.

use crate::fast_index::gather_rows;
use crate::nn::layers::sparsemax;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Registered names, so a caller owning the `VarMap` can reach them without
/// hard-coding a string this module chose.
pub const LOGITS_VAR_NAME: &str = "modules.logits";
pub const MU_VAR_NAME: &str = "modules.mu";

/// Membership logits `[D, M]` and the module dictionary `[M, H]`, both learned.
pub struct FeatureEmbedding {
    logits: Tensor,
    mu: Tensor,
    n_features: usize,
    n_modules: usize,
    embedding_dim: usize,
}

impl FeatureEmbedding {
    /// Register both parameters. The membership starts flat, which sparsemax
    /// maps to uniform over the whole support: no module is preferred and none
    /// is excluded, so the data alone decides. This is also what a feature the
    /// axis gains later is given, which is why it is a constant rather than a
    /// draw.
    pub fn new(
        n_features: usize,
        n_modules: usize,
        embedding_dim: usize,
        vs: VarBuilder,
    ) -> Result<Self> {
        let logits = vs.get_with_hints(
            (n_features, n_modules),
            LOGITS_VAR_NAME,
            candle_nn::Init::Const(0.0),
        )?;
        let mu = vs.get_with_hints(
            (n_modules, embedding_dim),
            MU_VAR_NAME,
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        Ok(Self {
            logits,
            mu,
            n_features,
            n_modules,
            embedding_dim,
        })
    }

    /// Every feature's membership `[D, M]`, rows on the simplex with exact
    /// zeros off it.
    pub fn membership(&self) -> Result<Tensor> {
        sparsemax(&self.logits)
    }

    /// The rows named by `ids`, composed without forming the whole table.
    ///
    /// The cheap path, and the one training uses: `[n, M]` gathered, projected,
    /// then one matmul against `μ`. Equal to [`Self::full`] followed by a
    /// select, which is what lets the encoder work in module space.
    pub fn gather(&self, ids: &Tensor) -> Result<Tensor> {
        sparsemax(&gather_rows(&self.logits, ids)?)?.matmul(&self.mu)
    }

    /// `α μᵀ` — a topic-by-module table, so a decoder can reach the feature axis
    /// through the modules instead of through a `[D, H]` product.
    pub fn topic_by_module(&self, alpha: &Tensor) -> Result<Tensor> {
        alpha.matmul(&self.mu.t()?.contiguous()?)
    }

    /// The whole `[D, H]` table. For export and diagnostics: forming it is the
    /// cost this parameterization exists to avoid, so it does not belong on a
    /// training path.
    pub fn full(&self) -> Result<Tensor> {
        self.membership()?.matmul(&self.mu)
    }

    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    #[must_use]
    pub fn n_modules(&self) -> usize {
        self.n_modules
    }

    #[must_use]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// The module dictionary `[M, H]`.
    #[must_use]
    pub fn dictionary(&self) -> &Tensor {
        &self.mu
    }
}

#[cfg(test)]
#[path = "feature_embedding_tests.rs"]
mod feature_embedding_tests;
