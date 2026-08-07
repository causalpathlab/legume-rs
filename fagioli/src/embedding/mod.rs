//! Embedding of GWAS summary statistics: the trait-by-variant bipartite graph
//! factored through a shared low-dimensional program space.
//!
//! The model is low-rank multi-trait RSS. Writing the observed z-scores as
//! `Z = R B Λ_N + E` with `E ~ MN(0, R, Ω)`, the embedding assumes the *direct*
//! (LD-free) effect matrix factors as `B = U V'`, so `β_gt = <u_g, v_t>`.
//!
//! What distinguishes it is factoring `B` rather than `Z`: the top components of
//! `Z` are dominated by LD blocks and by sample overlap, which is why they need
//! deconvolving before any of it means anything.
//!
//! LD is nuisance, not data — it is the sampling covariance of the graph being
//! embedded, not an observation of biology. It therefore belongs in the noise
//! model, never in an auxiliary loss or a smoother over an LD graph, either of
//! which would force `u_g` to re-encode exactly what whitening removes.
//!
//! [`diagnostics`] is deliberately the first thing here: it decides whether the
//! fitted geometry is real before there is a model to defend.

pub mod diagnostics;

pub use diagnostics::{
    compare_geometry, cross_trait_moments, estimate_trait_geometry, CrossTraitMoments,
    GeometryVerdict, TraitGeometry,
};
