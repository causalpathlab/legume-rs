//! MCMC posterior sampling over the bilinear embedding — the principled
//! alternative to the point-estimate NCE-SGD fit, so uncertainty can propagate
//! into the downstream annotation / selection calls.
//!
//! The score is `s(f,c) = ⟨e_f, e_c⟩ + b_f + b_c`. Freezing one side makes the
//! other side's rows conditionally independent under the Poisson likelihood, so
//! a sweep is embarrassingly parallel: [`sweep::sweep_side`] runs one elliptical-
//! slice chain per node (reusing `mcmc_util`'s `EssSampler`), in a `rayon`
//! `par_iter` — the same idiom as the phase-2 `cell_projection::project_cells`.
//!
//! ## Likelihood
//!
//! The default is the **Poisson likelihood with a sampled partition**
//! ([`lnpdf::poisson_lnpdf`]): every observed count enters exactly, and only the
//! rate normalizer `Σ_o exp(s)` is Monte-Carlo'd over a **frozen** other-side
//! slate (`partition` + `partition_scale = |pool|/K`). It factorizes on both
//! sides — unlike the softmax/InfoNCE trainer objective, whose feature sweep
//! couples the positive row with its negatives — so every sweep is one valid set
//! of per-node transitions.
//!
//! ## Scope of this module (Stage A)
//!
//! The core sampler + a planted-truth validation. The SuSiE feature gate
//! (per-gene PIP), the hierarchical hyperpriors, the pipeline wiring, and the GPU
//! batched executor are layered on top in later stages — see the plan.

pub mod diagnostics;
pub mod gate;
pub mod hyper;
pub mod hyper_ss;
pub mod hyper_sweep;
pub mod index;
pub mod lnpdf;
pub mod sweep;

pub use diagnostics::{chain_diagnostics, scalar_diagnostics, worst_case, ChainDiag};
pub use gate::{gate_posterior, GateConfig, GenePosterior};
pub use hyper::{sample_pi0, HalfCauchyVar};
pub use hyper_ss::{hyper_ss, HyperSsConfig, HyperSsResult};
pub use hyper_sweep::{hyper_sweep, HyperSweepConfig, HyperSweepResult};
pub use index::{build_gene_index, ContrastiveIndex};
pub use lnpdf::{poisson_ll, poisson_lnpdf, FrozenSide, NodeTerm};
pub use sweep::{sweep_side, NodePosterior, SweepConfig};
