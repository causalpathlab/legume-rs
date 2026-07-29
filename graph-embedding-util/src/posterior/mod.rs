//! MCMC posterior over the phase-1 **pseudobulk** bilinear model — the principled
//! alternative to its point-estimate NCE-SGD fit, so uncertainty can propagate
//! into the downstream selection and annotation calls.
//!
//! The score is `s(f,p) = ⟨e_f, e_p⟩ + b_f + b_p`. Freezing one side makes the
//! other side's anchors conditionally independent under the Poisson likelihood, so
//! a block is embarrassingly parallel: [`dim_block`] runs one elliptical-slice
//! chain per anchor in a `rayon` `par_iter`, and [`pb_gibbs`] alternates the two
//! sides to sample the joint.
//!
//! ## Likelihood
//!
//! The **Poisson likelihood with a frozen partition** ([`lnpdf::multinomial_ll`],
//! with the anchor intercept profiled out): every observed count enters exactly,
//! and only the rate normalizer `Σ_o exp(s)` is summed over a frozen slate. It
//! factorizes on both sides — unlike the softmax/InfoNCE trainer objective, whose
//! feature sweep couples the positive row with its negatives — so each block is
//! one valid set of per-anchor transitions.
//!
//! ## Pseudobulk, not cells
//!
//! Sampling runs against the pseudobulk axis, which is what phase 1 actually fits
//! (`--phase1-cells-per-pb` defaults to 0, i.e. pure pb). Earlier one-sided,
//! cell-anchored samplers — a SuSiE single-effect gate, a per-node ESS sweep, and
//! their hyper variants — were **retired** rather than kept for comparison: they
//! were unreachable from any CLI, and a second sampler contradicting "MCMC happens
//! at the pb level" is an invitation to wire it back up.
//!
//! ## Exposure
//!
//! The collapse emits per-cell RATES, not counts. [`pb_index`] converts to the
//! count scale (`rate · size_p`, with `log size_p` in the pb bias) — without it a
//! Poisson fit is a quasi-Poisson whose per-dim slab variance collapses toward
//! zero, which is exactly what this module's `σ₀h²` would then report.

pub mod diagnostics;
pub mod dim_block;
pub mod frozen_diag;
pub mod hyper;
pub mod index;
pub mod lnpdf;
pub mod pb_gibbs;
pub mod pb_index;
pub mod run;

pub use diagnostics::{chain_diagnostics, scalar_diagnostics, worst_case, ChainDiag};
pub use dim_block::{dim_block, DimBlockConfig, DimBlockResult};
pub use frozen_diag::{frozen_side_diag, FrozenSideDiag};
pub use hyper::{sample_pi0, HalfCauchyVar};
pub use index::ContrastiveIndex;
pub use lnpdf::{poisson_ll, poisson_lnpdf, AnchorMoment, FrozenSide, NodeTerm};
pub use run::{PosteriorArgs, PosteriorPlan, DEFAULT_PARTITION, DEFAULT_SAMPLES};
