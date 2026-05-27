//! `faba rna-mod-embed` — joint embedding of gene counts + RNA-modification
//! tracks (m6A, A-to-I, poly-A) via a CP-factored feature model and
//! count-weighted NCE with counterfactual negatives.
//!
//! See `faba/temp.md` for the design.

pub mod args;
pub mod feature_table;
pub mod loss;
pub mod manifest;
pub mod model;
pub mod pseudobulk;
pub mod sampling;
pub mod train;
