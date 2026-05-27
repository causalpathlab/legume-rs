//! `faba rna-mod-embed` — joint embedding of gene counts + RNA-modification
//! tracks (m6A, A-to-I, poly-A). For each (gene g, modality m) the
//! feature embedding is
//!
//!     e_{g,m} = ρ_g + Σ_k z_{g,k} · Q_{k, m, :},
//!
//! i.e. a per-gene baseline ρ_g plus a z_g-weighted pool over K
//! modality-specific signature vectors Q_{k, m, :}. Training uses
//! count-weighted NCE with counterfactual negatives.
//!
//! See `faba/temp.md` for the design.

pub mod args;
pub mod common;
pub mod feature_table;
pub mod loss;
pub mod manifest;
pub mod model;
pub mod pseudobulk;
pub mod sampling;
pub mod train;
