//! `faba gem` — joint embedding of gene counts + RNA-modification
//! tracks (m6A, A-to-I, poly-A). A feature row has identity
//! `(gene g, modality m, region r)` and embeds as a base gene vector β_g
//! deviated by an exp gate:
//!
//!     AGG  ({g}/AGG):      e_f = β_g
//!     comp ({g}/{m}/{c}):  e_f = β_g ⊙ exp(logdev_{g,m,r})
//!     logdev_{g,m,r}       = Σ_k z_{g,k} · δ_{k,m,:} + γ_{m,r,:}
//!
//! `δ_{k,m,:}` is the program×modality deviation **direction** (full
//! H-vector — a modification can move the gene in a new direction, not
//! just rescale β) and `γ_{m,r,:}` is an additive per-(modality, region)
//! offset, where region = a transcript-position bin. Together they
//! resolve multiple modification components per gene. Training uses
//! count-weighted NCE with counterfactual negatives (random,
//! swap-gene-mode, swap-modality).
//!
//! See `faba/temp.md` for the design.

pub mod args;
pub mod common;
pub mod feature_table;
pub mod gene_weight;
pub mod loss;
pub mod manifest;
pub mod model;
pub mod pseudobulk;
pub mod region;
pub mod sampling;
pub mod topics;
pub mod train;

#[cfg(test)]
mod sim_test;
