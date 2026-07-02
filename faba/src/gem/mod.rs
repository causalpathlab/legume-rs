//! `faba gem` — joint embedding of gene counts (spliced + unspliced) into one
//! cell/gene space, over the shared `graph_embedding_util` engine.
//!
//! Each feature row `{gene}/count/{spliced|unspliced}` maps to its gene, so a
//! gene's spliced and unspliced tracks embed identically as `β_g` (β-sharing).
//! A single Poisson likelihood on counts: cell **identity** comes from the
//! spliced projection θ (mature mRNA = current state), and the spliced↔unspliced
//! contrast is a **velocity** δ = dir(φ)−dir(θ) on the cell axis (φ = nascent
//! unspliced projection), tracking transcriptional dynamics rather than a second
//! (binomial) likelihood.

pub mod args;
pub mod common;
pub mod sample_id;
