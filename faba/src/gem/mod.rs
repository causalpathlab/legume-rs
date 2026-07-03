//! `faba gem` — joint embedding of gene counts (spliced + unspliced) into one
//! cell/gene space, over the shared `graph_embedding_util` engine.
//!
//! Each feature row `{gene}/count/{spliced|unspliced}` maps to its gene, so a
//! gene's spliced and unspliced tracks embed identically as `β_g` (β-sharing).
//! A single Poisson likelihood on counts: cell **identity** is the spliced
//! projection θ (written raw), and the **velocity** is the raw analytic increment
//! δ — a Poisson-MAP shift fit to the unspliced edges with θ held fixed (‖δ‖ =
//! speed) — tracking the spliced↔unspliced dynamics rather than a second (binomial)
//! likelihood.

pub mod args;
pub mod common;
pub mod sample_id;
