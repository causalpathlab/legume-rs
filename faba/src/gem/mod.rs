//! `faba gem` — joint embedding of gene counts (spliced + unspliced) into one
//! cell/gene space, over the shared `graph_embedding_util` engine.
//!
//! Each feature row `{gene}/count/{spliced|unspliced}` maps to its gene, so a
//! gene's spliced and unspliced tracks embed identically as `β_g` (β-sharing);
//! the splice deviation is recovered post-hoc on the cell axis by the dual
//! phase-2 projection (`{out}.axis_delta.parquet`).
//!
//! m6A is co-embedded as an optional second modality: a coverage-conditioned
//! binomial arm (methylated `M` vs unmethylated `U` reads) sharing the cell
//! axis, plugged into geu's generic `LossArm` / `PerCellAuxTerm` seams (see
//! [`m6a`]). A-to-I / poly-A are not modelled here.

pub mod args;
pub mod common;
pub mod m6a;
pub mod sample_id;
