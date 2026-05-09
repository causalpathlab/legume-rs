//! `senna gbe` — graph-based embedding (count-NCE) over the
//! (cell, feature) bipartite graph.
//!
//! Modality-agnostic discriminative embedding: each input file
//! contributes its rows to a unified feature axis; cell barcodes
//! union across files. Bilinear `E_f · E_c` scoring with NEG-style
//! count-noise-contrastive estimation, per-file rebalanced sampling
//! and same-file hard negatives. Outputs senna's standard
//! `latent.parquet` / `dictionary.parquet` so `senna {clustering,
//! annotate, layout, plot} --from` work directly.

pub mod coarsen;
pub mod data;
pub mod eval;
pub mod fit;
pub mod loss;
pub mod model;
pub mod training;

pub use fit::{fit_gbe, GbeArgs};
