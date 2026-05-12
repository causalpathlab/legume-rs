//! `pinto gbe` — count-NCE bipartite-graph embedding for spatial
//! transcriptomics. Thin wrapper around `graph-embedding-util` that
//! adds SRT-specific data loading and a spatial-KNN side output.

pub mod args;
pub mod cluster;
pub mod fit;

pub use args::SrtGbeArgs;
pub use fit::fit_srt_gbe;
