//! Posthoc directional ligand→receptor activity test per link community.
//!
//! Consumes the output of `pinto lc` (edge→community assignments) and a
//! user-supplied directional ligand→receptor list, and tests whether each LR
//! pair shows elevated activity within each community. The statistic is
//! conditional entropy `H(R_receiver | L_sender)` across the edges of each
//! (batch × community × connected component); null pairs are gene-swap decoys
//! matched on marginal expression and global Moran's I over the same edge
//! graph.

pub mod args;
pub mod entropy;
pub mod fit;
pub mod io;
pub mod matcher;
pub mod moran;
pub mod outputs;

pub use args::SrtLrActivityArgs;
pub use fit::fit_srt_lr_activity;
