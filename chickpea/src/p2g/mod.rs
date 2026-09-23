//! peak-to-gene: gene-centric cis-regulatory links for paired RNA + ATAC.
//! Entry point: [`run::run_peak_to_gene`].
//!
//! Peaks are aggregated onto genes through ABC contact weights and embedded as
//! a second track of the gene axis next to RNA ([`two_track`]). Peak rows are
//! folded in against the pseudobulk embeddings ([`peak_foldin`]), and each
//! gene attends over its cis peaks ([`attention`]); the shares are the links,
//! reported overall and per cell cluster ([`context`], [`workflow`]).

pub mod attention;
pub mod cis;
pub mod context;
pub mod gene_track;
pub mod input;
pub mod peak_foldin;
pub mod run;
pub mod tracks;
pub mod two_track;
pub mod workflow;

pub use run::{run_peak_to_gene, PeakToGeneArgs};
