//! peak-to-gene subcommand: summary-statistics fine-mapping of cis peak→gene
//! links. Entry point: [`run::run_peak_to_gene`].
//!
//! Pipeline: pseudobulk the matched RNA + ATAC cells, embed peaks (and the
//! projected genes) in a shared ATAC latent space, score each cis peak–gene
//! pair by a log-linear regression z in that space, then fine-map per gene
//! with SuSiE-RSS using the peak–peak correlation (LD) structure.

pub mod cis;
pub mod gene_track;
pub mod peak_foldin;
pub mod run;
pub mod tracks;
pub mod two_track;

mod embed;
mod finemap;
mod input;
mod knockoff;
mod output;
mod tmle;

pub use run::{run_peak_to_gene, PeakToGeneArgs};
