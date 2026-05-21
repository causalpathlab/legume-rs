//! peak-to-gene subcommand: summary-statistics fine-mapping of cis peak→gene
//! links. Entry point: [`run::run_peak_to_gene`].
//!
//! Pipeline: pseudobulk the matched RNA + ATAC cells, embed peaks (and the
//! projected genes) in a shared ATAC latent space, score each cis peak–gene
//! pair by a log-linear regression z in that space, then fine-map per gene
//! with SuSiE-RSS using the peak–peak correlation (LD) structure.

pub mod run;

mod embed;
mod finemap;
mod input;
mod knockoff;
mod output;

pub use run::{run_peak_to_gene, PeakToGeneArgs};
