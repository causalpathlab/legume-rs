//! Faceted "Miami plot" figure for `faba pileup` (SVG/PDF).
//!
//! A mirrored Manhattan per gene, faceted by cell type: one stacked panel
//! per cell type sharing the genomic x-axis. Within a panel, epi sites
//! (m6A / A-to-I / APA) rise as lollipops on top, read depth fills
//! downward on the bottom, and a GTF-derived gene model sits in between.
//!
//! Orchestration lives in [`crate::site_analysis::pileup::run_pileup`]
//! (the matrix reader is tightly coupled to its `Selector`); this module
//! owns the binning grid, the BAM depth track, the gene model, and the
//! SVG rendering.
//!
//! - [`bin`]: shared `BinEdges` grid + robust-max scaling.
//! - [`depth`]: per-cell-type read depth from BAM.
//! - [`genemodel`]: GTF -> exon/intron/strand SVG band.
//! - [`render`]: faceted SVG assembly + PNG/PDF output.

pub mod bin;
pub mod depth;
pub mod genemodel;
pub mod render;
