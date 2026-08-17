//! Benjamini–Hochberg FDR correction.
//!
//! The implementation moved to [`matrix_util::hypothesis`], which is where the
//! workspace's other callers (trajectory ordering, association testing, droplet
//! QC) reach it. It lived here and in one other crate as two independent copies
//! that agreed numerically by coincidence and disagreed on `NaN` handling; this
//! alias keeps `bh_fdr` available to the enrichment pipeline's call sites while
//! there is only one implementation to keep correct.

pub use matrix_util::hypothesis::benjamini_hochberg as bh_fdr;
