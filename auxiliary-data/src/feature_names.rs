//! Feature-name kind + canonicalizer hooks for multi-file data
//! alignment.
//!
//! Loaders that union rows across multiple sparse backends
//! ([`crate::data_loading::read_data_on_shared_rows`]) can opt into
//! generous matching by passing a [`FeatureNameKind`] — same row-name
//! canonicalization machinery used by `senna marker` /
//! `FeaturePairGraph::from_edge_list` (via
//! [`matrix_util::membership::GeneIndexResolver`]) but plumbed at the
//! [`data_beans::sparse_io_vector::SparseIoVec`] level so the row
//! intersection itself sees aligned names.
//!
//! The two flavors cover what biology pipelines see in practice:
//!
//! - [`FeatureNameKind::Gene`] for gene rows in scRNA / spatial-RNA
//!   data, where the same gene shows up as `TGFB1`, `ENSG00000105329`,
//!   or `ENSG00000105329_TGFB1` across cohorts;
//! - [`FeatureNameKind::Locus`] for chromosome-coordinate rows in ATAC
//!   / chickpea-style data, where `chr1:1000-2000`, `chr1_1000_2000`,
//!   and `1:1000-2000` should all resolve to the same peak.

use std::sync::Arc;

use data_beans::sparse_io_vector::RowNameCanonicalizer;
use matrix_util::membership::canon_locus;

/// Kind of feature row name + the matching policy that goes with it.
/// Default is [`FeatureNameKind::Exact`] — preserves prior behavior.
#[derive(Clone, Debug, Default)]
pub enum FeatureNameKind {
    /// Exact string match. The default; no canonicalization applied.
    #[default]
    Exact,
    /// Gene names with delimiter-based aliasing. The last component
    /// after splitting on `delim` is treated as canonical, so
    /// `ENSG00000000003_TSPAN6` (file A) and `TSPAN6` (file B) align
    /// when `delim = '_'`. Names without the delimiter pass through
    /// unchanged.
    Gene { delim: char },
    /// Genomic loci. Strips a leading `chr` (case-insensitive) and
    /// folds `:` and `-` separators to `_`, so `chr1:1000-2000` and
    /// `1_1000_2000` resolve to the same row.
    Locus,
}

impl FeatureNameKind {
    /// Canonicalize a single feature name under this kind's rule.
    pub fn canonicalize(&self, name: &str) -> Box<str> {
        match self {
            FeatureNameKind::Exact => name.into(),
            FeatureNameKind::Gene { delim } => {
                // rsplit returns "TGFB1" from "ENSG..._TGFB1" (last token).
                // Empty strings and names without the delimiter pass through.
                name.rsplit(*delim).next().unwrap_or(name).into()
            }
            FeatureNameKind::Locus => canon_locus(name),
        }
    }

    /// Whether this kind applies any canonicalization at all. Callers
    /// can use this to skip the SparseIoVec setter for the common
    /// (no-op) [`FeatureNameKind::Exact`] case.
    pub fn is_exact(&self) -> bool {
        matches!(self, FeatureNameKind::Exact)
    }

    /// Build a `RowNameCanonicalizer` suitable for
    /// [`SparseIoVec::with_row_canonicalizer`]. Returns `None` for
    /// [`FeatureNameKind::Exact`] so callers don't install a no-op
    /// closure.
    pub fn into_canonicalizer(self) -> Option<RowNameCanonicalizer> {
        if self.is_exact() {
            return None;
        }
        Some(Arc::new(move |name: &str| self.canonicalize(name)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_passthrough() {
        let k = FeatureNameKind::Exact;
        assert_eq!(
            k.canonicalize("ENSG00000000003_TSPAN6").as_ref(),
            "ENSG00000000003_TSPAN6"
        );
        assert!(k.is_exact());
        assert!(k.into_canonicalizer().is_none());
    }

    #[test]
    fn gene_takes_last_underscore_component() {
        let k = FeatureNameKind::Gene { delim: '_' };
        assert_eq!(k.canonicalize("ENSG00000000003_TSPAN6").as_ref(), "TSPAN6");
        // Symbol-only inputs survive unchanged.
        assert_eq!(k.canonicalize("TSPAN6").as_ref(), "TSPAN6");
        // Multi-underscore: still take the last token (caller's responsibility
        // to pick a sensible delim if ambiguity matters).
        assert_eq!(k.canonicalize("A_B_C").as_ref(), "C");
        assert!(!k.is_exact());
        assert!(k.into_canonicalizer().is_some());
    }

    #[test]
    fn locus_strips_chr_and_folds_separators() {
        let k = FeatureNameKind::Locus;
        assert_eq!(k.canonicalize("chr1:1000-2000").as_ref(), "1_1000_2000");
        assert_eq!(k.canonicalize("1_1000_2000").as_ref(), "1_1000_2000");
        assert_eq!(k.canonicalize("ChrX:5000-6000").as_ref(), "X_5000_6000");
    }
}
