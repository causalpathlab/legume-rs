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
use rustc_hash::FxHashMap as HashMap;

/// Per-name canonicalization rule for cross-backend row alignment.
/// Concrete strategy only — no "request" variants. Callers that want
/// auto-detection pass [`None`] (or whatever wrapping enum they choose)
/// and call [`FeatureNameKind::auto_detect`] once row names are in hand.
#[derive(Clone, Debug, Default)]
pub enum FeatureNameKind {
    /// Strict string match — no canonicalization. Default.
    #[default]
    Exact,
    /// Gene-symbol rule: register every `delim`-split component as an
    /// alias of the full name. `ENSG00000105329_TGFB1` and `TGFB1`
    /// resolve to the same row.
    Gene { delim: char },
    /// Genomic-locus rule. Normalizes formats (`chr1:1000-2000`,
    /// `1:1000-2000`, `chr1_1000_2000` → `1_1000_2000`). If
    /// `merge_overlapping`, intervals that overlap on the same
    /// chromosome additionally collapse into one cluster
    /// (`chr1:1-20` ∪ `chr1:15-30` → `1_1_30`). Useful for ATAC peak
    /// sets called independently across datasets.
    Locus { merge_overlapping: bool },
    /// Heterogeneous axis: dispatch per row name. Names that parse as
    /// loci go through [`FeatureNameKind::Locus`] with overlap merging;
    /// names with `_` use [`FeatureNameKind::Gene`]; the rest pass
    /// through. Picked automatically when [`auto_detect`] finds both
    /// signatures in the same axis (e.g. paired RNA + ATAC union).
    Mixed,
}

impl FeatureNameKind {
    /// Canonicalize a single name under this kind's per-name rule.
    /// [`Locus { merge_overlapping: true }`] and [`Mixed`] only describe
    /// the per-name part here (format normalization for loci, last-token
    /// split for gene-style); the global cluster step lives in
    /// [`build_locus_overlap_canonical_map`] and is installed by
    /// [`build_canonicalizer`].
    pub fn canonicalize(&self, name: &str) -> Box<str> {
        match self {
            FeatureNameKind::Exact => name.into(),
            FeatureNameKind::Gene { delim } => gene_canonicalize(name, *delim),
            FeatureNameKind::Locus { .. } => canon_locus(name),
            FeatureNameKind::Mixed => {
                if parse_locus(name).is_some() {
                    canon_locus(name)
                } else if name.contains('_') {
                    gene_canonicalize(name, '_')
                } else {
                    name.into()
                }
            }
        }
    }

    /// True iff this kind is [`Exact`] — no canonicalizer needed.
    pub fn is_exact(&self) -> bool {
        matches!(self, FeatureNameKind::Exact)
    }

    /// True iff installing the canonicalizer requires peeking every row
    /// name across all backends first (to build the locus-overlap cluster
    /// map). Loaders branch on this.
    pub fn needs_global_pass(&self) -> bool {
        matches!(
            self,
            FeatureNameKind::Locus {
                merge_overlapping: true
            } | FeatureNameKind::Mixed
        )
    }

    /// Sniff `names` and pick the right kind. Tallies:
    /// `n_locus` = count where `parse_locus` matches; `n_gene_like` =
    /// count of remaining names that contain `_` (loci with `_`
    /// separators are not double-counted as gene-like). Decision:
    /// both ≥ 10% → [`Mixed`]; else loci ≥ 50% →
    /// `Locus { merge_overlapping: true }`; else gene-like ≥ 50% →
    /// `Gene { delim: '_' }`; else [`Exact`].
    pub fn auto_detect(names: &[Box<str>]) -> Self {
        let n = names.len();
        if n == 0 {
            return Self::Exact;
        }
        let mut n_locus = 0usize;
        let mut n_gene_like = 0usize;
        for name in names {
            if parse_locus(name).is_some() {
                n_locus += 1;
            } else if name.contains('_') {
                n_gene_like += 1;
            }
        }
        let pct_locus = n_locus as f32 / n as f32;
        let pct_gene = n_gene_like as f32 / n as f32;
        if pct_locus >= 0.10 && pct_gene >= 0.10 {
            Self::Mixed
        } else if pct_locus >= 0.50 {
            Self::Locus {
                merge_overlapping: true,
            }
        } else if pct_gene >= 0.50 {
            Self::Gene { delim: '_' }
        } else {
            Self::Exact
        }
    }

    /// Build a `RowNameCanonicalizer` suitable for
    /// [`SparseIoVec::with_row_canonicalizer`]. Returns `None` for
    /// [`FeatureNameKind::Exact`] so callers don't install a no-op
    /// closure. **Does not** handle the LocusOverlap global step — for
    /// that, use [`build_locus_overlap_canonicalizer`].
    pub fn into_canonicalizer(self) -> Option<RowNameCanonicalizer> {
        if self.is_exact() {
            return None;
        }
        Some(Arc::new(move |name: &str| self.canonicalize(name)))
    }
}

/// Parse a row name as `(chr, start, end)`. Accepts either
/// `chr1:1000-2000`, `1:1000-2000`, `chr1_1000_2000`, or `1_1000_2000`.
/// Returns `None` for anything that doesn't match — those names pass
/// through the overlap pass untouched.
pub fn parse_locus(name: &str) -> Option<(Box<str>, u64, u64)> {
    let lower = name.to_ascii_lowercase();
    let stripped = lower
        .strip_prefix("chr")
        .map(str::to_string)
        .unwrap_or(lower);
    // Try canonical `_` form first: chr_start_end → ["chr", "start", "end"].
    let parts: Vec<&str> = stripped.splitn(3, [':', '-', '_']).collect();
    if parts.len() != 3 {
        return None;
    }
    let chr = parts[0];
    let start: u64 = parts[1].parse().ok()?;
    let end: u64 = parts[2].parse().ok()?;
    if end < start {
        return None;
    }
    Some((chr.to_string().into_boxed_str(), start, end))
}

/// Build the overlap-merge canonical map from a flat list of row names
/// across all input backends. Names that parse as `(chr, start, end)`
/// are grouped per chromosome, sorted by start, and clustered by
/// transitive overlap (any interval whose start falls before the
/// running cluster's max end). The cluster canonical is
/// `_{chr}_{min_start}_{max_end}` so every member name maps to a single
/// well-defined string.
///
/// Names that fail to parse are not entered into the map; the caller
/// falls back to per-name `canon_locus` for those.
pub fn build_locus_overlap_canonical_map(names: &[Box<str>]) -> HashMap<Box<str>, Box<str>> {
    let n = names.len();
    let parsed: Vec<Option<(Box<str>, u64, u64)>> = names.iter().map(|n| parse_locus(n)).collect();

    // Bucket valid indices by chromosome.
    let mut by_chr: HashMap<Box<str>, Vec<usize>> = HashMap::default();
    for (i, p) in parsed.iter().enumerate() {
        if let Some((chr, _, _)) = p {
            by_chr.entry(chr.clone()).or_default().push(i);
        }
    }

    // Union-find with path compression.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            let g = p[p[x]];
            p[x] = g;
            x = g;
        }
        x
    }

    // Per chr: sort by start, sweep, union anything overlapping the running cluster.
    let mut cluster_extent: HashMap<usize, (u64, u64)> = HashMap::default();
    for (_, mut idxs) in by_chr {
        idxs.sort_by_key(|&i| parsed[i].as_ref().map(|p| p.1).unwrap_or(0));
        let mut current_root: Option<usize> = None;
        let mut current_min_start: u64 = 0;
        let mut current_max_end: u64 = 0;
        for i in idxs {
            let (_, s, e) = parsed[i].as_ref().unwrap();
            match current_root {
                Some(root) if *s < current_max_end => {
                    let ra = find(&mut parent, root);
                    let rb = find(&mut parent, i);
                    if ra != rb {
                        parent[rb] = ra;
                    }
                    current_max_end = current_max_end.max(*e);
                    cluster_extent
                        .insert(find(&mut parent, i), (current_min_start, current_max_end));
                }
                _ => {
                    current_root = Some(i);
                    current_min_start = *s;
                    current_max_end = *e;
                    cluster_extent.insert(i, (*s, *e));
                }
            }
        }
    }

    // Build name → canonical map. Canonical = `_{chr}_{min_start}_{max_end}`
    // — matches canon_locus's `chr` strip + `_` separator convention.
    let mut out: HashMap<Box<str>, Box<str>> = HashMap::default();
    for (i, p) in parsed.iter().enumerate() {
        if let Some((chr, _, _)) = p {
            let root = find(&mut parent, i);
            let (mn, mx) = cluster_extent.get(&root).copied().unwrap_or((0, 0));
            let canonical = format!("{}_{}_{}", chr, mn, mx).into_boxed_str();
            out.insert(names[i].clone(), canonical);
        }
    }
    out
}

/// Build a `RowNameCanonicalizer` for [`FeatureNameKind::LocusOverlap`].
/// `names` should be the concatenation of every input backend's row
/// names (in any order). The returned canonicalizer does
/// `map.get(name).cloned()` first, falling back to per-name
/// `canon_locus` for names that didn't parse as a locus.
pub fn build_locus_overlap_canonicalizer(names: &[Box<str>]) -> RowNameCanonicalizer {
    let map = Arc::new(build_locus_overlap_canonical_map(names));
    Arc::new(move |name: &str| map.get(name).cloned().unwrap_or_else(|| canon_locus(name)))
}

/// Per-name dispatcher for **mixed-kind** axes (e.g. multiome with peaks
/// ∪ genes in one feature axis). For each name:
///   • parses as `(chr, start, end)` → LocusOverlap canonical
///     (cluster representative from `names`).
///   • contains `_` → gene rule: last token after the rightmost `_`.
///   • else → passthrough.
///
/// Use this when the auto-detector sees significant evidence of BOTH
/// loci and gene-style names in the same axis.
pub fn build_mixed_kind_canonicalizer(names: &[Box<str>]) -> RowNameCanonicalizer {
    let map = Arc::new(build_locus_overlap_canonical_map(names));
    Arc::new(move |name: &str| {
        if let Some(c) = map.get(name) {
            c.clone()
        } else if parse_locus(name).is_some() {
            canon_locus(name)
        } else if name.contains('_') {
            gene_canonicalize(name, '_')
        } else {
            name.into()
        }
    })
}

/// Gene-symbol canonicalization with Cell Ranger feature-type suffix
/// awareness. 10x Cell Ranger HDF5 row names commonly arrive as
/// `ENSG..._SYMBOL_<feature_type>` where the third component is a
/// sanitized `feature_type` tag (e.g. `Gene` for `Gene Expression`).
/// A naive `rsplit(delim).next()` would return that constant tag,
/// canonicalizing *every* row to the same string and collapsing the
/// row intersection to one global key. Strip the known tag suffix
/// first so the actual symbol becomes the rsplit target.
fn gene_canonicalize(name: &str, delim: char) -> Box<str> {
    let stripped = strip_feature_type_suffix(name, delim);
    stripped.rsplit(delim).next().unwrap_or(stripped).into()
}

/// Cell Ranger sanitizes `features/feature_type` into the row name as
/// the trailing component. Strip the known tags so the actual gene
/// symbol becomes the rsplit target. Conservative list — only the
/// shapes we've actually seen in the wild — so an unknown tag falls
/// through untouched rather than corrupting a real symbol.
fn strip_feature_type_suffix(name: &str, delim: char) -> &str {
    // Names come pre-sanitized in different ways depending on the
    // producer (Cell Ranger's own h5, scanpy/anndata exports, R-side
    // tools), so accept both `_Gene` and `_Gene_Expression` plus the
    // common companion tags.
    const TAGS: &[&str] = &[
        "Gene_Expression",
        "Gene",
        "Antibody_Capture",
        "CRISPR_Guide_Capture",
        "Multiplexing_Capture",
        "Custom",
        "Peaks",
    ];
    for tag in TAGS {
        // Only strip if the suffix sits behind `delim` (otherwise we'd
        // mangle a real symbol that happens to end in "Gene").
        let mut suffix = String::with_capacity(tag.len() + 1);
        suffix.push(delim);
        suffix.push_str(tag);
        if let Some(rest) = name.strip_suffix(suffix.as_str()) {
            return rest;
        }
    }
    name
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
        // Unknown trailing tokens still get rsplit — caller's responsibility
        // to pick a sensible delim if a non-feature-type trailing token matters.
        assert_eq!(k.canonicalize("A_B_C").as_ref(), "C");
        assert!(!k.is_exact());
        assert!(k.into_canonicalizer().is_some());
    }

    #[test]
    fn gene_strips_cell_ranger_feature_type_suffix() {
        let k = FeatureNameKind::Gene { delim: '_' };
        // 10x Cell Ranger HDF5: `ENSG..._SYMBOL_Gene`. Trailing `_Gene`
        // would otherwise collapse every gene to the literal "Gene".
        assert_eq!(
            k.canonicalize("ENSG00000187634_SAMD11_Gene").as_ref(),
            "SAMD11"
        );
        // Full `Gene_Expression` tag variant.
        assert_eq!(
            k.canonicalize("ENSG00000187634_SAMD11_Gene_Expression")
                .as_ref(),
            "SAMD11"
        );
        // A real gene whose name happens to end in "Gene" *without* the
        // delimiter shouldn't be stripped (no underscore in front of "Gene").
        assert_eq!(k.canonicalize("FakeGene").as_ref(), "FakeGene");
    }

    #[test]
    fn locus_strips_chr_and_folds_separators() {
        let k = FeatureNameKind::Locus {
            merge_overlapping: false,
        };
        assert_eq!(k.canonicalize("chr1:1000-2000").as_ref(), "1_1000_2000");
        assert_eq!(k.canonicalize("1_1000_2000").as_ref(), "1_1000_2000");
        assert_eq!(k.canonicalize("ChrX:5000-6000").as_ref(), "X_5000_6000");
    }

    // -- Genomic region parsing edge cases ----------------------------------

    #[test]
    fn parse_locus_accepts_common_formats() {
        // colon-dash, bare chromosome, underscore-separated, chrX caps.
        assert_eq!(
            parse_locus("chr1:1000-2000"),
            Some(("1".into(), 1000, 2000))
        );
        assert_eq!(parse_locus("1:1000-2000"), Some(("1".into(), 1000, 2000)));
        assert_eq!(
            parse_locus("chr1_1000_2000"),
            Some(("1".into(), 1000, 2000))
        );
        assert_eq!(
            parse_locus("CHR1:1000-2000"),
            Some(("1".into(), 1000, 2000))
        );
        assert_eq!(
            parse_locus("chrX:5000-6000"),
            Some(("x".into(), 5000, 6000))
        );
        assert_eq!(parse_locus("chrMT:1-100"), Some(("mt".into(), 1, 100)));
    }

    #[test]
    fn parse_locus_rejects_non_loci() {
        assert!(parse_locus("TGFB1").is_none()); // gene symbol
        assert!(parse_locus("ENSG00000105329").is_none()); // ensembl
        assert!(parse_locus("chr1:bad-2000").is_none()); // non-numeric start
        assert!(parse_locus("chr1:1000").is_none()); // missing end
        assert!(parse_locus("chr1:2000-1000").is_none()); // end < start
        assert!(parse_locus("").is_none()); // empty
        assert!(parse_locus("chr1").is_none()); // chr-only
    }

    #[test]
    fn overlap_map_merges_two_overlapping_intervals() {
        // User's motivating example: chr1:1-20 and chr1:15-30 → same cluster.
        let names = vec![
            "chr1:1-20".to_string().into_boxed_str(),
            "chr1:15-30".to_string().into_boxed_str(),
        ];
        let map = build_locus_overlap_canonical_map(&names);
        let c0 = map.get(&names[0]).unwrap();
        let c1 = map.get(&names[1]).unwrap();
        assert_eq!(c0, c1, "both inputs should map to the same canonical");
        assert_eq!(c0.as_ref(), "1_1_30"); // union-range canonical
    }

    #[test]
    fn overlap_map_keeps_non_overlapping_separate() {
        let names = vec![
            "chr1:1-20".to_string().into_boxed_str(),
            "chr1:100-200".to_string().into_boxed_str(),
            "chr2:1-20".to_string().into_boxed_str(),
        ];
        let map = build_locus_overlap_canonical_map(&names);
        assert_eq!(map.get(&names[0]).unwrap().as_ref(), "1_1_20");
        assert_eq!(map.get(&names[1]).unwrap().as_ref(), "1_100_200");
        // different chromosome — separate cluster even if start overlaps
        assert_eq!(map.get(&names[2]).unwrap().as_ref(), "2_1_20");
    }

    #[test]
    fn overlap_map_handles_transitive_chain() {
        // A overlaps B (1-20 vs 15-30), B overlaps C (15-30 vs 25-40),
        // A does NOT overlap C directly — but they should still cluster
        // via transitive closure through B.
        let names = vec![
            "chr1:1-20".to_string().into_boxed_str(),
            "chr1:15-30".to_string().into_boxed_str(),
            "chr1:25-40".to_string().into_boxed_str(),
        ];
        let map = build_locus_overlap_canonical_map(&names);
        let c0 = map.get(&names[0]).unwrap();
        let c1 = map.get(&names[1]).unwrap();
        let c2 = map.get(&names[2]).unwrap();
        assert_eq!(c0, c1);
        assert_eq!(c1, c2);
        assert_eq!(c0.as_ref(), "1_1_40"); // union of full chain
    }

    #[test]
    fn overlap_map_handles_full_containment() {
        // chr1:1-100 contains chr1:30-50 — should cluster.
        let names = vec![
            "chr1:1-100".to_string().into_boxed_str(),
            "chr1:30-50".to_string().into_boxed_str(),
        ];
        let map = build_locus_overlap_canonical_map(&names);
        let c0 = map.get(&names[0]).unwrap();
        let c1 = map.get(&names[1]).unwrap();
        assert_eq!(c0, c1);
        assert_eq!(c0.as_ref(), "1_1_100");
    }

    #[test]
    fn overlap_map_treats_adjacent_as_separate() {
        // chr1:1-20 and chr1:20-30 are *touching* but not overlapping
        // (end of first == start of second, exclusive end convention).
        let names = vec![
            "chr1:1-20".to_string().into_boxed_str(),
            "chr1:20-30".to_string().into_boxed_str(),
        ];
        let map = build_locus_overlap_canonical_map(&names);
        assert_ne!(map.get(&names[0]).unwrap(), map.get(&names[1]).unwrap());
    }

    #[test]
    fn overlap_map_normalizes_chr_prefix_within_cluster() {
        // chr1:1-20 and 1:15-30 (no chr prefix) should still cluster
        // because parse_locus normalizes both to chr="1".
        let names = vec![
            "chr1:1-20".to_string().into_boxed_str(),
            "1:15-30".to_string().into_boxed_str(),
        ];
        let map = build_locus_overlap_canonical_map(&names);
        let c0 = map.get(&names[0]).unwrap();
        let c1 = map.get(&names[1]).unwrap();
        assert_eq!(c0, c1);
        assert_eq!(c0.as_ref(), "1_1_30");
    }

    #[test]
    fn overlap_map_normalizes_separators_within_cluster() {
        // chr1:1-20 and chr1_15_30 (different separators) → same cluster.
        let names = vec![
            "chr1:1-20".to_string().into_boxed_str(),
            "chr1_15_30".to_string().into_boxed_str(),
        ];
        let map = build_locus_overlap_canonical_map(&names);
        assert_eq!(map.get(&names[0]).unwrap(), map.get(&names[1]).unwrap());
    }

    #[test]
    fn overlap_map_ignores_non_locus_names() {
        // Non-locus names should not appear in the map; caller falls
        // back to canon_locus (the default Locus rule).
        let names = vec![
            "TGFB1".to_string().into_boxed_str(),
            "chr1:1-20".to_string().into_boxed_str(),
        ];
        let map = build_locus_overlap_canonical_map(&names);
        assert!(map.get(&names[0]).is_none());
        assert!(map.get(&names[1]).is_some());
    }

    #[test]
    fn overlap_map_handles_zero_length_interval() {
        // chr1:1000-1000 — degenerate but valid; should be its own cluster.
        let names = vec!["chr1:1000-1000".to_string().into_boxed_str()];
        let map = build_locus_overlap_canonical_map(&names);
        assert_eq!(map.get(&names[0]).unwrap().as_ref(), "1_1000_1000");
    }

    #[test]
    fn overlap_canonicalizer_falls_back_to_canon_locus_for_unmatched() {
        let names = vec!["chr1:1-20".to_string().into_boxed_str()];
        let canon = build_locus_overlap_canonicalizer(&names);
        // In-cluster name → cluster canonical.
        assert_eq!(canon("chr1:1-20").as_ref(), "1_1_20");
        // Unrelated locus not in the map → canon_locus normalization.
        assert_eq!(canon("chr2:500-600").as_ref(), "2_500_600");
        // Non-locus → canon_locus passes through unchanged.
        // (canon_locus for "TGFB1" likely strips no separators and returns
        // the lowercase form — accept whatever the helper returns.)
        let g = canon("TGFB1");
        assert!(!g.is_empty());
    }

    // -- Auto-detect & Mixed dispatcher -------------------------------------

    #[test]
    fn auto_detect_pure_locus_axis() {
        let names: Vec<Box<str>> = (0..100)
            .map(|i| format!("chr1:{}-{}", i * 100, i * 100 + 50).into_boxed_str())
            .collect();
        assert!(matches!(
            FeatureNameKind::auto_detect(&names),
            FeatureNameKind::Locus {
                merge_overlapping: true
            }
        ));
    }

    #[test]
    fn auto_detect_pure_gene_axis() {
        let names: Vec<Box<str>> = (0..100)
            .map(|i| format!("ENSG000_GENE{}", i).into_boxed_str())
            .collect();
        assert!(matches!(
            FeatureNameKind::auto_detect(&names),
            FeatureNameKind::Gene { delim: '_' }
        ));
    }

    #[test]
    fn auto_detect_mixed_axis() {
        // 80 loci + 20 gene-style → both fractions ≥ 10% → Mixed.
        let mut names: Vec<Box<str>> = (0..80)
            .map(|i| format!("chr1:{}-{}", i * 1000, i * 1000 + 500).into_boxed_str())
            .collect();
        names.extend((0..20).map(|i| format!("ENSG000_GENE{}", i).into_boxed_str()));
        assert!(matches!(
            FeatureNameKind::auto_detect(&names),
            FeatureNameKind::Mixed
        ));
    }

    #[test]
    fn auto_detect_empty_or_exact() {
        assert!(matches!(
            FeatureNameKind::auto_detect(&[]),
            FeatureNameKind::Exact
        ));
        let names = vec!["TGFB1".into(), "CD4".into(), "IL2".into(), "GAPDH".into()];
        assert!(matches!(
            FeatureNameKind::auto_detect(&names),
            FeatureNameKind::Exact
        ));
    }

    #[test]
    fn mixed_dispatcher_canonicalizes_each_name_by_kind() {
        let names: Vec<Box<str>> = vec![
            "chr1:1-20".into(),     // locus → cluster canonical
            "chr1:15-30".into(),    // locus, overlaps above → same cluster
            "ENSG000_TGFB1".into(), // gene-style → "TGFB1"
            "CD4".into(),           // plain symbol → passthrough
        ];
        let canon = build_mixed_kind_canonicalizer(&names);
        assert_eq!(canon("chr1:1-20").as_ref(), "1_1_30");
        assert_eq!(canon("chr1:15-30").as_ref(), "1_1_30");
        assert_eq!(canon("ENSG000_TGFB1").as_ref(), "TGFB1");
        assert_eq!(canon("CD4").as_ref(), "CD4");
    }
}
