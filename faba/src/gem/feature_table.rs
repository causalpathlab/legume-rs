//! Parses faba's `{gene_key}/{modality}/{detail}` row names, stratifies
//! input rows into AGG / count-comp / modifier-comp / site, and builds
//! the compact (gene_id, modality_id) row identity used by the model.
//!
//! Per the embedding rule `e_f = β_g ⊙ exp(Σ_k z_{g,k}·δ_{k,m,:} + γ_{m,r,:})`,
//! every component row within a (g, m) pair shares the same e_f and the
//! same per-row bias b_f. So the model never sees raw row indices —
//! everything goes through (gene_id, Option<modality_id>), where `None`
//! tags the virtual AGG row.

use rustc_hash::FxHashMap;

use super::region::RegionMap;

/// Parsed feature-name parts. Owned `Box<str>` so the parsed value can
/// outlive the source string slice.
#[derive(Clone, Debug)]
pub struct FeatureKey {
    pub gene: Box<str>,
    pub modality: Box<str>,
    pub detail: Box<str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowStratum {
    /// `{gene}/count/{spliced|unspliced|...}`.
    CountComp,
    /// `{gene}/{m}/{component}` where `m ≠ count` and the detail is not
    /// a `chr:pos` site identifier.
    ModifierComp,
    /// `{gene}/{m}/{chr:pos}` — excluded from the embedding loss in v1.
    Site,
}

/// Split `"gene/modality/detail"` into its three parts. Returns `None`
/// for any row that doesn't have exactly two `/` separators.
pub fn parse_feature_name(name: &str) -> Option<FeatureKey> {
    let mut iter = name.splitn(3, '/');
    let gene = iter.next()?;
    let modality = iter.next()?;
    let detail = iter.next()?;
    if gene.is_empty() || modality.is_empty() || detail.is_empty() {
        return None;
    }
    Some(FeatureKey {
        gene: gene.into(),
        modality: modality.into(),
        detail: detail.into(),
    })
}

/// Detail strings shaped like `chr1:12345` mark per-site rows that we
/// exclude from the embedding loss. Strict shape check:
///   - exactly one `:` separator,
///   - the part before starts with `chr` (case-insensitive),
///   - the part after is non-empty and entirely ASCII digits.
///
/// This deliberately rejects component names like `exon:2` or
/// `donor:acceptor:3` that happen to contain a colon — those would be
/// silently dropped from training if we accepted any `<word>:<digit>`.
pub fn is_site_detail(detail: &str) -> bool {
    let mut parts = detail.splitn(2, ':');
    let Some(prefix) = parts.next() else {
        return false;
    };
    let Some(suffix) = parts.next() else {
        return false;
    };
    // Reject if there's a second colon — sites have exactly one.
    if suffix.contains(':') {
        return false;
    }
    if !prefix.to_ascii_lowercase().starts_with("chr") {
        return false;
    }
    !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
}

/// Stratum classification, given a parsed feature name. The `count`
/// modality label is what `data-beans-sim faba` (and `faba genes`) emit
/// for spliced/unspliced count rows.
pub fn classify(key: &FeatureKey) -> RowStratum {
    if &*key.modality == "count" {
        RowStratum::CountComp
    } else if is_site_detail(&key.detail) {
        RowStratum::Site
    } else {
        RowStratum::ModifierComp
    }
}

/// Parse the GMM component index from a modifier row's detail. The
/// mixture pipeline emits `{gene}/{mod}/{new_idx}` with a plain integer
/// component index, so the common case is `"0".parse()`. Returns `None`
/// for non-integer details (e.g. `comp_1`, `spliced`), which then fall
/// back to region 0 at lookup time.
pub fn parse_component_idx(detail: &str) -> Option<u32> {
    detail.parse::<u32>().ok()
}

/// Compact-id view of all input rows.
pub struct FeatureTable {
    /// Stratum per input row (`None` ⇒ unparseable / dropped).
    pub strata: Vec<Option<RowStratum>>,

    /// `gene_id` → gene_key.
    pub gene_names: Vec<Box<str>>,
    /// `modality_id` → modality name. Slot 0 is always `"count"`.
    pub modality_names: Vec<Box<str>>,

    /// Number of transcript-position region bins R. `region_id ∈ 0..R`.
    pub n_regions: usize,

    /// Per input row: `(gene_id, modality_id)` once classified into a
    /// CountComp or ModifierComp stratum. `None` for Site or
    /// unparseable rows.
    pub row_gene: Vec<Option<u32>>,
    pub row_modality: Vec<Option<u32>>,
    /// Per modifier-comp row: GMM component index parsed from the detail
    /// (`None` for count-comp / non-integer details).
    pub row_component: Vec<Option<u32>>,
    /// Per modifier-comp row: transcript-position region bin, resolved
    /// via the `RegionMap` from `(gene, modality, component)`. `None`
    /// for count-comp / Site / unparseable rows; satellite rows that
    /// miss the annotation default to region 0.
    pub row_region: Vec<Option<u32>>,

    /// Input row IDs in each loss-participating stratum.
    pub count_comp_rows: Vec<u32>,
    pub modifier_comp_rows: Vec<u32>,
    /// `modifier_rows_by_modality[m]` is the modifier-comp row IDs for
    /// modality `m`. Modality 0 (count) is empty by convention.
    pub modifier_rows_by_modality: Vec<Vec<u32>>,

    /// `measured[g][m]` = true iff gene `g` has at least one input row in
    /// modality `m` — a count-comp row for the count modality (slot 0), a
    /// modifier-comp row otherwise. sampling.rs reads only the modifier
    /// columns (slot ≥ 1) to restrict swap-z draws to genes with real
    /// measurement in modality m; the count column exists so the exported
    /// `measured_mask.parquet` reflects count coverage rather than reading
    /// as all-zero.
    pub measured: Vec<Vec<bool>>,
}

impl FeatureTable {
    pub fn n_genes(&self) -> usize {
        self.gene_names.len()
    }

    pub fn n_modalities(&self) -> usize {
        self.modality_names.len()
    }

    /// Classify the unified feature axis into row strata + compact ids.
    /// Modality 0 is forced to be `"count"` so AGG rows always reduce to
    /// "sum of modality-0 components". `regions` resolves each modifier
    /// component to a transcript-position region bin.
    pub fn build(feature_names: &[Box<str>], regions: &RegionMap) -> Self {
        let n = feature_names.len();

        let mut gene_to_id: FxHashMap<Box<str>, u32> = FxHashMap::default();
        let mut gene_names: Vec<Box<str>> = Vec::new();
        let mut modality_to_id: FxHashMap<Box<str>, u32> = FxHashMap::default();
        let mut modality_names: Vec<Box<str>> = Vec::new();
        // Reserve slot 0 for "count".
        modality_to_id.insert("count".into(), 0);
        modality_names.push("count".into());

        let mut strata = Vec::with_capacity(n);
        let mut row_gene = Vec::with_capacity(n);
        let mut row_modality = Vec::with_capacity(n);
        let mut row_component: Vec<Option<u32>> = Vec::with_capacity(n);
        let mut row_region: Vec<Option<u32>> = Vec::with_capacity(n);

        let mut count_comp_rows = Vec::new();
        let mut modifier_comp_rows = Vec::new();

        for (row, name) in feature_names.iter().enumerate() {
            let Some(key) = parse_feature_name(name) else {
                strata.push(None);
                row_gene.push(None);
                row_modality.push(None);
                row_component.push(None);
                row_region.push(None);
                continue;
            };

            let gene_id = match gene_to_id.get(&key.gene) {
                Some(&id) => id,
                None => {
                    let id = gene_names.len() as u32;
                    gene_to_id.insert(key.gene.clone(), id);
                    gene_names.push(key.gene.clone());
                    id
                }
            };

            let stratum = classify(&key);
            match stratum {
                RowStratum::CountComp => {
                    strata.push(Some(stratum));
                    row_gene.push(Some(gene_id));
                    row_modality.push(Some(0));
                    row_component.push(None);
                    row_region.push(None);
                    count_comp_rows.push(row as u32);
                }
                RowStratum::ModifierComp => {
                    let modality_id = match modality_to_id.get(&key.modality) {
                        Some(&id) => id,
                        None => {
                            let id = modality_names.len() as u32;
                            modality_to_id.insert(key.modality.clone(), id);
                            modality_names.push(key.modality.clone());
                            id
                        }
                    };
                    let component = parse_component_idx(&key.detail);
                    // Resolve the transcript-position region from the
                    // component annotation; un-annotated components fall
                    // back to region 0 (a valid γ index, masked nowhere).
                    let region = regions.lookup(&key.gene, &key.modality, component.unwrap_or(0));
                    strata.push(Some(stratum));
                    row_gene.push(Some(gene_id));
                    row_modality.push(Some(modality_id));
                    row_component.push(component);
                    row_region.push(Some(region));
                    modifier_comp_rows.push(row as u32);
                }
                RowStratum::Site => {
                    // Site modalities still register so that downstream
                    // diagnostic code can ask "did modality m appear at
                    // all?", even if no comp rows did.
                    if !modality_to_id.contains_key(&key.modality) {
                        let id = modality_names.len() as u32;
                        modality_to_id.insert(key.modality.clone(), id);
                        modality_names.push(key.modality.clone());
                    }
                    strata.push(Some(stratum));
                    row_gene.push(Some(gene_id));
                    row_modality.push(modality_to_id.get(&key.modality).copied());
                    row_component.push(None);
                    row_region.push(None);
                }
            }
        }

        let n_genes = gene_names.len();
        let n_modalities = modality_names.len();

        // measured[g][m]: any input row for (g, m)? Count-comp rows mark
        // the count modality (slot 0); modifier-comp rows mark the rest.
        let mut measured = vec![vec![false; n_modalities]; n_genes];
        for &row in &count_comp_rows {
            if let Some(g) = row_gene[row as usize] {
                measured[g as usize][0] = true;
            }
        }
        for &row in &modifier_comp_rows {
            if let (Some(g), Some(m)) = (row_gene[row as usize], row_modality[row as usize]) {
                measured[g as usize][m as usize] = true;
            }
        }

        // modifier_rows_by_modality[m]
        let mut modifier_rows_by_modality: Vec<Vec<u32>> = vec![Vec::new(); n_modalities];
        for &row in &modifier_comp_rows {
            if let Some(m) = row_modality[row as usize] {
                modifier_rows_by_modality[m as usize].push(row);
            }
        }

        Self {
            strata,
            n_regions: regions.n_regions,
            gene_names,
            modality_names,
            row_gene,
            row_modality,
            row_component,
            row_region,
            count_comp_rows,
            modifier_comp_rows,
            modifier_rows_by_modality,
            measured,
        }
    }

    /// Genes that have a modifier-comp row in modality `m`. Returned in
    /// ascending order, suitable for swap-z's `{g' : measured(g',m)=1}`
    /// restriction.
    pub fn measured_genes_for_modality(&self, modality: u32) -> Vec<u32> {
        (0..self.n_genes() as u32)
            .filter(|&g| self.measured[g as usize][modality as usize])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_three_part_names() {
        let k = parse_feature_name("ENSG001_BRCA2/m6A/comp_1").unwrap();
        assert_eq!(&*k.gene, "ENSG001_BRCA2");
        assert_eq!(&*k.modality, "m6A");
        assert_eq!(&*k.detail, "comp_1");
    }

    #[test]
    fn rejects_malformed_names() {
        assert!(parse_feature_name("foo").is_none());
        assert!(parse_feature_name("foo/bar").is_none());
        assert!(parse_feature_name("/m6A/comp_1").is_none());
        assert!(parse_feature_name("g//comp_1").is_none());
    }

    #[test]
    fn site_detail_detection() {
        assert!(is_site_detail("chr1:12345"));
        assert!(is_site_detail("chrX:999"));
        assert!(is_site_detail("CHR1:42")); // case-insensitive prefix
        assert!(!is_site_detail("comp_1"));
        assert!(!is_site_detail("spliced"));
        assert!(!is_site_detail(":12345")); // empty chromosome
        assert!(!is_site_detail("chr1:")); // empty pos
        assert!(!is_site_detail("chr1:abc")); // non-digit pos
                                              // Now strict: only `chr…:digits` is a site. Reject any other
                                              // colon-containing component name (the previous loose rule
                                              // silently dropped these).
        assert!(!is_site_detail("exon:2"));
        assert!(!is_site_detail("donor:42"));
        assert!(!is_site_detail("peak_3:bin_1"));
        assert!(!is_site_detail("chr1:1:2")); // multiple colons
    }

    #[test]
    fn stratification_basics() {
        let names: Vec<Box<str>> = [
            "geneA/count/spliced",
            "geneA/count/unspliced",
            "geneA/m6A/comp_1",
            "geneA/m6A/comp_2",
            "geneA/m6A/chr1:1234",
            "geneB/count/spliced",
            "geneB/A2I/comp_1",
            "geneC/pA/comp_1",
        ]
        .iter()
        .map(|s| (*s).into())
        .collect();

        let tab = FeatureTable::build(&names, &RegionMap::empty(5));
        assert_eq!(tab.n_genes(), 3); // geneA, geneB, geneC
        assert!(tab.modality_names.iter().any(|m| &**m == "count"));
        assert!(tab.modality_names.iter().any(|m| &**m == "m6A"));
        assert!(tab.modality_names.iter().any(|m| &**m == "A2I"));
        assert!(tab.modality_names.iter().any(|m| &**m == "pA"));

        // Two count-comp rows for geneA, one for geneB.
        assert_eq!(tab.count_comp_rows.len(), 3);
        // Two modifier-comp rows for geneA m6A (excluding the chr:pos), one A2I geneB, one pA geneC.
        assert_eq!(tab.modifier_comp_rows.len(), 4);

        let gene_a = tab.gene_names.iter().position(|g| &**g == "geneA").unwrap() as u32;
        let gene_c = tab.gene_names.iter().position(|g| &**g == "geneC").unwrap() as u32;

        // measured mask matches reality.
        let m6a = tab
            .modality_names
            .iter()
            .position(|m| &**m == "m6A")
            .unwrap();
        assert!(tab.measured[gene_a as usize][m6a]);
        assert!(!tab.measured[gene_c as usize][m6a]);

        // Count modality (slot 0) reflects count-comp coverage: geneA/geneB
        // have count rows, geneC (pA only) does not. Regression guard for
        // the previously all-zero count column in measured_mask.parquet.
        let gene_b = tab.gene_names.iter().position(|g| &**g == "geneB").unwrap() as u32;
        assert_eq!(tab.modality_names[0].as_ref(), "count");
        assert!(tab.measured[gene_a as usize][0]);
        assert!(tab.measured[gene_b as usize][0]);
        assert!(!tab.measured[gene_c as usize][0]);
    }
}
