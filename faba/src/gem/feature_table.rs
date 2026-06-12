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

/// Global gene/modality/region namespaces + the per-(gene, modality) aggregates
/// the model is sized and sampled by. Per-row classification (`(gene, modality,
/// component, region, stratum)`) lives in a per-backend [`BackendRowMap`], NOT
/// here — under the split-backend design no single positional array addresses
/// any one backend's rows.
pub struct FeatureTable {
    /// `gene_id` → gene_key.
    pub gene_names: Vec<Box<str>>,
    /// `modality_id` → modality name. Slot 0 is always `"count"`.
    pub modality_names: Vec<Box<str>>,

    /// Number of transcript-position region bins R. `region_id ∈ 0..R`.
    pub n_regions: usize,

    /// Counts of loss-participating rows across the union of backends — kept
    /// only for the load-time summary log (`.len()`); not addressable per
    /// backend.
    pub count_comp_rows: Vec<u32>,
    pub modifier_comp_rows: Vec<u32>,
    /// `modifier_rows_by_modality[m]` is the modifier-comp row IDs for
    /// modality `m`. Modality 0 (count base) and the count-splice
    /// modalities are empty by convention (their rows live in
    /// `count_comp_rows`).
    pub modifier_rows_by_modality: Vec<Vec<u32>>,

    /// `is_count_modality[m]` = true for the count-derived satellite
    /// modalities (`count/spliced`, `count/unspliced`, …). These are
    /// embedded with their own δ direction like any modality, but are
    /// **sampled** in the count/anchor stratum (drawn under `--f-count`,
    /// Fisher-penalised) rather than the modifier stratum, and are
    /// excluded from the τ_modality balance over true modifiers. Slot 0
    /// (the AGG/count base) is false — no comp row ever carries it.
    pub is_count_modality: Vec<bool>,
    /// The modality ids for which `is_count_modality` is true, ascending.
    /// Used by the sampler to spread the count budget over splice types.
    pub count_modality_ids: Vec<u32>,

    /// `measured[g][m]` = true iff gene `g` has at least one input row in
    /// modality `m` — a count-comp row for the count modality (slot 0), a
    /// modifier-comp row otherwise. sampling.rs reads only the modifier
    /// columns (slot ≥ 1) to restrict swap-z draws to genes with real
    /// measurement in modality m; the count column exists so the exported
    /// `measured_mask.parquet` reflects count coverage rather than reading
    /// as all-zero.
    pub measured: Vec<Vec<bool>>,
}

/// Per-backend row classification, aligned to one backend's compact rows.
///
/// With the satellite modalities held in **separate** backends (genes vs
/// m6A vs A2I vs …), the single positional `FeatureTable.row_*` arrays no
/// longer address any one backend. Instead the global [`FeatureTable`] owns
/// the gene/modality/region namespaces, and each backend gets its own
/// `BackendRowMap` mapping that backend's row index → the global
/// `(gene, modality, component, region, stratum)`. A row whose gene is not
/// in the global table (e.g. a dead gene dropped in the refine pass) maps to
/// `None` and is skipped by the aggregator.
pub struct BackendRowMap {
    pub stratum: Vec<Option<RowStratum>>,
    pub gene: Vec<Option<u32>>,
    pub modality: Vec<Option<u32>>,
    pub component: Vec<Option<u32>>,
    pub region: Vec<Option<u32>>,
}

/// Mutable accumulator shared by [`FeatureTable::build`] (single list) and
/// [`FeatureTable::build_layered`] (genes + satellite name lists). Genes and
/// modalities are interned by **name**, so ids stay stable no matter how the
/// rows are split across backends — the union of names yields exactly the
/// joint `(gene, modality)` space a single concatenated build would.
struct TableBuilder {
    gene_to_id: FxHashMap<Box<str>, u32>,
    gene_names: Vec<Box<str>>,
    modality_to_id: FxHashMap<Box<str>, u32>,
    modality_names: Vec<Box<str>>,
    strata: Vec<Option<RowStratum>>,
    row_gene: Vec<Option<u32>>,
    row_modality: Vec<Option<u32>>,
    row_component: Vec<Option<u32>>,
    row_region: Vec<Option<u32>>,
    count_comp_rows: Vec<u32>,
    modifier_comp_rows: Vec<u32>,
}

impl TableBuilder {
    fn new() -> Self {
        let mut modality_to_id: FxHashMap<Box<str>, u32> = FxHashMap::default();
        let mut modality_names: Vec<Box<str>> = Vec::new();
        // Reserve slot 0 for "count".
        modality_to_id.insert("count".into(), 0);
        modality_names.push("count".into());
        Self {
            gene_to_id: FxHashMap::default(),
            gene_names: Vec::new(),
            modality_to_id,
            modality_names,
            strata: Vec::new(),
            row_gene: Vec::new(),
            row_modality: Vec::new(),
            row_component: Vec::new(),
            row_region: Vec::new(),
            count_comp_rows: Vec::new(),
            modifier_comp_rows: Vec::new(),
        }
    }

    fn modality_id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.modality_to_id.get(name) {
            return id;
        }
        let id = self.modality_names.len() as u32;
        self.modality_to_id.insert(name.into(), id);
        self.modality_names.push(name.into());
        id
    }

    /// Ingest one backend's row names. `add_new_genes` is `true` for the
    /// gene-defining list (the genes backend) and `false` for satellite
    /// lists — a satellite row whose gene is absent from the gene namespace
    /// is dropped (pushed as `None`) rather than introducing a new gene, so
    /// the global gene set is exactly the genes backend's and dead genes
    /// stay filtered across modalities.
    fn ingest(&mut self, names: &[Box<str>], add_new_genes: bool, regions: &RegionMap) {
        for name in names {
            let row = self.strata.len() as u32;
            let Some(key) = parse_feature_name(name) else {
                self.push_dropped();
                continue;
            };
            let gene_id = match self.gene_to_id.get(&key.gene) {
                Some(&id) => id,
                None if add_new_genes => {
                    let id = self.gene_names.len() as u32;
                    self.gene_to_id.insert(key.gene.clone(), id);
                    self.gene_names.push(key.gene.clone());
                    id
                }
                None => {
                    self.push_dropped();
                    continue;
                }
            };

            match classify(&key) {
                RowStratum::CountComp => {
                    // Only an explicit `unspliced` detail gets the unspliced
                    // modality; everything else (`spliced`, or a plain
                    // gene-count track with no split) is treated as `spliced`,
                    // so a count matrix without the split still trains as a
                    // single spliced modality. Each split gets its own δ/γ.
                    let modality_name = if &*key.detail == "unspliced" {
                        "unspliced"
                    } else {
                        "spliced"
                    };
                    let modality_id = self.modality_id(modality_name);
                    self.strata.push(Some(RowStratum::CountComp));
                    self.row_gene.push(Some(gene_id));
                    self.row_modality.push(Some(modality_id));
                    self.row_component.push(None);
                    // Splice rows have no transcript-position component;
                    // region stays the sentinel 0.
                    self.row_region.push(Some(0));
                    self.count_comp_rows.push(row);
                }
                RowStratum::ModifierComp => {
                    let modality_id = self.modality_id(&key.modality);
                    let component = parse_component_idx(&key.detail);
                    let region = regions.lookup(&key.gene, &key.modality, component.unwrap_or(0));
                    self.strata.push(Some(RowStratum::ModifierComp));
                    self.row_gene.push(Some(gene_id));
                    self.row_modality.push(Some(modality_id));
                    self.row_component.push(component);
                    self.row_region.push(Some(region));
                    self.modifier_comp_rows.push(row);
                }
                RowStratum::Site => {
                    let modality_id = self.modality_id(&key.modality);
                    self.strata.push(Some(RowStratum::Site));
                    self.row_gene.push(Some(gene_id));
                    self.row_modality.push(Some(modality_id));
                    self.row_component.push(None);
                    self.row_region.push(None);
                }
            }
        }
    }

    fn push_dropped(&mut self) {
        self.strata.push(None);
        self.row_gene.push(None);
        self.row_modality.push(None);
        self.row_component.push(None);
        self.row_region.push(None);
    }

    fn finish(self, regions: &RegionMap) -> FeatureTable {
        let n_genes = self.gene_names.len();
        let n_modalities = self.modality_names.len();

        // measured[g][m]: any input row for (g, m)? A count-comp row marks
        // both its splice modality (≥1) and slot 0 (count coverage).
        // is_count_modality[m]: which modalities are count-derived splits.
        let mut measured = vec![vec![false; n_modalities]; n_genes];
        let mut is_count_modality = vec![false; n_modalities];
        for &row in &self.count_comp_rows {
            if let Some(g) = self.row_gene[row as usize] {
                measured[g as usize][0] = true;
                if let Some(m) = self.row_modality[row as usize] {
                    measured[g as usize][m as usize] = true;
                    if m != 0 {
                        is_count_modality[m as usize] = true;
                    }
                }
            }
        }
        for &row in &self.modifier_comp_rows {
            if let (Some(g), Some(m)) =
                (self.row_gene[row as usize], self.row_modality[row as usize])
            {
                measured[g as usize][m as usize] = true;
            }
        }

        let count_modality_ids: Vec<u32> = (0..n_modalities as u32)
            .filter(|&m| is_count_modality[m as usize])
            .collect();

        let mut modifier_rows_by_modality: Vec<Vec<u32>> = vec![Vec::new(); n_modalities];
        for &row in &self.modifier_comp_rows {
            if let Some(m) = self.row_modality[row as usize] {
                modifier_rows_by_modality[m as usize].push(row);
            }
        }

        FeatureTable {
            n_regions: regions.n_regions,
            gene_names: self.gene_names,
            modality_names: self.modality_names,
            count_comp_rows: self.count_comp_rows,
            modifier_comp_rows: self.modifier_comp_rows,
            modifier_rows_by_modality,
            is_count_modality,
            count_modality_ids,
            measured,
        }
    }
}

impl FeatureTable {
    pub fn n_genes(&self) -> usize {
        self.gene_names.len()
    }

    pub fn n_modalities(&self) -> usize {
        self.modality_names.len()
    }

    /// Modality id of the `"spliced"` track (the count rows whose detail is
    /// not `unspliced`), if present. Used to mask the collapse backend down
    /// to spliced rows only.
    pub fn spliced_modality_id(&self) -> Option<u32> {
        self.modality_names
            .iter()
            .position(|m| &**m == "spliced")
            .map(|i| i as u32)
    }

    /// Classify a single feature axis into row strata + compact ids.
    /// Modality 0 is forced to be `"count"` so AGG rows always reduce to
    /// "sum of modality-0 components". `regions` resolves each modifier
    /// component to a transcript-position region bin.
    pub fn build(feature_names: &[Box<str>], regions: &RegionMap) -> Self {
        let mut b = TableBuilder::new();
        b.ingest(feature_names, true, regions);
        b.finish(regions)
    }

    /// Build a global table spanning **separate** per-modality backends. The
    /// `genes` list defines the gene namespace (and the spliced/unspliced
    /// count modalities); each `satellites` list only *augments* existing
    /// genes with modifier modalities (m6A / A2I / pA) and their measured
    /// flags — satellite rows for genes absent from `genes` are dropped. The
    /// returned table's `row_*` arrays span the concatenation `genes ++
    /// satellites…` and are not used to index any single backend (see
    /// [`Self::map_backend_rows`]); only the namespaces, `measured`,
    /// `is_count_modality`, `count_modality_ids`, and the per-modality
    /// modifier-row non-emptiness are consumed globally.
    pub fn build_layered(
        genes: &[Box<str>],
        satellites: &[&[Box<str>]],
        regions: &RegionMap,
    ) -> Self {
        let mut b = TableBuilder::new();
        b.ingest(genes, true, regions);
        for s in satellites {
            b.ingest(s, false, regions);
        }
        b.finish(regions)
    }

    /// Map one backend's row names to the global `(gene, modality, …)` ids.
    /// Aligned to `names` (the backend's compact rows). A row whose gene is
    /// absent from this table maps to `None` everywhere (dropped downstream).
    pub fn map_backend_rows(&self, names: &[Box<str>], regions: &RegionMap) -> BackendRowMap {
        let gene_to_id: FxHashMap<&str, u32> = self
            .gene_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_ref(), i as u32))
            .collect();
        let modality_to_id: FxHashMap<&str, u32> = self
            .modality_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_ref(), i as u32))
            .collect();

        let n = names.len();
        let mut stratum = Vec::with_capacity(n);
        let mut gene = Vec::with_capacity(n);
        let mut modality = Vec::with_capacity(n);
        let mut component = Vec::with_capacity(n);
        let mut region = Vec::with_capacity(n);

        let push_none = |stratum: &mut Vec<Option<RowStratum>>,
                         gene: &mut Vec<Option<u32>>,
                         modality: &mut Vec<Option<u32>>,
                         component: &mut Vec<Option<u32>>,
                         region: &mut Vec<Option<u32>>| {
            stratum.push(None);
            gene.push(None);
            modality.push(None);
            component.push(None);
            region.push(None);
        };
        for name in names {
            let Some(key) = parse_feature_name(name) else {
                push_none(
                    &mut stratum,
                    &mut gene,
                    &mut modality,
                    &mut component,
                    &mut region,
                );
                continue;
            };
            let Some(&gid) = gene_to_id.get(&*key.gene) else {
                push_none(
                    &mut stratum,
                    &mut gene,
                    &mut modality,
                    &mut component,
                    &mut region,
                );
                continue;
            };
            match classify(&key) {
                RowStratum::CountComp => {
                    let modality_name = if &*key.detail == "unspliced" {
                        "unspliced"
                    } else {
                        "spliced"
                    };
                    stratum.push(Some(RowStratum::CountComp));
                    gene.push(Some(gid));
                    modality.push(modality_to_id.get(modality_name).copied());
                    component.push(None);
                    region.push(Some(0));
                }
                RowStratum::ModifierComp => {
                    let comp = parse_component_idx(&key.detail);
                    let reg = regions.lookup(&key.gene, &key.modality, comp.unwrap_or(0));
                    stratum.push(Some(RowStratum::ModifierComp));
                    gene.push(Some(gid));
                    modality.push(modality_to_id.get(&*key.modality).copied());
                    component.push(comp);
                    region.push(Some(reg));
                }
                RowStratum::Site => {
                    stratum.push(Some(RowStratum::Site));
                    gene.push(Some(gid));
                    modality.push(modality_to_id.get(&*key.modality).copied());
                    component.push(None);
                    region.push(None);
                }
            }
        }

        BackendRowMap {
            stratum,
            gene,
            modality,
            component,
            region,
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
    fn count_detail_defaults_to_spliced() {
        // A count track whose detail isn't `unspliced` (here a plain
        // `/count/total`) collapses to the `spliced` modality — no
        // per-detail modality is spawned — while `unspliced` stays distinct.
        let names: Vec<Box<str>> = [
            "geneA/count/total",     // → spliced
            "geneA/count/unspliced", // → unspliced
            "geneB/count/spliced",   // → spliced (merges with geneA's `total`)
        ]
        .iter()
        .map(|s| (*s).into())
        .collect();

        let tab = FeatureTable::build(&names, &RegionMap::empty(1));
        assert!(tab.modality_names.iter().any(|m| &**m == "spliced"));
        assert!(tab.modality_names.iter().any(|m| &**m == "unspliced"));
        // No `total` modality leaked in.
        assert!(!tab.modality_names.iter().any(|m| &**m == "total"));
        let spliced = tab
            .modality_names
            .iter()
            .position(|m| &**m == "spliced")
            .unwrap() as u32;
        let gene_a = tab.gene_names.iter().position(|g| &**g == "geneA").unwrap();
        let gene_b = tab.gene_names.iter().position(|g| &**g == "geneB").unwrap();
        // geneA's `/count/total` and geneB's `/count/spliced` both land on
        // the spliced modality.
        assert!(tab.measured[gene_a][spliced as usize]);
        assert!(tab.measured[gene_b][spliced as usize]);
        assert!(tab.is_count_modality[spliced as usize]);
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
        // Count details become their own modalities (≥1), distinct from the
        // reserved slot-0 "count" base — so spliced/unspliced get separate
        // δ directions.
        let spliced = tab
            .modality_names
            .iter()
            .position(|m| &**m == "spliced")
            .expect("spliced modality") as u32;
        let unspliced = tab
            .modality_names
            .iter()
            .position(|m| &**m == "unspliced")
            .expect("unspliced modality") as u32;
        assert_ne!(spliced, unspliced);
        assert_ne!(spliced, 0); // never the reserved base slot
        assert!(tab.is_count_modality[spliced as usize]);
        assert!(tab.is_count_modality[unspliced as usize]);
        assert!(!tab.is_count_modality[0]); // base slot is not a count modality
        assert_eq!(tab.count_modality_ids, vec![spliced, unspliced]);

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

    #[test]
    fn layered_build_and_backend_row_maps() {
        // Genes backend: count rows. Satellite (m6A) backend: modifier rows
        // for the same genes, plus a `geneZ` absent from the genes backend.
        let genes: Vec<Box<str>> = [
            "geneA/count/spliced",
            "geneA/count/unspliced",
            "geneB/count/spliced",
        ]
        .iter()
        .map(|s| (*s).into())
        .collect();
        let m6a: Vec<Box<str>> = ["geneA/m6A/0", "geneB/m6A/1", "geneZ/m6A/0"]
            .iter()
            .map(|s| (*s).into())
            .collect();

        let tab = FeatureTable::build_layered(&genes, &[&m6a], &RegionMap::empty(3));

        // Gene namespace comes from the genes backend only — geneZ is dropped.
        assert_eq!(tab.n_genes(), 2);
        assert!(!tab.gene_names.iter().any(|g| &**g == "geneZ"));

        // The m6A modality is still registered (augmented from the satellite).
        let m6a_id = tab
            .modality_names
            .iter()
            .position(|m| &**m == "m6A")
            .expect("m6A modality") as u32;
        let ga = tab.gene_names.iter().position(|g| &**g == "geneA").unwrap();
        let gb = tab.gene_names.iter().position(|g| &**g == "geneB").unwrap();
        assert!(tab.measured[ga][m6a_id as usize]);
        assert!(tab.measured[gb][m6a_id as usize]);
        assert!(!tab.is_count_modality[m6a_id as usize]);
        // m6A has modifier rows → non-empty (sampler relies on this).
        assert!(!tab.modifier_rows_by_modality[m6a_id as usize].is_empty());

        let spliced = tab.spliced_modality_id().expect("spliced id");
        assert!(tab.is_count_modality[spliced as usize]);

        // Satellite backend row map: geneZ row drops to None; others map to
        // the global ids.
        let rm = tab.map_backend_rows(&m6a, &RegionMap::empty(3));
        assert_eq!(rm.stratum[0], Some(RowStratum::ModifierComp));
        assert_eq!(rm.gene[0], Some(ga as u32));
        assert_eq!(rm.modality[0], Some(m6a_id));
        assert_eq!(rm.gene[2], None); // geneZ absent from the table
        assert_eq!(rm.stratum[2], None);

        // Genes backend row map: spliced vs unspliced distinguished, so the
        // spliced mask can be built from it.
        let grm = tab.map_backend_rows(&genes, &RegionMap::empty(3));
        assert_eq!(grm.stratum[0], Some(RowStratum::CountComp));
        assert_eq!(grm.modality[0], Some(spliced));
        assert_ne!(grm.modality[1], Some(spliced)); // unspliced
    }
}
