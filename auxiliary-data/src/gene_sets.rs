//! Gene-set sources for enrichment, reduced to one common `term → gene-set`
//! form regardless of origin:
//!   - **GAF** (GO Consortium gene-association format): one annotation per row,
//!     `gene → GO id`; propagated up the ontology (true-path rule) into
//!     `term → genes`. Read [`read_gaf`] then [`GafRaw::into_gene_sets`].
//!   - **GMT** (MSigDB): one gene-set per row, `term <TAB> description <TAB>
//!     genes…`; taken as-is, no propagation. Read [`read_gmt`].
//!
//! Each gene carries the identifier aliases the source provides (symbol /
//! accession / synonyms for GAF) so a downstream reconciler can match it against
//! an expression dictionary whether that uses HGNC, ENSG, or `ENSG_HGNC`.

use crate::ontology::Ontology;
use anyhow::{Context, Result};
use data_beans::utilities::name_matching::GeneIndex;
use matrix_util::common_io::open_buf_reader;
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::BufRead;

/// A collection of gene-sets keyed by term id — the common form all sources
/// (GAF/GMT/markers) reduce to before scoring.
#[derive(Default)]
pub struct GeneSets {
    /// term id → display name (absent ⇒ display the id itself).
    pub names: FxHashMap<Box<str>, Box<str>>,
    /// term id → member gene keys (upper-cased primary identifier).
    pub term_genes: FxHashMap<Box<str>, FxHashSet<Box<str>>>,
    /// gene key → all identifier aliases for that gene (including the key), used
    /// to reconcile against an expression dictionary (HGNC / ENSG / UniProt).
    pub gene_aliases: FxHashMap<Box<str>, FxHashSet<Box<str>>>,
}

impl GeneSets {
    /// Number of distinct terms (gene-sets).
    #[must_use]
    pub fn n_terms(&self) -> usize {
        self.term_genes.len()
    }

    /// Total membership entries summed over terms (post-propagation for GAF).
    #[must_use]
    pub fn n_annotations(&self) -> usize {
        self.term_genes.values().map(FxHashSet::len).sum()
    }

    /// Number of distinct member genes across all sets.
    #[must_use]
    pub fn n_genes(&self) -> usize {
        self.gene_aliases.len()
    }
}

/// Read an MSigDB-style GMT: each non-comment line is
/// `term <TAB> description <TAB> gene1 <TAB> gene2 …`. Terms are used verbatim
/// (they become tree nodes only if they resolve against an `--obo`); genes are
/// their own keys with no extra aliases; no propagation.
pub fn read_gmt(path: &str) -> Result<GeneSets> {
    let reader = open_buf_reader(path).with_context(|| format!("failed to open GMT: {path}"))?;
    let mut gs = GeneSets::default();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split('\t');
        let term = it.next().map(str::trim).unwrap_or_default();
        if term.is_empty() {
            continue;
        }
        let desc = it.next().map(str::trim).unwrap_or_default();
        let genes: FxHashSet<Box<str>> = it
            .map(str::trim)
            .filter(|g| !g.is_empty())
            .map(|g| g.to_uppercase().into_boxed_str())
            .collect();
        if genes.is_empty() {
            continue;
        }
        let term: Box<str> = term.into();
        for g in &genes {
            gs.gene_aliases
                .entry(g.clone())
                .or_default()
                .insert(g.clone());
        }
        if !desc.is_empty() {
            gs.names.insert(term.clone(), desc.into());
        }
        gs.term_genes.entry(term).or_default().extend(genes);
    }
    Ok(gs)
}

/// Options for [`read_gaf`].
#[derive(Default, Clone, Copy)]
pub struct GafOpts {
    /// Drop IEA (inferred-from-electronic-annotation) rows — the low-confidence
    /// bulk of a GAF (evidence-code column).
    pub no_iea: bool,
}

/// Raw GAF annotations before ontology propagation: gene key → direct GO ids,
/// plus the per-gene identifier aliases harvested from the GAF.
pub struct GafRaw {
    gene2direct: FxHashMap<Box<str>, FxHashSet<Box<str>>>,
    gene_aliases: FxHashMap<Box<str>, FxHashSet<Box<str>>>,
}

/// Parse a GAF (`.gaf` or `.gaf.gz`). Columns (1-based) used: 2 = object id /
/// accession, 3 = symbol (the gene key), 4 = qualifier (rows with `NOT` are
/// dropped), 5 = GO id, 7 = evidence code (for `no_iea`), 11 = synonyms. The
/// symbol, accession, and pipe-split synonyms are kept as match aliases.
pub fn read_gaf(path: &str, opts: &GafOpts) -> Result<GafRaw> {
    let reader = open_buf_reader(path).with_context(|| format!("failed to open GAF: {path}"))?;
    let mut gene2direct: FxHashMap<Box<str>, FxHashSet<Box<str>>> = FxHashMap::default();
    let mut gene_aliases: FxHashMap<Box<str>, FxHashSet<Box<str>>> = FxHashMap::default();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('!') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 15 {
            continue; // GAF 2.x has ≥15 columns
        }
        let symbol = f[2].trim();
        let go = f[4].trim();
        if symbol.is_empty() || !go.starts_with("GO:") {
            continue;
        }
        if f[3].split('|').any(|q| q.trim() == "NOT") {
            continue;
        }
        if opts.no_iea && f[6].trim() == "IEA" {
            continue;
        }
        let key: Box<str> = symbol.to_uppercase().into();
        let aliases = gene_aliases.entry(key.clone()).or_default();
        aliases.insert(key.clone());
        let acc = f[1].trim();
        if !acc.is_empty() {
            aliases.insert(acc.to_uppercase().into());
        }
        for syn in f[10].split('|').map(str::trim).filter(|s| !s.is_empty()) {
            aliases.insert(syn.to_uppercase().into());
        }
        gene2direct.entry(key).or_default().insert(go.into());
    }
    Ok(GafRaw {
        gene2direct,
        gene_aliases,
    })
}

impl GafRaw {
    /// Reduce to the common [`GeneSets`] form. With an ontology, propagate each
    /// gene up the `is_a` + `part_of` closure (true-path rule) and pull display
    /// names; without one, keep the direct annotations only.
    #[must_use]
    pub fn into_gene_sets(self, onto: Option<&Ontology>) -> GeneSets {
        let mut gs = GeneSets {
            gene_aliases: self.gene_aliases,
            ..Default::default()
        };
        for (gene, direct) in &self.gene2direct {
            let mut full: FxHashSet<Box<str>> = FxHashSet::default();
            for go in direct {
                match onto {
                    Some(o) if o.contains(go) => {
                        full.extend(o.ancestors_or_self_with_part_of(go));
                    }
                    _ => {
                        full.insert(go.clone());
                    }
                }
            }
            for t in full {
                gs.term_genes.entry(t).or_default().insert(gene.clone());
            }
        }
        if let Some(o) = onto {
            let terms: Vec<Box<str>> = gs.term_genes.keys().cloned().collect();
            for t in terms {
                if let Some(n) = o.name(&t) {
                    gs.names.insert(t, n.into());
                }
            }
        }
        gs
    }
}

/// Gene-sets reconciled against an expression dictionary: term → matched row
/// indices, plus coverage statistics. This is the sparse membership the
/// enrichment scorer consumes (term → its rows in the gene profile).
pub struct Reconciled {
    /// term id → matched dictionary row indices (deduped, ascending); only
    /// terms retaining ≥ the requested minimum members are kept.
    pub term_rows: FxHashMap<Box<str>, Vec<usize>>,
    /// term id → display name (carried through from the source).
    pub names: FxHashMap<Box<str>, Box<str>>,
    /// All matched dictionary rows that carry ≥1 annotation (the enrichment
    /// background "universe"), ascending. Drawn from every source term — NOT
    /// only the size-windowed `term_rows` — so the hypergeometric background is
    /// the full annotated set.
    pub universe: Vec<usize>,
    /// distinct gene keys in the source.
    pub n_genes_total: usize,
    /// gene keys that matched at least one dictionary row.
    pub n_genes_matched: usize,
    /// terms retained (≥ min members after reconciliation).
    pub n_terms_kept: usize,
    /// terms in the source before the min-member filter.
    pub n_terms_total: usize,
}

impl GeneSets {
    /// Reconcile each member gene against `index` (built over the expression
    /// dictionary), trying every identifier alias and taking the first hit, then
    /// keep only terms whose matched-member count is in
    /// `[min_members, max_members]`. The upper bound matters: GSEA `es_std` is
    /// ill-behaved for near-universal sets (their restandardization SD collapses
    /// → spurious huge z), so giant root-level terms must be excluded — the
    /// standard GSEA size window. `max_members = None` disables the cap. Handles
    /// HGNC / ENSG / `ENSG_HGNC` conventions via [`GeneIndex`].
    #[must_use]
    pub fn reconcile(
        &self,
        index: &GeneIndex,
        min_members: usize,
        max_members: Option<usize>,
    ) -> Reconciled {
        // gene key → matched dictionary row (first alias that resolves).
        let mut gene_row: FxHashMap<&str, usize> = FxHashMap::default();
        for (key, aliases) in &self.gene_aliases {
            let row = index
                .match_gene(key)
                .or_else(|| aliases.iter().find_map(|a| index.match_gene(a)));
            if let Some(r) = row {
                gene_row.insert(key, r);
            }
        }
        let n_genes_matched = gene_row.len();
        // Background universe = every matched annotated row (deduped, ascending),
        // independent of the term size window applied below.
        let mut universe: Vec<usize> = gene_row.values().copied().collect();
        universe.sort_unstable();
        universe.dedup();

        let mut term_rows: FxHashMap<Box<str>, Vec<usize>> = FxHashMap::default();
        for (term, genes) in &self.term_genes {
            let mut rows: Vec<usize> = genes
                .iter()
                .filter_map(|g| gene_row.get(g.as_ref()).copied())
                .collect();
            rows.sort_unstable();
            rows.dedup();
            let n = rows.len();
            if n >= min_members && max_members.is_none_or(|mx| n <= mx) {
                term_rows.insert(term.clone(), rows);
            }
        }

        Reconciled {
            n_terms_kept: term_rows.len(),
            n_terms_total: self.term_genes.len(),
            n_genes_total: self.gene_aliases.len(),
            n_genes_matched,
            names: self.names.clone(),
            universe,
            term_rows,
        }
    }
}

/// Below this matched-gene fraction, [`Reconciled::log_coverage`] warns rather
/// than logs at info — a thin overlap silently produces no enrichment.
const COVERAGE_WARN_FRAC: f32 = 0.5;

impl Reconciled {
    /// Fraction of source genes matched to the dictionary.
    #[must_use]
    pub fn match_frac(&self) -> f32 {
        if self.n_genes_total == 0 {
            0.0
        } else {
            self.n_genes_matched as f32 / self.n_genes_total as f32
        }
    }

    /// Total membership entries across retained terms.
    #[must_use]
    pub fn n_memberships(&self) -> usize {
        self.term_rows.values().map(Vec::len).sum()
    }

    /// Log a one-line coverage summary — `info`, or `warn` when overlap is thin
    /// (a near-empty map silently yields no enrichment).
    pub fn log_coverage(&self) {
        let frac = self.match_frac();
        let msg = format!(
            "gene-set coverage: {}/{} genes matched ({:.1}%); {}/{} terms kept (≥min members); {} memberships",
            self.n_genes_matched,
            self.n_genes_total,
            100.0 * frac,
            self.n_terms_kept,
            self.n_terms_total,
            self.n_memberships(),
        );
        if frac < COVERAGE_WARN_FRAC {
            log::warn!("{msg}");
        } else {
            log::info!("{msg}");
        }
    }

    /// Fail when coverage is too thin to produce meaningful enrichment.
    pub fn ensure_coverage(&self, min_frac: f32, min_terms: usize) -> Result<()> {
        let frac = self.match_frac();
        anyhow::ensure!(
            frac >= min_frac && self.n_terms_kept >= min_terms,
            "insufficient gene→term coverage: {:.1}% of genes matched (need ≥{:.0}%), \
             {} terms kept (need ≥{}). Check that gene-set ids (HGNC/ENSG) match the \
             expression dictionary's gene names.",
            100.0 * frac,
            100.0 * min_frac,
            self.n_terms_kept,
            min_terms,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(contents: &str, suffix: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(suffix).tempfile().unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn gmt_round_trip() {
        let f = tmp(
            "# comment\n\
             SET_A\tset A desc\tCD3D\tcd8a\tGZMK\n\
             SET_B\tset B desc\tMS4A1\tCD79A\n",
            ".gmt",
        );
        let gs = read_gmt(f.path().to_str().unwrap()).unwrap();
        assert_eq!(gs.n_terms(), 2);
        assert_eq!(gs.names.get("SET_A").map(|n| &**n), Some("set A desc"));
        // genes upper-cased, deduped into keys.
        assert!(gs.term_genes["SET_A"].contains("CD8A"));
        assert_eq!(gs.term_genes["SET_A"].len(), 3);
        assert!(gs.gene_aliases.contains_key("CD3D"));
    }

    /// GO-shaped OBO: GO:0000001 (leaf) is_a GO:0000002 (parent).
    fn write_go_obo() -> tempfile::NamedTempFile {
        tmp(
            "format-version: 1.2\n\n\
             [Term]\nid: GO:0000002\nname: parent process\n\n\
             [Term]\nid: GO:0000001\nname: leaf process\nis_a: GO:0000002 ! parent process\n",
            ".obo",
        )
    }

    #[test]
    fn gaf_parse_filters_and_propagates() {
        // 17-col GAF rows: FOO→GO:0000001 (IDA, kept); BAR→GO:0000001 (NOT, dropped);
        // BAZ→GO:0000001 (IEA, dropped under no_iea). Tabs are literal.
        let rows = "\
UniProtKB\tP11111\tFOO\t\tGO:0000001\tPMID:1\tIDA\t\tP\tFoo protein\tFOO_ALT|ENSG00000011111\tprotein\ttaxon:9606\t20200101\tUniProt\t\t\n\
UniProtKB\tP22222\tBAR\tNOT|enables\tGO:0000001\tPMID:2\tIDA\t\tP\tBar protein\t\tprotein\ttaxon:9606\t20200101\tUniProt\t\t\n\
UniProtKB\tP33333\tBAZ\t\tGO:0000001\tPMID:3\tIEA\t\tP\tBaz protein\t\tprotein\ttaxon:9606\t20200101\tUniProt\t\t\n";
        let gaf = tmp(rows, ".gaf");
        let raw = read_gaf(gaf.path().to_str().unwrap(), &GafOpts { no_iea: true }).unwrap();

        let onto = Ontology::load_obo(write_go_obo().path().to_str().unwrap()).unwrap();
        let gs = raw.into_gene_sets(Some(&onto));

        // Only FOO survives (NOT + IEA dropped).
        assert_eq!(gs.n_genes(), 1);
        // True-path: FOO is a member of the leaf AND its is_a parent.
        assert!(gs.term_genes["GO:0000001"].contains("FOO"));
        assert!(gs.term_genes["GO:0000002"].contains("FOO"));
        // names pulled from the ontology.
        assert_eq!(
            gs.names.get("GO:0000002").map(|n| &**n),
            Some("parent process")
        );
        // aliases harvested: accession + synonym (incl. an ENSG).
        let al = &gs.gene_aliases["FOO"];
        assert!(al.contains("P11111"));
        assert!(al.contains("ENSG00000011111"));
    }

    #[test]
    fn reconcile_matches_aliases_and_filters() {
        let f = tmp("SET_A\tdesc\tCD8A\tMS4A1\tGHOSTGENE\n", ".gmt");
        let gs = read_gmt(f.path().to_str().unwrap()).unwrap();
        // dict mixes ENSG_SYMBOL and bare symbol; GHOSTGENE is absent.
        let dict: Vec<Box<str>> = ["ENSG00000153563_CD8A", "MS4A1"]
            .iter()
            .map(|s| Box::from(*s))
            .collect();
        let idx = GeneIndex::build(&dict);

        let rec = gs.reconcile(&idx, 1, None);
        // CD8A (via ENSG_… symbol tier) + MS4A1 (exact) match; GHOSTGENE doesn't.
        assert_eq!(rec.n_genes_total, 3);
        assert_eq!(rec.n_genes_matched, 2);
        assert_eq!(rec.universe, vec![0, 1]); // both matched rows, deduped
        assert_eq!(rec.term_rows["SET_A"].len(), 2);
        assert!(rec.ensure_coverage(0.5, 1).is_ok());

        // min_members filter drops the term → coverage check fails.
        let rec2 = gs.reconcile(&idx, 3, None);
        assert!(rec2.term_rows.is_empty());
        assert!(rec2.ensure_coverage(0.5, 1).is_err());

        // max_members cap also drops the (2-member) term.
        let rec3 = gs.reconcile(&idx, 1, Some(1));
        assert!(rec3.term_rows.is_empty());
    }

    #[test]
    fn gaf_without_ontology_keeps_direct() {
        let rows = "\
UniProtKB\tP11111\tFOO\t\tGO:0000001\tPMID:1\tIDA\t\tP\tFoo\t\tprotein\ttaxon:9606\t20200101\tUniProt\t\t\n";
        let gaf = tmp(rows, ".gaf");
        let raw = read_gaf(gaf.path().to_str().unwrap(), &GafOpts::default()).unwrap();
        let gs = raw.into_gene_sets(None);
        assert_eq!(gs.n_terms(), 1); // no propagation → just the direct term
        assert!(gs.term_genes["GO:0000001"].contains("FOO"));
    }
}
