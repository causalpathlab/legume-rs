//! Generic OBO ontology (`cl-basic.obo`, `go-basic.obo`, …) parsed with
//! `fastobo` and stored as a `petgraph` directed graph (edges point
//! **child → parent**, tagged with the relation kind).
//!
//! This is a thin, reusable wrapper: parse once, then query each term's
//! ancestors via [`Ontology::ancestors_or_self`] (`is_a` only — the relation a
//! collapsed-label tree needs) or [`Ontology::ancestors_or_self_with_part_of`]
//! (`is_a` + `part_of`, the GO "true-path" closure). We never flatten or walk
//! the full ontology.
//!
//! Prefix-agnostic: any term id is kept (`CL:`, `GO:`, …); edges to
//! unknown/obsolete targets are dropped, as are obsolete terms. The
//! `{is_inferred="true"}` qualifier blocks fastobo handles natively (a hand
//! rolled parser missing them was the original prototype's bug).

use anyhow::{Context, Result};
use fastobo::ast::{EntityFrame, TermClause};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction::Outgoing;
use rustc_hash::{FxHashMap, FxHashSet};

/// Ontology relation kind carried on each `child → parent` edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rel {
    /// `is_a` subsumption (subclass).
    IsA,
    /// `relationship: part_of` (mereological containment; GO true-path).
    PartOf,
}

/// Parsed OBO DAG. Edges are `child → parent` (tagged [`Rel`]), so a node's
/// ancestors are reachable along `Outgoing` edges and its children along
/// `Incoming` edges.
pub struct Ontology {
    graph: DiGraph<Box<str>, Rel>,
    idx: FxHashMap<Box<str>, NodeIndex>,
    names: FxHashMap<Box<str>, Box<str>>,
}

/// One parsed, non-obsolete OBO term gathered in the first pass.
struct ParsedTerm {
    id: Box<str>,
    name: Option<Box<str>>,
    /// `is_a` parent ids.
    is_a: Vec<Box<str>>,
    /// `relationship: part_of` parent ids.
    part_of: Vec<Box<str>>,
}

impl Ontology {
    /// Parse an OBO file into the `is_a` + `part_of` DAG. Obsolete terms and
    /// edges to unknown/obsolete targets are skipped.
    pub fn load_obo(path: &str) -> Result<Self> {
        let doc = fastobo::from_file(path)
            .with_context(|| format!("failed to parse OBO file: {path}"))?;

        // Pass 1: collect non-obsolete terms.
        let mut terms: Vec<ParsedTerm> = Vec::new();
        for frame in doc.entities() {
            let EntityFrame::Term(term) = frame else {
                continue;
            };
            // fastobo's `Line<_>` Display can carry a trailing newline / inline
            // comment, so trim every extracted token down to the bare value.
            let id: Box<str> = term.id().to_string().trim().into();
            let mut name: Option<Box<str>> = None;
            let mut is_a: Vec<Box<str>> = Vec::new();
            let mut part_of: Vec<Box<str>> = Vec::new();
            let mut obsolete = false;
            for line in term.clauses() {
                match &**line {
                    TermClause::Name(n) => name = Some(n.to_string().trim().into()),
                    TermClause::IsObsolete(b) => obsolete = obsolete || *b,
                    TermClause::IsA(parent) => is_a.push(parent.to_string().trim().into()),
                    TermClause::Relationship(rel, target) => {
                        if rel.to_string().trim() == "part_of" {
                            part_of.push(target.to_string().trim().into());
                        }
                    }
                    _ => {}
                }
            }
            if !obsolete {
                terms.push(ParsedTerm {
                    id,
                    name,
                    is_a,
                    part_of,
                });
            }
        }

        // Pass 2: nodes first, then edges (only between known nodes).
        let mut graph: DiGraph<Box<str>, Rel> = DiGraph::new();
        let mut idx: FxHashMap<Box<str>, NodeIndex> = FxHashMap::default();
        let mut names: FxHashMap<Box<str>, Box<str>> = FxHashMap::default();
        for term in &terms {
            let node = graph.add_node(term.id.clone());
            idx.insert(term.id.clone(), node);
            if let Some(n) = &term.name {
                names.insert(term.id.clone(), n.clone());
            }
        }
        for term in &terms {
            let child = idx[&term.id];
            for (parents, rel) in [(&term.is_a, Rel::IsA), (&term.part_of, Rel::PartOf)] {
                for p in parents {
                    if let Some(&parent) = idx.get(p) {
                        graph.add_edge(child, parent, rel);
                    }
                }
            }
        }

        Ok(Self { graph, idx, names })
    }

    /// Number of (non-obsolete) terms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.node_count()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.idx.contains_key(id)
    }

    /// Human-readable term name (`None` if the term is unknown or unnamed).
    #[must_use]
    pub fn name(&self, id: &str) -> Option<&str> {
        self.names.get(id).map(|n| &**n)
    }

    /// All `is_a` ancestors of `id` plus `id` itself (empty if `id` is unknown).
    #[must_use]
    pub fn ancestors_or_self(&self, id: &str) -> FxHashSet<Box<str>> {
        self.ancestors_impl(id, false)
    }

    /// All `is_a` + `part_of` ancestors of `id` plus `id` itself — the GO
    /// "true-path" closure (empty if `id` is unknown).
    #[must_use]
    pub fn ancestors_or_self_with_part_of(&self, id: &str) -> FxHashSet<Box<str>> {
        self.ancestors_impl(id, true)
    }

    /// Walk ancestors in `NodeIndex` space (Copy — no per-visit string clones),
    /// following `is_a` edges and, when `with_part_of`, `part_of` edges too;
    /// then materialize each id exactly once at the boundary.
    fn ancestors_impl(&self, id: &str, with_part_of: bool) -> FxHashSet<Box<str>> {
        let Some(&start) = self.idx.get(id) else {
            return FxHashSet::default();
        };
        let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
        seen.insert(start);
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            for edge in self.graph.edges_directed(n, Outgoing) {
                let follow = matches!(edge.weight(), Rel::IsA)
                    || (with_part_of && matches!(edge.weight(), Rel::PartOf));
                if follow && seen.insert(edge.target()) {
                    stack.push(edge.target());
                }
            }
        }
        seen.iter().map(|&n| self.graph[n].clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal CL-shaped OBO: cell → lymphocyte → T → {CD4, CD8}; B sibling;
    /// one obsolete term; one `{is_inferred="true"}` qualifier on an is_a line;
    /// one `part_of` edge (CD8 T part_of an immune-system "compartment") to
    /// exercise the relation-filtered traversal.
    fn write_obo() -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            f,
            "format-version: 1.2\n\n\
             [Term]\nid: CL:0000000\nname: cell\n\n\
             [Term]\nid: CL:0000542\nname: lymphocyte\nis_a: CL:0000000 ! cell\n\n\
             [Term]\nid: CL:0000084\nname: T cell\nis_a: CL:0000542 {{is_inferred=\"true\"}} ! lymphocyte\n\n\
             [Term]\nid: CL:0000624\nname: CD4 T\nis_a: CL:0000084 ! T cell\n\n\
             [Term]\nid: CL:0000625\nname: CD8 T\nis_a: CL:0000084 ! T cell\nrelationship: part_of CL:1000000 ! compartment\n\n\
             [Term]\nid: CL:1000000\nname: immune compartment\n\n\
             [Term]\nid: CL:0000236\nname: B cell\nis_a: CL:0000542 ! lymphocyte\n\n\
             [Term]\nid: CL:9999999\nname: dead\nis_obsolete: true\n"
        )
        .unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn parses_and_resolves_ancestry() {
        let f = write_obo();
        let onto = Ontology::load_obo(f.path().to_str().unwrap()).unwrap();

        // obsolete term dropped; 7 live terms (incl. the compartment).
        assert_eq!(onto.len(), 7);
        assert!(!onto.contains("CL:9999999"));
        assert_eq!(onto.name("CL:0000084"), Some("T cell"));

        // The {is_inferred} qualifier must NOT break the edge: CD4/CD8 under T,
        // T under lymphocyte under cell.
        let anc = onto.ancestors_or_self("CL:0000624");
        for a in ["CL:0000624", "CL:0000084", "CL:0000542", "CL:0000000"] {
            assert!(anc.contains(a), "missing ancestor {a}");
        }
        assert!(!anc.contains("CL:0000236"));
    }

    #[test]
    fn part_of_only_followed_on_demand() {
        let f = write_obo();
        let onto = Ontology::load_obo(f.path().to_str().unwrap()).unwrap();

        // is_a-only walk does NOT cross the part_of edge to the compartment.
        let isa = onto.ancestors_or_self("CL:0000625");
        assert!(isa.contains("CL:0000084"), "is_a ancestor missing");
        assert!(
            !isa.contains("CL:1000000"),
            "part_of must not leak into is_a-only walk"
        );

        // is_a + part_of walk reaches the compartment.
        let full = onto.ancestors_or_self_with_part_of("CL:0000625");
        assert!(full.contains("CL:0000084"), "is_a ancestor missing");
        assert!(full.contains("CL:1000000"), "part_of ancestor missing");
    }
}
