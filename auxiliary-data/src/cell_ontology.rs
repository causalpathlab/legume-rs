//! Cell Ontology (`cl-basic.obo`) `is_a` DAG, parsed with `fastobo` and stored
//! as a `petgraph` directed graph (edges point **child → parent**).
//!
//! This is a thin, reusable wrapper: parse once, then query ancestors and the
//! sub-DAG *induced* by a small set of leaf terms (e.g. the cell-type labels of
//! a marker file). Downstream annotation builds its hypothesis tree on the
//! induced sub-DAG — we never flatten or walk the full ~3,500-term ontology.
//!
//! Obsolete terms and `is_a` targets outside the term set are dropped, and the
//! `{is_inferred="true"}` qualifier blocks fastobo handles natively (a hand
//! rolled parser missing them was the original prototype's bug).

use anyhow::{anyhow, Context, Result};
use fastobo::ast::{EntityFrame, TermClause};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction::{Incoming, Outgoing};
use rustc_hash::{FxHashMap, FxHashSet};

/// Parsed Cell Ontology `is_a` DAG. Edges are `child → parent`, so a node's
/// ancestors are reachable along `Outgoing` edges and its children along
/// `Incoming` edges.
pub struct CellOntology {
    graph: DiGraph<Box<str>, ()>,
    idx: FxHashMap<Box<str>, NodeIndex>,
    names: FxHashMap<Box<str>, Box<str>>,
}

impl CellOntology {
    /// Parse an OBO file into the `is_a` DAG. Non-`CL:` terms, obsolete terms,
    /// and `is_a` edges to unknown/obsolete targets are skipped.
    pub fn load_obo(path: &str) -> Result<Self> {
        let doc = fastobo::from_file(path)
            .with_context(|| format!("failed to parse OBO file: {path}"))?;

        // Pass 1: collect non-obsolete terms (id, name, is_a parents).
        let mut terms: Vec<(Box<str>, Option<Box<str>>, Vec<Box<str>>)> = Vec::new();
        for frame in doc.entities() {
            let EntityFrame::Term(term) = frame else {
                continue;
            };
            // fastobo's `Line<_>` Display can carry a trailing newline / inline
            // comment, so trim every extracted token down to the bare value.
            let id: Box<str> = term.id().to_string().trim().into();
            if !id.starts_with("CL:") {
                continue;
            }
            let mut name: Option<Box<str>> = None;
            let mut parents: Vec<Box<str>> = Vec::new();
            let mut obsolete = false;
            for line in term.clauses() {
                match &**line {
                    TermClause::Name(n) => name = Some(n.to_string().trim().into()),
                    TermClause::IsObsolete(b) => obsolete = obsolete || *b,
                    TermClause::IsA(parent) => {
                        let p = parent.to_string().trim().to_string();
                        if p.starts_with("CL:") {
                            parents.push(p.into());
                        }
                    }
                    _ => {}
                }
            }
            if !obsolete {
                terms.push((id, name, parents));
            }
        }

        // Pass 2: nodes first, then edges (only between known nodes).
        let mut graph: DiGraph<Box<str>, ()> = DiGraph::new();
        let mut idx: FxHashMap<Box<str>, NodeIndex> = FxHashMap::default();
        let mut names: FxHashMap<Box<str>, Box<str>> = FxHashMap::default();
        for (id, name, _) in &terms {
            let node = graph.add_node(id.clone());
            idx.insert(id.clone(), node);
            if let Some(n) = name {
                names.insert(id.clone(), n.clone());
            }
        }
        for (id, _, parents) in &terms {
            let child = idx[id];
            for p in parents {
                if let Some(&parent) = idx.get(p) {
                    graph.add_edge(child, parent, ());
                }
            }
        }

        Ok(Self { graph, idx, names })
    }

    /// Number of (non-obsolete `CL:`) terms.
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

    /// All ancestors of `id` plus `id` itself (empty if `id` is unknown).
    #[must_use]
    pub fn ancestors_or_self(&self, id: &str) -> FxHashSet<Box<str>> {
        let mut seen = FxHashSet::default();
        let Some(&start) = self.idx.get(id) else {
            return seen;
        };
        let mut stack = vec![start];
        seen.insert(self.graph[start].clone());
        while let Some(n) = stack.pop() {
            for parent in self.graph.neighbors_directed(n, Outgoing) {
                if seen.insert(self.graph[parent].clone()) {
                    stack.push(parent);
                }
            }
        }
        seen
    }

    /// Sub-DAG node set induced by `leaves`: the leaves and all their ancestors.
    /// Errors if any leaf is absent from the ontology (so a typo in a `label→CL`
    /// map fails loudly rather than silently dropping a cell type).
    pub fn induced_nodes(&self, leaves: &[Box<str>]) -> Result<FxHashSet<Box<str>>> {
        let mut induced = FxHashSet::default();
        for leaf in leaves {
            if !self.idx.contains_key(leaf) {
                return Err(anyhow!("CL term not found in ontology: {leaf}"));
            }
            induced.extend(self.ancestors_or_self(leaf));
        }
        Ok(induced)
    }

    /// Children of `id` (direct `is_a` descendants) restricted to `induced`.
    /// Because `induced` is ancestor-closed, direct edges suffice — Steiner
    /// chains are kept (single-child pass-through nodes), never collapsed.
    #[must_use]
    pub fn induced_children(&self, id: &str, induced: &FxHashSet<Box<str>>) -> Vec<Box<str>> {
        let Some(&node) = self.idx.get(id) else {
            return Vec::new();
        };
        let mut out: Vec<Box<str>> = self
            .graph
            .neighbors_directed(node, Incoming)
            .map(|c| self.graph[c].clone())
            .filter(|c| induced.contains(c))
            .collect();
        out.sort_unstable(); // deterministic order
        out
    }

    /// Direct parents of `id` restricted to `induced`.
    #[must_use]
    pub fn induced_parents(&self, id: &str, induced: &FxHashSet<Box<str>>) -> Vec<Box<str>> {
        let Some(&node) = self.idx.get(id) else {
            return Vec::new();
        };
        let mut out: Vec<Box<str>> = self
            .graph
            .neighbors_directed(node, Outgoing)
            .map(|p| self.graph[p].clone())
            .filter(|p| induced.contains(p))
            .collect();
        out.sort_unstable();
        out
    }

    /// Roots of the induced sub-DAG: induced nodes with no induced parent.
    #[must_use]
    pub fn induced_roots(&self, induced: &FxHashSet<Box<str>>) -> Vec<Box<str>> {
        let mut roots: Vec<Box<str>> = induced
            .iter()
            .filter(|id| self.induced_parents(id, induced).is_empty())
            .cloned()
            .collect();
        roots.sort_unstable();
        roots
    }

    /// Post-order over the induced sub-DAG (every node appears after all its
    /// induced descendants — i.e. children before parents). Suitable for
    /// bottom-up combination (e.g. Simes). Deterministic.
    #[must_use]
    pub fn induced_postorder(&self, induced: &FxHashSet<Box<str>>) -> Vec<Box<str>> {
        let mut order: Vec<Box<str>> = Vec::with_capacity(induced.len());
        let mut visited: FxHashSet<Box<str>> = FxHashSet::default();
        for root in self.induced_roots(induced) {
            self.postorder_visit(&root, induced, &mut visited, &mut order);
        }
        // Any node not reached from a root (shouldn't happen for ancestor-closed
        // sets) is appended so the result still covers `induced`.
        let mut leftover: Vec<Box<str>> = induced
            .iter()
            .filter(|n| !visited.contains(*n))
            .cloned()
            .collect();
        leftover.sort_unstable();
        order.extend(leftover);
        order
    }

    fn postorder_visit(
        &self,
        id: &str,
        induced: &FxHashSet<Box<str>>,
        visited: &mut FxHashSet<Box<str>>,
        order: &mut Vec<Box<str>>,
    ) {
        if visited.contains(id) {
            return;
        }
        visited.insert(id.into());
        for child in self.induced_children(id, induced) {
            self.postorder_visit(&child, induced, visited, order);
        }
        order.push(id.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal CL-shaped OBO: cell → lymphocyte → T → {CD4, CD8}; B sibling;
    /// one obsolete term; one `{is_inferred="true"}` qualifier on an is_a line.
    fn write_obo() -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            f,
            "format-version: 1.2\n\n\
             [Term]\nid: CL:0000000\nname: cell\n\n\
             [Term]\nid: CL:0000542\nname: lymphocyte\nis_a: CL:0000000 ! cell\n\n\
             [Term]\nid: CL:0000084\nname: T cell\nis_a: CL:0000542 {{is_inferred=\"true\"}} ! lymphocyte\n\n\
             [Term]\nid: CL:0000624\nname: CD4 T\nis_a: CL:0000084 ! T cell\n\n\
             [Term]\nid: CL:0000625\nname: CD8 T\nis_a: CL:0000084 ! T cell\n\n\
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
        let onto = CellOntology::load_obo(f.path().to_str().unwrap()).unwrap();

        // obsolete term dropped; 6 live terms.
        assert_eq!(onto.len(), 6);
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
    fn induced_subdag_structure() {
        let f = write_obo();
        let onto = CellOntology::load_obo(f.path().to_str().unwrap()).unwrap();

        let leaves: Vec<Box<str>> = ["CL:0000624", "CL:0000625", "CL:0000236"]
            .iter()
            .map(|s| (*s).into())
            .collect();
        let induced = onto.induced_nodes(&leaves).unwrap();
        // leaves + T + lymphocyte + cell = 6 (here all live terms).
        assert!(induced.contains("CL:0000084") && induced.contains("CL:0000000"));

        assert_eq!(onto.induced_roots(&induced), vec!["CL:0000000".into()]);
        assert_eq!(
            onto.induced_children("CL:0000084", &induced),
            vec!["CL:0000624".into(), "CL:0000625".into()]
        );
        // lymphocyte's induced children = {T cell, B cell}.
        assert_eq!(
            onto.induced_children("CL:0000542", &induced),
            vec!["CL:0000084".into(), "CL:0000236".into()]
        );

        // post-order: every node after its children.
        let order = onto.induced_postorder(&induced);
        let pos = |id: &str| order.iter().position(|n| &**n == id).unwrap();
        assert!(pos("CL:0000624") < pos("CL:0000084"));
        assert!(pos("CL:0000084") < pos("CL:0000542"));
        assert!(pos("CL:0000542") < pos("CL:0000000"));
    }

    #[test]
    fn unknown_leaf_errors() {
        let f = write_obo();
        let onto = CellOntology::load_obo(f.path().to_str().unwrap()).unwrap();
        let leaves: Vec<Box<str>> = vec!["CL:1234567".into()];
        assert!(onto.induced_nodes(&leaves).is_err());
    }
}
