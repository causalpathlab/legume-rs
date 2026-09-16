//! From edge files to a typed graph: nodes keyed by `(type, name)`, every
//! source one relation, edges deduplicated per relation.
//!
//! Two file shapes. A *pair* file is two gene names per line with an
//! optional weight, the classic PPI / co-expression list; each file is its
//! own relation `gene:gene/<stem>`. A *typed* file spells the node type of
//! both endpoints on every line, `lhs_type lhs rhs_type rhs [weight]`, and
//! its rows join the relation `lhs_type:rhs_type` whichever file they came
//! from. Gene names go through the caller's canonicaliser; every other type
//! is matched verbatim.

use auxiliary_data::feature_names::FeatureNameKind;
use auxiliary_data::gene_sets::{read_membership_pairs, GeneSets};
use auxiliary_data::ontology::{Ontology, Rel};
use genomic_data::coordinates::{parse_region, tile_windows};
use graph_embedding_util::fne::{NodeTypeTable, Relation, RelationTable, TypedEdgeList};
use log::{info, warn};
use matrix_util::common_io::read_lines_of_words_delim;
use matrix_util::membership::detect_delimiter;
use rustc_hash::FxHashMap;
use std::path::Path;

/// The node type whose names are canonicalised as gene symbols.
pub(crate) const GENE_TYPE: &str = "gene";
/// Ontology terms and gene sets.
pub(crate) const TERM_TYPE: &str = "term";
/// Fixed genomic windows.
pub(crate) const REGION_TYPE: &str = "region";

/// A node's text: a display name and a description, either optional.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeText {
    pub name: Option<Box<str>>,
    pub text: Option<Box<str>>,
}

/// `<stem>` of a path for relation names: the file name up to its first dot.
pub(crate) fn file_stem(path: &str) -> String {
    let stem = Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    stem.split('.')
        .next()
        .filter(|s| !s.is_empty())
        .map_or(stem.clone(), str::to_string)
}

struct TypeNodes {
    name: Box<str>,
    names: Vec<Box<str>>,
    index: FxHashMap<Box<str>, u32>,
    /// Text attached to some of the nodes, by local id.
    texts: FxHashMap<u32, NodeText>,
}

struct RelSpec {
    name: Box<str>,
    lhs_type: usize,
    rhs_type: usize,
    undirected: bool,
    weight: f32,
    /// `(lhs local, rhs local) → edge weight`; a repeated pair keeps the
    /// largest weight.
    edges: FxHashMap<(u32, u32), f32>,
    n_self_loops: usize,
    n_repeats: usize,
}

/// The assembled graph, ready for `graph_embedding_util::fne::train`.
pub(crate) struct TypedGraph {
    pub types: NodeTypeTable,
    pub relations: RelationTable,
    pub edges: TypedEdgeList,
    /// Node names in global-id order (type blocks in table order).
    pub node_names: Vec<Box<str>>,
    /// Node type name of every global id.
    pub node_types: Vec<Box<str>>,
    /// `(global id, text)` for every node that carries any, in id order.
    pub texts: Vec<(u32, NodeText)>,
}

pub(crate) struct TypedGraphBuilder {
    name_kind: FeatureNameKind,
    types: Vec<TypeNodes>,
    type_index: FxHashMap<Box<str>, usize>,
    relations: Vec<RelSpec>,
    rel_index: FxHashMap<Box<str>, usize>,
}

impl TypedGraphBuilder {
    pub(crate) fn new(name_kind: FeatureNameKind) -> Self {
        Self {
            name_kind,
            types: Vec::new(),
            type_index: FxHashMap::default(),
            relations: Vec::new(),
            rel_index: FxHashMap::default(),
        }
    }

    fn type_id(&mut self, ty: &str) -> usize {
        if let Some(&t) = self.type_index.get(ty) {
            return t;
        }
        let t = self.types.len();
        self.types.push(TypeNodes {
            name: ty.into(),
            names: Vec::new(),
            index: FxHashMap::default(),
            texts: FxHashMap::default(),
        });
        self.type_index.insert(ty.into(), t);
        t
    }

    /// `(type id, local id)` of a node, inserting it on first sight.
    fn node(&mut self, ty: &str, name: &str) -> (usize, u32) {
        let t = self.type_id(ty);
        let key: Box<str> = if ty == GENE_TYPE {
            self.name_kind.canonicalize(name)
        } else {
            name.into()
        };
        let nodes = &mut self.types[t];
        if let Some(&i) = nodes.index.get(&key) {
            return (t, i);
        }
        let i = nodes.names.len() as u32;
        nodes.names.push(key.clone());
        nodes.index.insert(key, i);
        (t, i)
    }

    fn relation(&mut self, name: &str, lhs: &str, rhs: &str) -> usize {
        if let Some(&r) = self.rel_index.get(name) {
            return r;
        }
        let lhs_type = self.type_id(lhs);
        let rhs_type = self.type_id(rhs);
        let r = self.relations.len();
        self.relations.push(RelSpec {
            name: name.into(),
            lhs_type,
            rhs_type,
            undirected: lhs == rhs,
            weight: 1.0,
            edges: FxHashMap::default(),
            n_self_loops: 0,
            n_repeats: 0,
        });
        self.rel_index.insert(name.into(), r);
        r
    }

    fn add_edge(&mut self, r: usize, lhs: u32, rhs: u32, weight: f32) {
        let rel = &mut self.relations[r];
        let key = if rel.undirected {
            if lhs == rhs {
                rel.n_self_loops += 1;
                return;
            }
            (lhs.min(rhs), lhs.max(rhs))
        } else {
            (lhs, rhs)
        };
        match rel.edges.get_mut(&key) {
            Some(w) => {
                rel.n_repeats += 1;
                if weight > *w {
                    *w = weight;
                }
            }
            None => {
                rel.edges.insert(key, weight);
            }
        }
    }

    /// A gene-gene pair file: `gene1 gene2 [weight]`, its own relation
    /// `gene:gene/<stem>`.
    pub(crate) fn add_pair_file(&mut self, path: &str) -> anyhow::Result<()> {
        let rel_name = format!("{GENE_TYPE}:{GENE_TYPE}/{}", file_stem(path));
        let r = self.relation(&rel_name, GENE_TYPE, GENE_TYPE);
        let read = read_lines_of_words_delim(path, detect_delimiter(path), -1)?;
        let mut n_rows = 0usize;
        for line in &read.lines {
            if line.len() < 2 || line[0].starts_with('#') {
                continue;
            }
            let weight = parse_weight(line.get(2).map(AsRef::as_ref), path)?;
            let (_, i) = self.node(GENE_TYPE, &line[0]);
            let (_, j) = self.node(GENE_TYPE, &line[1]);
            self.add_edge(r, i, j, weight);
            n_rows += 1;
        }
        let rel = &self.relations[r];
        info!(
            "fne: {path}: {n_rows} pair rows → relation `{}` with {} unique edges ({} self-loops, {} repeats dropped)",
            rel.name,
            rel.edges.len(),
            rel.n_self_loops,
            rel.n_repeats
        );
        Ok(())
    }

    /// A typed edge file: `lhs_type lhs rhs_type rhs [weight]`; rows join
    /// the relation `lhs_type:rhs_type`.
    pub(crate) fn add_typed_file(&mut self, path: &str) -> anyhow::Result<()> {
        let read = read_lines_of_words_delim(path, detect_delimiter(path), -1)?;
        let mut n_rows = 0usize;
        let mut n_short = 0usize;
        let mut touched: Vec<usize> = Vec::new();
        for line in &read.lines {
            if line.is_empty() || line[0].starts_with('#') {
                continue;
            }
            if line.len() < 4 {
                n_short += 1;
                continue;
            }
            let (lt, rt) = (line[0].as_ref(), line[2].as_ref());
            let rel_name = format!("{lt}:{rt}");
            let r = self.relation(&rel_name, lt, rt);
            let weight = parse_weight(line.get(4).map(AsRef::as_ref), path)?;
            let (_, i) = self.node(lt, &line[1]);
            let (_, j) = self.node(rt, &line[3]);
            self.add_edge(r, i, j, weight);
            if !touched.contains(&r) {
                touched.push(r);
            }
            n_rows += 1;
        }
        if n_short > 0 {
            warn!("fne: {path}: {n_short} rows had fewer than four columns and were skipped");
        }
        let names: Vec<String> = touched
            .iter()
            .map(|&r| format!("`{}`", self.relations[r].name))
            .collect();
        info!(
            "fne: {path}: {n_rows} typed rows → relation(s) {}",
            names.join(", ")
        );
        Ok(())
    }

    /// Attach text to a node (inserting it if new); a later call fills only
    /// the parts still missing.
    fn set_text(&mut self, ty: &str, name: &str, text: NodeText) {
        let (t, i) = self.node(ty, name);
        let slot = self.types[t].texts.entry(i).or_default();
        if slot.name.is_none() {
            slot.name = text.name;
        }
        if slot.text.is_none() {
            slot.text = text.text;
        }
    }

    /// A membership file `gene <TAB> label` → relation `gene:<ty>/<stem>`
    /// with the labels as nodes of type `ty`.
    pub(crate) fn add_membership_file(&mut self, ty: &str, path: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            !ty.is_empty() && ty != GENE_TYPE,
            "--membership: the label type must be a non-empty name other than `{GENE_TYPE}` (got `{ty}`)"
        );
        let pairs = read_membership_pairs(path)?;
        let rel_name = format!("{GENE_TYPE}:{ty}/{}", file_stem(path));
        let r = self.relation(&rel_name, GENE_TYPE, ty);
        for (gene, label) in &pairs {
            let (_, i) = self.node(GENE_TYPE, gene);
            let (_, j) = self.node(ty, label);
            self.add_edge(r, i, j, 1.0);
        }
        let rel = &self.relations[r];
        info!(
            "fne: {path}: {} membership rows → relation `{}` with {} unique edges over {} `{ty}` labels",
            pairs.len(),
            rel.name,
            rel.edges.len(),
            self.types[self.type_index[ty]].names.len()
        );
        Ok(())
    }

    /// Gene sets (a GAF after propagation, or a GMT) → relation
    /// `gene:term/<stem>`; sets outside `[min, max]` members are dropped
    /// (`max` 0 = no cap). Names and descriptions become term text.
    pub(crate) fn add_gene_sets(
        &mut self,
        sets: &GeneSets,
        stem: &str,
        min_members: usize,
        max_members: usize,
    ) -> usize {
        let rel_name = format!("{GENE_TYPE}:{TERM_TYPE}/{stem}");
        let r = self.relation(&rel_name, GENE_TYPE, TERM_TYPE);
        let mut terms: Vec<&Box<str>> = sets.term_genes.keys().collect();
        terms.sort();
        let mut n_kept = 0usize;
        for term in terms {
            let genes = &sets.term_genes[term];
            let n = genes.len();
            if n < min_members || (max_members > 0 && n > max_members) {
                continue;
            }
            n_kept += 1;
            let (_, j) = self.node(TERM_TYPE, term);
            let mut members: Vec<&Box<str>> = genes.iter().collect();
            members.sort();
            for g in members {
                let (_, i) = self.node(GENE_TYPE, g);
                self.add_edge(r, i, j, 1.0);
            }
            if let Some(desc) = sets.names.get(term) {
                self.set_text(
                    TERM_TYPE,
                    term,
                    NodeText {
                        name: Some(desc.clone()),
                        text: None,
                    },
                );
            }
        }
        let rel = &self.relations[r];
        info!(
            "fne: gene sets `{stem}`: {n_kept} of {} sets within [{min_members}, {}] members → relation `{}` with {} edges",
            sets.term_genes.len(),
            if max_members == 0 {
                "∞".to_string()
            } else {
                max_members.to_string()
            },
            rel.name,
            rel.edges.len()
        );
        n_kept
    }

    /// The ontology's hierarchy over the term nodes already in the graph:
    /// relations `term:term/is_a` and `term:term/part_of` (child → parent),
    /// plus the terms' names and definitions as text.
    pub(crate) fn add_ontology(&mut self, onto: &Ontology) {
        let Some(&t) = self.type_index.get(TERM_TYPE) else {
            warn!("fne: --obo given but no term nodes are in the graph; its hierarchy is skipped");
            return;
        };
        let present: Vec<Box<str>> = self.types[t].names.clone();
        for id in &present {
            let text = NodeText {
                name: onto.name(id).map(Box::from),
                text: onto.def(id).map(Box::from),
            };
            if text != NodeText::default() {
                self.set_text(TERM_TYPE, id, text);
            }
        }
        let r_is_a = self.relation(
            &format!("{TERM_TYPE}:{TERM_TYPE}/is_a"),
            TERM_TYPE,
            TERM_TYPE,
        );
        let r_part_of = self.relation(
            &format!("{TERM_TYPE}:{TERM_TYPE}/part_of"),
            TERM_TYPE,
            TERM_TYPE,
        );
        let index = &self.types[t].index;
        let mut edges: Vec<(u32, u32, Rel)> = onto
            .edges()
            .filter_map(|(child, parent, rel)| Some((*index.get(child)?, *index.get(parent)?, rel)))
            .collect();
        edges.sort_unstable_by_key(|(c, p, rel)| (*c, *p, matches!(rel, Rel::PartOf)));
        let n = edges.len();
        for (c, p, rel) in edges {
            let r = match rel {
                Rel::IsA => r_is_a,
                Rel::PartOf => r_part_of,
            };
            // Hierarchy edges are directed child → parent; the undirected
            // dedup of a same-type relation only folds the two directions.
            self.add_edge(r, c, p, 1.0);
        }
        info!(
            "fne: ontology hierarchy over {} present terms: {n} edges ({} is_a, {} part_of)",
            present.len(),
            self.relations[r_is_a].edges.len(),
            self.relations[r_part_of].edges.len()
        );
    }

    /// A region→gene link file `region gene [score]`; every region is tiled
    /// onto `window`-bp windows and each window links the gene in the
    /// relation `region:gene/<stem>` with the score as the edge weight.
    pub(crate) fn add_region_file(&mut self, path: &str, window: i64) -> anyhow::Result<()> {
        let rel_name = format!("{REGION_TYPE}:{GENE_TYPE}/{}", file_stem(path));
        let r = self.relation(&rel_name, REGION_TYPE, GENE_TYPE);
        let read = read_lines_of_words_delim(path, detect_delimiter(path), -1)?;
        let mut n_rows = 0usize;
        let mut n_bad = 0usize;
        for line in &read.lines {
            if line.len() < 2 || line[0].starts_with('#') {
                continue;
            }
            let Some(region) = parse_region(&line[0]) else {
                n_bad += 1;
                continue;
            };
            let weight = parse_weight(line.get(2).map(AsRef::as_ref), path)?;
            let (_, j) = self.node(GENE_TYPE, &line[1]);
            for w in tile_windows(&region, window) {
                let (_, i) = self.node(REGION_TYPE, &w.to_string());
                self.add_edge(r, i, j, weight);
            }
            n_rows += 1;
        }
        if n_bad > 0 {
            warn!(
                "fne: {path}: {n_bad} rows had a region that is not a coordinate and were skipped"
            );
        }
        let rel = &self.relations[r];
        info!(
            "fne: {path}: {n_rows} region rows on {}-bp windows → relation `{}` with {} edges over {} windows",
            window,
            rel.name,
            rel.edges.len(),
            self.types[self.type_index[REGION_TYPE]].names.len()
        );
        Ok(())
    }

    /// `name=weight` overrides; an unknown relation name is an error so a
    /// typo cannot silently leave a relation at 1.
    pub(crate) fn set_relation_weight(&mut self, spec: &str) -> anyhow::Result<()> {
        let (name, w) = spec
            .rsplit_once('=')
            .ok_or_else(|| anyhow::anyhow!("--relation-weight `{spec}`: expected `name=weight`"))?;
        let w: f32 = w
            .trim()
            .parse()
            .map_err(|e| anyhow::anyhow!("--relation-weight `{spec}`: {e}"))?;
        anyhow::ensure!(
            w.is_finite() && w >= 0.0,
            "--relation-weight `{spec}`: weight must be a non-negative number"
        );
        let known: Vec<&str> = self.relations.iter().map(|r| r.name.as_ref()).collect();
        let r = *self.rel_index.get(name.trim()).ok_or_else(|| {
            anyhow::anyhow!(
                "--relation-weight `{spec}`: no relation named `{}`; the run has {}",
                name.trim(),
                known.join(", ")
            )
        })?;
        self.relations[r].weight = w;
        Ok(())
    }

    pub(crate) fn n_edges(&self) -> usize {
        self.relations.iter().map(|r| r.edges.len()).sum()
    }

    /// Lay the types out as contiguous blocks and turn every edge into
    /// global ids. Relations that ended up with no edges are dropped.
    pub(crate) fn finish(self) -> anyhow::Result<TypedGraph> {
        anyhow::ensure!(
            self.n_edges() > 0,
            "fne: 0 usable edges across the input files"
        );
        let mut type_specs: Vec<(&str, usize)> = Vec::new();
        let mut node_names: Vec<Box<str>> = Vec::new();
        let mut node_types: Vec<Box<str>> = Vec::new();
        let mut texts: Vec<(u32, NodeText)> = Vec::new();
        // Types with no nodes (a relation declared them but every row was
        // dropped) are laid out with a placeholder count of zero and pruned.
        let mut kept_types: Vec<usize> = Vec::new();
        for (t, nodes) in self.types.iter().enumerate() {
            if nodes.names.is_empty() {
                continue;
            }
            kept_types.push(t);
            let offset = node_names.len() as u32;
            type_specs.push((&nodes.name, nodes.names.len()));
            let mut with_text: Vec<(&u32, &NodeText)> = nodes.texts.iter().collect();
            with_text.sort_unstable_by_key(|(i, _)| **i);
            texts.extend(with_text.into_iter().map(|(i, t)| (offset + *i, t.clone())));
            node_names.extend(nodes.names.iter().cloned());
            node_types.extend(std::iter::repeat_n(nodes.name.clone(), nodes.names.len()));
        }
        let types = NodeTypeTable::new(&type_specs)?;
        let type_pos = |t: usize| -> Option<usize> { kept_types.iter().position(|&k| k == t) };

        let mut relations = Vec::new();
        let mut edges = TypedEdgeList::default();
        let mut weights: Vec<f32> = Vec::new();
        for spec in &self.relations {
            if spec.edges.is_empty() {
                warn!("fne: relation `{}` has no edges and is dropped", spec.name);
                continue;
            }
            let (Some(lt), Some(rt)) = (type_pos(spec.lhs_type), type_pos(spec.rhs_type)) else {
                continue;
            };
            let r = relations.len() as u16;
            relations.push(Relation {
                name: spec.name.clone(),
                lhs_type: lt as u16,
                rhs_type: rt as u16,
                weight: spec.weight,
                undirected: spec.undirected,
            });
            // Sorted so the edge order is a function of the files, not of
            // the hash map.
            let mut pairs: Vec<(&(u32, u32), &f32)> = spec.edges.iter().collect();
            pairs.sort_unstable_by_key(|(k, _)| **k);
            for (&(i, j), &w) in pairs {
                edges.lhs.push(types.global(lt, i));
                edges.rhs.push(types.global(rt, j));
                edges.rel.push(r);
                weights.push(w);
            }
        }
        if weights.iter().any(|&w| w != 1.0) {
            edges.weight = Some(weights);
        }
        let relations = RelationTable::new(relations, &types)?;
        for (t, n) in &type_specs {
            info!("fne: node type `{t}`: {n} nodes");
        }
        Ok(TypedGraph {
            types,
            relations,
            edges,
            node_names,
            node_types,
            texts,
        })
    }
}

fn parse_weight(tok: Option<&str>, path: &str) -> anyhow::Result<f32> {
    match tok {
        None => Ok(1.0),
        Some(t) => {
            let w: f32 = t
                .parse()
                .map_err(|e| anyhow::anyhow!("{path}: weight column `{t}`: {e}"))?;
            anyhow::ensure!(
                w.is_finite() && w >= 0.0,
                "{path}: weight column `{t}` must be a non-negative number"
            );
            Ok(w)
        }
    }
}
