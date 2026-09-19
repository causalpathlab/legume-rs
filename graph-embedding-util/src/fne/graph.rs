//! The typed graph: node types as contiguous id ranges of one flat table,
//! relations as `(lhs type, rhs type, weight)`, edges as a structure of
//! arrays over global node ids.
//!
//! Every node lives in exactly one type and every type owns a contiguous
//! block of the embedding table, so a relation's uniform negatives are a
//! draw over one range and a batch's gathers are one `index_select` per
//! side. A relation's weight is a per-relation constant (PBG's
//! `relation.weight`); a per-edge weight, when present, multiplies it row
//! by row.

use rand::{Rng, RngExt};
use std::ops::Range;

/// Node types as contiguous ranges `offsets[t]..offsets[t + 1]` of the
/// flat table.
#[derive(Clone, Debug)]
pub struct NodeTypeTable {
    names: Vec<Box<str>>,
    offsets: Vec<u32>,
}

impl NodeTypeTable {
    /// Lay the types out in the given order; every type must have a name
    /// and at least one node.
    pub fn new(types: &[(&str, usize)]) -> anyhow::Result<Self> {
        anyhow::ensure!(!types.is_empty(), "fne: no node types");
        anyhow::ensure!(
            types.len() <= u16::MAX as usize,
            "fne: {} node types exceed the u16 index",
            types.len()
        );
        let mut offsets = Vec::with_capacity(types.len() + 1);
        let mut names: Vec<Box<str>> = Vec::with_capacity(types.len());
        let mut total = 0usize;
        offsets.push(0u32);
        for (name, n) in types {
            anyhow::ensure!(*n > 0, "fne: node type `{name}` has no nodes");
            anyhow::ensure!(
                names.iter().all(|m| &**m != *name),
                "fne: node type `{name}` declared twice"
            );
            total += n;
            anyhow::ensure!(
                total <= u32::MAX as usize,
                "fne: {total} nodes exceed the u32 index"
            );
            offsets.push(total as u32);
            names.push(Box::from(*name));
        }
        Ok(Self { names, offsets })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    #[must_use]
    pub fn name(&self, t: usize) -> &str {
        &self.names[t]
    }

    #[must_use]
    pub fn names(&self) -> &[Box<str>] {
        &self.names
    }

    /// Global ids of type `t`.
    #[must_use]
    pub fn range(&self, t: usize) -> Range<u32> {
        self.offsets[t]..self.offsets[t + 1]
    }

    #[must_use]
    pub fn n_nodes(&self, t: usize) -> usize {
        (self.offsets[t + 1] - self.offsets[t]) as usize
    }

    #[must_use]
    pub fn n_total(&self) -> usize {
        *self.offsets.last().expect("offsets never empty") as usize
    }

    /// Type index of a global id.
    #[must_use]
    pub fn type_of(&self, global: u32) -> usize {
        // offsets is ascending; the last offset ≤ global is the type.
        self.offsets.partition_point(|&o| o <= global) - 1
    }

    /// Type index by name.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n.as_ref() == name)
    }

    /// `(type, local id)` of a global id.
    #[must_use]
    pub fn local(&self, global: u32) -> (usize, u32) {
        let t = self.type_of(global);
        (t, global - self.offsets[t])
    }

    #[must_use]
    pub fn global(&self, t: usize, local: u32) -> u32 {
        self.offsets[t] + local
    }
}

/// Whether linked endpoints should attract (high Dot) or repel (low Dot).
/// Names are geometric only — callers choose polarity; FNE does not map
/// biological labels (e.g. genetic interaction sign) onto these variants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RelationPolarity {
    /// Softmax-NCE on Dot: the observed pair should rank above negatives.
    #[default]
    Friend,
    /// Softmax-NCE on −Dot: the observed pair should rank below negatives.
    Enemy,
}

impl RelationPolarity {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Friend => "friend",
            Self::Enemy => "enemy",
        }
    }

    /// Enemy scores −Dot into softmax-NCE; friend leaves Dot unchanged.
    #[must_use]
    pub fn flips_dot(self) -> bool {
        matches!(self, Self::Enemy)
    }

    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "friend" => Ok(Self::Friend),
            "enemy" => Ok(Self::Enemy),
            other => anyhow::bail!("unknown polarity `{other}` (expected friend or enemy)"),
        }
    }
}

/// One relation of the graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Relation {
    pub name: Box<str>,
    pub lhs_type: u16,
    pub rhs_type: u16,
    /// PBG's per-relation loss weight; 1.0 unless the caller grades its
    /// relations.
    pub weight: f32,
    /// Both endpoints are the same type and the pair is unordered. Metadata
    /// for loaders and reports only: the loss corrupts both sides of every
    /// relation regardless, so reverse edges are never inserted.
    pub undirected: bool,
    /// Friend = attract (Dot NCE); Enemy = repel (anti-Dot NCE).
    pub polarity: RelationPolarity,
}

#[derive(Clone, Debug)]
pub struct RelationTable {
    relations: Vec<Relation>,
}

impl RelationTable {
    pub fn new(relations: Vec<Relation>, types: &NodeTypeTable) -> anyhow::Result<Self> {
        anyhow::ensure!(!relations.is_empty(), "fne: no relations");
        anyhow::ensure!(
            relations.len() <= u16::MAX as usize,
            "fne: {} relations exceed the u16 index",
            relations.len()
        );
        for r in &relations {
            anyhow::ensure!(
                (r.lhs_type as usize) < types.len() && (r.rhs_type as usize) < types.len(),
                "fne: relation `{}` names a node type outside the {} declared",
                r.name,
                types.len()
            );
            anyhow::ensure!(
                r.weight.is_finite() && r.weight >= 0.0,
                "fne: relation `{}` has weight {}",
                r.name,
                r.weight
            );
            anyhow::ensure!(
                !r.undirected || r.lhs_type == r.rhs_type,
                "fne: relation `{}` is undirected across two node types",
                r.name
            );
        }
        Ok(Self { relations })
    }

    #[must_use]
    pub fn get(&self, r: usize) -> &Relation {
        &self.relations[r]
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.relations.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Relation> {
        self.relations.iter()
    }
}

/// The edges as parallel arrays over GLOBAL node ids, so an epoch's
/// shuffle is an in-place permutation with no index vector.
#[derive(Clone, Debug, Default)]
pub struct TypedEdgeList {
    pub lhs: Vec<u32>,
    pub rhs: Vec<u32>,
    pub rel: Vec<u16>,
    /// Per-edge weight multiplying the relation weight; `None` ⇒ all 1.
    pub weight: Option<Vec<f32>>,
}

impl TypedEdgeList {
    #[must_use]
    pub fn len(&self) -> usize {
        self.lhs.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lhs.is_empty()
    }

    #[must_use]
    pub fn edge_weight(&self, i: usize) -> f32 {
        self.weight.as_ref().map_or(1.0, |w| w[i])
    }

    pub(crate) fn swap(&mut self, i: usize, j: usize) {
        self.lhs.swap(i, j);
        self.rhs.swap(i, j);
        self.rel.swap(i, j);
        if let Some(w) = self.weight.as_mut() {
            w.swap(i, j);
        }
    }

    /// Fisher–Yates over `range` only; edges outside it stay where they are.
    pub(crate) fn shuffle_range<R: Rng>(&mut self, range: Range<usize>, rng: &mut R) {
        let start = range.start;
        let n = range.end - range.start;
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            self.swap(start + i, start + j);
        }
    }

    /// Every id inside its relation's declared type range, every relation
    /// index inside the table, parallel arrays of one length.
    pub fn validate(&self, types: &NodeTypeTable, rels: &RelationTable) -> anyhow::Result<()> {
        let n = self.len();
        anyhow::ensure!(
            self.rhs.len() == n && self.rel.len() == n,
            "fne: edge arrays disagree in length"
        );
        if let Some(w) = &self.weight {
            anyhow::ensure!(w.len() == n, "fne: edge weights disagree in length");
            anyhow::ensure!(
                w.iter().all(|x| x.is_finite() && *x >= 0.0),
                "fne: an edge weight is negative or not finite"
            );
        }
        for i in 0..n {
            let r = self.rel[i] as usize;
            anyhow::ensure!(r < rels.len(), "fne: edge {i} names relation {r}");
            let rel = rels.get(r);
            anyhow::ensure!(
                types.range(rel.lhs_type as usize).contains(&self.lhs[i]),
                "fne: edge {i} lhs {} is not a `{}` node",
                self.lhs[i],
                types.name(rel.lhs_type as usize)
            );
            anyhow::ensure!(
                types.range(rel.rhs_type as usize).contains(&self.rhs[i]),
                "fne: edge {i} rhs {} is not a `{}` node",
                self.rhs[i],
                types.name(rel.rhs_type as usize)
            );
        }
        Ok(())
    }

    /// Edges per relation.
    #[must_use]
    pub fn counts_per_relation(&self, n_rel: usize) -> Vec<usize> {
        let mut c = vec![0usize; n_rel];
        for &r in &self.rel {
            c[r as usize] += 1;
        }
        c
    }

    /// Stable partition so relation `r`'s edges occupy one contiguous block;
    /// returns the block of each relation.
    pub(crate) fn group_by_relation(&mut self, n_rel: usize) -> Vec<Range<usize>> {
        // Counting sort: one pass to size the blocks, one to scatter.
        let counts = self.counts_per_relation(n_rel);
        let mut starts = Vec::with_capacity(n_rel);
        let mut acc = 0usize;
        for &c in &counts {
            starts.push(acc);
            acc += c;
        }
        let n = self.len();
        let mut cursor = starts.clone();
        let mut lhs = vec![0u32; n];
        let mut rhs = vec![0u32; n];
        let mut rel = vec![0u16; n];
        let mut weight = self.weight.as_ref().map(|_| vec![0f32; n]);
        for i in 0..n {
            let r = self.rel[i] as usize;
            let j = cursor[r];
            cursor[r] += 1;
            lhs[j] = self.lhs[i];
            rhs[j] = self.rhs[i];
            rel[j] = self.rel[i];
            if let (Some(out), Some(w)) = (weight.as_mut(), self.weight.as_ref()) {
                out[j] = w[i];
            }
        }
        self.lhs = lhs;
        self.rhs = rhs;
        self.rel = rel;
        self.weight = weight;
        (0..n_rel)
            .map(|r| starts[r]..starts[r] + counts[r])
            .collect()
    }
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod graph_tests;

/// `pbg_train(auto_wd=True)`: the weight decay SIMBA fits to the edge count,
/// scaled off two reference graphs (`0.013` at 2,725,781 edges below 5e7
/// edges, `0.0004` at 59,103,481 edges above), rounded to 6 decimals
/// (half-away-from-zero here vs numpy's half-to-even: a tie needs the 7th
/// decimal to be exactly 5, which no edge count of interest produces).
#[must_use]
pub fn auto_wd(n_edges: usize) -> f64 {
    let n = n_edges.max(1) as f64;
    let wd = if n < 5e7 {
        0.013 * 2_725_781.0 / n
    } else {
        0.0004 * 59_103_481.0 / n
    };
    (wd * 1e6).round() / 1e6
}
