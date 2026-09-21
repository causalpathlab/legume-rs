//! One typed graph for the joint embedding: the peak→gene link relation, the
//! pb × feature count relations of every level (SIMBA's binned recipe, one
//! relation per expression level), and the child → parent edges.
//!
//! Node types: `region`, `gene`, then `pb@l` per level (finest first). Uniform
//! negatives stay inside a type, so a coarse pb never corrupts a fine one.

use crate::common::Mat;
use crate::p2g::link_map::PeakGeneEdge;
use crate::p2g::pb_levels::PbLevels;
use graph_embedding_util::fne::{
    NodeTypeTable, Relation, RelationPolarity, RelationTable, TypedEdgeList,
};
use graph_embedding_util::simba::{log_norm, Discretization, RelationTable as LevelWeights};

/// Relation name of the peak→gene link edges.
pub const LINK_RELATION: &str = "region:gene/link";

/// Node type index of the peaks, the genes, and the first level's pbs.
const REGION_T: u16 = 0;
const GENE_T: u16 = 1;
const PB_T0: u16 = 2;

/// The assembled graph plus the repeat counts that balance its relations.
pub struct ContextGraph {
    pub types: NodeTypeTable,
    pub rels: RelationTable,
    pub edges: TypedEdgeList,
    /// `FneConfig::relation_repeats`, by relation index.
    pub repeats: Vec<usize>,
}

/// Build the joint graph from link edges (peak/gene indices) and the levels.
/// `bins` is the number of SIMBA expression levels per modality.
pub fn build_context_graph(
    link: &[PeakGeneEdge],
    levels: &PbLevels,
    n_peaks: usize,
    n_genes: usize,
    bins: usize,
) -> anyhow::Result<ContextGraph> {
    anyhow::ensure!(n_peaks > 0 && n_genes > 0, "need ≥1 peak and ≥1 gene");
    anyhow::ensure!(bins > 0, "need ≥1 expression bin");
    let n_levels = levels.n_levels();
    anyhow::ensure!(n_levels > 0, "no pb level");
    anyhow::ensure!(
        levels.parent.len() + 1 == n_levels,
        "{} parent maps for {n_levels} levels",
        levels.parent.len()
    );
    if let Some(rna) = &levels.rna {
        anyhow::ensure!(rna.len() == n_levels, "RNA levels != ATAC levels");
    }
    for l in 0..n_levels {
        anyhow::ensure!(
            levels.atac[l].nrows() == n_peaks,
            "ATAC level {l} has {} rows, expected {n_peaks} peaks",
            levels.atac[l].nrows()
        );
        if let Some(rna) = &levels.rna {
            anyhow::ensure!(
                rna[l].nrows() == n_genes && rna[l].ncols() == levels.atac[l].ncols(),
                "RNA level {l} is {}×{}, expected {n_genes}×{}",
                rna[l].nrows(),
                rna[l].ncols(),
                levels.atac[l].ncols()
            );
        }
    }
    anyhow::ensure!(
        link.iter().all(|e| e.weight.is_finite() && e.weight > 0.0),
        "link weights must be finite and positive"
    );

    let pb_names: Vec<String> = (0..n_levels).map(|l| format!("pb@{l}")).collect();
    let mut spec = vec![("region", n_peaks), ("gene", n_genes)];
    spec.extend(
        pb_names
            .iter()
            .map(String::as_str)
            .zip(levels.n_pb_per_level()),
    );
    let types = NodeTypeTable::new(&spec)?;
    let pb_t = |l: usize| PB_T0 + l as u16;
    let friend = |name: String, lhs: u16, rhs: u16, weight: f32| Relation {
        name: name.into(),
        lhs_type: lhs,
        rhs_type: rhs,
        weight,
        undirected: false,
        polarity: RelationPolarity::Friend,
    };

    let mut relations = vec![friend(LINK_RELATION.into(), REGION_T, GENE_T, 1.0)];
    let mut edges = EdgeBuilder::default();
    for e in link {
        edges.push(
            types.global(REGION_T as usize, e.peak as u32),
            types.global(GENE_T as usize, e.gene as u32),
            0,
            e.weight,
        );
    }

    // Count relations, one modality at a time: one discretization over every
    // level's nonzero log-normalized entries, one relation per (level, bin)
    // present, then the edges. Three passes over the matrices, nothing held.
    let mut modalities: Vec<(&str, u16, &[Mat])> = Vec::new();
    if let Some(rna) = &levels.rna {
        modalities.push(("rna", GENE_T, rna.as_slice()));
    }
    modalities.push(("atac", REGION_T, levels.atac.as_slice()));
    for (label, feat_t, mats) in modalities {
        let disc = Discretization::fit_values(LogNormValues::new(mats), bins)?;
        for (l, m) in mats.iter().enumerate() {
            let mut present = vec![false; disc.n_levels() + 1];
            for_each_log_norm(m, |_, _, v| present[disc.level(v) as usize] = true);
            let present: Vec<u8> = (1..present.len())
                .filter(|&k| present[k])
                .map(|k| k as u8)
                .collect();
            let weights = LevelWeights::from_levels(&present);
            let first_rel = relations.len() as u16;
            for (&k, &w) in weights.levels.iter().zip(&weights.weights) {
                relations.push(friend(
                    format!("pb@{l}:{}/{label}@{k}", types.name(feat_t as usize)),
                    pb_t(l),
                    feat_t,
                    w,
                ));
            }
            edges.reserve(m.iter().filter(|&&x| x > 0.0).count());
            for_each_log_norm(m, |f, s, v| {
                edges.push(
                    types.global(pb_t(l) as usize, s),
                    types.global(feat_t as usize, f),
                    first_rel + weights.rel(disc.level(v)) as u16,
                    1.0,
                );
            });
        }
    }

    // Child → parent, one relation per consecutive level pair.
    for (l, map) in levels.parent.iter().enumerate() {
        anyhow::ensure!(
            map.len() == levels.n_pb(l),
            "parent map {l} has {} entries for {} pbs",
            map.len(),
            levels.n_pb(l)
        );
        let r = relations.len() as u16;
        relations.push(friend(
            format!("pb@{l}:pb@{}/parent", l + 1),
            pb_t(l),
            pb_t(l + 1),
            1.0,
        ));
        for (child, &parent) in map.iter().enumerate() {
            edges.push(
                types.global(pb_t(l) as usize, child as u32),
                types.global(pb_t(l + 1) as usize, parent as u32),
                r,
                1.0,
            );
        }
    }

    let rels = RelationTable::new(relations, &types)?;
    let repeats = balance_repeats(&edges.counts, &rels);
    Ok(ContextGraph {
        types,
        rels,
        edges: edges.finish(),
        repeats,
    })
}

/// Edge arrays plus the per-relation count they accumulate.
#[derive(Default)]
struct EdgeBuilder {
    lhs: Vec<u32>,
    rhs: Vec<u32>,
    rel: Vec<u16>,
    weight: Vec<f32>,
    counts: Vec<usize>,
}

impl EdgeBuilder {
    fn reserve(&mut self, n: usize) {
        self.lhs.reserve(n);
        self.rhs.reserve(n);
        self.rel.reserve(n);
        self.weight.reserve(n);
    }

    fn push(&mut self, lhs: u32, rhs: u32, rel: u16, weight: f32) {
        self.lhs.push(lhs);
        self.rhs.push(rhs);
        self.rel.push(rel);
        self.weight.push(weight);
        let r = rel as usize;
        if self.counts.len() <= r {
            self.counts.resize(r + 1, 0);
        }
        self.counts[r] += 1;
    }

    fn finish(self) -> TypedEdgeList {
        TypedEdgeList {
            lhs: self.lhs,
            rhs: self.rhs,
            rel: self.rel,
            weight: Some(self.weight),
        }
    }
}

/// `f(feature, sample, log_norm)` over the nonzero entries of one profile,
/// column by column (library size = the column sum).
fn for_each_log_norm(m: &Mat, mut f: impl FnMut(u32, u32, f32)) {
    for (s, col) in m.column_iter().enumerate() {
        let lib: f64 = col.iter().map(|&x| f64::from(x)).sum();
        if lib <= 0.0 {
            continue;
        }
        for (r, &x) in col.iter().enumerate() {
            if x > 0.0 {
                f(r as u32, s as u32, log_norm(x, lib));
            }
        }
    }
}

/// The nonzero log-normalized values of several profiles as a re-iterable
/// stream (what `Discretization::fit_values` scans twice).
#[derive(Clone)]
struct LogNormValues<'a> {
    mats: &'a [Mat],
    m: usize,
    s: usize,
    r: usize,
    lib: f64,
}

impl<'a> LogNormValues<'a> {
    fn new(mats: &'a [Mat]) -> Self {
        let mut it = Self {
            mats,
            m: 0,
            s: 0,
            r: 0,
            lib: 0.0,
        };
        it.lib = it.column_lib();
        it
    }

    fn column_lib(&self) -> f64 {
        self.mats
            .get(self.m)
            .filter(|m| self.s < m.ncols())
            .map_or(0.0, |m| {
                m.column(self.s).iter().map(|&x| f64::from(x)).sum()
            })
    }

    /// Step to the next column (or matrix), refreshing the library size.
    fn next_column(&mut self) {
        self.r = 0;
        self.s += 1;
        if self.s >= self.mats[self.m].ncols() {
            self.s = 0;
            self.m += 1;
        }
        self.lib = self.column_lib();
    }
}

impl Iterator for LogNormValues<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        while self.m < self.mats.len() {
            let m = &self.mats[self.m];
            while self.r < m.nrows() {
                let x = m[(self.r, self.s)];
                self.r += 1;
                if x > 0.0 && self.lib > 0.0 {
                    return Some(log_norm(x, self.lib));
                }
            }
            self.next_column();
        }
        None
    }
}

/// A relation is a count relation when it joins a pb type to a feature type.
fn is_count(r: &Relation) -> bool {
    r.lhs_type >= PB_T0 && r.rhs_type < PB_T0
}

/// Repeats that lift every non-count relation (the link and the parent edges)
/// to about the update share of one count relation.
fn balance_repeats(counts: &[usize], rels: &RelationTable) -> Vec<usize> {
    let count_of = |r: usize| counts.get(r).copied().unwrap_or(0);
    let count_sizes: Vec<usize> = (0..rels.len())
        .filter(|&r| is_count(rels.get(r)) && count_of(r) > 0)
        .map(count_of)
        .collect();
    if count_sizes.is_empty() {
        return vec![1; rels.len()];
    }
    let share = count_sizes.iter().sum::<usize>() as f64 / count_sizes.len() as f64;
    (0..rels.len())
        .map(|r| {
            let n = count_of(r);
            if is_count(rels.get(r)) || n == 0 {
                1
            } else {
                ((share / n as f64).round() as usize).max(1)
            }
        })
        .collect()
}
