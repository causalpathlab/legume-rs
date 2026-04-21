//! Graph-constrained DC-Poisson refinement for pinto's coarsened partitions.
//!
//! Pinto-specific front-end over the shared Poisson scoring core in
//! [`data_beans_alg::dc_poisson`]. Supplies:
//! - [`GraphProposer`] — candidates from spatial-KNN neighbors ∩ siblings.
//! - [`ConnectivityGuard`] — rejects moves that would disconnect the
//!   source cluster's induced subgraph on the entity axis.
//! - [`DcPoissonContext`] — bundles the reusable per-axis state (entity
//!   map, IDF-weighted profiles, guard) so it is built once for the whole
//!   coarsening dendrogram and reused per level.
//!
//! Entry points: [`DcPoissonContext::build`] + [`refine_level_dc_poisson`].

use crate::util::common::*;
use crate::util::knn_graph::KnnGraph;
use data_beans_alg::dc_poisson::{
    compact_labels, intersect_with_siblings_fallback, refine_with_proposer_guarded,
    CandidateProposer, MoveGuard, Profiles, RefineParams,
};
use matrix_util::utils::generate_minibatch_intervals;
use rand::rngs::SmallRng;
use std::cell::RefCell;
use std::collections::VecDeque;

/// Proposes candidates = (graph-adjacent groups) ∩ siblings per entity,
/// with sibling fallback when the intersection is empty.
pub struct GraphProposer {
    siblings: Vec<Vec<usize>>,
    neighbor_groups: Vec<Vec<usize>>,
}

impl GraphProposer {
    /// Gather per-entity graph-adjacent group labels by projecting cell-level
    /// KNN edges through `cell_to_entity`.
    pub fn new(
        graph: &KnnGraph,
        cell_to_entity: &[usize],
        entity_labels: &[usize],
        siblings: Vec<Vec<usize>>,
    ) -> Self {
        let num_entities = siblings.len();
        let mut neighbor_groups: Vec<Vec<usize>> = vec![Vec::new(); num_entities];

        for &(ci, cj) in &graph.edges {
            let ei = cell_to_entity[ci];
            let ej = cell_to_entity[cj];
            if ei == ej {
                continue;
            }
            neighbor_groups[ei].push(entity_labels[ej]);
            neighbor_groups[ej].push(entity_labels[ei]);
        }
        for v in neighbor_groups.iter_mut() {
            v.sort_unstable();
            v.dedup();
        }

        Self {
            siblings,
            neighbor_groups,
        }
    }
}

impl CandidateProposer for GraphProposer {
    fn propose(&self, labels: &[usize]) -> Vec<Vec<usize>> {
        (0..self.siblings.len())
            .map(|e| {
                intersect_with_siblings_fallback(
                    &self.siblings[e],
                    &self.neighbor_groups[e],
                    labels[e],
                )
            })
            .collect()
    }
}

/// Reusable scratch buffers for the articulation-point BFS.
#[derive(Default)]
struct ArticulationScratch {
    nbrs_in_cluster: Vec<usize>,
    visited: HashSet<usize>,
    remaining: HashSet<usize>,
    queue: VecDeque<usize>,
}

impl ArticulationScratch {
    fn reset(&mut self) {
        self.nbrs_in_cluster.clear();
        self.visited.clear();
        self.remaining.clear();
        self.queue.clear();
    }
}

/// Reject moves that would disconnect the source cluster's induced subgraph
/// over entities. Mirrors the per-cell check in
/// [`crate::util::graph_refine::would_disconnect_cluster`] but on the
/// coarsened entity axis.
pub struct ConnectivityGuard {
    entity_adj: Vec<Vec<usize>>,
    scratch: RefCell<ArticulationScratch>,
}

impl ConnectivityGuard {
    /// Build entity-level adjacency by projecting cell-level edges through
    /// `cell_to_entity`. Two entities are neighbors iff some KNN edge
    /// crosses between them.
    pub fn new(graph: &KnnGraph, cell_to_entity: &[usize], num_entities: usize) -> Self {
        let mut entity_adj: Vec<Vec<usize>> = vec![Vec::new(); num_entities];
        for &(ci, cj) in &graph.edges {
            let ei = cell_to_entity[ci];
            let ej = cell_to_entity[cj];
            if ei == ej {
                continue;
            }
            entity_adj[ei].push(ej);
            entity_adj[ej].push(ei);
        }
        for v in entity_adj.iter_mut() {
            v.sort_unstable();
            v.dedup();
        }
        Self {
            entity_adj,
            scratch: RefCell::new(ArticulationScratch::default()),
        }
    }
}

impl MoveGuard for ConnectivityGuard {
    fn accept_move(&self, e: usize, from: usize, labels: &[usize]) -> bool {
        let mut scratch = self.scratch.borrow_mut();
        let s = &mut *scratch;
        s.reset();

        s.nbrs_in_cluster.extend(
            self.entity_adj[e]
                .iter()
                .copied()
                .filter(|&j| labels[j] == from),
        );

        if s.nbrs_in_cluster.len() <= 1 {
            return true;
        }

        let start = s.nbrs_in_cluster[0];
        s.visited.insert(start);
        s.queue.push_back(start);
        // Split borrow: disjoint fields of the same scratch struct.
        s.remaining.extend(s.nbrs_in_cluster[1..].iter().copied());

        while let Some(node) = s.queue.pop_front() {
            for &nb in &self.entity_adj[node] {
                if nb == e || labels[nb] != from || !s.visited.insert(nb) {
                    continue;
                }
                s.remaining.remove(&nb);
                if s.remaining.is_empty() {
                    return true;
                }
                s.queue.push_back(nb);
            }
        }

        s.remaining.is_empty()
    }
}

/// Build per-entity sparse gene sums by streaming cells in blocks.
///
/// Stays sparse throughout: each block accumulates a per-entity
/// `Vec<(gene, value)>` stream directly from the cell's CSC row indices,
/// then sorts + merges duplicate genes. Block partials are merged
/// sparse-to-sparse. Avoids the `Vec<f32>[num_genes]`-per-entity dense
/// accumulators the naïve approach would allocate.
pub fn build_entity_gene_sums(
    data: &SparseIoVec,
    cell_to_entity: &[usize],
    num_entities: usize,
    num_genes: usize,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
    let n_cells = data.num_columns();
    let jobs = generate_minibatch_intervals(n_cells, num_genes, block_size);

    let partials: Vec<Vec<Vec<(u32, f32)>>> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<Vec<Vec<(u32, f32)>>> {
            let x = data.read_columns_csc(lb..ub)?;
            let mut local: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_entities];
            for local_col in 0..x.ncols() {
                let cell = lb + local_col;
                let e = cell_to_entity[cell];
                if e >= num_entities {
                    continue;
                }
                let s = x.col(local_col);
                let bucket = &mut local[e];
                bucket.reserve(s.nnz());
                for (&row, &val) in s.row_indices().iter().zip(s.values().iter()) {
                    bucket.push((row as u32, val));
                }
            }
            // Sort + coalesce duplicate genes within each entity's block buffer.
            for bucket in local.iter_mut() {
                if bucket.len() > 1 {
                    bucket.sort_unstable_by_key(|&(g, _)| g);
                    let mut write = 0;
                    for read in 1..bucket.len() {
                        if bucket[read].0 == bucket[write].0 {
                            bucket[write].1 += bucket[read].1;
                        } else {
                            write += 1;
                            bucket[write] = bucket[read];
                        }
                    }
                    bucket.truncate(write + 1);
                }
            }
            Ok(local)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Merge block partials per entity via sorted-sparse merge.
    let mut out: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_entities];
    for mut block in partials {
        for (e, bucket) in block.iter_mut().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            out[e] = merge_sorted_sparse(&out[e], bucket);
        }
    }
    Ok(out)
}

/// Merge two gene-sorted sparse vectors into a single gene-sorted sparse
/// vector with summed values at shared gene indices.
fn merge_sorted_sparse(a: &[(usize, f32)], b: &[(u32, f32)]) -> Vec<(usize, f32)> {
    if a.is_empty() {
        return b.iter().map(|&(g, v)| (g as usize, v)).collect();
    }
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (ga, va) = a[i];
        let (gb, vb) = (b[j].0 as usize, b[j].1);
        match ga.cmp(&gb) {
            std::cmp::Ordering::Less => {
                out.push((ga, va));
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push((gb, vb));
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push((ga, va + vb));
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend(b[j..].iter().map(|&(g, v)| (g as usize, v)));
    out
}

/// Compute per-entity sibling sets at one level.
///
/// Coarsest level (`parent_labels == None`): every entity's siblings = all
/// groups. Finer levels: siblings for entity `e` = other entities sharing
/// the same parent group, projected through `entity_labels` onto the
/// current level's group axis.
fn compute_siblings_per_entity(
    entity_labels: &[usize],
    parent_labels: Option<&[usize]>,
    k: usize,
) -> Vec<Vec<usize>> {
    let num_entities = entity_labels.len();
    let Some(parents) = parent_labels else {
        let all_groups: Vec<usize> = (0..k).collect();
        return vec![all_groups; num_entities];
    };
    debug_assert_eq!(parents.len(), num_entities);

    let mut parent_to_children_groups: HashMap<usize, Vec<usize>> = HashMap::default();
    for (e, &p) in parents.iter().enumerate() {
        let entry = parent_to_children_groups.entry(p).or_default();
        let g = entity_labels[e];
        if !entry.contains(&g) {
            entry.push(g);
        }
    }
    for v in parent_to_children_groups.values_mut() {
        v.sort_unstable();
    }

    (0..num_entities)
        .map(|e| {
            parent_to_children_groups
                .get(&parents[e])
                .cloned()
                .unwrap_or_default()
        })
        .collect()
}

/// Reusable per-axis state for DC-Poisson refinement across all levels.
///
/// Entities are fixed across the coarsening dendrogram (typically the
/// finest-cut cluster IDs); only their group assignment varies by level.
/// So the IDF-weighted profiles and entity-level adjacency can be built
/// once and reused for every level's refinement sweep.
pub struct DcPoissonContext<'a> {
    pub graph: &'a KnnGraph,
    pub cell_to_entity: Vec<usize>,
    pub profiles: Profiles,
    pub guard: ConnectivityGuard,
}

impl<'a> DcPoissonContext<'a> {
    /// Build the per-axis state once. `idf_weighting` matches
    /// [`RefineParams::idf_weighting`] — kept explicit here so a
    /// context can be constructed without a full [`RefineParams`].
    pub fn build(
        data: &SparseIoVec,
        graph: &'a KnnGraph,
        cell_to_entity: Vec<usize>,
        num_entities: usize,
        num_genes: usize,
        idf_weighting: bool,
    ) -> anyhow::Result<Self> {
        let gene_sums =
            build_entity_gene_sums(data, &cell_to_entity, num_entities, num_genes, None)?;
        let mut profiles = Profiles::from_gene_sums(&gene_sums, num_genes);
        if idf_weighting {
            let bg = profiles.empirical_marginal();
            profiles.weight_by_idf(&bg);
        }
        let guard = ConnectivityGuard::new(graph, &cell_to_entity, num_entities);
        Ok(Self {
            graph,
            cell_to_entity,
            profiles,
            guard,
        })
    }
}

/// Refine `entity_labels` in place at one coarsening level.
///
/// `parent_labels`, when provided, gives each entity's group at the
/// next-coarser level; moves are restricted to siblings sharing the same
/// parent. Pass `None` at the coarsest level.
pub fn refine_level_dc_poisson(
    ctx: &DcPoissonContext<'_>,
    entity_labels: &mut [usize],
    parent_labels: Option<&[usize]>,
    params: &RefineParams,
    rng: &mut SmallRng,
    level_label: &str,
) -> usize {
    let (compact, k) = compact_labels(entity_labels);
    entity_labels.copy_from_slice(&compact);
    if k <= 1 {
        return 0;
    }

    let siblings = compute_siblings_per_entity(entity_labels, parent_labels, k);
    let proposer = GraphProposer::new(ctx.graph, &ctx.cell_to_entity, entity_labels, siblings);

    let moves = refine_with_proposer_guarded(
        &ctx.profiles,
        entity_labels,
        k,
        &proposer,
        &ctx.guard,
        params,
        rng,
        level_label,
    );

    let (compact, _new_k) = compact_labels(entity_labels);
    entity_labels.copy_from_slice(&compact);

    moves
}
