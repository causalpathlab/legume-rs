//! Per-gene gated wrapper around `graph_embedding_util`'s chain sampler.
//!
//! Precompute the per-(gene, batch) positive distribution once at start
//! (sorted-intersection of gene-active edges with the batch's retained
//! edges, plus a `WeightedIndex` weighted by `a_g[u]·a_g[v]`), then on
//! every sample call just look up the cached entry and delegate to
//! `loss::sample_cell_chain_batch_with_pos`. The chain-aware sibling
//! negative pools live on the underlying `PerBatchCellSampler` and are
//! reused unchanged.

use graph_embedding_util::loss::{
    sample_cell_chain_batch_with_pos, CellChainBatch, CellChainBatchArgs, CellChainBatchStats,
    PerBatchCellSampler,
};
use rand::Rng;
use rand_distr::weighted::WeightedIndex;
use rayon::prelude::*;

use crate::cell_activity_embedding::gene_gating::CellActivities;

////////////////////////////////////////////////////////////////
//                                                            //
// GeneBatchCache                                             //
//                                                            //
////////////////////////////////////////////////////////////////

/// Precomputed positive distribution for one `(gene, batch)` pair.
pub struct GeneBatchEntry {
    /// `WeightedIndex` over the gene-batch intersected edge list. Weights
    /// are `a_g[u] · a_g[v]` (or uniform when the product would underflow).
    pub pos: WeightedIndex<f32>,
    /// Maps each local index in `pos` back to the global edge id in
    /// `srt_cell_pairs.graph.edges`. Required by
    /// `sample_cell_chain_batch_with_pos`.
    pub local_to_global: Vec<u32>,
}

/// Per-gene per-batch precomputed positive distributions. Built once at
/// the start of training; consumed (by reference) on every sample call.
///
/// Memory: O(n_genes · n_batches · mean_active_edges_per_gene), typically
/// 50-300 MB for an 18k-gene Visium dataset with one batch. Released
/// after training completes.
pub struct GeneBatchCache {
    /// `entries[gene][batch_idx]` — `None` when the gene has no active
    /// edges within that batch (sampling skips this pair).
    pub entries: Vec<Vec<Option<GeneBatchEntry>>>,
}

/// Build the per-(gene, batch) cache by intersecting each gene's
/// active-edge list with each batch's retained-edge list (both sorted
/// ascending → linear merge). Rayon over genes; per-gene cost is
/// O(n_batches · (|gene_active_edges| + |batch_edges|)).
pub fn build_gene_batch_cache(
    activities: &CellActivities,
    per_batch: &[Option<PerBatchCellSampler>],
) -> GeneBatchCache {
    let n_genes = activities.gene_active_edges.len();
    let n_batches = per_batch.len();
    let entries: Vec<Vec<Option<GeneBatchEntry>>> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let gene_edges = &activities.gene_active_edges[g];
            let gene_weights = &activities.gene_active_edge_weights[g];
            (0..n_batches)
                .map(|b| build_entry(gene_edges, gene_weights, per_batch[b].as_ref()))
                .collect()
        })
        .collect();
    GeneBatchCache { entries }
}

fn build_entry(
    gene_edges: &[u32],
    gene_weights: &[f32],
    batch_sampler: Option<&PerBatchCellSampler>,
) -> Option<GeneBatchEntry> {
    let bs = batch_sampler?;
    if gene_edges.is_empty() {
        return None;
    }
    let batch_edges = &bs.edge_indices;
    // Linear-time intersection of two sorted u32 lists.
    let (mut i, mut j) = (0usize, 0usize);
    let mut local_edges = Vec::<u32>::new();
    let mut local_weights = Vec::<f32>::new();
    while i < gene_edges.len() && j < batch_edges.len() {
        let a = gene_edges[i];
        let b = batch_edges[j];
        match a.cmp(&b) {
            std::cmp::Ordering::Equal => {
                local_edges.push(a);
                local_weights.push(gene_weights[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    if local_edges.is_empty() {
        return None;
    }
    // Guard against an all-zero weight column (e.g., underflow on tiny
    // products). Fall back to uniform within the active set.
    let total: f32 = local_weights.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        local_weights.fill(1.0);
    }
    let pos = WeightedIndex::new(local_weights).ok()?;
    Some(GeneBatchEntry {
        pos,
        local_to_global: local_edges,
    })
}

impl GeneBatchCache {
    /// Diagnostic: count of `(gene, batch)` cells with a non-empty
    /// active-edge intersection. Useful for an early sanity log.
    pub fn n_active_pairs(&self) -> usize {
        self.entries
            .iter()
            .flat_map(|row| row.iter())
            .filter(|e| e.is_some())
            .count()
    }
}

////////////////////////////////////////////////////////////////
//                                                            //
// GeneGatedChainSampler                                      //
//                                                            //
////////////////////////////////////////////////////////////////

pub struct GeneGatedChainSampler<'a> {
    pub edges: &'a [(u32, u32)],
    pub per_batch: &'a [Option<PerBatchCellSampler>],
    pub cache: &'a GeneBatchCache,
    pub pb_maps: &'a [&'a [usize]],
    pub batch_size: usize,
    pub n_negatives: usize,
}

impl<'a> GeneGatedChainSampler<'a> {
    /// Returns `None` when the gene × batch has no cached active edges.
    /// Hot path: one cache lookup, one `WeightedIndex` borrow, one
    /// delegate call — no per-step allocation beyond what the chain
    /// sampler already needs.
    pub fn sample<R: Rng>(
        &self,
        gene: usize,
        batch_idx: usize,
        rng: &mut R,
    ) -> Option<(CellChainBatch, CellChainBatchStats)> {
        let entry = self.cache.entries.get(gene)?.get(batch_idx)?.as_ref()?;
        let batch_sampler = self.per_batch.get(batch_idx)?.as_ref()?;

        let args = CellChainBatchArgs {
            edges: self.edges,
            batch_sampler,
            batch_size: self.batch_size,
            n_negatives: self.n_negatives,
            pb_maps: self.pb_maps,
        };
        Some(sample_cell_chain_batch_with_pos(
            args,
            &entry.pos,
            &entry.local_to_global,
            rng,
        ))
    }
}
