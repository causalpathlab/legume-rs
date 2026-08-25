//! Per-gene gated wrapper around `graph_embedding_util`'s chain sampler.
//!
//! The sampled UNIT is a finest-level super-cell (PB): edge ids here are
//! PB-PB super-edge ids, weights are per-gene activity sums folded onto
//! those super edges, and the chain's group maps are PB->parent labels.
//! The machinery is id-agnostic, so nothing below cares — but every
//! "edge" in this module is a super edge, never a cell pair.
//!
//! Precompute the per-(gene, batch) positive distribution once at start
//! (sorted-intersection of gene-active super edges with the batch's
//! retained super edges, plus a `WeightedIndex` weighted by the folded
//! activity), then on every sample call just look up the cached entry
//! and delegate to `loss::sample_unit_chain_batch_with_pos`. The
//! chain-aware sibling negative pools live on the underlying
//! `PerBatchUnitSampler` and are reused unchanged.

use graph_embedding_util::loss::{
    sample_unit_chain_batch_with_pos, PerBatchUnitSampler, UnitChainBatch, UnitChainBatchArgs,
    UnitChainBatchStats,
};
use rand::Rng;
use rand_distr::weighted::WeightedIndex;
use rayon::prelude::*;

use crate::cell_activity_graph_embedding::gene_gating::GeneActiveEdges;

////////////////////
// GeneExpBatchCache //
////////////////////

/// Precomputed positive distribution for one `(gene, batch)` pair.
pub struct GeneExpBatchEntry {
    /// `WeightedIndex` over the gene-batch intersected super-edge list.
    /// Weights are per-super-edge SUMS of the fine endpoint products
    /// `a_g[u] · a_g[v]` (or uniform when the sum would underflow).
    pub pos: WeightedIndex<f32>,
    /// Maps each local index in `pos` back to the global SUPER-EDGE id
    /// in `PbFrame::super_edges`. Required by
    /// `sample_unit_chain_batch_with_pos`.
    pub local_to_global: Vec<u32>,
}

/// Per-gene per-batch precomputed positive distributions. Built once at
/// the start of training; consumed (by reference) on every sample call.
///
/// Memory: O(n_genes · n_batches · mean_active_edges_per_gene), typically
/// 50-300 MB for an 18k-gene Visium dataset with one batch. Released
/// after training completes.
pub struct GeneExpBatchCache {
    /// `entries[gene][batch_idx]` — `None` when the gene has no active
    /// edges within that batch (sampling skips this pair).
    pub entries: Vec<Vec<Option<GeneExpBatchEntry>>>,
}

/// Build the per-(gene, batch) cache by intersecting each gene's
/// active-edge list with each batch's retained-edge list (both sorted
/// ascending → linear merge). Rayon over genes; per-gene cost is
/// O(n_batches · (|gene_active_edges| + |batch_edges|)).
pub fn build_gene_exp_batch_cache(
    activities: &GeneActiveEdges,
    samplers_per_exp_batch: &[Option<PerBatchUnitSampler>],
    activity_alpha: f32,
) -> GeneExpBatchCache {
    let n_genes = activities.gene_active_edges.len();
    let n_batches = samplers_per_exp_batch.len();
    let entries: Vec<Vec<Option<GeneExpBatchEntry>>> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let gene_edges = &activities.gene_active_edges[g];
            let gene_weights = &activities.gene_active_edge_weights[g];
            (0..n_batches)
                .map(|b| {
                    build_entry(
                        gene_edges,
                        gene_weights,
                        samplers_per_exp_batch[b].as_ref(),
                        activity_alpha,
                    )
                })
                .collect()
        })
        .collect();
    GeneExpBatchCache { entries }
}

fn build_entry(
    gene_edges: &[u32],
    gene_weights: &[f32],
    batch_sampler: Option<&PerBatchUnitSampler>,
    activity_alpha: f32,
) -> Option<GeneExpBatchEntry> {
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
                // Stage-2 coverage exponent (bge alpha_pb analog). α=1 keeps
                // the activity-proportional draw exactly; α<1 flattens toward
                // uniform over the gene's active edges. Identity fast-path so
                // the default is bit-for-bit the old behaviour.
                let w = if activity_alpha == 1.0 {
                    gene_weights[i]
                } else {
                    gene_weights[i].powf(activity_alpha)
                };
                local_weights.push(w);
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
    Some(GeneExpBatchEntry {
        pos,
        local_to_global: local_edges,
    })
}

impl GeneExpBatchCache {
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

///////////////////////////
// GeneGatedChainSampler //
///////////////////////////

pub struct GeneGatedChainSampler<'a> {
    pub super_edges: &'a [(u32, u32)],
    pub samplers_per_exp_batch: &'a [Option<PerBatchUnitSampler>],
    pub cache: &'a GeneExpBatchCache,
    pub unit_to_group_per_level: &'a [&'a [usize]],
    /// Positives drawn per `(gene, batch)`, the same for every gene.
    ///
    /// Deliberately ONE scalar. A per-gene budget was tried, on the theory that
    /// a gene active on 20 edges should not draw as hard as one active on 3000
    /// (`senna bge` weights its gene draws by count). It cannot work here: the
    /// loss reduces each gene to the MEAN over its positives
    /// (`cage_nce_loss_per_level`), so a gene's expected gradient contribution
    /// does not depend on how many it drew — only its variance does. bge's
    /// weighting changes how OFTEN a gene is drawn, i.e. how many gradient
    /// steps it gets, which is a different axis entirely. If that is wanted
    /// here, repeat genes in the epoch's visit order rather than varying this.
    /// Positive edges drawn per (gene, EXPERIMENTAL batch) draw.
    /// Not an SGD minibatch size: see `UnitChainBatchArgs::n_positives`.
    pub positives_per_draw: usize,
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
    ) -> Option<(UnitChainBatch, UnitChainBatchStats)> {
        let entry = self.cache.entries.get(gene)?.get(batch_idx)?.as_ref()?;
        let batch_sampler = self.samplers_per_exp_batch.get(batch_idx)?.as_ref()?;

        let args = UnitChainBatchArgs {
            edges: self.super_edges,
            batch_sampler,
            n_positives: self.positives_per_draw,
            n_negatives: self.n_negatives,
            unit_to_group_per_level: self.unit_to_group_per_level,
        };
        Some(sample_unit_chain_batch_with_pos(
            args,
            &entry.pos,
            &entry.local_to_global,
            rng,
        ))
    }
}
