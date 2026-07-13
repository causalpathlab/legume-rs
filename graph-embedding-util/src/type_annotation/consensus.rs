//! **Which partition do we report?**
//!
//! Leiden is a stochastic local optimiser over an objective with many near-equal optima, so the
//! partition it returns is a *draw*, not an answer. The bootstrap already knows this — it reseeds
//! Leiden on every replicate and ships the consensus **label**. But the cluster-level outputs
//! (`community`, the cluster × term `p`/`q`/`Q` matrices, the per-community calls) were still read
//! off one arbitrary partition: the one that happened to fall out of `--seed`.
//!
//! There is no need to run anything extra to do better. The bootstrap **already computed `B` of
//! them**. So instead of taking the first, take the most *typical* — the medoid:
//!
//! ```text
//! medoid = argmin_b  Σ_{b' ≠ b}  ( 1 − ARI(P_b, P_b') )
//! ```
//!
//! the partition with the smallest mean distance to all the others. It is the ensemble's centre of
//! mass, it is an actual partition (not an averaged object that no clustering would produce), and
//! choosing it costs one pass over pairs of vectors we are already holding.
//!
//! **Why the medoid rather than the best-scoring partition.** Picking `argmax` of Leiden's own
//! objective would report whichever seed the optimiser got luckiest on — selecting on the same
//! noisy criterion that produced the spread. That is the winner's curse, in the same shape it
//! takes in [`super::marker_bootstrap`], and it would make the reported partition *less*
//! reproducible, not more: the max of `B` noisy scores moves around more than a typical draw does.
//! The medoid selects on **agreement with the other draws**, which is the thing we actually want it
//! to be representative of.
//!
//! The mean ARI to the rest comes back with it, and is worth reading: it says how much the reported
//! partition is worth. At `0.9` the clustering is essentially determined and the medoid is a
//! formality; at `0.5` no single partition means much and the cluster-level outputs should be read
//! as one draw among many, whatever we print.
//!
//! # Cross-run reproducibility (now solved)
//!
//! `community` **is** reproducible between runs now. All `B` partitions within one run are Leiden
//! re-seeds of **one kNN graph**, and that graph is deterministic (matrix-util's seeded
//! instant-distance backend), so a second run rebuilds the identical graph and re-derives the
//! identical partition. Under the old un-seedable `hnsw_rs` backend this was false — the graph
//! differed by ~9% of edges between builds (edge Jaccard 0.91), and three runs agreed on
//! `community` at only ARI 0.91–0.93. That source is gone.
//!
//! So read this module for what it is: within a run, Leiden's choice among near-equal modularity
//! optima is still arbitrary, so the reported partition is the ensemble's centre (medoid) rather
//! than an arbitrary member of it, and it arrives with a stability number attached.

use super::term_ora::Partition;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[cfg(test)]
mod tests;

/// Cells used to compare two partitions.
///
/// ARI is `O(n)` per pair and there are `B(B−1)/2` pairs — at `B = 200`, `n = 15 000` that is
/// 3 × 10⁸ contingency updates for a *selection* that only has to get the ranking right. A
/// deterministic stride of a few thousand cells estimates every ARI to well inside the gaps that
/// separate candidate partitions, for ~2% of the cost.
const MAX_CELLS: usize = 4096;

/// The chosen partition, how well it agrees with the rest, and how well a *typical* member does.
///
/// The gap between the last two is exactly what the choice bought: if the ensemble's average
/// pairwise ARI is already 0.98, every partition is the medoid and this was a formality.
pub(super) struct Medoid {
    /// Index into `partitions`.
    pub best: usize,
    /// Mean ARI of the chosen partition to the other `B − 1`.
    pub agreement: f64,
    /// Mean ARI over *all* pairs — what an arbitrary draw would score on average.
    pub ensemble_mean: f64,
}

/// The most typical of `partitions`.
///
/// A single partition is its own medoid, perfectly.
pub(super) fn medoid(partitions: &[Partition]) -> Medoid {
    let b = partitions.len();
    if b < 2 {
        return Medoid {
            best: 0,
            agreement: 1.0,
            ensemble_mean: 1.0,
        };
    }
    let n = partitions[0].0.len();
    let stride = n.div_ceil(MAX_CELLS).max(1);
    let sub: Vec<Vec<usize>> = partitions
        .iter()
        .map(|(p, _)| p.iter().step_by(stride).copied().collect())
        .collect();

    // Mean ARI of each partition to all the others. The matrix is symmetric, so compute the upper
    // triangle once and scatter both ways.
    let pairs: Vec<(usize, usize, f64)> = (0..b)
        .into_par_iter()
        .flat_map_iter(|i| (i + 1..b).map(move |j| (i, j)))
        .map(|(i, j)| (i, j, ari(&sub[i], &sub[j])))
        .collect();

    let mut total = vec![0f64; b];
    for &(i, j, a) in &pairs {
        total[i] += a;
        total[j] += a;
    }
    let mean: Vec<f64> = total.iter().map(|&t| t / (b - 1) as f64).collect();
    let best = (0..b)
        .max_by(|&x, &y| mean[x].total_cmp(&mean[y]))
        .unwrap_or(0);
    Medoid {
        best,
        agreement: mean[best],
        ensemble_mean: pairs.iter().map(|&(_, _, a)| a).sum::<f64>() / pairs.len() as f64,
    }
}

/// Adjusted Rand Index between two labelings of the same cells.
///
/// Rand counts the pairs of cells the two partitions agree about (together in both, or apart in
/// both); the *adjustment* subtracts the agreement two independent partitions with these same
/// cluster sizes would reach by chance, so `0` is chance and `1` is identity. That correction is
/// what makes the number comparable across the ensemble at all: raw agreement rises with the
/// number of clusters, and Leiden's cluster count is itself one of the things that moves between
/// seeds.
fn ari(a: &[usize], b: &[usize]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    if a.len() < 2 {
        return 1.0;
    }
    let mut joint: FxHashMap<(u32, u32), f64> = FxHashMap::default();
    let mut ra: FxHashMap<u32, f64> = FxHashMap::default();
    let mut rb: FxHashMap<u32, f64> = FxHashMap::default();
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as u32, y as u32);
        *joint.entry((x, y)).or_default() += 1.0;
        *ra.entry(x).or_default() += 1.0;
        *rb.entry(y).or_default() += 1.0;
    }
    // Pairs within each cell of the contingency table, and within each margin.
    let choose2 = |v: f64| v * (v - 1.0) / 2.0;
    let s_ij: f64 = joint.values().copied().map(choose2).sum();
    let s_a: f64 = ra.values().copied().map(choose2).sum();
    let s_b: f64 = rb.values().copied().map(choose2).sum();
    let expected = s_a * s_b / choose2(n);
    let max = 0.5 * (s_a + s_b);
    if (max - expected).abs() < f64::EPSILON {
        // Both partitions put every cell in one cluster (or every cell alone): chance agreement is
        // total agreement, ARI is 0/0. They are identical, so say so.
        return 1.0;
    }
    (s_ij - expected) / (max - expected)
}
