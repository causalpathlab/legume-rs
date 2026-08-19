//! Which endpoint of a spatial edge is the sender.
//!
//! The test this module feeds is directional: the ligand is scored on the
//! send-side pseudobulk, the receptor on the recv-side, and only the ligand
//! side is permuted. So the orientation is not a presentational choice, it is
//! the estimand.
//!
//! It used to come from the edge list's own ordering. A `KnnGraph` emits every
//! edge once as canonical `(min_index, max_index)`, so "sender" meant "the
//! endpoint with the smaller column index in the expression matrix", which is
//! an artifact of barcode load order and carries no biology.
//!
//! Two repairs suggest themselves and both are wrong.
//!
//! **Orienting by the pair's own expression is circular.** Keeping whichever
//! direction scores higher for the ligand-receptor pair under test makes the
//! observed statistic a maximum over `2^E` orientations, while the null
//! permutes at the sample level, after the pseudobulks are built, and so never
//! re-selects. The null is then not matched to the statistic and the p-values
//! come out anticonservative. This is exactly why the plot layer may re-derive
//! an arrow direction from expression and this module may not.
//!
//! **Symmetrizing is valid but abandons the estimand.** Include both
//! orientations of every edge in one stratum and the send side equals the recv
//! side, so the statistic collapses to "do L and R covary here", with no
//! direction in it.
//!
//! Direction needs an asymmetry source, and it must be one the pair under test
//! cannot see. This module uses **community identity**: every cell has a
//! dominant link community, so an edge runs from community `a` to community
//! `b`, and `a -> b` and `b -> a` become two strata that are each tested. The
//! orientation is a function of the partition alone, never of `L` or `R`, so
//! the permutation null stays exactly valid; and every edge between the same
//! two communities orients the same way, so there is no per-edge coin flip.

use crate::util::common::*;
use rustc_hash::FxHashMap;

/// A directed community pair, and the cell-level assignment it rests on.
///
/// Strata are stored CSR-style: `pairs` is sorted and deduplicated, and
/// `items[offsets[s]..offsets[s + 1]]` lists the edges realizing stratum `s`
/// with the flag saying whether the stored `(i, j)` has to be swapped so `i`
/// sends. Bucketing once at resolve costs one pass over the edges; scanning
/// them per stratum instead would cost one pass per (batch, stratum) pair,
/// which is where the old shape spent its time.
pub struct DirectedStrata {
    /// `(sender, receiver)` per stratum, sorted, so the id is a function of
    /// the partition rather than of the order edges were serialized.
    pairs: Vec<(u32, u32)>,
    offsets: Vec<usize>,
    items: Vec<(u32, bool)>,
}

impl DirectedStrata {
    /// Derive the strata from the edge list, taking each cell's community to
    /// be the mode of its incident edges'.
    ///
    /// That mode is the argmax of the propensity row `compute_node_membership`
    /// builds and `lc` writes to `propensity.parquet`. It is recomputed here
    /// rather than read so `lra` keeps working against an edge list with no
    /// sibling propensity file; [`Self::new`] takes the assignment directly
    /// for a caller that would rather supply the artifact.
    ///
    /// Ties go to the lowest community index, which is arbitrary but stable,
    /// and a tie means the cell sits between two communities in any case.
    pub fn from_edge_modes(
        edges: &[(usize, usize, u32, Option<Box<str>>)],
        n_cells: usize,
    ) -> Self {
        let mut counts: Vec<FxHashMap<u32, usize>> =
            (0..n_cells).map(|_| FxHashMap::default()).collect();
        for &(i, j, k, _) in edges {
            *counts[i].entry(k).or_insert(0) += 1;
            *counts[j].entry(k).or_insert(0) += 1;
        }
        let dominant: Vec<u32> = counts
            .into_iter()
            .map(|m| {
                m.into_iter()
                    // `max_by_key` keeps the LAST maximum, so negate the
                    // community index to make the lowest one win a tie.
                    .max_by_key(|&(c, n)| (n, std::cmp::Reverse(c)))
                    .map_or(u32::MAX, |(c, _)| c)
            })
            .collect();
        Self::new(dominant, edges)
    }

    /// Build the strata from a per-cell community assignment.
    ///
    /// `dominant[i] == u32::MAX` marks a cell with no community; its edges sit
    /// out entirely.
    pub fn new(dominant: Vec<u32>, edges: &[(usize, usize, u32, Option<Box<str>>)]) -> Self {
        // A heterotypic edge realizes `a -> b` and `b -> a`; a homotypic one
        // realizes its single self stratum twice, once each way, which is what
        // makes that stratum symmetric.
        let realized = |i: usize, j: usize| -> Option<((u32, u32), (u32, u32))> {
            let (a, b) = (dominant[i], dominant[j]);
            (a != u32::MAX && b != u32::MAX).then_some(((a, b), (b, a)))
        };

        let mut counts: FxHashMap<(u32, u32), usize> = FxHashMap::default();
        for &(i, j, _, _) in edges {
            if let Some((ab, ba)) = realized(i, j) {
                *counts.entry(ab).or_insert(0) += 1;
                *counts.entry(ba).or_insert(0) += 1;
            }
        }
        let mut pairs: Vec<(u32, u32)> = counts.keys().copied().collect();
        pairs.sort_unstable();
        let index: FxHashMap<(u32, u32), usize> =
            pairs.iter().enumerate().map(|(s, &p)| (p, s)).collect();

        let mut offsets = vec![0usize; pairs.len() + 1];
        for (s, p) in pairs.iter().enumerate() {
            offsets[s + 1] = offsets[s] + counts[p];
        }
        let mut cursor = offsets.clone();
        let mut items = vec![(0u32, false); offsets[pairs.len()]];
        for (e, &(i, j, _, _)) in edges.iter().enumerate() {
            if let Some((ab, ba)) = realized(i, j) {
                for (key, flipped) in [(ab, false), (ba, true)] {
                    let s = index[&key];
                    items[cursor[s]] = (e as u32, flipped);
                    cursor[s] += 1;
                }
            }
        }
        Self {
            pairs,
            offsets,
            items,
        }
    }

    #[must_use]
    pub fn n_strata(&self) -> usize {
        self.pairs.len()
    }

    #[must_use]
    pub fn pair(&self, stratum: usize) -> (u32, u32) {
        self.pairs[stratum]
    }

    /// A stratum whose two endpoints share a community. The statistic is
    /// symmetric there by construction, so a direction must not be read off
    /// it. Kept rather than dropped because it is the homotypic baseline.
    #[must_use]
    pub fn is_homotypic(&self, stratum: usize) -> bool {
        let (a, b) = self.pairs[stratum];
        a == b
    }

    #[must_use]
    pub fn edges_in(&self, stratum: usize) -> usize {
        self.offsets[stratum + 1] - self.offsets[stratum]
    }

    /// How the stratum reads in a label: `C3` when both endpoints share a
    /// community, `C3->C7` when the test was directional.
    #[must_use]
    pub fn label(&self, stratum: usize) -> String {
        let (a, b) = self.pairs[stratum];
        if a == b {
            format!("C{a}")
        } else {
            format!("C{a}->C{b}")
        }
    }

    /// Which edges realize a stratum, and whether the stored `(i, j)` has to
    /// be swapped so that `i` is the sender.
    ///
    /// Indices rather than cells, so a caller keeps access to the edge's batch
    /// label, which it still has to filter on.
    #[must_use]
    pub fn oriented(&self, stratum: usize) -> &[(u32, bool)] {
        &self.items[self.offsets[stratum]..self.offsets[stratum + 1]]
    }

    /// Per-cell soft membership in each role, over the directed strata.
    ///
    /// Reads the bucketed strata, so a heterotypic edge lands in `a -> b` with
    /// its `a` endpoint sending and in `b -> a` with its `b` endpoint sending,
    /// and a homotypic edge lands in its one stratum both ways.
    #[must_use]
    pub fn role_memberships(
        &self,
        edges: &[(usize, usize, u32, Option<Box<str>>)],
        n_cells: usize,
    ) -> (Mat, Mat) {
        let n = self.n_strata();
        let mut p_send = Mat::zeros(n_cells, n);
        let mut p_recv = Mat::zeros(n_cells, n);
        for s in 0..n {
            for &(e, flipped) in self.oriented(s) {
                let (i, j, _, _) = edges[e as usize];
                let (send, recv) = if flipped { (j, i) } else { (i, j) };
                p_send[(send, s)] += 1.0;
                p_recv[(recv, s)] += 1.0;
            }
        }
        // Column-outer, because `Mat` is column-major: a row-wise sum-then-scale
        // strides by `n_cells` and takes a cache miss per element once the
        // stratum count is in the hundreds.
        let mut send_tot = vec![0.0f32; n_cells];
        let mut recv_tot = vec![0.0f32; n_cells];
        for s in 0..n {
            for i in 0..n_cells {
                send_tot[i] += p_send[(i, s)];
                recv_tot[i] += p_recv[(i, s)];
            }
        }
        for s in 0..n {
            for i in 0..n_cells {
                if send_tot[i] > 0.0 {
                    p_send[(i, s)] /= send_tot[i];
                }
                if recv_tot[i] > 0.0 {
                    p_recv[(i, s)] /= recv_tot[i];
                }
            }
        }
        (p_send, p_recv)
    }
}
