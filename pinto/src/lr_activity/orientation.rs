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
pub struct DirectedStrata {
    /// `dominant[i]` is cell `i`'s most frequent incident-edge community.
    dominant: Vec<u32>,
    /// `(sender, receiver)` community per stratum, in first-seen edge order.
    pairs: Vec<(u32, u32)>,
    index: FxHashMap<(u32, u32), u32>,
    /// Edges realizing each stratum, for the sparsity filter and the log.
    edge_counts: Vec<usize>,
}

impl DirectedStrata {
    /// Derive the strata from the edge list.
    ///
    /// A cell's dominant community is the mode of its incident edges'
    /// communities, which is the argmax of the propensity row
    /// `compute_node_membership` builds, without needing to read that file.
    /// Ties go to the lowest community index, which is arbitrary but stable,
    /// and a tie means the cell sits between two communities in any case.
    pub fn resolve(edges: &[(usize, usize, u32, Option<Box<str>>)], n_cells: usize) -> Self {
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
                    // max_by_key keeps the LAST maximum, so negate the
                    // community index to make the lowest one win a tie.
                    .max_by_key(|&(c, n)| (n, std::cmp::Reverse(c)))
                    .map_or(u32::MAX, |(c, _)| c)
            })
            .collect();

        // Collect the realized pairs, then index them in SORTED order rather
        // than in the order edges happen to arrive. A stratum id is written to
        // the manifest and joined on downstream, so it has to be a function of
        // the partition and not of how the edge list was serialized. Sorting
        // also makes swapping the two endpoint columns of the input a no-op,
        // which is the property this whole module exists to provide.
        let mut seen: Vec<(u32, u32)> = Vec::new();
        for &(i, j, _, _) in edges {
            let (a, b) = (dominant[i], dominant[j]);
            if a == u32::MAX || b == u32::MAX {
                continue;
            }
            seen.push((a, b));
            if a != b {
                seen.push((b, a));
            }
        }
        let mut pairs: Vec<(u32, u32)> = seen.clone();
        pairs.sort_unstable();
        pairs.dedup();
        let index: FxHashMap<(u32, u32), u32> = pairs
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i as u32))
            .collect();
        let mut edge_counts = vec![0usize; pairs.len()];
        for p in seen {
            edge_counts[index[&p] as usize] += 1;
        }
        Self {
            dominant,
            pairs,
            index,
            edge_counts,
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
        self.edge_counts[stratum]
    }

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
    /// Returned as indices rather than cells so a caller keeps access to the
    /// edge's batch label, which it still has to filter on.
    #[must_use]
    pub fn oriented(
        &self,
        stratum: usize,
        edges: &[(usize, usize, u32, Option<Box<str>>)],
    ) -> Vec<(usize, bool)> {
        let (want_a, want_b) = self.pairs[stratum];
        let mut out = Vec::new();
        for (e, &(i, j, _, _)) in edges.iter().enumerate() {
            let (a, b) = (self.dominant[i], self.dominant[j]);
            if a == u32::MAX || b == u32::MAX {
                continue;
            }
            // Heterotypic: exactly one arm fires, and it decides the swap.
            // Homotypic (`want_a == want_b`): a same-community edge fires BOTH
            // arms and is listed once each way, which is what makes the self
            // stratum symmetric, while an edge leaving the community fires
            // neither and belongs to another stratum.
            if a == want_a && b == want_b {
                out.push((e, false));
            }
            if b == want_a && a == want_b {
                out.push((e, true));
            }
        }
        out
    }

    /// Per-cell soft membership in each role, over the directed strata.
    ///
    /// A heterotypic edge contributes to both `a -> b` (with the `a` endpoint
    /// sending) and `b -> a` (with the `b` endpoint sending), so the two
    /// directions are each estimated on the same edges and can be compared. A
    /// homotypic edge has one stratum, in which both endpoints take both roles.
    #[must_use]
    pub fn role_memberships(
        &self,
        edges: &[(usize, usize, u32, Option<Box<str>>)],
        n_cells: usize,
    ) -> (Mat, Mat) {
        let s = self.n_strata();
        let mut p_send = Mat::zeros(n_cells, s);
        let mut p_recv = Mat::zeros(n_cells, s);
        for &(i, j, _, _) in edges {
            let (a, b) = (self.dominant[i], self.dominant[j]);
            if a == u32::MAX || b == u32::MAX {
                continue;
            }
            let ab = self.index[&(a, b)] as usize;
            p_send[(i, ab)] += 1.0;
            p_recv[(j, ab)] += 1.0;
            if a != b {
                let ba = self.index[&(b, a)] as usize;
                p_send[(j, ba)] += 1.0;
                p_recv[(i, ba)] += 1.0;
            } else {
                // One stratum, both roles, so the edge is counted both ways.
                p_send[(j, ab)] += 1.0;
                p_recv[(i, ab)] += 1.0;
            }
        }
        for i in 0..n_cells {
            let s = p_send.row(i).sum();
            if s > 0.0 {
                p_send.row_mut(i).scale_mut(1.0 / s);
            }
            let r = p_recv.row(i).sum();
            if r > 0.0 {
                p_recv.row_mut(i).scale_mut(1.0 / r);
            }
        }
        (p_send, p_recv)
    }
}
