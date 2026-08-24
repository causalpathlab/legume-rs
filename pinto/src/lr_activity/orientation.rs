//! Which link community a spatial edge belongs to.
//!
//! The test this module feeds is symmetric: ligand-receptor co-activity is
//! measured along the contacts of ONE link community at a time, with both
//! orientations of every edge counted, so no endpoint plays a privileged
//! role. Strata ARE link communities.
//!
//! This replaces an earlier directional design that paired communities into
//! `a -> b` strata. Two things killed it. Measured on a full-database run,
//! the cross-community arm was statistically inert — not one of its tests
//! reached p < 0.05 where a calibrated test puts 5% there by construction —
//! while it carried ~73% of the family-wise correction's cost. And its
//! sender/receiver roles descended from the edge list's canonical
//! `(min, max)` ordering, an artifact of barcode load order, so any true
//! directional signal was split across two strata and roughly halved before
//! testing. Within one community both orientations are enumerated, which is
//! what makes the surviving statistic symmetric by construction rather than
//! by hope.
//!
//! Homo-/heterotypy is deliberately absent from this vocabulary: it is a
//! property of a community's contacts under a cell-type annotation — a link
//! community can itself be an interface community — and nothing at this
//! layer can or should claim it.

use crate::util::common::*;
use rustc_hash::FxHashMap;

/// Per-community edge strata, and the cell-level assignment they rest on.
///
/// Strata are stored CSR-style: `communities` is sorted and deduplicated,
/// and `items[offsets[s]..offsets[s + 1]]` lists the edge instances
/// realizing stratum `s`. Every edge whose endpoints share a dominant
/// community contributes BOTH orientations, `(i -> j)` and `(j -> i)`, so
/// per-instance "first endpoint" carries no information and anything summed
/// over a stratum is symmetric in the pair.
pub struct CommunityStrata {
    /// The link community of each stratum, sorted, so the id is a function
    /// of the partition rather than of the order edges were serialized.
    communities: Vec<u32>,
    offsets: Vec<usize>,
    /// `(edge index, flipped)` — flipped instances read the stored `(i, j)`
    /// as `(j, i)`.
    items: Vec<(u32, bool)>,
}

impl CommunityStrata {
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
    ///
    /// `anchor_edges` and `tested_edges` differ once the pair graph carries
    /// expression-similar pairs. Those pairs belong in `anchor_edges`: which
    /// community a cell sits in is a statement about the partition, and more
    /// evidence is better. They must NOT appear in `tested_edges`: the
    /// co-activity estimand presupposes physical contact, and a pair that is
    /// merely similar has none.
    ///
    /// Every index this type later hands back refers to `tested_edges`, so a
    /// caller must pass that same slice to [`Self::memberships`].
    ///
    /// Passing the same slice twice is the unaugmented case.
    pub fn from_edge_modes(
        anchor_edges: &[(usize, usize, u32, Option<Box<str>>)],
        tested_edges: &[(usize, usize, u32, Option<Box<str>>)],
        n_cells: usize,
    ) -> Self {
        let mut counts: Vec<FxHashMap<u32, usize>> =
            (0..n_cells).map(|_| FxHashMap::default()).collect();
        for &(i, j, k, _) in anchor_edges {
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
        Self::new(dominant, tested_edges)
    }

    /// Build the strata from a per-cell community assignment.
    ///
    /// Only edges whose two endpoints share a dominant community realize a
    /// stratum; an edge bridging two communities is a statement about the
    /// partition's boundary, not a within-community contact, and sits out.
    /// `dominant[i] == u32::MAX` marks a cell with no community; its edges
    /// sit out entirely.
    pub fn new(dominant: Vec<u32>, edges: &[(usize, usize, u32, Option<Box<str>>)]) -> Self {
        let community_of = |i: usize, j: usize| -> Option<u32> {
            let (a, b) = (dominant[i], dominant[j]);
            (a != u32::MAX && a == b).then_some(a)
        };

        let mut counts: FxHashMap<u32, usize> = FxHashMap::default();
        for &(i, j, _, _) in edges {
            if let Some(c) = community_of(i, j) {
                // Both orientations, so the stratum is symmetric in the pair.
                *counts.entry(c).or_insert(0) += 2;
            }
        }
        let mut communities: Vec<u32> = counts.keys().copied().collect();
        communities.sort_unstable();
        let index: FxHashMap<u32, usize> = communities
            .iter()
            .enumerate()
            .map(|(s, &c)| (c, s))
            .collect();

        let mut offsets = vec![0usize; communities.len() + 1];
        for (s, c) in communities.iter().enumerate() {
            offsets[s + 1] = offsets[s] + counts[c];
        }
        let mut cursor = offsets.clone();
        let mut items = vec![(0u32, false); offsets[communities.len()]];
        for (e, &(i, j, _, _)) in edges.iter().enumerate() {
            if let Some(c) = community_of(i, j) {
                let s = index[&c];
                items[cursor[s]] = (e as u32, false);
                items[cursor[s] + 1] = (e as u32, true);
                cursor[s] += 2;
            }
        }
        Self {
            communities,
            offsets,
            items,
        }
    }

    #[must_use]
    pub fn n_strata(&self) -> usize {
        self.communities.len()
    }

    /// The link community this stratum measures.
    #[must_use]
    pub fn community(&self, stratum: usize) -> u32 {
        self.communities[stratum]
    }

    /// Edge INSTANCES in the stratum: twice the edge count, one per
    /// orientation.
    #[must_use]
    pub fn edges_in(&self, stratum: usize) -> usize {
        self.offsets[stratum + 1] - self.offsets[stratum]
    }

    /// How the stratum reads in a label.
    #[must_use]
    pub fn label(&self, stratum: usize) -> String {
        format!("C{}", self.communities[stratum])
    }

    /// The edge instances realizing a stratum. `flipped` reads the stored
    /// `(i, j)` as `(j, i)`; every edge appears once each way.
    ///
    /// Indices rather than cells, so a caller keeps access to the edge's
    /// batch label, which it still has to filter on.
    #[must_use]
    pub fn oriented(&self, stratum: usize) -> &[(u32, bool)] {
        &self.items[self.offsets[stratum]..self.offsets[stratum + 1]]
    }

    /// Per-cell soft membership over the strata: the fraction of a cell's
    /// within-community edge instances that fall in each stratum. One matrix,
    /// not a send/recv pair — with both orientations enumerated the two roles
    /// coincide exactly.
    #[must_use]
    pub fn memberships(
        &self,
        edges: &[(usize, usize, u32, Option<Box<str>>)],
        n_cells: usize,
    ) -> Mat {
        let n = self.n_strata();
        let mut p = Mat::zeros(n_cells, n);
        for s in 0..n {
            for &(e, flipped) in self.oriented(s) {
                let (i, j, _, _) = edges[e as usize];
                let first = if flipped { j } else { i };
                p[(first, s)] += 1.0;
            }
        }
        // Column-outer, because `Mat` is column-major.
        let mut tot = vec![0.0f32; n_cells];
        for s in 0..n {
            for i in 0..n_cells {
                tot[i] += p[(i, s)];
            }
        }
        for s in 0..n {
            for i in 0..n_cells {
                if tot[i] > 0.0 {
                    p[(i, s)] /= tot[i];
                }
            }
        }
        p
    }
}
