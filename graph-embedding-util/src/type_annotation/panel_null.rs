//! **Permutation null over the marker panel** — the guard the bootstrap cannot give you.
//!
//! The bootstrap ([`super::marker_bootstrap`]) resamples the panel and reports how often the
//! answer came back the same. That measures **variance**, and it is blind to **bias**: a call
//! that is *stably wrong* — because the type's listed markers simply do not identify it in this
//! embedding — comes back with support 1.0, every time, and looks like the most confident call
//! in the run.
//!
//! So ask a different question. Not *"would I get this answer again?"* but *"is this answer
//! better than one a panel that means nothing would have given me?"*
//!
//! ```text
//! bar[i][t] = min_{s≠t} d²(cell i, centroid s)          # the rivals, all real, all fixed
//! cost(t | panel) = Σ_i min( d²(cell i, centroid_t(panel)), bar[i][t] )
//!
//! for each type t, for each draw p:
//!     replace ONLY type t's panel with |live(t)| genes drawn at random from the LIVE marker
//!     pool, keeping t's IDF weight multiset, and leaving every rival type's panel REAL.
//! p[t] = P( cost(t | random genes) ≤ cost(t | t's own genes) )
//! ```
//!
//! **The statistic is the assignment's *cost*, not the type's cell count** — and getting this
//! wrong is the easy mistake. Counting cells (occupancy) asks "how many cells does this centroid
//! capture", which turns out to measure *whether any rival is nearby*, not whether the panel is
//! right. On a cleanly separated synthetic panel a random draw captures **exactly as many cells
//! as the real one** (measured: 0.337 vs 0.333, p = 0.995): once type t's real centroid is taken
//! out of the competition, its cells have no near rival left, so *anything* placed in the
//! neighbourhood sweeps them up by elimination. The count cannot see the difference. The cost
//! can: the real centroid sits **on** its cells and the random one sits far from them, so
//! `Σ min(d², bar)` separates the two by a wide margin. Cost also has no perverse optimum — a
//! centroid that captures nothing pays the full `Σ bar`, the worst score available — whereas a
//! *mean* distance over captured cells would be gamed by capturing a single nearby cell.
//!
//! **Why one type at a time.** Randomising every panel at once is a much weaker test and a
//! degenerate one: `m` random genes average toward the pool mean, so all `C` null centroids
//! collapse onto the same point, sit at nearly equal distance from every cell, and the whole
//! assignment becomes a coin flip between indistinguishable prototypes. The null would then
//! "fail" for reasons that have nothing to do with any particular type. Holding the rivals real
//! keeps the competition intact and puts exactly one type on trial.
//!
//! **Why the winner's curse cancels — the property nothing else in the design has.** A type with
//! few live markers has a high-variance centroid, and a noisy prototype *wins cells it should
//! not*: the maximum of a noisy score is biased upward. That is why a 7-marker type can quietly
//! capture 8% of a dataset. But the null panel is drawn at the **same size**, so it has the *same*
//! wobbly centroid and steals cells just as eagerly. The advantage appears on both sides of the
//! comparison and divides out. A small panel is no longer punished for being small — it is asked
//! only whether *these particular genes* beat *any* genes.
//!
//! **Why the null genes are drawn from the live pool.** A random gene that the embedding never
//! trained on carries no signal at all, so a null panel of dead genes would be trivially easy to
//! beat and every type would look significant. Drawing only from marker genes with a live β row,
//! and drawing exactly `|live(t)|` of them, holds "is this gene trained?" fixed and isolates the
//! one question worth asking: **are these the right genes?**

use super::gene_strata::GeneStrata;
use super::marker_bootstrap::sq_dist;
use super::markers::marker_gene_pool;
use super::term_ora::term_centroids;
use crate::null_call::live_row;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

#[cfg(test)]
mod tests;

/// What the panel null found. Everything is per **type** except [`Self::cell_p`].
pub struct PanelNull {
    /// Draws taken per type.
    pub n_perm: usize,
    /// Share of cells the type's real panel captures by nearest centroid. Reported because it is
    /// what people expect to see — but it is **not** what the p-value tests; see the module docs.
    pub occupancy: Vec<f32>,
    /// Mean share a same-size random panel captures from the same real rivals. Compare it with
    /// `occupancy` and be unsurprised when they agree: that is the point of the module docs.
    pub null_occupancy: Vec<f32>,
    /// Assignment cost `Σ_i min(d²(i, centroid_t), bar[i][t])` under the type's real panel —
    /// lower is a better-placed prototype.
    pub cost: Vec<f32>,
    /// Mean cost under same-size random panels.
    pub null_cost: Vec<f32>,
    /// `P(cost_null ≤ cost_real)`, +1-smoothed. **Large ⇒ this type's listed markers place its
    /// prototype no better than random genes of the same number would**, i.e. whatever cells it
    /// holds, it is not holding them on the strength of its own biology.
    pub p: Vec<f32>,
    /// Per cell: the share of null panels *for the type the cell was actually assigned to* that
    /// would have captured it too. **Large ⇒ this cell's label carries no marker-specific
    /// evidence** — anything of that size would have grabbed it. `NaN` for an unassigned cell.
    pub cell_p: Vec<f32>,
    /// Live markers per type (what the null draw is matched to).
    pub n_live: Vec<usize>,
}

/// Domain-separates this null's RNG stream from the ORA's label-shuffle and the bootstrap's
/// panel resampling, so the three cannot accidentally share draws off the same `--seed`.
const PANEL_NULL_STREAM: u64 = 0x009A_5E11_C0DE;

/// Run the per-type panel null. `feature_emb` / `cell_flat` are row-major `[g × h]` / `[n × h]`.
pub fn run_panel_null(
    feature_emb: &[f32],
    cell_flat: &[f32],
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
    n_perm: usize,
    seed: u64,
) -> PanelNull {
    let c = type_markers.len();
    let n = cell_flat.len() / h;
    let g = feature_emb.len() / h;

    let (centroids, n_live) = term_centroids(feature_emb, type_markers, h);

    // The universe the null draws from: marker genes the embedding actually **trained** (a live
    // β row). Drawing dead genes would make the null trivially beatable — see the module docs.
    let pool: Vec<u32> = marker_gene_pool(type_markers, g)
        .into_iter()
        .filter(|&gi| live_row(feature_emb, gi as usize, h).is_some())
        .collect();
    let pool_n = pool.len();

    // **Match the null on gene norm.** Draw uniformly from the pool and every null panel inherits
    // the *pool's* mean norm, so a type whose markers are longer than average beats its null on
    // norm alone and one whose markers are shorter loses on norm alone — with no biology tested
    // either way. Stratifying reproduces each type's own norm profile in its null. See
    // `super::gene_strata`.
    let strata = GeneStrata::by_norm(feature_emb, &pool, h);
    // `quota[t][k]` = how many of type t's live markers fall in norm stratum k. A null draw for t
    // takes exactly that many from each stratum, so its norm profile is t's own.
    let pos_of: std::collections::HashMap<u32, usize> =
        pool.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let quota: Vec<Vec<usize>> = (0..c)
        .map(|t| {
            let mut q = vec![0usize; strata.members.len()];
            for &(gi, _) in &type_markers[t] {
                if let Some(&i) = pos_of.get(&gi) {
                    q[strata.stratum[i]] += 1;
                }
            }
            q
        })
        .collect();

    ////////////////////////////////////////////////////////////////////////
    // The rivals are fixed, so the bar each type must clear is fixed too //
    ////////////////////////////////////////////////////////////////////////
    // Cell `i` goes to type `t` iff `d(i, t) < min_{s≠t} d(i, s)`. That right-hand side does not
    // depend on t's panel at all — so precompute it once and the whole null collapses to "how
    // often does a fake centroid get under a bar that is already sitting there". Each draw then
    // costs one pass over the cells instead of a full C-way re-assignment.
    let usable: Vec<bool> = (0..c).map(|t| n_live[t] > 0).collect();
    let mut bar = vec![f32::INFINITY; n * c]; // bar[i*c + t] = min_{s≠t, usable} d²(i, s)
    let mut own = vec![f32::INFINITY; n * c]; // own[i*c + t] = d²(i, t) under the REAL panel
    bar.par_chunks_mut(c)
        .zip(own.par_chunks_mut(c))
        .enumerate()
        .for_each(|(i, (bar_i, own_i))| {
            let cell = &cell_flat[i * h..(i + 1) * h];
            for (t, o) in own_i.iter_mut().enumerate() {
                *o = if usable[t] {
                    sq_dist(cell, &centroids[t * h..(t + 1) * h])
                } else {
                    f32::INFINITY
                };
            }
            // Best and second-best over the usable rivals ⇒ `min_{s≠t}` in O(c), not O(c²).
            let (mut b1, mut b2) = (f32::INFINITY, f32::INFINITY);
            let mut arg1 = usize::MAX;
            for (t, &d) in own_i.iter().enumerate() {
                if d < b1 {
                    b2 = b1;
                    b1 = d;
                    arg1 = t;
                } else if d < b2 {
                    b2 = d;
                }
            }
            for (t, b) in bar_i.iter_mut().enumerate() {
                *b = if t == arg1 { b2 } else { b1 };
            }
        });

    let occupancy: Vec<f32> = (0..c)
        .map(|t| {
            if !usable[t] {
                return 0.0;
            }
            let won = (0..n).filter(|&i| own[i * c + t] < bar[i * c + t]).count();
            won as f32 / n as f32
        })
        .collect();
    // What the p-value is actually about: how well this panel's prototype explains the cells,
    // given that every rival is real and fixed.
    let cost: Vec<f32> = (0..c)
        .map(|t| {
            (0..n)
                .map(|i| {
                    let b = bar[i * c + t];
                    if usable[t] {
                        own[i * c + t].min(b)
                    } else {
                        b
                    }
                })
                .sum()
        })
        .collect();

    /////////////////////////////////////
    // the draws: one type on trial    //
    /////////////////////////////////////
    // Keyed on (seed, type, draw) — not on rayon's schedule — so a seed reproduces exactly
    // however the work is split, matching `score::permutation_zscores`.
    let mut null_occ = vec![0f32; c * n_perm];
    let mut null_cost_all = vec![0f32; c * n_perm];
    // Times a null-t panel explained cell i at least as well as t's own panel did.
    let mut cell_hits = vec![0u32; n * c];

    // Ctrl+C: stop drawing and score each type against the draws it got. The types run in
    // parallel and so may stop at different counts, but that is harmless — every type's p is an
    // empirical tail against *its own* null draws, so each simply gets a coarser denominator.
    let stop = crate::stop::stop_flag();
    let mut n_done = vec![0usize; c];

    /// One type's trial: its null occupancies, its null costs, per-cell hits, and how many draws
    /// actually ran (a Ctrl+C can cut it short).
    struct Trial {
        occ: Vec<f32>,
        cost: Vec<f32>,
        hits: Vec<u32>,
        done: usize,
    }

    let per_type: Vec<Trial> = (0..c)
        .into_par_iter()
        .map(|t| {
            let m = n_live[t].min(pool_n);
            let mut occ_t = vec![0f32; n_perm];
            let mut cost_t = vec![f32::INFINITY; n_perm];
            let mut hits_t = vec![0u32; n];
            if m < 1 || !usable[t] {
                return Trial {
                    occ: occ_t,
                    cost: cost_t,
                    hits: hits_t,
                    done: 0,
                };
            }
            // t's IDF weights, in order, but only as many as it has live markers — the null
            // panel is the same size and carries the same weight multiset.
            let weights: Vec<f32> = type_markers[t]
                .iter()
                .filter(|&&(gi, _)| live_row(feature_emb, gi as usize, h).is_some())
                .map(|&(_, w)| w)
                .collect();

            let mut cent = vec![0f32; h];
            let mut drawn: Vec<usize> = Vec::with_capacity(m);
            let mut done = 0usize;
            for p in 0..n_perm {
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let mut rng = SmallRng::seed_from_u64(
                    seed ^ PANEL_NULL_STREAM
                        ^ ((t as u64) << 32)
                        ^ (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                // Stratum-matched: take t's own number of genes from each norm bin.
                drawn.clear();
                for (k, &want) in quota[t].iter().enumerate() {
                    let bin = &strata.members[k];
                    let take = want.min(bin.len());
                    if take == 0 {
                        continue;
                    }
                    drawn.extend(
                        rand::seq::index::sample(&mut rng, bin.len(), take)
                            .iter()
                            .map(|j| bin[j]),
                    );
                }
                cent.iter_mut().for_each(|v| *v = 0.0);
                let mut wsum = 0f32;
                for (&pool_i, &w) in drawn.iter().zip(&weights) {
                    let gi = pool[pool_i] as usize;
                    let ef = &feature_emb[gi * h..(gi + 1) * h];
                    wsum += w;
                    for (r, &e) in cent.iter_mut().zip(ef) {
                        *r += w * e;
                    }
                }
                if wsum > 0.0 {
                    cent.iter_mut().for_each(|v| *v /= wsum);
                }
                let (mut won, mut cost_p) = (0usize, 0f32);
                for i in 0..n {
                    let d = sq_dist(&cell_flat[i * h..(i + 1) * h], &cent);
                    let b = bar[i * c + t];
                    cost_p += d.min(b);
                    if d < b {
                        won += 1;
                    }
                    // Per cell: did random genes explain this cell at least as well as t's own?
                    if d <= own[i * c + t] {
                        hits_t[i] += 1;
                    }
                }
                occ_t[p] = won as f32 / n as f32;
                cost_t[p] = cost_p;
                done += 1;
            }
            Trial {
                occ: occ_t,
                cost: cost_t,
                hits: hits_t,
                done,
            }
        })
        .collect();

    for (t, trial) in per_type.iter().enumerate() {
        n_done[t] = trial.done;
        null_occ[t * n_perm..(t + 1) * n_perm].copy_from_slice(&trial.occ);
        null_cost_all[t * n_perm..(t + 1) * n_perm].copy_from_slice(&trial.cost);
        for i in 0..n {
            cell_hits[i * c + t] = trial.hits[i];
        }
    }

    // Every average and every tail is over the draws that *ran* (`n_done[t]`), not the draws that
    // were asked for — so an interrupted run is a coarser null, not a wrong one.
    let drawn = |t: usize| &null_occ[t * n_perm..t * n_perm + n_done[t]];
    let drawn_cost = |t: usize| &null_cost_all[t * n_perm..t * n_perm + n_done[t]];
    let mean_over = |v: &[f32]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    let null_occupancy: Vec<f32> = (0..c).map(|t| mean_over(drawn(t))).collect();
    let null_cost: Vec<f32> = (0..c).map(|t| mean_over(drawn_cost(t))).collect();
    let p: Vec<f32> = (0..c)
        .map(|t| {
            if !usable[t] || n_done[t] == 0 {
                return 1.0;
            }
            // Lower cost = better panel, so the null "wins" by coming in at or under the real.
            let le = drawn_cost(t).iter().filter(|&&x| x <= cost[t]).count();
            (le as f32 + 1.0) / (n_done[t] as f32 + 1.0)
        })
        .collect();

    // Per cell: how easily would ANY same-size panel have taken it?
    let cell_p: Vec<f32> = (0..n)
        .map(|i| {
            let assigned = (0..c).find(|&t| usable[t] && own[i * c + t] < bar[i * c + t]);
            match assigned {
                Some(t) if n_done[t] > 0 => {
                    (cell_hits[i * c + t] as f32 + 1.0) / (n_done[t] as f32 + 1.0)
                }
                _ => f32::NAN,
            }
        })
        .collect();

    PanelNull {
        n_perm: *n_done.iter().max().unwrap_or(&0),
        occupancy,
        null_occupancy,
        cost,
        null_cost,
        p,
        cell_p,
        n_live,
    }
}
