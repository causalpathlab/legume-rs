//! Steps 1–2 of the firm pipeline: where each type sits, and which cells it takes.
//!
//! The IDF-weighted term centroid, the drop of the types the panel cannot locate at all,
//! the nearest-centroid assignment, and the two distance readings it is judged by (the MAD
//! outlier prune, and the per-cell distance a bootstrap replicate re-derives). Grouped
//! because every one of them reads the same `[c × h]` prototype block under the same
//! Euclidean metric, and the guards against a short centroid have to agree across them.

use crate::null_call::live_row;
use crate::type_annotation::markers::MarkerSets;
use crate::type_annotation::UNASSIGNED;
use anyhow::Result;
use enrichment::consensus::MIN_LIVE_MARKERS;
use log::{info, warn};
use rayon::prelude::*;

/// `[c × h]` row-major IDF-weighted mean of each type's marker feature embeddings — the
/// **un-normalized** centroid (the Euclidean prototype) — plus the number of *live*
/// markers each type's centroid was actually built from ([`live_row`]). Empty types get
/// a zero row.
///
/// A dead marker is skipped in numerator *and* denominator. Counting it in `wsum` would
/// divide a partial sum by the full weight, pulling the centroid toward the origin in
/// proportion to the type's dead-marker fraction — and a short centroid is not a weak
/// competitor, it is a magnet (see [`assign_nearest`]). Skipping it makes the centroid
/// the honest mean over the markers carrying evidence, and leaves an all-dead type at the
/// origin, where [`assign_nearest`]'s guard excludes it.
pub(in crate::type_annotation) fn term_centroids(
    feature_emb: &[f32],
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
) -> (Vec<f32>, Vec<usize>) {
    let c = type_markers.len();
    let mut out = vec![0f32; c * h];
    let mut n_live = vec![0usize; c];
    out.par_chunks_mut(h)
        .zip(n_live.par_iter_mut())
        .zip(type_markers.par_iter())
        .for_each(|((row, live), markers)| {
            let mut wsum = 0f32;
            for &(gi, w) in markers {
                let Some(ef) = live_row(feature_emb, gi as usize, h) else {
                    continue;
                };
                *live += 1;
                wsum += w;
                for (r, &e) in row.iter_mut().zip(ef) {
                    *r += w * e;
                }
            }
            if wsum > 0.0 {
                for v in row.iter_mut() {
                    *v /= wsum;
                }
            }
        });
    (out, n_live)
}

/// Report the [`term_centroids`] live-marker counts, and warn about the types running on
/// a mostly-dead panel.
///
/// A dead marker still *matches* the panel and counts toward the "matched entries" tally,
/// but contributes nothing to its type's centroid — so a panel that is mostly dead looks
/// exactly like a healthy one from the outside. Say so, and name the flag that fixes it.
/// Drop the cell types the panel cannot locate: those left with fewer than `min_markers` markers
/// carrying a live feature row.
///
/// **A type is not "weakly supported", it is unlocated.** Its centroid is the mean of whatever
/// survived, and the mean of one or two points has no direction worth the name — the sampling
/// variance of a mean of one is not small, it is *undefined*. Worse, a centroid built from too few
/// markers tends to land short, and a short centroid sits near the middle of the cell cloud, where
/// it is close to every cell at once: it does not compete weakly, it becomes a **magnet** and takes
/// the dataset. The honest outcome is that it does not compete at all.
///
/// The drop is done by **emptying the type's marker list**, which is the one lever every consumer
/// downstream already respects: [`term_centroids`] leaves an empty type at the origin,
/// [`assign_nearest`] excludes the origin, and the marker bootstrap, the panel null and the support
/// null all build their pools from these same lists.
///
/// The type keeps its name and its column in every output — it simply never wins a cell. Silently
/// renumbering the types would be worse than the disease.
pub(super) fn drop_unsupported_types(
    beta_flat: &[f32],
    type_names: &[Box<str>],
    type_markers: &mut MarkerSets,
    h: usize,
    min_markers: usize,
) -> Result<usize> {
    // Never below the bootstrap's own invariant: you cannot resample a single point.
    let bar = min_markers.max(MIN_LIVE_MARKERS);

    let mut dropped: Vec<(usize, usize, usize)> = Vec::new(); // (type, matched, live)
    for (t, markers) in type_markers.iter_mut().enumerate() {
        let matched = markers.len();
        let live = markers
            .iter()
            .filter(|&&(gi, _)| live_row(beta_flat, gi as usize, h).is_some())
            .count();
        if live < bar {
            dropped.push((t, matched, live));
            markers.clear();
        }
    }
    if dropped.is_empty() {
        return Ok(0);
    }

    let preview: Vec<String> = dropped
        .iter()
        .take(12)
        .map(|&(t, matched, live)| format!("{} ({live}/{matched})", type_names[t]))
        .collect();
    let tail = if dropped.len() > preview.len() {
        format!(", … {} more", dropped.len() - preview.len())
    } else {
        String::new()
    };
    warn!(
        "dropping {} cell type(s) with fewer than {bar} live markers — they are not weakly \
         located, they are UNLOCATED, and a centroid built from too few markers lands short, near \
         the middle of the cell cloud, where it is close to every cell and becomes a magnet. \
         Shown as live/matched: {}{tail}. They keep their columns in the outputs but can no longer \
         win a cell. If this is unexpected, the markers are missing from the embedding rather than \
         from the data — re-run the fit with `--must-train-features <panel>`.",
        dropped.len(),
        preview.join(", "),
    );

    let surviving = type_markers.iter().filter(|m| !m.is_empty()).count();
    anyhow::ensure!(
        surviving >= 2,
        "only {surviving} cell type(s) have {bar} or more live markers — there is nothing left to \
         tell apart. Lower --min-markers, or (far more likely) the marker panel was never trained \
         into this embedding: re-run the fit with `--must-train-features <panel>`."
    );
    Ok(dropped.len())
}

pub(super) fn report_marker_liveness(
    type_names: &[Box<str>],
    type_markers: &[Vec<(u32, f32)>],
    n_live: &[usize],
) {
    let n_matched: usize = type_markers.iter().map(Vec::len).sum();
    if n_matched == 0 {
        return;
    }
    info!(
        "marker liveness: {}/{n_matched} matched markers carry a live β row",
        n_live.iter().sum::<usize>()
    );

    // (name, live, matched) for every type more than half dead, worst fraction first.
    // Compared as `live_a · matched_b` vs `live_b · matched_a` — the same order as the
    // ratio, in exact integer arithmetic.
    let mut starved: Vec<(&str, usize, usize)> = type_names
        .iter()
        .zip(type_markers)
        .zip(n_live)
        .filter(|&((_, m), &live)| !m.is_empty() && live * 2 < m.len())
        .map(|((name, m), &live)| (name.as_ref(), live, m.len()))
        .collect();
    if starved.is_empty() {
        return;
    }
    starved.sort_by(|&(_, al, am), &(_, bl, bm)| (al * bm).cmp(&(bl * am)));

    let mut preview: Vec<String> = starved
        .iter()
        .take(10)
        .map(|&(name, live, m)| format!("{name} {live}/{m}"))
        .collect();
    if starved.len() > preview.len() {
        preview.push("…".into());
    }
    warn!(
        "{} type(s) have under half their markers alive in the embedding: {}. \
         A dead marker is a gene the embedding never trained on and whose post-hoc \
         projection failed its null test; those types are scored off the survivors alone. \
         Re-run the embedding with `--must-train-features <panel>` to train on the panel.",
        starved.len(),
        preview.join(", "),
    );
}

/// Nearest-centroid assignment by squared Euclidean distance. Returns
/// `(assign[n], dist[n])` where `dist` is the Euclidean distance to the assigned
/// centroid.
///
/// **A zero-norm centroid can never win.** [`term_centroids`] leaves a type with no
/// usable markers at the origin (it only divides when `wsum > 0`). Cells here are
/// the *raw* embedding, so that centroid sits at squared distance `‖cell‖²` from
/// every cell — and therefore beats every real prototype for any cell nearer the
/// origin than to any of them. It is not a weak competitor, it is a magnet: on a
/// bone-marrow gem run, four types matched zero markers and one of them captured
/// 6814 / 15315 = 44.5% of all cells, while the other three captured none (the
/// strict `<` below lets only the first such type by index ever win).
///
/// `parse_and_match_markers` drops types that matched no gene, which is the common
/// cause but not the only one: a zero centroid also arises when every marker's gene
/// index is out of range, when the IDF weights all vanish, or when the embedding
/// rows cancel. None of those survive as an *empty marker list*, so the guard here
/// is a strictly larger net, not a duplicate of the drop. The cosine path
/// (`type_scores`) is immune for free, since a zero centroid scores 0.
pub(super) fn assign_nearest(
    cell_flat: &[f32],
    n: usize,
    centroids: &[f32],
    c: usize,
    h: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut live: Vec<bool> = (0..c)
        .map(|t| live_row(centroids, t, h).is_some())
        .collect();
    let n_degenerate = live.iter().filter(|&&l| !l).count();
    if n_degenerate == c {
        // Nothing to choose between; keep the old (arbitrary) behaviour rather than
        // returning an infinite distance that would poison the MAD prune downstream.
        warn!("all {c} term centroids are zero-norm — assignment is meaningless");
        live.iter_mut().for_each(|l| *l = true);
    } else if n_degenerate > 0 {
        warn!(
            "{n_degenerate} of {c} term centroid(s) are zero-norm and were excluded from \
             nearest-centroid assignment (they would sit at constant distance from every cell)"
        );
    }
    let mut assign = vec![0usize; n];
    let mut dist = vec![0f32; n];
    assign
        .par_iter_mut()
        .zip(dist.par_iter_mut())
        .enumerate()
        .for_each(|(i, (a, d))| {
            let cell = &cell_flat[i * h..(i + 1) * h];
            let mut best = 0usize;
            let mut best_d2 = f32::INFINITY;
            for t in 0..c {
                if !live[t] {
                    continue;
                }
                let ct = &centroids[t * h..(t + 1) * h];
                let mut s = 0f32;
                for (x, y) in cell.iter().zip(ct) {
                    let diff = x - y;
                    s += diff * diff;
                }
                if s < best_d2 {
                    best_d2 = s;
                    best = t;
                }
            }
            *a = best;
            *d = best_d2.max(0.0).sqrt();
        });
    (assign, dist)
}

/// Mark `assign[c] = UNASSIGNED` for cells whose distance to their assigned
/// centroid is a high-side robust outlier (`> median + k·MAD`) within that term
/// — the shared `data_beans::qc_lib` robust-band idiom. Terms with < 3 assigned
/// cells are left intact (too few to define a band). Returns the number pruned.
pub(super) fn prune_outliers(assign: &mut [usize], dist: &[f32], c: usize, k: f64) -> usize {
    use data_beans::qc_lib::{robust_outlier_keep, Tail};
    // Per-term cell indices (post-assignment).
    let mut per_term: Vec<Vec<usize>> = vec![Vec::new(); c];
    for (i, &t) in assign.iter().enumerate() {
        if t != UNASSIGNED {
            per_term[t].push(i);
        }
    }
    let mut pruned = 0usize;
    for cells in &per_term {
        if cells.len() < 3 {
            continue; // too few to define outliers
        }
        let dists: Vec<f32> = cells.iter().map(|&i| dist[i]).collect();
        let keep = robust_outlier_keep(&dists, k as f32, Tail::Upper, false, None);
        for (&i, &keep_i) in cells.iter().zip(&keep) {
            if !keep_i {
                assign[i] = UNASSIGNED;
                pruned += 1;
            }
        }
    }
    pruned
}

/// Distance from each cell to the centroid it was assigned to (`NaN` when unassigned).
pub(super) fn centroid_distances(
    cell_flat: &[f32],
    n: usize,
    centroids: &[f32],
    h: usize,
    assign: &[usize],
) -> Vec<f32> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let t = assign[i];
            if t == UNASSIGNED {
                return f32::NAN;
            }
            let cell = &cell_flat[i * h..(i + 1) * h];
            let ct = &centroids[t * h..(t + 1) * h];
            cell.iter()
                .zip(ct)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .max(0.0)
                .sqrt()
        })
        .collect()
}
