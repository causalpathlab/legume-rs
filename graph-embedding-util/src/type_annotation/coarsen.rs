use super::*;

/// `[n_comm × n_types]` mean fine score of the cells in each community —
/// the cell-grounded confusability that drives merging.
pub(super) fn community_enrichment(
    fine_z: &[f32],
    n_cells: usize,
    n_types: usize,
    community: &[usize],
    n_comm: usize,
) -> Vec<f32> {
    let width = n_comm * n_types;
    // Parallel scatter-accumulate: each cell chunk folds into its own
    // per-community (sum, count), then the partials are reduced.
    let (sum, cnt) = (0..n_cells)
        .into_par_iter()
        .fold(
            || (vec![0f64; width], vec![0usize; n_comm]),
            |(mut sum, mut cnt), c| {
                let k = community[c];
                cnt[k] += 1;
                let row = &fine_z[c * n_types..(c + 1) * n_types];
                for (s, &v) in sum[k * n_types..(k + 1) * n_types].iter_mut().zip(row) {
                    *s += v as f64;
                }
                (sum, cnt)
            },
        )
        .reduce(
            || (vec![0f64; width], vec![0usize; n_comm]),
            |(mut as_, mut ac), (bs, bc)| {
                for (a, b) in as_.iter_mut().zip(bs) {
                    *a += b;
                }
                for (a, b) in ac.iter_mut().zip(bc) {
                    *a += b;
                }
                (as_, ac)
            },
        );
    let mut enrich = vec![0f32; width];
    for k in 0..n_comm {
        let d = cnt[k].max(1) as f64;
        for t in 0..n_types {
            enrich[k * n_types + t] = (sum[k * n_types + t] / d) as f32;
        }
    }
    enrich
}

/// Subtract each fine type's cross-community mean from its `[n_comm × n_types]`
/// enrichment column, so the score reflects how distinctive a type is to a
/// community rather than its absolute (common-mode) level.
pub(super) fn center_columns(enrich: &mut [f32], n_comm: usize, n_types: usize) {
    if n_comm == 0 {
        return;
    }
    for t in 0..n_types {
        let mut mean = 0f64;
        for k in 0..n_comm {
            mean += enrich[k * n_types + t] as f64;
        }
        let mean = (mean / n_comm as f64) as f32;
        for k in 0..n_comm {
            enrich[k * n_types + t] -= mean;
        }
    }
}

/// Assign each fine type to the community where it is most enriched — the
/// type_map merge record (fine type → coarse group). Centered enrichment is
/// weighted by `√(community size)`: the mean over a community's cells has
/// standard error `∝ 1/√n`, so a tiny community's noise-inflated enrichment
/// would otherwise grab most types. The weight makes types prefer large,
/// reliably-enriched communities.
pub(super) fn build_merge_map(
    enrich: &[f32],
    sizes: &[usize],
    n_comm: usize,
    n_types: usize,
) -> Vec<usize> {
    let w: Vec<f32> = sizes.iter().map(|&n| (n as f32).sqrt()).collect();
    (0..n_types)
        .map(|t| {
            let mut best = 0;
            for k in 1..n_comm {
                if enrich[k * n_types + t] * w[k] > enrich[best * n_types + t] * w[best] {
                    best = k;
                }
            }
            best
        })
        .collect()
}

/// Per community, the up-to-`max_n` most-enriched fine types with positive
/// (above-null) enrichment — the lineage that defines the community's name
/// and coarse marker set. Always returns at least the single top type so no
/// community is left nameless.
///
/// Ranking is the **centered enrichment alone**. `enrich` is a mean of the
/// per-cell fine z-scores, which are already standardized against a *size-
/// matched* permutation null (each type's null draws random gene sets of the
/// same size), so the enrichment is comparable across marker-set sizes. An
/// earlier `√|markers_t|` weight here double-counted size and let large-marker
/// types (e.g. `Prog_Mk`) dominate the naming of lineages they don't belong to
/// — so it is removed.
pub(super) fn top_enriched_members(
    enrich: &[f32],
    n_comm: usize,
    n_types: usize,
    max_n: usize,
) -> Vec<Vec<usize>> {
    let score = |k: usize, t: usize| enrich[k * n_types + t];
    (0..n_comm)
        .map(|k| {
            let mut order: Vec<usize> = (0..n_types).collect();
            order.sort_by(|&a, &b| {
                score(k, b)
                    .partial_cmp(&score(k, a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sel: Vec<usize> = order
                .iter()
                .copied()
                .take(max_n)
                .filter(|&t| enrich[k * n_types + t] > 0.0)
                .collect();
            if sel.is_empty() {
                sel.push(order[0]);
            }
            sel
        })
        .collect()
}

/// Name a coarse group by the tokens shared by a **majority** of its member
/// type names (split on space/`_`, numeric tokens dropped), kept in the
/// representative (most-enriched, first) member's order: `{Naive B, Memory B,
/// pre B}` → `B`; `{CD8 Naive, CD8 Effector_1, CD8 Memory}` → `CD8`. A strict
/// intersection is brittle — one off-lineage member in a large community
/// would wipe the shared token — so we keep tokens present in ≥ 60% of
/// members (and ≥ 2). Falls back to the representative's full name when no
/// token clears the bar.
pub(super) fn lexical_label(members: &[usize], type_names: &[Box<str>]) -> Box<str> {
    let tok = |s: &str| -> Vec<String> {
        s.split([' ', '_'])
            .filter(|x| !x.is_empty() && !x.chars().all(|c| c.is_ascii_digit()))
            .map(str::to_string)
            .collect()
    };
    match members {
        [] => Box::from("NA"),
        [only] => type_names[*only].clone(),
        [first, rest @ ..] => {
            let m = 1 + rest.len();
            let thresh = (((m as f64) * 0.6).ceil() as usize).max(2);
            // document frequency of each token across members (deduped per member)
            let mut df: FxHashMap<String, usize> = FxHashMap::default();
            for &t in members {
                let mut seen: FxHashSet<String> = FxHashSet::default();
                for w in tok(&type_names[t]) {
                    if seen.insert(w.clone()) {
                        *df.entry(w).or_insert(0) += 1;
                    }
                }
            }
            // label = representative's tokens that clear the majority bar, in order
            let label: Vec<String> = tok(&type_names[*first])
                .into_iter()
                .filter(|w| df.get(w).copied().unwrap_or(0) >= thresh)
                .collect();
            if label.is_empty() {
                type_names[*first].clone()
            } else {
                label.join(" ").into_boxed_str()
            }
        }
    }
}

/// Coarse marker set per group = union of member `(gene, weight)` lists,
/// deduped by gene keeping the max weight.
pub(super) fn coarse_markers_from_groups(
    members: &[Vec<usize>],
    type_markers: &[Vec<(u32, f32)>],
) -> Vec<Vec<(u32, f32)>> {
    members
        .iter()
        .map(|grp| {
            let mut map: FxHashMap<u32, f32> = FxHashMap::default();
            for &t in grp {
                for &(g, w) in &type_markers[t] {
                    let e = map.entry(g).or_insert(f32::NEG_INFINITY);
                    if w > *e {
                        *e = w;
                    }
                }
            }
            map.into_iter().collect()
        })
        .collect()
}
