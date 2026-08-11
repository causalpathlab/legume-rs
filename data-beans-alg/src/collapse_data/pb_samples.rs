//! Pseudobulk-sample (pb-sample) construction.
//!
//! `build_pb_samples` aggregates per-(batch, group) intersections of the
//! finest hash partition into a layout + per-pb-sample gene sums used
//! downstream by the refinement step (`refine_multilevel` BBKNN +
//! DC-SBM) and by the cross-batch matched-stat path.

use super::*;

pub struct PbSampleLayout {
    /// Centroid matrix: proj_dim x num_pb_samples
    pub centroids: DMatrix<f32>,
    /// Number of cells in each pb-sample
    pub cell_counts: Vec<f32>,
    /// Batch index for each pb-sample
    pub pb_sample_to_batch: Vec<usize>,
    /// Sample/group index for each pb-sample
    pub pb_sample_to_group: Vec<usize>,
    /// Maps (batch, group) → pb-sample index
    pub bg_to_pbsamp: HashMap<(usize, usize), usize>,
    /// Global cell index → owning pb-sample index.
    pub cell_to_pbsamp: Vec<usize>,
    /// For a singleton pb-sample (an anchored reference column or a bulk
    /// sample), the one column it stands for; `None` for ordinary
    /// (batch, group) pb-samples. The layout is the single owner of "which
    /// pb-samples are already summaries" — downstream steps
    /// (`split_anchored_finest_groups`) read it here instead of re-deriving
    /// membership from batch labels.
    pub singleton_col: Vec<Option<usize>>,
    /// Batches whose columns are bulk samples: singletons like anchors, but
    /// excluded from cross-batch matching in BOTH directions (see
    /// [`bbknn_match_one_pbsamp`]).
    pub bulk_batches: Vec<usize>,
}

impl PbSampleLayout {
    /// Whether this pb-sample is a bulk sample (a singleton that takes no
    /// part in matching).
    #[must_use]
    pub fn is_bulk(&self, pbsamp: usize) -> bool {
        self.is_bulk_batch(self.pb_sample_to_batch[pbsamp])
    }

    /// Whether `batch` is a bulk batch — the one spelling of the predicate,
    /// so per-pb-sample and per-batch callers cannot drift apart.
    #[must_use]
    pub fn is_bulk_batch(&self, batch: usize) -> bool {
        self.bulk_batches.contains(&batch)
    }
}

/// Pre-aggregated pb-sample data for fast cross-batch matching.
/// Each pb-sample is the intersection of a (batch, sample) pair.
pub struct PbSampleCollection {
    pub layout: PbSampleLayout,
    /// Sparse gene sums per pb-sample: Vec of (gene_index, sum)
    pub gene_sums: Vec<Vec<(usize, f32)>>,
    /// Number of genes
    pub num_genes: usize,
}

/// Intermediate per-batch accumulator used during pb-sample construction.
pub(super) struct BatchAccumulator {
    centroid_sum: Vec<f32>,
    gene_sum: HashMap<usize, f32>,
    count: usize,
}

/// A single pb-sample produced from a (batch, group) intersection.
pub(super) struct PbSampleData {
    centroid: Vec<f32>,
    gene_sums: Vec<(usize, f32)>,
    cell_count: f32,
    batch: usize,
    group: usize,
}

/// Build the shared pb-sample layout from (batch, group) intersections.
///
/// For each non-empty (batch, group) block:
/// - Centroid = mean of projection vectors
/// - Cell count = number of cells in block
///
/// `anchor_batches` are batches whose columns are ALREADY pseudobulks — a
/// prior run's carried reference. Re-averaging those through this run's
/// partition would flatten resolution the parent already paid for (averages
/// of averages), so each anchored column becomes its own singleton pb-sample
/// instead, keeping the reference at its stored granularity for matching.
/// `bulk_batches` get the same singleton treatment (a bulk sample is already
/// a summary) but are additionally barred from matching — see
/// [`bbknn_match_one_pbsamp`].
///
/// This only uses `proj_kn` (no CSC reads), so it can be shared across layers.
pub(super) fn build_pb_sample_layout(
    group_to_cols: &[Vec<usize>],
    col_to_batch: &[usize],
    proj_kn: &DMatrix<f32>,
    col_weight: Option<&[f32]>,
    anchor_batches: &[usize],
    bulk_batches: &[usize],
) -> anyhow::Result<PbSampleLayout> {
    let proj_dim = proj_kn.nrows();

    /// Intermediate per-batch accumulator for centroid computation.
    struct CentroidAccum {
        centroid_sum: Vec<f32>,
        /// Summed multiplicity, not a raw column count: a column standing for
        /// `m` cells must weigh `m` in both the centroid and `cell_counts`,
        /// or a carried pseudobulk counts the same as one cell.
        count: f32,
    }
    let weight_of = |c: usize| col_weight.map_or(1.0, |w| w[c]);

    // Collect centroid data per group in parallel. The last tuple field is
    // `Some(column)` for a singleton (anchored) pb-sample, `None` for an
    // ordinary (batch, group) block.
    type CentroidTuple = (usize, usize, Vec<f32>, f32, Option<usize>);
    let per_group_results: Vec<Vec<CentroidTuple>> = group_to_cols
        .par_iter()
        .enumerate()
        .map(|(group, cells)| {
            let mut batch_data: HashMap<usize, CentroidAccum> = HashMap::default();
            let mut singletons: Vec<CentroidTuple> = Vec::new();

            for &glob_idx in cells {
                let batch = col_to_batch[glob_idx];
                let w = weight_of(glob_idx);
                if anchor_batches.contains(&batch) || bulk_batches.contains(&batch) {
                    let centroid: Vec<f32> =
                        (0..proj_dim).map(|d| proj_kn[(d, glob_idx)]).collect();
                    singletons.push((batch, group, centroid, w, Some(glob_idx)));
                    continue;
                }
                let acc = batch_data.entry(batch).or_insert_with(|| CentroidAccum {
                    centroid_sum: vec![0f32; proj_dim],
                    count: 0.0,
                });
                for d in 0..proj_dim {
                    acc.centroid_sum[d] += proj_kn[(d, glob_idx)] * w;
                }
                acc.count += w;
            }

            let mut out: Vec<CentroidTuple> = batch_data
                .into_iter()
                .filter(|(_, acc)| acc.count > 0.0)
                .map(|(batch, acc)| {
                    let inv_count = 1.0 / acc.count;
                    let centroid: Vec<f32> =
                        acc.centroid_sum.iter().map(|v| v * inv_count).collect();
                    (batch, group, centroid, acc.count, None)
                })
                .collect::<Vec<_>>();
            out.extend(singletons);
            out
        })
        .collect();

    // Flatten into a single list
    let all_pbsamp: Vec<_> = per_group_results.into_iter().flatten().collect();
    let num_pb = all_pbsamp.len();

    if num_pb == 0 {
        return Err(anyhow::anyhow!("no pb-samples built"));
    }

    // Build centroid matrix and metadata
    let mut centroids = DMatrix::<f32>::zeros(proj_dim, num_pb);
    let mut cell_counts = Vec::with_capacity(num_pb);
    let mut pbsamp_to_batch = Vec::with_capacity(num_pb);
    let mut pbsamp_to_group = Vec::with_capacity(num_pb);
    let mut singleton_col = Vec::with_capacity(num_pb);
    let mut bg_to_pbsamp = HashMap::default();

    let ncols = col_to_batch.len();
    let mut cell_to_pbsamp = vec![usize::MAX; ncols];
    for (i, (batch, group, centroid, count, sc)) in all_pbsamp.into_iter().enumerate() {
        for (d, &v) in centroid.iter().enumerate() {
            centroids[(d, i)] = v;
        }
        cell_counts.push(count);
        pbsamp_to_batch.push(batch);
        pbsamp_to_group.push(group);
        singleton_col.push(sc);
        match sc {
            // A singleton column IS its pb-sample; `(batch, group)` is not a
            // unique key for singletons, so they are mapped directly and stay
            // out of `bg_to_pbsamp`.
            Some(col) => cell_to_pbsamp[col] = i,
            None => {
                bg_to_pbsamp.insert((batch, group), i);
            }
        }
    }

    // Cell → pb-sample inversion for the ordinary (batch, group) blocks;
    // singleton columns were mapped directly above.
    for (group, cells) in group_to_cols.iter().enumerate() {
        for &c in cells {
            let b = col_to_batch[c];
            if cell_to_pbsamp[c] == usize::MAX {
                if let Some(&pbsamp) = bg_to_pbsamp.get(&(b, group)) {
                    cell_to_pbsamp[c] = pbsamp;
                }
            }
        }
    }

    Ok(PbSampleLayout {
        centroids,
        cell_counts,
        pb_sample_to_batch: pbsamp_to_batch,
        pb_sample_to_group: pbsamp_to_group,
        bg_to_pbsamp,
        cell_to_pbsamp,
        singleton_col,
        bulk_batches: bulk_batches.to_vec(),
    })
}

/// Collect gene sums for each pb-sample from a single `SparseIoVec` layer.
///
/// Keyed on the layout's `cell_to_pbsamp` so ordinary (batch, group) blocks
/// and singleton (anchored) pb-samples accumulate through one path.
pub(super) fn collect_pb_sample_gene_sums(
    data_vec: &SparseIoVec,
    group_to_cols: &[Vec<usize>],
    cell_to_pbsamp: &[usize],
    num_pb: usize,
) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
    use indicatif::ParallelProgressIterator;
    let prog_bar = styled_progress_bar(group_to_cols.len() as u64, "groups (pb-sample gene sums)");
    let gene_sum_maps: Vec<(usize, HashMap<usize, f32>)> = group_to_cols
        .par_iter()
        .progress_with(prog_bar.clone())
        .flat_map(|cells| {
            let yy = data_vec
                .read_columns_csc(cells.iter().cloned())
                .expect("read_columns_csc");

            let mut per_pbsamp: HashMap<usize, HashMap<usize, f32>> = HashMap::default();

            for (local_idx, y_j) in yy.col_iter().enumerate() {
                let col = cells[local_idx];
                let pbsamp = cell_to_pbsamp[col];
                if pbsamp == usize::MAX {
                    continue;
                }
                // Weighted to stay consistent with the weighted `cell_counts`
                // in the layout: `collect_matched_stat_coarse` divides one by
                // the other, and the quotient has to remain the per-cell rate.
                let w = data_vec.column_multiplicity(col);
                let gene_map = per_pbsamp.entry(pbsamp).or_default();
                for (&gene, &val) in y_j.row_indices().iter().zip(y_j.values().iter()) {
                    *gene_map.entry(gene).or_default() += val * w;
                }
            }

            per_pbsamp.into_iter().collect::<Vec<_>>()
        })
        .collect();

    let mut gene_sums: Vec<Vec<(usize, f32)>> = vec![vec![]; num_pb];
    for (pbsamp_idx, gene_map) in gene_sum_maps {
        let mut sorted: Vec<(usize, f32)> = gene_map.into_iter().collect();
        sorted.sort_unstable_by_key(|&(g, _)| g);
        gene_sums[pbsamp_idx] = sorted;
    }

    Ok(gene_sums)
}

/// Build pb-samples (layout + gene sums) from a single `SparseIoVec`.
pub(super) fn build_pb_samples(
    data_vec: &SparseIoVec,
    proj_kn: &DMatrix<f32>,
    num_genes: usize,
    anchor_batches: &[usize],
    bulk_batches: &[usize],
) -> anyhow::Result<PbSampleCollection> {
    let group_to_cols = data_vec
        .take_grouped_columns()
        .ok_or(anyhow::anyhow!("columns not assigned to groups"))?;
    let col_to_batch: Vec<usize> = (0..proj_kn.ncols())
        .map(|c| data_vec.get_batch_membership(std::iter::once(c))[0])
        .collect();

    let weights: Option<Vec<f32>> = data_vec.has_column_multiplicity().then(|| {
        (0..proj_kn.ncols())
            .map(|c| data_vec.column_multiplicity(c))
            .collect()
    });
    let layout = build_pb_sample_layout(
        group_to_cols,
        &col_to_batch,
        proj_kn,
        weights.as_deref(),
        anchor_batches,
        bulk_batches,
    )?;
    let num_pb = layout.cell_counts.len();
    let gene_sums =
        collect_pb_sample_gene_sums(data_vec, group_to_cols, &layout.cell_to_pbsamp, num_pb)?;

    Ok(PbSampleCollection {
        layout,
        gene_sums,
        num_genes,
    })
}

/// Recover up to `knn` *distinct* pb-samples (closest, by min cell distance)
/// from a single batch's cell-level HNSW for the given `query` centroid.
///
/// Grows the HNSW query adaptively until `knn` distinct pb-samples are found or
/// the batch's cells are exhausted (bounded by `bknn.len()`). This replaces the
/// old fixed `4·knn+1` oversample, which under-recovered whenever a batch's
/// nearest cells collapsed into fewer than `knn` pb-samples (small / sorting-
/// biased batches) — starving the cross-batch δ estimate of matches.
///
/// `own_pbsamp` and `usize::MAX` (unassigned) cells are skipped. Returned
/// distance per pb-sample is the minimum over its collapsed cells.
pub(crate) fn knn_distinct_pbsamples_in_batch(
    bknn: &ColumnDict<usize>,
    query: &[f32],
    knn: usize,
    cell_to_pbsamp: &[usize],
    own_pbsamp: usize,
) -> anyhow::Result<Vec<(usize, f32)>> {
    let n = bknn.num_points();
    if n == 0 || knn == 0 {
        return Ok(Vec::new());
    }
    // Start at the historical oversample, then grow ×4 per round. `search_by_
    // query_data` self-clamps the request to the batch size, so query_k == n
    // returns every cell — our exhaustion signal.
    let mut query_k = (knn * 4 + 1).min(n);
    let mut best: HashMap<usize, f32> = HashMap::default();
    loop {
        let (cell_ids, dists) = bknn.search_by_query_data(query, query_k)?;
        best.clear();
        for (&c, &d) in cell_ids.iter().zip(dists.iter()) {
            let other_pbsamp = cell_to_pbsamp[c];
            if other_pbsamp == usize::MAX || other_pbsamp == own_pbsamp {
                continue;
            }
            best.entry(other_pbsamp)
                .and_modify(|old| {
                    if d < *old {
                        *old = d;
                    }
                })
                .or_insert(d);
        }
        if best.len() >= knn || query_k >= n {
            break;
        }
        query_k = query_k.saturating_mul(4).min(n);
    }
    let mut per_batch: Vec<(usize, f32)> = best.drain().collect();
    per_batch.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    per_batch.truncate(knn);
    Ok(per_batch)
}

/// For a single `pbsamp`, gather its cross-batch match set: up to `knn`
/// distinct pb-samples from **each** non-own batch (closest to `pbsamp`'s
/// centroid), flattened across batches via [`knn_distinct_pbsamples_in_batch`].
///
/// Shared per-pb-sample body for both [`per_batch_sc_neighbors`] (keeps
/// distances) and `refine_multilevel::build_bbknn_neighbors` (ids only).
pub(crate) fn bbknn_match_one_pbsamp(
    layout: &PbSampleLayout,
    batch_knn_lookup: &[ColumnDict<usize>],
    knn: usize,
    pbsamp: usize,
    anchor_batches: Option<&[usize]>,
) -> anyhow::Result<Vec<(usize, f32)>> {
    // Bulk is a RECEIVER but never a SOURCE — greedy correction, the same
    // discipline the carried pb_reference uses. A bulk column draws its
    // counterfactual from the cell frame and is corrected toward it; the
    // cells self-match through the anchor path, so their frame never moves.
    // Letting bulk serve as a source is what would drag the cells, and that
    // is the only direction excluded here (below, and by keeping bulk out of
    // every anchor set).
    let pbsamp_batch = layout.pb_sample_to_batch[pbsamp];
    let centroid: Vec<f32> = layout.centroids.column(pbsamp).iter().copied().collect();
    let mut all_hits: Vec<(usize, f32)> = Vec::new();
    match anchor_batches {
        // Pooled matching: every non-own, non-bulk batch contributes
        // counterfactual candidates, symmetrically. A bulk batch cannot be a
        // source: matching a cell pb-sample to a mixture is the same
        // composition-into-δ leak in the other direction.
        None => {
            for (b, bknn) in batch_knn_lookup.iter().enumerate() {
                if b == pbsamp_batch || layout.is_bulk_batch(b) {
                    continue;
                }
                let per_batch = knn_distinct_pbsamples_in_batch(
                    bknn,
                    &centroid,
                    knn,
                    &layout.cell_to_pbsamp,
                    pbsamp,
                )?;
                all_hits.extend(per_batch);
            }
        }
        // Greedy (anchored) matching: counterfactuals come from the anchor
        // batches ONLY, for everyone. New batches are corrected toward the
        // anchor frame; an anchor pb-sample's nearest anchor is itself
        // (distance 0 → softmax self-match), so the frame stays fixed and its
        // δ settles at the prior — the reference is never re-adjusted.
        Some(anchors) => {
            for &b in anchors {
                let Some(bknn) = batch_knn_lookup.get(b) else {
                    continue;
                };
                let per_batch = knn_distinct_pbsamples_in_batch(
                    bknn,
                    &centroid,
                    knn,
                    &layout.cell_to_pbsamp,
                    usize::MAX, // self-match allowed
                )?;
                all_hits.extend(per_batch);
            }
        }
    }
    Ok(all_hits)
}

/// Per-pb-sample, for each non-own batch return up to `knn` distinct
/// pb-samples whose member cells are closest to `pbsamp`'s centroid.
///
/// Queries `SparseIoVec::batch_knn_lookup` (per-batch HNSW over cells) via
/// [`knn_distinct_pbsamples_in_batch`], which grows each per-batch search until
/// `knn` distinct pb-samples are recovered or the batch is exhausted.
///
/// Returns: `result[pbsamp] = Vec<(other_pbsamp, distance)>` flattened across all
/// non-own batches.
pub(super) fn per_batch_sc_neighbors(
    layout: &PbSampleLayout,
    batch_knn_lookup: &[ColumnDict<usize>],
    knn: usize,
    anchor_batches: Option<&[usize]>,
) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
    use indicatif::ParallelProgressIterator;
    let num_pb = layout.cell_counts.len();

    let prog_bar = styled_progress_bar(num_pb as u64, "pb-samples (BBKNN match)");
    let result = (0..num_pb)
        .into_par_iter()
        .progress_with(prog_bar.clone())
        .map(|pbsamp| bbknn_match_one_pbsamp(layout, batch_knn_lookup, knn, pbsamp, anchor_batches))
        .collect();
    prog_bar.finish_and_clear();
    result
}

/// Match pb-samples across batches and accumulate counterfactual
/// statistics into `stat.imputed_sum_ds` and `stat.residual_sum_ds`.
///
/// `pbsamp_to_group` is the per-pb-sample group assignment to use when writing
/// into stat columns; callers pass `&layout.pb_sample_to_group` for the
/// hash-partition mapping, or a refined mapping from
/// `refine_multilevel::refine_assignments`.
///
/// `knn` is now the per-other-batch neighbour count: each pb-sample draws
/// up to `knn` distinct foreign pb-samples from **each** non-own batch, so
/// the total match set is up to `knn · (num_batches − 1)`.
pub(super) fn build_pb_sample_to_cells(layout: &PbSampleLayout) -> Vec<Vec<usize>> {
    let num_pb = layout.cell_counts.len();
    let mut out: Vec<Vec<usize>> = vec![vec![]; num_pb];
    for (c, &pbsamp) in layout.cell_to_pbsamp.iter().enumerate() {
        if pbsamp != usize::MAX {
            out[pbsamp].push(c);
        }
    }
    out
}

#[cfg(test)]
#[path = "pb_samples_tests.rs"]
mod tests;
