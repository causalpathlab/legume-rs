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
/// This only uses `proj_kn` (no CSC reads), so it can be shared across layers.
pub(super) fn build_pb_sample_layout(
    group_to_cols: &[Vec<usize>],
    col_to_batch: &[usize],
    proj_kn: &DMatrix<f32>,
) -> anyhow::Result<PbSampleLayout> {
    let proj_dim = proj_kn.nrows();

    /// Intermediate per-batch accumulator for centroid computation.
    struct CentroidAccum {
        centroid_sum: Vec<f32>,
        count: usize,
    }

    // Collect centroid data per group in parallel
    type CentroidTuple = (usize, usize, Vec<f32>, f32);
    let per_group_results: Vec<Vec<CentroidTuple>> = group_to_cols
        .par_iter()
        .enumerate()
        .map(|(group, cells)| {
            let mut batch_data: HashMap<usize, CentroidAccum> = HashMap::default();

            for &glob_idx in cells {
                let batch = col_to_batch[glob_idx];
                let acc = batch_data.entry(batch).or_insert_with(|| CentroidAccum {
                    centroid_sum: vec![0f32; proj_dim],
                    count: 0,
                });
                for d in 0..proj_dim {
                    acc.centroid_sum[d] += proj_kn[(d, glob_idx)];
                }
                acc.count += 1;
            }

            batch_data
                .into_iter()
                .filter(|(_, acc)| acc.count > 0)
                .map(|(batch, acc)| {
                    let inv_count = 1.0 / acc.count as f32;
                    let centroid: Vec<f32> =
                        acc.centroid_sum.iter().map(|v| v * inv_count).collect();
                    (batch, group, centroid, acc.count as f32)
                })
                .collect::<Vec<_>>()
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
    let mut bg_to_pbsamp = HashMap::default();

    for (i, (batch, group, centroid, count)) in all_pbsamp.into_iter().enumerate() {
        for (d, &v) in centroid.iter().enumerate() {
            centroids[(d, i)] = v;
        }
        cell_counts.push(count);
        pbsamp_to_batch.push(batch);
        pbsamp_to_group.push(group);
        bg_to_pbsamp.insert((batch, group), i);
    }

    // Cell → pb-sample inversion (each cell belongs to exactly one pbsamp via
    // (batch, group)).
    let ncols = col_to_batch.len();
    let mut cell_to_pbsamp = vec![usize::MAX; ncols];
    for (group, cells) in group_to_cols.iter().enumerate() {
        for &c in cells {
            let b = col_to_batch[c];
            if let Some(&pbsamp) = bg_to_pbsamp.get(&(b, group)) {
                cell_to_pbsamp[c] = pbsamp;
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
    })
}

/// Collect gene sums for each pb-sample from a single `SparseIoVec` layer.
///
/// Uses the `bg_to_pbsamp` mapping from the layout to accumulate gene expression
/// per pb-sample, parallelized over groups.
pub(super) fn collect_pb_sample_gene_sums(
    data_vec: &SparseIoVec,
    group_to_cols: &[Vec<usize>],
    col_to_batch: &[usize],
    bg_to_pbsamp: &HashMap<(usize, usize), usize>,
    num_pb: usize,
) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
    use indicatif::ParallelProgressIterator;
    let prog_bar = styled_progress_bar(group_to_cols.len() as u64, "groups (pb-sample gene sums)");
    let gene_sum_maps: Vec<(usize, HashMap<usize, f32>)> = group_to_cols
        .par_iter()
        .enumerate()
        .progress_with(prog_bar.clone())
        .flat_map(|(group, cells)| {
            let yy = data_vec
                .read_columns_csc(cells.iter().cloned())
                .expect("read_columns_csc");

            let mut batch_gene_sums: HashMap<usize, HashMap<usize, f32>> = HashMap::default();

            for (local_idx, y_j) in yy.col_iter().enumerate() {
                let batch = col_to_batch[cells[local_idx]];
                let gene_map = batch_gene_sums.entry(batch).or_default();
                for (&gene, &val) in y_j.row_indices().iter().zip(y_j.values().iter()) {
                    *gene_map.entry(gene).or_default() += val;
                }
            }

            batch_gene_sums
                .into_iter()
                .filter_map(|(batch, gene_map)| {
                    bg_to_pbsamp
                        .get(&(batch, group))
                        .map(|&pbsamp_idx| (pbsamp_idx, gene_map))
                })
                .collect::<Vec<_>>()
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
) -> anyhow::Result<PbSampleCollection> {
    let group_to_cols = data_vec
        .take_grouped_columns()
        .ok_or(anyhow::anyhow!("columns not assigned to groups"))?;
    let col_to_batch: Vec<usize> = (0..proj_kn.ncols())
        .map(|c| data_vec.get_batch_membership(std::iter::once(c))[0])
        .collect();

    let layout = build_pb_sample_layout(group_to_cols, &col_to_batch, proj_kn)?;
    let num_pb = layout.cell_counts.len();
    let gene_sums = collect_pb_sample_gene_sums(
        data_vec,
        group_to_cols,
        &col_to_batch,
        &layout.bg_to_pbsamp,
        num_pb,
    )?;

    Ok(PbSampleCollection {
        layout,
        gene_sums,
        num_genes,
    })
}

/// Per-pb-sample, for each non-own batch return up to `knn` distinct
/// pb-samples whose member cells are closest to `pbsamp`'s centroid.
///
/// Queries `SparseIoVec::batch_knn_lookup` (per-batch HNSW over cells),
/// then dedups hits to pb-samples via `layout.cell_to_pbsamp`. Distances come
/// back as the minimum over collapsed cells for each pb-sample.
///
/// Returns: `result[pbsamp] = Vec<(other_pbsamp, distance)>` flattened across all
/// non-own batches.
pub(super) fn per_batch_sc_neighbors(
    layout: &PbSampleLayout,
    batch_knn_lookup: &[ColumnDict<usize>],
    knn: usize,
) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
    use indicatif::ParallelProgressIterator;
    use matrix_util::knn_match::MakeVecPoint;
    let num_pb = layout.cell_counts.len();
    // Oversample cells per batch so dedup-to-pbsamp still yields ~knn uniques.
    let cell_oversample = (knn * 4 + 1).max(knn);

    let prog_bar = styled_progress_bar(num_pb as u64, "pb-samples (BBKNN match)");
    let result: anyhow::Result<Vec<Vec<(usize, f32)>>> = (0..num_pb)
        .into_par_iter()
        .progress_with(prog_bar.clone())
        .map(|pbsamp| -> anyhow::Result<Vec<(usize, f32)>> {
            let pbsamp_batch = layout.pb_sample_to_batch[pbsamp];
            let centroid = layout.centroids.column(pbsamp).to_vp();
            let mut all_hits: Vec<(usize, f32)> = Vec::new();
            for (b, bknn) in batch_knn_lookup.iter().enumerate() {
                if b == pbsamp_batch {
                    continue;
                }
                let (cell_ids, dists) = bknn.search_by_query_data(&centroid, cell_oversample)?;
                let mut best: HashMap<usize, f32> = HashMap::default();
                for (&c, &d) in cell_ids.iter().zip(dists.iter()) {
                    let other_pbsamp = layout.cell_to_pbsamp[c];
                    if other_pbsamp == usize::MAX || other_pbsamp == pbsamp {
                        continue;
                    }
                    best.entry(other_pbsamp)
                        .and_modify(|old| {
                            if d < *old {
                                *old = d;
                            }
                        })
                        .or_insert(d);
                    if best.len() >= knn {
                        break;
                    }
                }
                let mut per_batch: Vec<(usize, f32)> = best.into_iter().collect();
                per_batch
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                per_batch.truncate(knn);
                all_hits.extend(per_batch);
            }
            Ok(all_hits)
        })
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
pub(super) fn build_pb_sample_to_cells(
    layout: &PbSampleLayout,
    group_to_cols_finest: &[Vec<usize>],
    col_to_batch: &[usize],
) -> Vec<Vec<usize>> {
    let num_pb = layout.cell_counts.len();
    let mut out: Vec<Vec<usize>> = vec![vec![]; num_pb];
    for (group, cells) in group_to_cols_finest.iter().enumerate() {
        for &c in cells {
            let batch = col_to_batch[c];
            if let Some(&pbsamp) = layout.bg_to_pbsamp.get(&(batch, group)) {
                out[pbsamp].push(c);
            }
        }
    }
    out
}
