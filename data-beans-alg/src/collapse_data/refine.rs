//! Multi-level pseudobulk refinement integration.
//!
//! Walks the finest hash partition through BBKNN + Poisson DC-SBM
//! refinement (`crate::refine_multilevel`), then descends level-by-
//! level via `merge_stat`-style aggregation to emit `CollapsedOut`
//! posteriors per coarsening level. Two flavors:
//!
//! - `refine_and_collect_single_layer`: `SparseIoVec` input (per-cell
//!   counts; pb-samples are `(batch, group)` intersections).
//! - `refine_and_collect_stack`: `SparseIoStack` input (per-layer
//!   stacked observations sharing the first-layer grouping decision).
//!
//! Also houses the level-descent helpers (`compute_level_sort_dims`,
//! `compute_fine_to_coarse_mapping`, `fine_to_coarse_from_refined`)
//! and the trivial-identity fallback (`refine_or_identity`) used by
//! the single-batch path.

use super::stats::DEFAULT_COARSEST_SORT_DIM;
use super::*;

pub(super) fn pad_numeric_labels(cell_to_group: &[usize], k: usize) -> Vec<String> {
    let width = {
        let mut w = 1usize;
        let mut n = k.max(1) - 1;
        while n >= 10 {
            w += 1;
            n /= 10;
        }
        w
    };
    cell_to_group
        .iter()
        .map(|g| format!("{:0width$}", g, width = width))
        .collect()
}

/// Derive a fine→coarse group mapping from two consecutive refined levels.
///
/// The refinement pass enforces hierarchy (sibling-constrained moves), so
/// all pb-samples sharing a level-`fine` group also share the same
/// level-`coarse` group. This picks the first pb-sample of each fine group
/// to read the coarse label.
pub(super) fn fine_to_coarse_from_refined(
    pbsamp_to_fine: &[usize],
    pbsamp_to_coarse: &[usize],
    num_fine: usize,
) -> Vec<usize> {
    let mut mapping = vec![usize::MAX; num_fine];
    for pbsamp in 0..pbsamp_to_fine.len() {
        let f = pbsamp_to_fine[pbsamp];
        if mapping[f] == usize::MAX {
            mapping[f] = pbsamp_to_coarse[pbsamp];
        } else {
            debug_assert_eq!(
                mapping[f], pbsamp_to_coarse[pbsamp],
                "refinement broke hierarchy at fine group {}",
                f
            );
        }
    }
    mapping
}

/// Per-level initial pb-sample → group, derived from the finest binary
/// hash codes by bit-masking each level's sort dim and compacting labels to
/// `0..k_level`. Each pb-sample's finest hash code is read from any of its
/// member cells (all cells in a pb-sample share the same finest group).
pub(super) fn initial_per_level_from_hash(
    fine_codes: &[usize],
    pb_sample_to_cells: &[Vec<usize>],
    level_dims: &[usize],
) -> Vec<Vec<usize>> {
    let num_pb = pb_sample_to_cells.len();
    level_dims
        .iter()
        .map(|&d| {
            let mask = if d >= usize::BITS as usize {
                usize::MAX
            } else {
                (1_usize << d).wrapping_sub(1)
            };
            let codes: Vec<usize> = (0..num_pb)
                .map(|pbsamp| fine_codes[pb_sample_to_cells[pbsamp][0]] & mask)
                .collect();
            crate::refine_multilevel::compact_labels(&codes).0
        })
        .collect()
}

/// Run refinement when `allow_refine`, else return the compacted initial
/// mapping unchanged (single-batch → no BBKNN candidates, nothing to refine).
pub(super) fn refine_or_identity(
    allow_refine: bool,
    inputs: &crate::refine_multilevel::RefineInputs<'_>,
    refine_params: &crate::refine_multilevel::RefineParams,
) -> anyhow::Result<crate::refine_multilevel::RefinedAssignment> {
    if allow_refine {
        crate::refine_multilevel::refine_assignments(inputs, refine_params)
    } else {
        let mut pbsamp_to_group: Vec<Vec<usize>> =
            Vec::with_capacity(inputs.initial_sc_to_group_per_level.len());
        let mut num_groups_per_level =
            Vec::with_capacity(inputs.initial_sc_to_group_per_level.len());
        for lvl in inputs.initial_sc_to_group_per_level {
            let (compact, k) = crate::refine_multilevel::compact_labels(lvl);
            num_groups_per_level.push(k);
            pbsamp_to_group.push(compact);
        }
        Ok(crate::refine_multilevel::RefinedAssignment {
            pbsamp_to_group,
            num_groups_per_level,
        })
    }
}

/// Shared inputs to both the `SparseIoVec` and `SparseIoStack` refinement
/// helpers. Keeps call-site signatures compact; every field is derivable
/// from `MultilevelParams` + the finest-level hash partition.
#[derive(Clone, Copy)]
pub(super) struct RefineCollectCtx<'a> {
    pub(super) fine_codes: &'a [usize],
    pub(super) group_to_cols_finest: &'a [Vec<usize>],
    pub(super) level_dims: &'a [usize],
    pub(super) num_features: usize,
    pub(super) num_batches: usize,
    pub(super) knn: usize,
    pub(super) opt_iter: usize,
    pub(super) refine_params: &'a crate::refine_multilevel::RefineParams,
}

/// Refinement integration path for `SparseIoVec`.
///
/// Walks each level of the hash-initialized hierarchy, runs
/// `refine_multilevel::refine_assignments` over pb-samples, then rebuilds
/// `CollapsedStat` per level from the refined cell → group assignment and
/// emits `CollapsedOut` with identical shape to the legacy path. Also
/// surfaces the per-level cell → pb mapping (finest-first, matching
/// `levels`) so consumers — e.g. `graph-embedding-util`'s nested chain
/// sampler — can build pb-tree parent/child maps without rerunning the
/// collapse internals.
pub(super) fn refine_and_collect_single_layer(
    data_vec: &mut SparseIoVec,
    proj_kn: &DMatrix<f32>,
    ctx: &RefineCollectCtx<'_>,
) -> anyhow::Result<MultilevelCollapseOut> {
    let RefineCollectCtx {
        fine_codes,
        group_to_cols_finest,
        level_dims,
        num_features,
        num_batches,
        knn,
        opt_iter,
        refine_params,
    } = *ctx;
    info!(
        "Multi-level refinement path (BBKNN + DC-SBM): {} levels",
        level_dims.len()
    );

    // 1. Build pb-samples (layout + gene sums) from the finest partition.
    let pb_samples = build_pb_samples(data_vec, proj_kn, num_features)?;
    let num_pb = pb_samples.layout.cell_counts.len();
    let ncells_dbg = proj_kn.ncols();
    info!(
        "Built {} pb-samples from {} cells (ratio {:.2}; knn={})",
        num_pb,
        ncells_dbg,
        num_pb as f32 / ncells_dbg.max(1) as f32,
        knn
    );
    if num_pb as f32 > 0.8 * ncells_dbg as f32 {
        warn!(
            "pb-sample count ({}) is close to cell count ({}) — hash partition is too fine \
             (many 1-cell pb-samples). Consider lowering --sort-dim.",
            num_pb, ncells_dbg
        );
    }

    // 2. pbsamp → cells and col → batch.
    let ncols = proj_kn.ncols();
    let col_to_batch: Vec<usize> = data_vec.get_batch_membership(0..ncols);
    let pb_sample_to_cells =
        build_pb_sample_to_cells(&pb_samples.layout, group_to_cols_finest, &col_to_batch);

    let initial_per_level =
        initial_per_level_from_hash(fine_codes, &pb_sample_to_cells, level_dims);
    let empty: [ColumnDict<usize>; 0] = [];
    let batch_knn: &[ColumnDict<usize>] = if num_batches >= 2 {
        data_vec
            .batch_knn_lookup()
            .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?
            .as_slice()
    } else {
        &empty
    };
    let inputs = crate::refine_multilevel::RefineInputs {
        layout: &pb_samples.layout,
        gene_sums: &pb_samples.gene_sums,
        num_genes: num_features,
        pb_sample_to_cells: &pb_sample_to_cells,
        batch_knn_lookup: batch_knn,
        k_per_batch: knn,
        initial_sc_to_group_per_level: &initial_per_level,
    };
    let refined = refine_or_identity(num_batches >= 2, &inputs, refine_params)?;

    // 5. Build finest CollapsedStat once from a full data pass, then derive
    //    coarser levels by `merge_stat` on column-aggregated sums — avoids
    //    re-reading all cells at every level (matches legacy merge descent).
    let num_levels = level_dims.len();
    let k_finest = refined.num_groups_per_level[0];
    let mut cell_to_group_finest = vec![0usize; ncols];
    for (pbsamp, cells) in pb_sample_to_cells.iter().enumerate() {
        let g = refined.pbsamp_to_group[0][pbsamp];
        for &c in cells {
            cell_to_group_finest[c] = g;
        }
    }
    let finest_str = pad_numeric_labels(&cell_to_group_finest, k_finest);
    let nthreads = rayon::current_num_threads();
    info!(
        "Assigning {} cells to {} finest pb-sample groups ({} rayon threads) ...",
        ncols, k_finest, nthreads
    );
    data_vec.assign_groups(&finest_str, None);
    debug_assert_eq!(data_vec.num_groups(), k_finest);

    let mut fine_stat = CollapsedStat::new(num_features, k_finest, num_batches);
    info!("Collecting basic stats over {} groups ...", k_finest);
    data_vec.collect_basic_stat(&mut fine_stat)?;
    if num_batches >= 2 {
        info!(
            "Collecting per-batch stats over {} groups × {} batches ...",
            k_finest, num_batches
        );
        data_vec.collect_batch_stat(&mut fine_stat)?;
        let batch_knn = data_vec
            .batch_knn_lookup()
            .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?;
        info!(
            "Collecting cross-batch matched stats (knn={}) over {} pb-samples ...",
            knn, num_pb
        );
        collect_matched_stat_coarse(
            &pb_samples.layout,
            &pb_samples.gene_sums,
            &refined.pbsamp_to_group[0],
            batch_knn.as_slice(),
            knn,
            &mut fine_stat,
        )?;
    }

    info!(
        "Level 1/{}: refined k={} (finest; {} cells read)",
        num_levels, k_finest, ncols
    );
    let mut results: Vec<CollapsedOut> = Vec::with_capacity(num_levels);
    let finest_out = optimize(
        &fine_stat,
        (1.0, 1.0),
        opt_iter,
        &format!("Coarsen L1/{}", num_levels),
    )?;
    results.push(finest_out);

    let mut prev_stat = fine_stat;
    for level in 1..num_levels {
        let k_prev = refined.num_groups_per_level[level - 1];
        let k_level = refined.num_groups_per_level[level];
        let fine_to_coarse = fine_to_coarse_from_refined(
            &refined.pbsamp_to_group[level - 1],
            &refined.pbsamp_to_group[level],
            k_prev,
        );
        let coarse_stat = merge_stat(&prev_stat, &fine_to_coarse, k_level);
        info!(
            "Level {}/{}: refined k={} (merged from {})",
            level + 1,
            num_levels,
            k_level,
            k_prev
        );
        let level_opt_iter = (opt_iter / 2).max(10);
        let out = optimize(
            &coarse_stat,
            (1.0, 1.0),
            level_opt_iter,
            &format!("Coarsen L{}/{}", level + 1, num_levels),
        )?;
        results.push(out);
        prev_stat = coarse_stat;
    }

    // Build per-level cell → pb mapping (finest-first) by walking
    // refined.pbsamp_to_group[level] through pb_sample_to_cells.
    let mut cell_to_pb_per_level: Vec<Vec<usize>> = Vec::with_capacity(num_levels);
    for level in 0..num_levels {
        let mut c2g = vec![0usize; ncols];
        for (pbsamp, cells) in pb_sample_to_cells.iter().enumerate() {
            let g = refined.pbsamp_to_group[level][pbsamp];
            for &c in cells {
                c2g[c] = g;
            }
        }
        cell_to_pb_per_level.push(c2g);
    }

    Ok(MultilevelCollapseOut {
        levels: results,
        cell_to_pb_per_level,
    })
}

/// Refinement integration path for `SparseIoStack`.
///
/// Shares one `RefinedAssignment` across all layers (first-layer-owns the
/// grouping decision, matching the existing stack convention). Per level ×
/// layer we rebuild `CollapsedStat` and emit `CollapsedOut`.
pub(super) fn refine_and_collect_stack(
    stack: &mut SparseIoStack,
    proj_kn: &DMatrix<f32>,
    ctx: &RefineCollectCtx<'_>,
) -> anyhow::Result<Vec<Vec<CollapsedOut>>> {
    let RefineCollectCtx {
        fine_codes,
        group_to_cols_finest,
        level_dims,
        num_features: _,
        num_batches,
        knn,
        opt_iter,
        refine_params,
    } = *ctx;
    let num_layers = stack.num_types();
    info!(
        "Multi-level stack refinement (BBKNN + DC-SBM): {} layers × {} levels",
        num_layers,
        level_dims.len()
    );

    let ncols = proj_kn.ncols();
    let col_to_batch: Vec<usize> = stack.stack[0].get_batch_membership(0..ncols);

    // Build shared pb-sample layout from layer[0]'s row count and the shared
    // projection. The layout only uses `proj_kn` + grouping, no raw reads.
    let layout = build_pb_sample_layout(group_to_cols_finest, &col_to_batch, proj_kn)?;
    let num_pb = layout.cell_counts.len();

    // Gene sums for layer[0] drive the refinement (first-layer-owns).
    let owner_num_features = stack.stack[0].num_rows();
    let gene_sums_owner = collect_pb_sample_gene_sums(
        &stack.stack[0],
        group_to_cols_finest,
        &col_to_batch,
        &layout.bg_to_pbsamp,
        num_pb,
    )?;

    let pb_sample_to_cells = build_pb_sample_to_cells(&layout, group_to_cols_finest, &col_to_batch);

    let initial_per_level =
        initial_per_level_from_hash(fine_codes, &pb_sample_to_cells, level_dims);
    let empty: [ColumnDict<usize>; 0] = [];
    let batch_knn: &[ColumnDict<usize>] = if num_batches >= 2 {
        stack.stack[0]
            .batch_knn_lookup()
            .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?
            .as_slice()
    } else {
        &empty
    };
    let inputs = crate::refine_multilevel::RefineInputs {
        layout: &layout,
        gene_sums: &gene_sums_owner,
        num_genes: owner_num_features,
        pb_sample_to_cells: &pb_sample_to_cells,
        batch_knn_lookup: batch_knn,
        k_per_batch: knn,
        initial_sc_to_group_per_level: &initial_per_level,
    };
    let refined = refine_or_identity(num_batches >= 2, &inputs, refine_params)?;

    // Per-layer gene_sums for the remaining layers (layer 0 reuses `gene_sums_owner`).
    let mut per_layer_gene_sums: Vec<GeneSums> = Vec::with_capacity(num_layers);
    for (d, layer) in stack.stack.iter().enumerate() {
        if d == 0 {
            per_layer_gene_sums.push(gene_sums_owner.clone());
        } else {
            per_layer_gene_sums.push(collect_pb_sample_gene_sums(
                layer,
                group_to_cols_finest,
                &col_to_batch,
                &layout.bg_to_pbsamp,
                num_pb,
            )?);
        }
    }

    // Finest CollapsedStat per layer via a single data pass, then descend
    //    into coarser levels by `merge_stat` on column aggregates.
    let num_levels = level_dims.len();
    let k_finest = refined.num_groups_per_level[0];
    let mut cell_to_group_finest = vec![0usize; ncols];
    for (pbsamp, cells) in pb_sample_to_cells.iter().enumerate() {
        let g = refined.pbsamp_to_group[0][pbsamp];
        for &c in cells {
            cell_to_group_finest[c] = g;
        }
    }
    let finest_str = pad_numeric_labels(&cell_to_group_finest, k_finest);
    let nthreads = rayon::current_num_threads();
    info!(
        "Assigning {} cells to {} finest pb-sample groups across {} layers ({} rayon threads) ...",
        ncols, k_finest, num_layers, nthreads
    );
    for layer in stack.stack.iter_mut() {
        layer.assign_groups(&finest_str, None);
    }

    let mut fine_stats: Vec<CollapsedStat> = Vec::with_capacity(num_layers);
    let mut finest_layer_results = Vec::with_capacity(num_layers);
    for (d, layer) in stack.stack.iter().enumerate() {
        let num_features = layer.num_rows();
        let mut stat = CollapsedStat::new(num_features, k_finest, num_batches);
        info!(
            "Layer {}/{}: collecting basic stats over {} groups ...",
            d + 1,
            num_layers,
            k_finest
        );
        layer.collect_basic_stat(&mut stat)?;
        if num_batches >= 2 {
            info!(
                "Layer {}/{}: collecting per-batch stats ({} batches) ...",
                d + 1,
                num_layers,
                num_batches
            );
            layer.collect_batch_stat(&mut stat)?;
            let batch_knn = layer
                .batch_knn_lookup()
                .ok_or_else(|| anyhow::anyhow!("batch_knn_lookup not built"))?;
            info!(
                "Layer {}/{}: collecting cross-batch matched stats (knn={}) over {} pb-samples ...",
                d + 1,
                num_layers,
                knn,
                num_pb
            );
            collect_matched_stat_coarse(
                &layout,
                &per_layer_gene_sums[d],
                &refined.pbsamp_to_group[0],
                batch_knn.as_slice(),
                knn,
                &mut stat,
            )?;
        }
        let out = optimize(
            &stat,
            (1.0, 1.0),
            opt_iter,
            &format!("Coarsen L1/{} layer {}/{}", num_levels, d + 1, num_layers),
        )?;
        finest_layer_results.push(out);
        fine_stats.push(stat);
    }
    info!(
        "Level 1/{}: refined k={} (finest; {} layers × {} cells)",
        num_levels, k_finest, num_layers, ncols
    );
    let mut results: Vec<Vec<CollapsedOut>> = Vec::with_capacity(num_levels);
    results.push(finest_layer_results);

    let mut prev_stats = fine_stats;
    for level in 1..num_levels {
        let k_prev = refined.num_groups_per_level[level - 1];
        let k_level = refined.num_groups_per_level[level];
        let fine_to_coarse = fine_to_coarse_from_refined(
            &refined.pbsamp_to_group[level - 1],
            &refined.pbsamp_to_group[level],
            k_prev,
        );
        let level_opt_iter = (opt_iter / 2).max(10);
        let mut layer_results = Vec::with_capacity(num_layers);
        let mut coarse_stats = Vec::with_capacity(num_layers);
        for (d, prev_stat) in prev_stats.iter().enumerate() {
            let coarse_stat = merge_stat(prev_stat, &fine_to_coarse, k_level);
            let out = optimize(
                &coarse_stat,
                (1.0, 1.0),
                level_opt_iter,
                &format!(
                    "Coarsen L{}/{} layer {}/{}",
                    level + 1,
                    num_levels,
                    d + 1,
                    num_layers
                ),
            )?;
            layer_results.push(out);
            coarse_stats.push(coarse_stat);
        }
        info!(
            "Level {}/{}: refined k={} (merged from {}, {} layers)",
            level + 1,
            num_levels,
            k_level,
            k_prev,
            num_layers
        );
        results.push(layer_results);
        prev_stats = coarse_stats;
    }

    Ok(results)
}

/// Compute sort dimensions for each level, linearly spaced from
/// finest to coarsest (fine→coarse). Duplicate dimensions are
/// removed so that extra levels don't repeat the same partitioning.
pub(super) fn compute_level_sort_dims(finest_sort_dim: usize, num_levels: usize) -> Vec<usize> {
    if num_levels <= 1 {
        return vec![finest_sort_dim];
    }
    let coarsest = DEFAULT_COARSEST_SORT_DIM.min(finest_sort_dim);
    let mut dims = Vec::with_capacity(num_levels);
    for level in 0..num_levels {
        // t goes from 0 (finest) to 1 (coarsest)
        let t = level as f32 / (num_levels - 1) as f32;
        let dim = finest_sort_dim as f32 - t * (finest_sort_dim - coarsest) as f32;
        let dim = dim.round() as usize;
        if dims.last() != Some(&dim) {
            dims.push(dim);
        }
    }
    dims
}

/// Compute the mapping from fine group indices to coarse group indices.
///
/// Each fine group's binary code is masked to `coarse_dim` bits to
/// produce its coarse code. Unique coarse codes are assigned
/// consecutive indices.
pub(super) fn compute_fine_to_coarse_mapping(
    group_to_cols: &[Vec<usize>],
    fine_codes: &[usize],
    coarse_dim: usize,
) -> (Vec<usize>, usize) {
    let coarse_mask = (1_usize << coarse_dim) - 1;

    // For each fine group, look up binary code from any member column
    let coarse_codes: Vec<usize> = group_to_cols
        .iter()
        .map(|cols| fine_codes[cols[0]] & coarse_mask)
        .collect();

    // Unique coarse codes → consecutive indices
    let mut unique_coarse: Vec<usize> = coarse_codes.to_vec();
    unique_coarse.sort_unstable();
    unique_coarse.dedup();
    let num_coarse = unique_coarse.len();

    let coarse_to_idx: HashMap<usize, usize> = unique_coarse
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect();

    let fine_to_coarse: Vec<usize> = coarse_codes.iter().map(|c| coarse_to_idx[c]).collect();

    (fine_to_coarse, num_coarse)
}
