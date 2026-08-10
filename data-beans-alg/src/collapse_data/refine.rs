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

/// Per-level reprojection offsets consumed by `refine_assignments`: for each
/// pb-sample, the finest hash bits *above* the parent level's sort dim
/// (`code >> parent_dim`, masked to `child_dim − parent_dim` bits). Crossing
/// the refined parent with these "extra bits" keeps each finer level bounded
/// by `2^child_dim` while staying hash-meaningful (cells agreeing on the finer
/// SVD dims group together), instead of an arbitrary positional index. The
/// coarsest level has no parent, so its entry is empty (unused).
pub(super) fn build_reproject_offsets(
    fine_codes: &[usize],
    pb_sample_to_cells: &[Vec<usize>],
    level_dims: &[usize],
) -> Vec<Vec<usize>> {
    let raw: Vec<usize> = pb_sample_to_cells
        .iter()
        .map(|cells| fine_codes[cells[0]])
        .collect();
    (0..level_dims.len())
        .map(|level| {
            if level + 1 < level_dims.len() {
                let parent_dim = level_dims[level + 1];
                let nbits = level_dims[level].saturating_sub(parent_dim);
                let mask = if nbits >= usize::BITS as usize {
                    usize::MAX
                } else {
                    (1_usize << nbits).wrapping_sub(1)
                };
                raw.iter().map(|&c| (c >> parent_dim) & mask).collect()
            } else {
                Vec::new()
            }
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

/// Append-only memory: give each anchored (carried) pb-sample its own
/// finest group, appended after the groups of the ordinary pb-samples.
///
/// The refined partition still decides the finest groups of the NEW columns;
/// carried columns are removed from those groups (a group left with only
/// carried members disappears in the compaction) and instead keep their
/// stored granularity as singleton groups, ordered by column index so a
/// re-emitted reference preserves the parent's column order. Every finest
/// stat for a singleton group then reduces to the carried column's own
/// values — observed sum `y·w` over size `w` is the stored rate again — so
/// carrying a reference through a round reproduces it instead of
/// re-averaging it into (batch × group) blends, which would compound
/// resolution loss every round. Coarser levels are left blended: they are
/// transient training aids recomputed each round from the frozen finest
/// columns, so averaging there does not compound.
///
/// Only level 0 is rewritten, and splitting a group cannot break the
/// sibling-constrained hierarchy the coarser levels assume. No-op when no
/// pb-sample belongs to an anchor batch.
pub(super) fn split_anchored_finest_groups(
    refined: &mut crate::refine_multilevel::RefinedAssignment,
    layout: &PbSampleLayout,
    pb_sample_to_cells: &[Vec<usize>],
    anchor_batches: &[usize],
) {
    let num_pb = layout.pb_sample_to_batch.len();
    // Anchored pb-samples are singletons (see `build_pb_sample_layout`), so
    // each maps to exactly one column; sort by that column to pin the
    // appended group order to the parent reference's column order.
    let mut anchored: Vec<(usize, usize)> = (0..num_pb)
        .filter(|&p| anchor_batches.contains(&layout.pb_sample_to_batch[p]))
        .map(|p| (pb_sample_to_cells[p][0], p))
        .collect();
    if anchored.is_empty() {
        return;
    }
    anchored.sort_unstable_by_key(|&(col, _)| col);
    let mut is_anchored = vec![false; num_pb];
    for &(_, p) in &anchored {
        is_anchored[p] = true;
    }

    let finest = &mut refined.pbsamp_to_group[0];
    let mut remap = vec![usize::MAX; refined.num_groups_per_level[0]];
    let mut k_new = 0usize;
    for p in 0..num_pb {
        if is_anchored[p] {
            continue;
        }
        let g = finest[p];
        if remap[g] == usize::MAX {
            remap[g] = k_new;
            k_new += 1;
        }
        finest[p] = remap[g];
    }
    for (j, &(_, p)) in anchored.iter().enumerate() {
        finest[p] = k_new + j;
    }
    refined.num_groups_per_level[0] = k_new + anchored.len();
    info!(
        "Append-only finest partition: {} new-data groups + {} carried singletons",
        k_new,
        anchored.len()
    );
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
    /// Posterior planes the emitted `CollapsedOut` should carry (threaded
    /// from `MultilevelParams::output_calibration`).
    pub(super) output_calibration: matrix_param::traits::CalibrateTarget,
    /// Resolved anchor-batch indices — see `MultilevelParams::anchor_batches`.
    pub(super) anchor_batches: Option<&'a [usize]>,
    /// See `MultilevelParams::observe_panels`.
    pub(super) observe_panels: bool,
    /// See `MultilevelParams::keep_finest_stats`.
    pub(super) keep_finest_stats: bool,
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
        group_to_cols_finest: _,
        level_dims,
        num_features,
        num_batches,
        knn,
        opt_iter,
        refine_params,
        output_calibration,
        anchor_batches: _,
        observe_panels: _,
        keep_finest_stats: _,
    } = *ctx;
    info!(
        "Multi-level refinement path (BBKNN + DC-SBM): {} levels",
        level_dims.len()
    );

    // 1. Build pb-samples (layout + gene sums) from the finest partition.
    let pb_samples = build_pb_samples(
        data_vec,
        proj_kn,
        num_features,
        ctx.anchor_batches.unwrap_or(&[]),
    )?;
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

    // 2. pbsamp → cells, via the layout's own column mapping.
    let ncols = proj_kn.ncols();
    let pb_sample_to_cells = build_pb_sample_to_cells(&pb_samples.layout);

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
    let reproject_offsets = build_reproject_offsets(fine_codes, &pb_sample_to_cells, level_dims);
    let inputs = crate::refine_multilevel::RefineInputs {
        layout: &pb_samples.layout,
        gene_sums: &pb_samples.gene_sums,
        num_genes: num_features,
        pb_sample_to_cells: &pb_sample_to_cells,
        batch_knn_lookup: batch_knn,
        k_per_batch: knn,
        initial_sc_to_group_per_level: &initial_per_level,
        reproject_offsets_per_level: &reproject_offsets,
    };
    let mut refined = refine_or_identity(num_batches >= 2, &inputs, refine_params)?;
    split_anchored_finest_groups(
        &mut refined,
        &pb_samples.layout,
        &pb_sample_to_cells,
        ctx.anchor_batches.unwrap_or(&[]),
    );
    let refined = refined;

    ///////////////////////////////////
    // collapse-structure diagnostic //
    ///////////////////////////////////
    // Resolve how the finest column count actually arises: distinct leaf
    // codes (initial finest groups) vs refined finest groups, and whether a
    // refined finest group spans multiple batches (i.e. batches are merged
    // within a group) or stays single-batch (a per-(leaf,batch) column).
    {
        let n_leaves = initial_per_level
            .first()
            .map(|lvl| lvl.iter().copied().max().map_or(0, |m| m + 1))
            .unwrap_or(0);
        let finest = &refined.pbsamp_to_group[0];
        let k_fin = refined.num_groups_per_level[0];
        // Bitmask of batches per finest group (handles up to 128 batches).
        let mut batch_mask = vec![0u128; k_fin];
        let b2g = &pb_samples.layout.pb_sample_to_batch;
        for (pb, &g) in finest.iter().enumerate() {
            let b = b2g[pb];
            if b < 128 {
                batch_mask[g] |= 1u128 << b;
            }
        }
        let spans: Vec<u32> = batch_mask.iter().map(|m| m.count_ones()).collect();
        let multi = spans.iter().filter(|&&c| c > 1).count();
        let max_b = spans.iter().copied().max().unwrap_or(0);
        let mean_b = spans.iter().map(|&c| c as f64).sum::<f64>() / k_fin.max(1) as f64;
        info!(
            "collapse structure: {} pb-samples, {} batches, {} leaf codes (finest init), \
             {} refined finest groups; finest groups spanning >1 batch: {}/{} \
             (max {} batches/group, mean {:.2})",
            num_pb, num_batches, n_leaves, k_fin, multi, k_fin, max_b, mean_b
        );
    }

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
            ctx.anchor_batches,
            &mut fine_stat,
        )?;
    }

    info!(
        "Level 1/{}: refined k={} (finest; {} cells read)",
        num_levels, k_finest, ncols
    );
    if ctx.observe_panels {
        attach_observability(&mut fine_stat, data_vec)?;
    }

    let mut results: Vec<CollapsedOut> = Vec::with_capacity(num_levels);
    let finest_out = optimize(
        &fine_stat,
        (1.0, 1.0),
        opt_iter,
        &format!("Fit L1/{}", num_levels),
        output_calibration,
        ctx.keep_finest_stats,
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
            &format!("Fit L{}/{}", level + 1, num_levels),
            output_calibration,
            false,
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
        output_calibration,
        anchor_batches: _,
        observe_panels: _,
        keep_finest_stats: _,
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
    let layout = build_pb_sample_layout(group_to_cols_finest, &col_to_batch, proj_kn, None, &[])?;
    let num_pb = layout.cell_counts.len();

    // Gene sums for layer[0] drive the refinement (first-layer-owns).
    let owner_num_features = stack.stack[0].num_rows();
    let gene_sums_owner = collect_pb_sample_gene_sums(
        &stack.stack[0],
        group_to_cols_finest,
        &layout.cell_to_pbsamp,
        num_pb,
    )?;

    let pb_sample_to_cells = build_pb_sample_to_cells(&layout);

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
    let reproject_offsets = build_reproject_offsets(fine_codes, &pb_sample_to_cells, level_dims);
    let inputs = crate::refine_multilevel::RefineInputs {
        layout: &layout,
        gene_sums: &gene_sums_owner,
        num_genes: owner_num_features,
        pb_sample_to_cells: &pb_sample_to_cells,
        batch_knn_lookup: batch_knn,
        k_per_batch: knn,
        initial_sc_to_group_per_level: &initial_per_level,
        reproject_offsets_per_level: &reproject_offsets,
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
                &layout.cell_to_pbsamp,
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
                ctx.anchor_batches,
                &mut stat,
            )?;
        }
        let out = optimize(
            &stat,
            (1.0, 1.0),
            opt_iter,
            &format!("Fit L1/{} layer {}/{}", num_levels, d + 1, num_layers),
            output_calibration,
            false,
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
                    "Fit L{}/{} layer {}/{}",
                    level + 1,
                    num_levels,
                    d + 1,
                    num_layers
                ),
                output_calibration,
                false,
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

#[cfg(test)]
#[path = "refine_tests.rs"]
mod reproject_tests;
