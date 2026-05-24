//! Barcode-keyed loader for arbitrary multi-modal panels. Each input
//! file contributes its rows to a unified feature axis; cells (barcodes)
//! are unioned across files. Per-cell batch labels (from `batch_files`,
//! `@batch` tags, or file names) are deduped into a global namespace so
//! downstream samplers can restrict negatives to within-batch.

use crate::progress::new_progress_bar;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use auxiliary_data::feature_names::FeatureNameKind;
use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use log::info;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Clone, Copy)]
pub struct Triplet {
    pub cell: u32,
    pub feature: u32,
    pub count: f32,
}

pub struct UnifiedData {
    pub triplets: Vec<Triplet>,
    pub feature_names: Vec<Box<str>>,
    pub per_file_data: Vec<SparseIoVec>,
    pub barcodes: Vec<Box<str>>,
    /// Global cell id → global batch id.
    pub batch_membership: Vec<u32>,
    /// Global batch labels in id order.
    pub batch_names: Vec<Box<str>>,
    /// For each compact feature id `i ∈ 0..n_features()`, the row
    /// index into `per_file_data[0]` (the underlying `SparseIoVec`).
    /// Identity (`0..n_features()`) when no HVG was applied; the
    /// HVG selection indices after `subset_features`. Used when an
    /// upstream pass (e.g. NB-Fisher) operates on the full backend
    /// and the result needs to be aligned to the compact axis.
    pub feature_to_backend_row: Vec<usize>,
    /// Per unified cell, a bitmask of which input files (modalities) the
    /// cell appears in: bit `b` set ⇔ the cell's barcode is present in
    /// backend `b`. Single-file / non-multiome loads set bit 0 for every
    /// cell. `count_ones() >= 2` marks a **matched** (multi-modal bridge)
    /// cell. Drives `--multiome` auto modality-batching and matched-cell
    /// up-weighting.
    pub cell_modality: Vec<u32>,
}

impl UnifiedData {
    pub fn n_cells(&self) -> usize {
        self.barcodes.len()
    }

    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    pub fn n_batches(&self) -> usize {
        self.batch_names.len()
    }

    /// Build a synthetic `UnifiedData` whose "cells" are pseudobulks
    /// (pb-samples from `collapse_columns_multilevel_vec`). Used by
    /// the two-stage fit: stage 1 trains `E_feat` on this batch-
    /// corrected pseudobulk count matrix, then freezes it for the
    /// per-cell stage-2 pass.
    ///
    /// `pb_count_ds` is `[D, S]` (features × pseudobulks). Nonzero
    /// entries become triplets. `feature_names` carries through the
    /// HVG-subsetted names so the final dictionary parquet is keyed
    /// on the right gene symbols. `feature_to_backend_row` is reused
    /// for any downstream pass that needs to address the raw backend.
    pub fn from_pseudobulks(
        pb_count_ds: &nalgebra::DMatrix<f32>,
        feature_names: Vec<Box<str>>,
        feature_to_backend_row: Vec<usize>,
    ) -> anyhow::Result<Self> {
        let n_features = pb_count_ds.nrows();
        let n_pb = pb_count_ds.ncols();
        anyhow::ensure!(
            feature_names.len() == n_features,
            "feature_names {} != pseudobulk rows {}",
            feature_names.len(),
            n_features
        );
        anyhow::ensure!(
            feature_to_backend_row.len() == n_features,
            "feature_to_backend_row {} != pseudobulk rows {}",
            feature_to_backend_row.len(),
            n_features
        );

        // Triplet-ize in parallel across pb columns. nalgebra is
        // column-major, so each worker scans one contiguous slab.
        // Per-pb output is locally sorted by feature → concat preserves
        // global sort by (cell, feature) for downstream consumers.
        let pb_bar = new_progress_bar(n_pb as u64);
        pb_bar.set_message("pb triplet-ize");
        let triplets: Vec<Triplet> = (0..n_pb)
            .into_par_iter()
            .progress_with(pb_bar.clone())
            .flat_map_iter(|s| {
                let s_u32 = s as u32;
                (0..n_features).filter_map(move |g| {
                    let c = pb_count_ds[(g, s)];
                    (c > 0.0).then_some(Triplet {
                        cell: s_u32,
                        feature: g as u32,
                        count: c,
                    })
                })
            })
            .collect();
        pb_bar.finish_and_clear();

        let barcodes: Vec<Box<str>> = (0..n_pb)
            .map(|i| format!("pb_{i}").into_boxed_str())
            .collect();

        // Pseudobulks are already batch-corrected, so we collapse to a
        // single synthetic "all" batch; the within-batch negative
        // sampler then becomes one global pool, which is appropriate
        // for stage-1 training on already-corrected counts.
        let batch_membership = vec![0u32; n_pb];
        let batch_names: Vec<Box<str>> = vec!["all".to_string().into_boxed_str()];

        Ok(UnifiedData {
            triplets,
            feature_names,
            per_file_data: Vec::new(),
            barcodes,
            batch_membership,
            batch_names,
            feature_to_backend_row,
            // Synthetic pseudobulk "cells" are single-modality.
            cell_modality: vec![1u32; n_pb],
        })
    }

    /// Restrict to a subset of features (by current global feature id).
    /// Drops triplets whose feature isn't in `selected_indices`, remaps
    /// remaining features to compact 0..k indices, updates
    /// `feature_names`, and composes `feature_to_backend_row` so the
    /// new compact index `i` still maps to the right row in
    /// `per_file_data[0]`.
    ///
    /// Used to apply HVG selection (`data_beans_alg::hvg::select_hvg_streaming`)
    /// at the gbe layer without reaching back into `SparseIoVec`.
    pub fn subset_features(&mut self, selected_indices: &[usize]) {
        if selected_indices.len() == self.n_features() {
            return; // no-op
        }
        let n_old = self.n_features();
        let mut old_to_new: Vec<i32> = vec![-1; n_old];
        for (new_i, &old_i) in selected_indices.iter().enumerate() {
            debug_assert!(old_i < n_old, "selected index {old_i} out of range");
            old_to_new[old_i] = new_i as i32;
        }

        let new_feature_names: Vec<Box<str>> = selected_indices
            .iter()
            .map(|&i| self.feature_names[i].clone())
            .collect();
        let new_feature_to_backend_row: Vec<usize> = selected_indices
            .iter()
            .map(|&i| self.feature_to_backend_row[i])
            .collect();

        let n_before = self.triplets.len();
        // Compact + remap in place: no second triplet vec, so peak memory
        // stays at the single edge list instead of briefly holding old+new.
        self.triplets.retain_mut(|t| {
            let new_id = old_to_new[t.feature as usize];
            if new_id >= 0 {
                t.feature = new_id as u32;
                true
            } else {
                false
            }
        });

        log::info!(
            "Feature subset: {} → {} features ({} → {} edges retained)",
            n_old,
            new_feature_names.len(),
            n_before,
            self.triplets.len()
        );
        self.feature_names = new_feature_names;
        self.feature_to_backend_row = new_feature_to_backend_row;
    }

    /// Build a `UnifiedData` from a single in-memory `SparseIoVec` plus
    /// per-cell batch labels. Internal helper for [`load_unified_data`].
    pub(crate) fn from_sparse_io(
        data: SparseIoVec,
        batch_labels: &[Box<str>],
        auto_modality_batch: bool,
    ) -> anyhow::Result<Self> {
        let feature_names = data.row_names()?;
        let barcodes = data.column_names()?;

        // Per-cell modality bitmask: which input backend(s) each unified
        // barcode appears in. Single-backend loads → bit 0 for all cells.
        let n_back = data.len();
        let cell_modality = if n_back > 1 {
            let bc_to_idx: FxHashMap<&str, usize> = barcodes
                .iter()
                .enumerate()
                .map(|(i, b)| (b.as_ref(), i))
                .collect();
            let mut mask = vec![0u32; barcodes.len()];
            for b in 0..n_back.min(32) {
                for name in data[b].column_names()? {
                    if let Some(&i) = bc_to_idx.get(name.as_ref()) {
                        mask[i] |= 1u32 << b;
                    }
                }
            }
            mask
        } else {
            vec![1u32; barcodes.len()]
        };

        // `--multiome` auto-batch: when no explicit batch labels were
        // resolved, group cells by their modality-presence pattern so the
        // δ-correction integrates matched + unimodal cohorts onto shared
        // axes. Otherwise honor the labels the loader resolved.
        let derived: Vec<Box<str>>;
        let batch_labels: &[Box<str>] = if auto_modality_batch && n_back > 1 {
            derived = cell_modality
                .iter()
                .map(|&m| modality_presence_label(m, n_back))
                .collect();
            &derived
        } else {
            batch_labels
        };
        anyhow::ensure!(
            batch_labels.len() == barcodes.len(),
            "batch labels {} != cells {}",
            batch_labels.len(),
            barcodes.len()
        );

        // Dedupe batch labels into a global namespace (in first-seen order).
        let mut batch_to_global: FxHashMap<Box<str>, u32> = FxHashMap::default();
        let mut batch_names: Vec<Box<str>> = Vec::new();
        let batch_membership: Vec<u32> = batch_labels
            .iter()
            .map(|label| {
                if let Some(&g) = batch_to_global.get(label) {
                    g
                } else {
                    let g = batch_names.len() as u32;
                    batch_to_global.insert(label.clone(), g);
                    batch_names.push(label.clone());
                    g
                }
            })
            .collect();

        info!(
            "Unified barcodes: {} total (single in-memory data set)",
            barcodes.len()
        );
        info!(
            "Unified batches: {} ({})",
            batch_names.len(),
            batch_names
                .iter()
                .map(|s| s.as_ref())
                .collect::<Vec<_>>()
                .join(", ")
        );

        info!("Materializing sparse triplets (unified feature space)...");
        // Stream nonzeros straight into compact 12-byte triplets: the wide
        // (u64,u64,f32) form is never built in full, only one bounded slab
        // at a time. Pre-size from the backend nnz so the edge list never
        // reallocs, and pick a slab width that keeps each backend read well
        // under a few hundred MB regardless of per-cell density.
        let n_cells = data.num_columns();
        let nnz_hint = data.num_non_zeros().unwrap_or(0);
        let avg_per_col = (nnz_hint / n_cells.max(1)).max(1);
        let chunk_cols = (8_000_000 / avg_per_col).clamp(1, n_cells.max(1));
        let pb_bar = new_progress_bar(nnz_hint.max(1) as u64);
        pb_bar.set_message("triplets");
        let mut triplets: Vec<Triplet> = Vec::with_capacity(nnz_hint);
        let mut since_tick = 0u64;
        data.for_each_triplet(0..n_cells, chunk_cols, |row, col, v| {
            if v != 0.0 {
                triplets.push(Triplet {
                    cell: col as u32,
                    feature: row as u32,
                    count: v,
                });
                since_tick += 1;
                if since_tick == 1 << 20 {
                    pb_bar.inc(since_tick);
                    since_tick = 0;
                }
            }
        })?;
        pb_bar.inc(since_tick);
        pb_bar.finish_and_clear();
        info!("  edges: {}", triplets.len());

        let n_features = feature_names.len();
        Ok(UnifiedData {
            triplets,
            feature_names,
            feature_to_backend_row: (0..n_features).collect(),
            per_file_data: vec![data],
            barcodes,
            batch_membership,
            batch_names,
            cell_modality,
        })
    }
}

/// Human-readable batch label for a modality-presence bitmask, e.g.
/// `m0`, `m1`, or `m0+m2`. Stable across cells with the same pattern, so
/// the dedupe in `from_sparse_io` collapses them into one batch each.
fn modality_presence_label(mask: u32, n_back: usize) -> Box<str> {
    let present: Vec<String> = (0..n_back.min(32))
        .filter(|&b| mask & (1u32 << b) != 0)
        .map(|b| format!("m{b}"))
        .collect();
    if present.is_empty() {
        "none".to_string().into_boxed_str()
    } else {
        present.join("+").into_boxed_str()
    }
}

/// Load one or more sparse files into a single [`UnifiedData`] by
/// delegating to `read_data_on_shared_rows` (the same loader senna's
/// run-manifest-driven flows use). This guarantees the barcodes stored
/// in `unified.barcodes` — and therefore the row names in
/// `latent.parquet` — line up byte-for-byte with what `senna plot
/// --from`, `senna clustering --from`, etc. observe when they re-load
/// the data. With multiple files, both paths see the
/// `<barcode>@<basename>` disambiguator from `SparseIoVec::push`.
///
/// Cells with the same raw barcode in two files are therefore treated
/// as **distinct** cells (matching the rest of senna). True multimodal
/// stacking — different feature schemas per file with shared cells —
/// is out of scope for this loader.
pub fn load_unified_data(
    data_files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
    feature_kind: FeatureNameKind,
    preload: bool,
    column_alignment: data_beans::sparse_io_vector::ColumnAlignment,
) -> anyhow::Result<UnifiedData> {
    use data_beans::sparse_io_vector::ColumnAlignment;
    if let Some(bf) = batch_files {
        // Under `ColumnAlignment::Union` a single barcode can be in
        // multiple files, so per-file batch lists no longer compose;
        // the loader requires exactly one batch file in that mode.
        // Mirror the same expectation here.
        match column_alignment {
            ColumnAlignment::Disjoint => anyhow::ensure!(
                bf.len() == data_files.len(),
                "batch_files length {} != data_files length {}",
                bf.len(),
                data_files.len()
            ),
            ColumnAlignment::Union => anyhow::ensure!(
                bf.len() == 1,
                "ColumnAlignment::Union requires exactly one --batch-files file \
                 (one label per unified cell); got {}",
                bf.len()
            ),
        }
    }

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: data_files.to_vec(),
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        preload,
        feature_kind: Some(feature_kind),
        column_alignment,
        ..Default::default()
    })?;

    // Under Union with no explicit batch file, auto-batch by
    // modality-presence so multiome cohorts (matched / unimodal) align.
    let auto_modality_batch =
        matches!(column_alignment, ColumnAlignment::Union) && batch_files.is_none();
    UnifiedData::from_sparse_io(loaded.data, &loaded.batch, auto_modality_batch)
}
