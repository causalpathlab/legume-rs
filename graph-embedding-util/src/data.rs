//! Barcode-keyed loader for arbitrary multi-modal panels. Each input
//! file contributes its rows to a unified feature axis; cells (barcodes)
//! are unioned across files. Per-cell batch labels (from `batch_files`,
//! `@batch` tags, or file names) are deduped into a global namespace so
//! downstream samplers can restrict negatives to within-batch.

use crate::progress::new_progress_bar;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use auxiliary_data::feature_names::FeatureNameKind;
use data_beans::sparse_io_vector::{ColumnAlignment, SparseIoVec};
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
    //////////////////
    // cell indices //
    //////////////////
    pub barcodes: Vec<Box<str>>,

    /// Global cell id → global batch id.
    pub batch_membership: Vec<u32>,

    /// Global batch labels in id order.
    pub batch_names: Vec<Box<str>>,

    /// Global cell id → global condition id, from optional per-cell
    /// condition labels. Defaults to a copy of `batch_membership` when no
    /// separate condition input is given. Retained loader state — the
    /// per-condition feature gate that consumed it has been removed, so no
    /// model currently reads this.
    pub condition_membership: Vec<u32>,

    /// Global condition labels in id order. Defaults to `batch_names`.
    pub condition_names: Vec<Box<str>>,

    /// Per unified cell, a bitmask of which input files (modalities) the
    /// cell appears in: bit `b` set ⇔ the cell's barcode is present in
    /// backend `b`. Single-file / non-multiome loads set bit 0 for every
    /// cell. `count_ones() >= 2` marks a **matched** (multi-modal bridge)
    /// cell. Drives `--multiome` auto modality-batching and matched-cell
    /// up-weighting.
    pub cell_modality: Vec<u32>,

    ///////////////////
    // feature index //
    ///////////////////
    pub feature_names: Vec<Box<str>>,
    /// For each compact feature id `i ∈ 0..n_features()`, the row index into
    /// `backend`. Identity (`0..n_features()`) when no HVG was applied; the
    /// HVG selection indices after `subset_features`. Used when an upstream
    /// pass (e.g. NB-Fisher) operates on the full backend and the result needs
    /// to be aligned to the compact axis.
    pub feature_to_backend_row: Vec<usize>,

    ///////////////
    // edge list //
    ///////////////
    /// Cell↔feature edges. Empty until `materialize_cell_triplets` is called
    /// (real-data path), or pre-built by `from_pseudobulks`.
    pub triplets: Vec<Triplet>,

    ////////////////////
    // sparse backend //
    ////////////////////
    /// The underlying sparse count matrix.  `Some` for real-data paths (set by
    /// `from_sparse_io`); `None` for synthetic pseudobulk `UnifiedData` (set by
    /// `from_pseudobulks`, which pre-builds `triplets` directly).
    pub backend: Option<SparseIoVec>,
}

impl UnifiedData {
    /// Borrow the sparse count backend. Panics when called on a synthetic
    /// pseudobulk `UnifiedData` (which has no backend).
    #[must_use]
    pub fn count_backend(&self) -> &SparseIoVec {
        self.backend
            .as_ref()
            .expect("count_backend called on synthetic pseudobulk UnifiedData (no backend)")
    }

    /// Mutably borrow the sparse count backend. Panics when called on a
    /// synthetic pseudobulk `UnifiedData` (which has no backend).
    pub fn count_backend_mut(&mut self) -> &mut SparseIoVec {
        self.backend
            .as_mut()
            .expect("count_backend_mut called on synthetic pseudobulk UnifiedData (no backend)")
    }
}

impl UnifiedData {
    #[must_use]
    pub fn n_cells(&self) -> usize {
        self.barcodes.len()
    }

    #[must_use]
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    #[must_use]
    pub fn n_batches(&self) -> usize {
        self.batch_names.len()
    }

    #[must_use]
    pub fn n_conditions(&self) -> usize {
        self.condition_names.len()
    }

    /// Per-cell batch labels (`batch_names[batch_membership[c]]`), in cell-id
    /// order. Length == `n_cells()`. The string form the projection /
    /// multilevel collapse consume for within-batch negatives.
    #[must_use]
    pub fn batch_labels(&self) -> Vec<Box<str>> {
        self.batch_membership
            .iter()
            .map(|&b| self.batch_names[b as usize].clone())
            .collect()
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
        // Pseudobulks are condition-corrected aggregates with no single
        // condition; collapse to one synthetic condition.
        let condition_membership = vec![0u32; n_pb];
        let condition_names: Vec<Box<str>> = vec!["all".to_string().into_boxed_str()];

        Ok(UnifiedData {
            barcodes,
            batch_membership,
            batch_names,
            condition_membership,
            condition_names,
            // Synthetic pseudobulk "cells" are single-modality.
            cell_modality: vec![1u32; n_pb],
            feature_names,
            feature_to_backend_row,
            triplets,
            backend: None,
        })
    }

    /// Restrict to a subset of features (by current global feature id).
    /// Drops triplets whose feature isn't in `selected_indices`, remaps
    /// remaining features to compact 0..k indices, updates
    /// `feature_names`, and composes `feature_to_backend_row` so the
    /// new compact index `i` still maps to the right row in `backend`.
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

        // The edge-count half is only meaningful when triplets are already
        // materialized; in the streaming path they're empty here (built later
        // from the subsetted axis), so don't report a misleading "0 → 0".
        if n_before > 0 {
            log::info!(
                "Feature subset: {} → {} features ({} → {} edges retained)",
                n_old,
                new_feature_names.len(),
                n_before,
                self.triplets.len()
            );
        } else {
            log::info!(
                "Feature subset: {} → {} features (triplets built later from this axis)",
                n_old,
                new_feature_names.len()
            );
        }
        self.feature_names = new_feature_names;
        self.feature_to_backend_row = new_feature_to_backend_row;
    }

    /// Restrict to a subset of cells (by current global cell id).
    /// Drops triplets whose cell isn't in `selected_indices`, remaps
    /// remaining cells to compact 0..k indices, and updates all
    /// per-cell metadata vecs (`barcodes`, `batch_membership`,
    /// `condition_membership`, `cell_modality`).
    ///
    /// `backend` is **not** touched — the `SparseIoVec` retains the
    /// original column layout.
    pub fn subset_cells(&mut self, selected_indices: &[usize]) {
        if selected_indices.len() == self.n_cells() {
            return; // no-op
        }
        let n_old = self.n_cells();
        let mut old_to_new: Vec<i32> = vec![-1; n_old];
        for (new_i, &old_i) in selected_indices.iter().enumerate() {
            debug_assert!(old_i < n_old, "selected index {old_i} out of range");
            old_to_new[old_i] = new_i as i32;
        }

        let new_barcodes: Vec<Box<str>> = selected_indices
            .iter()
            .map(|&i| self.barcodes[i].clone())
            .collect();
        let new_batch_membership: Vec<u32> = selected_indices
            .iter()
            .map(|&i| self.batch_membership[i])
            .collect();
        let new_condition_membership: Vec<u32> = selected_indices
            .iter()
            .map(|&i| self.condition_membership[i])
            .collect();
        let new_cell_modality: Vec<u32> = selected_indices
            .iter()
            .map(|&i| self.cell_modality[i])
            .collect();

        let n_before = self.triplets.len();
        self.triplets.retain_mut(|t| {
            let new_id = old_to_new[t.cell as usize];
            if new_id >= 0 {
                t.cell = new_id as u32;
                true
            } else {
                false
            }
        });

        if n_before > 0 {
            info!(
                "Cell subset: {} → {} cells ({} → {} edges retained)",
                n_old,
                new_barcodes.len(),
                n_before,
                self.triplets.len()
            );
        } else {
            info!(
                "Cell subset: {} → {} cells (triplets built later from this axis)",
                n_old,
                new_barcodes.len()
            );
        }
        self.barcodes = new_barcodes;
        self.batch_membership = new_batch_membership;
        self.condition_membership = new_condition_membership;
        self.cell_modality = new_cell_modality;
    }

    /// Build the cell↔feature edge list (`triplets`) from `backend`
    /// on the *current* unified feature axis. Deferred from construction so
    /// this training-only structure (~12 B/edge) doesn't coexist with the
    /// collapse's sufficient-stats during the peak collapse phase — call it
    /// after the collapse, before training. Honors any `subset_features`
    /// already applied: only kept backend rows are emitted, remapped to
    /// compact unified ids via `feature_to_backend_row`. Idempotent-ish: it
    /// replaces whatever `triplets` currently holds.
    pub fn materialize_cell_triplets(&mut self) -> anyhow::Result<()> {
        let data = self.count_backend();
        let n_cells = data.num_columns();
        let backend_rows = data.num_rows();

        // backend compact row → unified compact id (u32::MAX ⇒ dropped by
        // an HVG/frozen subset). Identity when no subset was applied.
        let mut backend_to_unified = vec![u32::MAX; backend_rows];
        for (uid, &brow) in self.feature_to_backend_row.iter().enumerate() {
            if brow < backend_rows {
                backend_to_unified[brow] = uid as u32;
            }
        }

        info!("Materializing sparse triplets (unified feature space)...");
        let nnz_hint = data.num_non_zeros().unwrap_or(0);
        // Bound the per-slab read even when nnz is unavailable (nnz_hint==0),
        // otherwise the slab would clamp to the whole matrix.
        let chunk_cols = if nnz_hint > 0 {
            let avg_per_col = (nnz_hint / n_cells.max(1)).max(1);
            (8_000_000 / avg_per_col).clamp(1, n_cells.max(1))
        } else {
            (1usize << 14).min(n_cells.max(1))
        };
        let pb_bar = new_progress_bar(nnz_hint.max(1) as u64);
        pb_bar.set_message("triplets");
        let mut triplets: Vec<Triplet> = Vec::with_capacity(nnz_hint);
        let mut since_tick = 0u64;
        data.for_each_triplet(0..n_cells, chunk_cols, |row, col, v| {
            if v != 0.0 {
                let uid = backend_to_unified[row as usize];
                if uid != u32::MAX {
                    triplets.push(Triplet {
                        cell: col as u32,
                        feature: uid,
                        count: v,
                    });
                    since_tick += 1;
                    if since_tick == 1 << 20 {
                        pb_bar.inc(since_tick);
                        since_tick = 0;
                    }
                }
            }
        })?;
        pb_bar.inc(since_tick);
        pb_bar.finish_and_clear();
        info!("  edges: {}", triplets.len());
        self.triplets = triplets;
        Ok(())
    }

    /// Build a `UnifiedData` from a single in-memory `SparseIoVec` plus
    /// per-cell batch labels. Internal helper for [`load_unified_data`].
    pub(crate) fn from_sparse_io(
        data: SparseIoVec,
        batch_labels: &[Box<str>],
        condition_labels: Option<&[Box<str>]>,
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
        let (batch_membership, batch_names) = dedup_labels(batch_labels);

        info!(
            "Unified barcodes: {} total (single in-memory data set)",
            barcodes.len()
        );
        info!(
            "Unified batches: {} ({})",
            batch_names.len(),
            truncate_names(&batch_names, 6)
        );

        // Condition axis: defaults to a copy of the batch axis when no
        // separate per-cell condition labels are supplied. When given,
        // dedupe into a global namespace the same first-seen way.
        let (condition_membership, condition_names): (Vec<u32>, Vec<Box<str>>) =
            match condition_labels {
                None => (batch_membership.clone(), batch_names.clone()),
                Some(labels) => {
                    anyhow::ensure!(
                        labels.len() == barcodes.len(),
                        "condition labels {} != cells {}",
                        labels.len(),
                        barcodes.len()
                    );
                    dedup_labels(labels)
                }
            };
        // Only log conditions when they differ from the batch axis.
        if condition_names != batch_names {
            info!(
                "Unified conditions: {} ({})",
                condition_names.len(),
                truncate_names(&condition_names, 6)
            );
        }

        // Triplets (the cell↔feature edge list for training) are NOT built
        // here. They're a training-only structure the collapse never reads, so
        // materializing them at load would needlessly stack ~12 B/edge on top
        // of the collapse's sufficient-stats during the (peak) collapse phase.
        // `materialize_cell_triplets` builds them after the collapse instead,
        // on the final (possibly HVG/frozen-subset) feature axis.
        let n_features = feature_names.len();
        Ok(UnifiedData {
            barcodes,
            batch_membership,
            batch_names,
            condition_membership,
            condition_names,
            cell_modality,
            feature_names,
            feature_to_backend_row: (0..n_features).collect(),
            triplets: Vec::new(),
            backend: Some(data),
        })
    }
}

/// Format the first `max_show` names from a slice, appending "… and N more" when
/// the slice is longer, so INFO log lines stay readable regardless of set size.
fn truncate_names(names: &[Box<str>], max_show: usize) -> String {
    if names.len() <= max_show {
        names
            .iter()
            .map(std::convert::AsRef::as_ref)
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        let shown: Vec<&str> = names[..max_show]
            .iter()
            .map(std::convert::AsRef::as_ref)
            .collect();
        format!("{} … and {} more", shown.join(", "), names.len() - max_show)
    }
}

/// Map per-cell string labels to compact `u32` ids in first-seen order,
/// returning `(membership, names)` where `names[membership[i]] == labels[i]`.
/// Shared by the batch and condition axes.
fn dedup_labels(labels: &[Box<str>]) -> (Vec<u32>, Vec<Box<str>>) {
    let mut to_global: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut names: Vec<Box<str>> = Vec::new();
    let membership: Vec<u32> = labels
        .iter()
        .map(|label| {
            *to_global.entry(label.clone()).or_insert_with(|| {
                let g = names.len() as u32;
                names.push(label.clone());
                g
            })
        })
        .collect();
    (membership, names)
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

/// Inputs for [`load_unified_data`]. Build with `..Default::default()` and
/// set only the fields a given flow needs — most callers leave the suffix /
/// condition axes at their `None` defaults.
#[derive(Default)]
pub struct LoadUnifiedArgs {
    /// Sparse data files (prefixes) to load, in order.
    pub data_files: Vec<Box<str>>,
    /// Optional per-cell batch labels. Under [`ColumnAlignment::Union`]
    /// exactly one file (one label per unified cell); under `Disjoint` one
    /// file per data file. `None` → infer from `@batch` tags / file names.
    pub batch_files: Option<Vec<Box<str>>>,
    /// Optional per-cell condition labels (same shape rules as `batch_files`).
    /// `None` → condition = batch.
    pub condition_files: Option<Vec<Box<str>>>,
    /// Cross-file row-name canonicalization. `None` = auto-detect.
    pub feature_kind: Option<FeatureNameKind>,
    /// Preload every sparse column into memory before any pass over cells.
    pub preload: bool,
    /// How to align cells (barcodes) across files.
    pub column_alignment: ColumnAlignment,
    /// Per-file row (feature) modality suffix — see
    /// [`ReadSharedRowsArgs::per_file_feature_suffix`].
    pub per_file_feature_suffix: Option<Vec<Box<str>>>,
    /// Per-file barcode (sample) suffix — see
    /// [`ReadSharedRowsArgs::per_file_barcode_suffix`].
    pub per_file_barcode_suffix: Option<Vec<Option<Box<str>>>>,
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
pub fn load_unified_data(args: LoadUnifiedArgs) -> anyhow::Result<UnifiedData> {
    use matrix_util::common_io::read_lines;
    let LoadUnifiedArgs {
        data_files,
        batch_files,
        condition_files,
        feature_kind,
        preload,
        column_alignment,
        per_file_feature_suffix,
        per_file_barcode_suffix,
    } = args;
    if let Some(bf) = batch_files.as_deref() {
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

    let batch_files_present = batch_files.is_some();
    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files,
        batch_files,
        preload,
        feature_kind,
        column_alignment,
        per_file_feature_suffix,
        per_file_barcode_suffix,
        ..Default::default()
    })?;

    // Optional per-cell condition labels. Read line-per-cell, concatenated
    // in file order (matches the unified column order: Disjoint stacks
    // files in order; Union takes one file with one label per unified
    // cell). Validated against the unified cell count below.
    let condition_labels: Option<Vec<Box<str>>> = match condition_files {
        None => None,
        Some(cf) => {
            let n_cells = loaded.data.num_columns();
            if matches!(column_alignment, ColumnAlignment::Union) {
                anyhow::ensure!(
                    cf.len() == 1,
                    "ColumnAlignment::Union requires exactly one --condition-files file \
                     (one label per unified cell); got {}",
                    cf.len()
                );
            }
            let mut labels: Vec<Box<str>> = Vec::with_capacity(n_cells);
            for f in &cf {
                info!("Reading condition file: {f}");
                labels.extend(read_lines(f)?);
            }
            anyhow::ensure!(
                labels.len() == n_cells,
                "condition files have {} lines but data has {} unified cells",
                labels.len(),
                n_cells
            );
            Some(labels)
        }
    };

    // Under Union with no explicit batch file AND no embedded @batch tags,
    // auto-batch by modality-presence so true multiome cohorts (matched +
    // unimodal cells) get corrected for modality drop-out.
    //
    // Do NOT use modality-presence when the loader resolved real per-cell
    // @batch sample IDs: that would discard sample structure and collapse
    // all cells in the same input file into one batch (e.g. every cell in
    // rep1_wt_genes.zarr.zip → "m0", every cell in rep1_wt_m6a.zarr.zip
    // → "m6"), even though they share the same biological samples.
    let batch_tags_are_trivial = loaded.batch.iter().all(|b| b.as_ref() == "all");
    let auto_modality_batch = matches!(column_alignment, ColumnAlignment::Union)
        && !batch_files_present
        && batch_tags_are_trivial;
    UnifiedData::from_sparse_io(
        loaded.data,
        &loaded.batch,
        condition_labels.as_deref(),
        auto_modality_batch,
    )
}

/// Validate multiome group structure against the post-load unified barcodes
/// and `cell_modality` bitmasks (bit `b` set ⇔ barcode present in file `b` of
/// the flattened file list).
///
/// Only **cross-group disjointness** is enforced: a barcode must not appear in
/// more than one group. Under `ColumnAlignment::Union` two groups sharing a raw
/// barcode would silently fold into one cell (summing counts from different
/// donors), so this guards against that mistake — fail fast with the offending
/// barcode. Partial overlap *within* a group (patchy multiome — RNA-only /
/// ATAC-only cells) is allowed: bge unions all cells regardless, and the
/// modality-presence auto-batch already labels each group.
///
/// No-op for a single group (nothing to be disjoint from).
///
/// `group_sizes` is the file count per group, in flattened file order (so
/// group 0 owns backend bits `0..group_sizes[0]`, group 1 the next block, …).
pub fn validate_multiome_groups(
    group_sizes: &[usize],
    barcodes: &[Box<str>],
    cell_modality: &[u32],
) -> anyhow::Result<()> {
    if group_sizes.len() <= 1 {
        return Ok(());
    }

    // Per-group bitmask covering that group's flattened file indices.
    let mut offset = 0usize;
    let group_masks: Vec<u32> = group_sizes
        .iter()
        .map(|&n| {
            let mask = (offset..offset + n).fold(0u32, |acc, i| acc | (1u32 << i));
            offset += n;
            mask
        })
        .collect();

    // Cross-group: no barcode's modality bits may span more than one group.
    for (cell_idx, &bits) in cell_modality.iter().enumerate() {
        let n_groups_hit = group_masks.iter().filter(|&&m| bits & m != 0).count();
        if n_groups_hit > 1 {
            anyhow::bail!(
                "multiome: barcode {:?} appears in multiple groups (modality bits \
                 {:#b}). Barcodes must be unique across groups — under Union loading \
                 a shared barcode merges cells from different samples into one.",
                barcodes[cell_idx],
                bits,
            );
        }
    }

    Ok(())
}
