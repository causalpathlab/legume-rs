//! Barcode-keyed loader for arbitrary multi-modal panels. Each input
//! file contributes its rows to a unified feature axis; cells (barcodes)
//! are unioned across files. Per-cell batch labels (from `batch_files`,
//! `@batch` tags, or file names) are deduped into a global namespace so
//! downstream samplers can restrict negatives to within-batch.

use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use data_beans::sparse_io_vector::SparseIoVec;
use log::{info, warn};
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

    /// Build a `UnifiedData` from a single in-memory `SparseIoVec` plus
    /// per-cell batch labels. Used by callers (e.g. `pinto gbe`) whose
    /// own SRT loaders already produced a concatenated `SparseIoVec` and
    /// shouldn't re-route through [`load_unified_data`].
    pub fn from_sparse_io(data: SparseIoVec, batch_labels: &[Box<str>]) -> anyhow::Result<Self> {
        let feature_names = data.row_names()?;
        let barcodes = data.column_names()?;
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
        let (_, raw) = data.columns_triplets(0..data.num_columns())?;
        let triplets: Vec<Triplet> = raw
            .into_iter()
            .filter(|&(_, _, v)| v != 0.0)
            .map(|(row, col, v)| Triplet {
                cell: col as u32,
                feature: row as u32,
                count: v,
            })
            .collect();
        info!("  edges: {}", triplets.len());

        Ok(UnifiedData {
            triplets,
            feature_names,
            per_file_data: vec![data],
            barcodes,
            batch_membership,
            batch_names,
        })
    }
}

pub fn load_unified_data(
    data_files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
) -> anyhow::Result<UnifiedData> {
    if let Some(bf) = batch_files {
        if bf.len() != data_files.len() {
            anyhow::bail!(
                "batch_files length {} != data_files length {}",
                bf.len(),
                data_files.len()
            );
        }
    }

    let mut per_file_data: Vec<SparseIoVec> = Vec::with_capacity(data_files.len());
    let mut per_file_features: Vec<Vec<Box<str>>> = Vec::with_capacity(data_files.len());
    let mut per_file_barcodes: Vec<Vec<Box<str>>> = Vec::with_capacity(data_files.len());
    let mut per_file_batch_labels: Vec<Vec<Box<str>>> = Vec::with_capacity(data_files.len());

    for (i, file) in data_files.iter().enumerate() {
        info!("Loading file {}/{}: {}", i + 1, data_files.len(), file);
        let bf = batch_files.map(|b| vec![b[i].clone()]);
        let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
            data_files: vec![file.clone()],
            batch_files: bf,
            preload: false,
        })?;
        per_file_features.push(loaded.data.row_names()?);
        per_file_barcodes.push(loaded.data.column_names()?);
        per_file_batch_labels.push(loaded.batch);
        per_file_data.push(loaded.data);
    }

    let total_bc_hint: usize = per_file_barcodes.iter().map(|v| v.len()).sum();
    let mut barcode_to_global: FxHashMap<Box<str>, u32> =
        FxHashMap::with_capacity_and_hasher(total_bc_hint, Default::default());
    let mut barcodes: Vec<Box<str>> = Vec::with_capacity(total_bc_hint);

    let mut per_file_local_to_global: Vec<Vec<u32>> = Vec::with_capacity(data_files.len());
    for bcs in &per_file_barcodes {
        let mut map = Vec::with_capacity(bcs.len());
        for bc in bcs.iter() {
            let g = if let Some(&g) = barcode_to_global.get(bc) {
                g
            } else {
                let g = barcodes.len() as u32;
                barcode_to_global.insert(bc.clone(), g);
                barcodes.push(bc.clone());
                g
            };
            map.push(g);
        }
        per_file_local_to_global.push(map);
    }

    info!(
        "Unified barcodes: {} total across {} file(s)",
        barcodes.len(),
        data_files.len()
    );

    // Dedupe batch labels into a global namespace and build cell → batch.
    // Resolve per-cell collisions across files (same barcode in two files
    // with conflicting batch labels) by keeping the first label seen and
    // warning — concatenating with `|` would create a fake "joint" batch
    // that the sampler can't actually populate consistently.
    let mut batch_to_global: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut batch_names: Vec<Box<str>> = Vec::new();
    let mut batch_membership: Vec<i32> = vec![-1; barcodes.len()];
    let mut conflicts = 0usize;
    for (file_idx, labels) in per_file_batch_labels.iter().enumerate() {
        let local_to_global = &per_file_local_to_global[file_idx];
        if labels.len() != local_to_global.len() {
            anyhow::bail!(
                "file {}: batch labels {} != cells {}",
                file_idx,
                labels.len(),
                local_to_global.len()
            );
        }
        for (local_cell, label) in labels.iter().enumerate() {
            let g_batch = if let Some(&g) = batch_to_global.get(label) {
                g
            } else {
                let g = batch_names.len() as u32;
                batch_to_global.insert(label.clone(), g);
                batch_names.push(label.clone());
                g
            };
            let g_cell = local_to_global[local_cell] as usize;
            match batch_membership[g_cell] {
                -1 => batch_membership[g_cell] = g_batch as i32,
                prev if prev as u32 != g_batch => conflicts += 1,
                _ => {}
            }
        }
    }
    if conflicts > 0 {
        warn!(
            "{} cell(s) appear in multiple files with conflicting batch labels — \
             keeping the first label seen",
            conflicts
        );
    }
    let batch_membership: Vec<u32> = batch_membership
        .into_iter()
        .map(|v| {
            debug_assert!(v >= 0, "every cell should have a batch label by now");
            v as u32
        })
        .collect();

    info!(
        "Unified batches: {} ({})",
        batch_names.len(),
        batch_names
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut feature_names: Vec<Box<str>> = Vec::new();
    let mut file_feature_offsets: Vec<u32> = Vec::with_capacity(data_files.len());
    for feats in &per_file_features {
        file_feature_offsets.push(feature_names.len() as u32);
        feature_names.extend(feats.iter().cloned());
    }

    info!(
        "Unified features: {} total ({})",
        feature_names.len(),
        per_file_features
            .iter()
            .map(|v| v.len().to_string())
            .collect::<Vec<_>>()
            .join(" + ")
    );

    info!("Materializing sparse triplets (unified feature space)...");
    let mut triplets: Vec<Triplet> = Vec::new();
    let mut per_file_edges: Vec<usize> = Vec::with_capacity(data_files.len());
    for (i, data) in per_file_data.iter().enumerate() {
        let start = triplets.len();
        let t = read_triplets(data, &per_file_local_to_global[i], file_feature_offsets[i])?;
        triplets.extend(t);
        per_file_edges.push(triplets.len() - start);
    }
    info!(
        "  edges per file: {} (total {})",
        per_file_edges
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(" + "),
        triplets.len()
    );

    Ok(UnifiedData {
        triplets,
        feature_names,
        per_file_data,
        barcodes,
        batch_membership,
        batch_names,
    })
}

fn read_triplets(
    data: &SparseIoVec,
    local_to_global: &[u32],
    feature_offset: u32,
) -> anyhow::Result<Vec<Triplet>> {
    let (_, raw) = data.columns_triplets(0..data.num_columns())?;
    Ok(raw
        .into_iter()
        .filter(|&(_, _, v)| v != 0.0)
        .map(|(row, col, v)| Triplet {
            cell: local_to_global[col as usize],
            feature: row as u32 + feature_offset,
            count: v,
        })
        .collect())
}
