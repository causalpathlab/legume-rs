//! Barcode-keyed loader for arbitrary multi-modal panels. Each input
//! file contributes its rows to a unified feature axis; cells (barcodes)
//! are unioned across files. Per-cell batch labels (from `batch_files`,
//! `@batch` tags, or file names) are deduped into a global namespace so
//! downstream samplers can restrict negatives to within-batch.

use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use auxiliary_data::feature_names::FeatureNameKind;
use data_beans::sparse_io_vector::SparseIoVec;
use log::info;
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
/// is out of scope for this loader; build a `UnifiedData` by hand via
/// [`UnifiedData::from_sparse_io`] (or a future helper) for that case.
pub fn load_unified_data(
    data_files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
    feature_kind: FeatureNameKind,
    preload: bool,
) -> anyhow::Result<UnifiedData> {
    if let Some(bf) = batch_files {
        anyhow::ensure!(
            bf.len() == data_files.len(),
            "batch_files length {} != data_files length {}",
            bf.len(),
            data_files.len()
        );
    }

    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: data_files.to_vec(),
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        preload,
        feature_kind,
    })?;

    UnifiedData::from_sparse_io(loaded.data, &loaded.batch)
}
