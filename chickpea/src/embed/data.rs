//! Barcode-keyed loader for paired-or-unpaired multiome — produces a
//! single unified feature axis where genes and peaks live together.
//!
//! Both modalities are loaded independently, but the loader emits a
//! single `Vec<Triplet>` over a shared feature index space:
//! `[0, n_genes)` are RNA genes, `[n_genes, n_genes + n_peaks)` are
//! ATAC peaks. Cells (barcodes) are unified via union — paired
//! barcodes share a row.
//!
//! This unification reflects the design choice that the embedding
//! geometry should treat genes and peaks as members of the same
//! "feature" type, with a single shared embedding table `E_feat` and
//! a single (feature, cell) relation in the loss. Modality identity
//! is retained only as a per-feature tag for diagnostics and
//! NB-Fisher weighting.

use crate::common::*;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use rustc_hash::FxHashMap;

/// One non-zero count edge in the unified `(cell, feature)` space.
#[derive(Clone, Copy)]
pub struct Triplet {
    /// Global cell index in the unified barcode space.
    pub cell: u32,
    /// Unified feature index. Genes occupy `[0, n_genes)`, peaks occupy
    /// `[n_genes, n_genes + n_peaks)`.
    pub feature: u32,
    /// Count value.
    pub count: f32,
}

/// Per-feature modality tag for diagnostics + per-modality bookkeeping
/// (e.g., NB-Fisher applies to genes only; peaks get unit weight).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Gene,
    Peak,
}

/// Unified multiome dataset: single feature axis (genes ∪ peaks),
/// single cell axis (barcode-unioned), single triplet stream.
pub struct UnifiedData {
    /// Triplets in `(unified_cell, unified_feature, count)` form.
    pub triplets: Vec<Triplet>,
    /// Feature names in unified order: genes first, then peaks.
    pub feature_names: Vec<Box<str>>,
    /// Per-feature modality tag, parallel to `feature_names`.
    pub feature_modality: Vec<Modality>,
    /// Number of gene features (= offset where peak features begin).
    pub n_genes: usize,
    /// Number of peak features.
    pub n_peaks: usize,
    /// Original RNA `SparseIoVec` — kept for NB-Fisher fitting on RNA.
    /// (ATAC is not retained; per-peak NB-Fisher isn't currently used.)
    pub rna_data: SparseIoVec,
    /// Unified barcode list, indexed by global cell idx.
    pub barcodes: Vec<Box<str>>,
}

impl UnifiedData {
    /// Total unique barcodes across both modalities.
    pub fn n_cells(&self) -> usize {
        self.barcodes.len()
    }

    /// Total unified features (genes + peaks).
    pub fn n_features(&self) -> usize {
        self.n_genes + self.n_peaks
    }
}

/// Load RNA + ATAC, build unified barcode index, materialize sparse triplets.
///
/// Paired vs unpaired is handled identically — barcode is the primary
/// key, and any barcode appearing in both modalities shares one global
/// cell row.
pub fn load_unified_data(
    rna_files: &[Box<str>],
    atac_files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
) -> anyhow::Result<UnifiedData> {
    let n_rna_files = rna_files.len();
    let n_atac_files = atac_files.len();

    let (rna_batch_files, atac_batch_files) = match batch_files {
        Some(bf) => {
            let expected = n_rna_files + n_atac_files;
            if bf.len() != expected {
                anyhow::bail!(
                    "batch_files length {} != rna_files ({}) + atac_files ({})",
                    bf.len(),
                    n_rna_files,
                    n_atac_files
                );
            }
            (
                Some(bf[..n_rna_files].to_vec()),
                Some(bf[n_rna_files..].to_vec()),
            )
        }
        None => (None, None),
    };

    info!("Loading RNA data ({} file(s))...", n_rna_files);
    let rna_loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: rna_files.to_vec(),
        batch_files: rna_batch_files,
        preload: false,
    })?;

    info!("Loading ATAC data ({} file(s))...", n_atac_files);
    let atac_loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: atac_files.to_vec(),
        batch_files: atac_batch_files,
        preload: false,
    })?;

    let rna_barcodes = rna_loaded.data.column_names()?;
    let atac_barcodes = atac_loaded.data.column_names()?;
    let rna_features = rna_loaded.data.row_names()?;
    let atac_features = atac_loaded.data.row_names()?;

    // Build unified barcode index. RNA barcodes get global indices first
    // (in RNA order); ATAC-only barcodes are appended.
    let mut barcode_to_global: FxHashMap<Box<str>, u32> = FxHashMap::with_capacity_and_hasher(
        rna_barcodes.len() + atac_barcodes.len(),
        Default::default(),
    );
    let mut barcodes: Vec<Box<str>> = Vec::with_capacity(rna_barcodes.len() + atac_barcodes.len());

    let mut rna_local_to_global = Vec::with_capacity(rna_barcodes.len());
    for bc in rna_barcodes.iter() {
        let g = barcodes.len() as u32;
        barcode_to_global.insert(bc.clone(), g);
        barcodes.push(bc.clone());
        rna_local_to_global.push(g);
    }

    let mut atac_local_to_global = Vec::with_capacity(atac_barcodes.len());
    let mut n_paired = 0usize;
    for bc in atac_barcodes.iter() {
        if let Some(&g) = barcode_to_global.get(bc) {
            atac_local_to_global.push(g);
            n_paired += 1;
        } else {
            let g = barcodes.len() as u32;
            barcode_to_global.insert(bc.clone(), g);
            barcodes.push(bc.clone());
            atac_local_to_global.push(g);
        }
    }

    info!(
        "Unified barcodes: {} total ({} RNA-only, {} ATAC-only, {} paired)",
        barcodes.len(),
        rna_barcodes.len() - n_paired,
        atac_barcodes.len() - n_paired,
        n_paired,
    );
    info!(
        "Loaded RNA: {} genes x {} cells, ATAC: {} peaks x {} cells",
        rna_features.len(),
        rna_barcodes.len(),
        atac_features.len(),
        atac_barcodes.len(),
    );

    let n_genes = rna_features.len();
    let n_peaks = atac_features.len();

    info!("Materializing sparse triplets (unified gene+peak space)...");
    let mut triplets = read_triplets(&rna_loaded.data, &rna_local_to_global, 0)?;
    let n_rna_edges = triplets.len();
    let atac_triplets = read_triplets(&atac_loaded.data, &atac_local_to_global, n_genes as u32)?;
    let n_atac_edges = atac_triplets.len();
    triplets.extend(atac_triplets);
    info!(
        "  {} RNA edges + {} ATAC edges = {} unified",
        n_rna_edges,
        n_atac_edges,
        triplets.len()
    );

    // Unified feature names + modality tags.
    let mut feature_names = Vec::with_capacity(n_genes + n_peaks);
    let mut feature_modality = Vec::with_capacity(n_genes + n_peaks);
    feature_names.extend(rna_features.iter().cloned());
    feature_modality.extend(std::iter::repeat_n(Modality::Gene, n_genes));
    feature_names.extend(atac_features.iter().cloned());
    feature_modality.extend(std::iter::repeat_n(Modality::Peak, n_peaks));

    Ok(UnifiedData {
        triplets,
        feature_names,
        feature_modality,
        n_genes,
        n_peaks,
        rna_data: rna_loaded.data,
        barcodes,
    })
}

/// Stream the entire `SparseIoVec` once and emit `Triplet`s with cell
/// indices in unified cell coordinates and feature indices offset by
/// `feature_offset` (0 for RNA, `n_genes` for ATAC).
fn read_triplets(
    data: &SparseIoVec,
    local_to_global: &[u32],
    feature_offset: u32,
) -> anyhow::Result<Vec<Triplet>> {
    let n_cols = data.num_columns();
    // Read in moderate-size column blocks to balance memory vs throughput.
    let block_size: usize = 4096;
    let mut out: Vec<Triplet> = Vec::new();

    let mut start = 0usize;
    while start < n_cols {
        let end = (start + block_size).min(n_cols);
        let csc = data.read_columns_csc(start..end)?;
        let col_offsets = csc.col_offsets();
        let row_indices = csc.row_indices();
        let values = csc.values();
        for c_local in 0..(end - start) {
            let glob = local_to_global[start + c_local];
            let lo = col_offsets[c_local];
            let hi = col_offsets[c_local + 1];
            for k in lo..hi {
                let v = values[k];
                if v == 0.0 {
                    continue;
                }
                out.push(Triplet {
                    cell: glob,
                    feature: row_indices[k] as u32 + feature_offset,
                    count: v,
                });
            }
        }
        start = end;
    }
    Ok(out)
}
