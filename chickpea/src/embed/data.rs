//! Barcode-keyed loader for arbitrary multi-modal panels. Each input
//! file contributes its rows to a unified feature axis; cells (barcodes)
//! are unioned across files.

use crate::common::*;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
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
    pub file_triplet_ranges: Vec<std::ops::Range<usize>>,
    pub file_feature_ranges: Vec<std::ops::Range<u32>>,
    pub barcodes: Vec<Box<str>>,
}

impl UnifiedData {
    pub fn n_cells(&self) -> usize {
        self.barcodes.len()
    }

    pub fn n_features(&self) -> usize {
        self.feature_names.len()
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

    let mut feature_names: Vec<Box<str>> = Vec::new();
    let mut file_feature_ranges: Vec<std::ops::Range<u32>> = Vec::with_capacity(data_files.len());
    for feats in &per_file_features {
        let start = feature_names.len() as u32;
        feature_names.extend(feats.iter().cloned());
        let end = feature_names.len() as u32;
        file_feature_ranges.push(start..end);
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
    let mut file_triplet_ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(data_files.len());
    for (i, data) in per_file_data.iter().enumerate() {
        let start = triplets.len();
        let t = read_triplets(
            data,
            &per_file_local_to_global[i],
            file_feature_ranges[i].start,
        )?;
        triplets.extend(t);
        let end = triplets.len();
        file_triplet_ranges.push(start..end);
    }
    info!(
        "  edges per file: {} (total {})",
        file_triplet_ranges
            .iter()
            .map(|r| (r.end - r.start).to_string())
            .collect::<Vec<_>>()
            .join(" + "),
        triplets.len()
    );

    Ok(UnifiedData {
        triplets,
        feature_names,
        per_file_data,
        file_triplet_ranges,
        file_feature_ranges,
        barcodes,
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
