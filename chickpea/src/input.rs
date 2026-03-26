use crate::common::*;
use data_beans::convert::try_open_or_convert;
use rustc_hash::FxHashMap as HashMap;

pub use genomic_data::coordinates::{
    find_cis_peaks, load_gene_tss, parse_peak_coordinates, GeneTss, PeakCoord,
};

/// Paired RNA + ATAC data with matched barcodes.
pub struct DualModalityData {
    pub rna: SparseIoVec,
    pub atac: SparseIoVec,
    /// Indices into RNA columns for shared cells (sorted by RNA index).
    pub shared_rna_idx: Vec<usize>,
    /// Indices into ATAC columns for shared cells (same order as shared_rna_idx).
    pub shared_atac_idx: Vec<usize>,
    pub gene_names: Vec<Box<str>>,
    pub peak_names: Vec<Box<str>>,
}

/// Load two sets of sparse backends and identify shared barcodes.
pub fn load_dual_modality(
    rna_files: &[Box<str>],
    atac_files: &[Box<str>],
    preload: bool,
) -> anyhow::Result<DualModalityData> {
    let mut rna = SparseIoVec::new();
    for f in rna_files {
        info!("Loading RNA: {}", f);
        let mut data = try_open_or_convert(f)?;
        if preload {
            data.preload_columns()?;
        }
        rna.push(Arc::from(data), None)?;
    }

    let mut atac = SparseIoVec::new();
    for f in atac_files {
        info!("Loading ATAC: {}", f);
        let mut data = try_open_or_convert(f)?;
        if preload {
            data.preload_columns()?;
        }
        atac.push(Arc::from(data), None)?;
    }

    let rna_cols = rna.column_names()?;
    let atac_cols = atac.column_names()?;

    // Build ATAC barcode → index map
    let atac_map: HashMap<&str, usize> = atac_cols
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), i))
        .collect();

    // Find shared barcodes in RNA order
    let mut shared_pairs: Vec<(usize, usize)> = Vec::new();
    for (rna_idx, name) in rna_cols.iter().enumerate() {
        if let Some(&atac_idx) = atac_map.get(name.as_ref()) {
            shared_pairs.push((rna_idx, atac_idx));
        }
    }

    let n_shared = shared_pairs.len();
    anyhow::ensure!(
        n_shared > 0,
        "No shared barcodes between RNA ({} cells) and ATAC ({} cells)",
        rna_cols.len(),
        atac_cols.len()
    );

    info!(
        "Barcode matching: RNA {} cells, ATAC {} cells, shared {} cells",
        rna_cols.len(),
        atac_cols.len(),
        n_shared
    );

    let shared_rna_idx: Vec<usize> = shared_pairs.iter().map(|&(r, _)| r).collect();
    let shared_atac_idx: Vec<usize> = shared_pairs.iter().map(|&(_, a)| a).collect();

    let gene_names = rna.row_names()?;
    let peak_names = atac.row_names()?;

    info!(
        "Loaded: {} genes × {} shared cells (RNA), {} peaks × {} shared cells (ATAC)",
        gene_names.len(),
        n_shared,
        peak_names.len(),
        n_shared
    );

    Ok(DualModalityData {
        rna,
        atac,
        shared_rna_idx,
        shared_atac_idx,
        gene_names,
        peak_names,
    })
}

/// Build pseudobulk matrices from sparse data by grouping cells.
///
/// Given cell-to-group assignments, sums columns within each group
/// to produce a [D × num_groups] matrix. Reads in blocks for efficiency.
pub fn build_pseudobulk_from_groups(
    data: &SparseIoVec,
    cell_indices: &[usize],
    cell_to_group: &[usize],
    num_groups: usize,
    block_size: usize,
) -> anyhow::Result<Mat> {
    let d = data.num_rows();
    let mut pb = Mat::zeros(d, num_groups);

    // Process in blocks
    for chunk_start in (0..cell_indices.len()).step_by(block_size) {
        let chunk_end = (chunk_start + block_size).min(cell_indices.len());
        let cols = cell_indices[chunk_start..chunk_end].iter().copied();
        let block = data.read_columns_dmatrix(cols)?; // [D × chunk_size]

        for (local_j, global_j) in (chunk_start..chunk_end).enumerate() {
            let g = cell_to_group[global_j];
            for i in 0..d {
                pb[(i, g)] += block[(i, local_j)];
            }
        }
    }

    Ok(pb)
}
