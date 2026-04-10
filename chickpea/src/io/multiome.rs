use data_beans::hdf5_io::read_hdf5_strings;
use data_beans::hdf5_io::resolve_backend_file;
use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io::SparseIo;
use data_beans::sparse_io::SparseIoBackend;
use data_beans::sparse_util::*;
use log::info;
use matrix_util::common_io::remove_file;

pub struct MultiomeBackends {
    pub rna: Box<dyn SparseIo<IndexIter = Vec<usize>>>,
    pub atac: Box<dyn SparseIo<IndexIter = Vec<usize>>>,
}

struct ModalitySplit {
    triplets: Vec<(u64, u64, f32)>,
    row_names: Vec<Box<str>>,
    n_features: usize,
}

/// Read a 10x Multiome HDF5 and split into separate RNA and ATAC backends.
///
/// The HDF5 file contains both modalities distinguished by `features/feature_type`:
///   - "Gene Expression" → RNA backend at `{out_prefix}.rna.{ext}`
///   - "Peaks" → ATAC backend at `{out_prefix}.atac.{ext}`
///
/// Follows the same field layout as `data_beans::convert::convert_h5_to_backend()`.
pub fn convert_multiome_h5(
    h5_file: &str,
    out_prefix: &str,
    backend: &SparseIoBackend,
) -> anyhow::Result<MultiomeBackends> {
    let file = hdf5::File::open(h5_file)?;
    info!("Opened multiome data file: {}", h5_file);

    let root_group_name = "matrix";
    let root = file.group(root_group_name).map_err(|_| {
        anyhow::anyhow!(
            "Unable to find root group '{}' in {}",
            root_group_name,
            h5_file,
        )
    })?;

    // Read CSC arrays and convert to COO triplets
    let CooTripletsShape { triplets, shape } = {
        let values = root
            .dataset("data")
            .map_err(|_| anyhow::anyhow!("missing 'data' dataset"))?
            .read_1d::<f32>()?
            .to_vec();
        let indices = root
            .dataset("indices")
            .map_err(|_| anyhow::anyhow!("missing 'indices' dataset"))?
            .read_1d::<u64>()?
            .to_vec();
        let indptr = root
            .dataset("indptr")
            .map_err(|_| anyhow::anyhow!("missing 'indptr' dataset"))?
            .read_1d::<u64>()?
            .to_vec();

        ValuesIndicesPointers {
            values: &values,
            indices: &indices,
            indptr: &indptr,
        }
        .to_coo(IndexPointerType::Column)?
    };

    let TripletsShape { nrows, ncols, nnz } = shape;
    info!("Read {} non-zero elements in {} x {}", nnz, nrows, ncols);

    // Row IDs
    let mut row_ids: Vec<Box<str>> = match root.dataset("features/id") {
        Ok(ds) => read_hdf5_strings(ds)?,
        _ => {
            info!("row (feature) IDs not found");
            (0..nrows).map(|x| x.to_string().into_boxed_str()).collect()
        }
    };

    // Row names
    let mut row_names: Vec<Box<str>> = match root.dataset("features/name") {
        Ok(ds) => read_hdf5_strings(ds)?,
        _ => {
            info!("row (feature) names not found");
            vec![Box::from(""); nrows]
        }
    };

    if nrows < row_ids.len() {
        row_ids.truncate(nrows);
    }
    if nrows < row_names.len() {
        row_names.truncate(nrows);
    }
    assert_eq!(nrows, row_ids.len());
    assert_eq!(nrows, row_names.len());

    // Composite row names: id_name
    let row_ids: Vec<Box<str>> = row_ids
        .into_iter()
        .zip(row_names)
        .map(|(id, name)| {
            if !name.is_empty() {
                format!("{}_{}", id, name).into_boxed_str()
            } else {
                id
            }
        })
        .collect();

    // Feature types
    let mut row_types: Vec<Box<str>> = match root.dataset("features/feature_type") {
        Ok(ds) => read_hdf5_strings(ds)?,
        _ => {
            info!("feature_type not found, assuming all Gene Expression");
            vec!["Gene Expression".to_string().into_boxed_str(); nrows]
        }
    };
    if nrows < row_types.len() {
        row_types.truncate(nrows);
    }
    assert_eq!(nrows, row_types.len());

    // Column names (barcodes, shared across modalities)
    let mut column_names: Vec<Box<str>> = match root.dataset("barcodes") {
        Ok(ds) => read_hdf5_strings(ds)?,
        _ => {
            info!("column (barcode) names not found");
            (0..ncols).map(|x| x.to_string().into_boxed_str()).collect()
        }
    };
    if ncols < column_names.len() {
        column_names.truncate(ncols);
    }
    assert_eq!(ncols, column_names.len());

    // Split triplets by feature type
    let rna_split = split_triplets(&triplets, &row_ids, &row_types, "gene expression");
    let atac_split = split_triplets(&triplets, &row_ids, &row_types, "peaks");

    info!(
        "RNA: {} features, {} non-zeros",
        rna_split.n_features,
        rna_split.triplets.len()
    );
    info!(
        "ATAC: {} features, {} non-zeros",
        atac_split.n_features,
        atac_split.triplets.len()
    );

    // Write RNA backend
    let rna_out_path = format!("{}.rna", out_prefix);
    let rna = write_backend(&rna_split, &rna_out_path, ncols, backend, &column_names)?;
    info!("Created RNA backend: {}", rna_out_path);

    // Write ATAC backend
    let atac_out_path = format!("{}.atac", out_prefix);
    let atac = write_backend(&atac_split, &atac_out_path, ncols, backend, &column_names)?;
    info!("Created ATAC backend: {}", atac_out_path);

    Ok(MultiomeBackends { rna, atac })
}

/// Partition triplets for a single feature type, remapping row indices.
fn split_triplets(
    triplets: &[(u64, u64, f32)],
    row_ids: &[Box<str>],
    row_types: &[Box<str>],
    select_type: &str,
) -> ModalitySplit {
    // Build old_row → new_row mapping for this type
    let mut new_row_idx: Vec<Option<u64>> = vec![None; row_ids.len()];
    let mut selected_names: Vec<Box<str>> = Vec::new();
    let mut count: u64 = 0;

    for (i, rtype) in row_types.iter().enumerate() {
        if rtype.to_lowercase().contains(select_type) {
            new_row_idx[i] = Some(count);
            selected_names.push(row_ids[i].clone());
            count += 1;
        }
    }

    let n_features = count as usize;

    // Single pass: filter and remap
    let remapped: Vec<(u64, u64, f32)> = triplets
        .iter()
        .filter_map(|&(row, col, val)| {
            new_row_idx[row as usize].map(|new_row| (new_row, col, val))
        })
        .collect();

    ModalitySplit {
        triplets: remapped,
        row_names: selected_names,
        n_features,
    }
}

/// Write a modality split to a backend file.
fn write_backend(
    split: &ModalitySplit,
    out_path: &str,
    ncols: usize,
    backend: &SparseIoBackend,
    column_names: &[Box<str>],
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    let (resolved_backend, backend_file) =
        resolve_backend_file(out_path, Some(backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    let mut out = create_sparse_from_triplets(
        &split.triplets,
        (split.n_features, ncols, split.triplets.len()),
        Some(&backend_file),
        Some(&resolved_backend),
    )?;

    out.register_row_names_vec(&split.row_names);
    out.register_column_names_vec(column_names);

    Ok(out)
}
