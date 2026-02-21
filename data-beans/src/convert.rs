use crate::misc::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::zarr_io::*;

use log::info;
use matrix_util::common_io::*;
use std::sync::Arc;
use tempfile::TempDir;

/// Convert a 10x-format HDF5 file (Cell Ranger / h5ad) to a data-beans backend.
///
/// Uses standard 10x field layout:
///   root_group/data, root_group/indices, root_group/indptr (CSC),
///   root_group/features/{id,name,feature_type}, root_group/barcodes
pub fn convert_h5_to_backend(h5_file: &str, output: &str) -> anyhow::Result<()> {
    let (backend, backend_file) =
        resolve_backend_file(&Box::from(output), Some(SparseIoBackend::Zarr))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    let file = hdf5::File::open(h5_file)?;
    info!("Opened data file: {}", h5_file);

    let root_group_name = "matrix";
    let root = file.group(root_group_name).map_err(|_| {
        anyhow::anyhow!(
            "Unable to find root group '{}' in {}",
            root_group_name,
            h5_file,
        )
    })?;

    // Read triplets
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
        Ok(rows) => read_hdf5_strings(rows)?,
        _ => {
            info!("row (feature) IDs not found");
            (0..nrows).map(|x| x.to_string().into_boxed_str()).collect()
        }
    };

    // Row names
    let mut row_names: Vec<Box<str>> = match root.dataset("features/name") {
        Ok(rows) => read_hdf5_strings(rows)?,
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

    // Row types for filtering
    let mut row_types: Vec<Box<str>> = match root.dataset("features/feature_type") {
        Ok(rows) => read_hdf5_strings(rows)?,
        _ => {
            info!("use all the types");
            vec!["Gene Expression".to_string().into_boxed_str(); nrows]
        }
    };
    if nrows < row_types.len() {
        row_types.truncate(nrows);
    }
    assert_eq!(nrows, row_types.len());

    // Column names
    let mut column_names: Vec<Box<str>> = match root.dataset("barcodes") {
        Ok(columns) => read_hdf5_strings(columns)?,
        _ => {
            info!("column (cell) names not found");
            (0..ncols).map(|x| x.to_string().into_boxed_str()).collect()
        }
    };
    if ncols < column_names.len() {
        column_names.truncate(ncols);
    }
    assert_eq!(ncols, column_names.len());

    // Create backend
    let mut out = create_sparse_from_triplets(
        &triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("Created sparse matrix: {}", backend_file);
    out.register_row_names_vec(&row_ids);
    out.register_column_names_vec(&column_names);

    // Filter by gene expression type
    let select_pattern = "gene expression";
    let remove_pattern = "aggregate";
    let select_rows: Vec<usize> = row_types
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            let lower = x.to_lowercase();
            if lower.contains(select_pattern) && !lower.contains(remove_pattern) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if select_rows.len() < nrows {
        info!(
            "Filtering features: {} -> {} rows of gene expression type",
            nrows,
            select_rows.len()
        );
        out.subset_columns_rows(None, Some(&select_rows))?;
    }

    info!("Conversion done: {}", backend_file);
    Ok(())
}

/// Convert a 10x-format Zarr file (Xenium zarr.zip or directory) to a data-beans backend.
///
/// Uses standard 10x Xenium field layout:
///   /cell_features/{data,indices,indptr} (CSC),
///   /cell_features/features/{id,name,feature_type},
///   /cell_features/cell_id
pub fn convert_zarr_to_backend(zarr_file: &str, output: &str) -> anyhow::Result<()> {
    let (backend, backend_file) =
        resolve_backend_file(&Box::from(output), Some(SparseIoBackend::Zarr))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    let (_dir, base, ext) = dir_base_ext(zarr_file)?;
    let temp_dir = TempDir::new()?;

    let store = if ext.as_ref() == "zip" {
        let temp_path = std::path::PathBuf::from(temp_dir.path());
        let temp_zarr = format!("{}/{}", temp_path.to_str().unwrap(), base);
        unzip_dir(zarr_file, Some(temp_zarr.as_ref()))?;
        info!("Unzipped to {}", temp_zarr);
        Arc::new(zarrs::filesystem::FilesystemStore::new(temp_zarr)?)
    } else {
        info!("Store at {}", zarr_file);
        Arc::new(zarrs::filesystem::FilesystemStore::new(zarr_file)?)
    };

    let indices: Vec<u64> = read_zarr_numerics(store.clone(), "/cell_features/indices")?;
    let indptr: Vec<u64> = read_zarr_numerics(store.clone(), "/cell_features/indptr")?;
    let values: Vec<f32> = read_zarr_numerics(store.clone(), "/cell_features/data")?;
    info!("Read the arrays");

    let CooTripletsShape { triplets, shape } = ValuesIndicesPointers {
        values: &values,
        indices: &indices,
        indptr: &indptr,
    }
    .to_coo(IndexPointerType::Column)?;

    let TripletsShape { nrows, ncols, nnz } = shape;
    info!("Read {} non-zero elements in {} x {}", nnz, nrows, ncols);

    let row_id_field = "/cell_features/features/id";
    let row_name_field = "/cell_features/features/name";
    let row_type_field = "/cell_features/features/feature_type";
    let column_name_field = "/cell_features/cell_id";

    let mut row_ids = read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), row_id_field)
        .or_else(|_| read_zarr_strings(store.clone(), row_id_field))
        .unwrap_or_else(|_| (0..nrows).map(|x| x.to_string().into_boxed_str()).collect());

    let mut row_names = read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), row_name_field)
        .or_else(|_| read_zarr_strings(store.clone(), row_name_field))
        .unwrap_or_else(|_| (0..nrows).map(|x| x.to_string().into_boxed_str()).collect());

    info!("Read {} row names", row_ids.len());
    if nrows < row_ids.len() {
        row_ids.truncate(nrows);
    }
    if nrows < row_names.len() {
        row_names.truncate(nrows);
    }
    assert_eq!(nrows, row_ids.len());
    assert_eq!(nrows, row_names.len());

    // Composite row names
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

    let mut row_types = read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), row_type_field)
        .or_else(|_| read_zarr_strings(store.clone(), row_type_field))
        .unwrap_or_else(|_| vec!["Gene Expression".to_string().into_boxed_str(); nrows]);
    if nrows < row_types.len() {
        row_types.truncate(nrows);
    }
    assert_eq!(nrows, row_types.len());

    let mut column_names =
        parse_10x_cell_id(read_zarr_ndarray::<u32>(store.clone(), column_name_field)?.view())
            .or_else(|_| read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), column_name_field))
            .or_else(|_| read_zarr_strings(store.clone(), column_name_field))
            .unwrap_or_else(|_| (0..ncols).map(|x| x.to_string().into_boxed_str()).collect());

    if ncols < column_names.len() {
        column_names.truncate(ncols);
    }
    assert_eq!(ncols, column_names.len());

    let mut out = create_sparse_from_triplets(
        &triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("Created sparse matrix: {}", backend_file);
    out.register_row_names_vec(&row_ids);
    out.register_column_names_vec(&column_names);

    // Filter by gene expression type
    let select_pattern = "gene expression";
    let remove_pattern = "aggregate";
    let select_rows: Vec<usize> = row_types
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            let lower = x.to_lowercase();
            if lower.contains(select_pattern) && !lower.contains(remove_pattern) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if select_rows.len() < nrows {
        info!(
            "Filtering features: {} -> {} rows of gene expression type",
            nrows,
            select_rows.len()
        );
        out.subset_columns_rows(None, Some(&select_rows))?;
    }

    info!("Conversion done: {}", backend_file);
    Ok(())
}

/// Try to open a data file directly; if that fails, attempt automatic
/// conversion from raw 10x formats (h5/h5ad, zarr/zarr.zip).
///
/// Converted backends are cached as `{data_file}.db.zarr` next to
/// the original file so subsequent calls skip conversion.
pub fn try_open_or_convert(
    data_file: &str,
) -> anyhow::Result<Box<dyn SparseIo<IndexIter = Vec<usize>>>> {
    let ext = file_ext(data_file)?;
    let backend = match ext.as_ref() {
        "h5" | "h5ad" => SparseIoBackend::HDF5,
        _ => SparseIoBackend::Zarr,
    };

    match open_sparse_matrix(data_file, &backend) {
        Ok(data) => Ok(data),
        Err(original_err) => {
            let converted = format!("{}.db.zarr", data_file);

            if std::path::Path::new(&converted).exists() {
                info!("Using cached conversion: {}", converted);
                return open_sparse_matrix(&converted, &SparseIoBackend::Zarr);
            }

            match ext.as_ref() {
                "h5" | "h5ad" => {
                    info!(
                        "Converting h5/h5ad to backend: {} -> {}",
                        data_file, converted
                    );
                    convert_h5_to_backend(data_file, &converted)?;
                }
                "zarr" | "zip" => {
                    info!("Converting zarr to backend: {} -> {}", data_file, converted);
                    convert_zarr_to_backend(data_file, &converted)?;
                }
                _ => return Err(original_err),
            }

            open_sparse_matrix(&converted, &SparseIoBackend::Zarr)
        }
    }
}
