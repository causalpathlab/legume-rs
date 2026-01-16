use crate::misc::*;
use crate::sparse_io::*;
use crate::sparse_util::*;

use matrix_util::common_io::*;
use log::info;
use std::sync::Arc;
use tempfile::TempDir;

// Import the argument structs from main
use crate::{FromH5Args, FromMtxArgs, FromZarrArgs, RunSqueezeArgs};

// Import the run_squeeze function from main
use crate::run_squeeze;

pub fn run_build_from_mtx(args: &FromMtxArgs) -> anyhow::Result<()> {
    if args.verbose > 0 {
        std::env::set_var("RUST_LOG", "info");
    }

    let mtx_file = args.mtx.as_ref();
    let row_file = args.row.as_ref();
    let col_file = args.col.as_ref();

    let (backend, backend_file) = resolve_backend_file(&args.output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        remove_file(&backend_file)?;
    }

    let mut data = create_sparse_from_mtx_file(mtx_file, Some(&backend_file), Some(&backend))?;

    if let Some(row_file) = row_file {
        data.register_row_names_file(row_file);
    } else if let Some(nrow) = data.num_rows() {
        let row_names: Vec<Box<str>> = (1..(nrow + 1)).map(|i| format!("{}", i).into()).collect();
        data.register_row_names_vec(&row_names);
    }

    if let Some(col_file) = col_file {
        data.register_column_names_file(col_file);
    } else if let Some(ncol) = data.num_columns() {
        let col_names: Vec<Box<str>> = (1..(ncol + 1)).map(|i| format!("{}", i).into()).collect();
        data.register_column_names_vec(&col_names);
    }

    if args.do_squeeze {
        let squeeze_args = RunSqueezeArgs {
            data_file: backend_file.clone(),
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            block_size: 100,
        };

        run_squeeze(&squeeze_args)?;
    }

    info!("done");
    Ok(())
}

pub fn run_build_from_zarr_triplets(args: &FromZarrArgs) -> anyhow::Result<()> {
    if args.verbose > 0 {
        std::env::set_var("RUST_LOG", "info");
    }

    let source_zarr_file_path = args.zarr_file.clone();

    let (backend, backend_file) = resolve_backend_file(&args.output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        remove_file(&backend_file)?;
    }

    let (_dir, base, ext) = dir_base_ext(&source_zarr_file_path)?;

    let temp_dir = TempDir::new()?; // should be kept outside

    let store = if ext.as_ref() == "zip" {
        let temp_path = std::path::PathBuf::from(temp_dir.path());
        let temp_zarr = format!("{}/{}", temp_path.to_str().unwrap(), base);
        // let temp_zarr = format!("{}/{}", dir, base);
        unzip_dir(&source_zarr_file_path, Some(temp_zarr.as_ref()))?;
        info!("Unzipped to {}", temp_zarr);
        Arc::new(zarrs::filesystem::FilesystemStore::new(temp_zarr)?)
    } else {
        info!("Store at {}", source_zarr_file_path);
        Arc::new(zarrs::filesystem::FilesystemStore::new(
            source_zarr_file_path.as_ref(),
        )?)
    };

    let indices: Vec<u64> = read_zarr_numerics(store.clone(), args.indices_field.as_ref())?;
    let indptr: Vec<u64> = read_zarr_numerics(store.clone(), args.indptr_field.as_ref())?;
    let values: Vec<f32> = read_zarr_numerics(store.clone(), args.data_field.as_ref())?;
    info!("Read the arrays");

    let CooTripletsShape { triplets, shape } = ValuesIndicesPointers {
        values: &values,
        indices: &indices,
        indptr: &indptr,
    }
    .into_coo(args.pointer_type.clone())?;

    let TripletsShape { nrows, ncols, nnz } = shape;
    info!("Read {} non-zero elements in {} x {}", nnz, nrows, ncols);

    let mut row_ids = read_zarr_attr::<Vec<Box<str>>>(store.clone(), &args.row_id_field)
        .or_else(|_| read_zarr_strings(store.clone(), args.row_id_field.as_ref()))
        .unwrap_or_else(|_| (0..nrows).map(|x| x.to_string().into_boxed_str()).collect());

    let mut row_names = read_zarr_attr::<Vec<Box<str>>>(store.clone(), &args.row_name_field)
        .or_else(|_| read_zarr_strings(store.clone(), args.row_id_field.as_ref()))
        .unwrap_or_else(|_| (0..nrows).map(|x| x.to_string().into_boxed_str()).collect());

    info!("Read {} row names", row_ids.len());
    if nrows < row_ids.len() {
        info!("data doesn't contain all the row IDs");
        row_ids.truncate(nrows);
    }

    if nrows < row_names.len() {
        info!("data doesn't contain all the row names");
        row_names.truncate(nrows);
    }

    assert_eq!(nrows, row_ids.len());
    assert_eq!(nrows, row_names.len());

    // have composite row names
    let row_ids: Vec<Box<str>> = row_ids
        .into_iter()
        .zip(row_names)
        .map(|(id, name)| {
            if name.len() > 0 {
                format!("{}_{}", id, name).into_boxed_str()
            } else {
                id
            }
        })
        .collect();

    let mut row_types = read_zarr_attr::<Vec<Box<str>>>(store.clone(), &args.row_type_field)
        .or_else(|_| read_zarr_strings(store.clone(), args.row_id_field.as_ref()))
        .unwrap_or_else(|_| vec![args.select_row_type.clone(); nrows]);
    if nrows < row_types.len() {
        info!("data doesn't contain all the rows");
        row_types.truncate(nrows);
    }
    assert_eq!(nrows, row_types.len());

    let select_pattern = args.select_row_type.to_lowercase();
    let remove_pattern = args.remove_row_type.to_lowercase();
    let select_rows = row_types
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            if x.to_lowercase().contains(&select_pattern)
                && !x.to_lowercase().contains(&remove_pattern)
            {
                Some(i)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut column_names =
        parse_10x_cell_id(read_zarr_ndarray::<u32>(store.clone(), &args.column_name_field)?.view())
            .or_else(|_| read_zarr_attr::<Vec<Box<str>>>(store.clone(), &args.column_name_field))
            .or_else(|_| read_zarr_strings(store.clone(), args.column_name_field.as_ref()))
            .unwrap_or_else(|_| (0..ncols).map(|x| x.to_string().into_boxed_str()).collect());

    if ncols < column_names.len() {
        info!("data doesn't contain all the columns");
        column_names.truncate(ncols);
    }
    assert_eq!(ncols, column_names.len());

    let mut out = create_sparse_from_triplets(
        &triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("created sparse matrix: {}", backend_file);
    out.register_row_names_vec(&row_ids);
    out.register_column_names_vec(&column_names);

    if select_rows.len() < nrows {
        info!("filtering in features of `{}` type", args.select_row_type);
        out.subset_columns_rows(None, Some(&select_rows))?;
    }

    if args.do_squeeze {
        info!("Squeeze the backend data {}", &backend_file);
        let squeeze_args = RunSqueezeArgs {
            data_file: backend_file.clone(),
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            block_size: 100,
        };

        run_squeeze(&squeeze_args)?;
    }

    info!("done");
    Ok(())
}

pub fn run_build_from_h5_triplets(args: &FromH5Args) -> anyhow::Result<()> {
    if args.verbose > 0 {
        std::env::set_var("RUST_LOG", "info");
    }

    let h5_source_file = args.h5_file.clone();

    let (backend, backend_file) = resolve_backend_file(&args.output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        remove_file(&backend_file)?;
    }

    let file = hdf5::File::open(h5_source_file.to_string())?;
    info!("Opened data file: {}", h5_source_file.clone());

    if let Ok(root) = file.group(args.root_group_name.as_ref()) {
        // Read triplets with the shape information
        let CooTripletsShape { triplets, shape } = if let (Ok(values), Ok(indices), Ok(indptr)) = (
            root.dataset(args.data_field.as_ref()),
            root.dataset(args.indices_field.as_ref()),
            root.dataset(args.indptr_field.as_ref()),
        ) {
            ValuesIndicesPointers {
                values: &values.read_1d::<f32>()?.to_vec(),
                indices: &indices.read_1d::<u64>()?.to_vec(),
                indptr: &indptr.read_1d::<u64>()?.to_vec(),
            }
            .into_coo(args.pointer_type.clone())?
        } else {
            return Err(anyhow::anyhow!("unable to read triplets"));
        };

        let TripletsShape { nrows, ncols, nnz } = shape;
        info!("Read {} non-zero elements in {} x {}", nnz, nrows, ncols);

        let mut row_ids: Vec<Box<str>> = match root.dataset(args.row_id_field.as_ref()) {
            Ok(rows) => read_hdf5_strings(rows)?,
            _ => {
                info!("row (feature) IDs not found");
                (0..nrows).map(|x| x.to_string().into_boxed_str()).collect()
            }
        };

        let mut row_names: Vec<Box<str>> = match root.dataset(args.row_name_field.as_ref()) {
            Ok(rows) => read_hdf5_strings(rows)?,
            _ => {
                info!("row (feature) names not found");
                vec![Box::from(""); nrows]
            }
        };

        info!("Read {} row names", row_ids.len());
        if nrows < row_ids.len() {
            info!("data doesn't contain all the row IDs");
            row_ids.truncate(nrows);
        }

        if nrows < row_names.len() {
            info!("data doesn't contain all the row names");
            row_names.truncate(nrows);
        }

        assert_eq!(nrows, row_ids.len());
        assert_eq!(nrows, row_names.len());

        // have composite row names
        let row_ids: Vec<Box<str>> = row_ids
            .into_iter()
            .zip(row_names)
            .map(|(id, name)| {
                if name.len() > 0 {
                    format!("{}_{}", id, name).into_boxed_str()
                } else {
                    id
                }
            })
            .collect();

        let mut row_types: Vec<Box<str>> = match root.dataset(args.row_type_field.as_ref()) {
            Ok(rows) => read_hdf5_strings(rows)?,
            _ => {
                info!("use all the types");
                vec![args.select_row_type.clone(); nrows]
            }
        };

        if nrows < row_types.len() {
            info!("data doesn't contain all the row types");
            row_types.truncate(nrows);
        }

        assert_eq!(nrows, row_types.len());

        let mut column_names: Vec<Box<str>> = match root.dataset(args.column_name_field.as_ref()) {
            Ok(columns) => read_hdf5_strings(columns)?,
            _ => {
                info!("column (cell) names not found");
                (0..ncols).map(|x| x.to_string().into_boxed_str()).collect()
            }
        };
        info!("Read {} column names", column_names.len());

        if ncols < column_names.len() {
            info!("data doesn't contain all the columns");
            column_names.truncate(ncols);
        }
        assert_eq!(ncols, column_names.len());

        let mut out = create_sparse_from_triplets(
            &triplets,
            (nrows, ncols, nnz),
            Some(&backend_file),
            Some(&backend),
        )?;
        info!("created sparse matrix: {}", backend_file);
        out.register_row_names_vec(&row_ids);
        out.register_column_names_vec(&column_names);

        let select_pattern = args.select_row_type.to_lowercase();
        let remove_pattern = args.remove_row_type.to_lowercase();
        let select_rows = row_types
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if x.to_lowercase().contains(&select_pattern)
                    && !x.to_lowercase().contains(&remove_pattern)
                {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if select_rows.len() < nrows {
            info!("filtering in features of `{}` type", args.select_row_type);
            out.subset_columns_rows(None, Some(&select_rows))?;
        }
    } else {
        return Err(anyhow::anyhow!(
            "unable to identify data with the specified root group name: {}",
            args.root_group_name
        ));
    }

    if args.do_squeeze {
        info!("Squeeze the backend data {}", &backend_file);
        let squeeze_args = RunSqueezeArgs {
            data_file: backend_file.clone(),
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            block_size: 100,
        };

        run_squeeze(&squeeze_args)?;
    }
    info!("done");
    Ok(())
}
