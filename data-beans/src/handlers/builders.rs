use crate::misc::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use data_beans::zarr_io::*;

use log::info;
use matrix_util::common_io::*;
use std::sync::Arc;
use tempfile::TempDir;

// Import the argument structs from main
use crate::{
    From10xMatrixArgs, From10xMoleculeArgs, FromH5adArgs, FromMtxArgs, FromZarrArgs, RunSqueezeArgs,
};

// Import the run_squeeze function from main
use crate::run_squeeze;

pub fn run_build_from_mtx(args: &FromMtxArgs) -> anyhow::Result<()> {
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
            data_files: vec![backend_file.clone()],
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            block_size: 100,
            preload: true,
            show_histogram: false,
            save_histogram: None,
            dry_run: false,
            interactive: false,
            output: None,
            row_align: crate::RowAlignMode::Common,
        };

        run_squeeze(&squeeze_args)?;
    }

    info!("done");
    Ok(())
}

pub fn run_build_from_zarr_triplets(args: &FromZarrArgs) -> anyhow::Result<()> {
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
    .to_coo(args.pointer_type.clone())?;

    let TripletsShape { nrows, ncols, nnz } = shape;
    info!("Read {} non-zero elements in {} x {}", nnz, nrows, ncols);

    let mut row_ids = read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), &args.row_id_field)
        .or_else(|_| read_zarr_strings(store.clone(), args.row_id_field.as_ref()))
        .unwrap_or_else(|_| (0..nrows).map(|x| x.to_string().into_boxed_str()).collect());

    let mut row_names = read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), &args.row_name_field)
        .or_else(|_| read_zarr_strings(store.clone(), args.row_name_field.as_ref()))
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
            if !name.is_empty() {
                format!("{}_{}", id, name).into_boxed_str()
            } else {
                id
            }
        })
        .collect();

    let mut row_types = read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), &args.row_type_field)
        .or_else(|_| read_zarr_strings(store.clone(), args.row_type_field.as_ref()))
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
            .or_else(|_| {
                read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), &args.column_name_field)
            })
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
            data_files: vec![backend_file.clone()],
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            block_size: 100,
            preload: true,
            show_histogram: false,
            save_histogram: None,
            dry_run: false,
            interactive: false,
            output: None,
            row_align: crate::RowAlignMode::Common,
        };

        run_squeeze(&squeeze_args)?;
    }

    info!("done");
    Ok(())
}

/// Build backend from 10X Genomics H5 format (Cell Ranger).
pub fn run_build_from_10x_matrix(args: &From10xMatrixArgs) -> anyhow::Result<()> {
    let file = hdf5::File::open(args.h5_file.to_string())?;
    info!("Opened 10X H5 file: {}", args.h5_file);
    let (backend, backend_file) = resolve_backend_file(&args.output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    let root = file.group(args.root_group_name.as_ref()).map_err(|_| {
        anyhow::anyhow!(
            "unable to identify data with the specified root group name: {}",
            args.root_group_name
        )
    })?;

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
        .to_coo(args.pointer_type.clone())?
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

    if nrows < row_ids.len() {
        row_ids.truncate(nrows);
    }
    if nrows < row_names.len() {
        row_names.truncate(nrows);
    }
    assert_eq!(nrows, row_ids.len());
    assert_eq!(nrows, row_names.len());

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

    let mut row_types: Vec<Box<str>> = match root.dataset(args.row_type_field.as_ref()) {
        Ok(rows) => read_hdf5_strings(rows)?,
        _ => {
            info!("use all the types");
            vec![args.select_row_type.clone(); nrows]
        }
    };
    if nrows < row_types.len() {
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
            let low = x.to_lowercase();
            if low.contains(&select_pattern) && !low.contains(&remove_pattern) {
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

    run_squeeze_if_needed(
        args.do_squeeze,
        args.row_nnz_cutoff,
        args.column_nnz_cutoff,
        &backend_file,
    )?;
    info!("done");
    Ok(())
}

/// Build backend from AnnData h5ad format (CELLxGENE).
///
/// Prefers `raw/X` over `X` for raw counts.
/// Reads all obs columns and writes metadata `.tsv.gz` files.
/// Creates `barcode@donor_id` column names when donor_id is available.
pub fn run_build_from_h5ad(args: &FromH5adArgs) -> anyhow::Result<()> {
    let file = hdf5::File::open(args.h5ad_file.to_string())?;
    info!("Opened AnnData h5ad file: {}", args.h5ad_file);

    let (backend, backend_file) = resolve_backend_file(&args.output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    // Prefer raw/X (raw counts) over X (processed/normalized)
    let (x_path, var_path) = if file.group("raw/X").is_ok() {
        info!("Using raw/X (raw counts)");
        ("raw/X", "raw/var")
    } else {
        info!("Using X");
        ("X", "var")
    };

    let x_group = file.group(x_path)?;
    let var_group = file.group(var_path)?;
    let obs_group = file.group("obs")?;

    // Detect CSR vs CSC from encoding-type attribute on X group.
    // AnnData stores (obs x var) = (cells x features), but data-beans
    // convention is (features x cells). We swap the pointer type to
    // transpose: CSR -> Column pointers, CSC -> Row pointers.
    let pointer_type = {
        use hdf5::types::VarLenUnicode;
        let encoding: String = x_group
            .attr("encoding-type")
            .and_then(|a| a.read_scalar::<VarLenUnicode>())
            .map(|s| s.to_string())
            .unwrap_or_default();
        if encoding.contains("csr") {
            info!("Sparse format: CSR -> transposing to (features x cells)");
            IndexPointerType::Column
        } else {
            info!("Sparse format: CSC -> transposing to (features x cells)");
            IndexPointerType::Row
        }
    };

    // Read triplets
    let CooTripletsShape { triplets, shape } = {
        let values = x_group.dataset("data")?.read_1d::<f32>()?.to_vec();
        let indices = x_group.dataset("indices")?.read_1d::<u64>()?.to_vec();
        let indptr = x_group.dataset("indptr")?.read_1d::<u64>()?.to_vec();
        ValuesIndicesPointers {
            values: &values,
            indices: &indices,
            indptr: &indptr,
        }
        .to_coo(pointer_type)?
    };

    let TripletsShape { nrows, ncols, nnz } = shape;
    info!("Read {} non-zero elements in {} x {}", nnz, nrows, ncols);

    // Feature IDs from var/_index
    let mut row_ids: Vec<Box<str>> = match var_group.dataset("_index") {
        Ok(ds) => read_hdf5_strings(ds)?,
        _ => {
            info!("var/_index not found, using numeric IDs");
            (0..nrows).map(|x| x.to_string().into_boxed_str()).collect()
        }
    };

    // Feature names from var/feature_name (often categorical)
    let mut row_names: Vec<Box<str>> =
        read_h5ad_column(&var_group, "feature_name").unwrap_or_else(|_| vec![Box::from(""); nrows]);

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

    // Feature types from var/feature_type (often categorical)
    let mut row_types: Vec<Box<str>> =
        read_h5ad_column(&var_group, "feature_type").unwrap_or_else(|_| vec![Box::from(""); nrows]);
    if nrows < row_types.len() {
        row_types.truncate(nrows);
    }
    assert_eq!(nrows, row_types.len());

    // Cell barcodes from obs/_index
    let mut column_names: Vec<Box<str>> = match obs_group.dataset("_index") {
        Ok(ds) => read_hdf5_strings(ds)?,
        _ => {
            info!("obs/_index not found, using numeric IDs");
            (0..ncols).map(|x| x.to_string().into_boxed_str()).collect()
        }
    };
    info!("Read {} barcodes", column_names.len());
    if ncols < column_names.len() {
        column_names.truncate(ncols);
    }
    assert_eq!(ncols, column_names.len());

    // Create sparse backend
    let mut out = create_sparse_from_triplets(
        &triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("created sparse matrix: {}", backend_file);
    out.register_row_names_vec(&row_ids);

    // Log feature type distribution (AnnData uses biotypes, not "Gene Expression")
    {
        let mut type_counts: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        for t in &row_types {
            *type_counts.entry(t.as_ref()).or_insert(0) += 1;
        }
        info!("Feature types: {:?}", type_counts);
    }

    // Filter by feature type only if user explicitly provides --select-row-type
    let select_rows: Vec<usize> = if !args.select_row_type.is_empty() {
        let select_pattern = args.select_row_type.to_lowercase();
        let remove_pattern = args.remove_row_type.to_lowercase();
        row_types
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                let low = x.to_lowercase();
                let selected = low.contains(&select_pattern);
                let removed = !remove_pattern.is_empty() && low.contains(&remove_pattern);
                if selected && !removed {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    } else {
        // Keep all features for AnnData (data is already curated)
        (0..nrows).collect()
    };

    // Read all obs metadata columns
    let obs_df = read_h5ad_dataframe(&obs_group)?;
    info!(
        "Read {} obs columns: {:?}",
        obs_df.col_names.len(),
        obs_df.col_names
    );

    // Output stem for metadata files (strip .zarr/.h5)
    let output_stem = backend_file
        .strip_suffix(".zarr")
        .or_else(|| backend_file.strip_suffix(".h5"))
        .unwrap_or(&backend_file);

    // Handle donor_id: barcode@donor_id column names + donor mapping files
    let donor_idx = obs_df
        .col_names
        .iter()
        .position(|c| c.as_ref() == "donor_id");

    if let Some(di) = donor_idx {
        let donors = &obs_df.col_data[di];
        let new_col_names: Vec<Box<str>> = column_names
            .iter()
            .zip(donors.iter())
            .map(|(b, d)| format!("{}@{}", b, d).into_boxed_str())
            .collect();
        out.register_column_names_vec(&new_col_names);
        info!("Column names set to barcode@donor_id");

        // barcode_to_donor.tsv.gz
        let donor_file = format!("{}.barcode_to_donor.tsv.gz", output_stem);
        let mut lines: Vec<Box<str>> = Vec::with_capacity(column_names.len() + 1);
        lines.push("barcode\tdonor_id".into());
        for (b, d) in column_names.iter().zip(donors.iter()) {
            lines.push(format!("{}\t{}", b, d).into());
        }
        write_lines(&lines, &donor_file)?;
        info!("Wrote {}", donor_file);

        // sample_metadata.tsv.gz (one row per unique donor)
        let sample_file = format!("{}.sample_metadata.tsv.gz", output_stem);
        write_sample_metadata(&sample_file, &obs_df.col_names, &obs_df.col_data, di)?;
        info!("Wrote {}", sample_file);
    } else {
        out.register_column_names_vec(&column_names);
    }

    if select_rows.len() < nrows {
        info!(
            "filtering features: {} -> {} of `{}` type",
            nrows,
            select_rows.len(),
            args.select_row_type
        );
        out.subset_columns_rows(None, Some(&select_rows))?;
    }

    // cell_metadata.tsv.gz — all obs columns indexed by barcode
    let meta_file = format!("{}.cell_metadata.tsv.gz", output_stem);
    let mut lines: Vec<Box<str>> = Vec::with_capacity(column_names.len() + 1);
    // header
    let header = std::iter::once("barcode")
        .chain(obs_df.col_names.iter().map(|s| s.as_ref()))
        .collect::<Vec<&str>>()
        .join("\t");
    lines.push(header.into());
    // rows
    for i in 0..column_names.len() {
        let row = std::iter::once(column_names[i].as_ref())
            .chain(obs_df.col_data.iter().map(|col| col[i].as_ref()))
            .collect::<Vec<&str>>()
            .join("\t");
        lines.push(row.into());
    }
    write_lines(&lines, &meta_file)?;
    info!("Wrote {}", meta_file);

    run_squeeze_if_needed(
        args.do_squeeze,
        args.row_nnz_cutoff,
        args.column_nnz_cutoff,
        &backend_file,
    )?;
    info!("done");
    Ok(())
}

/// Write sample-level metadata: one row per unique donor_id.
fn write_sample_metadata(
    path: &str,
    col_names: &[Box<str>],
    col_data: &[Vec<Box<str>>],
    donor_col_idx: usize,
) -> anyhow::Result<()> {
    use std::collections::BTreeMap;

    let donors = &col_data[donor_col_idx];
    let mut donor_first_row: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, d) in donors.iter().enumerate() {
        donor_first_row.entry(d.as_ref()).or_insert(i);
    }

    let mut lines: Vec<Box<str>> = Vec::with_capacity(donor_first_row.len() + 1);

    // header: donor_id + all other columns
    let header = std::iter::once("donor_id")
        .chain(
            col_names
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != donor_col_idx)
                .map(|(_, s)| s.as_ref()),
        )
        .collect::<Vec<&str>>()
        .join("\t");
    lines.push(header.into());

    // one row per unique donor
    for (donor_id, &row_i) in &donor_first_row {
        let row = std::iter::once(*donor_id)
            .chain(
                col_data
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != donor_col_idx)
                    .map(|(_, col)| col[row_i].as_ref()),
            )
            .collect::<Vec<&str>>()
            .join("\t");
        lines.push(row.into());
    }

    write_lines(&lines, path)?;
    Ok(())
}

/// Build backend from 10X Genomics molecule_info.h5.
///
/// Aggregates per-molecule counts into a feature x cell sparse matrix.
/// Handles multi-gem-group (cellranger aggr) and multi-library (CITE-seq).
pub fn run_build_from_10x_molecule(args: &From10xMoleculeArgs) -> anyhow::Result<()> {
    let file = hdf5::File::open(args.h5_file.to_string())?;
    info!("Opened molecule_info.h5: {}", args.h5_file);

    let (backend, backend_file) = resolve_backend_file(&args.output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    // 1. Read per-molecule arrays
    let barcode_idx = file.dataset("barcode_idx")?.read_1d::<u64>()?.to_vec();
    let feature_idx = file.dataset("feature_idx")?.read_1d::<u32>()?.to_vec();
    let count = file.dataset("count")?.read_1d::<u32>()?.to_vec();
    let gem_group = file.dataset("gem_group")?.read_1d::<u16>()?.to_vec();
    let library_idx = file.dataset("library_idx")?.read_1d::<u16>()?.to_vec();
    let n_molecules = barcode_idx.len();
    info!("Read {} molecules", n_molecules);

    // 2. Read lookup tables
    let barcodes = read_hdf5_strings(file.dataset("barcodes")?)?;
    let feature_group = file.group("features")?;
    let mut row_ids: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("id")?)?;
    let mut row_names: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("name")?)?;
    let mut row_types: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("feature_type")?)?;

    let n_features = row_ids.len();
    info!("Read {} barcodes, {} features", barcodes.len(), n_features);

    // 3. Parse library_info and filter by library type
    let valid_libraries: std::collections::HashSet<u16> = {
        let lib_info_ds = file.dataset("library_info")?;
        let lib_info_raw = read_hdf5_strings(lib_info_ds)?;
        let lib_info_json: String = lib_info_raw.iter().map(|s| s.as_ref()).collect();
        let lib_entries: Vec<serde_json::Value> = serde_json::from_str(&lib_info_json)?;

        let mut valid = std::collections::HashSet::new();
        for entry in &lib_entries {
            if let (Some(lib_id), Some(lib_type)) = (
                entry.get("library_id").and_then(|v| v.as_u64()),
                entry.get("library_type").and_then(|v| v.as_str()),
            ) {
                if lib_type.contains(args.library_type.as_ref()) {
                    valid.insert(lib_id as u16);
                }
            }
        }
        info!(
            "Library type '{}': {} of {} libraries match",
            args.library_type,
            valid.len(),
            lib_entries.len()
        );
        valid
    };

    // 4. Read pass_filter if needed
    let valid_cells: Option<std::collections::HashSet<(u64, u16)>> = if !args.no_pass_filter {
        let pf = file.dataset("barcode_info/pass_filter")?.read_2d::<u64>()?;
        let mut cells = std::collections::HashSet::new();
        for row in pf.rows() {
            let bc_idx = row[0];
            let lib_idx = row[1] as u16;
            if valid_libraries.contains(&lib_idx) {
                cells.insert((bc_idx, lib_idx));
            }
        }
        info!("pass_filter: {} valid cells", cells.len());
        Some(cells)
    } else {
        info!("Skipping pass_filter (--no-pass-filter)");
        None
    };

    // 5. Filter molecules and aggregate into triplets
    //    Column key = (barcode_idx, gem_group)
    use std::collections::{BTreeSet, HashMap};

    let mut col_keys = BTreeSet::new();
    let mut triplet_map: HashMap<(u64, u64), f32> = HashMap::new();

    for i in 0..n_molecules {
        // Filter by library type
        if !valid_libraries.contains(&library_idx[i]) {
            continue;
        }

        // Filter by pass_filter
        if let Some(ref cells) = valid_cells {
            if !cells.contains(&(barcode_idx[i], library_idx[i])) {
                continue;
            }
        }

        let col_key = (barcode_idx[i], gem_group[i]);
        col_keys.insert(col_key);

        // Will remap column index after collecting all keys
        let row = feature_idx[i] as u64;
        *triplet_map
            .entry((row, barcode_idx[i] * 65536 + gem_group[i] as u64))
            .or_insert(0.0) += count[i] as f32;
    }

    // Build dense column index mapping
    let col_keys_vec: Vec<(u64, u16)> = col_keys.into_iter().collect();
    let col_key_to_idx: HashMap<(u64, u16), u64> = col_keys_vec
        .iter()
        .enumerate()
        .map(|(idx, &key)| (key, idx as u64))
        .collect();
    let ncols = col_keys_vec.len();

    // Build column names: SEQUENCE-GEMGROUP
    let column_names: Vec<Box<str>> = col_keys_vec
        .iter()
        .map(|&(bc_idx, gg)| {
            let bc = barcodes[bc_idx as usize].as_ref();
            format!("{}-{}", bc, gg).into_boxed_str()
        })
        .collect();

    info!("Aggregated into {} columns (cells)", ncols);

    // Convert triplet_map to proper triplets with remapped column indices
    let triplets: Vec<(u64, u64, f32)> = triplet_map
        .into_iter()
        .map(|((row, packed_col), val)| {
            let bc_idx = packed_col / 65536;
            let gg = (packed_col % 65536) as u16;
            let col = col_key_to_idx[&(bc_idx, gg)];
            (row, col, val)
        })
        .collect();

    let nrows = n_features;
    let nnz = triplets.len();
    info!("Built {} triplets in {} x {} matrix", nnz, nrows, ncols);

    // 6. Build backend
    let mut out = create_sparse_from_triplets(
        &triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("Created sparse matrix: {}", backend_file);

    // 7. Register names
    // Composite row names: id_name
    if nrows < row_ids.len() {
        row_ids.truncate(nrows);
    }
    if nrows < row_names.len() {
        row_names.truncate(nrows);
    }

    let row_id_names: Vec<Box<str>> = row_ids
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

    out.register_row_names_vec(&row_id_names);
    out.register_column_names_vec(&column_names);

    // 8. Filter by feature type
    if nrows < row_types.len() {
        row_types.truncate(nrows);
    }

    let select_pattern = args.select_row_type.to_lowercase();
    let remove_pattern = args.remove_row_type.to_lowercase();
    let select_rows: Vec<usize> = row_types
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            let low = x.to_lowercase();
            let selected = low.contains(&select_pattern);
            let removed = !remove_pattern.is_empty() && low.contains(&remove_pattern);
            if selected && !removed {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if select_rows.len() < nrows {
        info!(
            "Filtering features: {} -> {} of '{}' type",
            nrows,
            select_rows.len(),
            args.select_row_type
        );
        out.subset_columns_rows(None, Some(&select_rows))?;
    }

    // 9. Squeeze if needed
    run_squeeze_if_needed(
        args.do_squeeze,
        args.row_nnz_cutoff,
        args.column_nnz_cutoff,
        &backend_file,
    )?;
    info!("done");
    Ok(())
}

fn run_squeeze_if_needed(
    do_squeeze: bool,
    row_nnz_cutoff: usize,
    column_nnz_cutoff: usize,
    backend_file: &str,
) -> anyhow::Result<()> {
    if do_squeeze {
        info!("Squeeze the backend data {}", backend_file);
        let squeeze_args = RunSqueezeArgs {
            data_files: vec![backend_file.into()],
            row_nnz_cutoff,
            column_nnz_cutoff,
            block_size: 100,
            preload: true,
            show_histogram: false,
            save_histogram: None,
            dry_run: false,
            interactive: false,
            output: None,
            row_align: crate::RowAlignMode::Common,
        };
        run_squeeze(&squeeze_args)?;
    }
    Ok(())
}
