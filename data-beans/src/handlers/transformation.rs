use crate::misc::*;
use crate::qc::*;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{read_row_names, read_col_names, MAX_ROW_NAME_IDX, MAX_COLUMN_NAME_IDX};

use matrix_util::common_io::*;
use log::info;
use fnv::FnvHashMap as HashMap;

// Import the argument structs from main.rs
// These will need to be public in main.rs
use crate::{SubsetColumnsArgs, ReorderRowsArgs, RunSqueezeArgs};

/// Subset columns from a sparse matrix data file
///
/// This function takes a subset of columns from the input data file and writes
/// them to a new output file. Columns can be specified either by indices or by
/// a file containing column names. Optionally, the output can be squeezed to
/// remove rows/columns with too few non-zero entries.
pub fn subset_columns(args: &SubsetColumnsArgs) -> anyhow::Result<()> {
    let columns_indices = args.column_indices.clone();
    let column_name_file = args.name_file.clone();

    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;

    let (_, output_file) = resolve_backend_file(&args.output, Some(backend.clone()))?;

    if let Some(out_dir) = dirname(&output_file) {
        mkdir(&out_dir)?;
    }

    recursive_copy(&data_file, &output_file)?;
    info!("copied the existing data file {}", data_file);

    let mut data = open_sparse_matrix(&output_file, &backend)?;

    let selected_columns = if let Some(idx) = columns_indices {
        idx
    } else if let Some(column_file) = column_name_file {
        let col_names = read_col_names(column_file, MAX_COLUMN_NAME_IDX)?;
        if col_names.is_empty() {
            return Err(anyhow::anyhow!("Empty column file"));
        }
        let col_names_map = data
            .column_names()
            .expect("column names not found in data file")
            .iter()
            .enumerate()
            .map(|(i, x)| (x.clone(), i))
            .collect::<HashMap<_, _>>();

        let col_names_order = col_names
            .into_iter()
            .filter_map(|x| col_names_map.get(&x))
            .collect::<Vec<_>>();

        let idx: Vec<usize> = col_names_order.iter().map(|&x| *x).collect();
        if idx.is_empty() {
            return Err(anyhow::anyhow!("Found empty columns"));
        }
        idx
    } else {
        return Err(anyhow::anyhow!(
            "either `column-indices` or `name-file` must be provided"
        ));
    };

    data.subset_columns_rows(Some(&selected_columns), None)?;

    if args.do_squeeze {
        info!("Squeeze the backend data {}", &output_file);
        let squeeze_args = RunSqueezeArgs {
            data_file: output_file.into(),
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            block_size: 100,
        };
        run_squeeze(&squeeze_args)?;
    }

    info!("done");
    Ok(())
}

/// Reorder rows in a sparse matrix data file
///
/// This function reorders the rows of the input data file according to the
/// order specified in a row name file, and writes the result to a new output file.
pub fn reorder_rows(args: &ReorderRowsArgs) -> anyhow::Result<()> {
    let row_names_order: Vec<Box<str>> = read_row_names(args.row_file.clone(), MAX_ROW_NAME_IDX)?;

    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;
    let (_, output_file) = resolve_backend_file(&args.output, Some(backend.clone()))?;

    if let Some(out_dir) = dirname(&output_file) {
        mkdir(&out_dir)?;
    }

    recursive_copy(&data_file, &output_file)?;
    info!("copied the existing data file {}", data_file);

    let mut data = open_sparse_matrix(&output_file, &backend)?;
    data.reorder_rows(&row_names_order)?;

    info!("done");
    Ok(())
}

/// Squeeze a sparse matrix by removing rows and columns with too few non-zero entries
///
/// This function removes rows and columns from the data file that have fewer
/// than the specified number of non-zero entries. The operation is performed
/// in-place on the data file.
pub fn run_squeeze(cmd_args: &RunSqueezeArgs) -> anyhow::Result<()> {
    let (backend, data_file) = resolve_backend_file(&cmd_args.data_file, None)?;
    let row_nnz_cutoff = cmd_args.row_nnz_cutoff;
    let col_nnz_cutoff = cmd_args.column_nnz_cutoff;

    let data = open_sparse_matrix(&data_file, &backend)?;

    info!(
        "before squeeze -- data: {} rows x {} columns",
        data.num_rows().unwrap(),
        data.num_columns().unwrap()
    );

    squeeze_by_nnz(
        data.as_ref(),
        SqueezeCutoffs {
            row: row_nnz_cutoff,
            column: col_nnz_cutoff,
        },
        cmd_args.block_size,
    )?;

    let data = open_sparse_matrix(&data_file, &backend)?;

    info!(
        "after squeeze -- data: {} rows x {} columns",
        data.num_rows().unwrap(),
        data.num_columns().unwrap()
    );

    Ok(())
}
