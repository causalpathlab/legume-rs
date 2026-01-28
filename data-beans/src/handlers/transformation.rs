use crate::misc::*;
use crate::qc::*;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{read_row_names, read_col_names, MAX_ROW_NAME_IDX, MAX_COLUMN_NAME_IDX};

use matrix_util::common_io::*;
use matrix_util::membership::Membership;
use log::info;

// Import the argument structs from main.rs
// These will need to be public in main.rs
use crate::{SubsetColumnsArgs, SubsetRowsArgs, ReorderRowsArgs, RunSqueezeArgs};

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

    let original_ncol = data.num_columns().unwrap_or(0);
    info!("original data: {} columns", original_ncol);

    let selected_columns = if let Some(idx) = columns_indices {
        info!("using {} column indices directly", idx.len());
        idx
    } else if let Some(column_file) = column_name_file {
        let col_names = read_col_names(column_file, MAX_COLUMN_NAME_IDX)?;
        if col_names.is_empty() {
            return Err(anyhow::anyhow!("Empty column file"));
        }
        info!("read {} column names from file", col_names.len());

        let data_col_names = data
            .column_names()
            .expect("column names not found in data file");

        // Build membership from column names in file (these are the ones we want to keep)
        // The keys are the column names from the file, values are their indices
        let membership = Membership::from_pairs(
            col_names.iter().enumerate().map(|(i, name)| (name.clone(), i.to_string().into_boxed_str())),
            args.allow_prefix,
        ).with_delimiter(args.delimiter);

        // Match data columns against the membership to find which ones to keep
        let (matched, stats) = membership.match_keys(&data_col_names);

        info!(
            "Column matching: {} exact + {} base_key + {} prefix = {}/{} matched",
            stats.exact,
            stats.base_key,
            stats.prefix,
            stats.total_matched(),
            data_col_names.len()
        );

        if matched.is_empty() {
            let file_sample: Vec<_> = col_names.iter().take(3).collect();
            let data_sample: Vec<_> = data_col_names.iter().take(3).collect();
            info!("column names from file (sample): {:?}", file_sample);
            info!("column names in data (sample): {:?}", data_sample);
            return Err(anyhow::anyhow!("Found empty columns - no matching column names"));
        }

        // Build index list: for each data column that matched, include its index
        let idx: Vec<usize> = data_col_names
            .iter()
            .enumerate()
            .filter_map(|(i, name)| matched.get(name).map(|_| i))
            .collect();

        info!("subsetting to {} columns", idx.len());
        idx
    } else {
        return Err(anyhow::anyhow!(
            "either `column-indices` or `name-file` must be provided"
        ));
    };

    data.subset_columns_rows(Some(&selected_columns), None)?;

    // Verify by re-opening the file
    drop(data);
    let data = open_sparse_matrix(&output_file, &backend)?;
    let new_ncol = data.num_columns().unwrap_or(0);
    info!("after subset: {} columns (verified)", new_ncol);

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

/// Subset rows from a sparse matrix data file
///
/// This function takes a subset of rows from the input data file and writes
/// them to a new output file. Rows can be specified either by indices or by
/// a file containing row names. Optionally, the output can be squeezed to
/// remove rows/columns with too few non-zero entries.
pub fn subset_rows(args: &SubsetRowsArgs) -> anyhow::Result<()> {
    let row_indices = args.row_indices.clone();
    let row_name_file = args.name_file.clone();

    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;

    let (_, output_file) = resolve_backend_file(&args.output, Some(backend.clone()))?;

    if let Some(out_dir) = dirname(&output_file) {
        mkdir(&out_dir)?;
    }

    recursive_copy(&data_file, &output_file)?;
    info!("copied the existing data file {}", data_file);

    let mut data = open_sparse_matrix(&output_file, &backend)?;

    let original_nrow = data.num_rows().unwrap_or(0);
    info!("original data: {} rows", original_nrow);

    let selected_rows = if let Some(idx) = row_indices {
        info!("using {} row indices directly", idx.len());
        idx
    } else if let Some(row_file) = row_name_file {
        let row_names = read_row_names(row_file, MAX_ROW_NAME_IDX)?;
        if row_names.is_empty() {
            return Err(anyhow::anyhow!("Empty row file"));
        }
        info!("read {} row names from file", row_names.len());

        let data_row_names = data
            .row_names()
            .expect("row names not found in data file");

        // Build membership from row names in file (these are the ones we want to keep)
        let membership = Membership::from_pairs(
            row_names.iter().enumerate().map(|(i, name)| (name.clone(), i.to_string().into_boxed_str())),
            args.allow_prefix,
        ).with_delimiter(args.delimiter);

        // Match data rows against the membership to find which ones to keep
        let (matched, stats) = membership.match_keys(&data_row_names);

        info!(
            "Row matching: {} exact + {} base_key + {} prefix = {}/{} matched",
            stats.exact,
            stats.base_key,
            stats.prefix,
            stats.total_matched(),
            data_row_names.len()
        );

        if matched.is_empty() {
            let file_sample: Vec<_> = row_names.iter().take(3).collect();
            let data_sample: Vec<_> = data_row_names.iter().take(3).collect();
            info!("row names from file (sample): {:?}", file_sample);
            info!("row names in data (sample): {:?}", data_sample);
            return Err(anyhow::anyhow!("Found empty rows - no matching row names"));
        }

        // Build index list: for each data row that matched, include its index
        let idx: Vec<usize> = data_row_names
            .iter()
            .enumerate()
            .filter_map(|(i, name)| matched.get(name).map(|_| i))
            .collect();

        info!("subsetting to {} rows", idx.len());
        idx
    } else {
        return Err(anyhow::anyhow!(
            "either `row-indices` or `name-file` must be provided"
        ));
    };

    info!("subsetting to {} rows", selected_rows.len());
    data.subset_columns_rows(None, Some(&selected_rows))?;

    // Verify by re-opening the file
    drop(data);
    let data = open_sparse_matrix(&output_file, &backend)?;
    let new_nrow = data.num_rows().unwrap_or(0);
    info!("after subset: {} rows (verified)", new_nrow);

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
