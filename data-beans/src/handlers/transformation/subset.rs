use super::{run_squeeze, ReorderRowsArgs, RowAlignMode, RunSqueezeArgs};
use super::{SubsetColumnsArgs, SubsetRowsArgs};
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{
    read_col_names, read_row_names, MAX_COLUMN_NAME_IDX, MAX_ROW_NAME_IDX,
};
use crate::zarr_io::{finalize_zarr_output, materialize_writable_backend};

use log::info;
use matrix_util::common_io::*;
use matrix_util::membership::Membership;

type EditStage = (
    SparseIoBackend,
    Box<str>,
    Box<dyn SparseIo<IndexIter = Vec<usize>>>,
);

/// Resolve `(backend, working_path, opened_data)` for an in-place edit handler:
/// stage the input at the output location (extracting `.zarr.zip` if needed),
/// mkdir the output parent, and open the backend for read/write.
///
/// Pair with [`finalize_zarr_output`] at the end of the handler so `.zarr.zip`
/// output paths get re-zipped.
fn stage_for_edit(input: &str, output: &str) -> anyhow::Result<EditStage> {
    let (backend, data_file) = resolve_backend_file(input, None)?;
    let (_, output_file) = resolve_backend_file(output, Some(backend.clone()))?;

    if let Some(out_dir) = dirname(&output_file) {
        mkdir(&out_dir)?;
    }

    materialize_writable_backend(&data_file, &output_file)?;
    info!("staged working copy at {}", output_file);

    let data = open_sparse_matrix(&output_file, &backend)?;
    Ok((backend, output_file, data))
}

fn build_squeeze_args(output_file: Box<str>, args_cols: usize, args_rows: usize) -> RunSqueezeArgs {
    RunSqueezeArgs {
        data_files: vec![output_file],
        row_nnz_cutoff: args_rows,
        column_nnz_cutoff: args_cols,
        block_size: None,
        preload: true,
        show_histogram: false,
        save_histogram: None,
        dry_run: false,
        interactive: false,
        auto_cutoff: false,
        output: None,
        row_align: RowAlignMode::Common,
    }
}

/// Subset columns from a sparse matrix data file
///
/// This function takes a subset of columns from the input data file and writes
/// them to a new output file. Columns can be specified either by indices or by
/// a file containing column names. Optionally, the output can be squeezed to
/// remove rows/columns with too few non-zero entries.
pub fn subset_columns(args: &SubsetColumnsArgs) -> anyhow::Result<()> {
    let columns_indices = args.column_indices.clone();
    let column_name_file = args.name_file.clone();

    let (_backend, output_file, mut data) = stage_for_edit(&args.data_file, &args.output)?;

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

        let membership = Membership::from_pairs(
            col_names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i.to_string().into_boxed_str())),
            args.allow_prefix,
        )
        .with_delimiter(args.delimiter);

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
            return Err(anyhow::anyhow!(
                "Found empty columns - no matching column names"
            ));
        }

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
    info!("after subset: {} columns", selected_columns.len());
    drop(data);

    if args.do_squeeze {
        info!("Squeeze the backend data {}", &output_file);
        let squeeze_args = build_squeeze_args(
            output_file.clone(),
            args.column_nnz_cutoff,
            args.row_nnz_cutoff,
        );
        run_squeeze(&squeeze_args)?;
    }

    finalize_zarr_output(&output_file, &args.output)?;
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

    let (_backend, output_file, mut data) = stage_for_edit(&args.data_file, &args.output)?;

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

        let data_row_names = data.row_names().expect("row names not found in data file");

        let membership = Membership::from_pairs(
            row_names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i.to_string().into_boxed_str())),
            args.allow_prefix,
        )
        .with_delimiter(args.delimiter);

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

    data.subset_columns_rows(None, Some(&selected_rows))?;
    info!("after subset: {} rows", selected_rows.len());
    drop(data);

    if args.do_squeeze {
        info!("Squeeze the backend data {}", &output_file);
        let squeeze_args = build_squeeze_args(
            output_file.clone(),
            args.column_nnz_cutoff,
            args.row_nnz_cutoff,
        );
        run_squeeze(&squeeze_args)?;
    }

    finalize_zarr_output(&output_file, &args.output)?;
    info!("done");
    Ok(())
}

/// Reorder rows in a sparse matrix data file
///
/// This function reorders the rows of the input data file according to the
/// order specified in a row name file, and writes the result to a new output file.
pub fn reorder_rows(args: &ReorderRowsArgs) -> anyhow::Result<()> {
    let row_names_order: Vec<Box<str>> = read_row_names(args.row_file.clone(), MAX_ROW_NAME_IDX)?;

    let (_backend, output_file, mut data) = stage_for_edit(&args.data_file, &args.output)?;

    data.reorder_rows(&row_names_order)?;
    drop(data);

    finalize_zarr_output(&output_file, &args.output)?;
    info!("done");
    Ok(())
}
