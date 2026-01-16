use matrix_util::common_io::*;
use crate::misc::*;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{read_col_names, MAX_ROW_NAME_IDX, MAX_COLUMN_NAME_IDX};
use crate::utilities::name_matching::match_by_substring;
use matrix_util::traits::IoOps;

// Import the argument types from main
use crate::{TakeColumnNamesArgs, TakeRowNamesArgs, InfoArgs, TakeColumnsArgs, TakeRowsArgs};

pub fn take_column_names(cmd_args: &TakeColumnNamesArgs) -> anyhow::Result<()> {

    let output = cmd_args.output.clone();
    let (backend, input) = resolve_backend_file(&cmd_args.data_file, None)?;
    let data = open_sparse_matrix(&input, &backend)?;
    let col_names = data.column_names()?;
    write_lines(&col_names, &output)?;

    Ok(())
}

pub fn take_row_names(cmd_args: &TakeRowNamesArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    let (backend, input) = resolve_backend_file(&cmd_args.data_file, None)?;
    let data = open_sparse_matrix(&input, &backend)?;

    let row_names = data.row_names()?;
    write_lines(&row_names, &output)?;

    Ok(())
}

pub fn show_info(cmd_args: &InfoArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();

    let (backend, input) = resolve_backend_file(&cmd_args.data_file, None)?;

    let data = open_sparse_matrix(&input, &backend)?;

    if let (Some(nrow), Some(ncol), Some(nnz)) =
        (data.num_rows(), data.num_columns(), data.num_non_zeros())
    {
        println!("number_of_rows:\t{}", nrow);
        println!("number_of_columns:\t{}", ncol);
        println!("number_of_nonzeros:\t{}", nnz);
    }

    if !output.is_empty() {
        use {mkdir, write_lines};
        mkdir(&output)?;
        let row_names = data.row_names()?;
        let col_names = data.column_names()?;
        write_lines(&row_names, &(output.to_string() + ".rows.gz"))?;
        write_lines(&col_names, &(output.to_string() + ".columns.gz"))?;
    }

    Ok(())
}

pub fn take_columns(args: &TakeColumnsArgs) -> anyhow::Result<()> {
    let columns = args.column_indices.clone();
    let column_name_file = args.name_file.clone();
    let column_names_arg = args.column_names.clone();

    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;

    let output = args.output.clone();

    let data = open_sparse_matrix(&data_file, &backend)?;
    let row_names = data.row_names()?;

    let (data, column_names) = if let Some(columns) = columns {
        let n_columns = data.num_columns().unwrap_or(0);
        let columns: Vec<usize> = columns.into_iter().filter(|&i| i < n_columns).collect();

        if columns.is_empty() {
            return Err(anyhow::anyhow!("invalid indexes"));
        }

        let _names = data.column_names()?;
        let column_names: Vec<Box<str>> = columns.iter().map(|&i| _names[i].clone()).collect();

        (data.read_columns_ndarray(columns)?, column_names)
    } else if let Some(column_file) = column_name_file {
        let col_names_to_match = read_col_names(column_file, MAX_COLUMN_NAME_IDX)?;
        let all_names = data.column_names()?;
        let (matched_indices, column_names) = match_by_substring(&all_names, &col_names_to_match, "column")?;
        (data.read_columns_ndarray(matched_indices)?, column_names)
    } else if let Some(col_names_to_match) = column_names_arg {
        let all_names = data.column_names()?;
        let (matched_indices, column_names) = match_by_substring(&all_names, &col_names_to_match, "column")?;
        (data.read_columns_ndarray(matched_indices)?, column_names)
    } else {
        return Err(anyhow::anyhow!(
            "either `column-indices`, `name-file`, or `column-names` must be provided"
        ));
    };

    if let Ok(ext) = file_ext(&output) {
        if ext.as_ref() == "parquet" {
            data.to_parquet(Some(&row_names), Some(&column_names), &output)?;
            return Ok(());
        }
    }

    data.to_tsv(&output)?;

    Ok(())
}

pub fn take_rows(args: &TakeRowsArgs) -> anyhow::Result<()> {
    let rows = args.row_indices.clone();
    let row_name_file = args.name_file.clone();
    let row_names_arg = args.row_names.clone();

    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;

    let output = args.output.clone();

    let data_backend = open_sparse_matrix(&data_file, &backend)?;

    let (data, row_names) = if let Some(rows) = rows {
        let n_rows = data_backend.num_rows().unwrap_or(0);
        let rows: Vec<usize> = rows.into_iter().filter(|&i| i < n_rows).collect();

        if rows.is_empty() {
            return Err(anyhow::anyhow!("invalid indexes"));
        }

        let _names = data_backend.row_names()?;
        let row_names: Vec<Box<str>> = rows.iter().map(|&i| _names[i].clone()).collect();

        (data_backend.read_rows_ndarray(rows)?, row_names)
    } else if let Some(row_name_file) = row_name_file {
        let row_names_to_match = read_col_names(row_name_file, MAX_ROW_NAME_IDX)?;
        let all_names = data_backend.row_names()?;
        let (matched_indices, row_names) = match_by_substring(&all_names, &row_names_to_match, "row")?;
        (data_backend.read_rows_ndarray(matched_indices)?, row_names)
    } else if let Some(row_names_to_match) = row_names_arg {
        let all_names = data_backend.row_names()?;
        let (matched_indices, row_names) = match_by_substring(&all_names, &row_names_to_match, "row")?;
        (data_backend.read_rows_ndarray(matched_indices)?, row_names)
    } else {
        return Err(anyhow::anyhow!(
            "either `row-indices`, `name-file`, or `row-names` must be provided"
        ));
    };

    let data_t = data.t().to_owned();

    if let Ok(ext) = file_ext(&output) {
        if ext.as_ref() == "parquet" {
            let column_names = data_backend.column_names()?;
            data_t.to_parquet(Some(&column_names), Some(&row_names), &output)?;
            return Ok(());
        }
    }

    data_t.to_tsv(&output)?;
    Ok(())
}
