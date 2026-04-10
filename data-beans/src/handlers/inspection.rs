use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{read_col_names, MAX_COLUMN_NAME_IDX, MAX_ROW_NAME_IDX};
use crate::utilities::name_matching::match_by_substring;
use clap::Args;
use matrix_util::common_io::*;
use matrix_util::traits::IoOps;
use ndarray::Array2;
use std::io::Write;

/// A quick information of the underlying matrix of a backend file.
#[derive(Args, Debug)]
pub struct InfoArgs {
    /// data file -- .zarr or .h5 file
    pub data_file: Box<str>,

    /// file header for {output}.{rows.gz,columns.gz}
    #[arg(short, long, default_value = "")]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
pub struct TakeColumnsArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// column indices to take: e.g., `0,1,2,3`
    #[arg(short = 'i', long, value_delimiter = ',')]
    pub column_indices: Option<Vec<usize>>,

    /// column name file where each line is a column name
    #[arg(short = 'f', long)]
    pub name_file: Option<Box<str>>,

    /// column names to take: e.g., `col1,col2,col3` (supports substring matching)
    #[arg(short = 'n', long, value_delimiter = ',')]
    pub column_names: Option<Vec<Box<str>>>,

    /// output `parquet` file
    #[arg(short, long, default_value = "stdout")]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
pub struct TakeRowsArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// row indices to take: e.g., `0,1,2,3`
    #[arg(short = 'i', long, value_delimiter = ',')]
    pub row_indices: Option<Vec<usize>>,

    /// row name file where each line is a row name
    #[arg(short = 'f', long)]
    pub name_file: Option<Box<str>>,

    /// row names to take: e.g., `gene1,gene2,gene3` (supports substring matching)
    #[arg(short = 'n', long, value_delimiter = ',')]
    pub row_names: Option<Vec<Box<str>>>,

    /// output `parquet` file
    #[arg(short, long, default_value = "stdout")]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
pub struct TakeColumnNamesArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// output file
    #[arg(short, long, default_value = "stdout")]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
pub struct TakeRowNamesArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// output file
    #[arg(short, long, default_value = "stdout")]
    pub output: Box<str>,
}

/// Write an Array2 as TSV with optional row/column names.
/// Arguments mirror `to_parquet_with_names`:
/// - `row_names`: `(Option<&[Box<str>]>, Option<&str>)` — names and label for the row dimension
/// - `column_names`: `Option<&[Box<str>]>` — column headers
fn write_named_tsv(
    output: &str,
    data: &Array2<f32>,
    row_names: (Option<&[Box<str>]>, Option<&str>),
    column_names: Option<&[Box<str>]>,
) -> anyhow::Result<()> {
    let nrows = data.nrows();
    let row_label = row_names.1.unwrap_or("row");
    let row_names: Vec<Box<str>> = match row_names.0 {
        Some(names) => names.to_vec(),
        None => (0..nrows).map(|i| format!("{}", i).into()).collect(),
    };
    let mut w = open_buf_writer(output)?;
    // Header
    write!(w, "{}", row_label)?;
    if let Some(col_names) = column_names {
        for name in col_names {
            write!(w, "\t{}", name)?;
        }
    } else {
        for j in 0..data.ncols() {
            write!(w, "\t{}", j)?;
        }
    }
    writeln!(w)?;
    // Data rows
    for (i, row) in data.rows().into_iter().enumerate() {
        write!(w, "{}", row_names[i])?;
        for val in row.iter() {
            write!(w, "\t{}", val)?;
        }
        writeln!(w)?;
    }
    Ok(())
}

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
        let (matched_indices, column_names) =
            match_by_substring(&all_names, &col_names_to_match, "column")?;
        (data.read_columns_ndarray(matched_indices)?, column_names)
    } else if let Some(col_names_to_match) = column_names_arg {
        let all_names = data.column_names()?;
        let (matched_indices, column_names) =
            match_by_substring(&all_names, &col_names_to_match, "column")?;
        (data.read_columns_ndarray(matched_indices)?, column_names)
    } else {
        return Err(anyhow::anyhow!(
            "either `column-indices`, `name-file`, or `column-names` must be provided"
        ));
    };

    if let Ok(ext) = file_ext(&output) {
        if ext.as_ref() == "parquet" {
            data.to_parquet_with_names(
                &output,
                (Some(&row_names), Some("row_name")),
                Some(&column_names),
            )?;
            return Ok(());
        }
    }

    write_named_tsv(
        &output,
        &data,
        (Some(&row_names), Some("row_name")),
        Some(&column_names),
    )?;

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
        let (matched_indices, row_names) =
            match_by_substring(&all_names, &row_names_to_match, "row")?;
        (data_backend.read_rows_ndarray(matched_indices)?, row_names)
    } else if let Some(row_names_to_match) = row_names_arg {
        let all_names = data_backend.row_names()?;
        let (matched_indices, row_names) =
            match_by_substring(&all_names, &row_names_to_match, "row")?;
        (data_backend.read_rows_ndarray(matched_indices)?, row_names)
    } else {
        return Err(anyhow::anyhow!(
            "either `row-indices`, `name-file`, or `row-names` must be provided"
        ));
    };

    let data_t = data.t().to_owned();
    let column_names = data_backend.column_names()?;

    if let Ok(ext) = file_ext(&output) {
        if ext.as_ref() == "parquet" {
            data_t.to_parquet_with_names(
                &output,
                (Some(&column_names), Some("column_name")),
                Some(&row_names),
            )?;
            return Ok(());
        }
    }

    write_named_tsv(
        &output,
        &data_t,
        (Some(&column_names), Some("column_name")),
        Some(&row_names),
    )?;
    Ok(())
}
