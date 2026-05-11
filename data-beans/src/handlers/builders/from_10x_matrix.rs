use super::run_squeeze_if_needed;
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::utilities::name_matching::{
    compose_id_name, filter_row_indices_by_type, make_names_unique,
};
use data_beans::zarr_io::*;

use clap::Args;
use log::info;
use matrix_util::common_io::*;

#[derive(Args, Debug)]
pub struct From10xMatrixArgs {
    #[arg(
        help = "Input HDF5 file containing sparse matrix triplets",
        long_help = "Specify the HDF5 file where triplets of sparse matrix data are stored. \n\
		     Supports 10X Genomics and H5AD formats."
    )]
    pub h5_file: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output",
        long_help = "Choose the backend format for the output file. \n\
		     Supported formats include 'zarr' and 'h5'"
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        help = "Output file header or name",
        long_help = "Specify the output file header. \n\
		     The output will be named as {output}.{backend}.\n\
		     With --zip, zarr backend produces {output}.zarr.zip instead of {output}.zarr.\n\
		     Redundant {backend} names will be ignored."
    )]
    pub output: Box<str>,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,

    #[arg(
        short = 'x',
        long,
        default_value = "matrix",
        help = "Root group name for sparse data triplets",
        long_help = "Set the root group name under which sparse data triplets are stored in the HDF5 file. \n\
		     Use the 'list-h5' command to inspect available groups."
    )]
    pub root_group_name: Box<str>,

    #[arg(
        short = 'd',
        long,
        default_value = "data",
        help = "Data field name",
        long_help = "Name of the dataset containing triplet values X(i,j) under the root group."
    )]
    pub data_field: Box<str>,

    #[arg(
        short = 'i',
        long,
        default_value = "indices",
        help = "Indices field name",
        long_help = "Name of the dataset containing indices. \n\
		     Row indices for CSC, column indices for CSR, under the root group."
    )]
    pub indices_field: Box<str>,

    #[arg(
        short = 'p',
        long,
        default_value = "indptr",
        help = "Indptr field name",
        long_help = "Name of the dataset containing indptr. \n\
		     Column pointers for CSC, row pointers for CSR, under the root group."
    )]
    pub indptr_field: Box<str>,

    #[arg(
        short = 't',
        long,
        value_enum,
        default_value = "column",
        help = "Pointer type (row or column)",
        long_help = "Specify whether the pointers are for row (gene) or column (cell) indices."
    )]
    pub pointer_type: IndexPointerType,

    #[arg(
        short = 'r',
        long,
        default_value = "features/id",
        help = "Row ID field name",
        long_help = "Group or dataset name for row, gene, or feature IDs under the root group."
    )]
    pub row_id_field: Box<str>,

    #[arg(
        short = 'n',
        long,
        default_value = "features/name",
        help = "Row name field name",
        long_help = "Group or dataset name for row, gene, or feature names under the root group."
    )]
    pub row_name_field: Box<str>,

    #[arg(
        short = 'f',
        long,
        default_value = "features/feature_type",
        help = "Row type field name",
        long_help = "Group or dataset name for row, gene, or feature types under the root group."
    )]
    pub row_type_field: Box<str>,

    #[arg(
        long,
        default_value = "gene,peak",
        help = "Select row type (comma-separated patterns; ANY match keeps the row)",
        long_help = "Select row type for inclusion. Comma-separated case-insensitive substrings; \n\
		     a row is kept if its type contains any pattern. Default 'gene,peak' \n\
		     keeps both Gene Expression and ATAC Peaks."
    )]
    pub select_row_type: Box<str>,

    #[arg(
        long,
        default_value = "aggregate",
        help = "Remove row type (comma-separated patterns; ANY match drops the row)",
        long_help = "Remove rows if their type contains any of these comma-separated patterns."
    )]
    pub remove_row_type: Box<str>,

    #[arg(
        short = 'c',
        long,
        default_value = "barcodes",
        help = "Column name field",
        long_help = "Group or dataset name for columns or cells under the root group."
    )]
    pub column_name_field: Box<str>,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows or columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros. \n\
		     This can help reduce file size and improve performance."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows. \n\
		     Rows with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns. \n\
		     Columns with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        help = "Cells per rayon job for the post-build squeeze pass \
                (omit for auto-scaling by feature count)"
    )]
    pub block_size: Option<usize>,
}
pub fn run_build_from_10x_matrix(args: &From10xMatrixArgs) -> anyhow::Result<()> {
    let file = hdf5::File::open(args.h5_file.to_string())?;
    info!("Opened 10X H5 file: {}", args.h5_file);
    let effective_output = apply_zip_flag(&args.output, args.zip);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;

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

    let CooTripletsShape { triplets, shape } =
        if let (Ok(values_ds), Ok(indices_ds), Ok(indptr_ds)) = (
            root.dataset(args.data_field.as_ref()),
            root.dataset(args.indices_field.as_ref()),
            root.dataset(args.indptr_field.as_ref()),
        ) {
            let values = values_ds.read_1d::<f32>()?;
            let indices = indices_ds.read_1d::<u64>()?;
            let indptr = indptr_ds.read_1d::<u64>()?;
            ValuesIndicesPointers {
                values: values.as_slice().expect("values not contiguous"),
                indices: indices.as_slice().expect("indices not contiguous"),
                indptr: indptr.as_slice().expect("indptr not contiguous"),
            }
            .to_coo(args.pointer_type)?
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

    let mut row_ids = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_ids);

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

    let mut out = create_sparse_from_triplets_owned(
        triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("created sparse matrix: {}", backend_file);
    out.register_row_names_vec(&row_ids);
    out.register_column_names_vec(&column_names);

    let select_rows =
        filter_row_indices_by_type(&row_types, &args.select_row_type, &args.remove_row_type);

    if select_rows.len() < nrows {
        info!("filtering in features of `{}` type", args.select_row_type);
        out.subset_columns_rows(None, Some(&select_rows))?;
    }

    run_squeeze_if_needed(
        args.do_squeeze,
        args.row_nnz_cutoff,
        args.column_nnz_cutoff,
        args.block_size,
        &backend_file,
    )?;
    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}
