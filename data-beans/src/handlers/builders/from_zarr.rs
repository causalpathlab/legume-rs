use super::run_squeeze_if_needed;
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::utilities::name_matching::{
    compose_id_name, filter_row_indices_by_type, make_names_unique,
};
use data_beans::zarr_io::*;

use log::info;
use matrix_util::common_io::*;

#[derive(clap::Args, Debug)]
pub struct FromZarrArgs {
    #[arg(
        help = "Input Zarr file containing sparse matrix triplets",
        long_help = "Specify the Zarr file where triplets of sparse matrix data are stored. \n\
		     For example, 10X Genomics Xenium's 'cell_feature_matrix.zarr.zip'."
    )]
    pub zarr_file: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output",
        long_help = "Choose the backend format for the output file. \n\
		     Supported formats include 'zarr' and 'h5'."
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
        short = 'd',
        long,
        default_value = "/cell_features/data",
        help = "Data field path",
        long_help = "Path to the dataset containing triplet values. \n\
		     Use the 'list-zarr' subcommand to inspect available fields."
    )]
    pub data_field: Box<str>,

    #[arg(
        short = 'i',
        long,
        default_value = "/cell_features/indices",
        help = "Indices field path",
        long_help = "Path to the dataset containing indices. \n\
		     Row indices for CSC, column indices for CSR."
    )]
    pub indices_field: Box<str>,

    #[arg(
        short = 'p',
        long,
        default_value = "/cell_features/indptr",
        help = "Indptr field path",
        long_help = "Path to the dataset containing indptr. \n\
		     Column pointers for CSC, row pointers for CSR."
    )]
    pub indptr_field: Box<str>,

    #[arg(
        short = 't',
        long,
        value_enum,
        default_value = "row",
        help = "Pointer type (row or column)",
        long_help = "Specify whether the pointers keep track of row (gene) or column (cell) indices."
    )]
    pub pointer_type: IndexPointerType,

    #[arg(
        short = 'r',
        long,
        default_value = "/cell_features/feature_ids",
        help = "Row ID field path",
        long_help = "Path to the group or dataset for row, gene, or feature IDs."
    )]
    pub row_id_field: Box<str>,

    #[arg(
        short = 'n',
        long,
        default_value = "/cell_features/feature_keys",
        help = "Row name field path",
        long_help = "Path to the group or dataset for row, gene, or feature names."
    )]
    pub row_name_field: Box<str>,

    #[arg(
        short = 'f',
        long,
        default_value = "/cell_features/feature_types",
        help = "Row type field path",
        long_help = "Path to the group or dataset for row, gene, or feature types."
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
        default_value = "/cell_features/cell_id",
        help = "Column name field path",
        long_help = "Path to the group or dataset for columns or cells. \n\
		     Will first attempt Xenium's Cell ID format mapping."
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
pub fn run_build_from_zarr_triplets(args: &FromZarrArgs) -> anyhow::Result<()> {
    let source_zarr_file_path = args.zarr_file.clone();

    let effective_output = apply_zip_flag(&args.output, args.zip);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        remove_file(&backend_file)?;
    }

    let store = open_zarr_store(&source_zarr_file_path)?;
    info!("Opened zarr store: {}", source_zarr_file_path);

    let indices: Vec<u64> = read_zarr_numerics(store.clone(), args.indices_field.as_ref())?;
    let indptr: Vec<u64> = read_zarr_numerics(store.clone(), args.indptr_field.as_ref())?;
    let values: Vec<f32> = read_zarr_numerics(store.clone(), args.data_field.as_ref())?;
    info!("Read the arrays");

    let CooTripletsShape { triplets, shape } = ValuesIndicesPointers {
        values: &values,
        indices: &indices,
        indptr: &indptr,
    }
    .to_coo(args.pointer_type)?;

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
    let mut row_ids = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_ids);

    let mut row_types = read_zarr_group_attr::<Vec<Box<str>>>(store.clone(), &args.row_type_field)
        .or_else(|_| read_zarr_strings(store.clone(), args.row_type_field.as_ref()))
        .unwrap_or_else(|_| vec![args.select_row_type.clone(); nrows]);
    if nrows < row_types.len() {
        info!("data doesn't contain all the rows");
        row_types.truncate(nrows);
    }
    assert_eq!(nrows, row_types.len());

    let select_rows =
        filter_row_indices_by_type(&row_types, &args.select_row_type, &args.remove_row_type);

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

    let mut out = create_sparse_from_triplets_owned(
        triplets,
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

