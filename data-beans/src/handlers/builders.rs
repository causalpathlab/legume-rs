use crate::handlers::transformation::{run_squeeze, RowAlignMode, RunSqueezeArgs};
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
pub struct FromMtxArgs {
    /// matrix market-formatted data file (`.mtx.gz` or `.mtx`)
    pub mtx: Box<str>,

    /// row/feature name file (name per each line; `.tsv.gz` or `.tsv`).
    /// For 10x `features.tsv[.gz]` the columns are `id`, `name`, `feature_type`;
    /// the third column, when present, is used for `--select-row-type` /
    /// `--remove-row-type` / `--hto-row-type` filtering.
    #[arg(short, long)]
    pub row: Option<Box<str>>,

    /// column/cell/barcode file (name per each line; `.tsv.gz` or `.tsv`)
    #[arg(short, long)]
    pub col: Option<Box<str>>,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    pub backend: SparseIoBackend,

    /// output file header: {output}.{backend} (or {output}.zarr.zip with --zip)
    #[arg(short, long)]
    pub output: Box<str>,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,

    /// maximum number of columns to read from the row/feature name file
    /// (columns are joined with '_'); e.g. 2 reads "ENSG…<tab>SYMBOL"
    #[arg(long, default_value_t = 2)]
    pub row_name_columns: usize,

    /// Keep only rows whose feature_type (column 3 of `--row`) contains this
    /// string. Default `Gene Expression` matches the 10x mtx default. Set to
    /// empty to keep everything.
    #[arg(long, default_value = "Gene Expression")]
    pub select_row_type: Box<str>,

    /// Drop rows whose feature_type contains this string. Applied after
    /// `--select-row-type`.
    #[arg(long, default_value = "")]
    pub remove_row_type: Box<str>,

    /// Feature type used as HTO/multiplexing tags (e.g. `Antibody Capture`).
    /// When set, each cell is assigned a batch label `barcode@{id}_{name}`
    /// based on the HTO row with the highest count. HTO rows are then
    /// removed from the final backend and cells with zero HTO signal are
    /// dropped.
    #[arg(long, default_value = "")]
    pub hto_row_type: Box<str>,

    /// squeeze
    #[arg(long, default_value_t = false)]
    pub do_squeeze: bool,

    /// minimum number of non-zero cutoff for rows
    #[arg(long, default_value_t = 1)]
    pub row_nnz_cutoff: usize,

    /// minimum number of non-zero cutoff for columns
    #[arg(long, default_value_t = 1)]
    pub column_nnz_cutoff: usize,

    /// block size for the post-build squeeze pass (columns per rayon job).
    /// Smaller values bound peak memory on very wide matrices at the cost of
    /// extra scheduling overhead.
    #[arg(long, default_value_t = 100)]
    pub block_size: usize,
}

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
        default_value = "gene",
        help = "Select row type",
        long_help = "Select row type for inclusion. Rows are included if their type contains this value."
    )]
    pub select_row_type: Box<str>,

    #[arg(
        long,
        default_value = "aggregate",
        help = "Remove row type",
        long_help = "Remove rows if their type contains this value."
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
        default_value_t = 100,
        help = "Block size for the post-build squeeze pass",
        long_help = "Number of columns handled per rayon job during the squeeze pass. \n\
		     Smaller values bound peak memory on very wide matrices at the cost \n\
		     of extra scheduling overhead."
    )]
    pub block_size: usize,
}

#[derive(Args, Debug)]
pub struct FromH5adArgs {
    #[arg(
        help = "Input AnnData h5ad file",
        long_help = "Specify the AnnData h5ad file (CELLxGENE schema v7).\n\
		     Expected layout: X/{data,indices,indptr}, obs/, var/.\n\
		     Prefers raw/X over X when available."
    )]
    pub h5ad_file: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output",
        long_help = "Choose the backend format for the output file."
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        help = "Output file header or name",
        long_help = "Specify the output file header.\n\
		     The output will be named as {output}.{backend}.\n\
		     With --zip, zarr backend produces {output}.zarr.zip instead of {output}.zarr.\n\
		     Metadata files will be named {output}.cell_metadata.tsv.gz, etc."
    )]
    pub output: Box<str>,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,

    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = [Box::<str>::from("_index"), Box::from("gene_id")],
        help = "var/ field(s) for feature ID (fallback list)",
        long_help = "Comma-separated list of var/ dataset names to try for the feature ID (e.g. Ensembl ID).\n\
		     The first one found is used. Default: '_index,gene_id'."
    )]
    pub row_id_field: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = [Box::<str>::from("feature_name"), Box::from("gene_name")],
        help = "var/ field(s) for gene symbol (fallback list)",
        long_help = "Comma-separated list of var/ column names to try for the gene symbol.\n\
		     The first column found is used. Joined with the ID to form 'ID_SYMBOL' row names.\n\
		     Default: 'feature_name,gene_name'."
    )]
    pub row_name_field: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = [Box::<str>::from("_index"), Box::from("cell")],
        help = "obs/ field(s) for cell barcode (fallback list)",
        long_help = "Comma-separated list of obs/ dataset names to try for the cell barcode.\n\
		     The first one found is used. Default: '_index,cell'."
    )]
    pub col_name_field: Vec<Box<str>>,

    #[arg(
        long,
        default_value = "",
        help = "Select row type (biotype) for filtering",
        long_help = "Filter features by biotype (e.g., 'protein_coding', 'lncRNA').\n\
		     Leave empty to keep all features (default).\n\
		     CELLxGENE uses biotype annotations rather than 'Gene Expression'."
    )]
    pub select_row_type: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Remove row type",
        long_help = "Remove rows if their biotype contains this value."
    )]
    pub remove_row_type: Box<str>,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows or columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for the post-build squeeze pass",
        long_help = "Number of columns handled per rayon job during the squeeze pass. \n\
		     Smaller values bound peak memory on very wide matrices at the cost \n\
		     of extra scheduling overhead."
    )]
    pub block_size: usize,
}

#[derive(Args, Debug)]
pub struct From10xMoleculeArgs {
    #[arg(
        help = "Input 10X molecule_info.h5 file",
        long_help = "Specify the molecule_info.h5 file from Cell Ranger count/multi.\n\
		     Contains per-molecule data: barcode_idx, feature_idx, count, gem_group, etc."
    )]
    pub h5_file: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output",
        long_help = "Choose the backend format for the output file."
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        help = "Output file header or name",
        long_help = "Specify the output file header.\n\
		     The output will be named as {output}.{backend}.\n\
		     With --zip, zarr backend produces {output}.zarr.zip instead of {output}.zarr."
    )]
    pub output: Box<str>,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,

    #[arg(
        long,
        default_value = "Gene Expression",
        help = "Library type to include",
        long_help = "Filter molecules to only those from libraries of this type.\n\
		     Common types: 'Gene Expression', 'Antibody Capture', 'CRISPR Guide Capture'.\n\
		     Reads library_info JSON to determine which library indices match."
    )]
    pub library_type: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Select row type (feature_type)",
        long_help = "Filter features by type. Rows are included if their type contains this value.\n\
		     Empty (default) keeps all features. 10X uses 'Gene Expression', 'Antibody Capture', etc."
    )]
    pub select_row_type: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Remove row type",
        long_help = "Remove rows if their type contains this value. Empty (default) removes nothing."
    )]
    pub remove_row_type: Box<str>,

    #[arg(
        long,
        default_value_t = false,
        help = "Skip pass_filter and include all barcodes",
        long_help = "By default, only barcodes that passed Cell Ranger cell calling are included.\n\
		     Set this flag to include ALL barcodes with at least one molecule."
    )]
    pub no_pass_filter: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows or columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for the post-build squeeze pass",
        long_help = "Number of columns handled per rayon job during the squeeze pass. \n\
		     Smaller values bound peak memory on very wide matrices at the cost \n\
		     of extra scheduling overhead."
    )]
    pub block_size: usize,
}

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
        default_value = "gene",
        help = "Select row type",
        long_help = "Select row type for inclusion. Rows are included if their type contains this value."
    )]
    pub select_row_type: Box<str>,

    #[arg(
        long,
        default_value = "aggregate",
        help = "Remove row type",
        long_help = "Remove rows if their type contains this value."
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
        default_value_t = 100,
        help = "Block size for the post-build squeeze pass",
        long_help = "Number of columns handled per rayon job during the squeeze pass. \n\
		     Smaller values bound peak memory on very wide matrices at the cost \n\
		     of extra scheduling overhead."
    )]
    pub block_size: usize,
}

pub fn run_build_from_mtx(args: &FromMtxArgs) -> anyhow::Result<()> {
    let mtx_file = args.mtx.as_ref();
    let row_file = args.row.as_ref();
    let col_file = args.col.as_ref();

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

    let mut data = create_sparse_from_mtx_file(mtx_file, Some(&backend_file), Some(&backend))?;

    // Row names + (optional) ids + feature types — parsed together from the
    // features.tsv file so we can support the 10x 3-column layout.
    //   col 1: id        col 2: name        col 3: feature_type (optional)
    let parsed_rows = if let Some(row_file) = row_file {
        let rows = read_mtx_feature_rows(row_file.as_ref(), args.row_name_columns)?;
        Some(rows)
    } else {
        None
    };

    if let Some(rows) = parsed_rows.as_ref() {
        let mut display = rows.build_display_names();
        make_names_unique(&mut display);
        data.register_row_names_vec(&display);
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

    let nrow_backend = data.num_rows().unwrap_or(0);
    let ncol_backend = data.num_columns().unwrap_or(0);
    if let Some(rows) = parsed_rows.as_ref() {
        if let Some(row_types) = rows.types.as_ref() {
            log_feature_type_histogram(mtx_file, row_types);

            // Single pass: classify each row as HTO, keep, or drop.
            let sel_lc = args.select_row_type.to_ascii_lowercase();
            let rem_lc = args.remove_row_type.to_ascii_lowercase();
            let hto_lc = args.hto_row_type.to_ascii_lowercase();
            let hto_enabled = !hto_lc.is_empty();

            let mut keep_rows: Vec<usize> = Vec::new();
            let mut hto_rows: Vec<usize> = Vec::new();
            for (i, t) in row_types.iter().enumerate() {
                let low = t.to_ascii_lowercase();
                if hto_enabled && low.contains(&hto_lc) {
                    hto_rows.push(i);
                    continue;
                }
                let selected = sel_lc.is_empty() || low.contains(&sel_lc);
                let removed = !rem_lc.is_empty() && low.contains(&rem_lc);
                if selected && !removed {
                    keep_rows.push(i);
                }
            }

            let keep_cols: Option<Vec<usize>> = if hto_enabled && !hto_rows.is_empty() {
                info!(
                    "HTO: {} multiplexing rows matching '{}'",
                    hto_rows.len(),
                    args.hto_row_type
                );

                // HTO matrices are tiny (<~20 rows), so dense is fine.
                let hto_dense = data.read_rows_dmatrix(hto_rows.clone())?;
                assert_eq!(hto_dense.nrows(), hto_rows.len());
                assert_eq!(hto_dense.ncols(), ncol_backend);

                struct HtoAssign {
                    col: usize,
                    hto_row: usize,
                    count: f32,
                }
                let mut assigns: Vec<HtoAssign> = Vec::with_capacity(ncol_backend);
                let mut zero_cells = 0usize;
                for j in 0..ncol_backend {
                    let col = hto_dense.column(j);
                    let mut best = 0usize;
                    let mut best_val = col[0];
                    let mut col_sum = 0.0f32;
                    for (k, &v) in col.iter().enumerate() {
                        col_sum += v;
                        if v > best_val {
                            best_val = v;
                            best = k;
                        }
                    }
                    if col_sum <= 0.0 {
                        zero_cells += 1;
                        continue;
                    }
                    assigns.push(HtoAssign {
                        col: j,
                        hto_row: hto_rows[best],
                        count: best_val,
                    });
                }

                let dropped_frac = if ncol_backend > 0 {
                    zero_cells as f64 / ncol_backend as f64
                } else {
                    0.0
                };
                info!(
                    "HTO: {}/{} cells have zero HTO signal and will be dropped ({:.1}%)",
                    zero_cells,
                    ncol_backend,
                    dropped_frac * 100.0
                );
                if dropped_frac > 0.5 {
                    log::warn!(
                        "HTO: dropped more than half of cells ({}/{}); \
                         check that the HTO library is consistently present",
                        zero_cells,
                        ncol_backend
                    );
                }

                let tag_for = |hto_row: usize| -> String {
                    let id = rows.ids[hto_row].as_ref();
                    let name = rows.names[hto_row].as_ref();
                    if name.is_empty() {
                        id.to_string()
                    } else {
                        format!("{}_{}", id, name)
                    }
                };

                let mut col_names = data.column_names()?;

                let output_stem = backend_file
                    .strip_suffix(".zarr")
                    .or_else(|| backend_file.strip_suffix(".h5"))
                    .unwrap_or(&backend_file);
                let hto_file = format!("{}.barcode_to_hto.tsv.gz", output_stem);
                {
                    use std::io::Write;
                    let mut w = open_buf_writer(&hto_file)?;
                    writeln!(w, "barcode\tid\tname\tcount")?;
                    for a in &assigns {
                        let bc = col_names[a.col].as_ref();
                        let id = rows.ids[a.hto_row].as_ref();
                        let name = rows.names[a.hto_row].as_ref();
                        writeln!(w, "{}\t{}\t{}\t{}", bc, id, name, a.count)?;
                    }
                    w.flush()?;
                }
                info!("Wrote {}", hto_file);

                for a in &assigns {
                    col_names[a.col] =
                        format!("{}@{}", col_names[a.col], tag_for(a.hto_row)).into_boxed_str();
                }
                data.register_column_names_vec(&col_names);

                Some(assigns.into_iter().map(|a| a.col).collect::<Vec<_>>())
            } else {
                if hto_enabled {
                    info!(
                        "HTO: no rows match feature type '{}'; skipping multiplexing",
                        args.hto_row_type
                    );
                }
                None
            };

            let need_row_subset = keep_rows.len() < nrow_backend;
            let need_col_subset = keep_cols.as_ref().is_some_and(|v| v.len() < ncol_backend);

            if need_row_subset || need_col_subset {
                info!(
                    "from-mtx subset: rows {} -> {}, cols {} -> {}",
                    nrow_backend,
                    keep_rows.len(),
                    ncol_backend,
                    keep_cols.as_ref().map(|v| v.len()).unwrap_or(ncol_backend),
                );
                data.subset_columns_rows(keep_cols.as_ref(), Some(&keep_rows))?;
            }
        }
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

struct MtxFeatureRows {
    ids: Vec<Box<str>>,
    names: Vec<Box<str>>,
    /// Tab-separated third column when present. `None` when the file had
    /// fewer than three columns (row-type filtering is then skipped).
    types: Option<Vec<Box<str>>>,
    row_name_columns: usize,
}

impl MtxFeatureRows {
    /// Build the composite `id{ROW_SEP}name{...}` display names used when
    /// registering rows on the backend. Derived on demand so the struct
    /// doesn't carry a third parallel vector.
    fn build_display_names(&self) -> Vec<Box<str>> {
        let take = self.row_name_columns.max(1);
        self.ids
            .iter()
            .zip(self.names.iter())
            .map(|(id, name)| {
                let id = id.as_ref();
                let name = name.as_ref();
                let joined = if take >= 2 && !name.is_empty() {
                    let mut s = String::with_capacity(id.len() + ROW_SEP.len() + name.len());
                    s.push_str(id);
                    s.push_str(ROW_SEP);
                    s.push_str(name);
                    s
                } else {
                    id.to_string()
                };
                joined.into_boxed_str()
            })
            .collect()
    }
}

fn read_mtx_feature_rows(
    row_file: &str,
    row_name_columns: usize,
) -> anyhow::Result<MtxFeatureRows> {
    use matrix_util::common_io::read_lines_of_words_delim;

    // 10x features.tsv is tab-separated and the 3rd column (feature_type)
    // can contain spaces ("Gene Expression", "Antibody Capture"), so
    // whitespace splitting would corrupt it.
    let lines = read_lines_of_words_delim(row_file, &['\t'], -1)?.lines;
    let n = lines.len();
    let mut ids = Vec::with_capacity(n);
    let mut names = Vec::with_capacity(n);
    let mut types = Vec::with_capacity(n);
    let mut any_type = false;

    for words in lines.into_iter() {
        let id: Box<str> = words.first().cloned().unwrap_or_else(|| "".into());
        let name: Box<str> = words.get(1).cloned().unwrap_or_else(|| "".into());
        let ty: Box<str> = words.get(2).cloned().unwrap_or_else(|| "".into());
        if !ty.is_empty() {
            any_type = true;
        }
        ids.push(id);
        names.push(name);
        types.push(ty);
    }

    Ok(MtxFeatureRows {
        ids,
        names,
        types: any_type.then_some(types),
        row_name_columns,
    })
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

/// Build backend from 10X Genomics H5 format (Cell Ranger).
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

/// Build backend from AnnData h5ad format (CELLxGENE).
///
/// Prefers `raw/X` over `X` for raw counts.
/// Reads all obs columns and writes metadata `.tsv.gz` files.
/// Creates `barcode@donor_id` column names when donor_id is available.
pub fn run_build_from_h5ad(args: &FromH5adArgs) -> anyhow::Result<()> {
    let file = hdf5::File::open(args.h5ad_file.to_string())?;
    info!("Opened AnnData h5ad file: {}", args.h5ad_file);

    let effective_output = apply_zip_flag(&args.output, args.zip);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;

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
        let values = x_group.dataset("data")?.read_1d::<f32>()?;
        let indices = x_group.dataset("indices")?.read_1d::<u64>()?;
        let indptr = x_group.dataset("indptr")?.read_1d::<u64>()?;
        ValuesIndicesPointers {
            values: values.as_slice().expect("values not contiguous"),
            indices: indices.as_slice().expect("indices not contiguous"),
            indptr: indptr.as_slice().expect("indptr not contiguous"),
        }
        .to_coo(pointer_type)?
    };

    let TripletsShape { nrows, ncols, nnz } = shape;
    info!("Read {} non-zero elements in {} x {}", nnz, nrows, ncols);

    // Feature IDs from var/ (try each candidate in order)
    let mut row_ids: Vec<Box<str>> =
        resolve_h5ad_field(&var_group, &args.row_id_field, "feature IDs").unwrap_or_else(|| {
            info!("No feature ID column found in var/; using numeric IDs");
            (0..nrows).map(|x| x.to_string().into_boxed_str()).collect()
        });

    // Feature names from var/ (try each candidate in order)
    let mut row_names: Vec<Box<str>> =
        resolve_h5ad_field(&var_group, &args.row_name_field, "gene symbols").unwrap_or_else(|| {
            info!("No gene symbol column found in var/; using index only");
            vec![Box::from(""); nrows]
        });

    if nrows < row_ids.len() {
        row_ids.truncate(nrows);
    }
    if nrows < row_names.len() {
        row_names.truncate(nrows);
    }
    assert_eq!(nrows, row_ids.len());
    assert_eq!(nrows, row_names.len());

    // Composite row names: id_name
    let mut row_ids = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_ids);

    // Feature types from var/feature_type (often categorical)
    let mut row_types: Vec<Box<str>> =
        read_h5ad_column(&var_group, "feature_type").unwrap_or_else(|_| vec![Box::from(""); nrows]);
    if nrows < row_types.len() {
        row_types.truncate(nrows);
    }
    assert_eq!(nrows, row_types.len());

    // Cell barcodes from obs/ (try each candidate in order)
    let mut column_names: Vec<Box<str>> =
        resolve_h5ad_field(&obs_group, &args.col_name_field, "barcodes").unwrap_or_else(|| {
            info!("No barcode column found in obs/; using numeric IDs");
            (0..ncols).map(|x| x.to_string().into_boxed_str()).collect()
        });
    info!("Read {} barcodes", column_names.len());
    if ncols < column_names.len() {
        column_names.truncate(ncols);
    }
    assert_eq!(ncols, column_names.len());

    // Create sparse backend
    let mut out = create_sparse_from_triplets_owned(
        triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("created sparse matrix: {}", backend_file);
    out.register_row_names_vec(&row_ids);

    log_feature_type_histogram(args.h5ad_file.as_ref(), &row_types);

    let select_rows =
        filter_row_indices_by_type(&row_types, &args.select_row_type, &args.remove_row_type);

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
        args.block_size,
        &backend_file,
    )?;
    finalize_zarr_output(&backend_file, &effective_output)?;
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

    let effective_output = apply_zip_flag(&args.output, args.zip);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    // 1. Read per-molecule arrays. Keep ndarray `Array1` owners — don't
    // `.to_vec()` them, as that doubles peak memory on large molecule files.
    let barcode_idx = file.dataset("barcode_idx")?.read_1d::<u64>()?;
    let feature_idx = file.dataset("feature_idx")?.read_1d::<u32>()?;
    let count = file.dataset("count")?.read_1d::<u32>()?;
    let gem_group = file.dataset("gem_group")?.read_1d::<u16>()?;
    let library_idx = file.dataset("library_idx")?.read_1d::<u16>()?;
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
    let valid_libraries: rustc_hash::FxHashSet<u16> = {
        let lib_info_ds = file.dataset("library_info")?;
        let lib_info_raw = read_hdf5_strings(lib_info_ds)?;
        let lib_info_json: String = lib_info_raw.iter().map(|s| s.as_ref()).collect();
        let lib_entries: Vec<serde_json::Value> = serde_json::from_str(&lib_info_json)?;

        let mut valid = rustc_hash::FxHashSet::default();
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
    let valid_cells: Option<rustc_hash::FxHashSet<(u64, u16)>> = if !args.no_pass_filter {
        let pf = file.dataset("barcode_info/pass_filter")?.read_2d::<u64>()?;
        let mut cells = rustc_hash::FxHashSet::default();
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
    use rustc_hash::FxHashMap as HashMap;
    use std::collections::BTreeSet;

    let mut col_keys = BTreeSet::new();
    let mut triplet_map: HashMap<(u64, u64), f32> = Default::default();

    {
        // Borrow raw arrays as slices once; avoids repeated bounds-checked
        // indexing through `Array1::Index` in the hot loop.
        let barcode_idx_s = barcode_idx.as_slice().expect("barcode_idx not contiguous");
        let feature_idx_s = feature_idx.as_slice().expect("feature_idx not contiguous");
        let count_s = count.as_slice().expect("count not contiguous");
        let gem_group_s = gem_group.as_slice().expect("gem_group not contiguous");
        let library_idx_s = library_idx.as_slice().expect("library_idx not contiguous");

        for i in 0..n_molecules {
            // Filter by library type
            if !valid_libraries.contains(&library_idx_s[i]) {
                continue;
            }

            // Filter by pass_filter
            if let Some(ref cells) = valid_cells {
                if !cells.contains(&(barcode_idx_s[i], library_idx_s[i])) {
                    continue;
                }
            }

            let col_key = (barcode_idx_s[i], gem_group_s[i]);
            col_keys.insert(col_key);

            // Will remap column index after collecting all keys
            let row = feature_idx_s[i] as u64;
            *triplet_map
                .entry((row, barcode_idx_s[i] * 65536 + gem_group_s[i] as u64))
                .or_insert(0.0) += count_s[i] as f32;
        }
    }

    // Free the raw per-molecule arrays before we materialize the (huge)
    // triplets Vec. These can easily be multiple GB on 10X Aggr outputs.
    drop(barcode_idx);
    drop(feature_idx);
    drop(count);
    drop(gem_group);
    drop(library_idx);
    drop(valid_cells);

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
    let mut out = create_sparse_from_triplets_owned(
        triplets,
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

    let mut row_id_names = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_id_names);

    out.register_row_names_vec(&row_id_names);
    out.register_column_names_vec(&column_names);

    // 8. Filter by feature type
    if nrows < row_types.len() {
        row_types.truncate(nrows);
    }

    let select_rows =
        filter_row_indices_by_type(&row_types, &args.select_row_type, &args.remove_row_type);

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
        args.block_size,
        &backend_file,
    )?;
    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}

fn log_feature_type_histogram(label: &str, row_types: &[Box<str>]) {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for t in row_types {
        *counts.entry(t.as_ref()).or_insert(0) += 1;
    }
    info!("Feature types in {}: {:?}", label, counts);
}

fn run_squeeze_if_needed(
    do_squeeze: bool,
    row_nnz_cutoff: usize,
    column_nnz_cutoff: usize,
    block_size: usize,
    backend_file: &str,
) -> anyhow::Result<()> {
    if do_squeeze {
        info!("Squeeze the backend data {}", backend_file);
        let squeeze_args = RunSqueezeArgs {
            data_files: vec![backend_file.into()],
            row_nnz_cutoff,
            column_nnz_cutoff,
            block_size,
            preload: true,
            show_histogram: false,
            save_histogram: None,
            dry_run: false,
            interactive: false,
            output: None,
            row_align: RowAlignMode::Common,
        };
        run_squeeze(&squeeze_args)?;
    }
    Ok(())
}
