use super::{log_feature_type_histogram, run_squeeze_if_needed};
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::sparse_util::*;
use crate::utilities::name_matching::{
    compose_id_name, filter_row_indices_by_type, make_names_unique,
};
use crate::zarr_io::*;

use clap::Args;
use log::info;
use matrix_util::common_io::*;

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
		     The zarr backend produces {output}.zarr.zip by default;\n\
		     pass --no-zip to keep a {output}.zarr directory instead.\n\
		     Metadata files will be named {output}.cell_metadata.tsv.gz, etc."
    )]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
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
        help = "Cells per rayon job for the post-build squeeze pass \
                (omit for auto-scaling by feature count)"
    )]
    pub block_size: Option<usize>,
}
pub fn run_build_from_h5ad(args: &FromH5adArgs) -> anyhow::Result<()> {
    let file = hdf5::File::open(args.h5ad_file.to_string())?;
    info!("Opened AnnData h5ad file: {}", args.h5ad_file);

    let effective_output = apply_zip_flag(&args.output, args.zip, &args.backend);
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
