use crate::hdf5_io::*;
use crate::sparse_io::*;

use clap::Args;
use hdf5::types::{H5Type, VarLenUnicode};
use log::info;
use matrix_util::common_io::read_lines;
use std::collections::HashMap;

/// gzip level for the numeric `X` datasets. gzip (DEFLATE) is chosen over the
/// backend's Blosc because h5py / anndata read it without any filter plugin.
const GZIP_LEVEL: u8 = 4;

/// Upper bound on the chunk length (elements) for the `X` datasets.
const MAX_CHUNK_ELEMS: usize = 1 << 20;

#[derive(Args, Debug)]
pub struct ToH5adArgs {
    /// backend data file to export (`.zarr`, `.zarr.zip`, or `.h5`)
    pub data_file: Box<str>,

    #[arg(
        short,
        long,
        help = "Output `.h5ad` file (default: <input stem>.h5ad)",
        long_help = "Output AnnData h5ad path. `.h5ad` is appended if the name \
                     has no such suffix. Defaults to the input file's stem \
                     with a `.h5ad` extension."
    )]
    pub output: Option<Box<str>>,

    #[arg(
        long,
        help = "Cell-metadata TSV to attach as `obs` columns",
        long_help = "Optional (gzip ok) TSV whose first column is the cell key. \
                     Keys are matched against the backend column names, with a \
                     `barcode@donor` -> `barcode` fallback. Remaining columns \
                     become `obs` columns; unmatched cells get `NA`. \
                     Pair this with the `*.cell_metadata.tsv.gz` that `from-h5ad` \
                     wrote to round-trip cell annotations."
    )]
    pub obs: Option<Box<str>>,

    #[arg(
        long,
        help = "Feature-metadata TSV to attach as `var` columns",
        long_help = "Optional (gzip ok) TSV whose first column is the feature key \
                     (matched against the backend row names). Remaining columns \
                     become `var` columns; unmatched features get `NA`."
    )]
    pub var: Option<Box<str>>,
}

/// Export a data-beans backend to an AnnData `.h5ad` file readable by
/// scanpy / anndata. This is the inverse of `from-h5ad`.
///
/// data-beans stores the matrix `M` as (features x cells) with `/by_column`
/// holding the CSC of `M` (column pointers over cells, row indices over
/// features). AnnData's `X` is (cells x features) = `Mᵀ`, stored as CSR.
/// Because CSR(`Mᵀ`) is byte-for-byte identical to CSC(`M`), the `/by_column`
/// arrays copy straight into `X` as a `csr_matrix` of shape `[n_cells,
/// n_features]` — no re-sort, exactly undoing the transpose `from-h5ad` did.
pub fn run_export_to_h5ad(args: &ToH5adArgs) -> anyhow::Result<()> {
    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;

    let output = match &args.output {
        Some(o) if o.ends_with(".h5ad") => o.to_string(),
        Some(o) => format!("{}.h5ad", o),
        None => format!("{}.h5ad", strip_backend_suffix(&data_file)),
    };

    info!("Exporting {} -> {}", data_file, output);

    let mut data = open_sparse_matrix(&data_file, &backend)?;

    let nrow = data
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?; // features -> var
    let ncol = data
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("backend has no `ncol`"))?; // cells -> obs
    let nnz = data
        .num_non_zeros()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nnz`"))?;

    let row_names = data.row_names()?; // features
    let col_names = data.column_names()?; // cells
    if row_names.len() != nrow {
        anyhow::bail!("row name count {} != nrow {}", row_names.len(), nrow);
    }
    if col_names.len() != ncol {
        anyhow::bail!("column name count {} != ncol {}", col_names.len(), ncol);
    }

    // Load the CSC (`/by_column`) arrays; they map directly onto X = CSR.
    info!("preloading columns ({} cells, {} non-zeros)...", ncol, nnz);
    data.preload_columns()?;
    let (indptr, indices, values) = data
        .csc_column_arrays()
        .ok_or_else(|| anyhow::anyhow!("failed to access preloaded CSC arrays"))?;
    debug_assert_eq!(indptr.len(), ncol + 1);
    debug_assert_eq!(indices.len(), values.len());

    let file = hdf5::File::create(&output)?;

    // Root: mark as an AnnData container.
    write_str_attr(&file, "encoding-type", "anndata")?;
    write_str_attr(&file, "encoding-version", "0.1.0")?;

    // X: csr_matrix of shape [n_obs = ncol, n_var = nrow].
    write_csr_x(&file, ncol, nrow, indptr, indices, values)?;

    // obs (cells) and var (features) dataframes.
    write_dataframe(&file, "obs", &col_names, args.obs.as_deref(), true)?;
    write_dataframe(&file, "var", &row_names, args.var.as_deref(), false)?;

    file.flush()?;
    info!(
        "done: {} ({} obs x {} var, {} non-zeros)",
        output, ncol, nrow, nnz
    );
    Ok(())
}

/// Write `/X` as an AnnData `csr_matrix` group from the backend's CSC arrays.
fn write_csr_x(
    file: &hdf5::File,
    n_obs: usize,
    n_var: usize,
    indptr: &[u64],
    indices: &[u64],
    values: &[f32],
) -> anyhow::Result<()> {
    let x = file.create_group("X")?;
    write_str_attr(&x, "encoding-type", "csr_matrix")?;
    write_str_attr(&x, "encoding-version", "0.1.0")?;
    x.new_attr::<i64>()
        .shape([2])
        .create("shape")?
        .write_raw(&[n_obs as i64, n_var as i64])?;

    // Values keep their f32 dtype. Column indices (features) and the row
    // pointer are stored as the narrowest signed int scipy will accept:
    // i32 unless the matrix is large enough to overflow it.
    write_gzip_dataset(&x, "data", values)?;

    if values.len() > i32::MAX as usize || n_var > i32::MAX as usize {
        write_index_arrays(&x, indptr, indices, |v| v as i64)?;
    } else {
        write_index_arrays(&x, indptr, indices, |v| v as i32)?;
    }
    Ok(())
}

/// Write the CSR `indices`/`indptr` datasets, casting the backend's `u64`
/// arrays to the chosen integer width.
fn write_index_arrays<T: H5Type>(
    x: &hdf5::Group,
    indptr: &[u64],
    indices: &[u64],
    cast: impl Fn(u64) -> T,
) -> anyhow::Result<()> {
    write_gzip_dataset(
        x,
        "indices",
        &indices.iter().map(|&v| cast(v)).collect::<Vec<T>>(),
    )?;
    write_gzip_dataset(
        x,
        "indptr",
        &indptr.iter().map(|&v| cast(v)).collect::<Vec<T>>(),
    )?;
    Ok(())
}

/// Write an AnnData `dataframe` group (`obs` or `var`).
///
/// * `index` — the observation / variable names (`_index` dataset).
/// * `meta_tsv` — optional metadata to align onto `index` and add as columns.
/// * `barcode_fallback` — when matching `meta_tsv` keys, also try the barcode
///   prefix before `@` (data-beans names multi-donor cells `barcode@donor`).
fn write_dataframe(
    file: &hdf5::File,
    name: &str,
    index: &[Box<str>],
    meta_tsv: Option<&str>,
    barcode_fallback: bool,
) -> anyhow::Result<()> {
    let group = file.create_group(name)?;
    write_str_attr(&group, "encoding-type", "dataframe")?;
    write_str_attr(&group, "encoding-version", "0.2.0")?;
    write_str_attr(&group, "_index", "_index")?;

    write_string_array(&group, "_index", index)?;

    let (col_names, columns) = match meta_tsv {
        Some(path) => load_aligned_metadata(path, index, barcode_fallback)?,
        None => (vec![], vec![]),
    };
    for (cname, col) in col_names.iter().zip(columns.iter()) {
        write_string_array(&group, cname, col)?;
    }

    // column-order lists the non-index columns (empty array when none).
    let order: Vec<VarLenUnicode> = col_names
        .iter()
        .map(|c| vlu(c))
        .collect::<anyhow::Result<_>>()?;
    group
        .new_attr::<VarLenUnicode>()
        .shape([order.len()])
        .create("column-order")?
        .write_raw(&order)?;

    Ok(())
}

/// Parse a metadata TSV and align its rows onto `names`.
///
/// The first TSV column is the join key; the rest become columns (a literal
/// `_index` column, if present, is dropped to avoid clashing with the index
/// dataset). Returns `(column_names, columns)` where every column has the same
/// length as `names`; unmatched entries are filled with `"NA"`.
#[allow(clippy::type_complexity)]
fn load_aligned_metadata(
    path: &str,
    names: &[Box<str>],
    barcode_fallback: bool,
) -> anyhow::Result<(Vec<Box<str>>, Vec<Vec<Box<str>>>)> {
    let lines = read_lines(path)?;
    if lines.is_empty() {
        log::warn!("metadata file {} is empty; no columns attached", path);
        return Ok((vec![], vec![]));
    }

    let header: Vec<&str> = lines[0].split('\t').collect();
    // Keep every non-key column except a literal "_index".
    let (keep_idx, col_names): (Vec<usize>, Vec<Box<str>>) = header
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, &c)| c != "_index")
        .map(|(i, &c)| (i, Box::<str>::from(c)))
        .unzip();

    let mut key2row: HashMap<Box<str>, Vec<Box<str>>> = HashMap::new();
    for line in &lines[1..] {
        let fields: Vec<Box<str>> = line.split('\t').map(|s| s.into()).collect();
        if let Some(key) = fields.first() {
            key2row.insert(key.clone(), fields);
        }
    }

    let mut columns: Vec<Vec<Box<str>>> = vec![Vec::with_capacity(names.len()); col_names.len()];
    let mut n_matched = 0usize;
    for nm in names {
        let row = key2row.get(nm).or_else(|| {
            if barcode_fallback {
                nm.split(COLUMN_SEP).next().and_then(|bc| key2row.get(bc))
            } else {
                None
            }
        });
        match row {
            Some(fields) => {
                n_matched += 1;
                for (ci, &hidx) in keep_idx.iter().enumerate() {
                    let val = fields.get(hidx).cloned().unwrap_or_else(|| "NA".into());
                    columns[ci].push(val);
                }
            }
            None => {
                for col in columns.iter_mut() {
                    col.push("NA".into());
                }
            }
        }
    }

    info!(
        "attached {} metadata column(s) from {}; matched {}/{} entries",
        col_names.len(),
        path,
        n_matched,
        names.len()
    );
    Ok((col_names, columns))
}

//////////////////////
// HDF5 write helpers //
//////////////////////

/// Encode a Rust string as an HDF5 variable-length UTF-8 string.
fn vlu(s: &str) -> anyhow::Result<VarLenUnicode> {
    s.parse::<VarLenUnicode>()
        .map_err(|e| anyhow::anyhow!("cannot encode '{}' as an HDF5 string: {}", s, e))
}

/// Write a scalar UTF-8 string attribute onto a group or dataset.
fn write_str_attr(loc: &hdf5::Location, name: &str, value: &str) -> anyhow::Result<()> {
    loc.new_attr::<VarLenUnicode>()
        .create(name)?
        .write_scalar(&vlu(value)?)?;
    Ok(())
}

/// Create a gzip-compressed 1-D numeric dataset and write `data` into it.
/// Falls back to a plain contiguous dataset when empty (a chunked layout
/// needs a non-zero chunk size).
fn write_gzip_dataset<T: H5Type>(
    group: &hdf5::Group,
    name: &str,
    data: &[T],
) -> anyhow::Result<()> {
    let n = data.len();
    let ds = if n > 0 {
        let chunk = n.min(MAX_CHUNK_ELEMS);
        group
            .new_dataset::<T>()
            .shape(n)
            .deflate(GZIP_LEVEL)
            .chunk([chunk])
            .create(name)?
    } else {
        group.new_dataset::<T>().shape(n).create(name)?
    };
    ds.write_raw(data)?;
    Ok(())
}

/// Write a `string-array`-encoded dataset of UTF-8 strings.
fn write_string_array(group: &hdf5::Group, name: &str, values: &[Box<str>]) -> anyhow::Result<()> {
    let encoded: Vec<VarLenUnicode> = values
        .iter()
        .map(|s| vlu(s))
        .collect::<anyhow::Result<_>>()?;
    let ds = group
        .new_dataset::<VarLenUnicode>()
        .shape(encoded.len())
        .create(name)?;
    ds.write_raw(&encoded)?;
    write_str_attr(&ds, "encoding-type", "string-array")?;
    write_str_attr(&ds, "encoding-version", "0.2.0")?;
    Ok(())
}
