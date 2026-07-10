use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::utilities::name_matching::split_id_name;

use clap::Args;
use log::{info, warn};
use matrix_util::common_io::{mkdir, open_buf_writer, write_lines};
use std::io::Write;

/// Cells per block when streaming the matrix out (bounds peak memory to one
/// block of triplets regardless of matrix size).
const COL_BLOCK: usize = 512;

#[derive(Args, Debug)]
pub struct ToMtxArgs {
    /// backend data file to export (`.zarr`, `.zarr.zip`, or `.h5`)
    pub data_file: Box<str>,

    #[arg(
        short,
        long,
        help = "Output directory for the 10x-style triplet",
        long_help = "Output directory. The 10x Cell Ranger MEX triplet is written inside it as\n\
                     matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz (loadable with\n\
                     `scanpy.read_10x_mtx(dir)` or Seurat `Read10X(dir)`). The directory\n\
                     is created if missing."
    )]
    pub output: Box<str>,

    /// keep the triplet uncompressed (`matrix.mtx` / `features.tsv` /
    /// `barcodes.tsv`) instead of the default gzipped Cell Ranger v3 layout
    #[arg(long = "no-gzip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub gzip: bool,

    #[arg(
        long,
        default_value = "Gene Expression",
        help = "feature_type written to column 3 of features.tsv",
        long_help = "Value written to the third column (feature_type) of features.tsv for\n\
                     every feature. 10x uses e.g. 'Gene Expression', 'Antibody Capture',\n\
                     'Peaks'. `from-mtx` reads this back for --select-row-type filtering."
    )]
    pub feature_type: Box<str>,
}

/// Export a data-beans backend to a 10x Genomics MEX (MatrixMarket) triplet
/// readable by scanpy / Seurat. This is the inverse of `from-mtx`.
///
/// data-beans already stores the matrix as (features x cells), which is 10x's
/// orientation (features are rows, barcodes are columns), so no transpose is
/// needed — unlike `to-h5ad`.
pub fn run_export_to_mtx(args: &ToMtxArgs) -> anyhow::Result<()> {
    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;
    let data = open_sparse_matrix(&data_file, &backend)?;

    let nrow = data
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?;
    let ncol = data
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("backend has no `ncol`"))?;
    let nnz = data
        .num_non_zeros()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nnz`"))?;

    let row_names = data.row_names()?; // features
    let col_names = data.column_names()?; // barcodes
    if row_names.len() != nrow {
        anyhow::bail!("row name count {} != nrow {}", row_names.len(), nrow);
    }
    if col_names.len() != ncol {
        anyhow::bail!("column name count {} != ncol {}", col_names.len(), ncol);
    }

    let dir = args.output.trim_end_matches('/');
    mkdir(dir)?;
    let suffix = if args.gzip { ".gz" } else { "" };
    let mtx_path = format!("{}/matrix.mtx{}", dir, suffix);
    let features_path = format!("{}/features.tsv{}", dir, suffix);
    let barcodes_path = format!("{}/barcodes.tsv{}", dir, suffix);

    info!(
        "Exporting {} -> {}/ ({} features x {} barcodes, {} non-zeros)",
        data_file, dir, nrow, ncol, nnz
    );

    // features.tsv: 10x 3-column layout (id, name, feature_type). Recover the
    // id/name split from the backend's composite `id_name` display name.
    let feature_lines: Vec<Box<str>> = row_names
        .iter()
        .map(|composite| {
            let (id, name) = split_id_name(composite);
            format!("{}\t{}\t{}", id, name, args.feature_type).into_boxed_str()
        })
        .collect();
    write_lines(&feature_lines, &features_path)?;
    info!("wrote {}", features_path);

    // barcodes.tsv: one barcode per line.
    write_lines(&col_names, &barcodes_path)?;
    info!("wrote {}", barcodes_path);

    // matrix.mtx: stream the CSC columns as 1-based (feature, barcode, count).
    write_mtx_10x(data.as_ref(), &mtx_path, nrow, ncol, nnz)?;
    info!("wrote {}", mtx_path);

    info!("done: {}/", dir);
    Ok(())
}

/// Write a Cell Ranger v3 style `matrix.mtx[.gz]`: a MatrixMarket coordinate
/// file with an `integer` field, a `%metadata_json` comment, 1-based indices
/// and features as rows / barcodes as columns. Values are streamed a column
/// block at a time. Non-integer values (a non-count backend) are rounded, with
/// a single warning — 10x MEX is defined over integer counts.
fn write_mtx_10x(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    path: &str,
    nrow: usize,
    ncol: usize,
    nnz: usize,
) -> anyhow::Result<()> {
    let mut buf = open_buf_writer(path)?;
    writeln!(buf, "%%MatrixMarket matrix coordinate integer general")?;
    writeln!(
        buf,
        "%metadata_json: {{\"software_version\": \"data-beans-{}\", \"format_version\": 2}}",
        env!("CARGO_PKG_VERSION")
    )?;
    writeln!(buf, "{} {} {}", nrow, ncol, nnz)?;

    let mut warned_fractional = false;
    let mut lo = 0usize;
    while lo < ncol {
        let hi = (lo + COL_BLOCK).min(ncol);
        let (_, _, triplets) = data.read_triplets_by_columns((lo..hi).collect())?;
        for (row_i, col_local, val) in triplets {
            let count = val.round();
            if !warned_fractional && (val - count).abs() > 1e-6 {
                warn!(
                    "matrix holds non-integer values (e.g. {}); rounding to integers \
                     for the 10x `integer` MatrixMarket field",
                    val
                );
                warned_fractional = true;
            }
            // 1-based; global column = block offset + local index.
            writeln!(
                buf,
                "{} {} {}",
                row_i + 1,
                lo as u64 + col_local + 1,
                count as i64
            )?;
        }
        lo = hi;
    }
    buf.flush()?;
    Ok(())
}
