use super::{log_feature_type_histogram, run_squeeze_if_needed};
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::utilities::name_matching::{make_names_unique, RowTypeFilter};
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

    /// Comma-separated patterns; row is kept if its feature_type (column 3
    /// of `--row`) contains ANY of them (case-insensitive). Default
    /// `Gene Expression,Peaks` covers both 10x scRNA and 10x ATAC.
    /// Set to empty to keep everything.
    #[arg(long, default_value = "Gene Expression,Peaks")]
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

    /// Cells per rayon job for the post-build squeeze pass.
    /// Omit for auto-scaling by feature count.
    #[arg(long)]
    pub block_size: Option<usize>,
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

            let sel = RowTypeFilter::parse(&args.select_row_type);
            let rem = RowTypeFilter::parse(&args.remove_row_type);
            let hto = RowTypeFilter::parse(&args.hto_row_type);
            let hto_enabled = !hto.is_empty();

            let mut keep_rows: Vec<usize> = Vec::new();
            let mut hto_rows: Vec<usize> = Vec::new();
            for (i, t) in row_types.iter().enumerate() {
                if hto_enabled && hto.matches(t) {
                    hto_rows.push(i);
                    continue;
                }
                let selected = sel.is_empty() || sel.matches(t);
                let removed = !rem.is_empty() && rem.matches(t);
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

    // BED-shaped peaks file (e.g. 10x ATAC `peaks.bed`): col1 is a chromosome
    // and cols 2/3 are integer start/end coordinates. Reshape so display
    // names come out as `chr:start-end` and feature_type filtering is
    // bypassed (BED has no feature_type column).
    if any_type && looks_like_bed(&ids, &names, &types) {
        info!(
            "{}: detected BED-shaped peaks file; row names will be formatted as chr:start-end",
            row_file
        );
        let n = ids.len();
        let mut new_ids = Vec::with_capacity(n);
        for i in 0..n {
            new_ids.push(format!("{}:{}-{}", ids[i], names[i], types[i]).into_boxed_str());
        }
        return Ok(MtxFeatureRows {
            ids: new_ids,
            names: vec![Box::from(""); n],
            types: None,
            row_name_columns,
        });
    }

    Ok(MtxFeatureRows {
        ids,
        names,
        types: any_type.then_some(types),
        row_name_columns,
    })
}

/// Heuristic: rows look like BED when col1 is a chromosome name and cols 2/3
/// parse as non-negative integers. We sample up to the first 16 non-empty
/// rows (so a stray header line doesn't fool us) and require ALL of them to
/// pass — one mismatch flips the file to "not BED".
fn looks_like_bed(ids: &[Box<str>], starts: &[Box<str>], ends: &[Box<str>]) -> bool {
    let mut checked = 0usize;
    let mut matched = 0usize;
    for ((id, start), end) in ids.iter().zip(starts).zip(ends) {
        if checked >= 16 {
            break;
        }
        let (id, start, end) = (id.as_ref(), start.as_ref(), end.as_ref());
        if id.is_empty() || start.is_empty() || end.is_empty() {
            continue;
        }
        checked += 1;
        if is_chromosome_name(id) && start.parse::<u64>().is_ok() && end.parse::<u64>().is_ok() {
            matched += 1;
        }
    }
    checked >= 1 && matched == checked
}

/// Accepts `chr1`/`chr10`/`chrX` (UCSC) and bare `1`/`X`/`MT` (Ensembl).
fn is_chromosome_name(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() >= 3 && bytes[..3].eq_ignore_ascii_case(b"chr") {
        return true;
    }
    if matches!(
        bytes,
        b"X" | b"x" | b"Y" | b"y" | b"M" | b"m" | b"MT" | b"Mt" | b"mt"
    ) {
        return true;
    }
    s.parse::<u32>().is_ok()
}
