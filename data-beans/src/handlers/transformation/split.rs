//! Train/test and K-fold splitting of a backend, by CELL.
//!
//! # Why this lives in data-beans
//!
//! Splitting is a data operation, not a model one: the folds have to be
//! identical no matter which engine consumes them, or two methods are compared
//! on different data and the comparison means nothing. Putting it beside
//! `subsample` and `subset-columns` gives every downstream crate — `senna`,
//! `pinto`, anything later — one splitter and one seed convention.
//!
//! # The grouped split is the point on spatial data
//!
//! A random cell split leaves nearly every test cell ringed by training cells.
//! On spatial data adjacent cells are near-duplicates, so the model has
//! effectively seen each test cell already and the score comes out optimistic —
//! it measures interpolation between neighbours, not generalization.
//!
//! So the split is by *region*, not by cell. Give it coordinates and a grid
//! resolution — `--coord positions.csv --grid 8` tiles the bounding box 8×8 and
//! assigns whole tiles — and a test region gets a training boundary rather than
//! a training interior. The gap between a random and a tiled split is worth
//! reporting rather than choosing between.
//!
//! `--groups` is the escape hatch for a grouping that is not geometric: donor,
//! sample, slide, or any precomputed labelling.
//!
//! # Ablating features
//!
//! Deliberately not a flag here. Dropping genes from the training half composes
//! out of what already exists:
//!
//! ```text
//! data-beans split data.zarr -o cv --test-frac 0.2
//! data-beans subsample cv.train.zarr.zip --gene-frac 0.8 -o cv.train.ablated
//! ```
//!
//! Keeping the two operations separate means the split stays reproducible from
//! its own seed regardless of what the ablation does.

use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::zarr_io::{finalize_output, prepare_output};

use clap::Args;
use log::info;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rustc_hash::FxHashMap as HashMap;

#[derive(Args, Debug)]
pub struct SplitArgs {
    /// data file -- `.zarr`, `.zarr.zip`, or `.h5`
    pub data_file: Box<str>,

    /// fraction of cells (columns) held out as the test half, in (0, 1)
    #[arg(long, conflicts_with = "folds")]
    pub test_frac: Option<f64>,

    /// number of cross-validation folds; writes {out}.fold{k}.{train,test}
    #[arg(long, conflicts_with = "test_frac")]
    pub folds: Option<usize>,

    /// cell coordinates -- split by REGION instead of by cell (CSV/TSV/parquet)
    ///
    /// First column is the cell name; the coordinate columns are chosen with
    /// --coord-columns. Pair with --grid.
    #[arg(long, value_name = "FILE", conflicts_with = "groups")]
    pub coord: Option<Box<str>>,

    /// 0-based coordinate column indices, comma separated [default: 1,2]
    #[arg(long, value_name = "I,J", value_delimiter = ',', requires = "coord")]
    pub coord_columns: Option<Vec<usize>>,

    /// tiles per axis over the coordinate bounding box; 8 gives an 8x8 grid
    #[arg(long, default_value_t = 8, requires = "coord")]
    pub grid: usize,

    /// two-column table (cell name, group label) -- keep cells sharing a label
    /// on the same side: donor, sample, slide. Use --coord for geometry instead.
    #[arg(long, value_name = "TABLE")]
    pub groups: Option<Box<str>>,

    /// RNG seed for a reproducible split
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// backend for the output files
    #[arg(long, value_enum, default_value = "zarr")]
    pub backend: SparseIoBackend,

    /// output prefix: writes {output}.train.* and {output}.test.*
    #[arg(short, long)]
    pub output: Box<str>,

    /// keep `.zarr` directories instead of producing `.zarr.zip` archives
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,
}

/// Split a backend into train and test halves by cell, or into K folds.
///
/// Every cell lands in exactly one side (or exactly one fold's test half), so
/// the halves are disjoint and together cover the input — checked, not assumed.
pub fn run_split(args: &SplitArgs) -> anyhow::Result<()> {
    let (backend_in, file_in) = resolve_backend_file(&args.data_file, None)?;
    let data = open_sparse_matrix(&file_in, &backend_in)?;
    let n_columns = data
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("backend has no `ncol`"))?;
    let column_names = data.column_names()?;

    // The unit that gets shuffled and assigned as a whole. One column each by
    // default, one grid tile under `--coord`, one label under `--groups`.
    // Everything below is identical in all three cases, which is why grouped
    // splitting costs nothing extra to support.
    let column_groups: Vec<Vec<usize>> = match (args.coord.as_deref(), args.groups.as_deref()) {
        (Some(path), _) => column_groups_from_grid(path, args, &column_names)?,
        (_, Some(path)) => column_groups_from_table(path, &column_names)?,
        _ => (0..n_columns).map(|i| vec![i]).collect(),
    };
    // Two, not one: a single group cannot be divided, and `--test-frac` would
    // reach `clamp(1, 0)` and panic. One group is a real outcome of a coarse
    // `--grid` or a `--groups` table with a single label, so it gets a message
    // that names the cause rather than a backtrace.
    anyhow::ensure!(
        column_groups.len() >= 2,
        "only {} group(s) to split: every cell falls in the same tile or group, \
         so there is no way to hold any of them out. Raise --grid, or check that \
         --groups / --coord really vary across cells.",
        column_groups.len()
    );

    // Group ids first, then the columns they cover: the partition never sees a
    // column, so a grouped split cannot leak one across the boundary by
    // construction rather than by a later check.
    let splits: Vec<(Vec<usize>, Vec<usize>)> =
        partition_groups(column_groups.len(), args.test_frac, args.folds, args.seed)?
            .into_iter()
            .map(|(train_groups, test_groups)| {
                (
                    columns_in_groups(&column_groups, &train_groups),
                    columns_in_groups(&column_groups, &test_groups),
                )
            })
            .collect();

    for (fold, (train_columns, test_columns)) in splits.iter().enumerate() {
        anyhow::ensure!(
            train_columns.len() + test_columns.len() == n_columns,
            "split lost cells: {} train + {} test != {n_columns}",
            train_columns.len(),
            test_columns.len()
        );
        let tag = |half: &str| match args.folds {
            Some(_) => format!("{}.fold{fold}.{half}", args.output),
            None => format!("{}.{half}", args.output),
        };
        write_half(&*data, train_columns, &tag("train"), args)?;
        write_half(&*data, test_columns, &tag("test"), args)?;
        info!(
            "fold {fold}: {} train cells, {} test cells",
            train_columns.len(),
            test_columns.len()
        );
    }
    Ok(())
}

/// Flatten group ids to the column indices they cover, sorted so the backend
/// read stays sequential.
fn columns_in_groups(column_groups: &[Vec<usize>], picked: &[usize]) -> Vec<usize> {
    let mut columns: Vec<usize> = picked
        .iter()
        .flat_map(|&g| column_groups[g].iter().copied())
        .collect();
    columns.sort_unstable();
    columns
}

/// Cells grouped into a regular grid over their coordinates.
///
/// A grid rather than a clustering on purpose: the tiles are then a property of
/// the tissue's geometry alone, reproducible from the coordinate file without a
/// model run, and a reader can say exactly how large a held-out region was.
/// Empty tiles simply do not appear.
fn column_groups_from_grid(
    path: &str,
    args: &SplitArgs,
    column_names: &[Box<str>],
) -> anyhow::Result<Vec<Vec<usize>>> {
    let tiles_per_axis = args.grid;
    anyhow::ensure!(tiles_per_axis >= 2, "--grid must be at least 2");
    let coord_columns = args.coord_columns.clone().unwrap_or_else(|| vec![1, 2]);
    anyhow::ensure!(
        coord_columns.len() == 2,
        "--coord-columns needs exactly two indices; got {}",
        coord_columns.len()
    );
    let coords = read_coords(path, &coord_columns)?;

    let mut coord_of_column: Vec<(f64, f64)> = Vec::with_capacity(column_names.len());
    let mut n_missing = 0usize;
    for name in column_names {
        match coords.get(name.as_ref()) {
            Some(&xy) => coord_of_column.push(xy),
            None => {
                n_missing += 1;
                coord_of_column.push((f64::NAN, f64::NAN));
            }
        }
    }
    anyhow::ensure!(
        n_missing == 0,
        "{path}: {n_missing} of {} cells have no coordinate",
        column_names.len()
    );

    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for &(x, y) in &coord_of_column {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    // A degenerate span would put every cell in tile 0; widen it so the grid is
    // still well defined rather than silently collapsing to one group.
    let span = |lo: f64, hi: f64| if hi > lo { hi - lo } else { 1.0 };
    let (width, height) = (span(x0, x1), span(y0, y1));

    let mut group_of_tile: HashMap<usize, usize> = HashMap::default();
    let mut column_groups: Vec<Vec<usize>> = Vec::new();
    for (column, &(x, y)) in coord_of_column.iter().enumerate() {
        let tx = (((x - x0) / width) * tiles_per_axis as f64)
            .floor()
            .clamp(0.0, (tiles_per_axis - 1) as f64) as usize;
        let ty = (((y - y0) / height) * tiles_per_axis as f64)
            .floor()
            .clamp(0.0, (tiles_per_axis - 1) as f64) as usize;
        let tile = ty * tiles_per_axis + tx;
        let next = column_groups.len();
        let group = *group_of_tile.entry(tile).or_insert(next);
        if group == next {
            column_groups.push(Vec::new());
        }
        column_groups[group].push(column);
    }
    info!(
        "spatial split: {} cells in {} non-empty tiles of a \
         {tiles_per_axis}x{tiles_per_axis} grid ({path})",
        column_names.len(),
        column_groups.len()
    );
    Ok(column_groups)
}

/// `cell -> (x, y)` from a CSV/TSV/parquet whose first column is the cell name.
fn read_coords(path: &str, cols: &[usize]) -> anyhow::Result<HashMap<Box<str>, (f64, f64)>> {
    let (i, j) = (cols[0], cols[1]);
    let mut out: HashMap<Box<str>, (f64, f64)> = HashMap::default();
    if path.ends_with(".parquet") {
        let names = matrix_util::parquet::read_parquet_string_column(path, 0)?;
        let mat = <nalgebra::DMatrix<f32> as matrix_util::traits::IoOps>::from_parquet(path)?;
        for (r, name) in names.iter().enumerate() {
            out.insert(
                name.clone(),
                (
                    f64::from(mat.mat[(r, i - 1)]),
                    f64::from(mat.mat[(r, j - 1)]),
                ),
            );
        }
        return Ok(out);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("reading coordinates {path}: {e}"))?;
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = if line.contains('\t') {
            line.split('\t').collect()
        } else if line.contains(',') {
            line.split(',').collect()
        } else {
            line.split_whitespace().collect()
        };
        let max = i.max(j);
        anyhow::ensure!(
            f.len() > max,
            "{path}:{}: {} columns, need index {max}",
            lineno + 1,
            f.len()
        );
        // A header line has non-numeric coordinates; skip it rather than failing.
        let (Ok(x), Ok(y)) = (f[i].trim().parse::<f64>(), f[j].trim().parse::<f64>()) else {
            continue;
        };
        out.insert(f[0].trim().into(), (x, y));
    }
    anyhow::ensure!(!out.is_empty(), "{path}: no coordinate rows parsed");
    Ok(out)
}

/// Cells grouped by their group label, from a two-column (cell, label) table.
///
/// A cell missing from the table is an error rather than a silent singleton:
/// a partial grouping would split part of the data by group and the rest by
/// cell, and the resulting fold is neither.
fn column_groups_from_table(
    path: &str,
    column_names: &[Box<str>],
) -> anyhow::Result<Vec<Vec<usize>>> {
    let (table_names, table_labels) = read_group_table(path)?;
    anyhow::ensure!(
        table_names.len() == table_labels.len(),
        "{path}: {} names but {} group labels",
        table_names.len(),
        table_labels.len()
    );
    let label_of_column: HashMap<&str, &str> = table_names
        .iter()
        .map(std::convert::AsRef::as_ref)
        .zip(table_labels.iter().map(std::convert::AsRef::as_ref))
        .collect();

    let mut group_of_label: HashMap<&str, usize> = HashMap::default();
    let mut column_groups: Vec<Vec<usize>> = Vec::new();
    let mut n_missing = 0usize;
    for (column, name) in column_names.iter().enumerate() {
        let Some(&label) = label_of_column.get(name.as_ref()) else {
            n_missing += 1;
            continue;
        };
        let next = column_groups.len();
        let group = *group_of_label.entry(label).or_insert(next);
        if group == next {
            column_groups.push(Vec::new());
        }
        column_groups[group].push(column);
    }
    anyhow::ensure!(
        n_missing == 0,
        "{path}: {n_missing} of {} cells have no group label",
        column_names.len()
    );
    info!(
        "grouped split: {} cells in {} groups from {path}",
        column_names.len(),
        column_groups.len()
    );
    Ok(column_groups)
}

/// The `(name, label)` columns of a group table.
type GroupTable = (Vec<Box<str>>, Vec<Box<str>>);

/// The `(name, label)` columns of a group table, from parquet or delimited text.
///
/// Both, because the flag says `TABLE` and the help says "two-column table" —
/// reading only parquet meant the TSV a user would naturally write failed with a
/// parquet decode error. The text path goes through the workspace reader, so it
/// sniffs the delimiter and handles `.gz` like every other list this tool takes.
///
/// Matching stays EXACT (see the caller). `Membership` offers base-key and prefix
/// fallbacks, which are right for joining annotations and wrong here: a split
/// that fuzzy-matches puts a cell in the wrong half, and nothing downstream can
/// detect it.
fn read_group_table(path: &str) -> anyhow::Result<GroupTable> {
    let is_parquet = std::path::Path::new(path)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|e| e.eq_ignore_ascii_case("parquet"));
    if is_parquet {
        return Ok((
            matrix_util::parquet::read_parquet_string_column(path, 0)?,
            matrix_util::parquet::read_parquet_string_column(path, 1)?,
        ));
    }

    let delim = matrix_util::membership::detect_delimiter(path);
    let out = matrix_util::common_io::read_lines_of_words_delim(path, delim, -1)?;
    let mut names = Vec::with_capacity(out.lines.len());
    let mut labels = Vec::with_capacity(out.lines.len());
    for (row, fields) in out.lines.iter().enumerate() {
        anyhow::ensure!(
            fields.len() >= 2,
            "{path}:{}: {} column(s), need a cell name and a group label",
            row + 1,
            fields.len()
        );
        names.push(fields[0].clone());
        labels.push(fields[1].clone());
    }
    Ok((names, labels))
}

/// Write one half as a new backend. Only the selected columns are read, so the
/// I/O scales with the half rather than the input.
///
/// MEMORY, which is a different story: `read_triplets_by_columns` returns every
/// non-zero of the half at once, and `create_sparse_from_triplets_owned` then
/// builds a second copy. At `(u64, u64, f32)` padded to 24 bytes that is a few
/// hundred MB for a slide-scale half and tens of GB for an imaging-scale one, so
/// the upper end OOMs rather than merely being slow.
///
/// The streaming writer next door (`handlers::merging`: `begin_streaming_csc` +
/// `append_csc_slab`) is the shape this wants, but it needs the total nnz up
/// front and the `SparseIo` trait exposes only whole-matrix `num_non_zeros()` —
/// no per-column indptr — so a half's nnz is not knowable without either a
/// counting pass or a new trait method on every backend. `subsample` has the
/// identical shape, so the two should move together. Left as-is deliberately:
/// it is a redesign of a write path whose output has to stay byte-equivalent,
/// not a cleanup.
fn write_half(
    data: &dyn SparseIo<IndexIter = Vec<usize>>,
    selected_columns: &[usize],
    output: &str,
    args: &SplitArgs,
) -> anyhow::Result<()> {
    let row_names = data.row_names()?;
    let all_column_names = data.column_names()?;
    let n_rows = data
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?;

    // The slow step of the whole subcommand, and it is one opaque call — say what
    // is happening before it, or a large half looks like a hang.
    info!("Reading {} columns for {output}", selected_columns.len());
    let (_, _, triplets) = data.read_triplets_by_columns(selected_columns.to_vec())?;
    let nnz = triplets.len();
    let kept_column_names: Vec<Box<str>> = selected_columns
        .iter()
        .map(|&c| all_column_names[c].clone())
        .collect();

    let (effective_output, backend_out, file_out) =
        prepare_output(output, args.backend.clone(), args.zip)?;
    let mut out = create_sparse_from_triplets_owned(
        triplets,
        (n_rows, selected_columns.len(), nnz),
        Some(file_out.as_ref()),
        Some(&backend_out),
    )?;
    out.register_row_names_vec(&row_names);
    out.register_column_names_vec(&kept_column_names);
    drop(out);

    let final_path = finalize_output(&file_out, &effective_output)?;
    info!(
        "wrote {final_path} ({n_rows} genes x {} cells, {nnz} non-zeros)",
        selected_columns.len()
    );
    Ok(())
}

/// Partition `n_groups` shuffled groups into `(train, test)` group-index pairs.
///
/// Separated from the column bookkeeping so the part that decides who is held
/// out can be tested without a backend on disk — this is the piece where an
/// off-by-one silently produces an overlapping or incomplete fold, which no
/// downstream number would reveal.
fn partition_groups(
    n_groups: usize,
    test_frac: Option<f64>,
    folds: Option<usize>,
    seed: u64,
) -> anyhow::Result<Vec<(Vec<usize>, Vec<usize>)>> {
    let mut shuffled_groups: Vec<usize> = (0..n_groups).collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    shuffled_groups.shuffle(&mut rng);

    match (test_frac, folds) {
        (Some(f), _) => {
            anyhow::ensure!(
                f > 0.0 && f < 1.0,
                "--test-frac must be in (0, 1); got {f}. A percentage goes in as e.g. 0.2"
            );
            // Clamped so neither side is ever empty: a fraction that rounds to
            // zero, or to everything, is a silently useless split rather than an
            // error the user would see.
            let n_test = ((f * n_groups as f64).round() as usize).clamp(1, n_groups - 1);
            let (test_groups, train_groups) = shuffled_groups.split_at(n_test);
            Ok(vec![(train_groups.to_vec(), test_groups.to_vec())])
        }
        (_, Some(k)) => {
            anyhow::ensure!(k >= 2, "--folds must be at least 2");
            anyhow::ensure!(
                k <= n_groups,
                "--folds {k} exceeds the {n_groups} groups available to split"
            );
            Ok((0..k)
                .map(|f| {
                    let take = |in_test: bool| -> Vec<usize> {
                        shuffled_groups
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| (i % k == f) == in_test)
                            .map(|(_, &g)| g)
                            .collect()
                    };
                    (take(false), take(true))
                })
                .collect())
        }
        (None, None) => anyhow::bail!("specify one of --test-frac or --folds"),
    }
}

#[cfg(test)]
mod tests;
