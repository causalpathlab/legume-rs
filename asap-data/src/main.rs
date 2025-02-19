mod simulate;
mod sparse_io;
mod sparse_matrix_hdf5;
mod sparse_matrix_zarr;
mod statistics;

use crate::sparse_io::*;
use crate::statistics::RunningStatistics;
use clap::{ArgAction, Args, Parser, Subcommand};
use env_logger;
use indicatif::ParallelProgressIterator;
use indicatif::{ProgressBar, ProgressDrawTarget};
use log::info;
use matrix_util::traits::IoOps;
use matrix_util::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type SData = dyn SparseIo<IndexIter = Vec<usize>>;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match &cli.commands {
        Commands::Build(args) => {
            run_build(args)?;
        }
        Commands::Simulate(args) => {
            run_simulate(args)?;
        }
        Commands::Stat(args) => {
            run_stat(args)?;
        }
        Commands::Statistics(args) => {
            run_stat(args)?;
        }
        Commands::Squeeze(args) => {
            run_squeeze(args)?;
        }
        Commands::Columns(args) => {
            take_columns(args)?;
        }
        Commands::SortRows(args) => {
            reorder_rows(args)?;
        }
        Commands::MergeMtx(args) => {
            run_merge(args)?;
        }
    }

    Ok(())
}

#[derive(Parser)]
#[command(version, about, long_about=None)]
#[command(propagate_version = true)]
///
/// Basic utility functions for processing sparse matrices
///
/// - build: build from .mtx fileset to another faster format
/// - stat: compute basic statistics of a sparse matrix
/// - squeeze: filter out rows and columns that are (nearly) empty
/// - simulate: simulate a sparse matrix
///
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build faster backend data from mtx
    Build(RunBuildArgs),

    /// Sort rows according to the order of row names specified in a
    /// row name file
    SortRows(SortRowsArgs),

    /// Take columns from the sparse matrix and save it to an `output`
    /// file as a dense matrix
    Columns(TakeColumnsArgs),

    /// Merge multiple 10x `.mtx` files
    MergeMtx(MergeMtxArgs),

    /// Filter out rows and columns by number of non-zeros (Q/C)
    Squeeze(RunSqueezeArgs),

    /// Take basic statistics from a sparse matrix
    Statistics(RunStatArgs),
    /// Simulate a sparse matrix data
    Stat(RunStatArgs),

    /// Simulate a sparse matrix data
    Simulate(RunSimulateArgs),
}

#[derive(Args)]
pub struct SortRowsArgs {
    /// Data file -- either `.zarr` or `.h5`
    data_file: Box<str>,

    /// Row/feature name file (name per each line; `.tsv.gz` or `.tsv`)
    #[arg(short, long, required = true)]
    row_file: Box<str>,

    /// backend
    #[arg(short, long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,
}

#[derive(Args)]
pub struct TakeColumnsArgs {
    /// Data file -- either `.zarr` or `.h5`
    data_file: Box<str>,

    /// Column indices to take: e.g., `0,1,2,3`
    #[arg(short, long, value_delimiter = ',')]
    columns: Option<Vec<usize>>,

    /// Column name file where each line is a column name
    #[arg(long)]
    name_file: Option<Box<str>>,

    /// Output file
    #[arg(short, long, required = true)]
    output: Box<str>,

    /// backend
    #[arg(short, long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,
}

#[derive(Args)]
pub struct RunBuildArgs {
    /// Matrix Market formatted data file (`.mtx.gz` or `.mtx`)
    mtx: Box<str>,

    /// Row/feature name file (name per each line; `.tsv.gz` or `.tsv`)
    #[arg(short, long)]
    row: Option<Box<str>>,

    /// Column/cell/barcode file (name per each line; `.tsv.gz` or `.tsv`)
    #[arg(short, long)]
    col: Option<Box<str>>,

    /// backend
    #[arg(long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,

    /// Output file header: {output}.{backend}
    #[arg(short, long)]
    output: Option<Box<str>>,

    /// verbose mode
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,
}

#[derive(Args)]
pub struct MergeMtxArgs {
    /// data file directories
    #[arg(value_delimiter = ',', required = true)]
    data_directories: Vec<Box<str>>,

    /// backend
    #[arg(long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,

    /// Output file header: {output}.{backend}
    #[arg(short, long, required = true)]
    output: Box<str>,

    /// matrix file name
    #[arg(short, long, default_value = "matrix.mtx")]
    mtx_file_name: Box<str>,

    /// feature/row file name
    #[arg(short, long, default_value = "genes.tsv.gz")]
    feature_file_name: Box<str>,

    /// number of words to use for feature names
    #[arg(long, default_value_t = 2)]
    num_feature_name_words: usize,

    /// barcode/column file name
    #[arg(short, long, default_value = "barcodes.tsv.gz")]
    barcode_file_name: Box<str>,

    /// number of words to use for barcode names
    #[arg(long, default_value_t = 5)]
    num_barcode_name_words: usize,

    /// verbose mode
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,
}

#[derive(Args)]
pub struct RunSqueezeArgs {
    /// Data file -- either `.zarr` or `.h5`
    data_file: Box<str>,

    /// Number of non-zero cutoff for rows
    #[arg(short, long, default_value = "0")]
    row_nnz_cutoff: usize,

    /// Number of non-zero cutoff for columns
    #[arg(short, long, default_value = "0")]
    column_nnz_cutoff: usize,

    /// Block_size for parallel processing
    #[arg(long, value_enum, default_value = "100")]
    block_size: usize,

    /// backend to use (hdf5 or zarr), default: zarr
    #[arg(short, long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,
}

#[derive(Args)]
pub struct RunStatArgs {
    /// Data file -- .zarr or .h5 file
    data_file: Box<str>,

    /// backend
    #[arg(short, long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,

    /// block_size
    #[arg(long, value_enum, default_value = "100")]
    block_size: usize,

    /// output file header
    #[arg(short, long)]
    output: Box<str>,
}

#[derive(Args)]
pub struct RunSimulateArgs {
    /// number of rows
    #[arg(short, long)]
    pub rows: usize,

    /// number of columns
    #[arg(short, long)]
    pub cols: usize,

    /// number of factors
    #[arg(short, long)]
    pub factors: Option<usize>,

    /// number of batches
    #[arg(short, long)]
    pub batches: Option<usize>,

    /// output file header: {output}.{backend}
    #[arg(short, long)]
    output: Box<str>,

    /// random seed
    #[arg(long)]
    pub seed: Option<u64>,

    /// backend
    #[arg(long, value_enum, default_value = "zarr")]
    backend: SparseIoBackend,
}

/////////////////////
// implementations //
/////////////////////

fn read_row_names(row_file: Box<str>, max_row_name_idx: usize) -> anyhow::Result<Vec<Box<str>>> {
    let (_names, _) = common_io::read_lines_of_words(&row_file, -1)?;
    Ok(_names
        .into_iter()
        .map(|x| {
            let s = (0..x.len().min(max_row_name_idx))
                .filter_map(|i| x.get(i))
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(ROW_SEP)
                .parse::<String>()
                .expect("invalid row name");
            s.into_boxed_str()
        })
        .collect())
}

fn read_col_names(col_file: Box<str>, max_column_name_idx: usize) -> anyhow::Result<Vec<Box<str>>> {
    let (_names, _) = common_io::read_lines_of_words(&col_file, -1)?;
    Ok(_names
        .into_iter()
        .map(|x| {
            let s = (0..x.len().min(max_column_name_idx))
                .filter_map(|i| x.get(i))
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(COLUMN_SEP)
                .parse::<String>()
                .expect("invalid col name");
            s.into_boxed_str()
        })
        .collect())
}

fn reorder_rows(args: &SortRowsArgs) -> anyhow::Result<()> {
    let data_file = args.data_file.clone();
    let row_names_order: Vec<Box<str>> = read_row_names(args.row_file.clone(), MAX_ROW_NAME_IDX)?;
    let backend = args.backend.clone();
    let mut data: Box<SData> = open_sparse_matrix(&data_file, &backend.clone())?;
    data.reorder_rows(&row_names_order)?;

    Ok(())
}

fn take_columns(args: &TakeColumnsArgs) -> anyhow::Result<()> {
    use matrix_util::traits::IoOps;

    let data_file = args.data_file.clone();

    let columns = args.columns.clone();
    let column_name_file = args.name_file.clone();
    let backend = args.backend.clone();
    let output = args.output.clone();

    if let Some(columns) = columns {
        let data: Box<SData> = open_sparse_matrix(&data_file, &backend.clone())?;
        data.read_columns_ndarray(columns)?.to_tsv(&output)?;
    } else if let Some(column_file) = column_name_file {
        let col_names = read_col_names(column_file, MAX_COLUMN_NAME_IDX)?;
        let data: Box<SData> = open_sparse_matrix(&data_file, &backend.clone())?;
        let col_names_map = data
            .column_names()
            .expect("column names not found in data file")
            .iter()
            .enumerate()
            .map(|(i, x)| (x.clone(), i))
            .collect::<HashMap<_, _>>();

        let col_names_order = col_names
            .into_iter()
            .filter_map(|x| col_names_map.get(&x))
            .collect::<Vec<_>>();

        let columns: Vec<usize> = col_names_order.iter().map(|&x| x.clone()).collect();
        data.read_columns_ndarray(columns)?.to_tsv(&output)?;
    } else {
        return Err(anyhow::anyhow!(
            "either `columns` or `name_file` must be provided"
        ));
    }

    Ok(())
}

fn run_merge(args: &MergeMtxArgs) -> anyhow::Result<()> {
    if args.verbose > 0 {
        std::env::set_var("RUST_LOG", "info");
    }

    use std::path::Path;

    let directories = args.data_directories.clone();

    let mut mtx_files = vec![];
    let mut row_files = vec![];
    let mut col_files = vec![];
    let mut batch_names = vec![];

    for dir in directories.iter() {
        let dir = dir.clone().into_string();

        info!("Searching subdir within: {}", &dir);

        let mut sub_dir_vec = std::fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();

        sub_dir_vec.sort_by_key(|entry| entry.file_name());

        for sub in sub_dir_vec {
            if let Some(sub_dir) = sub.path().to_str() {
                let mut mtx: Option<Box<str>> = None;
                let mut row: Option<Box<str>> = None;
                let mut col: Option<Box<str>> = None;

                if let Some(base) = Path::new(&sub_dir).file_stem() {
                    let base = base.to_str().expect("invalid base name").to_string();
                    info!("searching {} ...", &base);

                    let batch_name = Some(base);

                    for x in std::fs::read_dir(&sub_dir)? {
                        if let Some(_path) = x?.path().to_str() {
                            let _path = _path.to_string();

                            info!("found: {}", &_path);

                            if _path.ends_with(args.mtx_file_name.as_ref()) {
                                mtx = Some(_path.into_boxed_str());
                            } else if _path.ends_with(args.feature_file_name.as_ref()) {
                                row = Some(_path.into_boxed_str());
                            } else if _path.ends_with(args.barcode_file_name.as_ref()) {
                                col = Some(_path.into_boxed_str());
                            }
                        }
                    }

                    if let (Some(m), Some(r), Some(c), Some(b)) = (mtx, row, col, batch_name) {
                        info!("found: {}, {}, {} for {}", &m, &r, &c, &b);
                        mtx_files.push(m);
                        row_files.push(r);
                        col_files.push(c);
                        batch_names.push(b);
                    }
                }
            }
        }
    }

    let num_batches = batch_names.len();

    info!("merging over {} batches ...", num_batches);

    assert_eq!(num_batches, mtx_files.len());
    assert_eq!(num_batches, row_files.len());
    assert_eq!(num_batches, col_files.len());

    if num_batches == 0 {
        return Err(anyhow::anyhow!("No relevant files found"));
    }

    info!("Finding common rows/features ...");

    let mut row_hash: HashMap<Box<str>, usize> = HashMap::new();

    for i in 0..num_batches {
        let row_names = read_row_names(row_files[i].clone(), args.num_feature_name_words)?;
        for name in row_names.iter() {
            let n = row_hash.entry(name.clone()).or_insert(0);
            *n += 1;
        }
    }

    let mut common_rows: Vec<Box<str>> = row_hash
        .into_iter()
        .filter_map(|(k, v)| {
            if v == num_batches {
                Some(k.clone())
            } else {
                None
            }
        })
        .collect();

    common_rows.sort_by_key(|x| x.to_string());

    let row_pos: HashMap<Box<str>, usize> = common_rows
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();

    info!(
        "Found {} common row/feature names across {} file sets",
        row_pos.len(),
        num_batches
    );

    info!("Elongating column/barcode names ...");

    let column_names = col_files
        .iter()
        .zip(batch_names.iter())
        .map(|(col_file, batch_name)| {
            let col_names = read_col_names(col_file.clone(), args.num_barcode_name_words)
                .expect(format!("").as_str());
            col_names
                .into_iter()
                .map(|x| format!("{}{}{}", x, COLUMN_SEP, batch_name).into_boxed_str())
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    info!("Found {} columns/barcodes ...", column_names.len());

    info!("Renaming triplets...");

    let mut renamed_triplets = vec![];
    let mut offset = 0;
    let mut nnz_tot = 0;

    let pb = ProgressBar::new(num_batches as u64);

    if args.verbose > 0 {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    for b in 0..num_batches {
        let row_names = read_row_names(row_files[b].clone(), args.num_feature_name_words)?;

        let (triplets, shape) = mtx_io::read_mtx_triplets(&mtx_files[b].clone())?;

        if let Some((nrow, ncol, nnz)) = shape {
            info!(
                "{}: {} rows, {} columns, {} non-zeros",
                &mtx_files[b], nrow, ncol, nnz
            );

            let triplets = triplets
                .into_iter()
                .filter_map(|(batch_i, batch_j, x_ij)| {
                    if let Some(i) = row_pos.get(&row_names[batch_i as usize]) {
                        let i = *i as u64;
                        let j = batch_j + offset;
                        Some((i, j, x_ij))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            nnz_tot += triplets.len();
            renamed_triplets.extend(triplets);

            let batch_col_names =
                read_col_names(col_files[b].clone(), args.num_barcode_name_words)?;
            let batch_name = batch_names[b].clone();

            (0..ncol).into_iter().for_each(|batch_j| {
                let loc = format!("{}{}{}", batch_col_names[batch_j], COLUMN_SEP, batch_name)
                    .into_boxed_str();

                let glob_j = offset as usize + batch_j;
                let glob = column_names[glob_j as usize].clone();
                assert_eq!(glob, loc);
            });

            offset += ncol as u64;
            pb.inc(1);
        }
    }
    pb.finish_and_clear();

    let backend = args.backend.clone();
    let output = args.output.clone();

    let backend_file = match backend {
        SparseIoBackend::HDF5 => format!("{}.h5", &output),
        SparseIoBackend::Zarr => format!("{}.zarr", &output),
    };

    if Path::new(&backend_file).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        common_io::remove_file(&backend_file)?;
    }

    let mut data = create_sparse_from_triplets(
        renamed_triplets,
        (row_pos.len(), offset as usize, nnz_tot),
        Some(&backend_file),
        Some(&backend),
    )?;

    data.register_row_names_vec(&common_rows);
    data.register_column_names_vec(&column_names);

    Ok(())
}

fn run_build(args: &RunBuildArgs) -> anyhow::Result<()> {
    if args.verbose > 0 {
        std::env::set_var("RUST_LOG", "info");
    }

    use std::path::Path;

    let mtx_file = args.mtx.as_ref();
    let row_file = args.row.as_ref();
    let col_file = args.col.as_ref();

    let backend = args.backend.clone();

    let output = match args.output.clone() {
        Some(output) => output,
        None => {
            let (dir, mut base, ext) = common_io::dir_base_ext(mtx_file)?;

            if base.ends_with(".mtx") && ext.ends_with("gz") {
                base = base
                    .into_string()
                    .trim_end_matches(".mtx")
                    .to_string()
                    .into_boxed_str();
            }

            match (dir.len(), base.len()) {
                (0, 0) => format!("./").into_boxed_str(),
                (0, _) => format!("./{}", base).into_boxed_str(),
                _ => format!("{}/{}", dir, base).into_boxed_str(),
            }
        }
    };

    let backend_file = match backend {
        SparseIoBackend::HDF5 => format!("{}.h5", &output),
        SparseIoBackend::Zarr => format!("{}.zarr", &output),
    };

    if Path::new(&backend_file).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        common_io::remove_file(&backend_file)?;
    }

    let mut data = create_sparse_from_mtx_file(&mtx_file, Some(&backend_file), Some(&backend))?;

    if let Some(row_file) = row_file {
        data.register_row_names_file(row_file);
    } else {
        if let Some(nrow) = data.num_rows() {
            let row_names: Vec<Box<str>> =
                (1..(nrow + 1)).map(|i| format!("{}", i).into()).collect();
            data.register_row_names_vec(&row_names);
        }
    }

    if let Some(col_file) = col_file {
        data.register_column_names_file(col_file);
    } else {
        if let Some(ncol) = data.num_columns() {
            let col_names: Vec<Box<str>> =
                (1..(ncol + 1)).map(|i| format!("{}", i).into()).collect();
            data.register_column_names_vec(&col_names);
        }
    }

    Ok(())
}

fn run_stat(cmd_args: &RunStatArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    common_io::mkdir(&output)?;

    let input = cmd_args.data_file.clone();
    let backend = cmd_args.backend.clone();
    let block_size = cmd_args.block_size;

    match common_io::extension(&input)?.as_ref() {
        "zarr" => {
            assert_eq!(backend, SparseIoBackend::Zarr);
        }
        "h5" => {
            assert_eq!(backend, SparseIoBackend::HDF5);
        }
        _ => return Err(anyhow::anyhow!("Unknown file format: {}", input)),
    }

    let data: Box<SData> = open_sparse_matrix(&input, &backend.clone())?;
    let row_names = data.row_names()?;
    let col_names = data.column_names()?;

    if let Ok((row_stat, col_stat)) = collect_row_column_stats(&data, block_size) {
        let row_stat_file = format!("{}.row.stat.gz", output);
        let col_stat_file = format!("{}.col.stat.gz", output);
        row_stat.save(&row_stat_file, &row_names, "\t")?;
        col_stat.save(&col_stat_file, &col_names, "\t")?;
    }

    Ok(())
}

fn run_squeeze(cmd_args: &RunSqueezeArgs) -> anyhow::Result<()> {
    let data_file = cmd_args.data_file.clone();
    let row_nnz_cutoff = cmd_args.row_nnz_cutoff;
    let col_nnz_cutoff = cmd_args.column_nnz_cutoff;

    let block_size = cmd_args.block_size;
    let backend = cmd_args.backend.clone();

    match common_io::extension(&data_file)?.as_ref() {
        "zarr" => {
            assert_eq!(backend, SparseIoBackend::Zarr);
        }
        "h5" => {
            assert_eq!(backend, SparseIoBackend::HDF5);
        }
        _ => return Err(anyhow::anyhow!("Unknown file format: {}", data_file)),
    }

    let mut data = open_sparse_matrix(&data_file, &backend)?;

    info!(
        "before squeeze -- data: {} rows x {} columns",
        data.num_rows().unwrap(),
        data.num_columns().unwrap()
    );

    if let Ok((row_stat, col_stat)) = collect_row_column_stats(&data, block_size) {
        fn nnz_index(nnz: &Vec<f32>, cutoff: usize) -> Vec<usize> {
            nnz.iter()
                .enumerate()
                .filter_map(|(i, &x)| if (x as usize) > cutoff { Some(i) } else { None })
                .collect()
        }

        let row_nnz_vec = row_stat.count_positives().to_vec();
        let row_idx = nnz_index(&row_nnz_vec, row_nnz_cutoff);
        let col_nnz_vec = col_stat.count_positives().to_vec();
        let col_idx = nnz_index(&col_nnz_vec.to_vec(), col_nnz_cutoff);

        data.subset_columns_rows(Some(&col_idx), Some(&row_idx))?;

        info!(
            "after squeeze -- data: {} rows x {} columns",
            data.num_rows().unwrap(),
            data.num_columns().unwrap()
        );
    }

    Ok(())
}

fn run_simulate(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    common_io::mkdir(&output)?;

    let backend = cmd_args.backend.clone();

    let backend_file = match backend {
        SparseIoBackend::HDF5 => output.to_string() + ".h5",
        SparseIoBackend::Zarr => output.to_string() + ".zarr",
    };

    let mtx_file = output.to_string() + ".mtx.gz";
    let dict_file = mtx_file.replace(".mtx.gz", ".dict.gz");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.gz");
    let memb_file = mtx_file.replace(".mtx.gz", ".memb.gz");
    let ln_batch_file = mtx_file.replace(".mtx.gz", ".ln_batch.gz");

    common_io::remove_all_files(&vec![
        backend_file.clone().into_boxed_str(),
        mtx_file.clone().into_boxed_str(),
        dict_file.clone().into_boxed_str(),
        prop_file.clone().into_boxed_str(),
        memb_file.clone().into_boxed_str(),
        ln_batch_file.clone().into_boxed_str(),
    ])
    .expect("failed to clean up existing output files");

    let sim_args = simulate::SimArgs {
        rows: cmd_args.rows,
        cols: cmd_args.cols,
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        rseed: cmd_args.seed,
    };

    let sim = simulate::generate_factored_poisson_gamma_data(&sim_args);
    info!("successfully generated factored Poisson-Gamma data");

    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    common_io::write_lines(&batch_out, &memb_file)?;
    info!("batch membership: {:?}", &memb_file);

    sim.ln_delta_db.to_tsv(&ln_batch_file)?;
    sim.theta_kn.transpose().to_tsv(&prop_file)?;
    sim.beta_dk.to_tsv(&dict_file)?;

    info!(
        "wrote parameter files:\n{:?},\n{:?},\n{:?}",
        &ln_batch_file, &dict_file, &prop_file
    );

    let mtx_shape = (sim_args.rows, sim_args.cols, sim.triplets.len());

    info!("registering triplets ...");

    let mut data =
        create_sparse_from_triplets(sim.triplets, mtx_shape, Some(&backend_file), Some(&backend))?;

    info!("created sparse matrix: {}", backend_file);

    let rows: Vec<Box<str>> = (1..(sim_args.rows + 1))
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (1..(sim_args.cols + 1))
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);

    info!("done");
    Ok(())
}

/// Collect row and column statistics returns (row_stat, col_stat)
/// - row_stat: mean, variance, min, max
/// - col_stat: mean, variance, min, max
///
/// # Arguments
/// * data - sparse matrix data
/// * block_size - block size for parallel computation
///
fn collect_row_column_stats(
    data: &Box<SData>,
    block_size: usize,
) -> anyhow::Result<(RunningStatistics<Ix1>, RunningStatistics<Ix1>)> {
    if let (Some(nrow), Some(ncol)) = (data.num_rows(), data.num_columns()) {
        let mut row_stat = RunningStatistics::new(Ix1(nrow));
        let mut col_stat = RunningStatistics::new(Ix1(ncol));

        let arc_row_stat = Arc::new(Mutex::new(&mut row_stat));
        let arc_col_stat = Arc::new(Mutex::new(&mut col_stat));

        let nblock = (ncol + block_size - 1) / block_size;

        info!(
            "collecting row and column statistics over {} blocks",
            nblock
        );

        let arc_data = Arc::new(Mutex::new(data));

        (0..nblock)
            .into_par_iter()
            .progress_count(nblock as u64)
            .map(|b| {
                let lb: usize = b * block_size;
                let ub: usize = ((b + 1) * block_size).min(ncol);
                (lb, ub)
            })
            .for_each(|(lb, ub)| {
                let data_b = arc_data.lock().expect("failed to lock data");

                // This could be inefficient since we are populating a dense matrix
                let xx_b = data_b.read_columns_ndarray((lb..ub).collect()).unwrap();

                // accumulate rows' statistics
                {
                    let mut row_stat = arc_row_stat.lock().expect("failed to lock row_stat");
                    for x in xx_b.axis_iter(Axis(1)) {
                        row_stat.add(&x);
                    }
                }

                // accumulate columns' statistics
                {
                    let mut col_stat = arc_col_stat.lock().expect("failed to lock col_stat");

                    for x in xx_b.axis_iter(Axis(0)) {
                        for j in lb..ub {
                            let i = j - lb;
                            col_stat.add_element(&[j], x[i]);
                        }
                    }
                }
            }); // end of jobs

        Ok((row_stat, col_stat))
    } else {
        anyhow::bail!("No row/column info");
    }
}
