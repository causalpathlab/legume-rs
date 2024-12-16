use clap::{Args, Parser, Subcommand};
use std::sync::{Arc, Mutex};

mod common_io;
mod mtx_io;
mod simulate;
mod sparse_io;
mod sparse_matrix_hdf5;
mod sparse_matrix_zarr;
mod statistics;

use crate::sparse_io::*;
type SData = dyn SparseIo<IndexIter = Vec<usize>>;
use crate::statistics::RunningStatistics;

fn main() -> anyhow::Result<()> {
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
        Commands::Squeeze(args) => {
            run_squeeze(args)?;
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
    /// Filter out rows and columns (Q/C)
    Squeeze(RunSqueezeArgs),
    /// Take basic statistics from a sparse matrix
    Stat(RunStatArgs),
    /// Simulate a sparse matrix data
    Simulate(RunSimulateArgs),
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

    /// Backend to use (HDF5 or Zarr), default: Zarr
    #[arg(long, value_enum)]
    backend: Option<SparseIoBackend>,

    /// Output file header: {output}.{backend}
    #[arg(short, long)]
    output: Option<Box<str>>,
}

#[derive(Args)]
pub struct RunSqueezeArgs {
    /// Data file -- either `.zarr` or `.h5`
    data_file: Box<str>,

    /// Number of non-zero cutoff for rows (default: 0)
    #[arg(short, long)]
    row_nnz_cutoff: Option<usize>,

    /// Number of non-zero cutoff for columns (default: 0)
    #[arg(short, long)]
    column_nnz_cutoff: Option<usize>,

    /// Block_size for parallel processing (default: 100)
    #[arg(long, value_enum)]
    block_size: Option<usize>,

    /// backend to use (HDF5 or Zarr), default: Zarr
    #[arg(short, long, value_enum)]
    backend: Option<SparseIoBackend>,
}

#[derive(Args)]
pub struct RunStatArgs {
    /// Data file -- .zarr or .h5 file
    data_file: Box<str>,

    /// backend to use (HDF5 or Zarr), default: Zarr
    #[arg(short, long, value_enum)]
    backend: Option<SparseIoBackend>,

    /// block_size, default: 100
    #[arg(long, value_enum)]
    block_size: Option<usize>,

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

    /// backend to use (HDF5 or Zarr), default: Zarr
    #[arg(long, value_enum)]
    backend: Option<SparseIoBackend>,
}

/////////////////////
// implementations //
/////////////////////

fn run_build(args: &RunBuildArgs) -> anyhow::Result<()> {
    use std::path::Path;

    let mtx_file = args.mtx.as_ref();
    let row_file = args.row.as_ref();
    let col_file = args.col.as_ref();

    let backend = args.backend.clone().unwrap_or(SparseIoBackend::Zarr);

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
        eprintln!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        common_io::remove_file(&backend_file)?;
    }

    let mut data = create_sparse_matrix(&mtx_file, &backend_file, &backend)?;

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
    let backend = cmd_args.backend.clone().unwrap_or(SparseIoBackend::Zarr);
    let block_size = cmd_args.block_size.unwrap_or(100);

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
    let row_nnz_cutoff = cmd_args.row_nnz_cutoff.unwrap_or(0);
    let col_nnz_cutoff = cmd_args.column_nnz_cutoff.unwrap_or(0);

    let block_size = cmd_args.block_size.unwrap_or(100);
    let backend = cmd_args.backend.clone().unwrap_or(SparseIoBackend::Zarr);

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

    eprintln!(
        "data: {} rows x {} columns",
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

        eprintln!(
            "data: {} rows x {} columns",
            data.num_rows().unwrap(),
            data.num_columns().unwrap()
        );
    }

    Ok(())
}

fn run_simulate(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    common_io::mkdir(&output)?;

    let backend = cmd_args.backend.clone().unwrap_or(SparseIoBackend::Zarr);

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

    simulate::generate_factored_gamma_data_mtx(
        &sim_args,
        &mtx_file,
        &dict_file,
        &prop_file,
        &ln_batch_file,
        &memb_file,
    )
    .expect("something went wrong in factored gamma");

    let mut data = create_sparse_matrix(&mtx_file, &backend_file, &backend)?;

    let rows: Vec<Box<str>> = (1..(sim_args.rows + 1))
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (1..(sim_args.cols + 1))
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);

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
        let arc_row_stat = Arc::new(Mutex::new(RunningStatistics::new(Ix1(nrow))));
        let arc_col_stat = Arc::new(Mutex::new(RunningStatistics::new(Ix1(ncol))));

        let nblock = (ncol + block_size - 1) / block_size;
        let arc_data = Arc::new(Mutex::new(data));

        (0..nblock)
            .into_par_iter()
            .map(|b| {
                let lb: usize = b * block_size;
                let ub: usize = ((b + 1) * block_size).min(ncol);
                (lb, ub)
            })
            .for_each(|(lb, ub)| {
                let data_b = arc_data.lock().expect("failed to lock data");

                // This could be inefficient since we are populating a dense matrix
                let xx_b = data_b.read_columns((lb..ub).collect()).unwrap();

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

        let row_stat = arc_row_stat.lock().expect("failed to lock row_stat");
        let col_stat = arc_col_stat.lock().expect("failed to lock col_stat");

        let ret_row = row_stat.clone();
        let ret_col = col_stat.clone();

        Ok((ret_row, ret_col))
    } else {
        anyhow::bail!("No row/column info");
    }
}
