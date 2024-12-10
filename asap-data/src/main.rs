mod common_io;
mod mtx_io;
mod simulate;
mod sparse_io;
mod sparse_matrix_hdf5;
mod sparse_matrix_zarr;
mod statistics;

use crate::common_io::write_lines;
use crate::sparse_io::*;
use crate::statistics::RunningStatistics;
use clap::{Args, Parser, Subcommand};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(version, about, long_about=None)]
#[command(propagate_version = true)]
///
/// `asap-data` utility for processing sparse matrices
/// - build: build from .mtx fileset to another
/// - stat: compute basic statistics of a sparse matrix
/// - simulate: simulate a sparse matrix
///
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// build from one format to another
    Build(RunBuildArgs),
    /// filter out rows and columns (Q/C)
    Filter(RunFilterArgs),
    /// stat a sparse matrix
    Stat(RunStatArgs),
    /// simulate a sparse matrix data
    Simulate(RunSimulateArgs),
}

#[derive(Args)]
pub struct RunFilterArgs {
    /// input .zarr or .h5 file
    #[arg(short, long)]
    input: Box<str>,

    /// output file header
    #[arg(short, long)]
    output: Box<str>,
}

fn run_filtering(cmd_args: &RunFilterArgs) -> anyhow::Result<()> {
    todo!("need to implement add and remove row/columns");

// 1. compute scores

// 2. subset

    Ok(())
}

#[derive(Args)]
pub struct RunStatArgs {
    /// input .zarr or .h5 file
    #[arg(short, long)]
    input: Box<str>,

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

fn run_stat(cmd_args: &RunStatArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    common_io::mkdir(&output)?;

    let input = cmd_args.input.clone();
    let backend = cmd_args.backend.clone().unwrap_or(SparseIoBackend::Zarr);

    let data = open_sparse_matrix(&input, &backend)?;

    let block_size = cmd_args.block_size.unwrap_or(100);

    use ndarray::Ix1;

    if let (Some(nrow), Some(ncol)) = (data.num_rows(), data.num_columns()) {
        let row_names = data.row_names().expect("no row names");
        let col_names = data.column_names().expect("no col names");

        let arc_row_stat = Arc::new(Mutex::new(RunningStatistics::new(Ix1(nrow))));

        {
            let nblock = (ncol + block_size - 1) / block_size;
            let arc_data = Arc::new(Mutex::new(data));
            let jobs = (0..nblock).into_par_iter().map(|b| {
                let lb: usize = b * block_size;
                let ub: usize = ((b + 1) * block_size).min(ncol);
                (lb, ub)
            });

            let arc_col_stat_map = Arc::new(Mutex::new(HashMap::<usize, Box<str>>::new()));

            jobs.for_each(|(lb, ub)| {
                //////////////////////
                // take subset data //
                //////////////////////
                let nn_b = ub - lb;

                let data_b = arc_data.lock().expect("failed to lock data");
                let xx_b = data_b.read_columns((lb..ub).collect()).unwrap();

                // accumulate rows' statistics
                {
                    let mut row_stat = arc_row_stat.lock().expect("failed to lock row_stat");
                    for x in xx_b.axis_iter(Axis(1)) {
                        row_stat.add(&x);
                    }
                }

                let mut _names = vec![];
                for i in lb..ub {
                    _names.push(col_names[i].clone());
                }

                // accumulate columns' statistics
                {
                    let mut col_stat_map = arc_col_stat_map
                        .lock()
                        .expect("failed to lock col_stat_lines");

                    let mut _stat = RunningStatistics::new(Ix1(nn_b));
                    for x in xx_b.axis_iter(Axis(0)) {
                        _stat.add(&x);
                    }

                    let _lines = _stat
                        .to_string_vec(&_names, "\t")
                        .expect("failed to generate column stat lines");

                    for (glob, line) in (lb..ub).zip(_lines.iter()) {
                        col_stat_map.insert(glob, line.clone().into());
                    }
                }
            }); // end of jobs

            {
                let row_stat = arc_row_stat.lock().expect("failed to lock row_stat");
                let row_stat_file = format!("{}.row.stat.gz", output);
                row_stat.save(&row_stat_file, &row_names, "\t")?;
            }
            {
                let col_stat_file = format!("{}.column.stat.gz", output);
                let col_stat_map = arc_col_stat_map
                    .lock()
                    .expect("failed to lock col_stat_lines");

                let mut sorted: Vec<(usize, Box<str>)> = col_stat_map.clone().into_iter().collect();
                sorted.sort_by_key(|&(k, _)| k);
                let lines: Vec<Box<str>> = sorted.into_iter().map(|(_, v)| v).collect();
                write_lines(&lines, &col_stat_file)?;
            }
        }
    } else {
        return Err(anyhow::anyhow!(
            "failed to get the number of rows and columns"
        ));
    }

    Ok(())
}

#[derive(Args)]
pub struct RunBuildArgs {
    /// input `.mtx.gz` or `.mtx` file
    #[arg(short, long)]
    mtx: Box<str>,

    /// row/feature names `.tsv.gz` or `.tsv` file
    #[arg(short, long)]
    row: Box<str>,

    /// column/cell/barcode names `.tsv.gz` or `.tsv` file
    #[arg(short, long)]
    col: Box<str>,

    /// backend to use (HDF5 or Zarr), default: Zarr
    #[arg(short, long, value_enum)]
    backend: Option<SparseIoBackend>,

    /// output file header: {output}.{backend}
    #[arg(short, long)]
    output: Box<str>,
}

fn run_build(args: &RunBuildArgs) -> anyhow::Result<()> {
    let mtx_file = args.mtx.as_ref();
    let row_file = args.row.as_ref();
    let col_file = args.col.as_ref();

    let backend = args.backend.clone().unwrap_or(SparseIoBackend::Zarr);
    let backend_file = match backend {
        SparseIoBackend::HDF5 => format!("{}.h5", args.output),
        SparseIoBackend::Zarr => format!("{}.zarr", args.output),
    };
    let backend_file = backend_file.as_ref();

    let mut data = create_sparse_matrix(&mtx_file, &backend_file, &backend)?;

    data.register_row_names_file(row_file);
    data.register_column_names_file(col_file);

    Ok(())
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

    let rows: Vec<Box<str>> = (0..sim_args.rows)
        .map(|i| (i + 1).to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (0..sim_args.cols)
        .map(|i| (i + 1).to_string().into_boxed_str())
        .collect();

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);

    Ok(())
}

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
        _ => {
            todo!("not implemented yet");
        }
    }

    // let mtx_file = args().nth(1).expect("missing mtx file");

    // let zarr_backend = "temp.zarr";

    // if fs::metadata(zarr_backend).is_ok() {
    //     // fs::remove_file(zarr_backend)?;
    //     fs::remove_dir_all(zarr_backend)?;
    // }

    // let mut mtx_data = sparse_matrix::SparseMtxData::create(zarr_backend)?;

    // mtx_data.read_mtx_by_row(mtx_file.as_ref())?;

    // let h5_file = "temp.h5";
    // if fs::metadata(h5_file).is_ok() {
    //     fs::remove_file(h5_file)?;
    // }

    // let mut mtx_data = cell_feature_matrix::SparseMtxData::new(h5_file)?;
    // mtx_data.load_mtx_hdf5(&mtx_file)?;

    Ok(())
}
