mod common_io;
mod mtx_io;
mod simulate;
mod sparse_io;
mod sparse_matrix_hdf5;
mod sparse_matrix_zarr;
mod statistics;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(version, about, long_about=None)]
#[command(propagate_version = true)]
///
/// `asap-data` utility for processing sparse matrices
/// - build: build from .mtx fileset to another
/// - subset: subset a sparse matrix
/// - merge: merge multiple sparse matrices
/// - stat: compute basic statistics of a sparse matrix
/// - simulate: simulate a sparse matrix
///
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(ValueEnum, Clone, Debug)]
#[clap(rename_all = "lowercase")]
enum Backend {
    Zarr,
    HDF5,
}

#[derive(Subcommand)]
enum Commands {
    /// build from one format to another
    Build(RunBuildArgs),
    /// subset a sparse matrix
    Subset(RunSubsetArgs),
    /// stat a sparse matrix
    Stat(RunStatArgs),
    /// simulate a sparse matrix data
    Simulate(RunSimulateArgs),
}

#[derive(Args)]
pub struct RunStatArgs {
    /// input .zarr or .h5 file
    #[arg(short, long)]
    input: Box<str>,

    /// output file header
    #[arg(short, long)]
    output: Box<str>,
}

fn run_stat(cmd_args: &RunStatArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    common_io::mkdir(&output)?;

    Ok(())
}

#[derive(Args)]
pub struct RunSubsetArgs {
    /// input .zarr or .h5 file
    #[arg(short, long)]
    input: Box<str>,

    /// output file header
    #[arg(short, long)]
    output: Box<str>,
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

    /// backend to use (HDF5 or Zarr)
    #[arg(short, long, value_enum)]
    backend: Backend,

    /// output file header: {output}.{backend}
    #[arg(short, long)]
    output: Box<str>,
}

fn run_build(args: &RunBuildArgs) -> anyhow::Result<()> {
    let mtx_file = args.mtx.as_ref();
    let row_file = args.row.as_ref();
    let col_file = args.col.as_ref();

    let backend_file = match args.backend {
        Backend::HDF5 => format!("{}.h5", args.output),
        Backend::Zarr => format!("{}.zarr", args.output),
    };
    let backend_file = backend_file.as_ref();

    let mut x =
        sparse_matrix_hdf5::SparseMtxData::from_mtx_file(mtx_file, Some(backend_file), None)?;

    x.register_row_names(row_file);
    x.register_column_names(col_file);

    // match args.backend {
    // 	Backend::HDF5 => {
    // 		let h5_file = args.output.as_ref();
    // 		let mut mtx_data = sparse_matrix_hdf5::SparseMtxData::new(h5_file)?;
    // 		mtx_data.load_mtx_hdf5(mtx_file)?;
    // 	}
    // 	Backend::Zarr => {
    // 		let zarr_backend = args.output.as_ref();
    // 		let mut mtx_data = sparse_matrix_zarr::SparseMtxData::create(zarr_backend)?;
    // 		mtx_data.read_mtx_by_row(mtx_file)?;
    // 	}
    // }

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

    /// backend to use (HDF5 or Zarr), default: HDF5
    #[arg(long, value_enum)]
    backend: Option<Backend>,
}

fn run_simulate(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    common_io::mkdir(&output)?;

    let backend = cmd_args.backend.clone().unwrap_or(Backend::HDF5);

    let backend_file = match backend {
        Backend::HDF5 => output.to_string() + ".h5",
        Backend::Zarr => output.to_string() + ".zarr",
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
    match backend {
        Backend::HDF5 => {
            use sparse_matrix_hdf5::SparseMtxData;
            let _ = SparseMtxData::from_mtx_file(&mtx_file, Some(&backend_file), Some(true))?;
        }
        Backend::Zarr => {
            use sparse_matrix_zarr::SparseMtxData;
            let _ = SparseMtxData::from_mtx_file(&mtx_file, Some(&backend_file), Some(true))?;
        }
    }

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
