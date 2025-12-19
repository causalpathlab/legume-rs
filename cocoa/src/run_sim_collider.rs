use crate::common::*;

#[derive(Parser, Debug, Clone)]
pub struct SimColliderArgs {
    #[arg(
        short = 'r',
        required = true,
        help = "number of rows/genes",
        long_help = "Number of rows/genes/features"
    )]
    n_genes: usize,

    #[arg(
        short = 'c',
        required = true,
        help = "number of columns/cells",
        long_help = "Number of columns/cells"
    )]
    n_cells: usize,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1.0,1.0",
        help = "hyperparameter for gamma distribution",
        long_help = "Hyperparameter for gamma distribution"
    )]
    gamma_hyperparam: Vec<f32>,

    #[arg(
        long,
        default_value_t = 42,
        help = "random seed",
        long_help = "Random seed"
    )]
    rseed: u64,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "backend",
        long_help = "Backend"
    )]
    backend: SparseIoBackend,

    #[arg(
        long,
        default_value_t = false,
        help = "save mtx",
        long_help = "Save mtx"
    )]
    save_mtx: bool,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header"
    )]
    out: Box<str>,

    #[arg(long, short, help = "verbosity", long_help = "Set RUST_LOG=`info`")]
    verbose: bool,
}

pub fn run_sim_collider_data(args: SimColliderArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    unimplemented!("");

    Ok(())
}
