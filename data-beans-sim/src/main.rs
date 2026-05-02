use clap::{Parser, Subcommand};

use data_beans_sim::deconv::{generate_convoluted_data, SimConvArgs};
use data_beans_sim::handlers::{
    run_copula_sim, run_simulate, run_simulate_multimodal, CopulaSimArgs, RunSimulateArgs,
    RunSimulateMultimodalArgs,
};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    if let Some(n) = cli.n_threads {
        if n == 0 {
            anyhow::bail!("--n-threads must be >= 1");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    match &cli.commands {
        Commands::Simulate(args) => run_simulate(args)?,
        Commands::SimulateConv(args) => generate_convoluted_data(args)?,
        Commands::SimulateMultimodal(args) => run_simulate_multimodal(args)?,
        Commands::CopulaSim(args) => run_copula_sim(args)?,
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Synthetic single-cell-like data generators for the data-beans ecosystem.",
    long_about = "data-beans-sim: simulate sparse count matrices with factor-model, CNV, multimodal,\n\
                  and topic-conditioned bulk-deconvolution structure. Outputs are written in the same\n\
                  zarr/h5 backend formats consumed by `data-beans` so that simulated and real datasets\n\
                  share the same downstream tooling."
)]
struct Cli {
    #[arg(short = 'v', long, global = true)]
    verbose: bool,

    #[arg(
        long = "n-threads",
        visible_aliases = ["threads", "num-threads"],
        global = true,
        value_name = "N",
        help = "Limit the number of CPU threads"
    )]
    n_threads: Option<usize>,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Simulate matrix with Gamma topic",
        long_about = "`Y(i,j) ~ δ(i,B(j)) Σ β(i,k) θ(j,k)` with β,θ ~ Gamma topic;\n\
		      B(j)`=batch; `ln δ`~N(0,1)"
    )]
    Simulate(RunSimulateArgs),

    #[command(
        about = "Simulate convoluted (bulk) data matrix from real SC reference (experimental)"
    )]
    SimulateConv(SimConvArgs),

    #[command(
        about = "Simulate multimodal count data with shared base + delta dictionaries",
        long_about = "Generate M count matrices from shared latent topics with modality-specific\n\
                      dictionaries: reference = softmax(W_base), others = softmax(W_base + W_delta_m).\n\
                      Delta is sparse (spike-and-slab): n_delta_features genes per topic are perturbed."
    )]
    SimulateMultimodal(RunSimulateMultimodalArgs),

    #[command(
        name = "copula-sim",
        about = "Reference-conditioned copula simulator (scDesign / scDesign2 / scDesign3 style)",
        long_about = "Fit per-gene NB marginals + a Gaussian copula on a real reference dataset,\n\
                      then sample synthetic counts that preserve gene-gene dependence. Optionally\n\
                      partition cells via SVD + leiden (or an external label file) and layer batch,\n\
                      CNV (clonal tree), and housekeeping effects on top of the copula draws.\n\
                      \n\
                      References:\n\
                      - Li & Li 2019, Bioinformatics 35(14):i41-i50 (scDesign)\n\
                      - Sun, Song, Li & Li 2021, Genome Biology 22:163 (scDesign2)\n\
                      - Song et al. 2024, Nat Biotechnology 42(2):247-252 (scDesign3)"
    )]
    CopulaSim(CopulaSimArgs),
}
