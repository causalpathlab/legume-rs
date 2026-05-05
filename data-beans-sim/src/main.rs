use clap::{Parser, Subcommand};

use data_beans_sim::deconv::{generate_convoluted_data, SimConvArgs};
use data_beans_sim::handlers::{
    run_simulate, run_simulate_multimodal, RunSimulateArgs, RunSimulateMultimodalArgs,
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
        Commands::Topic(args) => run_simulate(args)?,
        Commands::Bulk(args) => generate_convoluted_data(args)?,
        Commands::Multimodal(args) => run_simulate_multimodal(args)?,
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
        about = "Gamma topic factor model (with optional reference-conditioned two-stage NB+copula sampling)",
        long_about = "Synthetic mode (no `--reference`):\n\
                      `Y(i,j) ~ Poisson(depth · δ(i,B(j)) · Σ_k β(i,k) θ(j,k))`\n\
                      with β,θ ~ Gamma; ln δ ~ N(0,1) z-scored per batch.\n\
                      \n\
                      Reference mode (`--reference <h5/zarr>`): two-stage GLM —\n\
                        stage 1:  log λ⁰ = log μ̂_g + √pve_topic·t + √pve_noise·ε\n\
                        stage 2:  log λ  = log λ⁰ + √pve_batch·δ_{g,b}\n\
                        sample :  y ~ NB(λ, r̂_g)  via  u=Φ(z*),  F⁻¹_NB(u; λ, r̂)\n\
                      where t is z-scored log(β·θ) per cell, ε iid N(0,1), and δ has unit\n\
                      per-gene variance under a low-rank-plus-residual covariance whose\n\
                      rank is `--batch-rank` (0=Splatter-style iid; 2-3=structured batch\n\
                      program along axes chosen by `--batch-program {random,biology}`).\n\
                      `--depth` is reinterpreted as a multiplicative scale so library size\n\
                      matches the reference's mean. scDesign / scDesign2 / scDesign3 lineage."
    )]
    Topic(RunSimulateArgs),

    #[command(about = "Bulk (convoluted) data matrix from real SC reference (experimental)")]
    Bulk(SimConvArgs),

    #[command(
        about = "Multimodal count data with shared base + delta dictionaries",
        long_about = "Generate M count matrices from shared latent topics with modality-specific\n\
                      dictionaries: reference = softmax(W_base), others = softmax(W_base + W_delta_m).\n\
                      Delta is sparse (spike-and-slab): n_delta_features genes per topic are perturbed."
    )]
    Multimodal(RunSimulateMultimodalArgs),
}
