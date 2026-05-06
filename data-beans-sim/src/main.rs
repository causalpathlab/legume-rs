use clap::{Parser, Subcommand};

use data_beans_sim::deconv::{generate_convoluted_data, SimConvArgs};
use data_beans_sim::handlers::{
    run_simulate, run_simulate_multimodal, RunSimulateArgs, RunSimulateMultimodalArgs,
};
use data_beans_sim::multiome::{run_multiome, MultiomeArgs};

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
        Commands::Multiome(args) => run_multiome(args)?,
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
        about = "Log-normal topic factor model (with optional reference-conditioned NB+copula sampling)",
        long_about = "Synthetic mode (no `--reference`):\n\
                      `Y(g,j) ~ Poisson( (depth/G) · δ(g,B(j)) · Σ_k β(g,k) θ(k,j) )`\n\
                      with explicit log-space variance decomposition for both β and δ:\n\
                        log β(g,k) = σ_β · [√pve_topic · u_{g,k} + √(1−pve_topic) · v_g] − σ_β²/2\n\
                        log δ(g,b) =        √pve_batch · z_{g,b} + √(1−pve_batch) · w_g\n\
                      with u, v, z, w ~ N(0, 1) iid. `--beta-scale` controls σ_β. `pve_topic`\n\
                      and `pve_batch` are independent variance shares (both can be 1). `depth`\n\
                      is the **expected** library size (emergent — no per-cell rescaling).\n\
                      \n\
                      Reference mode (`--reference <h5/zarr>`): two-stage GLM, NB+copula sampling —\n\
                        stage 1:  log λ⁰ = log μ̂_g + √pve_topic·t + √pve_noise·ε\n\
                        stage 2:  log λ  = log λ⁰ + √pve_batch·z_{g,b} + √(1−pve_batch)·w_g\n\
                        sample :  y ~ NB(λ, r̂_g)  via  u=Φ(z*),  F⁻¹_NB(u; λ, r̂)\n\
                      where t is z-scored log(β·θ) per cell with β drawn the same way as in\n\
                      synthetic mode; ε iid N(0, 1); z_{g,b} from the gene-gene copula factor\n\
                      (rank `--batch-rank`, axes chosen by `--batch-program {random,biology}`);\n\
                      w_g iid N(0, 1) gives the batch-invariant per-gene shift. `--depth` is\n\
                      reinterpreted as a multiplicative scale so library size matches the\n\
                      reference's mean. scDesign / scDesign2 / scDesign3 lineage."
    )]
    Topic(RunSimulateArgs),

    #[command(about = "Bulk (convoluted) data matrix from real SC reference (experimental)")]
    Bulk(SimConvArgs),

    #[command(
        about = "Multimodal count data with shared base + delta dictionaries",
        long_about = "Generate M count matrices from shared latent topics θ with modality-specific\n\
                      dictionaries:\n\
                        β_0(:,k) = softmax_g( W_base[k,:]            )    (reference modality)\n\
                        β_m(:,k) = softmax_g( W_base[k,:] + Δ_m[k,:] )    (m = 1..M-1)\n\
                      where W_base ~ N(0, base_scale²) (or stick-breaking-derived logits in\n\
                      hierarchical mode), and Δ_m is sparse spike-and-slab: `n_delta_features`\n\
                      genes per topic carry an iid N(0, delta_scale²) perturbation; the rest\n\
                      are zero.\n\
                      \n\
                      Per-modality counts (batch effects independent per modality unless\n\
                      `--shared-batch-effects`):\n\
                        log δ_m(g,b) = √pve_batch · z_{g,b} + √(1−pve_batch) · w_g\n\
                        Y_m(g,j)    ~ Poisson( depth_m · δ_m(g,B(j)) · Σ_k β_m(g,k) θ(k,j) )\n\
                      depth_m is the **expected** library size for modality m (emergent — no\n\
                      per-cell rescaling). Because each β_m(:,k) sums to 1 over genes, the\n\
                      cell-level total of (β·θ) sums to 1 deterministically and `depth_m`\n\
                      directly sets E[lib(j) | m]."
    )]
    Multimodal(RunSimulateMultimodalArgs),

    #[command(
        about = "Paired ATAC + RNA simulator with peak-gene ground truth and optional per-modality NB+copula sampling",
        long_about = "Synthetic mode (no `--reference-*`): chickpea sim-link Poisson model.\n\
                      Shared topics θ drive ATAC counts via β_atac and RNA counts via\n\
                      W = M·β_ext, where M[G,P] is the sparse peak-gene ground truth.\n\
                      \n\
                      Reference mode (`--reference-rna` and/or `--reference-atac`):\n\
                      per-modality two-stage GLM + NB+copula PIT sampling — same shape as\n\
                      `topic --reference`. The supplied reference's row count overrides\n\
                      `--n-genes` / `--n-peaks`; per-feature dispersion r̂ and a global Σ̂\n\
                      are fitted independently per modality. Cross-modality coupling stays\n\
                      implicit through the shared θ and indicator M."
    )]
    Multiome(MultiomeArgs),
}
