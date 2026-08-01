use clap::{Parser, Subcommand};

use data_beans_sim::deconv::{generate_convoluted_data, SimConvArgs};
use data_beans_sim::faba::{run_faba, FabaArgs};
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
        Commands::Faba(args) => run_faba(args)?,
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Synthetic single-cell-like data generators for the data-beans ecosystem.",
    long_about = "data-beans-sim simulates sparse count matrices. Available structure:\n\
                  factor-model, CNV, multimodal, and topic-conditioned bulk-deconvolution.\n\
                  \n\
                  Outputs use the same zarr/h5 backend formats `data-beans` reads.\n\
                  Simulated and real datasets therefore share one toolchain."
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
                      `Y(g,j) ~ Poisson( (depth/G) · δ(g,B(j)) · Σ_k β(g,k) θ(k,j) )`,\n\
            with an explicit log-space variance decomposition for both β and δ:\n\
                      log β(g,k) = σ_β · [√pve_topic · u_{g,k} + √(1−pve_topic) · v_g] − σ_β²/2\n\
                      log δ(g,b) =        √pve_batch · z_{g,b} + √(1−pve_batch) · w_g\n\
                      with u, v, z, w ~ N(0, 1) iid. `--beta-scale` controls σ_β.\n\
                      `pve_topic` and `pve_batch` are independent variance shares;\n\
                      both can be 1. `depth` is the **expected** library size. It is emergent,\n\
                      with no per-cell rescaling.\n\
                      \n\
                      Reference mode (`--reference <h5/zarr>`): two-stage GLM,\n\
                      NB+copula sampling —\n\
                      stage 1:  log λ⁰ = log μ̂_g + √pve_topic·t + √pve_noise·ε\n\
                      stage 2:  log λ  = log λ⁰ + √pve_batch·z_{g,b} + √(1−pve_batch)·w_g\n\
                      sample :  y ~ NB(λ, r̂_g)  via  u=Φ(z*),  F⁻¹_NB(u; λ, r̂)\n\
                      where t is z-scored log(β·θ) per cell. β is drawn as in synthetic mode,\n\
                      and ε is iid N(0, 1). z_{g,b} comes from the gene-gene copula factor,\n\
                      at rank `--batch-rank`, on axes chosen by `--batch-program`.\n\
                      w_g iid N(0, 1) gives the batch-invariant per-gene shift.\n\
                      \n\
                      `--depth` is reinterpreted as a multiplicative scale.\n\
                      Library size then matches the reference's mean. This follows the scDesign,\n\
                      scDesign2 and scDesign3 lineage.\n\
                      \n\
                      See data-beans-sim/docs/topic.md for the full derivation."
    )]
    Topic(RunSimulateArgs),

    #[command(
        about = "Bulk (convoluted) data matrix from real SC reference (experimental)",
        long_about = "Synthesise bulk pseudo-samples from real single-cell counts.\n\
                      Cells are Dirichlet-mixed under a supplied per-cell topic membership.\n\
                      \n\
                      Each sample is an exact weighted sum of cells with known topic memberships.\n\
                      There is no extra noise model. So ground-truth fractions are recovered,\n\
                      up to whatever the cell-pool sampling implies.\n\
                      \n\
                      See data-beans-sim/docs/bulk.md for the full derivation."
    )]
    Bulk(SimConvArgs),

    #[command(
        about = "Multimodal count data with shared base + delta dictionaries",
        long_about = "Generate M count matrices from shared latent topics θ with modality-specific dictionaries:\n\
                      β_0(:,k) = softmax_g( W_base[k,:]            )    (reference modality)\n\
                      β_m(:,k) = softmax_g( W_base[k,:] + Δ_m[k,:] )    (m = 1..M-1)\n\
                      where W_base ~ N(0, base_scale²).\n\
                      In hierarchical mode the logits are stick-breaking-derived.\n\
                      Δ_m is sparse spike-and-slab.\n\
                      `n_delta_features` genes per topic carry an iid N(0, delta_scale²) perturbation;\n\
                      the rest are zero.\n\
                      \n\
                      Per-modality counts (batch effects independent per modality unless `--shared-batch-effects`):\n\
                      log δ_m(g,b) = √pve_batch · z_{g,b} + √(1−pve_batch) · w_g\n\
                      Y_m(g,j)    ~ Poisson( depth_m · δ_m(g,B(j)) · Σ_k β_m(g,k) θ(k,j) )\n\
                      depth_m is the **expected** library size for modality m. It is emergent,\n\
                      with no per-cell rescaling. Each β_m(:,k) sums to 1 over genes.\n\
                      So the cell-level total of (β·θ) sums to 1 deterministically,\n\
                      and `depth_m` directly sets E[lib(j) | m].\n\
                      \n\
                      See data-beans-sim/docs/multimodal.md for the full derivation."
    )]
    Multimodal(RunSimulateMultimodalArgs),

    #[command(
        about = "Paired ATAC + RNA simulator: cell types switch peaks on/off,\n\
                 genes inherit their enhancers (two-step), with peak-gene ground truth",
        long_about = "Synthetic mode, used when no `--reference-*` is given.\n\
                      It is a two-step generative model. Cis links are cell-type-INVARIANT.\n\
                      Cell-type-specific expression arises because upstream peaks switch on and off per cell type.\n\
                      \n\
                      Step 1 — ATAC from topics. Per cell j, topic mix θ_j (concentration\n\
                      --topic-concentration). Each peak p:\n\
                      \u{20} A_pj = base_p + σ·( √π_topic·T_p + √π_priv·P_p + √π_noise·N_p [+ batch] )\n\
                      \u{20} T = std(log(β_p·θ))  cell-type on/off;  P = peak-PRIVATE fluctuation\n\
                      \u{20} peak budget {topic, private, noise, batch} normalized to 1\n\
                      --invariant-causal-fraction sets a share of causal peaks\n\
                      to be topic-invariant, so purely private.\n\
                      Their links are cleanly recoverable.\n\
                      \n\
                      Step 2 — RNA conditional on enhancers.\n\
                      A linked gene inherits its causal peaks' signal,\n\
                      sig = √π_topic·T + √π_priv·P, via an invariant cis link:\n\
                      \u{20} E_gj = σ·( √pve_cis·std(Σ_{p∈M_g} sig_p) + √(1−pve_cis)·N_g [+ batch] ) The gene has no topic path of its own.\n\
                      Cell-type specificity propagates through its peaks.\n\
                      Unlinked genes are noise. Counts are Poisson(depth·softmax).\n\
                      \n\
                      Identifiability. Only an enhancer's PRIVATE part reaches its gene.\n\
                      Co-active bystanders share only T. So --pve-private is the recoverability dial.\n\
                      --pve-cis sets the gene's cis-dependence strength.\n\
                      Ground truth is M[G,P].\n\
                      \n\
                      Reference mode uses `--reference-rna` or `--reference-atac`.\n\
                      Each modality gets a two-stage GLM, with NB+copula PIT sampling.\n\
                      A normalized {topic, noise, batch} budget weights the log-rate;\n\
                      there is no cis term.\n\
                      Reference row counts override --n-genes and --n-peaks.\n\
                      \n\
                      See data-beans-sim/docs/multiome.md for the full derivation."
    )]
    Multiome(MultiomeArgs),

    #[command(
        about = "RNA modification + processing simulator (counts + m6A + A-to-I + APA)",
        long_about = "Generate sparse per-track count matrices.\n\
                      They are shaped like a `faba all` run.\n\
                      There is one .zarr.zip per RNA track: expression counts, m6A methylation,\n\
                      A-to-I editing, and alternative polyadenylation.\n\
                      Rows are named '{gene}/{track}/{detail}'.\n\
                      A full set of ground-truth parquets ships alongside.\n\
                      \n\
                      Substrate-level coupling is encoded.\n\
                      m6A and pA share a long-3'UTR substrate axis. A-to-I rides on Alu/dsRNA.\n\
                      Writer and editor programs are shared,\n\
                      so one cell-state topic can drive multiple tracks.\n\
                      \n\
                      Generative model summary:\n\
                      \u{20} cell state    θ_{k,j} ~ Dirichlet (shared with writer/editor activity)\n\
                      \u{20} substrate     s_g ~ N(0, I_S);  φ_{g,m} = Bernoulli(σ(s_g·w_m + b_m))\n\
                      \u{20} programs      A_{m,k} ~ N(0,σ_A²)·Bern(π_A);  z_{g,k} ~ N(0,σ_z²)·Bern(π_z)\n\
                      \u{20} mRNA pool     log μ_{g,j} = β_g + log((β_topic·θ)_{g,j}) + δ_{g,B(j)}\n\
                      \u{20} mod. rate     log r_{g,m,j} = base_{g,m} + φ_{g,m}·Σ_k z·A·θ + δ_m\n\
                      \u{20} mixture       α_{g,m} ~ Dir(α_mix·1_{C_m})\n\
                      \u{20} counts:       λ ∝ α · μ                      → Poisson, depth_count\n\
                      \u{20} modifiers:    λ ∝ α · μ · r  (only if φ=1)    → Poisson, depth_modifier\n\
                      \n\
                      See data-beans-sim/docs/faba.md for the full derivation."
    )]
    Faba(FabaArgs),
}
