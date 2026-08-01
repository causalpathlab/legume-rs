mod cnv_call;
mod collapse_cocoa_data;
mod common;
mod input;
mod randomly_partition_data;
mod run_collapse;
mod run_diff;
mod run_sim_collider;
mod run_sim_one_type;
mod stat;

use crate::run_collapse::*;
use crate::run_diff::*;
use crate::run_sim_collider::*;
use crate::run_sim_one_type::*;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "CoCoA — Counterfactual Confounder Adjustment for single-cell DE",
    long_about = "CoCoA implements counterfactual confounder adjustment.\n\
                  It targets single-cell pseudobulk differential expression,\n\
                  after Park & Kellis, 2021.\n\
                  \n\
                  It pairs individual-level matching with topic-aware\n\
                  pseudobulk collapsing.\n\
                  That breaks collider bias.\n\
                  The bias arises when cell type is a downstream effect of\n\
                  both exposure and an unmeasured confounder.\n\
                  \n\
                  Subcommands:\n  \
                  \x20 collapse          — pseudobulk SC counts per (individual, topic)\n  \
                  \x20 diff              — differential expression, confounder-adjusted\n  \
                  \x20 simulate-one      — one cell type, with W → X → Y\n  \
                  \x20 simulate-collider — many cell types, with an X → A ← U collider"
)]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Increase output verbosity")]
    verbose: bool,

    #[arg(
        long = "n-threads",
        visible_aliases = ["threads", "num-threads"],
        global = true,
        value_name = "N",
        help = "Limit the number of CPU threads (rayon global pool)"
    )]
    n_threads: Option<usize>,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Differential expression analysis with pseudobulk",
        long_about = "\
Differential expression on pseudobulk data, confounder-adjusted.\n\
Adjustment is by cross-condition and cross-exposure matching,\n\
after Park & Kellis, 2021.\n\
\n\
By default, topic proportions are residualized.\n\
That removes the exposure-driven shift before analysis.\n\
It breaks collider bias.\n\
The bias arises when cell type A is a common effect of exposure X\n\
and cell-level confounder U, so X → A ← U.\n\
Use --no-residualize-topics to disable it.\n\
\n\
References:\n  \
  Park & Kellis (2021) Genome Biol — CoCoA-diff framework\n  \
  Hartwig et al. (2023) Eur J Epidemiol — residual collider stratification\
"
    )]
    Diff(DiffArgs),

    #[command(
        about = "Collapse single-cell counts into pseudobulk per (individual, topic)",
        alias = "pseudobulk"
    )]
    Collapse(CollapseArgs),

    #[command(
        about = "Simulate single-cell counts under one cell type with a W-confounded exposure",
        long_about = "\
Simulate single-cell counts under a single cell type with a confounded exposure.\n\
\n\
Causal DAG:\n\
\n    \
    W ──→ X ──→ Y\n    \
    └──────────↗\n\
\n  \
  Edges: W→X, W→Y, X→Y (causal, causal genes only)\n\
\n  \
  W is a confounder.\n  \
  It affects both exposure assignment X and gene expression Y.\n  \
  That creates a spurious X–Y association, even for non-causal genes.\n\
\n\
Generative model:\n\
\n  \
  Phase 1 — individual level (i = 1..N):\n    \
    W_i  ~ N(0, I)                        [--n-covariates dims]\n    \
    X_i  ~ Cat(softmax(√pve_wx · W·α + √(1-pve_wx) · ε))\n                                           \
                                           [--n-exposure categories]\n\
\n  \
  Phase 2 — individual-level gene expression (gene g = 1..G):\n    \
    log μ_ig = √pve_xy · β_g·X̃_i         (causal, causal genes only)\n             \
             + √pve_wy · W_i·γ_g          (confounding)\n             \
             + √pve_res · ε_g             (noise)\n\
\n  \
  Phase 3 — cell-level counts (cell j in individual i):\n    \
    Y_ijg ~ Poisson(ρ_j · exp(log μ_ig))\n    \
    ρ_j   ~ Gamma(shape, rate)            [--gamma-hyperparam]\n\
\n\
PVE parameters control signal strength at each DAG edge.\
"
    )]
    SimulateOne(SimOneTypeArgs),

    #[command(
        about = "Simulate single-cell counts with collider bias (X → A ← U) across cell types",
        long_about = "\
Simulate single-cell counts under a collider bias model with multiple cell types.\n\
\n\
Causal DAG:\n\
\n    \
    ┌──→ A ←───┐\n    \
    │          │\n    \
    X ──→ Y ←─ U\n    \
    ↑     ↑\n    \
    └─ V ─┘\n\
\n  \
  Edges: V→X, V→Y, X→Y (causal), X→A, U→A, U→Y\n\
\n  \
  A is a collider: X → A ← U.\n  \
  Conditioning on cell type A opens the path X → A ← U → Y.\n  \
  That induces a non-causal X–Y association, through U.\n\
\n\
Generative model:\n\
\n  \
  Phase 1 — individual level (i = 1..N):\n    \
    V_i  ~ N(0, I)                        [--n-covariates dims]\n    \
    X_i  ~ Cat(softmax(√pve_vx · V·α + √(1-pve_vx) · ε))\n                                           \
                                           [--n-exposure categories]\n\
\n  \
  Phase 2 — cell level (j = 1..n_j for individual i):\n    \
    U_j  ~ N(0, I)                        [--n-cell-covariates dims]\n    \
    A_ij ~ Cat(softmax(√pve_ua · U·δ + √pve_xa · X·η + √pve_res · ε))\n           \
           where A is the collider variable [--n-cell-types categories]\n\
\n  \
  Phase 3 — gene expression (gene g = 1..G):\n    \
    log μ_ijg = √pve_xy · β_g·X̃_i        (causal, causal genes only)\n              \
              + √pve_vy · V_i·γ_g         (individual confounding)\n              \
              + √pve_uy · U_j·ξ_g         (cell confounding)\n              \
              + √pve_res · ε_g            (noise)\n    \
    Y_ijg ~ Poisson(ρ_j · exp(log μ))\n    \
    ρ_j   ~ Gamma(shape, rate)            [--gamma-hyperparam]\n\
\n\
PVE parameters control signal strength at each DAG edge.\n\
\n\
References:\n  \
  Cole et al. (2010) Int J Epidemiol — collider bias via DAGs\n  \
  Davey Smith & Munafò (2019) Int J Epidemiol — selection bias severity\n  \
  Akimova et al. (2021) Sci Rep — continuous conditioning on a collider\n  \
  Hartwig et al. (2023) Eur J Epidemiol — residual collider stratification\
"
    )]
    SimulateCollider(SimColliderArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let default_filter = if cli.verbose {
        matrix_util::common_io::VERBOSE_LOG_FILTER
    } else {
        matrix_util::common_io::QUIET_LOG_FILTER
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
        .init();

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
        Commands::Diff(args) => {
            run_cocoa_diff(args.clone())?;
        }
        Commands::Collapse(args) => {
            run_collapse(args.clone())?;
        }
        Commands::SimulateOne(args) => {
            run_sim_one_type_data(args.clone())?;
        }
        Commands::SimulateCollider(args) => {
            run_sim_collider_data(args.clone())?;
        }
    }

    Ok(())
}
