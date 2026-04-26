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
    about = "CoCoA (Counterfactual Confounder Adjustment)",
    long_about = "Routines in CoCoA will be useful "
)]
struct Cli {
    #[arg(short = 'v', long, global = true, help = "Increase output verbosity.")]
    verbose: bool,

    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Differential expression analysis with pseudobulk",
        long_about = "\
Differential expression analysis on pseudobulk data
while adjusting confounding effects by cross-condition
or cross-exposure/treatment matching (Park & Kellis, 2021).

By default, topic proportions are residualized to remove the
exposure-driven shift before analysis. This breaks collider bias
when cell type A is a common effect of exposure X and cell-level
confounder U (X -> A <- U). Use --no-residualize-topics to disable.

References:
  Park & Kellis (2021) Genome Biol — CoCoA-diff framework
  Hartwig et al. (2023) Eur J Epidemiol — residual collider stratification
"
    )]
    Diff(DiffArgs),

    #[command(
        about = "Collapse single-cell counts into pseudobulk per individual and cell type",
        alias = "pseudobulk"
    )]
    Collapse(CollapseArgs),

    #[command(
        about = "Simulate single-cell data with confounded exposure (one cell type)",
        long_about = "\
Simulate single-cell count data with confounded exposure assignment (one cell type).

Causal DAG:

    W ──→ X ──→ Y
    └──────────↗

  Edges: W->X, W->Y, X->Y (causal, causal genes only)

  W is a confounder: it affects both exposure assignment X and
  gene expression Y, creating a spurious X-Y association even
  for non-causal genes.

Generative model:

  Phase 1 — individual level (i = 1..N):
    W_i  ~ N(0, I)                        [--n-covariates dims]
    X_i  ~ Cat(softmax(√pve_wx · W·α + √(1-pve_wx) · ε))
                                           [--n-exposure categories]

  Phase 2 — individual-level gene expression (gene g = 1..G):
    log μ_ig = √pve_xy · β_g·X̃_i         (causal, causal genes only)
             + √pve_wy · W_i·γ_g          (confounding)
             + √pve_res · ε_g             (noise)

  Phase 3 — cell-level counts (cell j in individual i):
    Y_ijg ~ Poisson(ρ_j · exp(log μ_ig))
    ρ_j   ~ Gamma(shape, rate)            [--gamma-hyperparam]

PVE parameters control signal strength at each DAG edge.
",
        alias = "sim-v1"
    )]
    SimulateOne(SimOneTypeArgs),

    #[command(
        about = "Simulate single-cell data with collider bias (X -> A <- U)",
        long_about = "\
Simulate single-cell count data under a collider bias model.

Causal DAG:

    ┌──→ A ←───┐
    │          │
    X ──→ Y ←─ U
    ↑     ↑
    └─ V ─┘

  Edges: V->X, V->Y, X->Y (causal), X->A, U->A, U->Y

  A is a collider (X -> A <- U). Conditioning on cell type A
  opens the spurious path X -> A <- U -> Y, inducing a non-causal
  association between exposure X and expression Y through U.

Generative model:

  Phase 1 — individual level (i = 1..N):
    V_i  ~ N(0, I)                        [--n-covariates dims]
    X_i  ~ Cat(softmax(√pve_vx · V·α + √(1-pve_vx) · ε))
                                           [--n-exposure categories]

  Phase 2 — cell level (j = 1..n_j for individual i):
    U_j  ~ N(0, I)                        [--n-cell-covariates dims]
    A_ij ~ Cat(softmax(√pve_ua · U·δ + √pve_xa · X·η + √pve_res · ε))
           where A is the collider variable [--n-cell-types categories]

  Phase 3 — gene expression (gene g = 1..G):
    log μ_ijg = √pve_xy · β_g·X̃_i        (causal, causal genes only)
              + √pve_vy · V_i·γ_g         (individual confounding)
              + √pve_uy · U_j·ξ_g         (cell confounding)
              + √pve_res · ε_g            (noise)
    Y_ijg ~ Poisson(ρ_j · exp(log μ))
    ρ_j   ~ Gamma(shape, rate)            [--gamma-hyperparam]

PVE parameters control signal strength at each DAG edge.

References:
  Cole et al. (2010) Int J Epidemiol — collider bias via DAGs
  Davey Smith & Munafò (2019) Int J Epidemiol — selection bias severity
  Akimova et al. (2021) Sci Rep — continuous conditioning on a collider
  Hartwig et al. (2023) Eur J Epidemiol — residual collider stratification
",
        aliases = ["simulate", "sim-v2"]
    )]
    SimulateCollider(SimColliderArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

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
