mod collapse_cocoa_data;
mod common;
mod input;
mod randomly_partition_data;
mod run_collapse;
mod run_diff;
mod run_sim_collider;
mod run_sim_one_type;
mod run_sim_spatial;
mod run_spatial_diff;
mod spatial_match;
mod stat;

use crate::run_collapse::*;
use crate::run_diff::*;
use crate::run_sim_collider::*;
use crate::run_sim_one_type::*;
use crate::run_sim_spatial::*;
use crate::run_spatial_diff::*;

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
        about = "Per-topic spatial differential expression",
        long_about = "\
Per-topic spatial differential expression with topic-confounder adjustment.

Problem:
  Given pinto's cell-to-topic propensity θ and spatial coordinates, find
  genes whose expression differs across *spatial context* within each
  latent topic. Topic identity is a confounder between location and
  expression — we condition on topic by stratification rather than by
  propensity modeling (which is poorly posed when spatial features are
  high-dimensional and locally smooth).

Algorithm (per topic k):
  1. Stratify cells on θ_{·,k}:
        HIGH  if θ ≥ q_high  (default 0.75)
        LOW   if θ ≤ q_low   (default 0.25)
        DROP  otherwise
  2. Build a spatial kNN graph over all cells (--spatial-knn, optionally
     clipped by --spatial-radius).
  3. For each HIGH cell, collect its spatial-kNN neighbors that are in
     the LOW stratum. Weight each (HIGH, LOW_neighbor) pair by
        w_ij = θ_i · exp(−d_ij) · (1 − θ_j) / Σ_j'
  4. Accumulate sufficient stats into pseudobulks per individual
     (or a single cohort bin if --indv-files is absent):
        y_high[g, i] = Σ_cell∈HIGH θ · y_{g,cell}
        y_low [g, i] = Σ matched-LOW contributions (same gene, weighted)
  5. Optionally row-scale y_high / y_low by NB-Fisher housekeeping
     weights (default ON, --no-adjust-housekeeping disables).
  6. Fit independent Gamma posteriors τ_high, τ_low per (gene × indv)
     and report log τ_high − log τ_low as the contrast.
  7. Hybrid permutation + CLT p-values (--n-permutations N):
       - Shuffle HIGH/LOW labels within each individual, keeping the
         spatial graph and DROP cells fixed.
       - Recompute matching, accumulation, and Gamma fit per permutation.
       - Estimate null mean/sd per gene via Welford; report
         z = (observed − null_mean) / null_sd and
         p = erfc(|z|/√2) (two-sided Gaussian).
       - 100 reps is usually enough for z-based ranking; increase for
         stable tail p-values.

Outputs:
  {out}.spatial_diff.tsv.gz
      columns: gene, topic, contrast_log, null_mean, null_sd, z, pval
  {out}.spatial_diff_indiv.tsv.gz  (only when --indv-files is set)
      columns: gene, topic, individual, log_fold_change
      — per-individual log τ_high − log τ_low, analogous to cocoa
        diff's τ_{d,i} output.

References:
  Park & Kellis (2021) Genome Biol — CoCoA-diff (counterfactual match)
  Hartwig et al. (2023) Eur J Epidemiol — residual collider stratification
"
    )]
    SpatialDiff(SpatialDiffArgs),

    #[command(
        about = "Simulate spatial single-cell data with ground-truth within-topic DE markers",
        long_about = "\
Minimal spatial-DE simulator for validating `cocoa spatial-diff`.

Generative model:
  Cells are placed on an integer grid of size --grid-x × --grid-y and
  split along x into --n-indv contiguous individual blocks. For each
  cell i:
    θ_i  ~ Dirichlet(1, …, 1) with --n-topics components
           (intermixed HIGH/LOW cells so spatial kNN can bridge them)
    y_{g,i} ~ Poisson(λ_{g,i})
    λ_{g,i} = --baseline-rate, or --effect-size if cell i is a marker
              cell for gene g (see below).

Marker injection:
  --n-spatial-markers random (gene, topic k, region R) triples are
  drawn. A cell is a \"marker cell\" for gene g iff
     θ_{i,k} ≥ quantile_{--topic-high-quantile}(θ_{·,k})  AND  (x_i, y_i) ∈ R
  where R is one of {x<median, x≥median, y<median, y≥median}.
  Each gene is used at most once.

Outputs (prefixed by --out):
  {out}.zarr (or .h5)              — sparse G × N counts
  {out}.topic.parquet              — N × K Dirichlet propensities
                                     (directly usable as `spatial-diff
                                     --topic-proportion-files`)
  {out}.coords.tsv.gz              — row-order `x<TAB>y` (no header)
  {out}.indv.tsv.gz                — cell<TAB>individual
  {out}.ground_truth.tsv.gz        — gene<TAB>topic<TAB>region<TAB>effect

Typical workflow:
  cocoa simulate-spatial --out sim/demo
  cocoa spatial-diff sim/demo.zarr \\
        -r sim/demo.topic.parquet \\
        -i sim/demo.indv.tsv.gz \\
        --coords-file sim/demo.coords.tsv.gz \\
        --n-permutations 100 \\
        --out sim/demo_result
  # compare sim/demo.ground_truth.tsv.gz to the top-|z| hits per topic
",
        aliases = ["sim-spatial"]
    )]
    SimulateSpatial(SimSpatialArgs),

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
        Commands::SpatialDiff(args) => {
            run_cocoa_spatial_diff(args.clone())?;
        }
        Commands::SimulateSpatial(args) => {
            run_sim_spatial(args.clone())?;
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
