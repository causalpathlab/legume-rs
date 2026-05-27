//! RNA modification + processing simulator (`data-beans-sim faba`).
//!
//! Emits one sparse `.zarr.zip` per RNA track (expression counts, m6A
//! methylation, A-to-I editing, alternative polyadenylation) with rows
//! named in faba's `{gene_key}/{track}/{detail}` convention, plus a
//! full set of ground-truth parquet files. Designed to look like the
//! output of one `faba all` run so downstream consumers
//! (`senna bge --multiome`, `faba mome`, …) can be exercised on
//! simulated data without code changes.
//!
//! Generative model (see `faba/temp.md` for the embedding-side
//! motivation; this simulator does NOT enforce a recovery-friendly
//! geometry — it samples from the underlying biology):
//!
//! ```text
//! Cell state          θ_{k, j}  ~ Dirichlet  (K_topic; shared with
//!                                              writer/editor activity)
//! Structural          s_g       ~ N(0, I_S)   gene substrate score
//! Substrate gate      φ_{g, m}  ~ Bernoulli( σ(s_g · w_m + b_m) )
//! Writer/editor       A_{m, k}  ~ N(0, σ_A²) · Bernoulli(π_A)
//! Gene response       z_{g, k}  ~ N(0, σ_z²) · Bernoulli(π_z)
//! Per-(g,m) base      base_{g,m}~ N(0, σ_b²)
//! mRNA pool           log μ_{g,j} = β_g + Σ_k β_topic_{g,k} θ_{k,j}
//!                                  + δ_{g,B(j)}
//! Modification rate   log r_{g,m,j} = base_{g,m}
//!                                   + φ_{g,m} · Σ_k z_{g,k}·A_{m,k}·θ_{k,j}
//!                                   + γ_m · δ_{g,B(j)}
//! Mixture split       α_{g, m}  ~ Dir(α_mix · 1_{C_m})
//!
//! count    λ_{g,c,j}     = α_{g,count,c} · μ_{g,j}            · depth_count
//! modifier λ_{g,m,c,j}   = α_{g,m,c} · μ_{g,j} · r_{g,m,j}    · depth_m
//!                          (only if φ_{g,m}=1)
//! y                     ~ Poisson(λ)
//! ```
//!
//! Cross-talk is captured by:
//! 1. **Substrate correlation** — m6A and pA share s[0] ("long 3'UTR"),
//!    so their coverage masks are positively correlated; A-to-I rides on
//!    a separate s[2] ("Alu/dsRNA").
//! 2. **Shared writer/editor programs** — many columns of `A` are non-zero
//!    in ≥ 2 modalities, so a single cell-state program (e.g. a
//!    "neuronal activity" topic) can drive m6A, A2I, and pA together.

mod latents;
mod output;
mod sample;

use clap::Args;
use data_beans::sparse_io::SparseIoBackend;
use log::info;
use matrix_util::common_io::mkdir_parent;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};

use crate::core::sample_log_batch_effects;

#[derive(Args, Debug)]
pub struct FabaArgs {
    #[arg(long, short, required = true, help = "Output prefix for all files")]
    pub out: Box<str>,

    #[arg(long, default_value_t = 2000, help = "Number of genes G")]
    pub n_genes: usize,

    #[arg(long, default_value_t = 2000, help = "Number of cells N")]
    pub n_cells: usize,

    #[arg(
        long,
        default_value_t = 8,
        help = "Cell-state topics K (single axis — also drives writer/editor activity, \
                so A_{m,k} couples topic k to modality m's modification machinery)"
    )]
    pub k_topics: usize,

    #[arg(
        long,
        default_value = "1.0,0.30,0.20,0.40",
        help = "Target substrate coverage per modality (count,m6A,A2I,pA)"
    )]
    pub pi_measured: Box<str>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of mixture components per modifier (g, m); count is fixed at 2 \
                (spliced, unspliced)"
    )]
    pub components_per_modifier: usize,

    #[arg(long, default_value_t = 0.5, help = "Dirichlet concentration on α")]
    pub alpha_mix: f32,

    #[arg(long, default_value_t = 1.0, help = "Std-dev of z_{g,k} ~ N(0, σ_z²)")]
    pub sigma_z: f32,

    #[arg(long, default_value_t = 1.0, help = "Std-dev of A_{m,k} ~ N(0, σ_A²)")]
    pub sigma_a: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Std-dev of per-(g,m) base intercept base_{g,m} ~ N(0, σ_b²)"
    )]
    pub sigma_base: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Std-dev of per-gene baseline β_g ~ N(0, σ_β²)"
    )]
    pub sigma_beta: f32,

    #[arg(
        long,
        default_value_t = 0.25,
        help = "Sparsity of z: P(z_{g,k} ≠ 0) = π_z"
    )]
    pub pi_z: f32,

    #[arg(
        long,
        default_value_t = 0.6,
        help = "Sparsity of A: P(A_{m,k} ≠ 0) = π_A"
    )]
    pub pi_a: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Log-normal scale σ_β for the topic dictionary"
    )]
    pub beta_scale: f32,

    #[arg(
        long,
        default_value_t = 5000,
        help = "Target library size for count rows"
    )]
    pub depth_count: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Target per-modality library size for modifier rows"
    )]
    pub depth_modifier: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Topic-PVE for β_topic and θ (reuses core sampling)"
    )]
    pub pve_topic: f32,

    #[arg(long, default_value_t = 1.0, help = "Batch-PVE")]
    pub pve_batch: f32,

    #[arg(long, default_value_t = 1, help = "Number of batches B")]
    pub batches: usize,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Fraction of substrate-positive (g, m) pairs held out (no rows emitted) \
                for imputation evaluation"
    )]
    pub held_out_frac: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix backend"
    )]
    pub backend: SparseIoBackend,

    #[arg(
        long,
        help = "Write plain .zarr directories instead of .zarr.zip archives \
                (zip is the default — pass this to opt out)"
    )]
    pub no_zip: bool,
}

impl FabaArgs {
    /// Resolve the effective zip-output flag: zip is on by default,
    /// `--no-zip` turns it off. This indirection keeps the user-facing
    /// CLI consistent with the project-wide `.zarr.zip` default while
    /// remaining toggleable (a bare `default_value_t = true` bool can't
    /// be flipped from the command line).
    pub fn zip_output(&self) -> bool {
        !self.no_zip
    }
}

/// The set of modalities the v1 simulator emits. Index order is fixed
/// (count=0, m6A=1, A2I=2, pA=3) and used to interpret `--pi-measured`
/// and the rows of A and φ.
pub const MODALITIES: [&str; 4] = ["count", "m6A", "A2I", "pA"];

/// Substrate-feature dimension. Fixed at 3 because the axes are
/// biology-specific (long-3′UTR, DRACH, Alu/dsRNA) and the default
/// substrate weights below would need separate per-modality calibration
/// for any other S.
pub const N_SUBSTRATE_FEATURES: usize = 3;

/// Substrate-weight rows (one per modality, length `S = 3`).
/// `count` is degenerate (no substrate — always measured).
/// `m6A`  loads on long-3'UTR (s[0]) and DRACH (s[1]).
/// `A2I`  loads on Alu/dsRNA (s[2]).
/// `pA`   loads on long-3'UTR (s[0]).
pub fn default_substrate_weights() -> [[f32; N_SUBSTRATE_FEATURES]; MODALITIES.len()] {
    [
        [0.0, 0.0, 0.0], // count (unused — φ_count = 1)
        [0.7, 1.0, 0.0], // m6A
        [0.0, 0.0, 1.2], // A2I
        [1.0, 0.0, 0.0], // pA
    ]
}

/// Human-readable names for the substrate axes (length `N_SUBSTRATE_FEATURES`).
pub const SUBSTRATE_AXIS_NAMES: [&str; N_SUBSTRATE_FEATURES] = ["utr_length", "drach", "alu"];

pub fn run_faba(args: &FabaArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let g = args.n_genes;
    let n = args.n_cells;
    let bb = args.batches.max(1);
    let pi_meas = parse_pi_measured(&args.pi_measured, MODALITIES.len())?;

    info!(
        "faba sim: G={}, N={}, K={}, S={}, B={}",
        g, n, args.k_topics, N_SUBSTRATE_FEATURES, bb
    );
    info!("modalities {:?} target coverage {:?}", MODALITIES, pi_meas);

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.rseed);

    // Batch membership (uniform).
    let runif = Uniform::new(0, bb)?;
    let batch_membership: Vec<usize> = (0..n).map(|_| runif.sample(&mut rng)).collect();

    // Per-(modality, gene) batch effect log δ. One matrix per modality
    // — gives modality-specific batch confounding without forcing
    // shared batch programs (cheap; uses existing helper).
    let ln_delta_per_mod: Vec<nalgebra::DMatrix<f32>> = (0..MODALITIES.len())
        .map(|_| sample_log_batch_effects(g, bb, args.pve_batch, &mut rng))
        .collect();

    // Latents.
    let lats = latents::sample_all(args, &pi_meas, &mut rng)?;

    // Held-out mask (subset of substrate-positive pairs).
    let held_out = latents::sample_held_out(&lats.phi, args.held_out_frac, &mut rng);

    // Precompute the modality-invariant log((β_topic · θ)) — shared
    // across the count and every modifier sampler call instead of
    // recomputed inside each.
    let log_topic = sample::precompute_log_topic(&lats);

    // Per-modality log-rate matrices and triplet sampling.
    let rseed_count = args.rseed.wrapping_add(0x436F_756E); // 'Coun'
    let count_triplets = sample::sample_count_modality(
        &lats,
        &log_topic,
        &ln_delta_per_mod[0],
        &batch_membership,
        args.depth_count,
        rseed_count,
    );

    let mut modifier_triplets: Vec<Vec<(u64, u64, f32)>> = Vec::with_capacity(3);
    let mut modifier_row_keys: Vec<Vec<(usize, usize)>> = Vec::with_capacity(3);
    for m in 1..MODALITIES.len() {
        // Per-modality seed offset uses a 32-bit stride so it stays
        // collision-free even if MODALITIES later grows beyond a
        // handful of tracks.
        let seed = args.rseed.wrapping_add((m as u64) << 32);
        let (trips, row_keys) = sample::sample_modifier_modality(
            m,
            &lats,
            &log_topic,
            &held_out,
            &ln_delta_per_mod[m],
            &batch_membership,
            args.depth_modifier,
            seed,
        );
        info!(
            "modality '{}': {} rows × {} cells → {} non-zero triplets",
            MODALITIES[m],
            row_keys.len(),
            n,
            trips.len()
        );
        modifier_triplets.push(trips);
        modifier_row_keys.push(row_keys);
    }
    info!(
        "modality 'count': {}×{} → {} non-zero triplets",
        2 * g,
        n,
        count_triplets.len()
    );

    // Write outputs.
    output::write_all(
        args,
        &lats,
        &held_out,
        &batch_membership,
        &ln_delta_per_mod,
        &count_triplets,
        &modifier_triplets,
        &modifier_row_keys,
    )?;

    info!("faba simulation done — prefix '{}'", args.out);
    Ok(())
}

fn parse_pi_measured(s: &str, expect: usize) -> anyhow::Result<Vec<f32>> {
    let parts: Vec<f32> = s
        .split(',')
        .map(|t| t.trim().parse::<f32>())
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(
        parts.len() == expect,
        "--pi-measured needs {} values, got {} ({:?})",
        expect,
        parts.len(),
        parts
    );
    for (i, &p) in parts.iter().enumerate() {
        anyhow::ensure!(
            (0.0..=1.0).contains(&p),
            "--pi-measured[{}] ({}) out of [0, 1]",
            i,
            p
        );
    }
    Ok(parts)
}
