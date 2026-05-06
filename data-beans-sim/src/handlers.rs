use crate::copula::marginals::{nb_cdf_table, nb_inverse_cdf_from_table, nb_table_cap, phi, NbFit};
use crate::copula::reference::open_reference;
use crate::copula::{fit_global_copula, GlobalCopulaArgs};
use crate::core as simulate;
use crate::multimodal as simulate_multimodal;
use data_beans::hdf5_io::*;
use data_beans::sparse_io::*;
use data_beans::zarr_io::{apply_zip_flag, finalize_zarr_output};

use clap::{Args, ValueEnum};
use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::common_io::*;
use matrix_util::mtx_io;
use matrix_util::traits::*;
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::core::{inject_housekeeping, sample_lognormal_dictionary};

use crate::copula::gaussian::CopulaCovariance;

/// How the batch-program covariance `F_batch` is constructed.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BatchProgram {
    /// `F_batch` is a fresh `(D, batch_rank)` random factor independent of
    /// the gene-gene biology. Batch shifts ride an arbitrary low-dim
    /// subspace — easier to disentangle from biology, useful as a sanity
    /// baseline.
    Random,
    /// `F_batch` reuses the top `batch_rank` columns of the reference's
    /// gene-gene copula factor `Σ̂_gene`. Batch shifts mimic biological
    /// co-expression axes — the worst-case stress test for batch-correction
    /// methods, since batch programs look like cell-state programs.
    Biology,
}

#[derive(Args, Debug)]
pub struct RunSimulateArgs {
    #[arg(
        short,
        long,
        help = "Number of rows, genes, or features (ignored when --reference is set)"
    )]
    pub rows: Option<usize>,

    #[arg(short, long, help = "Number of columns or cells")]
    pub cols: usize,

    /// Real single-cell reference (`.h5`, `.zarr`, `.zarr.zip`). When set,
    /// the GLM pipeline is unchanged, but the final count generation step
    /// swaps `Poisson(λ)` for a copula-coupled NB draw using per-gene
    /// dispersion `r̂_g` and a global Σ̂ fitted from this reference.
    #[arg(long)]
    pub reference: Option<Box<str>>,

    /// HVG count for the gene-gene copula. Genes outside the HVG set are
    /// sampled independently from `NB(λ_{g,j}, r̂_g)`. Used only with `--reference`.
    #[arg(long, default_value_t = 2000)]
    pub n_hvg: usize,

    /// Maximum rank of the low-rank `Σ̂` factor `F = U·diag(σ)/√N`. Effective
    /// rank is `min(rank, n_hvg, n_reference_cells)`. Used only with `--reference`.
    #[arg(long, default_value_t = 100)]
    pub copula_rank: usize,

    /// Per-gene isotropic ridge variance added at sample time on top of `Σ̂`.
    /// Used only with `--reference`.
    #[arg(long, default_value_t = 1e-3)]
    pub regularization: f32,

    /// Lower bound on the NB size parameter `r̂_g`. Tames runaway dispersion
    /// when MoM yields a near-zero `r` for noisy genes. Used only with `--reference`.
    #[arg(long, default_value_t = 1e-2)]
    pub r_floor: f32,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Expected library size E[Σ_g Y(g,j)] per cell (emergent — no per-cell rescaling). \
                Synthetic mode: enters as λ_scale = depth/G. Reference mode: multiplicative \
                scale on the reference's per-gene mean μ̂_g."
    )]
    pub depth: usize,

    #[arg(
        short,
        long,
        default_value_t = 1,
        help = "Number of factors K (cell types / topics / states); width of β and θ"
    )]
    pub factors: usize,

    #[arg(short, long, default_value_t = 1, help = "Number of batches B")]
    pub batches: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Topic-PVE π_topic ∈ [0, 1]. Variance share between topic-specific and \
                topic-invariant components in log β: \
                log β(g,k) = σ_β·[√π_topic · u(g,k) + √(1−π_topic) · v(g)] − σ_β²/2. \
                Also softens θ from one-hot toward uniform: θ(k*,j) = π_topic + (1−π_topic)/K. \
                π_topic = 0 ⇒ no topic structure; π_topic = 1 ⇒ pure topic structure. \
                Independent of pve_batch (both can be 1)."
    )]
    pub pve_topic: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Batch-PVE π_batch ∈ [0, 1]. Variance share between batch-specific and \
                batch-invariant components in log δ: \
                log δ(g,b) = √π_batch · z(g,b) + √(1−π_batch) · w(g). \
                π_batch = 0 ⇒ batches share a single per-gene shift exp(w); \
                π_batch = 1 ⇒ fully batch-specific. Independent of pve_topic."
    )]
    pub pve_batch: f32,

    /// PVE-style magnitude for the per-cell residual log-mean noise term
    /// in stage 1 of the reference-mode two-stage simulator. Setting this
    /// \> 0 adds `√pve_noise · ε_{g,j}` (`ε ~ N(0,1)` iid per gene per cell)
    /// on top of the topic perturbation, before batch is applied. Default
    /// 0 keeps stage 1 fully topic + reference-baseline driven.
    #[arg(long, default_value_t = 0.0)]
    pub pve_noise: f32,

    /// Rank of the batch-program subspace (reference mode). `0` = each
    /// gene's batch shift is iid log-normal (Splatter-style). `2-3` =
    /// genes co-shift along a low-dim subspace ("batch program"). Higher
    /// ranks make batch effects look more like biology.
    #[arg(long, default_value_t = 2)]
    pub batch_rank: usize,

    /// Where the batch-program subspace comes from when `--batch-rank > 0`.
    /// `random` = fresh low-dim random factor (default; arbitrary subspace).
    /// `biology` = top columns of the reference's gene-gene copula
    /// factor (worst case: batch shifts mimic biological axes).
    #[arg(long, value_enum, default_value_t = BatchProgram::Random)]
    pub batch_program: BatchProgram,

    #[arg(short, long, help = "Output file header")]
    pub output: Box<str>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Log-normal scale σ_β. Total log-variance per gene-topic entry is σ_β² \
                (independent of pve_topic); E[β(g,k)] = 1 by centering. Higher = more \
                variable expression across genes and topics."
    )]
    pub beta_scale: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(long, help = "Hierarchical tree depth for binary tree dictionary")]
    pub hierarchical_depth: Option<usize>,

    #[arg(long, default_value_t = 0, help = "Number of housekeeping genes")]
    pub n_housekeeping: usize,

    #[arg(long, default_value_t = 10.0, help = "Housekeeping fold change")]
    pub housekeeping_fold: f32,

    #[arg(long, default_value_t = false, help = "Save output in MTX format")]
    pub save_mtx: bool,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output"
    )]
    pub backend: SparseIoBackend,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,
}

#[derive(Args, Debug)]
pub struct RunSimulateMultimodalArgs {
    #[arg(short, long, help = "Number of features (shared across modalities)")]
    pub rows: usize,

    #[arg(short, long, help = "Number of cells")]
    pub cols: usize,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Expected library size per modality, comma-separated (e.g. 1000,500). \
                One entry per modality; length defines M. Each modality's β columns are \
                softmax-normalized over genes, so depth_m directly sets E[lib(j)|m]."
    )]
    pub depth: Vec<usize>,

    #[arg(
        short,
        long,
        default_value_t = 5,
        help = "Number of topics K (shared across modalities)"
    )]
    pub factors: usize,

    #[arg(short, long, default_value_t = 1, help = "Number of batches B")]
    pub batches: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Scale σ_base of base-dictionary logits W_base ~ N(0, σ_base²) when \
                hierarchical mode is off."
    )]
    pub base_scale: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Scale σ_Δ of non-zero spike entries Δ_m[k,g] ~ N(0, σ_Δ²)"
    )]
    pub delta_scale: f32,

    #[arg(
        long,
        default_value_t = 5,
        help = "Spike-and-slab support: number of non-zero genes per topic in each Δ_m"
    )]
    pub n_delta_features: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Topic-PVE π_topic ∈ [0, 1]. Softens θ from one-hot toward uniform: \
                θ(k*,j) = π_topic + (1−π_topic)/K. Independent of pve_batch."
    )]
    pub pve_topic: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Batch-PVE π_batch ∈ [0, 1]. Variance share in log δ: \
                log δ_m(g,b) = √π_batch · z(g,b) + √(1−π_batch) · w(g). \
                Independent of pve_topic."
    )]
    pub pve_batch: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(
        long,
        default_value_t = false,
        help = "Share batch effects across modalities"
    )]
    pub shared_batch_effects: bool,

    #[arg(long, help = "Hierarchical tree depth for base dictionary")]
    pub hierarchical_depth: Option<usize>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Log-normal scale σ_β for the hierarchical-base dictionary"
    )]
    pub beta_scale: f32,

    #[arg(long, default_value_t = 0, help = "Number of housekeeping genes")]
    pub n_housekeeping: usize,

    #[arg(long, default_value_t = 10.0, help = "Housekeeping fold change")]
    pub housekeeping_fold: f32,

    #[arg(short, long, help = "Output file header")]
    pub output: Box<str>,

    #[arg(long, default_value_t = false, help = "Save output in MTX format")]
    pub save_mtx: bool,

    #[arg(long, value_enum, default_value = "zarr", help = "Backend format")]
    pub backend: SparseIoBackend,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,
}

/// Simulate log-normal factored count data.
///
/// Without `--reference`: pure log-normal · Poisson model
/// `y(g,j) ~ Poisson( (depth/G) · δ(g,B(j)) · Σ_k β(g,k) θ(k,j) )` with
/// log β decomposed into topic-specific + topic-invariant components by
/// `pve_topic`, and log δ decomposed into batch-specific + batch-invariant
/// by `pve_batch`. Library size is emergent.
///
/// With `--reference`: same β / θ / δ structure, but counts are sampled
/// via NB(λ, r̂_g) coupled across HVGs by the reference-fitted gene-gene
/// copula Σ̂. Per-gene baseline becomes log μ̂_g (replacing log(depth/G)).
pub fn run_simulate(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    if cmd_args.reference.is_some() && cmd_args.rows.is_some() {
        anyhow::bail!(
            "`--rows` and `--reference` are mutually exclusive. \
             Under `--reference` the gene count is taken from the reference."
        );
    }
    if cmd_args.reference.is_some() {
        return run_simulate_with_reference(cmd_args);
    }

    let rows = cmd_args
        .rows
        .ok_or_else(|| anyhow::anyhow!("--rows is required when --reference is not set"))?;

    let effective_output = apply_zip_flag(&cmd_args.output, cmd_args.zip);
    let output: Box<str> = strip_backend_suffix(&effective_output).into();

    dirname(&output).as_deref().map(mkdir).transpose()?;

    let backend = cmd_args.backend.clone();
    let (_, backend_file) = resolve_backend_file(&effective_output, Some(backend.clone()))?;

    let mtx_file = output.to_string() + ".mtx.gz";
    let row_file = output.to_string() + ".rows.gz";
    let col_file = output.to_string() + ".cols.gz";

    let dict_file = mtx_file.replace(".mtx.gz", ".dict.parquet");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.parquet");
    let batch_memb_file = mtx_file.replace(".mtx.gz", ".batch.gz");
    let ln_batch_file = mtx_file.replace(".mtx.gz", ".ln_batch.parquet");

    remove_all_files(&vec![
        backend_file.clone(),
        mtx_file.clone().into_boxed_str(),
        dict_file.clone().into_boxed_str(),
        prop_file.clone().into_boxed_str(),
        batch_memb_file.clone().into_boxed_str(),
        ln_batch_file.clone().into_boxed_str(),
    ])
    .expect("failed to clean up existing output files");

    let sim_args = simulate::SimArgs {
        rows,
        cols: cmd_args.cols,
        depth: cmd_args.depth,
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        beta_scale: cmd_args.beta_scale,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        hierarchical_depth: cmd_args.hierarchical_depth,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
    };

    let sim = simulate::generate_factored_poisson_gamma_data(&sim_args)?;
    info!("successfully generated log-normal factored Poisson data");

    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    write_lines(&batch_out, &batch_memb_file)?;
    info!("batch membership: {:?}", &batch_memb_file);

    let mtx_shape = (sim_args.rows, sim_args.cols, sim.triplets.len());

    let rows: Vec<Box<str>> = (0..sim_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    sim.ln_delta_db
        .to_parquet_with_names(&ln_batch_file, (Some(&rows), Some("feature")), None)?;
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;
    sim.beta_dk
        .to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

    if let Some(ref node_probs) = sim.hierarchy_node_probs {
        let hierarchy_file = mtx_file.replace(".mtx.gz", ".hierarchy.parquet");
        node_probs.to_parquet_with_names(&hierarchy_file, (Some(&rows), Some("feature")), None)?;
        info!("wrote hierarchy node probabilities: {:?}", &hierarchy_file);
    }

    info!(
        "wrote parameter files:\n{:?},\n{:?},\n{:?}",
        &ln_batch_file, &dict_file, &prop_file
    );

    if cmd_args.save_mtx {
        let mut triplets = sim.triplets.clone();
        triplets.par_sort_by_key(|&(row, _, _)| row);
        triplets.par_sort_by_key(|&(_, col, _)| col);

        mtx_io::write_mtx_triplets(&triplets, sim_args.rows, sim_args.cols, &mtx_file)?;
        write_lines(&rows, &row_file)?;
        write_lines(&cols, &col_file)?;

        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    info!("registering triplets ...");

    let mut data = create_sparse_from_triplets(
        &sim.triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&backend),
    )?;

    info!("created sparse matrix: {}", backend_file);

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);

    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}

/// Reference-conditioned variant of [`run_simulate`]. Two-stage
/// architecture:
///
/// **Stage 1 (clean cell, topic-only):**
/// `log λ⁰_{g,j} = log μ̂_g + √pve_topic · t_{g,j} + √pve_noise · ε_{g,j}`
/// where `t = log(β·θ)` z-scored across genes per cell (unit log-variance
/// per cell) and `ε ~ N(0, 1)` iid per gene per cell.
///
/// **Stage 2 (batch perturbation, post-hoc):**
/// `log λ_{g,j} = log λ⁰_{g,j} + √pve_batch · δ_{g, b(j)}`
/// where `δ_{:,b} ~ N(0, F_b · F_bᵀ + diag(1 − ‖F_b‖²))` has unit per-gene
/// variance by construction. `F_b`'s rank is `--batch-rank`; its construction
/// is controlled by `--batch-program` (`random` = fresh low-rank factor,
/// `biology` = top columns of the gene-gene copula `Σ̂`).
///
/// Counts are sampled via a unified copula PIT pipeline:
/// `u = Φ(z*)`, `y = F⁻¹_NB(u; λ, r̂_g)`, with `z*` drawn from the gene-gene
/// copula for HVGs and iid `N(0, 1)` for non-HVGs. No per-cell depth
/// renormalization — library size emerges from `μ̂_g · exp(stage-1 + stage-2)`.
fn run_simulate_with_reference(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    let reference_path = cmd_args
        .reference
        .as_ref()
        .expect("run_simulate_with_reference called without --reference");

    let effective_output = apply_zip_flag(&cmd_args.output, cmd_args.zip);
    let output: Box<str> = strip_backend_suffix(&effective_output).into();
    dirname(&output).as_deref().map(mkdir).transpose()?;

    let backend = cmd_args.backend.clone();
    let (_, backend_file) = resolve_backend_file(&effective_output, Some(backend.clone()))?;

    info!("opening reference: {}", reference_path);
    let sc = open_reference(reference_path)?;

    let global_args = GlobalCopulaArgs {
        sc: &sc,
        n_hvg: cmd_args.n_hvg,
        copula_rank: cmd_args.copula_rank,
        regularization: cmd_args.regularization,
        r_floor: cmd_args.r_floor,
    };
    let fit = fit_global_copula(&global_args)?;

    let dd = fit.n_genes;
    let nn = cmd_args.cols;
    let bb = cmd_args.batches.max(1);
    let kk = if let Some(depth) = cmd_args.hierarchical_depth {
        1usize << (depth - 1)
    } else {
        cmd_args.factors.max(1)
    };
    let pve_topic = cmd_args.pve_topic.clamp(0.0, 1.0);
    let pve_batch = cmd_args.pve_batch.clamp(0.0, 1.0);
    let pve_noise = cmd_args.pve_noise.clamp(0.0, 1.0);
    let alpha_topic = pve_topic.sqrt();
    let alpha_batch = pve_batch.sqrt();
    let alpha_noise = pve_noise.sqrt();

    let mut rng = rand::rngs::StdRng::seed_from_u64(cmd_args.rseed);

    let runif = rand_distr::Uniform::new(0, bb).expect("unif [0 .. bb)");
    let batch_membership: Vec<usize> = (0..nn).map(|_| runif.sample(&mut rng)).collect();

    // Stage-1 baseline: log μ̂_g rescaled so the reference's mean library size
    // matches `--depth`. log_mu_hat[g] = log μ̂_g + log(depth / Σ_g μ̂_g).
    let lib_ref: f32 = fit.mu_hat.iter().sum::<f32>().max(1e-30);
    let depth_log_offset = ((cmd_args.depth as f32).max(1.0) / lib_ref).ln();
    let log_mu_hat: DVector<f32> = DVector::from_iterator(
        dd,
        fit.mu_hat
            .iter()
            .map(|&m| m.max(1e-30).ln() + depth_log_offset),
    );
    info!(
        "stage-1 baseline: μ̂ rescaled by depth/lib_ref = {}/{:.0} = {:.4}",
        cmd_args.depth,
        lib_ref,
        (cmd_args.depth as f32) / lib_ref
    );

    // Stage-2 batch covariance. rank=0 → all-isotropic (Splatter-style);
    // rank>0 → low-rank factor + iid residual.
    let batch_cov: CopulaCovariance = if cmd_args.batch_rank == 0 {
        CopulaCovariance::random_low_rank(dd, 0, &mut rng)
    } else {
        match cmd_args.batch_program {
            BatchProgram::Biology => fit.copula.truncate_rank(cmd_args.batch_rank),
            BatchProgram::Random => {
                CopulaCovariance::random_low_rank(dd, cmd_args.batch_rank, &mut rng)
            }
        }
    };
    info!(
        "stage-2 batch program: rank={} ({:?}), {} batches",
        batch_cov.rank(),
        cmd_args.batch_program,
        bb
    );
    // Pre-sample δ_{:, b} per batch with explicit log-space variance
    // decomposition mirroring the synthetic-mode batch effects:
    //   log δ_{g,b} = √π_batch · z_{g,b} + √(1−π_batch) · w_g
    // z_{g,b} from the (gene-gene-correlated) copula, w_g iid N(0,1) shared
    // across batches (the batch-invariant per-gene shift).
    let alpha_invariant_batch = (1.0 - pve_batch).sqrt();
    let normal01 = Normal::new(0.0_f32, 1.0_f32).expect("standard normal");
    let w_invariant: DVector<f32> =
        DVector::from_fn(dd, |_, _| normal01.sample(&mut rng) * alpha_invariant_batch);
    let batch_delta: Vec<DVector<f32>> = (0..bb)
        .map(|_| batch_cov.sample(&mut rng).scale(alpha_batch) + &w_invariant)
        .collect();

    // Topic dictionary β: log-normal with explicit topic-PVE decomposition,
    // matching synthetic mode. Hierarchical mode uses the stick-breaking
    // tree (which encodes topic structure by construction; no extra blend).
    let (mut beta_dk, hierarchy_node_probs) = if let Some(tree_depth) = cmd_args.hierarchical_depth
    {
        let (beta, node_probs) = crate::core::generate_hierarchical_dictionary(
            dd,
            tree_depth,
            cmd_args.beta_scale,
            &mut rng,
        );
        info!(
            "generated hierarchical dictionary: depth={}, K={} leaves",
            tree_depth,
            beta.ncols()
        );
        (beta, Some(node_probs))
    } else {
        (
            sample_lognormal_dictionary(dd, kk, pve_topic, cmd_args.beta_scale, &mut rng),
            None,
        )
    };

    inject_housekeeping(
        &mut beta_dk,
        cmd_args.n_housekeeping,
        cmd_args.housekeeping_fold,
        &mut rng,
    );

    let theta_kn = crate::core::sample_theta_kn(kk, nn, pve_topic, &mut rng)?;

    let triplets: Vec<(u64, u64, f32)> = (0..nn)
        .into_par_iter()
        .progress_count(nn as u64)
        .map(|j| -> Vec<(u64, u64, f32)> {
            let mut local_rng = rand::rngs::StdRng::seed_from_u64(
                cmd_args
                    .rseed
                    .wrapping_add(0x5a5a_5a5a)
                    .wrapping_add(j as u64),
            );
            let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();

            // Stage 1: t = z-scored log(β·θ); ε iid per gene.
            let bt = &beta_dk * theta_kn.column(j);
            let mut t_z: DVector<f32> = bt.map(|x| x.max(1e-30).ln());
            let m_t = t_z.mean();
            let s_t = {
                let mut s2 = 0.0_f32;
                for v in t_z.iter() {
                    let d = *v - m_t;
                    s2 += d * d;
                }
                (s2 / dd as f32).sqrt().max(1e-12)
            };
            for v in t_z.iter_mut() {
                *v = (*v - m_t) / s_t;
            }

            let b = batch_membership[j];
            let mut log_lambda = log_mu_hat.clone();
            for g in 0..dd {
                let topic_term = alpha_topic * t_z[g];
                let noise_term = if alpha_noise > 0.0 {
                    alpha_noise * normal.sample(&mut local_rng)
                } else {
                    0.0
                };
                log_lambda[g] += topic_term + noise_term + batch_delta[b][g];
            }

            // Unified PIT sampling: u = Φ(z*), y = F⁻¹_NB(u; λ, r̂).
            // z* from gene-gene copula for HVGs, iid N(0,1) for non-HVGs.
            // Iterate only active genes (μ̂ ≥ threshold); undetectable genes
            // can't produce a nonzero count even at maximum perturbation.
            let z_hvg = fit.copula.sample(&mut local_rng);
            let mut counts: Vec<(u64, u64, f32)> = Vec::with_capacity(fit.active_genes.len() / 8);
            for &g in &fit.active_genes {
                let mu_g = log_lambda[g].exp();
                if !mu_g.is_finite() || mu_g <= 0.0 {
                    continue;
                }
                let nb = NbFit {
                    mu: mu_g,
                    r: fit.r_hat[g],
                };
                let z_g = match fit.hvg_pos[g] {
                    Some(h) => z_hvg[h as usize],
                    None => normal.sample(&mut local_rng),
                };
                let u = phi(z_g as f64).clamp(1e-7, 1.0 - 1e-7);
                let table = nb_cdf_table(nb, nb_table_cap(nb));
                let x = if table.is_empty() {
                    0
                } else {
                    nb_inverse_cdf_from_table(u, &table)
                };
                if x > 0 {
                    counts.push((g as u64, j as u64, x as f32));
                }
            }
            counts
        })
        .flatten()
        .collect();

    info!(
        "sampled {} cells producing {} nonzero triplets via two-stage NB+copula",
        nn,
        triplets.len()
    );

    // Output: same companion files as before. `.ln_batch.parquet` now stores
    // `α_batch · δ_{g, b}` — the actual log-shift applied per gene per batch.
    let mtx_file = output.to_string() + ".mtx.gz";
    let row_file = output.to_string() + ".rows.gz";
    let col_file = output.to_string() + ".cols.gz";
    let dict_file = format!("{}.dict.parquet", output);
    let prop_file = format!("{}.prop.parquet", output);
    let batch_memb_file = format!("{}.batch.gz", output);
    let ln_batch_file = format!("{}.ln_batch.parquet", output);
    let r_file = format!("{}.r.parquet", output);
    let hvg_file = format!("{}.hvg.gz", output);

    remove_all_files(&vec![
        backend_file.clone(),
        mtx_file.clone().into_boxed_str(),
        dict_file.clone().into_boxed_str(),
        prop_file.clone().into_boxed_str(),
        batch_memb_file.clone().into_boxed_str(),
        ln_batch_file.clone().into_boxed_str(),
        r_file.clone().into_boxed_str(),
        hvg_file.clone().into_boxed_str(),
    ])
    .expect("failed to clean up existing output files");

    let row_names: Vec<Box<str>> = fit.gene_names.clone();
    let col_names: Vec<Box<str>> = (0..nn)
        .map(|j| {
            let argmax_k = theta_kn.column(j).imax();
            format!("synthetic_{}_{}@{}", j, argmax_k, batch_membership[j]).into_boxed_str()
        })
        .collect();

    let batch_lines: Vec<Box<str>> = batch_membership
        .iter()
        .map(|b: &usize| b.to_string().into_boxed_str())
        .collect();
    write_lines(&batch_lines, &batch_memb_file)?;
    info!("batch membership: {}", batch_memb_file);

    // batch_delta is already α_batch-scaled. Stack into D × B for parquet.
    let mut ln_delta_db = DMatrix::<f32>::zeros(dd, bb);
    for (b, col) in batch_delta.iter().enumerate() {
        ln_delta_db.set_column(b, col);
    }
    ln_delta_db.to_parquet_with_names(&ln_batch_file, (Some(&row_names), Some("feature")), None)?;
    theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&col_names), Some("cell")),
        None,
    )?;
    beta_dk.to_parquet_with_names(&dict_file, (Some(&row_names), Some("feature")), None)?;

    if let Some(node_probs) = hierarchy_node_probs.as_ref() {
        let hierarchy_file = format!("{}.hierarchy.parquet", output);
        node_probs.to_parquet_with_names(
            &hierarchy_file,
            (Some(&row_names), Some("feature")),
            None,
        )?;
        info!("hierarchy node probabilities: {}", hierarchy_file);
    }

    // Poisson collapses (`r = ∞`) are encoded as a large finite sentinel
    // so parquet readers don't choke on infinity.
    let mut r_col = DMatrix::<f32>::zeros(dd, 1);
    for g in 0..dd {
        let r = fit.r_hat[g];
        r_col[(g, 0)] = if r.is_finite() { r } else { 1e9 };
    }
    let r_label = ["r_hat".to_string().into_boxed_str()];
    r_col.to_parquet_with_names(&r_file, (Some(&row_names), Some("feature")), Some(&r_label))?;
    info!("per-gene NB dispersion r̂: {}", r_file);

    let hvg_lines: Vec<Box<str>> = fit
        .hvg_indices
        .iter()
        .map(|&g| row_names[g].clone())
        .collect();
    write_lines(&hvg_lines, &hvg_file)?;
    info!("HVGs used by copula: {}", hvg_file);

    let mut triplets = triplets;
    if cmd_args.save_mtx {
        triplets.par_sort_by_key(|&(row, _, _)| row);
        triplets.par_sort_by_key(|&(_, col, _)| col);
        mtx_io::write_mtx_triplets(&triplets, dd, nn, &mtx_file)?;
        write_lines(&row_names, &row_file)?;
        write_lines(&col_names, &col_file)?;
        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    let mtx_shape = (dd, nn, triplets.len());
    let mut data =
        create_sparse_from_triplets(&triplets, mtx_shape, Some(&backend_file), Some(&backend))?;
    data.register_row_names_vec(&row_names);
    data.register_column_names_vec(&col_names);
    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("wrote sparse backend: {}", backend_file);

    info!("done");
    Ok(())
}

/// Run multimodal simulation with shared base + delta dictionaries.
pub fn run_simulate_multimodal(cmd_args: &RunSimulateMultimodalArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    dirname(&output).as_deref().map(mkdir).transpose()?;

    let sim_args = simulate_multimodal::MultimodalSimArgs {
        rows: cmd_args.rows,
        cols: cmd_args.cols,
        depth_per_modality: cmd_args.depth.clone(),
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        base_scale: cmd_args.base_scale,
        delta_scale: cmd_args.delta_scale,
        n_delta_features: cmd_args.n_delta_features,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        shared_batch_effects: cmd_args.shared_batch_effects,
        hierarchical_depth: cmd_args.hierarchical_depth,
        beta_scale: cmd_args.beta_scale,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
    };

    let mm = sim_args.depth_per_modality.len();
    let sim = simulate_multimodal::generate_multimodal_data(&sim_args)?;
    info!("generated multimodal data: {} modalities", mm);

    let rows: Vec<Box<str>> = (0..cmd_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    // Shared outputs
    let prop_file = format!("{}.prop.parquet", output);
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;

    let batch_file = format!("{}.batch.gz", output);
    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();
    write_lines(&batch_out, &batch_file)?;

    // W_base
    let base_file = format!("{}.w_base.parquet", output);
    sim.w_base_kd
        .to_parquet_with_names(&base_file, (None, None), None)?;

    // Per-modality outputs
    let backend = cmd_args.backend.clone();
    for m in 0..mm {
        let suffix = format!(".m{}", m);
        let modality_output = apply_zip_flag(&format!("{}{}", output, suffix), cmd_args.zip);
        let (_, backend_file) = resolve_backend_file(&modality_output, Some(backend.clone()))?;

        let mtx_shape = (cmd_args.rows, cmd_args.cols, sim.triplets[m].len());

        // Dictionary
        let dict_file = format!("{}{}.dict.parquet", output, suffix);
        sim.beta_dk[m].to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

        // Batch effects
        let ln_batch_file = format!("{}{}.ln_batch.parquet", output, suffix);
        sim.ln_delta_db[m].to_parquet_with_names(
            &ln_batch_file,
            (Some(&rows), Some("feature")),
            None,
        )?;

        // Delta (non-reference only)
        if m > 0 {
            let delta_file = format!("{}.w_delta{}.parquet", output, suffix);
            sim.w_delta_kd[m - 1].to_parquet_with_names(&delta_file, (None, None), None)?;

            let mask_file = format!("{}.spike_mask{}.parquet", output, suffix);
            sim.spike_mask_kd[m - 1].to_parquet_with_names(&mask_file, (None, None), None)?;
        }

        // MTX
        if cmd_args.save_mtx {
            let mtx_file = format!("{}{}.mtx.gz", output, suffix);
            let mut triplets = sim.triplets[m].clone();
            triplets.par_sort_by_key(|&(row, col, _)| (col, row));
            mtx_io::write_mtx_triplets(&triplets, cmd_args.rows, cmd_args.cols, &mtx_file)?;
        }

        // Sparse backend
        let mut data = create_sparse_from_triplets(
            &sim.triplets[m],
            mtx_shape,
            Some(&backend_file),
            Some(&backend),
        )?;

        data.register_row_names_vec(&rows);
        data.register_column_names_vec(&cols);

        finalize_zarr_output(&backend_file, &modality_output)?;
        info!("modality {}: {}", m, backend_file);
    }

    info!("done");
    Ok(())
}
