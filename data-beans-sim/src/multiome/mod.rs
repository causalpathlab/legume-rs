//! Paired ATAC + RNA simulator (`data-beans-sim multiome`).
//!
//! Without any `--reference-*`: pure chickpea-style Poisson simulator —
//! shared latent topics drive ATAC counts via β_atac and RNA counts via the
//! derived dictionary `W = M · β_ext`. Peak-gene ground truth `M` is sparse
//! and laid out so cis-window recovery is well-defined.
//!
//! With `--reference-rna` and/or `--reference-atac`: per-modality two-stage
//! GLM with NB+copula sampling, mirroring `handlers::run_simulate_with_reference`.
//! Each reference is fitted independently (`fit_global_copula`) — there is
//! no cross-modality copula. Cross-modality coupling stays implicit through
//! the shared θ and the indicator M.

mod sample;

use crate::copula::gaussian::CopulaCovariance;
use crate::copula::marginals::{nb_cdf_table, nb_inverse_cdf_from_table, nb_table_cap, phi, NbFit};
use crate::copula::reference::{open_reference, SparseRef};
use crate::copula::{fit_global_copula, GlobalCopulaArgs, GlobalCopulaFit};
use crate::handlers::BatchProgram;

use clap::Args;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::zarr_io::{apply_zip_flag, finalize_zarr_output};
use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::common_io::{mkdir_parent, open_buf_writer, write_lines};
use matrix_util::traits::*;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::io::Write;

use sample::Mat;

type Triplets = Vec<(u64, u64, f32)>;

const N_CHROMOSOMES: usize = 22;
const PEAK_BIN_WIDTH: usize = 500;
const PEAK_GAP: usize = 500;

/// Simulated gene TSS (chr + position). Local mirror of
/// `genomic_data::coordinates::GeneTss` to avoid pulling in that crate.
struct GeneTss {
    chr: Box<str>,
    tss: i64,
}

#[derive(Args, Debug)]
pub struct MultiomeArgs {
    #[arg(long, short, required = true, help = "Output prefix for all files")]
    pub out: Box<str>,

    #[arg(
        long,
        default_value_t = 2000,
        help = "Number of genes (G); overridden by --reference-rna"
    )]
    pub n_genes: usize,

    #[arg(
        long,
        default_value_t = 10000,
        help = "Number of ATAC peaks (P); overridden by --reference-atac"
    )]
    pub n_peaks: usize,

    #[arg(long, default_value_t = 5000, help = "Number of cells (N)")]
    pub n_cells: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Coarse topics (K), shared by ATAC and RNA"
    )]
    pub n_topics: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "RNA subtypes per coarse topic; K_total = K × K_sub"
    )]
    pub n_sub_topics: usize,

    #[arg(long, default_value_t = 3, help = "Causal peaks per linked gene")]
    pub n_causal_per_gene: usize,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "Fraction of genes with causal peak links"
    )]
    pub linked_gene_fraction: f32,

    #[arg(
        long,
        default_value_t = 5000,
        help = "Expected RNA library size per cell. Synthetic mode: enters as the per-cell \
                depth multiplier ρ_j (with `--cell-sd-log-depth-rna` log-normal noise). \
                Reference mode: rescales the reference's per-gene mean μ̂_g so the simulated \
                cells' average RNA library size matches this value."
    )]
    pub depth_rna: usize,

    #[arg(
        long,
        default_value_t = 2000,
        help = "Expected ATAC library size per cell. Symmetric to --depth-rna."
    )]
    pub depth_atac: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "SD of log-normal per-cell depth noise (ATAC, no-ref mode)"
    )]
    pub cell_sd_log_depth_atac: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "SD of log-normal per-cell depth noise (RNA, no-ref mode)"
    )]
    pub cell_sd_log_depth_rna: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Topic-PVE π_topic ∈ [0, 1]. Softens θ from one-hot toward uniform: \
                θ_coarse(k*,j) = π_topic + (1−π_topic)/K. Multiome β is softmax-normalized \
                over features (sums to 1 per topic), so π_topic acts on θ only — there is \
                no separate β-PVE knob (like multimodal). Independent of pve_batch."
    )]
    pub pve_topic: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Subtype-PVE π_sub ∈ [0, 1]. Same θ-softening but at the subtype level \
                within the dominant coarse topic. Only used when K_sub > 1."
    )]
    pub pve_sub_topic: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Gene-topic effect SD; LogNormal modulation. 0 = disabled"
    )]
    pub gene_topic_sd: f32,

    #[arg(long, default_value_t = 42, help = "Random seed for reproducibility")]
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
        default_value_t = false,
        help = "Write `.zarr.zip` archive instead of `.zarr` directory"
    )]
    pub zip: bool,

    // ---- Reference / copula flags (per modality) ----------------------------------------
    /// Real single-cell ATAC reference (`.h5`, `.zarr`, `.zarr.zip`). When set,
    /// the ATAC sampler switches to two-stage GLM + NB+copula PIT (per-peak
    /// `r̂` + global Σ̂ from the reference). The reference's row count
    /// overrides `--n-peaks`.
    #[arg(long)]
    pub reference_atac: Option<Box<str>>,

    /// Real single-cell RNA reference. Symmetric to `--reference-atac`. The
    /// reference's row count overrides `--n-genes`.
    #[arg(long)]
    pub reference_rna: Option<Box<str>>,

    /// HVG / HVP count for each modality's gene-gene (peak-peak) copula.
    /// Features outside the HV set are sampled independently from `NB(λ, r̂)`.
    #[arg(long, default_value_t = 2000)]
    pub n_hvg: usize,

    /// Maximum rank of the per-modality low-rank `Σ̂` factor.
    #[arg(long, default_value_t = 100)]
    pub copula_rank: usize,

    /// Per-feature isotropic ridge variance added at sample time on top of `Σ̂`.
    #[arg(long, default_value_t = 1e-3)]
    pub regularization: f32,

    /// Lower bound on the NB size parameter `r̂`. Tames runaway dispersion when
    /// MoM yields a near-zero `r` for noisy features.
    #[arg(long, default_value_t = 1e-2)]
    pub r_floor: f32,

    /// Number of batches (per-cell membership is uniform). Stage-2 batch
    /// perturbation is fitted per modality; same membership across modalities.
    #[arg(long, default_value_t = 1)]
    pub batches: usize,

    /// Rank of the batch-program subspace in reference mode. `0` = iid
    /// (Splatter-style); `2-3` = co-shifted batch program.
    #[arg(long, default_value_t = 2)]
    pub batch_rank: usize,

    /// Where the batch-program subspace comes from when `--batch-rank > 0`.
    #[arg(long, value_enum, default_value_t = BatchProgram::Random)]
    pub batch_program: BatchProgram,

    /// PVE-style magnitude for the per-cell residual log-mean noise term in
    /// stage 1 of the reference-mode sampler.
    #[arg(long, default_value_t = 0.0)]
    pub pve_noise: f32,

    /// Proportion of variance explained by batch effects (reference mode).
    #[arg(long, default_value_t = 1.0)]
    pub pve_batch: f32,
}

pub fn run_multiome(args: &MultiomeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let mut rng = StdRng::seed_from_u64(args.rseed);
    let nn = args.n_cells;
    let kk = args.n_topics;
    let k_sub = args.n_sub_topics.max(1);
    let k_total = kk * k_sub;

    // ---- Open references (if any) and resolve modality dims --------------------------------
    let atac_fit: Option<GlobalCopulaFit> = if let Some(path) = args.reference_atac.as_ref() {
        info!("opening ATAC reference: {}", path);
        let sc = open_reference(path)?;
        Some(fit_modality_copula(&sc, args)?)
    } else {
        None
    };
    let rna_fit: Option<GlobalCopulaFit> = if let Some(path) = args.reference_rna.as_ref() {
        info!("opening RNA reference: {}", path);
        let sc = open_reference(path)?;
        Some(fit_modality_copula(&sc, args)?)
    } else {
        None
    };

    let p = atac_fit.as_ref().map(|f| f.n_genes).unwrap_or(args.n_peaks);
    let g = rna_fit.as_ref().map(|f| f.n_genes).unwrap_or(args.n_genes);

    info!(
        "simulating: {} genes, {} peaks, {} cells, {} topics × {} subtypes = {} total \
         (pve_coarse={}, pve_sub={}, gene_topic_sd={}, atac_ref={}, rna_ref={})",
        g,
        p,
        nn,
        kk,
        k_sub,
        k_total,
        args.pve_topic,
        args.pve_sub_topic,
        args.gene_topic_sd,
        atac_fit.is_some(),
        rna_fit.is_some(),
    );

    // ---- Topic proportions (nested) -----------------------------------------------------
    let theta_seed = rng.next_u64();
    let (theta_full, theta_coarse) = sample::sample_nested_topic_proportions(
        kk,
        k_sub,
        nn,
        args.pve_topic,
        args.pve_sub_topic,
        theta_seed,
    );

    // ---- Dictionaries -------------------------------------------------------------------
    let beta_ext = sample::sample_dictionary(p, k_total, &mut rng);
    let beta_atac = sample::marginalize_dictionary(&beta_ext, kk, k_sub);

    // ---- Names --------------------------------------------------------------------------
    let peak_names: Vec<Box<str>> = atac_fit
        .as_ref()
        .map(|f| f.gene_names.clone())
        .unwrap_or_else(|| generate_peak_names(p));
    let gene_names: Vec<Box<str>> = rna_fit
        .as_ref()
        .map(|f| f.gene_names.clone())
        .unwrap_or_else(|| generate_indexed_names(g, "gene"));
    let cell_names = generate_indexed_names(nn, "cell");
    let gene_coords = generate_gene_coords(g);

    // ---- Indicator matrix M[G × P] -------------------------------------------------------
    let n_linked = (g as f32 * args.linked_gene_fraction) as usize;
    let (indicator_genes, indicator_peaks) = sample::sample_indicator_matrix(
        g,
        p,
        n_linked,
        args.n_causal_per_gene,
        N_CHROMOSOMES,
        &mut rng,
    );
    info!(
        "{} linked genes, {} total entries in M",
        n_linked,
        indicator_genes.len()
    );

    // ---- Derived RNA dictionary W[G × K_total] = M · β_ext ----------------------------
    let w_gk = sample::build_derived_dictionary(&indicator_genes, &indicator_peaks, &beta_ext, g);

    // ---- Optional gene-topic effect γ[G × K_total] ------------------------------------
    let gamma_gk = if args.gene_topic_sd > 0.0 {
        Some(sample::sample_gene_topic_effects(
            g,
            k_total,
            args.gene_topic_sd,
            &mut rng,
        ))
    } else {
        None
    };

    // ---- Batch membership (shared across modalities) ----------------------------------
    let bb = args.batches.max(1);
    let runif = rand_distr::Uniform::new(0, bb).expect("unif [0 .. bb)");
    let batch_membership: Vec<usize> = (0..nn).map(|_| runif.sample(&mut rng)).collect();

    // ---- Synthetic-mode batch effects (per modality, log-space PVE decomposition) -------
    // Mirrors the topic / multimodal subcommands:
    //   log δ_m(g, b) = √π_batch · z_{g,b} + √(1−π_batch) · w_g
    // Built only when `bb > 1` AND the corresponding modality is in synthetic mode.
    // Reference-mode modalities draw their own batch perturbations inside
    // `sample_with_reference`, where δ rides the gene-gene copula structure.
    let synth_atac_delta: Option<DMatrix<f32>> = if atac_fit.is_none() && bb > 1 {
        Some(sample_synth_batch_delta(p, bb, args.pve_batch, &mut rng))
    } else {
        None
    };
    let synth_rna_delta: Option<DMatrix<f32>> = if rna_fit.is_none() && bb > 1 {
        Some(sample_synth_batch_delta(g, bb, args.pve_batch, &mut rng))
    } else {
        None
    };

    // =====================================================================================
    //   ATAC counts
    // =====================================================================================
    let (atac_triplets, atac_batch_delta) = if let Some(fit) = atac_fit.as_ref() {
        let (trips, batch_delta) = sample_with_reference(
            &beta_atac,
            &theta_coarse,
            None,
            fit,
            &batch_membership,
            bb,
            args.pve_topic,
            args.pve_noise,
            args.pve_batch,
            args.batch_rank,
            args.batch_program,
            Some(args.depth_atac),
            args.rseed.wrapping_add(0x4154_4143), // "ATAC"
            "ATAC",
        )?;
        (trips, Some(batch_delta))
    } else {
        let rho =
            sample::sample_cell_depths(nn, args.depth_atac, args.cell_sd_log_depth_atac, &mut rng);
        info!(
            "sampling ATAC counts: {} peaks × {} cells (Poisson{})",
            p,
            nn,
            if synth_atac_delta.is_some() {
                ", with batch δ"
            } else {
                ""
            }
        );
        let atac_seed = rng.next_u64();
        let memb_ref = synth_atac_delta
            .as_ref()
            .map(|_| batch_membership.as_slice());
        let trips = sample::sample_poisson_counts(
            &beta_atac,
            &theta_coarse,
            &rho,
            None,
            synth_atac_delta.as_ref(),
            memb_ref,
            atac_seed,
        );
        (trips, None)
    };
    info!("ATAC: {} non-zeros", atac_triplets.len());

    // =====================================================================================
    //   RNA counts
    // =====================================================================================
    let (rna_triplets, rna_batch_delta) = if let Some(fit) = rna_fit.as_ref() {
        let (trips, batch_delta) = sample_with_reference(
            &w_gk,
            &theta_full,
            gamma_gk.as_ref(),
            fit,
            &batch_membership,
            bb,
            args.pve_topic,
            args.pve_noise,
            args.pve_batch,
            args.batch_rank,
            args.batch_program,
            Some(args.depth_rna),
            args.rseed.wrapping_add(0x524e_4100), // "RNA\0"
            "RNA",
        )?;
        (trips, Some(batch_delta))
    } else {
        let tau =
            sample::sample_cell_depths(nn, args.depth_rna, args.cell_sd_log_depth_rna, &mut rng);
        info!(
            "sampling RNA counts: {} genes × {} cells (Poisson{})",
            g,
            nn,
            if synth_rna_delta.is_some() {
                ", with batch δ"
            } else {
                ""
            }
        );
        let rna_seed = rng.next_u64();
        let memb_ref = synth_rna_delta
            .as_ref()
            .map(|_| batch_membership.as_slice());
        let trips = sample::sample_poisson_counts(
            &w_gk,
            &theta_full,
            &tau,
            gamma_gk.as_ref(),
            synth_rna_delta.as_ref(),
            memb_ref,
            rna_seed,
        );
        (trips, None)
    };
    info!("RNA: {} non-zeros", rna_triplets.len());

    // ---- Persist sparse outputs --------------------------------------------------------
    let backend = args.backend.clone();
    let backend_suffix = match backend {
        SparseIoBackend::Zarr => "zarr",
        SparseIoBackend::HDF5 => "h5",
    };

    let atac_dir = format!("{}.atac.{}", args.out, backend_suffix);
    let atac_final = apply_zip_flag(&atac_dir, args.zip);
    let mut atac_data = create_sparse_from_triplets(
        &atac_triplets,
        (p, nn, atac_triplets.len()),
        Some(&atac_dir),
        Some(&backend),
    )?;
    atac_data.register_row_names_vec(&peak_names);
    atac_data.register_column_names_vec(&cell_names);
    finalize_zarr_output(&atac_dir, &atac_final)?;
    info!("wrote ATAC sparse backend: {}", atac_final);

    let rna_dir = format!("{}.rna.{}", args.out, backend_suffix);
    let rna_final = apply_zip_flag(&rna_dir, args.zip);
    let mut rna_data = create_sparse_from_triplets(
        &rna_triplets,
        (g, nn, rna_triplets.len()),
        Some(&rna_dir),
        Some(&backend),
    )?;
    rna_data.register_row_names_vec(&gene_names);
    rna_data.register_column_names_vec(&cell_names);
    finalize_zarr_output(&rna_dir, &rna_final)?;
    info!("wrote RNA sparse backend: {}", rna_final);

    // ---- Companion parquet / TSV files -------------------------------------------------
    let dict_file = format!("{}.dict.parquet", args.out);
    beta_atac.to_parquet_with_names(&dict_file, (Some(&peak_names), Some("peak")), None)?;
    info!("wrote ATAC dictionary (marginalized) to {}", dict_file);

    let prop_file = format!("{}.prop.parquet", args.out);
    theta_coarse.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cell_names), Some("cell")),
        None,
    )?;
    info!("wrote coarse proportions to {}", prop_file);

    let derived_file = format!("{}.derived_dict.parquet", args.out);
    w_gk.to_parquet_with_names(&derived_file, (Some(&gene_names), Some("gene")), None)?;
    info!("wrote derived RNA dictionary to {}", derived_file);

    if k_sub > 1 {
        let ext_file = format!("{}.beta_ext.parquet", args.out);
        beta_ext.to_parquet_with_names(&ext_file, (Some(&peak_names), Some("peak")), None)?;
        info!("wrote extended dictionary to {}", ext_file);

        let full_prop_file = format!("{}.theta_full.parquet", args.out);
        theta_full.transpose().to_parquet_with_names(
            &full_prop_file,
            (Some(&cell_names), Some("cell")),
            None,
        )?;
        info!("wrote full (nested) proportions to {}", full_prop_file);
    }

    if let Some(ref gamma) = gamma_gk {
        let gamma_file = format!("{}.gamma.parquet", args.out);
        gamma.to_parquet_with_names(&gamma_file, (Some(&gene_names), Some("gene")), None)?;
        info!("wrote gene-topic effects to {}", gamma_file);
    }

    write_ground_truth(
        &indicator_genes,
        &indicator_peaks,
        &peak_names,
        &gene_names,
        &args.out,
    )?;
    write_names(&args.out, &peak_names, &gene_names, &cell_names)?;
    write_gene_coords(&gene_names, &gene_coords, &args.out)?;

    // Batch membership (whenever bb > 1, synthetic OR reference mode).
    if bb > 1 {
        let batch_lines: Vec<Box<str>> = batch_membership
            .iter()
            .map(|b| b.to_string().into_boxed_str())
            .collect();
        let batch_file = format!("{}.batch.gz", args.out);
        write_lines(&batch_lines, &batch_file)?;
        info!("batch membership: {}", batch_file);
    }

    // Synthetic-mode log-δ matrices (one per modality where bb > 1 and no ref).
    if let Some(ref delta) = synth_atac_delta {
        let f = format!("{}.atac.ln_batch.parquet", args.out);
        delta
            .map(|x| x.ln())
            .to_parquet_with_names(&f, (Some(&peak_names), Some("peak")), None)?;
        info!("wrote ATAC synthetic-mode log-δ: {}", f);
    }
    if let Some(ref delta) = synth_rna_delta {
        let f = format!("{}.rna.ln_batch.parquet", args.out);
        delta
            .map(|x| x.ln())
            .to_parquet_with_names(&f, (Some(&gene_names), Some("gene")), None)?;
        info!("wrote RNA synthetic-mode log-δ: {}", f);
    }

    if let (Some(fit), Some(delta)) = (atac_fit.as_ref(), atac_batch_delta.as_ref()) {
        write_reference_extras(&args.out, "atac", fit, delta, &peak_names)?;
    }
    if let (Some(fit), Some(delta)) = (rna_fit.as_ref(), rna_batch_delta.as_ref()) {
        write_reference_extras(&args.out, "rna", fit, delta, &gene_names)?;
    }

    info!(
        "done. outputs at {}.{{rna,atac}}.{}",
        args.out, backend_suffix
    );

    Ok(())
}

fn fit_modality_copula(sc: &SparseRef, args: &MultiomeArgs) -> anyhow::Result<GlobalCopulaFit> {
    let global_args = GlobalCopulaArgs {
        sc,
        n_hvg: args.n_hvg,
        copula_rank: args.copula_rank,
        regularization: args.regularization,
        r_floor: args.r_floor,
    };
    fit_global_copula(&global_args)
}

/// Two-stage GLM + NB+copula PIT sampler for one modality.
///
/// `dict_dk` is the modality's dictionary (β_atac for ATAC, W for RNA), and
/// `theta_kn` is the topic proportions matrix paired with it (θ_coarse for
/// ATAC since its dictionary marginalizes subtypes; θ_full for RNA since W
/// is K_total-wide). `gamma_dk` is an optional element-wise modulation
/// applied to `dict_dk` before computing the stage-1 baseline.
///
/// Returns triplets and the per-batch δ (for parquet export).
#[allow(clippy::too_many_arguments)]
fn sample_with_reference(
    dict_dk: &Mat,
    theta_kn: &Mat,
    gamma_dk: Option<&Mat>,
    fit: &GlobalCopulaFit,
    batch_membership: &[usize],
    bb: usize,
    pve_topic: f32,
    pve_noise: f32,
    pve_batch: f32,
    batch_rank: usize,
    batch_program: BatchProgram,
    depth_target: Option<usize>,
    rseed: u64,
    label: &str,
) -> anyhow::Result<(Triplets, Vec<DVector<f32>>)> {
    let dd = fit.n_genes;
    let nn = batch_membership.len();
    if dict_dk.nrows() != dd {
        anyhow::bail!(
            "{}: dictionary rows ({}) != reference features ({})",
            label,
            dict_dk.nrows(),
            dd
        );
    }

    let alpha_topic = pve_topic.clamp(0.0, 1.0).sqrt();
    let alpha_noise = pve_noise.clamp(0.0, 1.0).sqrt();
    let alpha_batch = pve_batch.clamp(0.0, 1.0).sqrt();
    let alpha_invariant_batch = (1.0 - pve_batch.clamp(0.0, 1.0)).sqrt();

    // Effective dictionary (γ ⊙ β).
    let eff_dk: std::borrow::Cow<Mat> = match gamma_dk {
        Some(g) => std::borrow::Cow::Owned(g.component_mul(dict_dk)),
        None => std::borrow::Cow::Borrowed(dict_dk),
    };

    // Stage-1 baseline `log μ̂_g`, optionally rescaled so the reference's
    // mean library size matches `depth_target` (mirrors topic ref mode).
    let lib_ref: f32 = fit.mu_hat.iter().sum::<f32>().max(1e-30);
    let depth_log_offset = match depth_target {
        Some(d) if d > 0 => ((d as f32).max(1.0) / lib_ref).ln(),
        _ => 0.0,
    };
    let log_mu_hat: DVector<f32> = DVector::from_iterator(
        dd,
        fit.mu_hat
            .iter()
            .map(|&m| m.max(1e-30).ln() + depth_log_offset),
    );
    if depth_log_offset != 0.0 {
        info!(
            "{}: μ̂ rescaled by depth/lib_ref = {:?}/{:.0} = {:.4}",
            label,
            depth_target,
            lib_ref,
            depth_log_offset.exp(),
        );
    }

    let mut rng = StdRng::seed_from_u64(rseed);

    // Stage-2 batch covariance.
    let batch_cov: CopulaCovariance = if batch_rank == 0 {
        CopulaCovariance::random_low_rank(dd, 0, &mut rng)
    } else {
        match batch_program {
            BatchProgram::Empirical => fit.copula.truncate_rank(batch_rank),
            BatchProgram::Random => CopulaCovariance::random_low_rank(dd, batch_rank, &mut rng),
        }
    };
    info!(
        "{}: stage-2 batch program rank={} ({:?}), {} batches",
        label,
        batch_cov.rank(),
        batch_program,
        bb
    );
    // Explicit log-space variance decomposition mirroring synthetic mode:
    //   log δ_{g,b} = √π_batch · z_{g,b} + √(1−π_batch) · w_g
    // z from the (gene-gene-correlated) batch copula; w iid N(0,1) shared
    // across batches (the batch-invariant per-gene shift).
    let normal01 = Normal::new(0.0_f32, 1.0_f32).unwrap();
    let w_invariant: DVector<f32> =
        DVector::from_fn(dd, |_, _| normal01.sample(&mut rng) * alpha_invariant_batch);
    let batch_delta: Vec<DVector<f32>> = (0..bb)
        .map(|_| batch_cov.sample(&mut rng).scale(alpha_batch) + &w_invariant)
        .collect();

    let triplets: Vec<(u64, u64, f32)> = (0..nn)
        .into_par_iter()
        .progress_count(nn as u64)
        .map(|j| -> Vec<(u64, u64, f32)> {
            let mut local_rng = StdRng::seed_from_u64(rseed.wrapping_add(j as u64).wrapping_add(1));
            let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();

            // Stage 1: t = z-scored log(eff · θ_col).
            let bt = &*eff_dk * theta_kn.column(j);
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

            let z_hvg = fit.copula.sample(&mut local_rng);
            let mut counts: Vec<(u64, u64, f32)> = Vec::with_capacity(fit.active_genes.len() / 8);
            for &gidx in &fit.active_genes {
                let mu_g = log_lambda[gidx].exp();
                if !mu_g.is_finite() || mu_g <= 0.0 {
                    continue;
                }
                let nb = NbFit {
                    mu: mu_g,
                    r: fit.r_hat[gidx],
                };
                let z_g = match fit.hvg_pos[gidx] {
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
                    counts.push((gidx as u64, j as u64, x as f32));
                }
            }
            counts
        })
        .flatten()
        .collect();

    Ok((triplets, batch_delta))
}

/// Mean-space wrapper around `core::sample_log_batch_effects`. Returns
/// `exp(log δ)` so callers can multiply directly into the Poisson rate
/// without re-exponentiating per cell.
fn sample_synth_batch_delta(
    dd: usize,
    bb: usize,
    pve_batch: f32,
    rng: &mut impl Rng,
) -> DMatrix<f32> {
    crate::core::sample_log_batch_effects(dd, bb, pve_batch, rng).map(|x| x.exp())
}

fn write_reference_extras(
    out_prefix: &str,
    modality: &str,
    fit: &GlobalCopulaFit,
    batch_delta: &[DVector<f32>],
    feature_names: &[Box<str>],
) -> anyhow::Result<()> {
    let dd = fit.n_genes;
    let bb = batch_delta.len();

    let ln_batch_file = format!("{}.{}.ln_batch.parquet", out_prefix, modality);
    let mut ln_delta_db = DMatrix::<f32>::zeros(dd, bb);
    for (b, col) in batch_delta.iter().enumerate() {
        ln_delta_db.set_column(b, col);
    }
    ln_delta_db.to_parquet_with_names(
        &ln_batch_file,
        (Some(feature_names), Some("feature")),
        None,
    )?;
    info!("wrote {} batch delta: {}", modality, ln_batch_file);

    // Poisson collapses (`r = ∞`) encoded as a large finite sentinel.
    let r_file = format!("{}.{}.r.parquet", out_prefix, modality);
    let mut r_col = DMatrix::<f32>::zeros(dd, 1);
    for g in 0..dd {
        let r = fit.r_hat[g];
        r_col[(g, 0)] = if r.is_finite() { r } else { 1e9 };
    }
    let r_label = ["r_hat".to_string().into_boxed_str()];
    r_col.to_parquet_with_names(
        &r_file,
        (Some(feature_names), Some("feature")),
        Some(&r_label),
    )?;
    info!("wrote {} per-feature NB dispersion r̂: {}", modality, r_file);

    let hvg_file = format!("{}.{}.hvg.gz", out_prefix, modality);
    let hvg_lines: Vec<Box<str>> = fit
        .hvg_indices
        .iter()
        .map(|&g| feature_names[g].clone())
        .collect();
    write_lines(&hvg_lines, &hvg_file)?;
    info!("wrote {} HVG list: {}", modality, hvg_file);

    Ok(())
}

fn generate_peak_names(n_peaks: usize) -> Vec<Box<str>> {
    (0..n_peaks)
        .map(|i| {
            let chr = (i % N_CHROMOSOMES) + 1;
            let start = (i / N_CHROMOSOMES) * (PEAK_BIN_WIDTH + PEAK_GAP);
            let end = start + PEAK_BIN_WIDTH;
            format!("chr{}:{}-{}", chr, start, end).into_boxed_str()
        })
        .collect()
}

fn generate_indexed_names(n: usize, prefix: &str) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{}_{}", prefix, i).into_boxed_str())
        .collect()
}

fn generate_gene_coords(n_genes: usize) -> Vec<GeneTss> {
    (0..n_genes)
        .map(|i| {
            let chr = (i % N_CHROMOSOMES) + 1;
            let gene_on_chr = i / N_CHROMOSOMES;
            let tss =
                gene_on_chr as i64 * (PEAK_BIN_WIDTH + PEAK_GAP) as i64 + PEAK_BIN_WIDTH as i64 / 2;
            GeneTss {
                chr: format!("chr{}", chr).into(),
                tss,
            }
        })
        .collect()
}

fn write_gene_coords(
    gene_names: &[Box<str>],
    coords: &[GeneTss],
    out_prefix: &str,
) -> anyhow::Result<()> {
    let path = format!("{}.gene_coords.tsv.gz", out_prefix);
    let mut writer = open_buf_writer(&path)?;
    writeln!(writer, "gene\tchr\ttss")?;
    for (name, coord) in gene_names.iter().zip(coords.iter()) {
        writeln!(writer, "{}\t{}\t{}", name, coord.chr, coord.tss)?;
    }
    info!("wrote gene coordinates to {}", path);
    Ok(())
}

fn write_ground_truth(
    indicator_genes: &[usize],
    indicator_peaks: &[usize],
    peak_names: &[Box<str>],
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    let path = format!("{}.ground_truth.tsv.gz", out_prefix);
    let mut writer = open_buf_writer(&path)?;
    writeln!(writer, "gene\tpeak")?;
    for i in 0..indicator_genes.len() {
        writeln!(
            writer,
            "{}\t{}",
            gene_names[indicator_genes[i]], peak_names[indicator_peaks[i]]
        )?;
    }
    info!("wrote ground truth to {}", path);
    Ok(())
}

fn write_names(
    out_prefix: &str,
    peak_names: &Vec<Box<str>>,
    gene_names: &Vec<Box<str>>,
    cell_names: &Vec<Box<str>>,
) -> anyhow::Result<()> {
    write_lines(gene_names, &format!("{}.gene_names.txt", out_prefix))?;
    write_lines(peak_names, &format!("{}.peak_names.txt", out_prefix))?;
    write_lines(cell_names, &format!("{}.barcodes.txt", out_prefix))?;
    Ok(())
}
