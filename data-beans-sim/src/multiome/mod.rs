//! Paired ATAC + RNA simulator (`data-beans-sim multiome`).
//!
//! Without any `--reference-*`: a **two-step** generative model.
//!
//! Step 1 — ATAC from topics (`build_peak_logits`). Peak `p`'s log-accessibility is
//! `A_pj = base_p + σ·(√π_topic·T_p + √π_priv·P_p + √π_noise·N_p [+ √π_batch·B_p])`,
//! where `T = std(log(β_p·θ))` is cell-type on/off switching and `P` a peak-PRIVATE
//! fluctuation; the peak budget `{topic, private, noise, batch}` is normalized to 1.
//! A fraction of causal peaks are topic-INVARIANT (pure-private → cleanly identifiable).
//!
//! Step 2 — RNA conditional on enhancers (`build_gene_logits`). A linked gene inherits
//! its causal peaks' regulatory signal `sig = √π_topic·T + √π_priv·P` through a
//! cell-type-INVARIANT cis link: `E_gj = σ·(√pve_cis·std(Σ_{p∈M_g} sig_p) +
//! √(1−pve_cis)·N_g [+ batch])`. The gene has no topic path of its own — cell-type
//! specificity propagates through its peaks; unlinked genes are noise. Counts are
//! `Poisson(depth_j · softmax(·))`; peak-gene ground truth is `M[G,P]`.
//!
//! With `--reference-rna`/`--reference-atac`: per-modality two-stage GLM + NB+copula
//! sampling (`fit_global_copula`, no cross-modality copula); the `{topic, noise, batch}`
//! budget (normalized, no cis) weights the log-rate.

mod sample;

pub use sample::{sample_nested_topic_proportions, sample_poisson_from_logits};

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
        default_value_t = 0.0,
        help = "Cis propagation (gene-level, in [0,1]): proportion of a LINKED gene's \
                log-expression variance inherited from its causal peaks' regulatory \
                signal (the rest is gene-intrinsic noise). 0 ⇒ gene decoupled from its \
                enhancers; 1 ⇒ fully enhancer-explained. A gene has no topic path of its \
                own — cell-type specificity propagates through its peaks."
    )]
    pub pve_cis: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Fraction of causal peaks made topic-INVARIANT (pure-private accessibility): \
                their cis links are cleanly recoverable (no topic confounding). A clean \
                positive-control set alongside the harder topic-driven links."
    )]
    pub invariant_causal_fraction: f32,

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
        help = "Peak topic weight: (unnormalized) share of a PEAK's log-accessibility \
                variance from the shared cell-state program (cell-type on/off). The peak \
                budget {topic, private, noise, batch} is normalized to sum to 1."
    )]
    pub pve_topic: f32,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "Peak-private weight: (unnormalized) share of a peak's log-accessibility \
                from a peak-PRIVATE fluctuation (independent of cell type). This is the \
                identifiability dial — only a true enhancer's private signal reaches its \
                gene; co-active bystanders share only the topic part. 0 ⇒ peaks collinear \
                within a cell type ⇒ cis links unidentifiable."
    )]
    pub pve_private: f32,

    #[arg(
        long,
        default_value_t = 0.8,
        help = "θ geometry: coarse-topic concentration per cell (1 = one-hot cell \
                states, 0 = uniform). Sets how distinct cell states are; separate \
                from the topic budget weight."
    )]
    pub topic_concentration: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "θ geometry: subtype concentration within the dominant coarse topic \
                (only used when --n-sub-topics > 1)."
    )]
    pub subtopic_concentration: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Total systematic log-rate SD σ (overall dynamic range), orthogonal \
                to the variance-budget shares."
    )]
    pub log_signal_sd: f32,

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

    /////////////////////////////////////////////
    // Reference / copula flags (per modality) //
    /////////////////////////////////////////////
    #[arg(
        long,
        help = "Real single-cell ATAC reference (.h5, .zarr, .zarr.zip)",
        long_help = "Real single-cell ATAC reference (`.h5`, `.zarr`, `.zarr.zip`). When set,\n\
                     the ATAC sampler switches to two-stage GLM + NB+copula PIT (per-peak\n\
                     `r̂` + global Σ̂ from the reference). The reference's row count\n\
                     overrides `--n-peaks`."
    )]
    pub reference_atac: Option<Box<str>>,

    #[arg(
        long,
        help = "Real single-cell RNA reference. Symmetric to `--reference-atac`",
        long_help = "Real single-cell RNA reference. Symmetric to `--reference-atac`. The\n\
                     reference's row count overrides `--n-genes`."
    )]
    pub reference_rna: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 2000,
        help = "HVG / HVP count for each modality's gene-gene (peak-peak) copula",
        long_help = "HVG / HVP count for each modality's gene-gene (peak-peak) copula.\n\
                     Features outside the HV set are sampled independently from `NB(λ, r̂)`."
    )]
    pub n_hvg: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Maximum rank of the per-modality low-rank Σ̂ factor"
    )]
    pub copula_rank: usize,

    #[arg(
        long,
        default_value_t = 1e-3,
        help = "Per-feature isotropic ridge variance added at sample time on top of Σ̂"
    )]
    pub regularization: f32,

    #[arg(
        long,
        default_value_t = 1e-2,
        help = "Lower bound on the NB size parameter r̂",
        long_help = "Lower bound on the NB size parameter `r̂`. Tames runaway dispersion when\n\
                     MoM yields a near-zero `r` for noisy features."
    )]
    pub r_floor: f32,

    #[arg(
        long,
        default_value_t = 1,
        help = "Number of batches (per-cell membership is uniform)",
        long_help = "Number of batches (per-cell membership is uniform). Stage-2 batch\n\
                     perturbation is fitted per modality; same membership across modalities."
    )]
    pub batches: usize,

    #[arg(
        long,
        default_value_t = 2,
        help = "Rank of the batch-program subspace in reference mode",
        long_help = "Rank of the batch-program subspace in reference mode. `0` = iid\n\
                     (Splatter-style); `2-3` = co-shifted batch program."
    )]
    pub batch_rank: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = BatchProgram::Random,
        help = "Where the batch-program subspace comes from when --batch-rank > 0"
    )]
    pub batch_program: BatchProgram,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Peak-noise weight: (unnormalized) share of a peak's log-accessibility from a per-cell residual noise term",
        long_help = "Peak-noise weight: (unnormalized) share of a peak's log-accessibility from a\n\
                     per-cell residual noise term. Normalized with the other peak-budget weights."
    )]
    pub pve_noise: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Batch weight: (unnormalized) share of log-rate variance from batch effects",
        long_help = "Batch weight: (unnormalized) share of log-rate variance from batch effects\n\
                     (used when --batches > 1). Normalized into the peak and gene budgets."
    )]
    pub pve_batch: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Fraction of cells observed in BOTH modalities (1.0 = fully paired, 0.0 = fully disjoint)",
        long_help = "Fraction of cells observed in BOTH modalities. `1.0` (default)\n\
                     keeps the historical paired-multiome behavior — ATAC and RNA\n\
                     share `--n-cells` barcodes one-to-one. `0.0` makes the two\n\
                     modalities fully disjoint (each gets `--n-cells` unique\n\
                     barcodes; no cell is in both files). In-between gives **patchy\n\
                     multiome**: `floor(n_cells * fraction)` shared cells plus\n\
                     `n_cells - floor(...)` modality-only cells per modality.\n\
                     \n\
                     Shared cells are named `cell_<i>` and appear identically in\n\
                     both files. Modality-only cells are named `atac_cell_<i>` or\n\
                     `rna_cell_<i>` and appear only in their file. Use to drive\n\
                     `senna gbe --multiome` / `senna itopic --multiome` integration\n\
                     tests at known overlap fractions."
    )]
    pub cell_overlap_fraction: f32,
}

pub fn run_multiome(args: &MultiomeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let mut rng = StdRng::seed_from_u64(args.rseed);
    let nn = args.n_cells;
    let kk = args.n_topics;
    let k_sub = args.n_sub_topics.max(1);
    let k_total = kk * k_sub;

    ////////////////////////////////////
    // Patchy multiome cell partition //
    ////////////////////////////////////
    // `cell_overlap_fraction = 1.0` keeps today's matched behavior
    // (nn cells shared between ATAC and RNA, identical barcodes).
    // `0.0` makes the two modalities fully disjoint. Anything in
    // between gives partial overlap — drives the `--multiome`
    // integration test path.
    let overlap = args.cell_overlap_fraction.clamp(0.0, 1.0);
    let nn_shared: usize = ((nn as f32) * overlap).round() as usize;
    let nn_atac_only: usize = nn.saturating_sub(nn_shared);
    let nn_rna_only: usize = nn.saturating_sub(nn_shared);
    let nn_total: usize = nn_shared + nn_atac_only + nn_rna_only;
    if overlap < 1.0 {
        info!(
            "patchy multiome: {} shared cells, {} ATAC-only, {} RNA-only \
             (total unique = {})",
            nn_shared, nn_atac_only, nn_rna_only, nn_total
        );
    }
    // Global cell indices [0, nn_total): the layout is
    //   [0, nn_shared)                      → shared "cell_<i>"
    //   [nn_shared, nn_shared + nn_atac_only) → "atac_cell_<i>"
    //   [nn_shared + nn_atac_only, nn_total)  → "rna_cell_<i>"
    let atac_indices: Vec<usize> = (0..(nn_shared + nn_atac_only)).collect();
    let rna_indices: Vec<usize> = (0..nn_shared)
        .chain((nn_shared + nn_atac_only)..nn_total)
        .collect();
    debug_assert_eq!(atac_indices.len(), nn);
    debug_assert_eq!(rna_indices.len(), nn);

    ////////////////////////////////////////////////////////
    // Open references (if any) and resolve modality dims //
    ////////////////////////////////////////////////////////
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
         (topic_conc={}, subtopic_conc={}, gene_topic_sd={}, atac_ref={}, rna_ref={})",
        g,
        p,
        nn,
        kk,
        k_sub,
        k_total,
        args.topic_concentration,
        args.subtopic_concentration,
        args.gene_topic_sd,
        atac_fit.is_some(),
        rna_fit.is_some(),
    );

    ////////////////////////////////
    // Topic proportions (nested) //
    ////////////////////////////////
    // Sample for all `nn_total` unique cells; slice per modality below.
    let theta_seed = rng.next_u64();
    let (theta_full, theta_coarse) = sample::sample_nested_topic_proportions(
        kk,
        k_sub,
        nn_total,
        args.topic_concentration,
        args.subtopic_concentration,
        theta_seed,
    );
    // ATAC consumes the coarse topic axis (K), RNA the full nested
    // axis (K * K_sub). Slice per modality.
    let theta_coarse_atac = theta_coarse.select_columns(&atac_indices);
    let theta_full_rna = theta_full.select_columns(&rna_indices);

    //////////////////
    // Dictionaries //
    //////////////////
    let beta_ext = sample::sample_dictionary(p, k_total, &mut rng);
    let beta_atac = sample::marginalize_dictionary(&beta_ext, kk, k_sub);

    ///////////
    // Names //
    ///////////
    let peak_names: Vec<Box<str>> = atac_fit
        .as_ref()
        .map(|f| f.gene_names.clone())
        .unwrap_or_else(|| generate_peak_names(p));
    let gene_names: Vec<Box<str>> = rna_fit
        .as_ref()
        .map(|f| f.gene_names.clone())
        .unwrap_or_else(|| generate_indexed_names(g, "gene"));
    // Unified cell-name index — keyed to the `theta_*` layout above.
    // Used for ground-truth parquets (theta, proportions, etc.). Per-
    // modality cell name vectors below are what gets registered on
    // each .zarr backend (which is what `senna gbe`/`itopic` will see).
    let cell_names: Vec<Box<str>> = {
        let mut v: Vec<Box<str>> = Vec::with_capacity(nn_total);
        v.extend(generate_indexed_names(nn_shared, "cell"));
        v.extend(generate_indexed_names(nn_atac_only, "atac_cell"));
        v.extend(generate_indexed_names(nn_rna_only, "rna_cell"));
        v
    };
    let atac_cell_names: Vec<Box<str>> = atac_indices
        .iter()
        .map(|&i| cell_names[i].clone())
        .collect();
    let rna_cell_names: Vec<Box<str>> =
        rna_indices.iter().map(|&i| cell_names[i].clone()).collect();
    let gene_coords = generate_gene_coords(g);

    ///////////////////////////////
    // Indicator matrix M[G × P] //
    ///////////////////////////////
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

    ///////////////////////////////////////////////////////
    // Derived RNA dictionary W[G × K_total] = M · β_ext //
    ///////////////////////////////////////////////////////
    let w_gk = sample::build_derived_dictionary(&indicator_genes, &indicator_peaks, &beta_ext, g);

    ///////////////////////////////////////////////
    // Optional gene-topic effect γ[G × K_total] //
    ///////////////////////////////////////////////
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

    ////////////////////////////////////////////////////////////////////
    // Batch membership (per unified cell; per-modality slices below) //
    ////////////////////////////////////////////////////////////////////
    let bb = args.batches.max(1);
    let runif = rand_distr::Uniform::new(0, bb).expect("unif [0 .. bb)");
    let batch_membership: Vec<usize> = (0..nn_total).map(|_| runif.sample(&mut rng)).collect();
    let batch_membership_atac: Vec<usize> =
        atac_indices.iter().map(|&i| batch_membership[i]).collect();
    let batch_membership_rna: Vec<usize> =
        rna_indices.iter().map(|&i| batch_membership[i]).collect();

    /////////////////////////////////////////////////////////////
    // Synthetic two-step generative model (no-reference mode) //
    /////////////////////////////////////////////////////////////
    // Step 1: ATAC accessibility from topics. A peak's regulatory signal mixes a topic
    // component (cell-type on/off) and a peak-PRIVATE fluctuation; the private share is
    // the identifiability dial. A fraction of causal peaks are topic-INVARIANT (pure
    // private) → cleanly recoverable links.
    // Step 2: a linked gene inherits Σ of its causal peaks' regulatory signal scaled by
    // `pve_cis`; the rest is gene-intrinsic noise. The gene has no topic path of its own
    // — cell-type specificity propagates through its enhancers.
    let synth = atac_fit.is_none() && rna_fit.is_none();
    let mut causal_by_gene: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (&gi, &pi) in indicator_genes.iter().zip(indicator_peaks.iter()) {
        causal_by_gene.entry(gi).or_default().push(pi);
    }
    let (batch_log_atac, batch_log_rna): (Option<DMatrix<f32>>, Option<DMatrix<f32>>) =
        if synth && bb > 1 {
            let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();
            let mut ba = DMatrix::<f32>::zeros(p, bb);
            ba.iter_mut().for_each(|v| *v = normal.sample(&mut rng));
            let mut br = DMatrix::<f32>::zeros(g, bb);
            br.iter_mut().for_each(|v| *v = normal.sample(&mut rng));
            (Some(ba), Some(br))
        } else {
            (None, None)
        };
    let (peak_logits, gene_logits): (Option<DMatrix<f32>>, Option<DMatrix<f32>>) = if synth {
        // Topic-invariant causal peaks (pure-private accessibility).
        let mut is_invariant = vec![false; p];
        if args.invariant_causal_fraction > 0.0 {
            let mut cps: Vec<usize> = indicator_peaks.clone();
            cps.sort_unstable();
            cps.dedup();
            cps.shuffle(&mut rng);
            let n_inv = (cps.len() as f32 * args.invariant_causal_fraction.clamp(0.0, 1.0)).round()
                as usize;
            for &pp in cps.iter().take(n_inv) {
                is_invariant[pp] = true;
            }
            info!("topic-invariant causal peaks: {}/{}", n_inv, cps.len());
        }
        // Peak-private fluctuation [P × nn] (identifiability source).
        let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();
        let mut priv_mat = DMatrix::<f32>::zeros(p, nn);
        priv_mat
            .iter_mut()
            .for_each(|v| *v = normal.sample(&mut rng));

        info!(
            "synthetic two-step: ATAC←topics, RNA←enhancers (pve_cis={})",
            args.pve_cis
        );
        let (pl, sig) = build_peak_logits(
            &beta_atac,
            &theta_coarse_atac,
            &is_invariant,
            &priv_mat,
            batch_log_atac.as_ref(),
            &batch_membership_atac,
            (
                args.pve_topic,
                args.pve_private,
                args.pve_noise,
                args.pve_batch,
            ),
            args.log_signal_sd,
            &mut rng,
        );
        let gl = build_gene_logits(
            &sig,
            &causal_by_gene,
            g,
            nn,
            nn_shared,
            batch_log_rna.as_ref(),
            &batch_membership_rna,
            args.pve_cis,
            args.pve_batch,
            args.log_signal_sd,
            &mut rng,
        );
        (Some(pl), Some(gl))
    } else {
        (None, None)
    };

    /////////////////
    // ATAC counts //
    /////////////////
    let (atac_triplets, atac_batch_delta) = if let Some(fit) = atac_fit.as_ref() {
        let (trips, batch_delta) = sample_with_reference(
            &beta_atac,
            &theta_coarse_atac,
            None,
            fit,
            &batch_membership_atac,
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
            "sampling ATAC counts: {} peaks × {} cells (two-step)",
            p, nn
        );
        let logits = peak_logits.as_ref().expect("synthetic peak logits");
        let trips = sample::sample_poisson_from_logits(logits, &rho, rng.next_u64());
        (trips, None)
    };
    info!("ATAC: {} non-zeros", atac_triplets.len());

    ////////////////
    // RNA counts //
    ////////////////
    let (rna_triplets, rna_batch_delta) = if let Some(fit) = rna_fit.as_ref() {
        let (trips, batch_delta) = sample_with_reference(
            &w_gk,
            &theta_full_rna,
            gamma_gk.as_ref(),
            fit,
            &batch_membership_rna,
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
        info!("sampling RNA counts: {} genes × {} cells (two-step)", g, nn);
        let logits = gene_logits.as_ref().expect("synthetic gene logits");
        let trips = sample::sample_poisson_from_logits(logits, &tau, rng.next_u64());
        (trips, None)
    };
    info!("RNA: {} non-zeros", rna_triplets.len());

    ////////////////////////////
    // Persist sparse outputs //
    ////////////////////////////
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
    atac_data.register_column_names_vec(&atac_cell_names);
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
    rna_data.register_column_names_vec(&rna_cell_names);
    finalize_zarr_output(&rna_dir, &rna_final)?;
    info!("wrote RNA sparse backend: {}", rna_final);

    ///////////////////////////////////
    // Companion parquet / TSV files //
    ///////////////////////////////////
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

    // Synthetic-mode raw batch log-effects (one per modality where bb > 1 and no ref).
    if let Some(ref bl) = batch_log_atac {
        let f = format!("{}.atac.ln_batch.parquet", args.out);
        bl.to_parquet_with_names(&f, (Some(&peak_names), Some("peak")), None)?;
        info!("wrote ATAC batch log-effects: {}", f);
    }
    if let Some(ref bl) = batch_log_rna {
        let f = format!("{}.rna.ln_batch.parquet", args.out);
        bl.to_parquet_with_names(&f, (Some(&gene_names), Some("gene")), None)?;
        info!("wrote RNA batch log-effects: {}", f);
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

/// Standardize a row to mean 0, unit variance (in place). Zero-variance rows
/// (e.g. an absent component) are zeroed.
fn standardize_inplace(v: &mut [f32]) {
    let n = v.len().max(1) as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let sd = var.sqrt();
    if sd < 1e-8 {
        v.iter_mut().for_each(|x| *x = 0.0);
    } else {
        v.iter_mut().for_each(|x| *x = (*x - mean) / sd);
    }
}

/// Step 1 — peak log-accessibility from topics. Per peak, mix a standardized topic
/// component `T = std(log(β·θ))` (cell-type on/off), a peak-PRIVATE fluctuation `P`,
/// noise, and batch, with √π weights from the normalized peak budget
/// `{topic, private, noise, batch}`. Topic-invariant peaks move their topic mass to
/// `P` (pure-private accessibility → cleanly identifiable links). Returns the peak
/// logits `[P×N]` AND the regulatory signal `sig = √π_topic·T + √π_priv·P` `[P×N]`
/// (no noise/batch) that genes inherit in step 2. A per-peak topic baseline preserves
/// abundance.
#[allow(clippy::too_many_arguments)]
fn build_peak_logits(
    beta: &DMatrix<f32>,
    theta: &DMatrix<f32>,
    is_invariant: &[bool],
    priv_mat: &DMatrix<f32>,
    batch_log: Option<&DMatrix<f32>>,
    batch_membership: &[usize],
    pve: (f32, f32, f32, f32), // topic, private, noise, batch
    sigma: f32,
    rng: &mut StdRng,
) -> (DMatrix<f32>, DMatrix<f32>) {
    let p = beta.nrows();
    let kk = beta.ncols();
    let ncol = theta.ncols();
    let (pt, ppriv, pn, pbt) = pve;
    let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();

    let mut logits = DMatrix::<f32>::zeros(p, ncol);
    let mut sig = DMatrix::<f32>::zeros(p, ncol);
    for f in 0..p {
        let mut t_row: Vec<f32> = (0..ncol)
            .map(|j| {
                let mut s = 0.0f32;
                for k in 0..kk {
                    s += beta[(f, k)] * theta[(k, j)];
                }
                (s + 1e-8).ln()
            })
            .collect();
        let base_f = t_row.iter().sum::<f32>() / ncol as f32;
        standardize_inplace(&mut t_row);

        let mut p_row: Vec<f32> = (0..ncol).map(|j| priv_mat[(f, j)]).collect();
        standardize_inplace(&mut p_row);

        let b_row: Option<Vec<f32>> = batch_log.map(|bl| {
            let mut br: Vec<f32> = (0..ncol).map(|j| bl[(f, batch_membership[j])]).collect();
            standardize_inplace(&mut br);
            br
        });

        let mut n_row: Vec<f32> = (0..ncol).map(|_| normal.sample(rng)).collect();
        standardize_inplace(&mut n_row);

        // Invariant peaks: topic mass folds into the private share.
        let (topic_w, priv_w) = if is_invariant[f] {
            (0.0, pt.max(0.0) + ppriv.max(0.0))
        } else {
            (pt.max(0.0), ppriv.max(0.0))
        };
        let cbt = if b_row.is_some() { pbt.max(0.0) } else { 0.0 };
        let ssum = topic_w + priv_w + pn.max(0.0) + cbt;
        let (wt, wp, wn, wb) = if ssum <= 1e-12 {
            (0.0, 1.0, 0.0, 0.0)
        } else {
            (
                (topic_w / ssum).sqrt(),
                (priv_w / ssum).sqrt(),
                (pn.max(0.0) / ssum).sqrt(),
                (cbt / ssum).sqrt(),
            )
        };

        for j in 0..ncol {
            let s = wt * t_row[j] + wp * p_row[j]; // regulatory signal (no noise/batch)
            sig[(f, j)] = s;
            let mut v = s + wn * n_row[j];
            if let Some(br) = b_row.as_ref() {
                v += wb * br[j];
            }
            logits[(f, j)] = base_f + sigma * v;
        }
    }
    (logits, sig)
}

/// Step 2 — gene log-expression conditional on upstream enhancers. A linked gene
/// inherits its causal peaks' regulatory signal, `C = std(Σ_{p∈M_g} sig_p)` over the
/// shared cells, with proportion `pve_cis` of its variance; the rest is gene noise
/// (and batch). The cis weights are cell-type-INVARIANT — a gene has no topic path of
/// its own. Unlinked genes get noise only. `sig` is indexed by ATAC cell; the first
/// `nn_shared` columns are the cells shared with RNA.
#[allow(clippy::too_many_arguments)]
fn build_gene_logits(
    sig: &DMatrix<f32>,
    causal_by_gene: &std::collections::HashMap<usize, Vec<usize>>,
    g: usize,
    ncol: usize,
    nn_shared: usize,
    batch_log: Option<&DMatrix<f32>>,
    batch_membership: &[usize],
    pve_cis: f32,
    pve_batch: f32,
    sigma: f32,
    rng: &mut StdRng,
) -> DMatrix<f32> {
    let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();
    let pc = pve_cis.clamp(0.0, 1.0);
    let shared = nn_shared.min(ncol);

    let mut logits = DMatrix::<f32>::zeros(g, ncol);
    for gi in 0..g {
        let linked = causal_by_gene.get(&gi);
        let mut c_row = vec![0.0f32; ncol];
        if let Some(peaks) = linked {
            for s in 0..shared {
                c_row[s] = peaks.iter().map(|&pp| sig[(pp, s)]).sum();
            }
            standardize_inplace(&mut c_row);
        }

        let b_row: Option<Vec<f32>> = batch_log.map(|bl| {
            let mut br: Vec<f32> = (0..ncol).map(|j| bl[(gi, batch_membership[j])]).collect();
            standardize_inplace(&mut br);
            br
        });

        let mut n_row: Vec<f32> = (0..ncol).map(|_| normal.sample(rng)).collect();
        standardize_inplace(&mut n_row);

        // Gene budget: cis + noise = 1 (+ batch, renormalized).
        let cis_w = if linked.is_some() { pc } else { 0.0 };
        let noise_w = 1.0 - cis_w;
        let cbt = if b_row.is_some() {
            pve_batch.max(0.0)
        } else {
            0.0
        };
        let ssum = cis_w + noise_w + cbt;
        let (wc, wn, wb) = (
            (cis_w / ssum).sqrt(),
            (noise_w / ssum).sqrt(),
            (cbt / ssum).sqrt(),
        );

        for j in 0..ncol {
            let mut v = wc * c_row[j] + wn * n_row[j];
            if let Some(br) = b_row.as_ref() {
                v += wb * br[j];
            }
            logits[(gi, j)] = sigma * v; // base_g = 0 (uniform gene abundance)
        }
    }
    logits
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

    // Normalize the {topic, noise, batch} budget to a simplex (reference mode has
    // no cis path); √π_x are the standardized-component coefficients.
    let bt = pve_topic.max(0.0);
    let bn = pve_noise.max(0.0);
    let bbt = pve_batch.max(0.0);
    let bsum = (bt + bn + bbt).max(1e-12);
    let (pi_topic, pi_noise, pi_batch) = (bt / bsum, bn / bsum, bbt / bsum);
    let alpha_topic = pi_topic.sqrt();
    let alpha_noise = pi_noise.sqrt();
    let alpha_batch = pi_batch.sqrt();
    let alpha_invariant_batch = (1.0 - pi_batch).sqrt();

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The budget is normalized to Σπ = 1, so a feature with all components
    /// present has log-rate variance ≈ σ² regardless of how the weights split
    /// (independent, per-feature-standardized components).
    #[test]
    fn budget_logits_total_variance_is_sigma_sq() {
        let (d, k, ncol) = (40usize, 4usize, 600usize);
        let mut rng = StdRng::seed_from_u64(7);
        let fill = |rng: &mut StdRng, rows: usize, cols: usize, lo: f32, hi: f32| {
            let mut m = DMatrix::<f32>::zeros(rows, cols);
            m.iter_mut().for_each(|v| *v = rng.random_range(lo..hi));
            m
        };
        let dict = fill(&mut rng, d, k, 0.05, 1.0);
        let mut theta = fill(&mut rng, k, ncol, 0.0, 1.0);
        for j in 0..ncol {
            let s: f32 = (0..k).map(|kk| theta[(kk, j)]).sum::<f32>().max(1e-6);
            for kk in 0..k {
                theta[(kk, j)] /= s;
            }
        }
        let priv_mat = fill(&mut rng, d, ncol, -1.0, 1.0);
        let batch = fill(&mut rng, d, 2, -1.0, 1.0);
        let memb: Vec<usize> = (0..ncol).map(|j| j % 2).collect();
        let is_invariant = vec![false; d];
        let sigma = 1.3f32;

        let (logits, _sig) = build_peak_logits(
            &dict,
            &theta,
            &is_invariant,
            &priv_mat,
            Some(&batch),
            &memb,
            (1.0, 2.0, 0.5, 0.5), // topic, private, noise, batch
            sigma,
            &mut rng,
        );

        let s2 = (sigma * sigma) as f64;
        for f in 0..d {
            let row: Vec<f64> = (0..ncol).map(|j| logits[(f, j)] as f64).collect();
            let mean = row.iter().sum::<f64>() / ncol as f64;
            let var = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / ncol as f64;
            assert!(
                (var - s2).abs() < 0.3 * s2,
                "feature {f}: var {var:.3} vs σ²={s2:.3}"
            );
        }
    }
}
