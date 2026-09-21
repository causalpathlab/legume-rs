//! `peak-to-gene` CLI args and orchestrator.
//!
//! Load multiome or ATAC-only → cell QC → data-beans multilevel pb collapse
//! (optional batch-adjusted rates) → ge-util workflow → E2G-like parquet.
//!
//! ATAC-only: omit `--rna-files`; gene activity is an ArchR-style
//! distance-weighted sum of cis peaks (gene body + exponential decay).

use crate::common::*;
use crate::p2g::abc_map::AbcMapParams;
use crate::p2g::gene_activity::{gene_activity_from_atac_pb, GeneActivityParams};
use crate::p2g::input::{
    load_all_gene_coords_tsv, load_all_gene_loci_from_gff, load_atac_data, load_gene_coords_tsv,
    load_paired_data, GeneUniverse,
};
use crate::p2g::workflow::{run_from_pseudobulk, PbMultiome, WorkflowParams};
use data_beans::alg::collapse_data::MultilevelParams;
use data_beans::alg::refine_multilevel::RefineParams;
use genomic_data::coordinates::{load_gene_tss, parse_peak_coordinates, GeneTss};
use graph_embedding_util::fne::FneConfig;
use legume_numeric::candle::candle_core::Device;
use log::info;

#[derive(Args, Debug)]
pub struct PeakToGeneArgs {
    /* Input */
    #[arg(
        long,
        value_delimiter = ',',
        help = "RNA sparse matrices (zarr/h5), comma-separated.\n\
                Omit for ATAC-only (ArchR-style gene activity from cis peaks)"
    )]
    rna_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "ATAC sparse matrices (zarr/h5), comma-separated"
    )]
    atac_files: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Batch label files.\n\
                Multiome: one per file in RNA-then-ATAC order.\n\
                ATAC-only: one per ATAC file"
    )]
    batch_files: Option<Vec<Box<str>>>,

    /// Shared cell QC (on by default; `--no-qc` to disable).
    #[command(flatten)]
    qc: data_beans::qc_lib::QcArgs,

    /* Cis */
    #[arg(
        long,
        default_value_t = 500_000,
        help = "Cis-window in bp around each gene TSS (peak midpoint distance).\n\
                Used for ABC / refine; gene activity uses --gene-activity-window"
    )]
    cis_window: i64,

    #[arg(
        long,
        default_value_t = 100_000,
        help = "ArchR-style gene-activity window (bp from extended gene body)"
    )]
    gene_activity_window: i64,

    #[arg(
        long,
        default_value_t = 5000.0,
        help = "ArchR geneModel decay lengthscale (bp) for gene activity"
    )]
    gene_activity_decay: f32,

    #[arg(long, help = "Gene coordinates TSV (gene<TAB>chr<TAB>tss)")]
    gene_coords: Option<Box<str>>,

    #[arg(
        long,
        help = "GFF/GTF annotation for gene TSS / body. Alternative to --gene-coords"
    )]
    gff_file: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 200,
        help = "Max cis-candidate peaks per gene after ranking by weight"
    )]
    max_cis: usize,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Drop ABC / refine edges with Pearson weight ≤ this floor"
    )]
    min_weight: f32,

    /* Pseudobulk / collapse (data-beans multilevel + refine) */
    #[arg(
        long,
        default_value_t = 64,
        help = "Random projection dimension for cell grouping"
    )]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 14,
        help = "Binary sort dimension. Yields ~2^sort_dim pseudobulk samples"
    )]
    sort_dim: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Use batch-adjusted pseudobulk (mu_adjusted) when available"
    )]
    use_adjusted: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Hierarchical refinement levels;\n\
                refined finest level is used (1 = single level)"
    )]
    num_levels: usize,

    /* Embedding / cluster */
    #[arg(
        long,
        default_value_t = 32,
        help = "FNE embedding dimension for peak/gene nodes"
    )]
    embedding_dim: usize,

    #[arg(long, default_value_t = 10, help = "FNE training epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 42, help = "RNG seed for FNE")]
    seed: u64,

    #[arg(
        long,
        default_value_t = 10,
        help = "Min pb samples to keep a cluster / run within-cluster refine"
    )]
    min_cluster_samples: usize,

    #[arg(
        long,
        help = "Optional Leiden target cluster count for pb-sample clustering"
    )]
    num_clusters: Option<usize>,

    /* Output */
    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix: E2G tables under `{out}/`,\n\
                `{out}.peak_embedding.parquet` and `{out}.gene_embedding.parquet`"
    )]
    out: Box<str>,
}

pub fn run_peak_to_gene(args: &PeakToGeneArgs) -> anyhow::Result<()> {
    mkdir_parent(&format!("{}/peaks.parquet", args.out))?;

    let atac_only = args
        .rna_files
        .as_ref()
        .map(|v| v.is_empty())
        .unwrap_or(true);

    if atac_only {
        run_atac_only(args)
    } else {
        run_multiome(args)
    }
}

fn run_multiome(args: &PeakToGeneArgs) -> anyhow::Result<()> {
    let rna_files = args
        .rna_files
        .as_ref()
        .expect("multiome path requires --rna-files");

    let mut paired = load_paired_data(rna_files, &args.atac_files, args.batch_files.as_deref())?;

    if let Some(cfg) = args.qc.to_config() {
        let report = data_beans::qc_lib::compute_qc(&paired.data_stack.stack[0], &cfg, None)?;
        info!(
            "cell QC (RNA-driven): dropping {} / {} cells before peak-to-gene linkage",
            report.n_cells_dropped,
            report.train_keep.len(),
        );
        if report.n_cells_dropped > 0 {
            paired.data_stack.mask_columns_all(&report.train_keep)?;
            paired.batch_membership =
                data_beans::qc_lib::filter_by_keep(&paired.batch_membership, &report.train_keep);
        }
    }

    let gene_names = paired.data_stack.stack[0].row_names()?;
    let peak_names = paired.data_stack.stack[1].row_names()?;

    let (rna_pb, atac_pb) = collapse_finest_pb(&mut paired, args)?;
    info!(
        "Pseudobulk: RNA {}x{}, ATAC {}x{}{}",
        rna_pb.nrows(),
        rna_pb.ncols(),
        atac_pb.nrows(),
        atac_pb.ncols(),
        if args.use_adjusted {
            " (batch-adjusted)"
        } else {
            ""
        }
    );
    if rna_pb.ncols() < 50 {
        info!(
            "warning: only {} pseudobulk samples; correlations may be unstable",
            rna_pb.ncols()
        );
    }

    let peak_coords = parse_peak_coordinates(&peak_names);
    let gene_tss = load_gene_tss_aligned(args, &gene_names)?;

    finish_workflow(
        args,
        PbMultiome {
            rna_pb: &rna_pb,
            atac_pb: &atac_pb,
            gene_tss: &gene_tss,
            peak_coords: &peak_coords,
            gene_names: &gene_names,
            peak_names: &peak_names,
        },
    )
}

fn run_atac_only(args: &PeakToGeneArgs) -> anyhow::Result<()> {
    let mut paired = load_atac_data(&args.atac_files, args.batch_files.as_deref())?;

    if let Some(cfg) = args.qc.to_config() {
        let report = data_beans::qc_lib::compute_qc(&paired.data_stack.stack[0], &cfg, None)?;
        info!(
            "cell QC (ATAC-driven): dropping {} / {} cells before peak-to-gene linkage",
            report.n_cells_dropped,
            report.train_keep.len(),
        );
        if report.n_cells_dropped > 0 {
            paired.data_stack.mask_columns_all(&report.train_keep)?;
            paired.batch_membership =
                data_beans::qc_lib::filter_by_keep(&paired.batch_membership, &report.train_keep);
        }
    }

    let peak_names = paired.data_stack.stack[0].row_names()?;
    let (_, atac_pb) = collapse_finest_pb_single(&mut paired, args)?;
    info!(
        "Pseudobulk (ATAC-only): ATAC {}x{}{}",
        atac_pb.nrows(),
        atac_pb.ncols(),
        if args.use_adjusted {
            " (batch-adjusted)"
        } else {
            ""
        }
    );
    if atac_pb.ncols() < 50 {
        info!(
            "warning: only {} pseudobulk samples; correlations may be unstable",
            atac_pb.ncols()
        );
    }

    let peak_coords = parse_peak_coordinates(&peak_names);
    let (gene_names_all, gene_locs_all) = load_gene_universe(args)?;

    let ga_params = GeneActivityParams {
        window: args.gene_activity_window,
        decay: args.gene_activity_decay,
        ..GeneActivityParams::default()
    };
    let activity = gene_activity_from_atac_pb(&atac_pb, &peak_coords, &gene_locs_all, &ga_params)?;

    // Drop genes with no cis accessibility signal (keeps ABC / FNE tractable).
    let keep: Vec<usize> = (0..activity.nrows())
        .filter(|&g| (0..activity.ncols()).any(|j| activity[(g, j)] > 0.0))
        .collect();
    anyhow::ensure!(
        !keep.is_empty(),
        "ATAC-only gene activity is all-zero; check gene-activity window and gene/peak coordinates"
    );
    info!(
        "Gene activity (ArchR-style): kept {} / {} genes with cis ATAC signal",
        keep.len(),
        gene_names_all.len()
    );

    let gene_names: Vec<Box<str>> = keep.iter().map(|&i| gene_names_all[i].clone()).collect();
    let gene_tss: Vec<Option<GeneTss>> = keep
        .iter()
        .map(|&i| {
            gene_locs_all[i].as_ref().map(|loc| GeneTss {
                chr: loc.chr.clone(),
                tss: loc.tss,
            })
        })
        .collect();
    let mut rna_pb = Mat::zeros(keep.len(), activity.ncols());
    for (new_g, &old_g) in keep.iter().enumerate() {
        for j in 0..activity.ncols() {
            rna_pb[(new_g, j)] = activity[(old_g, j)];
        }
    }

    finish_workflow(
        args,
        PbMultiome {
            rna_pb: &rna_pb,
            atac_pb: &atac_pb,
            gene_tss: &gene_tss,
            peak_coords: &peak_coords,
            gene_names: &gene_names,
            peak_names: &peak_names,
        },
    )
}

fn finish_workflow(args: &PeakToGeneArgs, pb: PbMultiome<'_>) -> anyhow::Result<()> {
    let params = WorkflowParams {
        abc: AbcMapParams {
            cis_window: args.cis_window,
            max_cis: args.max_cis,
            min_weight: args.min_weight,
        },
        fne: FneConfig {
            dim: args.embedding_dim,
            epochs: args.epochs,
            seed: args.seed,
            device: Device::Cpu,
            ..FneConfig::default()
        },
        min_cluster_samples: args.min_cluster_samples,
        target_clusters: args.num_clusters,
    };
    run_from_pseudobulk(&pb, &args.out, &params)
}

fn load_gene_tss_aligned(
    args: &PeakToGeneArgs,
    gene_names: &[Box<str>],
) -> anyhow::Result<Vec<Option<GeneTss>>> {
    anyhow::ensure!(args.cis_window > 0, "--cis-window must be > 0");
    if let Some(path) = &args.gene_coords {
        load_gene_coords_tsv(path, gene_names)
    } else if let Some(path) = &args.gff_file {
        load_gene_tss(path, gene_names)
    } else {
        anyhow::bail!("--cis-window > 0 requires either --gene-coords or --gff-file")
    }
}

fn load_gene_universe(args: &PeakToGeneArgs) -> anyhow::Result<GeneUniverse> {
    anyhow::ensure!(args.cis_window > 0, "--cis-window must be > 0");
    if let Some(path) = &args.gene_coords {
        load_all_gene_coords_tsv(path)
    } else if let Some(path) = &args.gff_file {
        load_all_gene_loci_from_gff(path)
    } else {
        anyhow::bail!("ATAC-only requires --gene-coords or --gff-file to define the gene universe")
    }
}

fn collapse_finest_pb(
    paired: &mut crate::p2g::input::PairedDataWithBatch,
    args: &PeakToGeneArgs,
) -> anyhow::Result<(Mat, Mat)> {
    let levels = collapse_levels(paired, args)?;
    let finest = levels
        .iter()
        .max_by_key(|lvl| pick_pseudobulk(&lvl[0], args.use_adjusted).ncols())
        .expect("levels is non-empty");
    anyhow::ensure!(
        finest.len() >= 2,
        "expected RNA+ATAC collapsed layers, got {}",
        finest.len()
    );
    let rna_pb = pick_pseudobulk(&finest[0], args.use_adjusted).clone();
    let atac_pb = pick_pseudobulk(&finest[1], args.use_adjusted).clone();
    Ok((rna_pb, atac_pb))
}

fn collapse_finest_pb_single(
    paired: &mut crate::p2g::input::PairedDataWithBatch,
    args: &PeakToGeneArgs,
) -> anyhow::Result<((), Mat)> {
    let levels = collapse_levels(paired, args)?;
    let finest = levels
        .iter()
        .max_by_key(|lvl| pick_pseudobulk(&lvl[0], args.use_adjusted).ncols())
        .expect("levels is non-empty");
    let atac_pb = pick_pseudobulk(&finest[0], args.use_adjusted).clone();
    Ok(((), atac_pb))
}

fn collapse_levels(
    paired: &mut crate::p2g::input::PairedDataWithBatch,
    args: &PeakToGeneArgs,
) -> anyhow::Result<Vec<Vec<data_beans::alg::collapse_data::CollapsedOut>>> {
    let block_size: Option<usize> = None;
    info!(
        "Random projection (dim={}, {} cells)...",
        args.proj_dim,
        paired.data_stack.num_columns()?
    );
    let proj = paired.data_stack.project_columns_with_batch_correction(
        args.proj_dim,
        block_size,
        Some(&paired.batch_membership),
    )?;

    let levels = paired.data_stack.collapse_columns_multilevel_vec(
        &proj.proj,
        &paired.batch_membership,
        &MultilevelParams {
            knn_pb_samples: DEFAULT_KNN,
            num_levels: args.num_levels.max(1),
            sort_dim: args.sort_dim,
            num_opt_iter: DEFAULT_OPT_ITER,
            refine: RefineParams::default(),
            output_calibration: legume_numeric::param::traits::CalibrateTarget::All,
            anchor_batches: None,
            bulk_batches: None,
            observe_panels: true,
            keep_finest_stats: false,
            pb_tree: None,
            strata: None,
        },
    )?;
    if levels.is_empty() {
        anyhow::bail!("collapse produced no levels");
    }
    Ok(levels)
}

/// Pick `mu_adjusted` when requested and available, else `mu_observed`.
fn pick_pseudobulk(co: &data_beans::alg::collapse_data::CollapsedOut, use_adjusted: bool) -> &Mat {
    if use_adjusted {
        if let Some(adj) = co.mu_adjusted.as_ref() {
            return adj.posterior_mean();
        }
    }
    co.mu_observed.posterior_mean()
}
