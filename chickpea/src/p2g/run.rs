//! `peak-to-gene` CLI args and orchestrator.
//!
//! Load multiome or ATAC-only → cell QC → data-beans multilevel pb collapse
//! (optional batch-adjusted rates) → ge-util workflow → E2G-like parquet.
//!
//! ATAC-only: omit `--rna-files`; gene activity is an ArchR-style
//! distance-weighted sum of cis peaks (gene body + exponential decay). Under
//! `--link-score pearson` it is the RNA stand-in of the link too (and so
//! correlates peaks with a sum of themselves); under `abc` the link reads
//! only ATAC and the surrogate feeds the pb-sample projection alone.

use crate::common::*;
use crate::p2g::gene_activity::{gene_activity_from_atac_pb, GeneActivityParams};
use crate::p2g::input::{
    load_all_gene_coords_tsv, load_all_gene_loci_from_gff, load_atac_data, load_gene_coords_tsv,
    load_paired_data, GeneUniverse,
};
use crate::p2g::link_map::{LinkParams, LinkScore};
use crate::p2g::pb_levels::{parent_maps, PbLevels};
use crate::p2g::workflow::{run_from_pseudobulk, PbMultiome, WorkflowParams};
use data_beans::alg::collapse_data::MultilevelParams;
use data_beans::alg::refine_multilevel::RefineParams;
use genomic_data::coordinates::{load_gene_tss, parse_peak_coordinates, GeneTss};
use graph_embedding_util::fne::FneConfig;
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

    /* Link score */
    #[arg(
        long,
        default_value_t = LinkScore::Pearson,
        value_enum,
        help = "Peak-gene link statistic.\n\
                pearson: correlation of log1p pseudobulk profiles.\n\
                abc: Engreitz activity x contact, normalized over the window (ATAC only)"
    )]
    link_score: LinkScore,

    #[arg(
        long,
        default_value_t = 500_000,
        help = "Cis-window in bp around each gene TSS (peak midpoint distance).\n\
                Used for the link and the refine; gene activity uses --gene-activity-window"
    )]
    cis_window: i64,

    #[arg(
        long,
        default_value_t = 0,
        help = "Keep only the K best-scoring peaks per gene, globally and per cluster.\n\
                0 keeps every peak above --min-weight"
    )]
    top_k_per_gene: usize,

    #[arg(
        long,
        default_value_t = 0.87,
        help = "abc: power-law contact exponent, C(d) = d^(-gamma)"
    )]
    contact_decay_gamma: f32,

    #[arg(
        long,
        default_value_t = 5_000,
        help = "abc: distances below this many bp count as this distance"
    )]
    contact_min_distance: i64,

    #[arg(
        long,
        default_value_t = 1_000_000,
        help = "abc: the contact at this distance is added to every contact\n\
                (ABC's Hi-C pseudocount)"
    )]
    contact_pseudocount_distance: i64,

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
        help = "Drop link / refine edges with score at or below this floor\n\
                (a correlation for pearson, a window share for abc)"
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
        default_value_t = 3,
        help = "Pseudobulk tree levels.\n\
                Links are scored on the finest level;\n\
                every level joins the embedding as sample nodes"
    )]
    num_levels: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Expression levels per modality\n\
                in the pb-sample × feature relations of the embedding"
    )]
    context_bins: usize,

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
        value_enum,
        default_value_t = ComputeDevice::Cpu,
        help = "Compute device for the FNE training"
    )]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal for cuda / metal")]
    device_no: usize,

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
        help = "Output prefix: E2G tables under `{out}/`;\n\
                `{out}.peak_embedding.parquet`, `.gene_embedding.parquet`,\n\
                `.cell_embedding.parquet` (finest pseudobulks);\n\
                `.pb_tree_embedding.parquet` with more than one level"
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

    let levels = collapse_pb_levels(&mut paired, args)?;
    info!(
        "Pseudobulk: {} genes × {} peaks{}; levels {:?}",
        gene_names.len(),
        peak_names.len(),
        if args.use_adjusted {
            " (batch-adjusted)"
        } else {
            ""
        },
        levels.n_pb_per_level()
    );
    warn_few_samples(levels.n_pb(0), args);

    let peak_coords = parse_peak_coordinates(&peak_names);
    let gene_tss = load_gene_tss_aligned(args, &gene_names)?;

    finish_workflow(
        args,
        PbMultiome {
            levels: &levels,
            gene_activity: None,
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
    let levels = collapse_pb_levels(&mut paired, args)?;
    let atac_pb = &levels.atac[0];
    info!(
        "Pseudobulk (ATAC-only): {} peaks{}; levels {:?}",
        peak_names.len(),
        if args.use_adjusted {
            " (batch-adjusted)"
        } else {
            ""
        },
        levels.n_pb_per_level()
    );
    warn_few_samples(levels.n_pb(0), args);

    let peak_coords = parse_peak_coordinates(&peak_names);
    let (gene_names_all, gene_locs_all) = load_gene_universe(args)?;

    let ga_params = GeneActivityParams {
        window: args.gene_activity_window,
        decay: args.gene_activity_decay,
        ..GeneActivityParams::default()
    };
    let activity = gene_activity_from_atac_pb(atac_pb, &peak_coords, &gene_locs_all, &ga_params)?;

    // Every gene with a locus is linked; a gene whose surrogate row is all
    // zero simply yields no edges (Pearson has no variance, ABC reads ATAC).
    let n_active = (0..activity.nrows())
        .filter(|&g| (0..activity.ncols()).any(|j| activity[(g, j)] > 0.0))
        .count();
    let keep: Vec<usize> = (0..activity.nrows())
        .filter(|&g| gene_locs_all[g].is_some())
        .collect();
    anyhow::ensure!(
        !keep.is_empty(),
        "ATAC-only: no gene with coordinates to link"
    );
    info!(
        "Gene activity (ArchR-style): {n_active} / {} genes with cis ATAC signal; linking {} genes",
        gene_names_all.len(),
        keep.len()
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
            levels: &levels,
            gene_activity: Some(&rna_pb),
            gene_tss: &gene_tss,
            peak_coords: &peak_coords,
            gene_names: &gene_names,
            peak_names: &peak_names,
        },
    )
}

fn warn_few_samples(n_pb: usize, args: &PeakToGeneArgs) {
    if n_pb < 50 && args.link_score == LinkScore::Pearson {
        info!("warning: only {n_pb} pseudobulk samples; correlations may be unstable");
    }
}

fn finish_workflow(args: &PeakToGeneArgs, pb: PbMultiome<'_>) -> anyhow::Result<()> {
    let params = WorkflowParams {
        abc: LinkParams {
            score: args.link_score,
            cis_window: args.cis_window,
            max_cis: args.max_cis,
            min_weight: args.min_weight,
            top_k_per_gene: args.top_k_per_gene,
            contact_gamma: args.contact_decay_gamma,
            contact_min_distance: args.contact_min_distance,
            contact_pseudocount_distance: args.contact_pseudocount_distance,
        },
        fne: FneConfig {
            dim: args.embedding_dim,
            epochs: args.epochs,
            seed: args.seed,
            device: args.device.to_device(args.device_no)?,
            ..FneConfig::default()
        },
        context_bins: args.context_bins,
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

/// Collapse every layer of the stack into the pseudobulk levels. Layer 0
/// (RNA, or ATAC when that is all there is) drives the partition and returns
/// the per-level cell membership; the other layer is collapsed onto that same
/// partition, so the finest matrices match what a stack collapse would give.
fn collapse_pb_levels(
    paired: &mut crate::p2g::input::PairedDataWithBatch,
    args: &PeakToGeneArgs,
) -> anyhow::Result<PbLevels> {
    use data_beans::alg::collapse_data::{
        collapse_columns_multilevel_with_hierarchy, collapse_columns_multilevel_with_partition,
    };
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
    let params = MultilevelParams {
        knn_pb_samples: DEFAULT_KNN,
        num_levels: args.num_levels.max(1),
        sort_dim: args.sort_dim,
        num_opt_iter: DEFAULT_OPT_ITER,
        refine: RefineParams::default(),
        // `All` keeps the Gamma sufficient statistics, which
        // `supported_means` reads; `MeanOnly` would drop them.
        output_calibration: legume_numeric::param::traits::CalibrateTarget::All,
        anchor_batches: None,
        bulk_batches: None,
        observe_panels: true,
        keep_finest_stats: false,
        pb_tree: None,
        strata: None,
    };

    let driver = collapse_columns_multilevel_with_hierarchy(
        &mut paired.data_stack.stack[0],
        &proj.proj,
        &paired.batch_membership,
        &params,
    )?;
    anyhow::ensure!(!driver.levels.is_empty(), "collapse produced no levels");
    let driver_pb = supported_means(&driver.levels, args.use_adjusted);
    let parent = parent_maps(
        &driver.cell_to_pb_per_level,
        &driver_pb.iter().map(Mat::ncols).collect::<Vec<_>>(),
    )?;

    if paired.data_stack.num_types() == 1 {
        return Ok(PbLevels {
            rna: None,
            atac: driver_pb,
            parent,
        });
    }
    let follower = collapse_columns_multilevel_with_partition(
        &mut paired.data_stack.stack[1],
        &proj.proj,
        &paired.batch_membership,
        &params,
        &driver.cell_to_pb_per_level,
    )?;
    Ok(PbLevels {
        rna: Some(driver_pb),
        atac: supported_means(&follower.levels, args.use_adjusted),
        parent,
    })
}

/// Posterior means per level, zeroed where the pb holds no data for the
/// entry (the Gamma prior's floor is regularization, not signal).
fn supported_means(
    levels: &[data_beans::alg::collapse_data::CollapsedOut],
    use_adjusted: bool,
) -> Vec<Mat> {
    levels
        .iter()
        .map(|co| {
            let mean = pick_pseudobulk(co, use_adjusted);
            Mat::from_fn(mean.nrows(), mean.ncols(), |r, c| {
                if co.mu_observed.has_data_support(r, c) {
                    mean[(r, c)]
                } else {
                    0.0
                }
            })
        })
        .collect()
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
