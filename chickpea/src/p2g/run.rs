//! `peak-to-gene` CLI args and orchestrator.
//!
//! Load paired multiome → cell QC → data-beans multilevel pb collapse
//! (optional batch-adjusted rates) → ge-util workflow → E2G-like parquet.

use crate::common::*;
use crate::p2g::abc_map::AbcMapParams;
use crate::p2g::input::{load_gene_coords_tsv, load_paired_data};
use crate::p2g::workflow::{run_from_pseudobulk, PbMultiome, WorkflowParams};
use data_beans::alg::collapse_data::MultilevelParams;
use data_beans::alg::refine_multilevel::RefineParams;
use genomic_data::coordinates::{load_gene_tss, parse_peak_coordinates};
use graph_embedding_util::fne::FneConfig;
use legume_numeric::candle::candle_core::Device;
use log::info;

#[derive(Args, Debug)]
pub struct PeakToGeneArgs {
    /* Input */
    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "RNA sparse matrices (zarr/h5), comma-separated"
    )]
    rna_files: Vec<Box<str>>,

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
        help = "Batch label files, one per data file in RNA-then-ATAC order"
    )]
    batch_files: Option<Vec<Box<str>>>,

    /// Shared cell QC (on by default; `--no-qc` to disable).
    #[command(flatten)]
    qc: data_beans::qc_lib::QcArgs,

    /* Cis */
    #[arg(
        long,
        default_value_t = 500_000,
        help = "Cis-window in bp around each gene TSS (peak midpoint distance)"
    )]
    cis_window: i64,

    #[arg(long, help = "Gene coordinates TSV (gene<TAB>chr<TAB>tss)")]
    gene_coords: Option<Box<str>>,

    #[arg(
        long,
        help = "GFF/GTF annotation for gene TSS. Alternative to --gene-coords"
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
        help = "Output directory prefix (writes peaks/clusters/peak_gene parquet)"
    )]
    out: Box<str>,
}

pub fn run_peak_to_gene(args: &PeakToGeneArgs) -> anyhow::Result<()> {
    mkdir_parent(&format!("{}/peaks.parquet", args.out))?;

    /* 1. Load paired RNA + ATAC */
    let mut paired = load_paired_data(
        &args.rna_files,
        &args.atac_files,
        args.batch_files.as_deref(),
    )?;

    /* 1b. Cell QC (RNA-driven), mask both modalities */
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

    /* 2. data-beans multilevel pb collapse (+ optional batch adjustment) */
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
    let finest = levels
        .iter()
        .max_by_key(|lvl| pick_pseudobulk(&lvl[0], args.use_adjusted).ncols())
        .expect("levels is non-empty");
    let rna_pb = pick_pseudobulk(&finest[0], args.use_adjusted);
    let atac_pb = pick_pseudobulk(&finest[1], args.use_adjusted);
    let s = rna_pb.ncols();
    info!(
        "Pseudobulk: RNA {}x{}, ATAC {}x{} ({} refinement level(s), {} samples{})",
        rna_pb.nrows(),
        s,
        atac_pb.nrows(),
        atac_pb.ncols(),
        levels.len(),
        s,
        if args.use_adjusted {
            ", batch-adjusted"
        } else {
            ""
        }
    );
    if s < 50 {
        info!("warning: only {s} pseudobulk samples; correlations may be unstable");
    }

    /* 3. Coordinates */
    let peak_coords = parse_peak_coordinates(&peak_names);
    let gene_tss = if args.cis_window > 0 {
        if let Some(path) = &args.gene_coords {
            load_gene_coords_tsv(path, &gene_names)?
        } else if let Some(path) = &args.gff_file {
            load_gene_tss(path, &gene_names)?
        } else {
            anyhow::bail!("--cis-window > 0 requires either --gene-coords or --gff-file");
        }
    } else {
        anyhow::bail!("--cis-window must be > 0");
    };

    /* 4. ge-util workflow */
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
    run_from_pseudobulk(
        &PbMultiome {
            rna_pb,
            atac_pb,
            gene_tss: &gene_tss,
            peak_coords: &peak_coords,
            gene_names: &gene_names,
            peak_names: &peak_names,
        },
        &args.out,
        &params,
    )
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
