use crate::chickpea_input::load_paired_data;
use crate::cis_mask::*;
use crate::coarsening::{log_spaced_coarsenings, log_spaced_genomic_coarsenings};
use crate::common::*;
use crate::topic::eval::{save_outputs, EvalContext};
use crate::topic::training::{TrainingContext, TrainingParams};
use candle_util::candle_core::Device;
use data_beans_alg::collapse_data::MultilevelParams;

#[derive(Args, Debug)]
pub struct FitTopicArgs {
    /* Input files */
    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "RNA count matrix files",
        long_help = "Comma-separated paths to RNA count matrices (sparse zarr/h5).\n\
                     Multiple files are merged on shared row names (genes).\n\
                     Example: --rna-files sample1.rna.zarr,sample2.rna.zarr"
    )]
    rna_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "ATAC count matrix files",
        long_help = "Comma-separated paths to ATAC count matrices (sparse zarr/h5).\n\
                     Multiple files are merged on shared row names (peaks).\n\
                     Cell barcodes must match the RNA files exactly."
    )]
    atac_files: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Batch membership files",
        long_help = "One file per data file, in RNA-then-ATAC order.\n\
                     Each file has one batch label per cell (one per line).\n\
                     If omitted, each data file is treated as its own batch."
    )]
    batch_files: Option<Vec<Box<str>>>,

    /* Model */
    #[arg(
        long,
        default_value_t = 10,
        help = "Number of latent topics",
        long_help = "Number of latent topics K, shared between RNA and ATAC.\n\
                     Controls the rank of the topic proportion matrix theta[K,N]\n\
                     and the ATAC dictionary beta[P,K]."
    )]
    n_topics: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "SER components per gene",
        long_help = "Number of Single Effect Regression (SER) components\n\
                     in the SuSiE model for gene-peak linkage.\n\
                     Each component selects one peak per gene via softmax.\n\
                     Sets the maximum number of causal peaks per gene."
    )]
    n_ser_components: usize,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "SuSiE prior variance on effect sizes",
        long_help = "Prior variance σ₀² for the Gaussian prior N(0, σ₀²)\n\
                     on SuSiE effect sizes. Larger values allow stronger\n\
                     peak-gene linkage effects."
    )]
    prior_var: f64,

    #[arg(
        long,
        default_value_t = 50,
        help = "Max cis-candidates per gene",
        long_help = "Maximum number of cis-candidate peaks per gene.\n\
                     When using distance-based filtering (--cis-window),\n\
                     peaks within the window are ranked by distance and\n\
                     the closest max_cis are selected.\n\
                     When using correlation-based filtering (--cis-window 0),\n\
                     peaks are ranked by absolute Pearson correlation."
    )]
    max_cis: usize,

    #[arg(
        long,
        default_value_t = 500000,
        help = "Cis-window size in bp",
        long_help = "Genomic distance window (in base pairs) around each gene's\n\
                     TSS for selecting candidate peaks.\n\
                     Only peaks on the same chromosome within ±cis_window bp\n\
                     of the gene TSS are considered.\n\
                     Set to 0 to fall back to correlation-based selection."
    )]
    cis_window: i64,

    #[arg(
        long,
        help = "Gene coordinates file (TSV: gene, chr, tss)",
        long_help = "Path to a TSV file with gene coordinates.\n\
                     Expected columns: gene, chr, tss (tab-separated, with header).\n\
                     Gene names must match the RNA matrix row names.\n\
                     Produced by sim-link as {out}.gene_coords.tsv.gz.\n\
                     Mutually exclusive with --gff-file."
    )]
    gene_coords: Option<Box<str>>,

    #[arg(
        long,
        help = "GFF/GTF file for gene TSS positions",
        long_help = "Path to a GFF3 or GTF annotation file.\n\
                     Gene TSS positions are extracted and matched to\n\
                     RNA matrix row names by gene_name or gene_id.\n\
                     Mutually exclusive with --gene-coords."
    )]
    gff_file: Option<Box<str>>,

    /* Collapsing */
    #[arg(
        long,
        default_value_t = 64,
        help = "Projection dimension",
        long_help = "Dimension of the shared random projection for\n\
                     cell grouping across both modalities.\n\
                     Higher values preserve more cell-cell distance structure."
    )]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 14,
        help = "Sort dimension",
        long_help = "Binary partitioning dimension for pseudobulk grouping.\n\
                     Produces up to 2^sort_dim super-cell groups.\n\
                     Higher values give more samples but noisier estimates."
    )]
    sort_dim: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Coarsening levels",
        long_help = "Number of hierarchical coarsening levels.\n\
                     Level 0 is coarsest, last level is finest.\n\
                     Training uses the finest level."
    )]
    num_levels: usize,

    /* Training */
    #[arg(
        long,
        default_value_t = 100,
        help = "Training epochs",
        long_help = "Total training epochs."
    )]
    epochs: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "Learning rate",
        long_help = "AdamW learning rate for all parameters.",
        alias = "lr"
    )]
    learning_rate: f64,

    #[arg(
        long,
        default_value_t = 256,
        help = "Minibatch size",
        long_help = "Number of pseudobulk samples per minibatch.\n\
                     Clamped to total sample count if larger."
    )]
    minibatch_size: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Data jitter interval",
        long_help = "Data jitter interval.\n\
                     Controls how often pseudobulk data is resampled from\n\
                     the posterior during training. Samples are reused for\n\
                     this many epochs before refreshing."
    )]
    jitter_interval: usize,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "Topic smoothing alpha",
        long_help = "Mix encoder topic proportions with uniform to prevent\n\
                     dead topics: z_smooth = (1-α)*z + α/K.\n\
                     Set to 0 to disable."
    )]
    topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 0,
        help = "Per-level row budget (0 = no subsampling)",
        long_help = "Maximum pseudobulk samples per level per jitter interval.\n\
                     Each level is capped at this many samples.\n\
                     Set to 0 to use all samples (default)."
    )]
    row_budget: usize,

    /* Device */
    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda)")]
    device_no: usize,

    /* Feature coarsening */
    #[arg(
        long,
        default_value_t = 0,
        help = "Max coarsened gene features (0 = disabled)",
        long_help = "Maximum number of coarsened gene modules at the finest level.\n\
                     Coarser levels use log-spaced smaller targets.\n\
                     Gene features are grouped by co-expression in pseudobulk.\n\
                     Set to 0 to disable feature coarsening (full resolution)."
    )]
    max_coarse_genes: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Max coarsened peak features (0 = disabled)",
        long_help = "Maximum number of coarsened peak modules at the finest level.\n\
                     Coarser levels use log-spaced smaller targets.\n\
                     Peak features are grouped by genomic proximity.\n\
                     Set to 0 to disable feature coarsening (full resolution)."
    )]
    max_coarse_peaks: usize,

    /* Output */
    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix for all result files.\n\
                     Produces: {out}.atac_dict.parquet, {out}.rna_dict.parquet,\n\
                     {out}.results.bed.gz, {out}.log_beta.parquet,\n\
                     {out}.prop.parquet"
    )]
    out: Box<str>,
}

pub fn fit_topic_model(args: &FitTopicArgs) -> anyhow::Result<()> {
    /* 1. Load paired RNA + ATAC data */
    let mut paired = load_paired_data(
        &args.rna_files,
        &args.atac_files,
        args.batch_files.as_deref(),
    )?;

    let ncells = paired.data_stack.num_columns()?;
    let block_size = Some(ncells.min(4096));

    /* 2. Shared random projection */
    info!(
        "Random projection (dim={}, {} cells)...",
        args.proj_dim, ncells
    );
    let proj = paired.data_stack.project_columns_with_batch_correction(
        args.proj_dim,
        block_size,
        Some(&paired.batch_membership),
    )?;

    /* 3. Multi-level collapsing */
    let collapsed: Vec<Vec<_>> = paired
        .data_stack
        .collapse_columns_multilevel_vec(
            &proj.proj,
            &paired.batch_membership,
            &MultilevelParams {
                knn_super_cells: DEFAULT_KNN,
                num_levels: args.num_levels,
                sort_dim: args.sort_dim,
                num_opt_iter: DEFAULT_OPT_ITER,
            },
        )?
        .into_iter()
        .rev()
        .collect();

    for (level, ld) in collapsed.iter().enumerate() {
        info!(
            "Level {}: RNA {}x{}, ATAC {}x{}",
            level,
            ld[0].mu_observed.nrows(),
            ld[0].mu_observed.ncols(),
            ld[1].mu_observed.nrows(),
            ld[1].mu_observed.ncols(),
        );
    }

    let n_genes = collapsed[0][0].mu_observed.nrows();
    let n_peaks = collapsed[0][1].mu_observed.nrows();

    /* 4. Build cis-mask */
    let dev = match args.device {
        ComputeDevice::Cpu => Device::Cpu,
        ComputeDevice::Cuda => Device::new_cuda(args.device_no)?,
        ComputeDevice::Metal => Device::new_metal(args.device_no)?,
    };

    let gene_names = paired.data_stack.stack[0].row_names()?;
    let peak_names = paired.data_stack.stack[1].row_names()?;
    let peak_coords = genomic_data::coordinates::parse_peak_coordinates(&peak_names);

    let (cis_indices, cis_mask) = if args.cis_window > 0 {
        let gene_tss = if let Some(ref coords_path) = args.gene_coords {
            load_gene_coords_tsv(coords_path, &gene_names)?
        } else if let Some(ref gff_path) = args.gff_file {
            load_gene_tss(gff_path, &gene_names)?
        } else {
            anyhow::bail!(
                "--cis-window > 0 requires either --gene-coords or --gff-file \
                 to provide gene TSS positions"
            );
        };
        build_cis_mask_by_distance(&peak_coords, &gene_tss, args.cis_window, args.max_cis, &dev)?
    } else {
        let rna_mat = collapsed.last().unwrap()[0].mu_observed.posterior_mean();
        let atac_mat = collapsed.last().unwrap()[1].mu_observed.posterior_mean();
        build_cis_mask_by_correlation(rna_mat, atac_mat, args.max_cis, &dev)?
    };

    /* 5. Feature coarsening */
    let num_levels = collapsed.len();
    let c_max = cis_indices.dim(1)?;
    let flat_cis_indices = cis_indices.flatten_all()?;

    let rna_coarsenings = log_spaced_coarsenings(
        collapsed.last().unwrap()[0].mu_observed.posterior_mean(),
        num_levels,
        args.max_coarse_genes,
    )?;
    let atac_coarsenings =
        log_spaced_genomic_coarsenings(&peak_coords, num_levels, args.max_coarse_peaks);

    /* 6. Train */
    let ctx = TrainingContext {
        collapsed: &collapsed,
        rna_coarsenings: &rna_coarsenings,
        atac_coarsenings: &atac_coarsenings,
        cis_mask: &cis_mask,
        flat_cis_indices: &flat_cis_indices,
        n_genes,
        n_peaks,
        c_max,
        dev: &dev,
    };

    let params = TrainingParams {
        n_topics: args.n_topics,
        n_ser_components: args.n_ser_components,
        prior_var: args.prior_var,
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        minibatch_size: args.minibatch_size,
        jitter_interval: args.jitter_interval,
        topic_smoothing: args.topic_smoothing,
        row_budget: args.row_budget,
        sort_dim: args.sort_dim,
    };

    let model = crate::topic::training::train(&ctx, &params)?;

    /* 7. Save outputs */
    let eval_ctx = EvalContext {
        rna_coarsenings: &rna_coarsenings,
        atac_coarsenings: &atac_coarsenings,
        cis_indices: &cis_indices,
        flat_cis_indices: &flat_cis_indices,
        gene_names: &gene_names,
        peak_names: &peak_names,
        peak_coords: &peak_coords,
        data_stack: &paired.data_stack,
        c_max,
        dev: &dev,
    };
    save_outputs(&model, &eval_ctx, &args.out)?;

    Ok(())
}
