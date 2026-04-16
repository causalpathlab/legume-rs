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

    /* Model */
    #[arg(long, default_value_t = 10, help = "Number of latent topics (K)")]
    n_topics: usize,

    #[arg(
        long,
        default_value_t = 128,
        help = "Per-feature embedding dimension for indexed encoder"
    )]
    embedding_dim: usize,

    #[arg(
        long,
        default_value_t = 512,
        help = "Top-K genes per sample for encoder context"
    )]
    context_size: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "SuSiE SER components per gene (max causal peaks)"
    )]
    n_ser_components: usize,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "SuSiE prior variance on effect sizes"
    )]
    prior_var: f64,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Per-gene gate prior (0.5=uninformative, lower=sparser)"
    )]
    gate_prior: f64,

    /* Cis-window */
    #[arg(long, default_value_t = 50, help = "Max cis-candidate peaks per gene")]
    max_cis: usize,

    #[arg(
        long,
        default_value_t = 500000,
        help = "Cis-window in bp around TSS. 0 = correlation-based"
    )]
    cis_window: i64,

    #[arg(
        long,
        help = "Gene coordinates TSV (gene, chr, tss). From sim-link gene_coords.tsv.gz"
    )]
    gene_coords: Option<Box<str>>,

    #[arg(
        long,
        help = "GFF/GTF annotation for gene TSS. Alternative to --gene-coords"
    )]
    gff_file: Option<Box<str>>,

    /* Collapsing */
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
        default_value_t = 3,
        help = "Multi-level coarsening depth (coarse → fine)"
    )]
    num_levels: usize,

    /* Training */
    #[arg(long, default_value_t = 100, help = "Training epochs")]
    epochs: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "AdamW learning rate",
        alias = "lr"
    )]
    learning_rate: f64,

    #[arg(
        long,
        default_value_t = 256,
        help = "Minibatch size (clamped to sample count)"
    )]
    minibatch_size: usize,

    #[arg(
        long,
        short = 'j',
        default_value_t = 5,
        help = "Epochs between RNA posterior resampling"
    )]
    jitter_interval: usize,

    #[arg(
        long,
        default_value_t = 1e-4,
        help = "Topic smoothing: z = (1-α)z + α/K. 0 = off"
    )]
    topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 0,
        help = "Max pseudobulk samples per level. 0 = all"
    )]
    row_budget: usize,

    /* Device */
    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    device_no: usize,

    /* Feature coarsening */
    #[arg(
        long,
        default_value_t = 0,
        help = "Max coarsened gene modules at finest level. 0 = off"
    )]
    max_coarse_genes: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Max coarsened peak modules at finest level. 0 = off"
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
                oversample: false,
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
        gate_prior: args.gate_prior,
        row_budget: args.row_budget,
        sort_dim: args.sort_dim,
        embedding_dim: args.embedding_dim,
        context_size: args.context_size,
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
