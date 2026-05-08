//! `chickpea embed-graph` entry point: parse args, load data, build
//! coarsenings + bias initializations, construct model, dispatch to
//! [`crate::embed::training::train`], save outputs.

use crate::common::*;
use crate::embed::coarsen::{
    build_cell_coarsenings, build_unified_feature_coarsenings, CellCoarseningArgs,
    UnifiedFeatureCoarseningArgs,
};
use crate::embed::data::load_unified_data;
use crate::embed::eval::{save_outputs, OutputContext};
use crate::embed::data::Triplet;
use crate::embed::loss::build_negative_sampler;
use rand_distr::weighted::WeightedIndex;
use crate::embed::model::{BiasInit, JointEmbedModel, ModelArgs};
use crate::embed::training::{train, TrainingContext, TrainingParams};
use candle_util::candle_core::Device;
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans_alg::gene_weighting::compute_nb_fisher_weights;

#[derive(Args, Debug)]
pub struct EmbedGraphArgs {
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
    #[arg(long, default_value_t = 64, help = "Embedding dimension H")]
    embedding_dim: usize,

    /* Coarsening */
    #[arg(long, default_value_t = 8, help = "Number of coarsening seeds")]
    num_coarsen_seeds: usize,

    #[arg(
        long,
        default_value_t = 200,
        help = "Target super-cell blocks (cell axis)"
    )]
    super_cells: usize,

    #[arg(long, default_value_t = 200, help = "Target gene blocks")]
    gene_blocks: usize,

    #[arg(long, default_value_t = 2000, help = "Target peak blocks")]
    peak_blocks: usize,

    #[arg(long, default_value_t = 32, help = "Sketch dim for coarsening RP")]
    sketch_dim: usize,

    /* Training */
    #[arg(long, default_value_t = 200, help = "Training epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 100, help = "Batches per epoch")]
    batches_per_epoch: usize,

    #[arg(long, default_value_t = 1024, help = "Positive edges per batch")]
    batch_size: usize,

    #[arg(long, default_value_t = 16, help = "Negative samples per positive")]
    num_negatives: usize,

    #[arg(
        long,
        default_value_t = 0.01,
        help = "AdamW learning rate",
        alias = "lr"
    )]
    learning_rate: f64,

    #[arg(long, default_value_t = 1, help = "Random seed (base)")]
    seed: u64,

    /* Device */
    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    device_no: usize,

    /* Output */
    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix; produces {out}.e_feat.tsv.gz, \
                     {out}.e_cell.tsv.gz, {out}.b_feat.tsv, \
                     {out}.b_cell.tsv"
    )]
    out: Box<str>,
}

pub fn embed_graph(args: &EmbedGraphArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let dev = match args.device {
        ComputeDevice::Cpu => Device::Cpu,
        ComputeDevice::Cuda => Device::new_cuda(args.device_no)?,
        ComputeDevice::Metal => Device::new_metal(args.device_no)?,
    };

    /* 1. Load data: unified barcode index + unified (gene + peak) feature axis. */
    let unified = load_unified_data(
        &args.rna_files,
        &args.atac_files,
        args.batch_files.as_deref(),
    )?;

    let n_cells = unified.n_cells();
    let n_genes = unified.n_genes;
    let n_peaks = unified.n_peaks;
    let n_features = unified.n_features();

    /* 2. Multi-seed coarsenings. Cell axis from unified triplets;
     *    feature axis is modality-preserving — gene blocks and peak
     *    blocks live in the same FeatureCoarsening but never mix. */
    info!(
        "Building coarsenings (K={} seeds): cells→~{}, gene blocks→~{}, peak blocks→~{}",
        args.num_coarsen_seeds, args.super_cells, args.gene_blocks, args.peak_blocks
    );

    let cell_axis = build_cell_coarsenings(CellCoarseningArgs {
        triplets: &unified.triplets,
        n_cells,
        n_features,
        target_blocks: args.super_cells,
        sketch_dim: args.sketch_dim,
        n_seeds: args.num_coarsen_seeds,
        base_seed: args.seed.wrapping_add(0xC347),
    })?;

    let feat_axis = build_unified_feature_coarsenings(UnifiedFeatureCoarseningArgs {
        triplets: &unified.triplets,
        n_genes,
        n_peaks,
        n_cells,
        target_gene_blocks: args.gene_blocks,
        target_peak_blocks: args.peak_blocks,
        sketch_dim: args.sketch_dim,
        n_seeds: args.num_coarsen_seeds,
        base_seed: args.seed.wrapping_add(0x6E7E),
    })?;

    info!(
        "Avg coarse blocks: cells {:.0}, features {:.0}",
        cell_axis.avg_n_coarse(),
        feat_axis.avg_n_coarse(),
    );

    /* 3. Per-feature loss weights: NB-Fisher for genes, 1.0 for peaks.
     *    Vector is over the unified feature index (genes first). */
    info!("Computing RNA NB-Fisher weights...");
    let rna_weights = compute_nb_fisher_weights(&unified.rna_data, None)?;
    let mut feat_weights = Vec::with_capacity(n_features);
    feat_weights.extend_from_slice(&rna_weights);
    feat_weights.extend(std::iter::repeat_n(1.0_f32, n_peaks));

    /* 4. Bias init at zero — embeddings get fair gradient (bias learns marginal
     *    via AdamW). */
    let zeros_feat = vec![0f32; n_features];
    let zeros_cell = vec![0f32; n_cells];

    /* 5. Construct model. */
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_util::candle_core::DType::F32, &dev);
    let model = JointEmbedModel::new(
        ModelArgs {
            n_features,
            n_cells,
            embedding_dim: args.embedding_dim,
        },
        &BiasInit {
            b_feat: &zeros_feat,
            b_cell: &zeros_cell,
        },
        &varmap,
        vs,
        &dev,
    )?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        },
    )?;

    /* 6. Samplers.
     *    - Positive sampler: count × NB-Fisher weight over unified
     *      triplets so HVG positives dominate wall-clock learning.
     *    - Negative sampler: marginal^0.75 over unified feature blocks
     *      per coarsening seed (word2vec) so positive and negative
     *      distributions share the same marginal.
     */
    info!("Building edge samplers...");
    // Per-modality positive samplers — same triplet stream, but the
    // WeightedIndex zeros out the other modality so each batch can be
    // split 50/50 between RNA and ATAC. Avoids the large modality
    // (here, ATAC with 5× more edges) drowning out the small one.
    let rna_sampler =
        build_modality_edge_sampler(&unified.triplets, &feat_weights, true, n_genes as u32);
    let atac_sampler =
        build_modality_edge_sampler(&unified.triplets, &feat_weights, false, n_genes as u32);

    let alpha_neg = 0.75_f32;
    let neg_samplers: Vec<_> = feat_axis
        .coarsenings
        .iter()
        .map(|c| build_negative_sampler(&unified.triplets, c, alpha_neg))
        .collect();

    /* 7. Train. */
    let train_ctx = TrainingContext {
        unified: &unified,
        cell_axis: &cell_axis,
        feat_axis: &feat_axis,
        feat_weights: &feat_weights,
        rna_sampler: &rna_sampler,
        atac_sampler: &atac_sampler,
        neg_samplers: &neg_samplers,
        dev: &dev,
    };
    let train_params = TrainingParams {
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch_size: args.batch_size,
        num_negatives: args.num_negatives,
        seed: args.seed,
    };
    train(&model, &mut opt, &train_ctx, &train_params)?;

    /* 8. Save outputs. */
    save_outputs(
        &model,
        &OutputContext {
            feature_names: &unified.feature_names,
            feature_modality: &unified.feature_modality,
            barcodes: &unified.barcodes,
        },
        &args.out,
    )?;

    info!(
        "Done — outputs at {}.e_*.tsv.gz / {}.b_*.tsv",
        args.out, args.out
    );

    Ok(())
}

/// Build a count×Fisher edge sampler restricted to one modality's
/// triplets in the unified stream. Triplets in the other modality are
/// given near-zero weight so they're never sampled.
fn build_modality_edge_sampler(
    triplets: &[Triplet],
    fisher_weights: &[f32],
    rna_side: bool,
    n_genes: u32,
) -> WeightedIndex<f32> {
    let weights: Vec<f32> = triplets
        .iter()
        .map(|t| {
            let in_modality = if rna_side {
                t.feature < n_genes
            } else {
                t.feature >= n_genes
            };
            if in_modality {
                let w = fisher_weights[t.feature as usize];
                (t.count * w).max(1e-8)
            } else {
                1e-12
            }
        })
        .collect();
    WeightedIndex::new(weights).expect("non-empty triplet stream")
}
