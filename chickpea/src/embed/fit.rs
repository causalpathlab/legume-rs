//! `chickpea embed-graph` entry point: parse args, load data, build
//! coarsenings + bias initializations, construct model, dispatch to
//! [`crate::embed::training::train`], save outputs.

use crate::common::*;
use crate::embed::coarsen::{build_cell_coarsenings, CellCoarseningArgs};
use crate::embed::data::load_unified_data;
use crate::embed::eval::{save_outputs, OutputContext};
use crate::embed::loss::build_per_file_samplers;
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
        help = "Sparse count matrices (zarr/h5), comma-separated. Each \
                file contributes its rows to the unified feature axis; \
                cells unify by barcode across files."
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file"
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
        long_help = "Output prefix; produces {out}.e_feat.parquet, \
                     {out}.e_cell.parquet, {out}.b_feat.parquet, \
                     {out}.b_cell.parquet"
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

    let unified = load_unified_data(&args.data_files, args.batch_files.as_deref())?;

    let n_cells = unified.n_cells();
    let n_features = unified.n_features();

    info!(
        "Building cell coarsenings (K={} seeds, target ~{} super-cells)",
        args.num_coarsen_seeds, args.super_cells
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

    info!("Avg coarse blocks: cells {:.0}", cell_axis.avg_n_coarse());

    info!("Computing NB-Fisher weights per file...");
    let mut feat_weights: Vec<f32> = Vec::with_capacity(n_features);
    for (i, data) in unified.per_file_data.iter().enumerate() {
        let w = compute_nb_fisher_weights(data, None)?;
        info!(
            "  file {}: {} features, mean Fisher weight {:.3}",
            i,
            w.len(),
            w.iter().sum::<f32>() / w.len().max(1) as f32
        );
        feat_weights.extend(w);
    }

    let zeros_feat = vec![0f32; n_features];
    let zeros_cell = vec![0f32; n_cells];

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

    info!("Building per-file edge samplers...");
    let alpha_neg = 0.75_f32;
    let file_samplers = build_per_file_samplers(
        &unified.triplets,
        &unified.file_triplet_ranges,
        &unified.file_feature_ranges,
        &feat_weights,
        alpha_neg,
    );

    let train_ctx = TrainingContext {
        unified: &unified,
        cell_axis: &cell_axis,
        feat_weights: &feat_weights,
        file_samplers: &file_samplers,
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

    save_outputs(
        &model,
        &OutputContext {
            feature_names: &unified.feature_names,
            barcodes: &unified.barcodes,
        },
        &args.out,
    )?;

    write_manifest(&args.out, &args.data_files, args.batch_files.as_deref())?;

    info!(
        "Done — outputs at {}.{{e,b}}_{{feat,cell}}.parquet (+ {}.senna.json)",
        args.out, args.out
    );

    Ok(())
}

fn write_manifest(
    out_prefix: &str,
    data_files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
) -> anyhow::Result<()> {
    use crate::manifest::*;
    let manifest_path_str = manifest_path(out_prefix);
    let manifest_dir = std::path::Path::new(&manifest_path_str)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    let m = RunManifest {
        version: MANIFEST_VERSION,
        kind: "svd".to_string(),
        prefix: out_prefix.to_string(),
        data: RunData {
            input: data_files
                .iter()
                .map(|p| rel_to_manifest(&manifest_dir, p))
                .collect(),
            batch: batch_files
                .map(|b| {
                    b.iter()
                        .map(|p| rel_to_manifest(&manifest_dir, p))
                        .collect()
                })
                .unwrap_or_default(),
        },
        outputs: RunOutputs {
            latent: Some(rel_to_manifest(
                &manifest_dir,
                &format!("{out_prefix}.e_cell.parquet"),
            )),
            dictionary: Some(rel_to_manifest(
                &manifest_dir,
                &format!("{out_prefix}.e_feat.parquet"),
            )),
        },
        layout: serde_json::json!({}),
        cluster: RunCluster::default(),
        annotate: RunAnnotate::default(),
        pseudotime: serde_json::json!({}),
        defaults: serde_json::json!({"colour_by": "cluster"}),
    };
    save(&m, &manifest_path_str)?;
    info!("Wrote run manifest {manifest_path_str}");
    Ok(())
}
