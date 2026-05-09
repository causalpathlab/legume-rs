//! Public entry point for `graph-embedding`. Callers translate their
//! own CLI args into a [`FitConfig`] and pass already-loaded
//! [`UnifiedData`] (so this crate stays free of file/path concerns).

use crate::coarsen::{build_cell_coarsenings, AxisCoarsenings, CellCoarseningArgs};
use crate::data::UnifiedData;
use crate::feature_network::FeatureNetworkSmoother;
use crate::loss::{build_per_batch_samplers, PerBatchSampler};
use crate::model::{BiasInit, JointEmbedModel, ModelArgs};
use crate::stop::setup_stop_handler;
use crate::training::{train, TrainingContext, TrainingParams};
use candle_util::candle_core::Device;
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans_alg::gene_weighting::compute_nb_fisher_weights;
use log::{info, warn};
use matrix_util::pair_graph::FeaturePairGraph;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Hyperparameter / configuration bundle for [`fit`]. Constructed by
/// each caller from its own CLI arguments — this crate doesn't import
/// `clap`.
pub struct FitConfig {
    pub embedding_dim: usize,
    pub num_coarsen_seeds: usize,
    pub super_cells: usize,
    pub sketch_dim: usize,
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub num_negatives: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub device: Device,
    pub feature_network: Option<FeatureNetworkConfig>,
    /// Optional caller-provided stop flag (so a single SIGINT handler
    /// can be shared with surrounding orchestration). When `None`,
    /// [`fit`] installs its own. Callers running `fit` more than once
    /// in the same process MUST pass `Some(...)` with a single shared
    /// flag — `ctrlc::set_handler` panics on the second registration.
    pub stop: Option<Arc<AtomicBool>>,
}

/// Optional SGC feature-network smoother configuration. The graph is
/// already loaded and aligned to the model's feature axis — caller
/// owns the file → graph translation.
pub struct FeatureNetworkConfig {
    pub graph: FeaturePairGraph,
    pub k_hops: usize,
    pub alpha: f32,
    pub refresh_epochs: usize,
}

/// CLI-flag-shaped helper that resolves an edge-list file against the
/// unified feature axis and packages it as a [`FeatureNetworkConfig`].
/// Both `senna gbe` and `pinto gbe` reach this with identical args, so
/// the resolution + zero-edge guard live here to keep error messages
/// consistent.
pub struct FeatureNetworkArgs<'a> {
    pub path: &'a str,
    pub feature_names: &'a [Box<str>],
    pub prefix_match: bool,
    pub delim: Option<char>,
    pub k_hops: usize,
    pub alpha: f32,
    pub refresh_epochs: usize,
}

pub fn load_feature_network(args: FeatureNetworkArgs) -> anyhow::Result<FeatureNetworkConfig> {
    info!("Loading feature network from {}...", args.path);
    let graph = FeaturePairGraph::from_edge_list(
        args.path,
        args.feature_names.to_vec(),
        args.prefix_match,
        args.delim,
    )?;
    if graph.num_edges() == 0 {
        anyhow::bail!(
            "Feature network has 0 matched edges — check name resolution \
             (--feature-network-delim / --feature-network-prefix-match)."
        );
    }
    Ok(FeatureNetworkConfig {
        graph,
        k_hops: args.k_hops,
        alpha: args.alpha,
        refresh_epochs: args.refresh_epochs,
    })
}

/// Trained model + its `VarMap`. The varmap is exposed so callers can
/// save checkpoints or re-run inference; current callers (`senna gbe`,
/// `pinto gbe`) only consume `model`, so it sits unused but kept alive.
pub struct FitOutput {
    pub model: JointEmbedModel,
    pub varmap: VarMap,
}

pub fn fit(unified: &UnifiedData, config: FitConfig) -> anyhow::Result<FitOutput> {
    let n_cells = unified.n_cells();
    let n_features = unified.n_features();

    info!(
        "Building cell coarsenings (K={} seeds, target ~{} super-cells)",
        config.num_coarsen_seeds, config.super_cells
    );

    let cell_axis: AxisCoarsenings = build_cell_coarsenings(CellCoarseningArgs {
        triplets: &unified.triplets,
        n_cells,
        n_features,
        target_blocks: config.super_cells,
        sketch_dim: config.sketch_dim,
        n_seeds: config.num_coarsen_seeds,
        base_seed: config.seed.wrapping_add(0xC347),
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
    let vs = VarBuilder::from_varmap(
        &varmap,
        candle_util::candle_core::DType::F32,
        &config.device,
    );
    let model = JointEmbedModel::new(
        ModelArgs {
            n_features,
            n_cells,
            embedding_dim: config.embedding_dim,
        },
        &BiasInit {
            b_feat: &zeros_feat,
            b_cell: &zeros_cell,
        },
        &varmap,
        vs,
        &config.device,
    )?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            ..Default::default()
        },
    )?;

    info!(
        "Building per-batch edge samplers ({} batch(es); negatives drawn within batch)...",
        unified.n_batches()
    );
    let alpha_neg = 0.75_f32;
    let batch_samplers_all = build_per_batch_samplers(
        &unified.triplets,
        &unified.batch_membership,
        unified.n_batches(),
        n_features,
        &feat_weights,
        alpha_neg,
    );
    let mut empty_batches: Vec<&str> = Vec::new();
    let batch_samplers: Vec<PerBatchSampler> = batch_samplers_all
        .into_iter()
        .enumerate()
        .filter_map(|(b, s)| {
            if s.is_none() {
                empty_batches.push(unified.batch_names[b].as_ref());
            }
            s
        })
        .collect();
    if !empty_batches.is_empty() {
        warn!(
            "Skipping {} batch(es) with no observed edges: {}",
            empty_batches.len(),
            empty_batches.join(", ")
        );
    }
    if batch_samplers.is_empty() {
        anyhow::bail!("no non-empty batches available for sampling");
    }

    let mut smoother = match config.feature_network {
        None => None,
        Some(FeatureNetworkConfig {
            graph,
            k_hops,
            alpha,
            refresh_epochs,
        }) => {
            if graph.num_edges() == 0 {
                anyhow::bail!(
                    "feature network has 0 matched edges — check name resolution at the caller"
                );
            }
            info!(
                "SGC smoothing: K={}, α={}, refresh={} epochs over {} edges",
                k_hops,
                alpha,
                refresh_epochs,
                graph.num_edges()
            );
            Some(FeatureNetworkSmoother::new(
                &graph,
                n_features,
                config.embedding_dim,
                alpha,
                k_hops,
                refresh_epochs,
            )?)
        }
    };

    let stop = config.stop.unwrap_or_else(setup_stop_handler);

    let train_ctx = TrainingContext {
        unified,
        cell_axis: &cell_axis,
        feat_weights: &feat_weights,
        batch_samplers: &batch_samplers,
        dev: &config.device,
        stop: &stop,
    };
    let train_params = TrainingParams {
        epochs: config.epochs,
        batches_per_epoch: config.batches_per_epoch,
        batch_size: config.batch_size,
        num_negatives: config.num_negatives,
        seed: config.seed,
    };
    train(
        &model,
        &mut opt,
        &train_ctx,
        &train_params,
        smoother.as_mut(),
    )?;

    Ok(FitOutput { model, varmap })
}
