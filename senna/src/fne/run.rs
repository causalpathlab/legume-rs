//! The `senna fne` driver: read the edge files into a typed graph, train
//! the PBG table, write the artifacts and the manifest.

use super::args::FneArgs;
use super::graph::TypedGraphBuilder;
use super::output::write_outputs;
use crate::run_manifest::{record_train_args, write_run_manifest, RunDescription, RunKind};
use graph_embedding_util::fne::{train, FneConfig};
use graph_embedding_util::stop::setup_stop_handler;
use log::info;
use matrix_util::common_io::mkdir_parent;

pub fn fit_fne(args: &FneArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    anyhow::ensure!(
        !args.networks.is_empty() || !args.edges.is_empty(),
        "fne: no input files; pass gene-gene pair files and/or --edges typed files"
    );

    let mut builder = TypedGraphBuilder::new(args.name_kind());
    for path in &args.networks {
        builder.add_pair_file(path)?;
    }
    for path in &args.edges {
        builder.add_typed_file(path)?;
    }
    for spec in &args.relation_weight {
        builder.set_relation_weight(spec)?;
    }
    let graph = builder.finish()?;
    info!(
        "fne: {} nodes in {} types, {} edges in {} relations",
        graph.node_names.len(),
        graph.types.len(),
        graph.edges.len(),
        graph.relations.len()
    );

    let stop = setup_stop_handler();
    let cfg = FneConfig {
        dim: args.embedding_dim,
        epochs: args.epochs,
        lr: args.learning_rate,
        batch_size: args.batch_size,
        num_batch_negs: args.num_batch_negs,
        num_uniform_negs: args.num_uniform_negs,
        wd: args.weight_decay,
        wd_interval: args.wd_interval,
        eval_fraction: args.eval_fraction,
        eval_min_per_relation: args.eval_min_per_relation,
        seed: args.seed,
        device: args.device.to_device(args.device_no)?,
    };
    let out = train(
        graph.edges.clone(),
        graph.types.clone(),
        graph.relations.clone(),
        &cfg,
    )?;
    if args.weight_decay.is_none() && out.wd > 1.0 {
        log::warn!(
            "fne: the automatic weight decay came out at {:.3}; it is SIMBA's calibration, \
             which scales inversely with the edge count and suits graphs of millions of \
             edges. On a graph this size pass --weight-decay explicitly (0 disables it).",
            out.wd
        );
    }
    write_outputs(&out, &graph, &args.out)?;

    let input: Vec<String> = args
        .networks
        .iter()
        .chain(args.edges.iter())
        .map(ToString::to_string)
        .collect();
    write_run_manifest(&RunDescription {
        train_args: Some(record_train_args(args)?),
        kind: RunKind::Fne,
        prefix: &args.out,
        data_input: &input,
        data_multiome: None,
        data_batch: &[],
        data_input_null: &[],
        dictionary_suffix: None,
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_reference_suffix: None,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_loading_suffix: None,
        module_membership_suffix: None,
        module_dictionary_suffix: None,
        softmax_dictionary_suffix: None,
        cell_embedding_suffix: None,
        cell_encoder_suffix: None,
        feature_contrast_suffix: None,
        feature_contrast_bias_suffix: None,
        track_encoder_suffixes: vec![],
        default_colour_by: "cluster",
        has_latent: false,
        has_cell_to_pb: false,
        has_pb_tree: false,
    })?;

    if stop.load(std::sync::atomic::Ordering::SeqCst) {
        info!(
            "Stopped early — outputs reflect partial training ({} of {} epochs)",
            out.epochs.len(),
            args.epochs
        );
    } else {
        info!(
            "Done — outputs at {}.{{feature_embedding,feature_types,relations,log_likelihood}}.parquet",
            args.out
        );
    }
    Ok(())
}
