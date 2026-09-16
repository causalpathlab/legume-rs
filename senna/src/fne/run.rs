//! The `senna fne` driver: read the edge files into a typed graph, train
//! the PBG table, write the artifacts and the manifest.

use super::args::FneArgs;
use super::graph::{file_stem, TypedGraphBuilder};
use super::output::{write_outputs, write_text_export};
use crate::run_manifest::{record_train_args, write_run_manifest, RunDescription, RunKind};
use auxiliary_data::gene_sets::{read_gaf, read_gmt, GafOpts};
use auxiliary_data::ontology::Ontology;
use graph_embedding_util::fne::{train, FneConfig};
use graph_embedding_util::stop::setup_stop_handler;
use log::info;
use matrix_util::common_io::mkdir_parent;

pub fn fit_fne(args: &FneArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    let any_input = !args.networks.is_empty()
        || !args.edges.is_empty()
        || !args.membership.is_empty()
        || args.gaf.is_some()
        || !args.gmt.is_empty()
        || !args.region_gene.is_empty();
    anyhow::ensure!(
        any_input,
        "fne: no input files; pass gene-gene pair files, --edges, --membership, --gaf, --gmt or --region-gene"
    );
    anyhow::ensure!(
        args.gaf.is_none() || args.obo.is_some(),
        "fne: --gaf needs --obo to propagate the annotations up the ontology"
    );

    let mut builder = TypedGraphBuilder::new(args.name_kind());
    for path in &args.networks {
        builder.add_pair_file(path)?;
    }
    for path in &args.edges {
        builder.add_typed_file(path)?;
    }
    for spec in &args.membership {
        let (ty, path) = spec
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--membership `{spec}`: expected `type=path`"))?;
        builder.add_membership_file(ty.trim(), path.trim())?;
    }
    let onto = match &args.obo {
        Some(obo) => {
            let onto = Ontology::load_obo(obo)?;
            info!("fne: loaded ontology: {} terms from {obo}", onto.len());
            Some(onto)
        }
        None => None,
    };
    if let Some(gaf) = &args.gaf {
        info!(
            "fne: reading GAF annotations from {gaf} (no_iea={})",
            args.no_iea
        );
        let sets = read_gaf(
            gaf,
            &GafOpts {
                no_iea: args.no_iea,
            },
        )?
        .into_gene_sets(onto.as_ref());
        builder.add_gene_sets(&sets, &file_stem(gaf), args.min_gene_set, args.max_gene_set);
    }
    for gmt in &args.gmt {
        info!("fne: reading GMT gene sets from {gmt}");
        let sets = read_gmt(gmt)?;
        builder.add_gene_sets(&sets, &file_stem(gmt), args.min_gene_set, args.max_gene_set);
    }
    if let Some(onto) = &onto {
        builder.add_ontology(onto);
    }
    for path in &args.region_gene {
        builder.add_region_file(path, args.region_window)?;
    }
    for spec in &args.relation_weight {
        builder.set_relation_weight(spec)?;
    }
    let mut graph = builder.finish()?;
    if let Some(path) = &args.export_text {
        write_text_export(&graph, path)?;
    }
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
    // The trainer shuffles the edge list in place; hand it over rather
    // than copying every edge. The two tables are small and stay with the
    // graph for the writers.
    let edges = std::mem::take(&mut graph.edges);
    let out = train(edges, graph.types.clone(), graph.relations.clone(), &cfg)?;
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
        .chain(args.membership.iter())
        .chain(args.gaf.iter())
        .chain(args.gmt.iter())
        .chain(args.region_gene.iter())
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
