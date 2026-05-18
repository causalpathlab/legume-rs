//! Orchestrator for `pinto lc-etm`.
//!
//! Pipeline:
//!   1. Load data + coordinates (or expression-only).
//!   2. Build cell-cell KNN graph (spatial if coords, else expression KNN).
//!   3. Estimate batch effects and write `.delta.parquet`.
//!   4. Build per-edge count matrix `y_e = x_i + x_j` `[E, G]`.
//!   5. Compute per-gene shortlist weights from total counts.
//!   6. Instantiate `IndexedEmbeddingEncoder` + `EmbeddedTopicDecoder`
//!      (shared ρ via ETM tying).
//!   7. Train via [`candle_util::vae::indexed_topic::train_mixed`]
//!      (single-level — no V-cycle in v1).
//!   8. Inference + writers via
//!      [`crate::link_community_etm::post::run_inference_and_write`].

pub use crate::link_community_etm::args::SrtLinkCommunityEtmArgs;
use crate::link_community::profiles::build_super_edges;
use crate::link_community_etm::data;
use crate::link_community_etm::post;
use crate::util::batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
use crate::util::cell_pairs::*;
use crate::util::graph_coarsen::*;
use crate::util::common::Mat;
use crate::util::input::*;
use candle_util::candle_core::{self, Device};
use candle_util::decoder::embedded_topic::EmbeddedTopicDecoder;
use candle_util::candle_nn::{VarBuilder, VarMap};
use candle_util::encoder::indexed::{
    IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs,
};
use candle_util::value_transform::ValueEmbeddingConfig;
use candle_util::vae;
use data_beans_alg::random_projection::RandProjOps;
use graph_embedding_util::stop::setup_stop_handler;
use log::info;
use matrix_util::common_io::mkdir_parent;

/// Main entry point for `pinto lc-etm`.
pub fn fit_srt_link_community_etm(args: &SrtLinkCommunityEtmArgs) -> anyhow::Result<()> {
    let c = &args.common;
    mkdir_parent(&c.out)?;

    anyhow::ensure!(args.n_communities > 0, "n_communities must be > 0");
    anyhow::ensure!(args.context_size > 0, "context_size must be > 0");
    anyhow::ensure!(args.embedding_dim > 0, "embedding_dim must be > 0");
    anyhow::ensure!(args.num_epochs > 0, "num_epochs must be > 0");
    anyhow::ensure!(args.batch_edges > 0, "batch_edges must be > 0");
    anyhow::ensure!(c.knn_spatial > 0, "knn_spatial must be > 0");

    //////////////////////////////////////////////////////
    // 1. Load data                                      //
    //////////////////////////////////////////////////////
    info!("Loading data files...");
    let has_coords = c.has_coordinates();

    let SRTData {
        data: mut data_vec,
        mut coordinates,
        mut coordinate_names,
        batches: mut batch_membership,
    } = if has_coords {
        read_data_with_coordinates(c.to_read_args_with_kind(auxiliary_data::feature_names::FeatureNameKind::Exact))?
    } else {
        info!("No coordinate files provided — using expression mode");
        read_data_without_coordinates(c.to_read_args_with_kind(auxiliary_data::feature_names::FeatureNameKind::Exact))?
    };

    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();
    anyhow::ensure!(
        c.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({})",
        c.knn_spatial,
        n_cells
    );

    //////////////////////////////////////////////////////
    // 2. Build KNN graph                                //
    //////////////////////////////////////////////////////
    let graph = if has_coords {
        info!("Building spatial KNN graph (k={})...", c.knn_spatial);
        build_spatial_graph(
            &coordinates,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?
    } else {
        info!(
            "Building expression KNN graph (k={}, proj_dim={})...",
            c.knn_spatial, c.proj_dim
        );
        let cell_proj_pre = data_vec.project_columns_with_batch_correction(
            c.proj_dim,
            c.block_size,
            None::<&[Box<str>]>,
        )?;
        let (g, embedding) = build_expression_graph(
            &cell_proj_pre.proj,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?;
        coordinates = embedding;
        coordinate_names = vec!["pc_1".into(), "pc_2".into()];
        g
    };

    if c.auto_batch && c.batch_files.is_none() {
        crate::util::input::auto_batch_from_components(&graph, &mut batch_membership);
    }

    //////////////////////////////////////////////////////
    // 3. Batch effects                                  //
    //////////////////////////////////////////////////////
    let batch_sort_dim = c.proj_dim.min(10);
    let _batch_db = estimate_and_write_batch_effects(
        &mut data_vec,
        &batch_membership,
        EstimateBatchArgs {
            proj_dim: c.proj_dim,
            sort_dim: batch_sort_dim,
            block_size: c.block_size,
            batch_knn: c.batch_knn,
            num_levels: c.num_levels,
        },
        &c.out,
    )?;

    let srt_cell_pairs = SrtCellPairs::with_graph(&data_vec, &coordinates, graph);
    srt_cell_pairs.to_parquet(
        &(c.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    let edges = srt_cell_pairs.graph.edges.clone();
    let n_edges = edges.len();
    info!(
        "{} cells, {} edges, {} genes",
        n_cells, n_edges, n_genes
    );

    //////////////////////////////////////////////////////
    // 4. Multi-level cell coarsening (V-cycle pyramid)  //
    //////////////////////////////////////////////////////
    info!(
        "Graph coarsening ({} levels, {} coarse clusters)...",
        c.num_levels, c.n_pseudobulk
    );

    let batch_arg: Option<&[Box<str>]> = if _batch_db.is_some() {
        Some(&batch_membership)
    } else {
        None
    };
    let cell_proj =
        data_vec.project_columns_with_batch_correction(c.proj_dim, c.block_size, batch_arg)?;

    let ml = graph_coarsen_multilevel(
        &srt_cell_pairs.graph,
        &mut cell_proj.proj.clone(),
        &srt_cell_pairs.pairs,
        CoarsenConfig {
            n_clusters: c.n_pseudobulk,
            num_levels: c.num_levels,
            refine_iterations: c.refine_iterations,
            seeding: has_coords.then(|| SeedingParams {
                coordinates: &coordinates,
                batch_membership: Some(&batch_membership),
            }),
            modularity_veto: None,
            dc_poisson: None,
        },
    );

    //////////////////////////////////////////////////////
    // 5. Per-level super-edge profiles + fine profile   //
    //////////////////////////////////////////////////////
    info!("Reading cell counts and assembling edge profiles...");
    let cell_counts_csc = data::read_cells_csc(&data_vec)?;
    let gene_totals = data::gene_total_counts(&cell_counts_csc);

    // Per-level training profiles. Cascade levels (coarse → finest cascade
    // level) feed the V-cycle's coarse-to-fine warm-up, and the **fine
    // edge profile** (one row per actual cell-cell edge `y_e = x_i + x_j`)
    // is appended as the final level so training and inference both
    // operate at the same resolution.
    let mut level_profiles: Vec<Mat> = Vec::with_capacity(ml.all_cell_labels.len() + 1);
    for (l, cell_labels) in ml.all_cell_labels.iter().enumerate() {
        let (super_edges, _f2s) = build_super_edges(&edges, cell_labels);
        let super_counts = data::build_super_cell_counts(&cell_counts_csc, cell_labels);
        let profile = data::build_super_edge_profiles(&super_counts, &super_edges);
        info!(
            "  L{}: {} super-cells → {} super-edges, profile [{}×{}] ({:.0} MB)",
            l,
            super_counts.len(),
            super_edges.len(),
            profile.nrows(),
            profile.ncols(),
            (profile.nrows() * profile.ncols() * 4) as f64 / 1_048_576.0
        );
        level_profiles.push(profile);
        drop(super_counts);
    }

    let fine_profile = data::build_edge_profiles(&cell_counts_csc, &edges);
    info!(
        "  L{} (fine): {} edges, profile [{}×{}] ({:.0} MB)",
        level_profiles.len(),
        edges.len(),
        fine_profile.nrows(),
        fine_profile.ncols(),
        (fine_profile.nrows() * fine_profile.ncols() * 4) as f64 / 1_048_576.0
    );
    level_profiles.push(fine_profile);

    //////////////////////////////////////////////////////
    // 6. Shortlist weights from gene totals             //
    //////////////////////////////////////////////////////
    let shortlist_weights = data::shortlist_weights(&gene_totals, args.min_gene_count);
    let n_kept = shortlist_weights.iter().filter(|&&w| w > 0.0).count();
    info!(
        "Kept {}/{} genes for top-K shortlist (min_count={:.0})",
        n_kept, n_genes, args.min_gene_count
    );

    // For v1 we don't use the Anscombe baseline (`feature_mean`) or the
    // NB-Fisher loss weighting (`feature_fisher_weights`) — they require
    // pseudobulk fit, which we don't have here. Pass uniform-ish neutral
    // values: zero baseline, all-ones Fisher weights.
    let feature_mean: Vec<f32> = vec![0.0; n_genes];
    let feature_fisher_weights: Vec<f32> = vec![1.0; n_genes];

    //////////////////////////////////////////////////////
    // 7. Encoder + per-level decoders (ETM ρ tying)     //
    //////////////////////////////////////////////////////
    let dev = Device::Cpu;
    let parameters = VarMap::new();
    let dtype = candle_core::DType::F32;
    let param_builder = VarBuilder::from_varmap(&parameters, dtype, &dev);

    let value_embedding = ValueEmbeddingConfig { n_value_bins: 8 };
    let encoder_layers: Vec<usize> = vec![args.embedding_dim];

    let encoder = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: n_genes,
            n_topics: args.n_communities,
            embedding_dim: args.embedding_dim,
            layers: &encoder_layers,
            value_embedding,
            use_gcn: false,
        },
        &parameters,
        param_builder.pp("enc"),
    )?;

    // One decoder per coarsening level; all share ρ via the Var clone.
    // Each level holds its own α (under separate `dec_{l}` scopes).
    let shared_rho = encoder.feature_embeddings().clone();
    let num_levels = level_profiles.len();
    let decoders: Vec<EmbeddedTopicDecoder> = (0..num_levels)
        .map(|l| {
            EmbeddedTopicDecoder::new(
                args.n_communities,
                shared_rho.clone(),
                param_builder.pp(format!("dec_{l}")),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    info!(
        "lc-etm: {} genes -> indexed encoder (emb={}, ctx={}) -> {} decoders (K={})",
        n_genes, args.embedding_dim, args.context_size, num_levels, args.n_communities
    );

    //////////////////////////////////////////////////////
    // 8. V-cycle training via candle-util               //
    //////////////////////////////////////////////////////
    let stop = setup_stop_handler();
    let train_cfg = vae::indexed_topic::IndexedTrainConfig {
        parameters: &parameters,
        dev: &dev,
        epochs: args.num_epochs,
        minibatch_size: args.batch_edges,
        learning_rate: args.lr,
        topic_smoothing: args.topic_smoothing,
        enc_context_size: args.context_size,
        dec_context_size: args.context_size,
        stop: &stop,
        shortlist_weights: &shortlist_weights,
        feature_mean: &feature_mean,
        feature_fisher_weights: &feature_fisher_weights,
        grad_clip: args.grad_clip,
        feature_graph: None,
        feature_embedding_l2: args.feature_embedding_l2,
        weight_decay: args.weight_decay,
        frozen_feature_var: None,
    };

    // Per-level (input == target) borrowed view — no clones of the
    // multi-GB profile matrices.
    let level_data: Vec<vae::indexed_topic::LevelData> = level_profiles
        .iter()
        .map(|m| (m, None, m))
        .collect();
    let scores =
        vae::indexed_topic::train_mixed(&level_data, &encoder, &decoders, &train_cfg, None)?;

    write_score_trace(&scores, &c.out)?;

    //////////////////////////////////////////////////////
    // 8. Inference + writers                            //
    //////////////////////////////////////////////////////
    let cell_names = data_vec.column_names()?;
    let gene_names = data_vec.row_names()?;
    // Inference at the fine level: the final entry of `level_profiles`
    // is the per-cell-pair profile, and `decoders[num_levels - 1]` is the
    // α head trained against it (so β = softmax(α · ρᵀ) is the
    // fine-resolution dictionary).
    let fine_profile = level_profiles.last().expect("level_profiles non-empty");
    post::run_inference_and_write(post::InferenceArgs {
        encoder: &encoder,
        decoder: &decoders[num_levels - 1],
        edge_profiles: fine_profile,
        edges: &edges,
        n_cells,
        n_communities: args.n_communities,
        cell_names: &cell_names,
        gene_names: &gene_names,
        shortlist_weights: &shortlist_weights,
        context_size: args.context_size,
        dev: &dev,
        out_prefix: &c.out,
    })?;

    Ok(())
}

/// Write `(epoch, llik, kl)` training trace to `.scores.parquet`.
fn write_score_trace(scores: &vae::TrainScores, out_prefix: &str) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    use parquet::basic::Type as ParquetType;

    let n_rows = scores.llik.len();
    let col_names: Vec<Box<str>> = vec!["llik".into(), "kl".into()];
    let col_types = vec![ParquetType::FLOAT, ParquetType::FLOAT];

    let writer = ParquetWriter::new(
        &format!("{}.scores.parquet", out_prefix),
        (n_rows, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("epoch"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_numeric_column(&mut row_group, &scores.llik)?;
    parquet_add_numeric_column(&mut row_group, &scores.kl)?;

    row_group.close()?;
    writer.close()?;
    Ok(())
}
