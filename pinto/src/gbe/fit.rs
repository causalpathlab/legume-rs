//! `pinto gbe` — graph-based embedding for spatial transcriptomics.
//!
//! Loads SRT data with coordinates, builds a spatial KNN, persists it
//! to `{out}.spatial_knn.parquet` for downstream consumers, and (when
//! `--cell-cell-lambda > 0`, the default) feeds the same edge list
//! into `graph-embedding-util`'s cell-cell NCE loss term so spatial
//! coherence shows up directly in the cell embedding.

use crate::gbe::args::SrtGbeArgs;
use crate::util::cell_pairs::{build_spatial_graph, SrtCellPairs, SrtCellPairsArgs};
use crate::util::common::*;
use crate::util::input::{read_data_with_coordinates, read_data_without_coordinates, SRTData};
use graph_embedding_util as ge;
use matrix_util::common_io::mkdir_parent;

pub fn fit_srt_gbe(args: &SrtGbeArgs) -> anyhow::Result<()> {
    let c = &args.common;

    mkdir_parent(&c.out)?;

    let dev = args.device.to_device(args.device_no)?;

    info!("Loading data files...");
    let has_coords = c.has_coordinates();

    let mut read_args = c.to_read_args();
    read_args.feature_kind = if args.feature_name_exact {
        ge::FeatureNameKind::Exact
    } else {
        ge::FeatureNameKind::Gene {
            delim: args.feature_name_delim,
        }
    };

    let SRTData {
        data: data_vec,
        coordinates,
        coordinate_names,
        batches: batch_membership,
    } = if has_coords {
        read_data_with_coordinates(read_args)?
    } else {
        info!("No coordinate files provided — using expression mode");
        read_data_without_coordinates(read_args)?
    };

    let n_cells = data_vec.num_columns();
    anyhow::ensure!(c.knn_spatial > 0, "knn_spatial must be > 0");
    anyhow::ensure!(
        c.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({})",
        c.knn_spatial,
        n_cells
    );
    anyhow::ensure!(
        args.cell_cell_lambda >= 0.0,
        "--cell-cell-lambda must be >= 0 (got {})",
        args.cell_cell_lambda
    );
    if args.cell_cell_lambda > 0.0 && !has_coords {
        anyhow::bail!(
            "--cell-cell-lambda > 0 requires --coord (cell-cell positives come from \
             the spatial KNN). Pass --cell-cell-lambda 0 to disable the term."
        );
    }

    let mut cell_cell_edges: Option<Vec<(u32, u32)>> = None;

    if has_coords {
        info!(
            "Building spatial KNN graph (k={}) and writing side-output...",
            c.knn_spatial
        );
        let graph = build_spatial_graph(
            &coordinates,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?;
        if args.cell_cell_lambda > 0.0 {
            cell_cell_edges = Some(
                graph
                    .edges
                    .iter()
                    .map(|&(i, j)| (i as u32, j as u32))
                    .collect(),
            );
        }
        let cell_pairs = SrtCellPairs::with_graph(&data_vec, &coordinates, graph);
        // Named `coord_pairs.parquet` to match what `pinto plot` /
        // `pinto lc` expect — same schema, same file, single source of
        // truth across pinto subcommands.
        cell_pairs.to_parquet(
            &format!("{}.coord_pairs.parquet", c.out),
            Some(coordinate_names.clone()),
        )?;
    } else {
        info!("Skipping spatial KNN side-output (no coordinates provided).");
    }

    info!("Constructing unified bipartite stream...");
    let mut unified = ge::UnifiedData::from_sparse_io(data_vec, &batch_membership)?;

    let hvg_enabled = args.hvg.n_hvg > 0 || args.hvg.feature_list_file.is_some();
    let hvg_weights: Option<Vec<f32>> = if hvg_enabled {
        let hvg = data_beans_alg::hvg::select_hvg_streaming(
            &unified.per_file_data[0],
            (args.hvg.n_hvg > 0).then_some(args.hvg.n_hvg),
            args.hvg.feature_list_file.as_deref(),
            c.block_size,
        )?;
        Some(hvg.row_weights(unified.n_features()))
    } else {
        None
    };

    let feature_network = args
        .feature_network
        .as_deref()
        .map(|path| {
            ge::load_feature_network(ge::FeatureNetworkArgs {
                path,
                feature_names: &unified.feature_names,
                prefix_match: args.feature_network_prefix_match,
                delim: args.feature_network_delim,
                k_hops: args.feature_network_k,
                alpha: args.feature_network_alpha,
                refresh_epochs: args.feature_network_refresh,
            })
        })
        .transpose()?;

    let cell_cell = cell_cell_edges.map(|edges| ge::CellCellConfig {
        edges,
        lambda: args.cell_cell_lambda,
        n_negatives: args.cell_cell_negatives,
        pb_levels: parse_pb_levels(&args.cell_cell_pb_levels),
        lambda_per_level: args.cell_cell_lambda_per_level.clone(),
    });

    let refine = if args.no_refine {
        None
    } else {
        Some(ge::RefineParams {
            num_gibbs: args.refine_gibbs,
            num_greedy: args.refine_greedy,
            feature_weighting: args.refine_weighting.into(),
            seed: args.refine_seed,
            ..ge::RefineParams::default()
        })
    };

    let config = ge::FitConfig {
        embedding_dim: args.embedding_dim,
        num_coarsen_seeds: args.num_coarsen_seeds,
        max_features: args.max_features,
        hvg_weights,
        composite_mode: args.composite_mode.into(),
        refine,
        super_cells: args.super_cells,
        sketch_dim: args.sketch_dim,
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch_size: args.batch_size,
        num_negatives: args.num_negatives,
        learning_rate: args.learning_rate,
        seed: c.seed,
        device: dev,
        block_size: c.block_size,
        fisher_weights_cache: if args.no_fisher_cache {
            None
        } else {
            Some(format!("{}.fisher_weights.parquet", c.out).into_boxed_str())
        },
        feature_network,
        cell_cell,
        stop: None,
    };

    let out = ge::fit(&mut unified, config)?;

    ge::save_outputs(
        &out.model,
        &ge::OutputContext {
            feature_names: &unified.feature_names,
            barcodes: &unified.barcodes,
        },
        &c.out,
    )?;

    let cluster_labels: Option<Vec<usize>> = match args.cluster {
        Some(_) => Some(run_post_fit_clustering(
            args,
            &out.model,
            &unified.barcodes,
        )?),
        None => None,
    };

    info!(
        "Done — outputs at {}.{{latent,dictionary,*_bias}}.parquet{}{}",
        c.out,
        if has_coords {
            ", plus .coord_pairs.parquet"
        } else {
            ""
        },
        if cluster_labels.is_some() {
            ", plus .clusters.parquet"
        } else {
            ""
        }
    );

    Ok(())
}

/// Run leiden / kmeans on the fitted cell embedding and write
/// `{out}.clusters.parquet`. Returns the label vector for downstream
/// use (e.g. the spatial scatter plot).
fn run_post_fit_clustering(
    args: &SrtGbeArgs,
    model: &ge::JointEmbedModel,
    barcodes: &[Box<str>],
) -> anyhow::Result<Vec<usize>> {
    let method = args
        .cluster
        .expect("caller guarded on args.cluster.is_some()")
        .to_method();
    let latent = Mat::from_tensor(&model.e_cell)?;
    let labels = match method {
        crate::gbe::cluster::ClusterMethod::Leiden => crate::gbe::cluster::leiden_clusters(
            &latent,
            &crate::gbe::cluster::LeidenParams {
                knn: args.cluster_knn,
                resolution: args.cluster_resolution,
                target_clusters: args.num_clusters,
                seed: Some(args.common.seed),
            },
        )?,
        crate::gbe::cluster::ClusterMethod::Kmeans => {
            let k = args.num_clusters.unwrap_or_else(|| {
                let default_k = model.e_cell.dim(1).unwrap_or(8);
                info!("K-means: --num-clusters not set, defaulting to embedding dim ({default_k})");
                default_k
            });
            crate::gbe::cluster::kmeans_clusters(&latent, k, args.kmeans_max_iter)?
        }
    };
    let out_path = format!("{}.clusters.parquet", args.common.out);
    crate::gbe::cluster::write_clusters_parquet(&labels, barcodes, &out_path)?;
    info!("Wrote cluster assignments → {out_path}");
    Ok(labels)
}

/// Translate `--cell-cell-pb-levels` into the optional level list
/// passed to [`ge::CellCellConfig::pb_levels`]. `all` (or empty) →
/// `None` (every collapse level); a comma list like `0,2,4` →
/// `Some(vec![0,2,4])`. Disabling the cell-cell loss entirely is
/// controlled by `--cell-cell-lambda 0`, not this flag.
fn parse_pb_levels(levels_arg: &str) -> Option<Vec<usize>> {
    let trimmed = levels_arg.trim();
    if trimmed.eq_ignore_ascii_case("all") || trimmed.is_empty() {
        return None;
    }
    Some(
        trimmed
            .split(',')
            .filter_map(|s| {
                let t = s.trim();
                (!t.is_empty()).then(|| t.parse::<usize>())
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("invalid --cell-cell-pb-levels: expected `all` or a comma list of non-negative integers"),
    )
}
