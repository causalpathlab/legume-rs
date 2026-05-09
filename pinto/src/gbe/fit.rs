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
        cell_pairs.to_parquet(
            &format!("{}.spatial_knn.parquet", c.out),
            Some(coordinate_names.clone()),
        )?;
    } else {
        info!("Skipping spatial KNN side-output (no coordinates provided).");
    }

    info!("Constructing unified bipartite stream...");
    let unified = ge::UnifiedData::from_sparse_io(data_vec, &batch_membership)?;

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
    });

    let config = ge::FitConfig {
        embedding_dim: args.embedding_dim,
        num_coarsen_seeds: args.num_coarsen_seeds,
        super_cells: args.super_cells,
        sketch_dim: args.sketch_dim,
        epochs: args.epochs,
        batches_per_epoch: args.batches_per_epoch,
        batch_size: args.batch_size,
        num_negatives: args.num_negatives,
        learning_rate: args.learning_rate,
        seed: c.seed,
        device: dev,
        feature_network,
        cell_cell,
        stop: None,
    };

    let out = ge::fit(&unified, config)?;

    ge::save_outputs(
        &out.model,
        &ge::OutputContext {
            feature_names: &unified.feature_names,
            barcodes: &unified.barcodes,
        },
        &c.out,
    )?;

    info!(
        "Done — outputs at {}.{{latent,dictionary,*_bias}}.parquet{}",
        c.out,
        if has_coords {
            ", plus .spatial_knn.parquet"
        } else {
            ""
        }
    );

    Ok(())
}
