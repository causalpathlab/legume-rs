//! `pinto cage` entrypoint: activity-gated cell-graph embedding.
//!
//! Pipeline:
//!   1. preprocess_srt → data + spatial KNN + batch effects
//!   2. SrtCellPairs::with_graph + coord_pairs.parquet
//!   3. graph_coarsen_multilevel (no DC-Poisson — embedding-only)
//!   4. build_per_batch_cell_samplers with PbChainFilter
//!   5. build_cell_activities (per-gene cell activity + active edges)
//!   6. allocate JointEmbedModel + GeneGating in one VarMap, AdamW
//!   7. training loop — rayon sample, serial fwd/bwd
//!   8. parquet outputs + .pinto.json

use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
use crate::cell_activity_graph_embedding::cluster::{
    edge_community_from_propensity, propensity_against_centroids, run_leiden_and_propensity,
    LeidenPropensityArgs, LeidenPropensityResult,
};
use crate::cell_activity_graph_embedding::gene_chain_sampler::{
    build_gene_batch_cache, GeneGatedChainSampler,
};
use crate::cell_activity_graph_embedding::gene_gating::{
    build_cell_activities, softplus_floored, LevelDimGate,
};
use crate::link_community::outputs::write_link_communities;
use crate::util::cell_pairs::SrtCellPairs;
use crate::util::common::*;
use crate::util::graph_coarsen::{graph_coarsen_multilevel, CoarsenConfig, SeedingParams};
use crate::util::metadata::{create_cage_metadata, CageClusterInfo, RunInputs};
use crate::util::score_trace::{write_score_trace, ScoreEntry};
use crate::util::srt_pipeline::{preprocess_srt, SrtPreprocessConfig, SrtPreprocessed};

use candle_util::candle_core::{DType, Tensor};
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use data_beans_alg::hvg::select_hvg_streaming;
use data_beans_alg::random_projection::RandProjOps;
use graph_embedding_util::loss::{
    build_per_batch_cell_samplers, cell_cell_nce_loss_per_level_batched_gated, PbChainFilter,
};
use graph_embedding_util::model::{JointEmbedModel, ModelArgs, ModelInit};
use graph_embedding_util::stop::setup_stop_handler;
use matrix_util::common_io::mkdir_parent;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::atomic::Ordering;

pub fn fit_cell_activity_graph_embedding(
    args: &CellActivityGraphEmbeddingArgs,
) -> anyhow::Result<()> {
    let c = &args.common;
    mkdir_parent(&c.out)?;

    anyhow::ensure!(args.embedding_dim > 0, "embedding-dim must be > 0");
    anyhow::ensure!(args.epochs > 0, "epochs must be > 0");
    anyhow::ensure!(
        !args.chain_levels.is_empty(),
        "chain-levels must be non-empty"
    );

    // Peek the first data file's row names so `auto` can dispatch
    // FeatureNameKind::auto_detect without paying for a full sparse
    // load up front.
    let peek_names = data_beans::convert::try_open_or_convert(&c.data_files[0])?.row_names()?;
    let feature_kind = args.gene_name_mode.resolve_kind(&peek_names);
    info!(
        "Gene-name canonicalization: {:?} (mode = {:?}, peeked {} names from {})",
        feature_kind,
        args.gene_name_mode,
        peek_names.len(),
        c.data_files[0]
    );

    //////////////////////////////////////////////////////
    // 1-3. Load + KNN + batch effects                  //
    //////////////////////////////////////////////////////
    let SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        batch_effects: batch_db,
        graph,
        gene_weights: _,
        n_cells,
        n_genes,
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        fisher_weights: false,
        batch_effects: true,
        feature_kind: Some(feature_kind),
    })?;

    let has_coords = c.has_coordinates();
    let cell_names = data_vec.column_names()?;
    let gene_names = data_vec.row_names()?;

    let srt_cell_pairs = SrtCellPairs::with_graph(&data_vec, &coordinates, graph);
    srt_cell_pairs.to_parquet(
        &(c.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;
    let edges_owned: Vec<(u32, u32)> = srt_cell_pairs
        .graph
        .edges
        .iter()
        .map(|&(i, j)| (i as u32, j as u32))
        .collect();
    let n_edges = edges_owned.len();
    info!("{} cells, {} genes, {} edges", n_cells, n_genes, n_edges);

    //////////////////////////////////////////////////////
    // 4. Coarsening (no DC-Poisson — embedding-only)   //
    //////////////////////////////////////////////////////
    let batch_arg: Option<&[Box<str>]> = if batch_db.is_some() {
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

    // Chain levels must be valid indices into all_cell_labels.
    for &lvl in &args.chain_levels {
        anyhow::ensure!(
            lvl < ml.all_cell_labels.len(),
            "--chain-levels entry {} out of range (num_levels = {})",
            lvl,
            ml.all_cell_labels.len()
        );
    }
    let n_chain_levels = args.chain_levels.len();

    //////////////////////////////////////////////////////
    // 5. Per-batch chain samplers                       //
    //////////////////////////////////////////////////////
    let batch_id_of: HashMap<Box<str>, u32> = {
        let mut uniq: Vec<Box<str>> = batch_membership.to_vec();
        uniq.sort();
        uniq.dedup();
        uniq.into_iter()
            .enumerate()
            .map(|(i, b)| (b, i as u32))
            .collect()
    };
    let n_batches = batch_id_of.len().max(1);
    let batch_membership_u32: Vec<u32> = batch_membership
        .iter()
        .map(|b| *batch_id_of.get(b).expect("batch id"))
        .collect();

    let pb_filter = PbChainFilter {
        cell_to_pb_per_level: &ml.all_cell_labels,
        levels: &args.chain_levels,
    };
    let (per_batch, sampler_stats) = build_per_batch_cell_samplers(
        &edges_owned,
        &batch_membership_u32,
        n_batches,
        n_cells,
        args.alpha_neg,
        Some(pb_filter),
    );
    info!(
        "Per-batch samplers: {} batches; cross_batch_dropped={}, pb_mismatch_dropped={}",
        n_batches, sampler_stats.cross_batch_dropped, sampler_stats.pb_mismatch_dropped
    );

    let active_batch_count = per_batch.iter().filter(|s| s.is_some()).count();
    anyhow::ensure!(
        active_batch_count > 0,
        "no batch retained any within-batch within-pb edges; consider --chain-levels or --reciprocal"
    );

    //////////////////////////////////////////////////////
    // 6. Per-gene activities + (gene, batch) cache      //
    //////////////////////////////////////////////////////
    info!("Computing per-gene cell activities...");
    let activities =
        build_cell_activities(&data_vec, &edges_owned, c.block_size, args.activity_norm)?;
    let nonzero_genes = activities
        .gene_active_edges
        .iter()
        .filter(|v| !v.is_empty())
        .count();
    info!("{}/{} genes have ≥1 active edge", nonzero_genes, n_genes);

    info!("Precomputing per-(gene, batch) positive distributions...");
    let cache = build_gene_batch_cache(&activities, &per_batch);
    info!(
        "Gene-batch cache: {} active (gene, batch) pairs",
        cache.n_active_pairs()
    );
    // List of genes with at least one cached (gene, batch) entry — used
    // to skip empty genes in every epoch's permutation. Computed now so
    // we can free `activities` afterwards.
    let mut trainable_genes: Vec<usize> = (0..n_genes)
        .filter(|&g| cache.entries[g].iter().any(|e| e.is_some()))
        .collect();
    info!(
        "{} trainable genes (≥1 active batch)",
        trainable_genes.len()
    );
    // Activities can be dropped now — the cache owns everything the
    // sampler needs for v1.
    drop(activities);

    // HVG subset: senna-style top-K via `select_hvg_streaming` (or a
    // user-provided `--feature-list-file`), applied as a subset of the
    // training gene axis. Non-HVG genes keep their randn-init e_gene
    // rows untouched and are dropped from the parquet output below.
    let hvg_selected: Option<Vec<usize>> = if args.hvg.n_hvg > 0
        || args.hvg.feature_list_file.is_some()
    {
        let hvg = select_hvg_streaming(
            &data_vec,
            (args.hvg.n_hvg > 0).then_some(args.hvg.n_hvg),
            args.hvg.feature_list_file.as_deref(),
            c.block_size,
        )?;
        let kept: std::collections::HashSet<usize> = hvg.selected_indices.iter().copied().collect();
        let before = trainable_genes.len();
        trainable_genes.retain(|g| kept.contains(g));
        info!(
                "HVG subset: kept {} / {} genes after intersection with trainable set ({} HVGs selected)",
                trainable_genes.len(),
                before,
                hvg.selected_indices.len()
            );
        Some(hvg.selected_indices)
    } else {
        None
    };
    anyhow::ensure!(
        !trainable_genes.is_empty(),
        "no trainable genes after HVG / active-edge filter"
    );

    //////////////////////////////////////////////////////
    // 7. Model + gates + optimizer                      //
    //////////////////////////////////////////////////////
    let dev = args.device.to_device(args.device_no)?;
    info!("Using device: {:?}", dev);
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    // `JointEmbedModel.e_feat` / `b_feat` ARE the gene embedding
    // (cells and genes share the same D-dim space). `n_features =
    // n_genes` and `b_feat` is zero-init per gene; both are learned
    // alongside the cell side via AdamW over `varmap.all_vars()`.
    let model = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features: n_genes,
            n_cells,
            embedding_dim: args.embedding_dim,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &vec![0.0_f32; n_genes],
            b_cell: &vec![0.0_f32; n_cells],
        },
        &varmap,
        vs,
        &dev,
    )?;
    let gate = LevelDimGate::new(n_chain_levels, args.embedding_dim, &varmap, &dev)?;
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.lr as f64,
            ..Default::default()
        },
    )?;

    //////////////////////////////////////////////////////
    // 8. Training loop                                  //
    //////////////////////////////////////////////////////
    let pb_maps: Vec<&[usize]> = args
        .chain_levels
        .iter()
        .map(|&lvl| ml.all_cell_labels[lvl].as_slice())
        .collect();

    let sampler = GeneGatedChainSampler {
        edges: &edges_owned,
        per_batch: &per_batch,
        cache: &cache,
        pb_maps: &pb_maps,
        batch_size: args.per_gene_batch,
        n_negatives: args.n_negatives,
    };

    let mut score_trace: Vec<ScoreEntry> = Vec::new();
    let mut rng_master = SmallRng::seed_from_u64(c.seed);

    // First ^C = graceful stop after current chunk, finalize outputs;
    // second ^C = hard abort. See graph_embedding_util::stop.
    let stop = setup_stop_handler();

    'epochs: for epoch in 0..args.epochs {
        let mut perm: Vec<usize> = trainable_genes.clone();
        perm.shuffle(&mut rng_master);

        let mut skip_count: usize = 0;
        let mut sample_count: usize = 0;

        let sampler_ref = &sampler;
        // Tensor accumulators — avoid per-step CUDA → CPU sync. The
        // `.detach()` strips autograd so we can keep adding into them
        // without retaining the backward graph across chunks.
        let mut epoch_loss_acc: Option<Tensor> = None;
        let mut per_level_acc: Option<Tensor> = None;
        let mut chunk_count: usize = 0;
        for chunk in perm.chunks(args.gene_batch_size) {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            // (a) Parallel sampling — pure CPU, no candle.
            let mini: Vec<(usize, _)> = chunk
                .par_iter()
                .flat_map_iter(|&g| {
                    let seed = c
                        .seed
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(g as u64)
                        .wrapping_add((epoch as u64).wrapping_mul(1_000_003));
                    let mut rng = SmallRng::seed_from_u64(seed);
                    (0..n_batches)
                        .filter_map(move |b| {
                            sampler_ref.sample(g, b, &mut rng).map(|(cb, _st)| (g, cb))
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            if mini.is_empty() {
                skip_count += chunk.len() * n_batches;
                continue;
            }

            let (gene_ids, cb_batches): (Vec<usize>, Vec<_>) = mini.into_iter().unzip();
            sample_count += gene_ids.len();

            // Gene identity enters the score function via the gated
            // loss. The chunk's gene ids parallel-replicate to align
            // with the [G*B] cell-side gathers.
            let gene_ids_u32: Vec<u32> = gene_ids.iter().map(|&g| g as u32).collect();

            // Per-level per-dim gate γ enters the score: rebuilt once
            // per chunk; the loss applies it per chain level via
            // `dim_gates`.
            let dim_gates = softplus_floored(&gate.gamma)?; // [L, D]

            // (b) ONE forward / backward over the whole chunk.
            let per_level_gl = cell_cell_nce_loss_per_level_batched_gated(
                &model,
                cb_batches,
                &gene_ids_u32,
                Some(&dim_gates),
                None, // smoother — wired later for --gene-network
                &dev,
            )?; // [G, L]
            let loss = per_level_gl.sum_all()?;
            let mut total = loss.clone();
            if args.gate_l2 > 0.0 {
                let reg = (dim_gates.sqr()?.sum_all()? * (args.gate_l2 as f64))?;
                total = (total + reg)?;
            }
            if args.embedding_l2 > 0.0 {
                let lam = args.embedding_l2 as f64;
                let cell_l2 = (model.e_cell.sqr()?.mean_all()? * lam)?;
                let gene_l2 = (model.e_feat.sqr()?.mean_all()? * lam)?;
                total = (total + cell_l2)?;
                total = (total + gene_l2)?;
            }
            opt.backward_step(&total)?;

            // Diagnostics (no host sync) — accumulate detached tensors.
            // Per-gene mean of per_level → [L] for this chunk; running sum.
            let per_level_chunk_mean = per_level_gl.mean(0)?.detach();
            let loss_chunk = loss.detach();
            epoch_loss_acc = Some(match epoch_loss_acc {
                Some(prev) => (prev + loss_chunk)?,
                None => loss_chunk,
            });
            per_level_acc = Some(match per_level_acc {
                Some(prev) => (prev + per_level_chunk_mean)?,
                None => per_level_chunk_mean,
            });
            chunk_count += 1;
        }

        // ONE host sync per epoch — pull accumulated tensors only now.
        let (mean_loss, mean_per_level) = if chunk_count > 0 {
            let loss_sum: f32 = epoch_loss_acc
                .as_ref()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap_or(f32::NAN);
            let pl_sum: Vec<f32> = per_level_acc
                .as_ref()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap_or_default();
            let scale = chunk_count as f64;
            (
                (loss_sum as f64) / scale,
                pl_sum.iter().map(|v| *v as f64 / scale).collect::<Vec<_>>(),
            )
        } else {
            (f64::NAN, vec![0.0; n_chain_levels])
        };
        info!(
            "epoch {}: mean loss = {:.4e} (per-level: {:?}), samples = {}, skipped pairs = {}",
            epoch, mean_loss, mean_per_level, sample_count, skip_count
        );

        // Push one summary row per epoch. `level = epoch`, `sweep = 0`.
        // `total_mass = #samples`, `mutual_information` = mean of per-level loss
        // (informally summarizing chain effectiveness).
        let mi = mean_per_level.iter().sum::<f64>() / (mean_per_level.len().max(1) as f64);
        score_trace.push(ScoreEntry {
            level: epoch as i32,
            sweep: 0,
            score: mean_loss,
            n_edges: sample_count,
            total_mass: sample_count as f64,
            mutual_information: mi,
        });

        // Convergence check: if the last `convergence_window` epochs'
        // mean losses are within `convergence_tol` relative range,
        // exit. `convergence_window == 0` disables.
        if args.convergence_window > 0 && score_trace.len() >= args.convergence_window {
            let window = &score_trace[score_trace.len() - args.convergence_window..];
            let losses: Vec<f64> = window.iter().map(|e| e.score).collect();
            let mean = losses.iter().sum::<f64>() / (losses.len() as f64);
            let (lo, hi) = losses
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            let rel = if mean.abs() > 1e-30 {
                (hi - lo) / mean.abs()
            } else {
                f64::INFINITY
            };
            if rel.is_finite() && rel < args.convergence_tol as f64 {
                info!(
                    "converged at epoch {}: last {} losses within rel-range {:.4} < tol {:.4}",
                    epoch, args.convergence_window, rel, args.convergence_tol
                );
                break;
            }
        }

        if stop.load(Ordering::SeqCst) {
            info!(
                "Stopping early at epoch {}/{} — finalizing outputs",
                epoch + 1,
                args.epochs
            );
            break 'epochs;
        }
    }

    //////////////////////////////////////////////////////
    // 9. Outputs                                        //
    //////////////////////////////////////////////////////
    info!("Writing cage outputs...");

    // Cell embedding [N × D]
    let e_cell_mat = tensor_to_mat(&model.e_cell)?;
    e_cell_mat.to_parquet_with_names(
        &(c.out.to_string() + ".cell_embedding.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&embedding_col_names(args.embedding_dim)),
    )?;

    // Cell bias [N]
    let b_cell_mat = tensor_to_mat_1d(&model.b_cell)?;
    b_cell_mat.to_parquet_with_names(
        &(c.out.to_string() + ".cell_bias.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&[Box::from("b_cell")]),
    )?;

    // Gene embedding [G × D] — same shared D-dim space as cells. When
    // HVG selection was active, restrict the parquet to the K selected
    // rows so downstream tools don't see untrained randn rows for
    // dropped genes.
    let e_gene_mat = tensor_to_mat(&model.e_feat)?;
    let b_gene_mat = tensor_to_mat_1d(&model.b_feat)?;

    let (e_gene_out, b_gene_out, gene_names_out): (Mat, Mat, Vec<Box<str>>) =
        if let Some(sel) = hvg_selected.as_ref() {
            let kept: Vec<usize> = sel.to_vec();
            (
                subset_rows(&e_gene_mat, kept.iter().copied())?,
                subset_rows(&b_gene_mat, kept.iter().copied())?,
                kept.iter().map(|&i| gene_names[i].clone()).collect(),
            )
        } else {
            (e_gene_mat, b_gene_mat, gene_names.clone())
        };

    e_gene_out.to_parquet_with_names(
        &(c.out.to_string() + ".feature_embedding.parquet"),
        (Some(&gene_names_out), Some("feature")),
        Some(&embedding_col_names(args.embedding_dim)),
    )?;

    b_gene_out.to_parquet_with_names(
        &(c.out.to_string() + ".gene_bias.parquet"),
        (Some(&gene_names_out), Some("gene")),
        Some(&[Box::from("b_gene")]),
    )?;

    // Per-level per-dim gate γ [L × D] post-softplus_floored, the
    // learned "which embedding dim matters at this chain level" map.
    let gates_t = gate.snapshot_gates()?;
    let gates_mat = tensor_to_mat(&gates_t)?;
    let level_names: Vec<Box<str>> = args
        .chain_levels
        .iter()
        .map(|&lvl| format!("level_{lvl}").into_boxed_str())
        .collect();
    gates_mat.to_parquet_with_names(
        &(c.out.to_string() + ".level_dim_gates.parquet"),
        (Some(&level_names), Some("level")),
        Some(&embedding_col_names(args.embedding_dim)),
    )?;

    write_score_trace(&(c.out.to_string() + ".scores.parquet"), &score_trace)?;

    //////////////////////////////////////////////////////
    // 10. Optional Leiden clustering + propensity      //
    //////////////////////////////////////////////////////
    // Cells: L2-normalized cosine kNN → Leiden → hard labels +
    // soft propensity (cells × K). Genes: same softmax-over-centroids
    // recipe re-applied to e_gene_out → feature dictionary (genes × K).
    let cluster_info: Option<CageClusterInfo> = if args.leiden_knn > 0 {
        let leiden_args = LeidenPropensityArgs {
            knn: args.leiden_knn,
            resolution: args.leiden_resolution,
            target_clusters: args.leiden_target_clusters,
            min_cluster_size: args.leiden_min_cluster_size,
            propensity_temp: args.propensity_temp,
            seed: c.seed,
        };
        let LeidenPropensityResult {
            labels,
            n_clusters,
            propensity,
            centroids,
        } = run_leiden_and_propensity(&e_cell_mat, &leiden_args)?;

        let cluster_col_names: Vec<Box<str>> = (0..n_clusters)
            .map(|k| format!("cluster_{k}").into_boxed_str())
            .collect();

        // Hard cluster labels [N × 1], NaN for filtered cells.
        let mut hard = Mat::zeros(n_cells, 1);
        for (i, &lab) in labels.iter().enumerate() {
            hard[(i, 0)] = if lab == usize::MAX {
                f32::NAN
            } else {
                lab as f32
            };
        }
        hard.to_parquet_with_names(
            &(c.out.to_string() + ".clusters.parquet"),
            (Some(&cell_names), Some("cell")),
            Some(&[Box::from("cluster")]),
        )?;

        // Soft cell propensity [N × K].
        propensity.to_parquet_with_names(
            &(c.out.to_string() + ".cluster_propensity.parquet"),
            (Some(&cell_names), Some("cell")),
            Some(&cluster_col_names),
        )?;

        // Feature dictionary [G × K] — same softmax-over-centroids
        // recipe on the gene side. Uses the HVG-subset gene matrix
        // when HVG was active, so the row axis matches
        // feature_embedding.parquet.
        let feature_dict =
            propensity_against_centroids(&e_gene_out, &centroids, args.propensity_temp);
        feature_dict.to_parquet_with_names(
            &(c.out.to_string() + ".feature_dictionary.parquet"),
            (Some(&gene_names_out), Some("feature")),
            Some(&cluster_col_names),
        )?;

        // Per-edge community via Hadamard-argmax of endpoint propensity.
        // Lands at `link_community.parquet` so `pinto lr-activity
        // --lc-prefix {prefix}` can run directly without an adapter.
        let edges_usize: Vec<(usize, usize)> = edges_owned
            .iter()
            .map(|&(u, v)| (u as usize, v as usize))
            .collect();
        let edge_community = edge_community_from_propensity(&edges_owned, &propensity);
        write_link_communities(
            &(c.out.to_string() + ".link_community.parquet"),
            &edges_usize,
            &edge_community,
            &cell_names,
        )?;

        info!(
            "Wrote leiden cluster artifacts: {} clusters, {} cells × {} = propensity, \
             {} features × {} = dictionary, {} edges = link_community",
            n_clusters,
            n_cells,
            n_clusters,
            gene_names_out.len(),
            n_clusters,
            edges_usize.len(),
        );
        Some(CageClusterInfo { n_clusters })
    } else {
        None
    };

    // Metadata
    {
        let coord_file_str = c.coord_files_joined();
        let meta = create_cage_metadata(
            &RunInputs {
                prefix: &c.out,
                data_files: &c.data_files,
                coord_file: coord_file_str.as_deref(),
                coord_columns: &coordinate_names,
                n_cells,
                n_genes,
                n_edges,
                k: args.embedding_dim,
            },
            batch_db.is_some(),
            cluster_info,
        );
        let meta_path = std::path::PathBuf::from(format!("{}.pinto.json", c.out));
        meta.write(&meta_path)?;
        info!("Wrote {}", meta_path.display());
    }

    info!("Done");
    Ok(())
}

fn embedding_col_names(d: usize) -> Vec<Box<str>> {
    (0..d).map(|i| format!("e{i}").into_boxed_str()).collect()
}

/// Convert a 2-D `[R × C]` candle Tensor into an `nalgebra::DMatrix<f32>`
/// suitable for `to_parquet_with_names`.
fn tensor_to_mat(t: &Tensor) -> anyhow::Result<Mat> {
    let rows = t.dim(0)?;
    let cols = t.dim(1)?;
    let data: Vec<f32> = t.to_vec2::<f32>()?.into_iter().flatten().collect();
    // `from_row_slice` expects row-major, matching `to_vec2`.
    Ok(Mat::from_row_slice(rows, cols, &data))
}

fn tensor_to_mat_1d(t: &Tensor) -> anyhow::Result<Mat> {
    let n = t.dim(0)?;
    let data: Vec<f32> = t.to_vec1::<f32>()?;
    Ok(Mat::from_column_slice(n, 1, &data))
}
