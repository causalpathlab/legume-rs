//! `pinto cage` entrypoint: activity-gated cell-graph embedding.
//!
//! Learns a per-CELL embedding `e_cell [n_cells x D]` by contrastive (NCE)
//! prediction of spatial adjacency, one gene at a time, with a Gibbs-sampled
//! selection over `(gene, dim)` pairs.
//!
//! ```text
//! load -> spatial KNN -> batch effects                  util::srt_pipeline
//! HVG-weighted projection -> graph_coarsen_multilevel   nested super-cell levels
//! build_cell_activities                                 per-gene activity, active edges
//! collapse + SVD + block Gibbs                          selection::select_features -> pip
//! training loop                                         the only SGD
//! parquet outputs + .pinto.json
//! ```
//!
//! # What the training loop optimizes
//!
//! Positives are REAL cell-cell edges from the spatial KNN. Negatives are cells
//! drawn per chain level: at level `l` a negative sits in a different super-cell
//! at `l` but the SAME parent at `l-1`, so the contrast sharpens with depth.
//! The coarsening hierarchy is therefore live in every step — as the negative
//! sampling structure, not as a separately trained entity. There is one
//! embedding, over cells; `e_pb` from the collapse is a frozen SVD the loss
//! never touches.
//!
//! ```text
//! s(u,v) = theta_g . (e_u (*) e_v) + b_u + b_v
//! theta_g = z_g (*) e_feat[g]        z ~ Bern(pip), redrawn once per EPOCH
//! ```
//!
//! `e_u (*) e_v` is the EDGE embedding: the pair's joint participation in each
//! latent community, high only when both endpoints load on it.
//!
//! # Two cadences, and they are different things
//!
//! - `pip` — the PROBABILITY. Estimated by block Gibbs, then RE-estimated every
//!   `--selection-refresh-epochs` against the live `e_cell` folded up into the
//!   pseudobulks, warm-started from the previous round. Without that refresh it
//!   would stay fixed at what a cold chain saw against an SVD basis the shipped
//!   model never uses.
//! - `z` — the SAMPLE. Drawn once per epoch from the current `pip`
//!   (`resample_gate_mask`), so each epoch trains a different sub-network and
//!   SGD optimizes `E_z[loss]` rather than `loss` at `E[z]`.
//!
//! `e_feat` is randomly initialized and the selection enters as this gate.
//! Deliberately NOT initialized to `E[z*beta]`: that already carries the
//! sampler's shrinkage, so gating it would apply the same selection twice. The
//! two are alternatives, not complements — `feature_posterior_mean.parquet`
//! ships `E[z*beta]` for downstream use.

use crate::cell_activity_graph_embedding::args::CellActivityGraphEmbeddingArgs;
use crate::cell_activity_graph_embedding::cluster::{
    edge_community_from_propensity, propensity_against_centroids, run_kmeans_clustering,
    KmeansClusteringArgs, KmeansClusteringResult,
};
use crate::cell_activity_graph_embedding::gene_chain_sampler::{
    build_gene_batch_cache, GeneGatedChainSampler,
};
use crate::cell_activity_graph_embedding::gene_gating::build_cell_activities;
use crate::cell_activity_graph_embedding::loss::{cage_nce_loss_per_level, CageLossOut};
use crate::link_community::outputs::write_link_communities;
use crate::util::cell_pairs::SrtCellPairs;
use crate::util::common::*;
use crate::util::graph_coarsen::{
    graph_coarsen_multilevel, CoarsenConfig, DcPoissonConfig, SeedingParams,
};
use crate::util::metadata::{create_cage_metadata, CageClusterInfo, RunInputs};
use crate::util::score_trace::{write_score_trace, ScoreEntry};
use crate::util::srt_pipeline::{preprocess_srt, SrtPreprocessConfig, SrtPreprocessed};

use candle_util::candle_core::Tensor;
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use data_beans_alg::gene_weighting::save_fisher_weights;
use data_beans_alg::hvg::select_hvg_streaming;
use data_beans_alg::random_projection::RandProjOps;
use graph_embedding_util::embedding_col_names;
use graph_embedding_util::loss::{build_per_batch_cell_samplers, PbChainFilter};
use graph_embedding_util::model::{GateKind, JointEmbedModel, ModelArgs, ModelInit};
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

    /////////////////////////////////////
    // 1-3. Load + KNN + batch effects //
    /////////////////////////////////////
    let SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        batch_effects: batch_db,
        graph,
        gene_weights: fisher_weights,
        n_cells,
        n_genes,
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        fisher_weights: !args.no_fisher_weights,
        batch_effects: true,
        feature_kind: Some(feature_kind),
    })?;

    let has_coords = c.has_coordinates();
    let cell_names = data_vec.column_names()?;
    let gene_names = data_vec.row_names()?;

    // Persist the NB-Fisher precision weights (when computed) so downstream
    // tools can reload them, matching the senna/chickpea convention.
    if let Some(w) = fisher_weights.as_ref() {
        save_fisher_weights(&c.out, w, &gene_names)?;
    }

    let srt_cell_pairs = SrtCellPairs::with_graph(&data_vec, &coordinates, &graph);
    srt_cell_pairs.to_parquet(
        &(c.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;
    let edges_owned: Vec<(u32, u32)> = srt_cell_pairs
        .inner
        .pairs()
        .iter()
        .map(|&(i, j)| (i as u32, j as u32))
        .collect();
    let n_edges = edges_owned.len();
    info!("{} cells, {} genes, {} edges", n_cells, n_genes, n_edges);

    /////////////////////////////////////////
    // 4. HVG-weighted proj -> coarsening //
    /////////////////////////////////////////
    let batch_arg: Option<&[Box<str>]> = if batch_db.is_some() {
        Some(&batch_membership)
    } else {
        None
    };
    // HVG selection WEIGHTS this projection; it does not subset the trained
    // gene axis. Non-selected genes get projection weight 0, so they sit out
    // the basis the coarsening hierarchy is built from, but they stay on the
    // feature axis: still trained, still sampled, still in the PIP table. The
    // selection shapes WHERE the pseudobulks land rather than which genes the
    // model may use — matching `senna bge` and `faba gem`.
    //
    // cage used to hard-subset here, which would now mean running a
    // variance-based selector in front of the spike-and-slab gate, with the
    // cruder one first and irreversible.
    let hvg_enabled = args.hvg.n_hvg > 0 || args.hvg.feature_list_file.is_some();
    let must_train =
        data_beans_alg::hvg::load_must_train(args.hvg.must_train_features.as_deref(), hvg_enabled)?;
    let hvg_weights: Option<Vec<f32>> = if hvg_enabled {
        let hvg = select_hvg_streaming(
            &data_vec,
            (args.hvg.n_hvg > 0).then_some(args.hvg.n_hvg),
            args.hvg.feature_list_file.as_deref(),
            must_train.as_ref(),
            c.block_size,
        )?;
        info!(
            "HVG-weighted projection: {} of {} genes weighted; all stay on the trained axis",
            hvg.selected_indices.len(),
            n_genes
        );
        Some(hvg.row_weights(n_genes))
    } else {
        None
    };

    let cell_proj = match hvg_weights.as_deref() {
        Some(w) => data_vec.project_columns_weighted(c.proj_dim, c.block_size, batch_arg, w)?,
        None => {
            data_vec.project_columns_with_batch_correction(c.proj_dim, c.block_size, batch_arg)?
        }
    };
    let ml = graph_coarsen_multilevel(
        &graph,
        &mut cell_proj.proj.clone(),
        srt_cell_pairs.inner.pairs(),
        CoarsenConfig {
            n_clusters: c.n_pseudobulk,
            num_levels: c.num_levels,
            refine_iterations: c.refine_iterations,
            seeding: has_coords.then(|| SeedingParams {
                coordinates: &coordinates,
                batch_membership: Some(&batch_membership),
            }),
            modularity_veto: None,
            // Second-opinion refinement on RAW counts, matching `pinto lc`.
            // cage was the only caller passing `None` here, so its levels were
            // cut on cosine-of-projection alone while `lc` got a degree-corrected
            // Poisson pass over the actual counts.
            //
            // This is also the path that consumes the parent labels for its
            // sibling constraint, and those were fed the WRONG level until the
            // coarsening loop was split into cut-then-refine passes — so
            // enabling it before that fix would have constrained moves against
            // the finer level rather than the coarser one.
            //
            // Opt out with `--no-dc-poisson`; the context build reads the count
            // matrix once, then every level reuses it.
            dc_poisson: (!args.no_dc_poisson).then(|| DcPoissonConfig {
                params: data_beans_alg::dc_poisson::RefineParams {
                    num_gibbs: 10,
                    num_greedy: 5,
                    feature_weighting: data_beans_alg::dc_poisson::FeatureWeighting::FisherInfoNb,
                    seed: c.seed,
                    gibbs_stagnation: 0.005,
                    profile_source: data_beans_alg::dc_poisson::ProfileSource::Raw,
                    ..Default::default()
                },
                data: &data_vec,
                num_genes: n_genes,
            }),
        },
    );

    // Collapse cells into super-cells, then sample the gene selection.
    let (selection, sel_state, mut sel_z) = {
        let pb = crate::cell_activity_graph_embedding::selection::build_pseudobulks(
            crate::cell_activity_graph_embedding::selection::PseudobulkArgs {
                data: &data_vec,
                all_cell_labels: &ml.all_cell_labels,
                graph: &graph,
                cell_features: &cell_proj.proj,
                embedding_dim: args.embedding_dim,
                block_size: c.block_size,
            },
        )?;
        // Diagnostic export: per CELL, its spatial coords, its finest-level
        // super-cell id, and that super-cell's embedding. Lets the basis be
        // inspected as a UMAP and as a spatial map before anything is built on it.
        if let Some(fine) = pb.levels.last() {
            let d = fine.e_pb.ncols();
            let mut out = Mat::zeros(n_cells, 3 + d);
            for i in 0..n_cells {
                let p = fine.cell_labels[i];
                out[(i, 0)] = coordinates[(i, 0)];
                out[(i, 1)] = coordinates[(i, 1)];
                out[(i, 2)] = p as f32;
                for h in 0..d {
                    out[(i, 3 + h)] = fine.e_pb[(p, h)];
                }
            }
            let mut cols: Vec<Box<str>> = vec![Box::from("x"), Box::from("y"), Box::from("pb")];
            cols.extend(embedding_col_names(d));
            out.to_parquet_with_names(
                &(c.out.to_string() + ".pseudobulk_cells.parquet"),
                (Some(&cell_names), Some("cell")),
                Some(&cols),
            )?;
            info!("Wrote {}.pseudobulk_cells.parquet", c.out);
        }

        // Sample which (gene, community) pairs are included.
        let (sel, sel_state, sel_z) =
            crate::cell_activity_graph_embedding::selection::select_features(
                &pb,
                args.embedding_dim,
                &crate::cell_activity_graph_embedding::selection::SelectArgs {
                    sweeps: args.selection_sweeps,
                    burnin: args.selection_sweeps / 2,
                    seed: c.seed,
                },
            )?;
        sel.log_summary();
        // The PIP table itself is NOT written. Under this arm `pip` is a set of
        // dropout keep-rates, not a reported estimand: it gates the gradient
        // during training and nothing downstream reads it, the same way a
        // dropout mask is never shipped. `senna bge` writes one because there
        // the gate is baked into the dictionary (`materialize_e_feat`) and the
        // table documents what was applied; cage ships the raw embedding, so a
        // table would only invite someone to re-apply a selection that was
        // regularization rather than a coefficient.
        //
        // The posterior mean EFFECTIVE loading `E[z·β]`.
        //
        // Not consumed by training under this arm: `e_feat` is randomly
        // initialized and the selection enters as a gate instead, because
        // `E[z·β]` already carries the shrinkage and gating it would apply the
        // same selection twice. It ships because it is the quantity downstream
        // wants when asking "how much does gene g load on dim h", and because
        // it is what an init-from-posterior arm would consume.
        sel.mean_beta_matrix().to_parquet_with_names(
            &(c.out.to_string() + ".feature_posterior_mean.parquet"),
            (Some(&gene_names), Some("gene")),
            Some(&embedding_col_names(args.embedding_dim)),
        )?;
        info!("Wrote {}.feature_posterior_mean.parquet", c.out);
        (sel, sel_state, sel_z)
    };

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

    ////////////////////////////////////////////
    // 5. Restrict pairs to the coarsened data //
    ////////////////////////////////////////////
    //
    // Despite the callee's name this step is about the PAIR DATA, not about
    // sampling. It decides which cell pairs cage is allowed to train on:
    //
    //   - cross-batch pairs are dropped outright;
    //   - a pair survives only if BOTH endpoints share a super-cell at EVERY
    //     `--chain-levels` entry, so the positives are within-super-cell pairs
    //     of the coarsened data rather than raw spatial edges;
    //   - per-cell degree over the retained pairs becomes the `--alpha-neg`
    //     negative weight, and per-level sibling pools are precomputed for the
    //     hard-negative draw.
    //
    // All of it is a deterministic function of the coarsening labels and the
    // edge list — no RNG here; randomness enters only at draw time.
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
        "Coarsened pair data: {} batches; dropped {} cross-batch, {} not within one super-cell at all chain levels",
        n_batches, sampler_stats.cross_batch_dropped, sampler_stats.pb_mismatch_dropped
    );

    let active_batch_count = per_batch.iter().filter(|s| s.is_some()).count();
    anyhow::ensure!(
        active_batch_count > 0,
        "no batch retained any within-batch within-pb edges; consider --chain-levels or --reciprocal"
    );

    //////////////////////////////////////////////////
    // 6. Per-gene activities + (gene, batch) cache //
    //////////////////////////////////////////////////
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
    let cache = build_gene_batch_cache(&activities, &per_batch, args.activity_alpha);
    info!(
        "Gene-batch cache: {} active (gene, batch) pairs",
        cache.n_active_pairs()
    );
    // List of genes with at least one cached (gene, batch) entry — used
    // to skip empty genes in every epoch's permutation. Computed now so
    // we can free `activities` afterwards.
    let trainable_genes: Vec<usize> = (0..n_genes)
        .filter(|&g| cache.entries[g].iter().any(|e| e.is_some()))
        .collect();
    info!(
        "{} trainable genes (≥1 active batch)",
        trainable_genes.len()
    );
    // Activities can be dropped now — the cache owns everything the
    // sampler needs for v1.
    drop(activities);

    anyhow::ensure!(
        !trainable_genes.is_empty(),
        "no genes have an active edge in any batch — nothing to train"
    );

    //////////////////////////
    // 7. Model + optimizer //
    //////////////////////////
    let dev = args.device.to_device(args.device_no)?;
    info!("Using device: {:?}", dev);
    let varmap = VarMap::new();
    // `JointEmbedModel.e_feat` / `b_feat` ARE the gene embedding
    // (cells and genes share the same D-dim space). `n_features =
    // n_genes` and `b_feat` is zero-init per gene; both are learned
    // alongside the cell side via AdamW over `varmap.all_vars()`.
    //
    // `e_feat` is randomly initialized and the SELECTION enters as a gate:
    // The sampled `pip` is installed as drop rates, and a fresh `z ~ Bern(pip)` is drawn
    // once per epoch (see the training loop). This is `senna bge`'s arm, using
    // the same `JointEmbedModel` methods rather than a local re-derivation.
    //
    // Deliberately NOT initialized to `E[z·β]`: that already carries the
    // sampler's shrinkage, so gating it would apply the same selection twice.
    // The two are alternatives, not complements.
    let mut model = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features: n_genes,
            n_cells,
            embedding_dim: args.embedding_dim,
            seed: c.seed,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &vec![0.0_f32; n_genes],
            b_cell: &vec![0.0_f32; n_cells],
        },
        &varmap,
        &dev,
    )?;
    // The cold sampler's `pip`. `resample_gate_mask` is a no-op without it, so
    // this install is what turns per-epoch spike sampling on at all.
    {
        let pip_t = Tensor::from_vec(selection.pip.clone(), (n_genes, args.embedding_dim), &dev)?;
        model.install_gate_pip(GateKind::Identity, &pip_t)?;
    }

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.lr as f64,
            ..Default::default()
        },
    )?;

    //////////////////////
    // 8. Training loop //
    //////////////////////
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
        // ONE `z ~ Bern(pip)` draw for the whole epoch. Per-EPOCH, not
        // per-chunk: `z` is a latent for the DATASET, so a per-chunk draw would
        // model each chunk as having its own inclusion state and add gradient
        // variance for nothing. Same call, same cadence as `senna bge`
        // (`geu::training::train_composite`).
        //
        // The initial draw is NOT the informative one — a single sample of a
        // 36591x16 Bernoulli table says little. What the model sees is the
        // AVERAGE over epochs: SGD optimizes `E_z[loss]`, marginalized over the
        // selection posterior, rather than `loss` at `E[z]`. Those differ by
        // Jensen, and the first is what the generative model implies.
        // RE-ESTIMATE `pip` against what SGD has learned so far, then draw
        // this epoch's `z` from it. Skipped at epoch 0, whose `pip` is the
        // cold chain's `pip`, already installed above.
        //
        // This is NOT MCEM, and should not be read as one. EM needs both steps
        // on the SAME objective; here the refresh is a Poisson fit on
        // super-cell counts while the SGD is edge NCE, so there is no joint
        // likelihood being monotonically improved and none of EM's convergence
        // guarantees apply.
        //
        // What it actually is: DROPOUT-style stochastic gating whose per-(gene,
        // dim) keep-probability is refreshed from a companion count model. The
        // hard 0/1 `z` below zeroes the gradient for excluded coordinates, so
        // each epoch trains a different sub-network — the regularizer. The
        // refresh exists only to stop those keep-probabilities going stale:
        // the cold `pip` was fit against `e_pb` from an SVD of pseudobulk
        // log-counts, which is NOT the embedding the model ships, so without it
        // the gate would keep dropping genes on the strength of a cell side
        // nothing uses.
        //
        // The chain is WARM-STARTED from the previous round's `final_z`, so a
        // refresh needs far fewer sweeps than the cold run — that is exactly
        // what `DimBlockResult::final_z` exists for.
        if epoch > 0
            && args.selection_refresh_epochs > 0
            && epoch % args.selection_refresh_epochs == 0
        {
            let e_cell_now = tensor_to_mat(&model.e_cell)?;
            let e_flat = sel_state.frozen_e_from_cells(&e_cell_now);
            let (sel_new, z_new) = sel_state.sample(
                &e_flat,
                &crate::cell_activity_graph_embedding::selection::SelectArgs {
                    sweeps: args.selection_refresh_sweeps,
                    burnin: args.selection_refresh_sweeps / 2,
                    seed: c.seed.wrapping_add(epoch as u64),
                },
                Some(sel_z.clone()),
            );
            sel_z = z_new;
            let pip_t = Tensor::from_vec(sel_new.pip.clone(), (n_genes, args.embedding_dim), &dev)?;
            model.install_gate_pip(GateKind::Identity, &pip_t)?;
            let m: f32 = sel_new.pip.iter().sum::<f32>() / sel_new.pip.len() as f32;
            info!("epoch {epoch}: pip refreshed against e_cell, mean = {m:.4}");
        }
        model.resample_gate_mask()?;
        perm.shuffle(&mut rng_master);
        // Optional cost lever: visit a random subset of the gene axis this
        // epoch. Stochastic coverage, NOT axis selection — every gene stays
        // in the model, keeps its gate, and appears in the output tables; it
        // just waits for a later epoch.
        if args.genes_per_epoch > 0 && args.genes_per_epoch < perm.len() {
            perm.truncate(args.genes_per_epoch);
        }

        let mut skip_count: usize = 0;
        let mut sample_count: usize = 0;

        let sampler_ref = &sampler;
        // Tensor accumulators — avoid per-step CUDA → CPU sync. The
        // `.detach()` strips autograd so we can keep adding into them
        // without retaining the backward graph across chunks.
        let mut epoch_loss_acc: Option<Tensor> = None;
        let mut per_level_acc: Option<Tensor> = None;
        let mut pair_acc: Option<Tensor> = None;
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

            // (b) ONE forward / backward over the whole chunk.
            let CageLossOut {
                per_level: per_level_gl,
                mean_abs_pair,
            } = cage_nce_loss_per_level(&model, cb_batches, &gene_ids_u32, &dev)?; // [G, L]
                                                                                   // NB-Fisher per-gene precision: scale each gene's row of the
                                                                                   // per-level loss by w_g ∈ (0,1] before summing, so high-mean /
                                                                                   // high-dispersion housekeeping genes contribute less gradient
                                                                                   // (the loss-side analog of bge's count·fisher positive draw and
                                                                                   // lc's `apply_gene_weights` on the gene basis). Coverage stays
                                                                                   // uniform — every gene is still visited once per epoch.
            let per_level_gl = match fisher_weights.as_ref() {
                Some(w) => {
                    let w_chunk: Vec<f32> = gene_ids.iter().map(|&g| w[g]).collect();
                    let w_g1 = Tensor::from_vec(w_chunk, (gene_ids.len(), 1), &dev)?;
                    per_level_gl.broadcast_mul(&w_g1)?
                }
                None => per_level_gl,
            };
            let loss = per_level_gl.sum_all()?;
            let mut total = loss.clone();

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
            pair_acc = Some(match pair_acc {
                Some(prev) => (prev + &mean_abs_pair)?,
                None => mean_abs_pair,
            });
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

        // Pair-term magnitude: the collapse detector. If this decays toward
        // zero while the loss still falls, the ungated cell biases have taken
        // over the objective and the gene direction is doing nothing.
        if chunk_count > 0 {
            let pair_mean = pair_acc
                .as_ref()
                .and_then(|t| t.to_scalar::<f32>().ok())
                .map_or(f64::NAN, |v| v as f64 / chunk_count as f64);
            info!("epoch {}: mean |pair| = {:.4e}", epoch, pair_mean);
        }

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

    ////////////////
    // 9. Outputs //
    ////////////////
    // Drop the epoch's spike draw so every emitted table reports the MEAN
    // `E[z] = pip`, not whichever sub-network the last epoch happened to run.
    model.clear_gate_mask();

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

    // Gene embedding [G × D] — same shared D-dim space as cells, over the FULL
    // gene axis. There is no HVG row subset to mirror any more: HVG weights the
    // projection, so every gene is trained and every gene gets a row here.
    //
    // These are the trained effects as-is. The selection is NOT re-applied on
    // top and no `pip` ships to let anyone re-apply it: the gate was dropout
    // over the gradient, so what it did is already expressed in the values
    // below. Multiplying them by a keep-rate would shrink a second time for a
    // selection that was never a coefficient.
    let e_gene_out = tensor_to_mat(&model.e_feat)?;
    let b_gene_out = tensor_to_mat_1d(&model.b_feat)?;

    e_gene_out.to_parquet_with_names(
        &(c.out.to_string() + ".feature_embedding.parquet"),
        (Some(&gene_names), Some("feature")),
        Some(&embedding_col_names(args.embedding_dim)),
    )?;

    b_gene_out.to_parquet_with_names(
        &(c.out.to_string() + ".gene_bias.parquet"),
        (Some(&gene_names), Some("gene")),
        Some(&[Box::from("b_gene")]),
    )?;

    write_score_trace(&(c.out.to_string() + ".scores.parquet"), &score_trace)?;

    //////////////////////////////////////////////////
    // 10. Optional k-means clustering + propensity //
    //////////////////////////////////////////////////
    // k-means++ (via matrix-util) on the L2-normalized cell embedding.
    // Soft propensity comes from temperature-softmax of cosine to the
    // per-cluster centroid; genes get the same recipe against the same
    // centroids (feature dictionary [G × K]).
    let run_clustering = args.n_clusters > 0;
    let cluster_info: Option<CageClusterInfo> = if run_clustering {
        let kmeans_args = KmeansClusteringArgs {
            n_clusters: args.n_clusters,
            propensity_temp: args.propensity_temp,
            max_iter: args.kmeans_max_iter,
        };
        let KmeansClusteringResult {
            labels,
            n_clusters,
            propensity,
            centroids,
        } = run_kmeans_clustering(&e_cell_mat, &kmeans_args)?;

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

        // Feature dictionary [G × K] — same softmax-over-centroids recipe on
        // the gene side, over the full gene axis.
        let feature_dict =
            propensity_against_centroids(&e_gene_out, &centroids, args.propensity_temp);
        feature_dict.to_parquet_with_names(
            &(c.out.to_string() + ".feature_dictionary.parquet"),
            (Some(&gene_names), Some("feature")),
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
            gene_names.len(),
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
