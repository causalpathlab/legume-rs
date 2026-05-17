//! `pinto cage-mcmc` orchestration.
//!
//! Pipeline mirrors `pinto cage` up through the gene-batch cache; from
//! there it diverges to: pure-CPU nalgebra state, ESS sweeps over four
//! parameter blocks, and streaming posterior moments via
//! [`SparseRunningStatistics`](matrix_util::sparse_stat::SparseRunningStatistics)
//! — each MCMC sample contributes one length-`N·D` (resp. `G·D`, `L·D`,
//! `N`) flattened column, so per-element mean/std land without
//! retaining all samples.
//!
//! Hot inner loop ([`super::loglik::dot`]) uses `wide::f32x8` SIMD; on
//! supporting CPUs, `RUSTFLAGS="-C target-cpu=native"` unlocks AVX2/AVX-512
//! codegen for an additional uplift.

use crate::cell_activity_graph_embedding::gene_chain_sampler::{
    build_gene_batch_cache, GeneGatedChainSampler,
};
use crate::cell_activity_graph_embedding::gene_gating::build_cell_activities;
use crate::cell_activity_graph_embedding_mcmc::args::CageMcmcArgs;
use crate::cell_activity_graph_embedding_mcmc::loglik::{
    build_bias_cache, init_const_dot_from_env, loglik_total, loglik_total_packed, loglik_with_bias,
    RowMajor,
};
use crate::cell_activity_graph_embedding_mcmc::model::{
    randn_matrix, randn_vector, softplus_floored, McmcState,
};
use crate::util::cell_pairs::SrtCellPairs;
use crate::util::common::*;
use crate::util::graph_coarsen::{graph_coarsen_multilevel, CoarsenConfig, SeedingParams};
use crate::util::metadata::{create_cage_metadata, RunInputs};
use crate::util::srt_pipeline::{preprocess_srt, SrtPreprocessConfig, SrtPreprocessed};

use data_beans_alg::hvg::select_hvg_streaming;
use data_beans_alg::random_projection::RandProjOps;
use graph_embedding_util::loss::{build_per_batch_cell_samplers, CellChainBatch, PbChainFilter};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_util::subset_rows;
use matrix_util::sparse_stat::SparseRunningStatistics;
use matrix_util::traits::RunningStatOps;
use mcmc_util::engine::elliptical_slice_step;
use nalgebra::{DMatrix, DVector};
use parquet::basic::Type as ParquetType;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

// The per-block ESS pattern intentionally calls `llik(&state.<block>)`
// once to refresh `cached_ll` and then passes `&llik` into
// `elliptical_slice_step` (where it's invoked many more times). Clippy's
// `redundant_closure_call` sees only the first explicit call site and
// fires a false positive; suppress at the function level.
#[allow(clippy::redundant_closure_call)]
pub fn fit_cage_mcmc(args: &CageMcmcArgs) -> anyhow::Result<()> {
    let c = &args.common;
    mkdir_parent(&c.out)?;

    let const_dot_on = init_const_dot_from_env();
    info!("cage-mcmc opts: const_dot_dispatch={}", const_dot_on);

    anyhow::ensure!(args.embedding_dim > 0, "embedding-dim must be > 0");
    anyhow::ensure!(args.n_samples > 0, "n-samples must be > 0");
    anyhow::ensure!(args.thin > 0, "thin must be > 0");
    anyhow::ensure!(args.resample_every > 0, "resample-every must be > 0");
    anyhow::ensure!(
        !args.chain_levels.is_empty(),
        "chain-levels must be non-empty"
    );
    anyhow::ensure!(
        args.prior_sd_cell > 0.0
            && args.prior_sd_gene > 0.0
            && args.prior_sd_gate > 0.0
            && args.prior_sd_bias > 0.0,
        "all --prior-sd-* must be > 0"
    );

    //////////////////////////////////////////////////////
    // Gene-name canonicalization                       //
    //////////////////////////////////////////////////////
    let peek_names = data_beans::convert::try_open_or_convert(&c.data_files[0])?.row_names()?;
    let feature_kind = args.gene_name_mode.resolve_kind(&peek_names);
    info!(
        "Gene-name canonicalization: {:?} (mode = {:?})",
        feature_kind, args.gene_name_mode
    );

    //////////////////////////////////////////////////////
    // 1-3. Load + KNN + batch effects                   //
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
    // 4. Coarsening                                     //
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
    anyhow::ensure!(
        per_batch.iter().any(|s| s.is_some()),
        "no batch retained any within-batch within-pb edges"
    );

    //////////////////////////////////////////////////////
    // 6. Activities + (gene, batch) cache               //
    //////////////////////////////////////////////////////
    info!("Computing per-gene cell activities...");
    let activities =
        build_cell_activities(&data_vec, &edges_owned, c.block_size, args.activity_norm)?;
    info!("Precomputing per-(gene, batch) positive distributions...");
    let cache = build_gene_batch_cache(&activities, &per_batch);
    info!(
        "Gene-batch cache: {} active (gene, batch) pairs",
        cache.n_active_pairs()
    );
    let mut trainable_genes: Vec<usize> = (0..n_genes)
        .filter(|&g| cache.entries[g].iter().any(|e| e.is_some()))
        .collect();
    drop(activities);

    // HVG subset (same convention as cage).
    let hvg_selected: Option<Vec<usize>> =
        if args.hvg.n_hvg > 0 || args.hvg.feature_list_file.is_some() {
            let hvg = select_hvg_streaming(
                &data_vec,
                (args.hvg.n_hvg > 0).then_some(args.hvg.n_hvg),
                args.hvg.feature_list_file.as_deref(),
                c.block_size,
            )?;
            let kept: HashSet<usize> = hvg.selected_indices.iter().copied().collect();
            trainable_genes.retain(|g| kept.contains(g));
            info!(
                "HVG subset: {} trainable genes after intersection",
                trainable_genes.len()
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
    // 7. Allocate state (DMatrix / DVector)             //
    //////////////////////////////////////////////////////
    let mut rng = SmallRng::seed_from_u64(c.seed);
    let d = args.embedding_dim;
    let mut state = McmcState {
        e_cell: randn_matrix(&mut rng, n_cells, d, args.prior_sd_cell),
        e_gene: randn_matrix(&mut rng, n_genes, d, args.prior_sd_gene),
        gamma: DMatrix::zeros(n_chain_levels, d), // pre-softplus zero → softplus_floored ≈ ln 2
        b_cell: DVector::zeros(n_cells),
    };
    info!(
        "Allocated MCMC state: e_cell [{}×{}], e_gene [{}×{}], γ [{}×{}], b_cell [{}]",
        n_cells, d, n_genes, d, n_chain_levels, d, n_cells
    );

    //////////////////////////////////////////////////////
    // 8. Sampler + minibatch closure                    //
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
    let sampler_ref = &sampler;

    // One minibatch resample. Returns `Vec<chunk>` where each chunk is a
    // `Vec<(gene_id, CellChainBatch)>` — the unit of rayon parallelism
    // inside `loglik_total`.
    let resample = |rng_master: &mut SmallRng,
                    trainable_genes: &[usize]|
     -> Vec<Vec<(usize, CellChainBatch)>> {
        let mut perm: Vec<usize> = trainable_genes.to_vec();
        perm.shuffle(rng_master);
        // One seed drawn from the master RNG; each chunk derives its own
        // RNG deterministically from (seed, chunk_idx, gene). Keeps the
        // parallel iterator borrow-free.
        let resample_seed: u64 = rng_master.random();
        let chunks: Vec<Vec<(usize, CellChainBatch)>> = perm
            .par_chunks(args.gene_batch_size.max(1))
            .enumerate()
            .map(|(ci, chunk)| {
                let mut local: Vec<(usize, CellChainBatch)> = Vec::new();
                for (i, &g) in chunk.iter().enumerate() {
                    let mut rng = SmallRng::seed_from_u64(
                        resample_seed
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add((ci as u64).wrapping_mul(1_000_003))
                            .wrapping_add(g as u64)
                            .wrapping_add(i as u64),
                    );
                    for b in 0..n_batches {
                        if let Some((cb, _)) = sampler_ref.sample(g, b, &mut rng) {
                            local.push((g, cb));
                        }
                    }
                }
                local
            })
            .collect();
        chunks
    };

    //////////////////////////////////////////////////////
    // 9. Streaming posterior moments via matrix-util's   //
    //    SparseRunningStatistics: treat each flattened   //
    //    sample as one "column" of length N·D (resp. G·D)//
    //    so we get per-element mean/std without storing  //
    //    all samples.                                    //
    //////////////////////////////////////////////////////
    let mut stat_e_cell = SparseRunningStatistics::<f32>::new(n_cells * d);
    let mut stat_e_gene = SparseRunningStatistics::<f32>::new(n_genes * d);
    let mut stat_gamma = SparseRunningStatistics::<f32>::new(n_chain_levels * d);
    let mut stat_b_cell = SparseRunningStatistics::<f32>::new(n_cells);

    let mut loglik_rows: Vec<LoglikRow> = Vec::new();

    //////////////////////////////////////////////////////
    // 10. MCMC loop                                     //
    //////////////////////////////////////////////////////
    let total = args.warmup + args.n_samples * args.thin;
    info!(
        "Running ESS: warmup={}, n_samples={}, thin={}, total sweeps={}",
        args.warmup, args.n_samples, args.thin, total
    );

    let mut chunks = resample(&mut rng, &trainable_genes);
    let mut cached_ll = loglik_total(
        &state.e_cell,
        &state.e_gene,
        &state.gamma,
        &state.b_cell,
        &chunks,
    );
    info!("Initial log-likelihood: {:.4e}", cached_ll);

    for sweep in 0..total {
        if sweep > 0 && sweep % args.resample_every == 0 {
            chunks = resample(&mut rng, &trainable_genes);
        }

        // ESS's bracket-shrinkage loop assumes `lnpdf(current) == cached_ll`.
        // With rayon parallel f32 summation, the same (state, chunks) pair
        // can yield log-lik values differing by ~1e-3 across calls; when
        // `ln(U)` is close to 0 the slice threshold falls between those
        // values and bracket shrinkage can fail to terminate at φ=0. Each
        // block refreshes `cached_ll` from the same closure ESS will call.

        // --- Block 1: e_cell --- (e_gene + γ fixed, packed once)
        // ll discarded — next block refreshes cached_ll from its own closure.
        let (new_e_cell, _) = {
            let prior = randn_matrix(&mut rng, n_cells, d, args.prior_sd_cell);
            let e_gene_rm = RowMajor::from_dmatrix(&state.e_gene);
            let gated_gamma = softplus_floored(&state.gamma);
            let llik = |cand: &DMatrix<f32>| {
                let e_cell_rm = RowMajor::from_dmatrix(cand);
                loglik_total_packed(&e_cell_rm, &e_gene_rm, &gated_gamma, &state.b_cell, &chunks)
            };
            cached_ll = llik(&state.e_cell);
            elliptical_slice_step(&state.e_cell, &prior, &llik, cached_ll, &mut rng)
        };
        state.e_cell = new_e_cell;

        // --- Block 2: e_gene --- (e_cell + γ fixed, packed once)
        let (new_e_gene, _) = {
            let prior = randn_matrix(&mut rng, n_genes, d, args.prior_sd_gene);
            let e_cell_rm = RowMajor::from_dmatrix(&state.e_cell);
            let gated_gamma = softplus_floored(&state.gamma);
            let llik = |cand: &DMatrix<f32>| {
                let e_gene_rm = RowMajor::from_dmatrix(cand);
                loglik_total_packed(&e_cell_rm, &e_gene_rm, &gated_gamma, &state.b_cell, &chunks)
            };
            cached_ll = llik(&state.e_gene);
            elliptical_slice_step(&state.e_gene, &prior, &llik, cached_ll, &mut rng)
        };
        state.e_gene = new_e_gene;

        // --- Block 3: γ --- (e_cell + e_gene fixed, packed once)
        let (new_gamma, _) = {
            let prior = randn_matrix(&mut rng, n_chain_levels, d, args.prior_sd_gate);
            let e_cell_rm = RowMajor::from_dmatrix(&state.e_cell);
            let e_gene_rm = RowMajor::from_dmatrix(&state.e_gene);
            let llik = |cand: &DMatrix<f32>| {
                let gated_gamma = softplus_floored(cand);
                loglik_total_packed(&e_cell_rm, &e_gene_rm, &gated_gamma, &state.b_cell, &chunks)
            };
            cached_ll = llik(&state.gamma);
            elliptical_slice_step(&state.gamma, &prior, &llik, cached_ll, &mut rng)
        };
        state.gamma = new_gamma;

        // --- Block 4: b_cell (BiasCache fast path) ---
        // The b_cell ESS step sees fixed (e_cell, e_gene, γ). Precompute
        // the interaction terms proj_u·proj_v / proj_u·proj_w_k once;
        // each bracket eval is then just bias add + log_sigmoid.
        let (new_b_cell, ll4) = {
            let prior = randn_vector(&mut rng, n_cells, args.prior_sd_bias);
            let cache = build_bias_cache(&state.e_cell, &state.e_gene, &state.gamma, &chunks);
            let llik = |cand: &DVector<f32>| loglik_with_bias(&cache, cand);
            cached_ll = llik(&state.b_cell);
            elliptical_slice_step(&state.b_cell, &prior, &llik, cached_ll, &mut rng)
        };
        state.b_cell = new_b_cell;
        cached_ll = ll4;

        let is_warmup = sweep < args.warmup;
        let post_idx = sweep.wrapping_sub(args.warmup);
        let sample_idx = if is_warmup {
            None
        } else if post_idx % args.thin == 0 {
            Some(post_idx / args.thin)
        } else {
            None
        };
        loglik_rows.push(LoglikRow {
            sweep,
            loglik: cached_ll,
            is_warmup,
            sample_idx,
        });

        if let Some(_idx) = sample_idx {
            // Snapshot post-softplus γ so the posterior summary is in
            // the same space as cage's `level_dim_gates`.
            let gated = softplus_floored(&state.gamma);
            // DMatrix is column-major; `as_slice()` gives the flat
            // column-major buffer SparseRunningStatistics expects.
            stat_e_cell.add_dense_column(state.e_cell.as_slice());
            stat_e_gene.add_dense_column(state.e_gene.as_slice());
            stat_gamma.add_dense_column(gated.as_slice());
            stat_b_cell.add_dense_column(state.b_cell.as_slice());
        }

        if sweep % 10 == 0 || sweep + 1 == total {
            info!(
                "sweep {:>5} {} loglik = {:.4e}",
                sweep,
                if is_warmup { "[warmup]" } else { "        " },
                cached_ll
            );
        }
    }

    //////////////////////////////////////////////////////
    // 11. Outputs                                       //
    //////////////////////////////////////////////////////
    info!("Writing cage-mcmc outputs...");

    let prefix: &str = &c.out;
    let col_names = embedding_col_names(d);
    let level_names: Vec<Box<str>> = args
        .chain_levels
        .iter()
        .map(|&lvl| format!("level_{lvl}").into_boxed_str())
        .collect();

    // Reshape `Vec<f32>` of length `rows*cols` back to a `[rows, cols]`
    // column-major DMatrix (matches the `as_slice()` layout we ingested).
    let to_mat =
        |v: Vec<f32>, rows: usize, cols: usize| -> Mat { Mat::from_column_slice(rows, cols, &v) };

    // Cell embedding mean / std
    let e_cell_mean = to_mat(stat_e_cell.mean(), n_cells, d);
    let e_cell_std = to_mat(stat_e_cell.std(), n_cells, d);
    e_cell_mean.to_parquet_with_names(
        &format!("{prefix}.cell_embedding.mean.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&col_names),
    )?;
    e_cell_std.to_parquet_with_names(
        &format!("{prefix}.cell_embedding.std.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&col_names),
    )?;

    // Cell bias mean / std (Nx1)
    let b_cell_mean = Mat::from_column_slice(n_cells, 1, &stat_b_cell.mean());
    let b_cell_std = Mat::from_column_slice(n_cells, 1, &stat_b_cell.std());
    b_cell_mean.to_parquet_with_names(
        &format!("{prefix}.cell_bias.mean.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&[Box::from("b_cell")]),
    )?;
    b_cell_std.to_parquet_with_names(
        &format!("{prefix}.cell_bias.std.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&[Box::from("b_cell")]),
    )?;

    // Feature embedding mean / std (HVG-subset on output).
    let e_gene_mean_full = to_mat(stat_e_gene.mean(), n_genes, d);
    let e_gene_std_full = to_mat(stat_e_gene.std(), n_genes, d);
    let (e_feature_mean, e_feature_std, feature_names_out): (Mat, Mat, Vec<Box<str>>) =
        if let Some(sel) = hvg_selected.as_ref() {
            let mean = subset_rows(&e_gene_mean_full, sel.iter().copied())?;
            let std = subset_rows(&e_gene_std_full, sel.iter().copied())?;
            let names: Vec<Box<str>> = sel.iter().map(|&i| gene_names[i].clone()).collect();
            (mean, std, names)
        } else {
            (e_gene_mean_full, e_gene_std_full, gene_names.clone())
        };

    e_feature_mean.to_parquet_with_names(
        &format!("{prefix}.feature_embedding.mean.parquet"),
        (Some(&feature_names_out), Some("feature")),
        Some(&col_names),
    )?;
    e_feature_std.to_parquet_with_names(
        &format!("{prefix}.feature_embedding.std.parquet"),
        (Some(&feature_names_out), Some("feature")),
        Some(&col_names),
    )?;

    // Level-dim gates (post-softplus) mean / std
    let gamma_mean = to_mat(stat_gamma.mean(), n_chain_levels, d);
    let gamma_std = to_mat(stat_gamma.std(), n_chain_levels, d);
    gamma_mean.to_parquet_with_names(
        &format!("{prefix}.level_dim_gates.mean.parquet"),
        (Some(&level_names), Some("level")),
        Some(&col_names),
    )?;
    gamma_std.to_parquet_with_names(
        &format!("{prefix}.level_dim_gates.std.parquet"),
        (Some(&level_names), Some("level")),
        Some(&col_names),
    )?;

    // Log-likelihood trace
    write_loglik_trace(&format!("{prefix}.loglik_trace.parquet"), &loglik_rows)?;

    // Thinned trace: post-warmup recorded samples only. Subset of
    // `loglik_rows` filtered by `sample_idx.is_some()`. Per-element
    // traces would be huge; the mean/std parquets already carry the
    // per-element posterior summary.
    if !args.no_trace {
        let trace: Vec<&LoglikRow> = loglik_rows
            .iter()
            .filter(|r| r.sample_idx.is_some())
            .collect();
        write_thinned_trace(&format!("{prefix}.trace.parquet"), &trace)?;
    }

    // Metadata — point standard cage fields at `.mean.parquet`.
    {
        let coord_file_str = c.coord_files_joined();
        let mut meta = create_cage_metadata(
            &RunInputs {
                prefix,
                data_files: &c.data_files,
                coord_file: coord_file_str.as_deref(),
                coord_columns: &coordinate_names,
                n_cells,
                n_genes,
                n_edges,
                k: d,
            },
            batch_db.is_some(),
            None,
        );
        meta.command = "cage-mcmc".to_string();
        meta.outputs.cell_embedding = Some(format!("{prefix}.cell_embedding.mean.parquet"));
        meta.outputs.cell_bias = Some(format!("{prefix}.cell_bias.mean.parquet"));
        meta.outputs.feature_embedding = Some(format!("{prefix}.feature_embedding.mean.parquet"));
        meta.outputs.level_dim_gates = Some(format!("{prefix}.level_dim_gates.mean.parquet"));
        meta.outputs.gene_bias = None; // cage-mcmc has no b_gene
        meta.outputs.scores = Some(format!("{prefix}.loglik_trace.parquet"));
        let meta_path = std::path::PathBuf::from(format!("{prefix}.metadata.json"));
        meta.write(&meta_path)?;
        info!("Wrote {}", meta_path.display());
    }

    info!("Done");
    Ok(())
}

////////////////////////////////////////////////////////////////
//                                                            //
// Helpers                                                    //
//                                                            //
////////////////////////////////////////////////////////////////

fn embedding_col_names(d: usize) -> Vec<Box<str>> {
    (0..d).map(|i| format!("e{i}").into_boxed_str()).collect()
}

////////////////////////////////////////////////////////////////
//                                                            //
// Parquet writers for the loglik trace and sample trace      //
//                                                            //
////////////////////////////////////////////////////////////////

/// One log-likelihood trace row. `sample_idx` is `Some(_)` only for
/// post-warmup sweeps that pass the thinning filter; those rows also
/// feed the Welford accumulators that produce the posterior parquets.
struct LoglikRow {
    sweep: usize,
    loglik: f32,
    is_warmup: bool,
    sample_idx: Option<usize>,
}

fn write_loglik_trace(path: &str, rows: &[LoglikRow]) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    let n = rows.len();
    let sweep: Vec<i32> = rows.iter().map(|r| r.sweep as i32).collect();
    let loglik: Vec<f64> = rows.iter().map(|r| r.loglik as f64).collect();
    let is_warmup: Vec<i32> = rows.iter().map(|r| r.is_warmup as i32).collect();

    let col_names: Vec<Box<str>> = vec!["sweep".into(), "loglik".into(), "is_warmup".into()];
    let col_types = vec![ParquetType::INT32, ParquetType::DOUBLE, ParquetType::INT32];
    let writer = ParquetWriter::new(
        path,
        (n, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("step"),
    )?;
    let row_names = writer.row_names_vec();
    let mut w = writer.get_writer()?;
    let mut rg = w.next_row_group()?;
    parquet_add_bytearray(&mut rg, row_names)?;
    parquet_add_numeric_column(&mut rg, &sweep)?;
    parquet_add_numeric_column(&mut rg, &loglik)?;
    parquet_add_numeric_column(&mut rg, &is_warmup)?;
    rg.close()?;
    w.close()?;
    Ok(())
}

fn write_thinned_trace(path: &str, rows: &[&LoglikRow]) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    let n = rows.len();
    let sweep: Vec<i32> = rows.iter().map(|r| r.sweep as i32).collect();
    let loglik: Vec<f64> = rows.iter().map(|r| r.loglik as f64).collect();
    let sample_idx: Vec<i32> = rows
        .iter()
        .map(|r| r.sample_idx.expect("filtered to Some") as i32)
        .collect();

    let col_names: Vec<Box<str>> = vec!["sweep".into(), "loglik".into(), "sample_idx".into()];
    let col_types = vec![ParquetType::INT32, ParquetType::DOUBLE, ParquetType::INT32];
    let writer = ParquetWriter::new(
        path,
        (n, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("step"),
    )?;
    let row_names = writer.row_names_vec();
    let mut w = writer.get_writer()?;
    let mut rg = w.next_row_group()?;
    parquet_add_bytearray(&mut rg, row_names)?;
    parquet_add_numeric_column(&mut rg, &sweep)?;
    parquet_add_numeric_column(&mut rg, &loglik)?;
    parquet_add_numeric_column(&mut rg, &sample_idx)?;
    rg.close()?;
    w.close()?;
    Ok(())
}
