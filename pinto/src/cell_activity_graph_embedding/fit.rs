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
//! collapse + SVD + block Gibbs (sampled arm only)       selection::select_features -> pip
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
//! # Two gate arms (`--gate-mode`)
//!
//! Both fill geu's one `GateKind::Identity` slot, and an installed `pip` takes
//! that slot over completely (`model/gate.rs:503-511`), so they are mutually
//! exclusive by construction.
//!
//! `sampled` (default) runs cage's block Gibbs over super-cell counts and installs
//! the result as a per-epoch Bernoulli mask. It scans every gene on every sweep,
//! so it costs about a third more wall-clock than the alternative — and measured
//! better, which is why it is the default.
//!
//! `learned` is the one `senna bge` uses: a variational spike-and-slab gate
//! `σ((S + IBP ladder)/τ)` fit by SGD alongside the embedding. Nothing scans the
//! count matrix. Because `σ(·)` multiplied the loading on every step it IS part
//! of the fitted coefficient, so `materialize_e_feat` bakes it into the shipped
//! dictionary and `α` ships as `feature_selection.parquet`. Prefer it when
//! runtime matters more than the last of the resolution.
//!
//! An earlier learned arm selected nothing there, every inclusion probability
//! staying above 0.95. That was the INCLUSION KL it carried at the time: a
//! penalty with a free coefficient under an NCE objective, needing λ≈1000 here
//! against 1/1024 in geu. `c802f7a3` replaced it with the truncated-IBP ladder,
//! which has nothing to calibrate, and both arms now draw selection from that
//! same ladder. Do not cite the old measurement against the current gate.
//!
//! Under `sampled` there are two cadences, and they are different things:
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
//! On that arm `e_feat` is randomly initialized and the selection enters as the
//! gate. Deliberately NOT initialized to `E[z*beta]`: that already carries the
//! sampler's shrinkage, so gating it would apply the same selection twice. The
//! two are alternatives, not complements — `feature_posterior_mean.parquet`
//! ships `E[z*beta]` for downstream use, and the raw embedding ships as-is
//! because a dropout keep-rate is not a coefficient.

use crate::cell_activity_graph_embedding::args::{CellActivityGraphEmbeddingArgs, GateMode};
use crate::cell_activity_graph_embedding::gene_chain_sampler::{
    build_gene_batch_cache, GeneGatedChainSampler,
};
use crate::cell_activity_graph_embedding::gene_gating::build_cell_activities;
use crate::cell_activity_graph_embedding::loss::{cage_nce_loss_per_level, CageLossOut};
use crate::cell_activity_graph_embedding::pair_projection::{
    project_pairs, PairBatchDivisor, PairLatent, PairProjectionArgs, ProjectionArgs,
};
use crate::link_community::profiles::{
    compute_propensity_and_gene_community_stat, PropensityReportConfig,
};
use crate::util::cell_pairs::SrtCellPairs;
use crate::util::common::*;
use crate::util::graph_coarsen::{
    graph_coarsen_multilevel, CoarsenConfig, DcPoissonConfig, SeedingParams,
};
use crate::util::metadata::{create_cage_metadata, RunInputs, SpliceTrackInfo, DELTA_BASE_SPLICED};
use crate::util::score_trace::{write_score_trace, ScoreEntry};
use crate::util::srt_pipeline::{
    preprocess_srt, topology_graph, GeneAxisMode, SrtPreprocessConfig, SrtPreprocessed,
};

use candle_util::candle_core::Tensor;
// `Optimizer` is what puts `AdamW::new` in scope; backward and the step run
// separately (`loss.backward()` then `clip_and_step_dense`) so the phase timers
// can attribute them apart.
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use candle_util::vae::{clip_and_step_dense, PhaseTimers};
use data_beans_alg::gene_weighting::save_fisher_weights;
use data_beans_alg::hvg::select_hvg_streaming;
use data_beans_alg::random_projection::RandProjOps;
use graph_embedding_util::embedding_col_names;
use graph_embedding_util::loss::{build_per_batch_cell_samplers, embedding_ridge, PbChainFilter};
use graph_embedding_util::model::{
    FeatureGateSpec, GateKind, JointEmbedModel, ModelArgs, ModelInit,
};
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
    // Each gate arm has knobs the other one cannot act on. Setting one for the
    // wrong arm is a mistake worth reporting, not silently absorbing — a flag
    // that is read and ignored is indistinguishable from one that did not work.
    // Keyed on "differs from the default" so an explicit default still passes.
    match args.gate_mode {
        GateMode::Learned => anyhow::ensure!(
            args.selection_sweeps == 10
                && args.selection_refresh_sweeps == 10
                && args.selection_refresh_epochs == 5,
            "--selection-sweeps / --selection-refresh-sweeps / --selection-refresh-epochs \
             configure the Gibbs sampler, which --gate-mode learned does not run. \
             Pass --gate-mode sampled to use them."
        ),
        GateMode::Sampled => {
            anyhow::ensure!(
                args.feature_gate_temp == 1.0,
                "--feature-gate-temp sharpens the LEARNED gate's sigmoid, and \
                 --gate-mode sampled installs a Gibbs-sampled mask instead, which \
                 takes the gate slot over entirely. Pass --gate-mode learned to use it."
            );
            // Same rule for the IBP ladder: under `sampled` no learned gate is
            // installed, so the ladder has no logits to tilt and this flag reaches
            // nothing at all. Without this the run completes and produces output
            // byte-identical to the default, which reads as "the gate cannot be
            // made to select" rather than "the flag did nothing". (The sampler has
            // its own IBP — `posterior::hyper::ibp_pi0` — driven by the
            // --selection-* flags.)
            anyhow::ensure!(
                args.gate_ibp_alpha.is_none(),
                "--gate-ibp-alpha tilts the LEARNED gate's inclusion ladder, and \
                 --gate-mode sampled installs a Gibbs-sampled mask instead, which \
                 draws from its own IBP under the --selection-* flags. Pass \
                 --gate-mode learned to use it."
            );
        }
    }
    if let Some(a) = args.gate_ibp_alpha {
        anyhow::ensure!(
            a.is_finite() && a > 0.0,
            "--gate-ibp-alpha must be finite and > 0 (got {a}); alpha sets \
             v = alpha/(alpha+1), the ladder's per-dim inclusion ratio, which is \
             only a probability for positive alpha."
        );
    }

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
        spatial_graph,
        edge_source,
        cell_proj: shared_cell_proj,
        gene_axis,
        row_weights: fisher_weights,
        row_stats,
        gene_weights: gene_fisher_weights,
        gene_stats: _,
        n_cells,
        n_rows,
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        fisher_weights: !args.no_fisher_weights,
        batch_effects: true,
        gene_axis: GeneAxisMode::Strict,
        cell_projection: true,
        feature_kind: Some(feature_kind),
    })?;

    let has_coords = c.has_coordinates();
    let cell_names = data_vec.column_names()?;
    let row_names = data_vec.row_names()?;

    // What a ROW means, decided once. On a splice-channelized matrix (faba's
    // `{gene}/count/{spliced,unspliced}`) two rows are one gene, and everything
    // gene-side below folds through this; on any other matrix the axis is the
    // identity and every fold is a pass-through. The two are NOT interchangeable
    // names for one number, so `n_rows` and `n_genes` stay apart from here on.
    let gene_axis = gene_axis.expect("GeneAxisMode::Strict must yield Some");
    let n_genes = gene_axis.n_genes();
    let gene_names: Vec<Box<str>> = gene_axis.gene_names().to_vec();

    // Persist the NB-Fisher precision weights (when computed) so downstream
    // tools can reload them, matching the senna/chickpea convention. These are
    // per-ROW NB precisions, consumed by the projection and the Poisson
    // refinement, both of which read the matrix — so they keep the row names.
    if let Some(w) = fisher_weights.as_ref() {
        save_fisher_weights(&c.out, w, &row_names)?;
    }

    // Fisher weights, on the axis each consumer actually indexes.
    //
    // `fisher_weights` is per ROW and stays that way: the projection and the
    // degree-corrected Poisson refinement both read the matrix. The TRAINING
    // LOOP does not — it weights a per-GENE loss and indexes by gene id — so on
    // a channelized matrix `w[g]` would hand gene `g` an unrelated row's
    // precision, silently, because `n_genes < n_rows` means it never goes out
    // of bounds.
    //
    // Both vectors come from one streaming pass, and the gene-axis one is
    // computed from statistics folded INSIDE that pass rather than from the row
    // statistics folded afterwards. That distinction is not cosmetic: `s2` of a
    // sum is not the sum of `s2`, so a post-hoc fold loses the cross term
    // between a gene's two tracks and hands the dispersion trend a variance
    // that is too small exactly where the two tracks covary most.

    let srt_cell_pairs = SrtCellPairs::with_graph_and_source(
        &data_vec,
        &coordinates,
        &graph,
        edge_source.as_deref(),
    );
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
    if gene_axis.is_channelized() {
        info!(
            "{} cells, {} genes ({} channel rows), {} edges",
            n_cells, n_genes, n_rows, n_edges
        );
    } else {
        info!("{} cells, {} genes, {} edges", n_cells, n_genes, n_edges);
    }

    // Per-ROW count totals, streamed ONCE. Two consumers need them and both are
    // whole-matrix passes: the splice go/no-go here, and the pair projection's
    // partition at the end of the run. The counts do not change in between —
    // batch effects and QC are both already applied — so a second pass would
    // re-read every column of the zarr to rebuild the same vector.
    // Off the streaming pass when it ran; `sum()` IS the per-row total. Only
    // `--no-fisher-weights` leaves it absent, and that is the one case worth a
    // second read.
    let row_totals: Vec<f64> = match row_stats {
        Some(st) => st.sum().iter().map(|&x| f64::from(x)).collect(),
        None => crate::link_community::profiles::compute_row_totals(&data_vec, c.block_size)?,
    };
    let gene_totals = gene_axis.pool_totals(&row_totals);

    // The go/no-go for everything a velocity contrast would be built on, taken
    // BEFORE the expensive fit so a thin input is caught by reading the log
    // rather than by reading a model. Only a channelized input has a contrast to
    // report on at all.
    let splice_report = gene_axis
        .report_delta_identifiability(&row_totals)
        .map(|r| SpliceTrackInfo {
            n_rows,
            n_delta_identified: r.n_identified,
            nascent_count_fraction: r.nascent_fraction,
            delta_base: DELTA_BASE_SPLICED.to_string(),
            // Filled in after training, once it is known whether any refresh
            // ran and what the sampler actually saw.
            delta_from_refresh: None,
            delta_median_counts: None,
            delta_counts_per_pseudobulk: None,
        });

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
    // model may use — matching `senna bge` and `senna gem`.
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
        // Selection ranks ROWS, and the projection it weights reads rows, so
        // both stay on the row axis. What cannot stay there is a gene: picking a
        // gene's spliced row without its unspliced one would weight half a gene
        // into the basis the coarsening hierarchy is cut from, so the weights are
        // promoted to whole genes. `--n-hvg N` therefore still counts N ROWS —
        // between N/2 and N genes on a channelized matrix — which is what the
        // realized count below reports.
        let mut w = hvg.row_weights(n_rows);
        let n_weighted = gene_axis.promote_row_weights(&mut w);
        info!(
            "HVG-weighted projection: {} of {} genes weighted ({} of {} rows); \
             all stay on the trained axis",
            n_weighted,
            n_genes,
            w.iter().filter(|&&x| x > 0.0).count(),
            n_rows
        );
        Some(w)
    } else {
        None
    };

    let cell_proj = match hvg_weights.as_deref() {
        // HVG weighting makes this a genuinely different projection, so the
        // shared one cannot stand in for it.
        Some(w) => data_vec.project_columns_weighted(c.proj_dim, c.block_size, batch_arg, w)?,
        None => shared_cell_proj
            .expect("cell_projection = true above must yield a projection"),
    };
    let topology = topology_graph(&graph, &spatial_graph);
    let ml = graph_coarsen_multilevel(
        topology,
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
                num_genes: n_rows,
            }),
        },
    );

    // Collapse cells into super-cells, then fit the gene selection against
    // them — the whole stage runs on the SAMPLED arm only.
    //
    // The collapse and its shared SVD exist to give the Gibbs a frozen side, so
    // under `--gate-mode learned` nothing consumes them and they were costing a
    // full pseudobulk build plus an [n_genes x D] SVD per run purely to emit a
    // diagnostic. `pseudobulk_cells.parquet` is therefore a SAMPLED-ARM output;
    // `pinto cage --help` says so.
    // The most recent refreshed selection, kept because its `delta` is the only
    // one commensurate with the shipped gene embedding — see the refresh below.
    let mut latest_refreshed: Option<crate::cell_activity_graph_embedding::selection::Selection> =
        None;
    let mut sampled_selection = if args.gate_mode == GateMode::Learned {
        None
    } else {
        let pb = crate::cell_activity_graph_embedding::selection::build_pseudobulks(
            crate::cell_activity_graph_embedding::selection::PseudobulkArgs {
                data: &data_vec,
                all_cell_labels: &ml.all_cell_labels,
                graph: &graph,
                cell_features: &cell_proj.proj,
                embedding_dim: args.embedding_dim,
                gene_axis: &gene_axis,
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

        {
            // Sample which (gene, community) pairs are included.
            let (sel, sel_state, sel_z) =
                crate::cell_activity_graph_embedding::selection::select_features(
                    &pb,
                    &gene_axis,
                    args.embedding_dim,
                    &crate::cell_activity_graph_embedding::selection::SelectArgs {
                        sweeps: args.selection_sweeps,
                        burnin: args.selection_sweeps / 2,
                        seed: c.seed,
                        nested_delta: !args.independent_delta_gate,
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

            Some((sel, sel_state, sel_z))
        }
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
    let activities = build_cell_activities(
        &data_vec,
        &edges_owned,
        c.block_size,
        args.activity_norm,
        &gene_axis,
    )?;
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

    // Positives drawn per (gene, batch), one scalar for every gene.
    // `--positives-per-epoch` overrides `--per-gene-batch` with an absolute
    // total. See `GeneGatedChainSampler::batch_size` for why the budget is not
    // split per-gene.
    let positives_per_gene = match args.positives_per_epoch {
        Some(total) => (total / (trainable_genes.len() * n_batches).max(1)).max(1),
        None => args.per_gene_batch,
    };
    info!(
        "Positive budget: {} per (gene, batch), {} genes x {} batch(es) = {} per epoch{}",
        positives_per_gene,
        trainable_genes.len(),
        n_batches,
        positives_per_gene * trainable_genes.len() * n_batches,
        if args.positives_per_epoch.is_some() {
            " (--positives-per-epoch)"
        } else {
            " (--per-gene-batch)"
        }
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
    // Install the feature gate. Both arms fill geu's one `GateKind::Identity`
    // slot, and an installed `pip` takes the slot over completely
    // (`model/gate.rs:503-511`), so they are mutually exclusive by construction.
    match &sampled_selection {
        // The cold sampler's `pip`. `resample_gate_mask` is a no-op without it,
        // so this install is what turns per-epoch spike sampling on at all.
        Some((selection, _, _)) => {
            let pip_t =
                Tensor::from_vec(selection.pip.clone(), (n_genes, args.embedding_dim), &dev)?;
            model.install_gate_pip(GateKind::Identity, &pip_t)?;
        }
        // Variational spike-and-slab: SGD fits `σ((S + IBP ladder)/τ)` alongside
        // the embedding. Sparsity comes from the ladder's structure, not from a
        // weighted penalty — see `geu::model::ibp_gate_logit_bias`.
        None => {
            model.enable_feature_gate(
                FeatureGateSpec {
                    temperature: args.feature_gate_temp,
                    ibp_alpha: args.gate_ibp_alpha,
                },
                &varmap,
                &dev,
            )?;
            info!(
                "Learned feature gate enabled (temperature = {:.3}); no count-scan sampler runs",
                args.feature_gate_temp
            );
        }
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
        batch_size: positives_per_gene,
        n_negatives: args.n_negatives,
    };

    let mut score_trace: Vec<ScoreEntry> = Vec::new();
    let mut rng_master = SmallRng::seed_from_u64(c.seed);
    // Wall-clock split of the SGD loop, to rank optimization work by evidence
    // rather than guess. Accumulated across all epochs; logged once at the end.
    let mut timers = PhaseTimers::default();
    // Steps the optimizer refused because the global gradient norm was not
    // finite. Reported at the end: a high count means the fit did not train.
    let mut skipped_steps: usize = 0;

    // One bar across the WHOLE training phase, not one per epoch: the useful
    // question while waiting is "how long until training is done", and a bar
    // that restarts five times cannot answer it. The total is the real step
    // count — chunks per epoch is `ceil(genes_visited / --gene-batch-size)` —
    // so the ETA is derived from observed step rate rather than guessed.
    // `--genes-per-epoch` shortens every epoch equally, so this stays exact.
    // Early exit (convergence or Ctrl-C) simply finishes the bar short.
    let genes_per_epoch_actual = if args.genes_per_epoch > 0 {
        args.genes_per_epoch.min(trainable_genes.len())
    } else {
        trainable_genes.len()
    };
    let steps_per_epoch = genes_per_epoch_actual.div_ceil(args.gene_batch_size);
    let total_steps = args.epochs * steps_per_epoch.max(1);
    let train_bar = new_progress_bar(total_steps as u64).with_message("training steps");

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
        if let Some((_, sel_state, sel_z)) = sampled_selection.as_mut() {
            if epoch > 0
                && args.selection_refresh_epochs > 0
                && epoch % args.selection_refresh_epochs == 0
            {
                let e_cell_now = tensor_to_mat(&model.e_cell)?;
                let e_flat = sel_state.frozen_e_from_cells(&e_cell_now);
                let (sel_new, warm_new) = sel_state.sample(
                    &e_flat,
                    &crate::cell_activity_graph_embedding::selection::SelectArgs {
                        sweeps: args.selection_refresh_sweeps,
                        burnin: args.selection_refresh_sweeps / 2,
                        seed: c.seed.wrapping_add(epoch as u64),
                        nested_delta: !args.independent_delta_gate,
                    },
                    std::mem::take(sel_z),
                );
                *sel_z = warm_new;
                // Keep it: the frozen side this saw is the LIVE `e_cell` folded
                // into super-cells, so its `delta` is in the same D-dim frame as
                // the gene embedding the run ships. The cold chain's is not — it
                // was fit against an SVD of pseudobulk log-counts, a frame the
                // model never uses — so a delta dictionary read off the cold run
                // would be mixed into `e_feat`'s frame and mean nothing.
                let sel_new = latest_refreshed.insert(sel_new);
                let pip_t =
                    Tensor::from_vec(sel_new.pip.clone(), (n_genes, args.embedding_dim), &dev)?;
                model.install_gate_pip(GateKind::Identity, &pip_t)?;
                let m: f32 = sel_new.pip.iter().sum::<f32>() / sel_new.pip.len() as f32;
                info!("epoch {epoch}: pip refreshed against e_cell, mean = {m:.4}");
            }
            // Draws this epoch's hard `z ~ Bern(pip)`. A no-op without a pip,
            // and meaningless under the learned gate, which carries the
            // differentiable expectation `E[z·β] = α·E` instead of a draw.
            model.resample_gate_mask()?;
        }
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
        // Genes this epoch will actually visit — the KL amortization divides by
        // it (see the `gate_kl` block). NOTE this does NOT leave the prior's
        // strength alone: shortening the epoch shrinks the data total while the
        // per-epoch KL total stays put, so `--genes-per-epoch 5000` on a 36k
        // axis pulls ~7x harder. Deferred, with the `--chain-levels` gap.
        debug_assert_eq!(perm.len(), genes_per_epoch_actual);
        // The KL weight has to divide by the units the DATA TERM actually
        // carries. That term is Fisher-weighted (`per_level_gl.broadcast_mul`
        // below) before `sum_all()`, so its mass is `Σ_g w_g`, not the gene
        // count — dividing by the count would inflate the prior's share by
        // `1/mean(w_g)`, which varies with the dataset's dispersion profile and
        // would make the same `--gate-kl-weight` mean different things on
        // different panels. Falls back to the plain count when weighting is off.
        let epoch_mass: f64 = match gene_fisher_weights.as_ref() {
            Some(w) => perm
                .iter()
                .map(|&g| f64::from(w[g]))
                .sum::<f64>()
                .max(1e-12),
            None => genes_per_epoch_actual as f64,
        };
        for chunk in perm.chunks(args.gene_batch_size) {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            // (a) Parallel sampling — pure CPU, no candle.
            let t_sample = std::time::Instant::now();
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

            // Attribute sampling BEFORE the early-out: a chunk whose genes all
            // failed to sample still paid for the rayon draw, and dropping it
            // under-reports the sampler exactly where it did the most futile
            // work. Advance the bar too, or its total (which assumes every
            // chunk steps) leaves it short and the ETA running long.
            timers.precompute += t_sample.elapsed();
            if mini.is_empty() {
                skip_count += chunk.len() * n_batches;
                train_bar.inc(1);
                continue;
            }

            let (gene_ids, cb_batches): (Vec<usize>, Vec<_>) = mini.into_iter().unzip();
            sample_count += gene_ids.len();
            let t_fwd = std::time::Instant::now();

            // Gene identity enters the score function via the gated
            // loss. The chunk's gene ids parallel-replicate to align
            // with the [G*B] cell-side gathers.
            let gene_ids_u32: Vec<u32> = gene_ids.iter().map(|&g| g as u32).collect();

            // (b) ONE forward / backward over the whole chunk.
            let CageLossOut {
                per_level: per_level_gl,
                mean_abs_pair,
            } = cage_nce_loss_per_level(
                &model,
                cb_batches,
                &gene_ids_u32,
                args.nce_objective.to_ge(),
                &dev,
            )?; // [G, L]
                // NB-Fisher per-gene precision: scale each gene's row of the
                // per-level loss by w_g ∈ (0,1] before summing, so high-mean /
                // high-dispersion housekeeping genes contribute less gradient
                // (the loss-side analog of bge's count·fisher positive draw and
                // lc's `apply_gene_weights` on the gene basis). Coverage stays
                // uniform — every gene is still visited once per epoch.
            let (per_level_gl, chunk_mass) = match gene_fisher_weights.as_ref() {
                Some(w) => {
                    let w_chunk: Vec<f32> = gene_ids.iter().map(|&g| w[g]).collect();
                    let mass: f64 = w_chunk.iter().map(|&x| f64::from(x)).sum();
                    let w_g1 = Tensor::from_vec(w_chunk, (gene_ids.len(), 1), &dev)?;
                    (per_level_gl.broadcast_mul(&w_g1)?, mass)
                }
                None => (per_level_gl, gene_ids.len() as f64),
            };
            let loss = per_level_gl.sum_all()?;
            let mut total = loss.clone();

            if args.embedding_l2 > 0.0 {
                // geu's ridge, not a local copy: the reduction is the whole
                // content of this penalty and it was wrong here in the same way.
                let lam = args.embedding_l2 as f64;
                total = (total + embedding_ridge(&model.e_cell, lam)?)?;
                total = (total + embedding_ridge(&model.e_feat, lam)?)?;
            }
            // The learned gate is variational, so its spike-and-slab KL is part
            // of the objective — without it the gate has nothing pulling it
            // toward the prior and simply never selects. Both KL calls return
            // `None` under a sampled mask (no learned Vars to regularize), so
            // this is a no-op on that arm.
            //
            // Over the WHOLE table, deliberately, even though
            // `JointEmbedModel::gate_kl_rows` offers an unbiased chunk-sized
            // estimate that is ~12s/5ep cheaper here.
            //
            // The per-epoch gradient SUMS are equal — full table gives each gene
            // `G/chunk` gradients of `w/(G·H)`, the subset gives one of
            // `w/(chunk·H)` — but the optimizer is AdamW, not SGD. A parameter
            // touched once per 283 steps has its second moment decay and its
            // momentum fill with zeros between visits, so the same gradient mass
            // delivered sparsely moves it far less. Measured at λ=1000: the
            // full-table KL leaves 32% of α below 0.5, the row subset leaves 0%
            // and the gate stops selecting. Switching would need λ re-tuned.
            //
            // Weight is `chunk_mass / epoch_mass`, Fisher-weighted to match the
            // data term, so the term's SHARE does not move with
            // `--gene-batch-size`, the experimental-batch count, or the dataset's
            // mean gene weight.
            //
            // There is no λ here any more, and no per-caller reference to
            // calibrate against. What survives `gate_kl()` is the α-weighted ridge
            // on the loading — see `geu::model::gate`'s `single_gate_kl` — and the
            // SELECTION prior it used to carry is now structural, a fixed IBP
            // ladder on the gate logits with no weight to choose.
            if let Some(kl) = model.gate_kl()? {
                total = (total + (kl * (chunk_mass / epoch_mass))?)?;
            }
            // Global-norm clip before the step. The NCE loss spikes when a
            // chunk draws a gene whose active edges are nearly all positives,
            // and an unbounded step there inflates the embedding norms every
            // later epoch has to work against. A step whose global norm is
            // non-finite is skipped rather than laundered into the parameters.
            timers.decoder_fwd += t_fwd.elapsed();
            let t_bwd = std::time::Instant::now();
            let grads = total.backward()?;
            timers.backward += t_bwd.elapsed();
            let t_opt = std::time::Instant::now();
            // A step is SKIPPED when the global gradient norm is not finite.
            // Count those: a run where every step is skipped moves no parameter
            // at all, and without this the bar would still reach 100% and the
            // phase timings would still look normal — the instrumentation would
            // make the failure less visible rather than more.
            let stepped = clip_and_step_dense(&mut opt, grads, f64::from(args.grad_clip))?;
            if !stepped {
                skipped_steps += 1;
            }
            timers.optimize += t_opt.elapsed();
            train_bar.inc(1);

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
            train_bar.set_message(format!(
                "training — epoch {}/{}, mean |pair| {:.2e}",
                epoch + 1,
                args.epochs,
                pair_mean
            ));
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

    train_bar.finish_and_clear();
    if skipped_steps > 0 {
        warn!(
            "{}/{} optimizer steps were SKIPPED for non-finite gradients — those \
             steps moved no parameters. Lower --lr or --grad-clip if this is a \
             large fraction.",
            skipped_steps, total_steps
        );
    }

    // Where the SGD loop's PER-STEP work went: `precompute` is the rayon
    // gene/edge sampling, `decoder_fwd` the loss forward, then backward and the
    // AdamW step. Reported so optimization work is ranked on measurement.
    //
    // These four do NOT sum to the training wall clock. Per-EPOCH work sits
    // outside them: the `--selection-refresh-epochs` Gibbs refresh (which on the
    // default sampled arm fires every 5th epoch and copies `e_cell` to the host),
    // the per-epoch diagnostic syncs, and the permutation shuffle. Read the
    // percentages as a split of per-step cost, not of the whole phase.
    timers.log_summary();

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
    // What ships depends on which gate ran, because the two mean different
    // things and `e_gene_out` is ALSO the frozen dictionary `project_pairs`
    // scores every cell pair against. Shipping the wrong one would put the pair
    // latent — and so every link community — on a feature side the fit never
    // trained under, which is the defect commit 91c50a65 fixed in geu.
    //
    //   sampled: the trained effects as-is. The mask is NOT re-applied and no
    //   `pip` ships to let anyone re-apply it — it was dropout over the
    //   gradient, so its effect is already in these values, and multiplying by
    //   a keep-rate would shrink twice for something that was never a
    //   coefficient.
    //
    //   learned: `σ(S/τ)` multiplied the loading on every step, so it IS part
    //   of the fitted coefficient. `materialize_e_feat` bakes it in, reading
    //   the raw Var so a second call cannot gate twice.
    if sampled_selection.is_none() {
        model.materialize_e_feat()?;
        info!("Baked the learned gate into the shipped feature embedding");
    }
    let e_gene_out = tensor_to_mat(&model.e_feat)?;
    let b_gene_out = tensor_to_mat_1d(&model.b_feat)?;

    e_gene_out.to_parquet_with_names(
        &(c.out.to_string() + ".feature_embedding.parquet"),
        (Some(&gene_names), Some("feature")),
        Some(&embedding_col_names(args.embedding_dim)),
    )?;

    // The nascent deviation, on a splice-channelized input only, and written HERE
    // rather than beside the cold selection because of which FRAME it is in.
    //
    // `delta_feature_embedding` is meant to be consumed as a dictionary — an
    // unspliced row loads `e_feat[g] + delta_g` — so it is only meaningful in the
    // same D-dim frame as `feature_embedding.parquet` directly above. The cold
    // chain was fit against an SVD of pseudobulk log-counts, a basis the shipped
    // model never uses; only a refresh sees the live `e_cell` folded into
    // super-cells. Shipping the cold delta would hand downstream a vector to add
    // to `e_feat` that was measured in a different coordinate system.
    //
    // Two tables rather than one because the second is not derivable from the
    // first: this is HOW MUCH an unspliced row loads on top of the identity,
    // `delta_selection` is WHETHER it loads at all. Genes whose delta is not
    // identified are NaN in both.
    let delta_from_refresh = latest_refreshed.is_some();
    let delta_source = latest_refreshed
        .as_ref()
        .or_else(|| sampled_selection.as_ref().map(|(sel, _, _)| sel));
    if let Some(delta) = delta_source.and_then(|s| s.delta.as_ref()) {
        if !delta_from_refresh {
            warn!(
                "delta was fit by the COLD chain, against an SVD of pseudobulk log-counts \
                 rather than the live cell embedding, so it is NOT in the same frame as \
                 feature_embedding.parquet and must not be added to it. Raise \
                 --selection-refresh-epochs above 0 to get a usable delta dictionary."
            );
        }
        let cols = embedding_col_names(args.embedding_dim);
        delta
            .mean_matrix(n_genes, args.embedding_dim)
            .to_parquet_with_names(
                &(c.out.to_string() + ".delta_feature_embedding.parquet"),
                (Some(&gene_names), Some("gene")),
                Some(&cols),
            )?;
        delta
            .pip_matrix(n_genes, args.embedding_dim)
            .to_parquet_with_names(
                &(c.out.to_string() + ".delta_selection.parquet"),
                (Some(&gene_names), Some("gene")),
                Some(&cols),
            )?;
        info!(
            "Wrote {}.delta_feature_embedding.parquet and {}.delta_selection.parquet \
             ({} of {} genes identified, the rest NaN; fit against {})",
            c.out,
            c.out,
            delta.n_identified(),
            n_genes,
            if delta_from_refresh {
                "the live cell embedding"
            } else {
                "the COLD pseudobulk SVD basis"
            }
        );
    }

    // The learned gate's inclusion table `α = σ(S/τ)`. Reported here — unlike
    // the sampled arm's `pip`, which stays unwritten — because α is a fitted
    // coefficient the run stands behind, not a dropout rate. `feature_selection`
    // returns `None` under an installed pip, so this is empty on that arm.
    if let Some(alpha) = model.feature_selection()? {
        tensor_to_mat(&alpha)?.to_parquet_with_names(
            &(c.out.to_string() + ".feature_selection.parquet"),
            (Some(&gene_names), Some("feature")),
            Some(&embedding_col_names(args.embedding_dim)),
        )?;
        info!("Wrote {}.feature_selection.parquet", c.out);
    }

    b_gene_out.to_parquet_with_names(
        &(c.out.to_string() + ".gene_bias.parquet"),
        (Some(&gene_names), Some("gene")),
        Some(&[Box::from("b_gene")]),
    )?;

    write_score_trace(&(c.out.to_string() + ".scores.parquet"), &score_trace)?;

    /////////////////////////////////////////////////////////
    // 10. Pair projection -> link communities -> propensity  //
    /////////////////////////////////////////////////////////
    // cage's downstream contract is the one `lc` and `dsvd` publish, and it is
    // built on the CELL PAIR, not the cell: a latent per pair, k-means over
    // those pairs for link communities, and a cell's propensity as the mix of
    // communities its incident edges carry.
    //
    // The pair latent comes from projecting each pair's POOLED counts onto the
    // frozen gene embedding (`pair_projection`) — the same phase-2 move `senna
    // bge` makes for cells, with the pair as the node. cage's own score
    // decomposes an edge as `⟨θ_g, e_u ⊙ e_v⟩`, so a Hadamard product is the
    // closed form this generalizes: the projection lets the pair's own expression
    // move it off that point, which is what puts a boundary pair between the
    // two programs it pools instead of on either endpoint.
    let pair_batch = batch_db.as_ref().map(|delta| PairBatchDivisor {
        delta,
        batch_of_cell: &batch_membership_u32,
    });
    let PairLatent {
        latent: pair_latent,
        bias: pair_bias,
    } = project_pairs(
        &data_vec,
        &edges_owned,
        &e_gene_out,
        pair_batch,
        &PairProjectionArgs {
            projection: ProjectionArgs {
                ridge: args.pair_ridge,
                steps: args.pair_steps,
                gene_sample: args.pair_gene_sample,
            },
            seed: c.seed,
            pair_block: args.pair_block,
        },
        &gene_axis,
        &gene_totals,
    )?;
    {
        // `β_uv` is the pair's log pooled depth; it never leaves this function,
        // but its spread is the cheapest check that the projection saw real data.
        let mut sorted = pair_bias;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if let (Some(&lo), Some(&hi)) = (sorted.first(), sorted.last()) {
            info!(
                "Pair intercept β: min {:.3}, median {:.3}, max {:.3}",
                lo,
                sorted[sorted.len() / 2],
                hi
            );
        }
    }

    let edges_usize: Vec<(usize, usize)> = edges_owned
        .iter()
        .map(|&(u, v)| (u as usize, v as usize))
        .collect();

    // `[D × E]` with every pair L2-normalized — the exact shape and
    // normalization `pinto dsvd` hands the shared routine, so edge k-means
    // clusters on composition and not on pooled depth.
    let mut proj_kn = pair_latent.transpose();
    proj_kn.normalize_columns_inplace();

    // One `[E × D]` copy, written out and then clustered — the shared routine
    // takes pairs as rows, so this is the same buffer both times.
    let proj_ne = proj_kn.transpose();
    proj_ne.to_parquet_with_names(
        &(c.out.to_string() + ".latent.parquet"),
        (None, Some("cell_pair")),
        Some(&embedding_col_names(args.embedding_dim)),
    )?;

    // Cluster the pairs -> per-edge community -> cell propensity (incident-edge
    // fractions) + entropy + the Poisson-Gamma gene x community dictionary. One
    // routine, shared verbatim with `lc` and `dsvd`, so every subcommand's
    // propensity means the same thing.
    // Leiden by default: the pair latent has no reason to carry exactly
    // `embedding_dim` interaction regimes, so the graph decides the count and
    // `n_edge_clusters` is only a target (or nothing, when left unset). Under
    // k-means the requested count IS the count.
    let clustering = args.edge_clustering.resolve(c.seed);
    let n_edge_clusters = compute_propensity_and_gene_community_stat(
        &proj_ne,
        &edges_usize,
        &data_vec,
        n_cells,
        &PropensityReportConfig {
            clustering,
            block_size: c.block_size,
            // Deliberately row-keyed, and documented as such in `pinto cage
            // --help`: a channelized run lists a gene's two tracks as two rows,
            // which is the one place the nascent/mature contrast is reportable.
            // `lc` folds instead, because its dictionary merge has to index the
            // same axis its gene filter chose on.
            gene_axis: None,
        },
        &c.out,
    )?;

    // Metadata
    {
        let mut splice_report = splice_report;
        if let Some(sr) = splice_report.as_mut() {
            // Set ONLY when a delta table was actually written, so "absent" and
            // "written against the cold basis" stay distinguishable.
            if let Some(d) = delta_source.and_then(|s| s.delta.as_ref()) {
                sr.delta_from_refresh = Some(delta_from_refresh);
                sr.delta_median_counts = Some(d.strength.median_counts);
                sr.delta_counts_per_pseudobulk = Some(d.strength.counts_per_pb());
            }
        }
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
                k: n_edge_clusters,
            },
            batch_db.is_some(),
            splice_report,
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
