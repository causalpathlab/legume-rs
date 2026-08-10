//! Entry point for `faba gem` (alias `gem-embedding`).
//!
//! Genes-only joint embedding: each row `{gene}/count/{spliced|unspliced}` is an
//! independent feature sharing the cell axis, and a gene's two tracks embed
//! identically as `β_g` (β-sharing) via the per-gene factorization. Driven
//! straight through the shared `graph_embedding_util` engine — the bilinear
//! score `e_feat·e_cell + b_feat + b_cell`, phase-1 multilevel-pseudobulk
//! training + phase-2 analytical per-cell projection. Cell **identity** is
//! resolved by the SPLICED edges (mature mRNA = current state) and written **raw**
//! (`{out}.cell_embedding.parquet`, magnitude kept); the same phase-2 pass fits an analytic
//! velocity increment `δ` to the unspliced edges (identity held fixed) and writes
//! it **raw** (`{out}.velocity.parquet`, ‖δ‖ = speed). Everything is the model's
//! actual MAP estimate — no post-hoc unit-norm, no aggregation. The nascent state
//! is just `θ + δ` = latent + velocity (derivable). Per-gene velocity, if wanted, is
//! the in-model `δ_g` (`--delta-l2`). No softmax co-embedding is written (see the
//! NOTE in `run_gem_genes_bge`: not every gene can be co-embedded).
//!
//! NOTE — `cell_embedding` is **raw** (its norm carries library size), so cluster /
//! UMAP it with **cosine** distance, or L2-normalize the rows first; plain Euclidean
//! would be dominated by the depth axis. (Only the gem/splice path stores raw; `senna
//! bge` still writes the L2 direction.)

use anyhow::Context;
use candle_util::candle_core::Tensor;
use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util::data::UnifiedData;
use graph_embedding_util::{load_unified_data, FeatureNameKind, LoadUnifiedArgs};
use log::info;
use matrix_util::common_io::{basename, mkdir_parent};
use rayon::ThreadPoolBuilder;
use rustc_hash::FxHashMap;

use crate::gem::args::GemArgs;
use crate::gem::sample_id::{file_sample_id, longest_common_underscore_suffix};

/// Default ridge on the per-gene splice offset δ_g, applied automatically whenever
/// the input carries unspliced rows and the user did not set `--delta-l2`. Keeping a
/// mild ridge on by default means every spliced+unspliced gem run always emits a δ_g
/// dictionary (`{out}.delta_feature_embedding.parquet`) for downstream `faba annotate
/// --track velocity`, without over-shrinking. Matches the documented `--delta-l2`
/// range (0.01–1.0).
const DEFAULT_DELTA_L2: f32 = 1.0;

// NOTE: `feature_embedding_l2` MUST be 0 for gem. It penalizes a free `E_feat`
// Var, but gem is β-sharing (`feat_factor = Some`) — the trained params are β_g
// and δ_g (δ_g already regularized by `--delta-l2`), and `E_feat` is a
// materialized snapshot, not a Var. A nonzero value trips the engine's
// `feat_factor + feature_embedding_l2 > 0` guard (fit/mod.rs) and aborts the fit.

pub fn run_gem_embedding(args: &GemArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    validate_args(args)?;
    // Reconcile --posterior with --mcmc/--jitter BEFORE any I/O, so a
    // contradictory pair fails in the first second rather than after the fit.
    let posterior_plan = args.posterior.resolve(args.runtime.seed)?;

    let n_threads = if args.runtime.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        args.runtime.threads
    };
    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok(); // ignore error if pool already initialised
    info!(
        "rayon thread pool: {} threads",
        rayon::current_num_threads()
    );

    let feature_kind = if args.collapse.feature_name_exact {
        FeatureNameKind::Exact
    } else {
        FeatureNameKind::Gene {
            delim: args.collapse.feature_name_delim,
        }
    };

    let batch_files: Option<&[Box<str>]> = if args.collapse.ignore_batch {
        if args.batch_files.is_some() {
            info!("--ignore-batch: dropping batch labels; treating all cells as one batch");
        }
        None
    } else {
        args.batch_files.as_deref()
    };

    run_gem_genes_bge(args, feature_kind, batch_files, posterior_plan)
}

/// Load the `--genes` files into one `UnifiedData`, tagging each file's
/// barcodes with its `@sample` id when there is >1 file and no explicit
/// `--batch-files` (so batch identity survives the Union merge).
fn load_modality(
    files: &[Box<str>],
    strip: &str,
    do_tag: bool,
    batch_files: Option<&[Box<str>]>,
    feature_kind: FeatureNameKind,
    preload: bool,
) -> anyhow::Result<UnifiedData> {
    let mut data_files: Vec<Box<str>> = Vec::with_capacity(files.len());
    let mut sample_ids: Vec<Box<str>> = Vec::with_capacity(files.len());
    for f in files {
        sample_ids.push(file_sample_id(f, strip)?);
        data_files.push(f.clone());
    }
    let per_file_barcode_suffix: Option<Vec<Option<Box<str>>>> = if do_tag {
        Some(sample_ids.into_iter().map(Some).collect())
    } else {
        None
    };
    load_unified_data(LoadUnifiedArgs {
        data_files,
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        feature_kind: Some(feature_kind),
        preload,
        column_alignment: ColumnAlignment::Union,
        per_file_barcode_suffix,
        ..Default::default()
    })
}

/// Genes-only joint embedding over the shared `graph_embedding_util` engine.
/// Writes `{out}.{cell_embedding,feature_embedding,feature_bias,cell_bias}.parquet`
/// (cell_embedding = raw spliced θ) and `{out}.velocity.parquet` (raw velocity
/// increment δ). gem spells the two embedding tables out rather than using senna's
/// `latent` / `dictionary` — it is not a topic model. No softmax co-embedding; no
/// nascent/driver post-hoc.
fn run_gem_genes_bge(
    args: &GemArgs,
    feature_kind: FeatureNameKind,
    batch_files: Option<&[Box<str>]>,
    posterior_plan: Option<graph_embedding_util::posterior::PosteriorPlan>,
) -> anyhow::Result<()> {
    use graph_embedding_util as ge;

    // Genes load (+ per-file `@sample` tag for batch identity when >1 file and
    // no explicit --batch-files). The sample-id strip is the explicit
    // `--genes-sample-strip`, else the longest common `_`-suffix across the
    // genes basenames.
    let genes = args.genes()?;
    let do_tag = batch_files.is_none() && genes.len() > 1;
    let genes_strip: Box<str> = if !args.collapse.genes_sample_strip.is_empty() {
        args.collapse.genes_sample_strip.clone()
    } else if do_tag {
        let genes_bn: Vec<Box<str>> = genes
            .iter()
            .map(|f| basename(f))
            .collect::<anyhow::Result<_>>()?;
        let s = longest_common_underscore_suffix(&genes_bn);
        if !s.is_empty() {
            info!("auto-strip: --genes-sample-strip = {:?}", s.as_ref());
        }
        s
    } else {
        "".into()
    };
    if do_tag {
        info!("tagging barcodes with per-file @sample id for batch identity");
    }
    let mut unified = load_modality(
        genes,
        &genes_strip,
        do_tag,
        batch_files,
        feature_kind,
        args.runtime.preload_data,
    )
    .context("load genes backend")?;
    info!(
        "genes loaded: {} features × {} cells, {} batch(es)",
        unified.n_features(),
        unified.n_cells(),
        unified.n_batches()
    );

    // Optional gene-level HVG feature filter. NOTE this is NOT what `--n-hvg` does in
    // `senna bge`, and the difference is deliberate at both ends: bge keeps the full
    // feature axis and uses the selection only to WEIGHT its random projection, while here
    // it hard-subsets. Selecting the top-N most variable GENES drops the rest — both the
    // spliced and unspliced rows of a dropped gene together, so the β-sharing
    // factorization stays aligned — and `subset_features` narrows the dictionary/co-embed
    // accordingly, with the uniform `hvg_weights` over the survivors then restricting the
    // pb projection / membership to those genes too. `None` (n_hvg = 0, the DEFAULT) keeps
    // every gene and lets the feature gate select, which is the recommended path.
    //
    // What the subset is for is a smaller dictionary, and nothing more. An earlier version
    // of this comment justified it as "removing the low-detection empty genes that pile at
    // the co-embed centre" — the same premise measured FALSE on bge's control arm, where
    // 0.0% of genes sit within 0.1 cell-radii of the centroid and the median distance is
    // 0.803. It has never been measured here either way, so it is not a reason to cite.
    //
    // `--must-train-features` force-includes a curated panel on top of that cut, at
    // the GENE level (so both splice tracks of a kept gene come along). Loaded only
    // when the HVG cut is on (the feature gate handles selection otherwise).
    //
    // The feature gate is gem's selector now — an INDEPENDENT Bernoulli inclusion
    // probability per (gene, dim), so a gene with no cell-state signal simply has
    // σ(S) → 0 in every dim and contributes ≈0. There is no null slot to send mass to:
    // that was the per-dim softmax over genes, which this replaced. The old ash-QC /
    // LRT two-pass
    // refit is retired. An explicit `--n-hvg N` still hard-subsets to the top-N genes
    // (a smaller dictionary; the remainder is restored by the post-hoc projection).
    let hvg_on = args.collapse.n_hvg > 0;
    let selection_on = hvg_on;
    // `--markers` is force-trained alongside `--must-train-features`. The annotators read
    // only the TRAINED feature rows, so a marker off the trained axis is absent from the
    // panel rather than merely down-weighted — naming the panel here is what keeps the genes
    // the calls are made on and the genes the model fit the same set.
    //
    // The panel is kept separately as well as unioned in, so the coverage log below can say
    // what share of the trained axis it is *without* re-reading the file.
    let explicit = data_beans_alg::hvg::load_must_train(
        args.collapse.must_train_features.as_deref(),
        selection_on,
    )?;
    let panel =
        data_beans_alg::hvg::load_must_train(args.collapse.markers.as_deref(), selection_on)?;
    let parts: Vec<&data_beans_alg::hvg::MustTrainFeatures> = [explicit.as_ref(), panel.as_ref()]
        .into_iter()
        .flatten()
        .collect();
    let must_train =
        (!parts.is_empty()).then(|| data_beans_alg::hvg::MustTrainFeatures::union(&parts));

    // Per-ROW projection weights over the FULL feature axis, matching what `senna bge`
    // does. `None` when `--n-hvg 0`.
    let mut hvg_row_weights: Option<Vec<f32>> = None;

    if hvg_on {
        use data_beans_alg::hvg::select_hvg_by_stats;
        use data_beans_alg::sparse_streaming::streaming_sparse_running_stats;
        use matrix_util::traits::RunningStatOps;
        // Select the top-N most variable GENES (not rows): compute per-row running
        // stats, POOL a gene's spliced + unspliced tracks onto one gene entry, and
        // rank genes by NB dispersion-trend excess. `--n-hvg N` keeps exactly the N
        // most variable genes (both tracks of each together) — pooling spliced +
        // unspliced total mirrors the pipeline's CR-style gene filter. Ranking on rows
        // instead would return well under N genes (the two correlated tracks of a gene
        // both rank high and collapse to one gene on dedup).
        let (row_gene, genes) = intern_gene_keys(&unified.feature_names);
        let n_genes = genes.len();
        let stat = streaming_sparse_running_stats(unified.count_backend(), None, "HVG")
            .context("HVG streaming stats")?;
        let (means, vars) = (stat.mean(), stat.variance());
        // Pooled gene stats: mean is exact (E[s+u]=E[s]+E[u]); var sums the tracks (a
        // lower bound ignoring cross-track covariance — fine for ranking).
        let mut gmean = vec![0f32; n_genes];
        let mut gvar = vec![0f32; n_genes];
        for (r, (&m, &v)) in means.iter().zip(vars.iter()).enumerate() {
            gmean[row_gene[r] as usize] += m;
            gvar[row_gene[r] as usize] += v;
        }
        let mut selected = select_hvg_by_stats(&gmean, &gvar, args.collapse.n_hvg);

        // Force-include, resolved against the GENE keys (not the count rows), so a
        // `CD8A` panel entry keeps `CD8A/count/spliced` AND `CD8A/count/unspliced`.
        // The index lowercases and hash-indexes the whole gene vocabulary, so build it once
        // and resolve both lists (the force-train union, then the panel) against it.
        let gene_index = data_beans::utilities::name_matching::GeneIndex::build(&genes);
        let forced = if let Some(must_train) = must_train.as_ref() {
            let forced = must_train.resolve_with(&gene_index);
            let added = data_beans_alg::hvg::union_indices(&mut selected, &forced);
            info!(
                "force-train: {added} gene(s) added on top of the HVG cut \
                 ({} of the {} matched were already HVGs)",
                forced.len() - added,
                forced.len()
            );
            added
        } else {
            0
        };

        let keep_genes: rustc_hash::FxHashSet<usize> = selected.into_iter().collect();

        // What share of the trained axis IS the marker panel? Worth saying out loud, because
        // it is the price of forcing the panel in: the embedding is now partly built to
        // separate the very compartments the panel will later be used to call, so a
        // downstream "the markers agree with the clusters" check is a check on the grouping,
        // not an independent confirmation. Small share ⇒ the axis still has its own opinion.
        if let Some(panel) = panel.as_ref() {
            let on_axis = panel
                .resolve_quiet_with(&gene_index)
                .into_iter()
                .filter(|g| keep_genes.contains(g))
                .count();
            info!(
                "--markers: {on_axis} panel gene(s) on the trained axis = {:.0}% of its {} \
                 gene(s). The embedding is trained to separate what the panel will later \
                 call, so read `annotate`'s agreement as a check on the grouping, not an \
                 independent one.",
                100.0 * on_axis as f32 / keep_genes.len().max(1) as f32,
                keep_genes.len()
            );
        }
        // WEIGHT, do not subset — the `senna bge` semantics. Non-selected genes get
        // projection weight 0, so they sit out the basis the pseudobulk partition is built
        // from, but they stay on the feature axis: still trained, still gated, still in the
        // dictionary, the co-embedding and the posterior's anchor set.
        //
        // This used to call `subset_features`, which dropped them outright. Weighting is
        // strictly more informative for the same cost — the RP sees the same genes either
        // way — and it makes `--n-hvg N` mean the same thing on both CLIs, so a gem run and
        // a bge run at the same N are finally the same experiment. What it gives up is the
        // smaller dictionary and faster fit a hard cut bought; if that is wanted back it
        // belongs behind its own flag rather than overloading this one.
        let mut w = vec![0.0f32; unified.n_features()];
        for (r, slot) in w.iter_mut().enumerate() {
            if keep_genes.contains(&(row_gene[r] as usize)) {
                *slot = 1.0;
            }
        }
        let weighted_rows = w.iter().filter(|&&x| x > 0.0).count();
        hvg_row_weights = Some(w);
        info!(
            "HVG weighting (--n-hvg {}): {} of {} genes selected ({} HVG + {} force-kept) \
             → {} of {} feature rows carry the projection; every gene still trains",
            args.collapse.n_hvg,
            keep_genes.len(),
            n_genes,
            keep_genes.len() - forced,
            forced,
            weighted_rows,
            unified.n_features()
        );
    }

    // Compute device. gem does a single gated fit (the feature gate selects
    // features during training), so there is no second pass to reconcile here.
    let dev = args
        .runtime
        .device
        .to_device(args.runtime.device_no)
        .context("candle device init")?;
    info!("compute device = {:?}", dev);

    // Build a `FitConfig` for the CURRENT feature axis of `unified`: the per-gene
    // β-sharing factor is derived from the live feature names, and the δ_g ridge /
    // HVG weights align to that axis. gem fits ONCE (the feature gate is the
    // selector; the senna-bge post-QC refit is retired). Returns the config plus the
    // axis-derived gene names
    // and resolved δ ridge the downstream dictionary writers need.
    //
    // Per-gene β-sharing factorization: each row `{gene}/count/{spliced|unspliced}`
    // maps to its gene, so a gene's two tracks embed identically as `β_g`; the splice
    // deviation is recovered as the phase-2 velocity increment δ on the CELL axis. δ_g
    // is auto-on with a mild ridge whenever both tracks are present (unless the user
    // set `--delta-l2`) so a δ_g dictionary is always emitted for `faba annotate
    // --track velocity`; a spliced-only input keeps δ off.
    // Returns the axis-derived gene names and δ ridge the dictionary writers need,
    // plus the row→gene map and unspliced mask — the posterior anchors on genes per
    // splice track, and re-deriving them later would both cost a second intern pass
    // and let the two copies desync if the feature axis ever moved between.
    type CfgParts = (ge::FitConfig, Vec<Box<str>>, f32, Vec<u32>, Vec<bool>);
    let build_cfg = |unified: &UnifiedData| -> anyhow::Result<CfgParts> {
        let (factor, gene_names) = build_splice_factor(&unified.feature_names);
        let (row_to_gene, unspliced_rows) =
            (factor.row_to_gene.clone(), factor.unspliced_rows.clone());
        info!(
            "β-sharing factor: {} genes from {} count rows ({} unspliced rows); \
             splice δ → cell-axis velocity increment",
            gene_names.len(),
            unified.feature_names.len(),
            factor.unspliced_rows.iter().filter(|&&b| b).count(),
        );
        let has_unspliced = factor.unspliced_rows.iter().any(|&b| b);
        let delta_l2 = if args.model.delta_l2 > 0.0 {
            args.model.delta_l2
        } else if has_unspliced {
            DEFAULT_DELTA_L2
        } else {
            0.0
        };
        // The selection's own per-row weights over the full axis. Previously this was a
        // UNIFORM vector over the survivors of a subset, which weighted nothing — the
        // selection had already happened by deletion.
        let hvg_weights = hvg_row_weights.clone();
        let cfg = ge::FitConfig {
            embedding_dim: args.model.embedding_dim,
            // gem has no carried-reference update path.
            anchor_batches: None,
            emit_finest_collapse: false,
            num_levels: args.collapse.num_levels,
            sort_dim: args.collapse.sort_dim,
            knn_pb_samples: args.collapse.knn_pb,
            num_opt_iter: args.collapse.num_opt_iter,
            proj_dim: args.collapse.proj_dim,
            hvg_weights,
            // geu's multilevel collapse requires a refine spec (it surfaces the
            // per-level cell→pb maps phase-2 needs). Use geu's defaults — same as a
            // `senna bge` run without `--no-refine`.
            refine: Some(ge::RefineParams::default()),
            epochs: args.train.epochs,
            batches_per_epoch: args.train.batches_per_epoch,
            batch_size: args.train.batch_size,
            num_negatives: 4,
            learning_rate: args.train.learning_rate,
            seed: args.runtime.seed,
            device: dev.clone(),
            block_size: None,
            feature_embedding_l2: 0.0, // must be 0 for β-sharing (see note above)
            weight_decay: args.train.weight_decay,
            max_grad_norm: args.train.max_grad_norm,
            cell_weight_mult: None,
            phase1_cells_per_pb: args.collapse.phase1_cells_per_pb,
            feat_factor: Some(factor),
            delta_l2,
            lineage_dag: args.train.lineage_dag,
            lineage_smooth: args.train.lineage_smooth,
            lineage_mst: !args.train.dense_dag,
            joint_velocity: !args.train.sequential_velocity,
            // Restore backend genes the trained axis is missing — the `--n-hvg`
            // remainder — solved (with velocity) against the frozen pseudobulk side.
            // `null_fdr: 0` = restore ALL of them (no ash-null gate; the feature gate
            // is now the selector). Self-disables when the trained axis is the backend
            // (the default `--n-hvg 0`): nothing is held out. Cell outputs unaffected.
            // `--posterior N` drives the phase-1 pb Gibbs over BOTH gates. Leaving
            // this `None` while still resolving `posterior_plan` made the flag a
            // silent no-op that the run manifest nonetheless advertised.
            pb_posterior: posterior_plan.map(|plan| plan.pb_gibbs_config()),
            pb_posterior_nested_delta: !args.model.independent_delta_gate,
            // `--nce-objective` (default softmax = InfoNCE: on gem's dense count data
            // the positive competing against its negatives in one distribution
            // separates cell types better than the per-pair logistic SGNS loss).
            nce_objective: args.model.nce_objective.to_ge(),
            // Per-gene softmax feature gate — the SuSiE variational spike-and-slab
            // single-effect, ALWAYS ON. Gates β_g (identity) AND, independently, δ_g
            // (velocity → velocity_selection); null absorber + categorical + Gaussian
            // effect KL at the fixed internal weight. Temperature is the one knob.
            feature_gate: Some(ge::FeatureGateConfig {
                temperature: args.model.feature_gate_temp,
                ibp_alpha: args.model.gate_ibp_alpha,
            }),
        };
        Ok((cfg, gene_names, delta_l2, row_to_gene, unspliced_rows))
    };

    // Single-pass gated fit — the feature gate selects features DURING training (a
    // junk gene sends its gate mass to null → β̃_g ≈ 0), so there is no LRT null-call
    // or two-pass refit. The `--n-hvg` remainder (if any) is restored post-hoc.
    // `row_to_gene` / `unspliced_rows` now travel to the sampler inside `FitConfig`
    // (`feat_factor`), so nothing outside `fit` re-derives the splice keying.
    let (cfg, gene_names, delta_l2, _row_to_gene, _unspliced_rows) = build_cfg(&unified)?;
    let out = ge::fit(&mut unified, cfg).context("ge::fit (genes bge)")?;
    let n_genes = gene_names.len();

    // Both gates' posteriors, keyed by gene so they join against the β dictionary.
    if let Some(post) = out.splice_posterior.as_ref() {
        ge::eval::write_splice_posterior_tables(&args.out, post, &gene_names)?;
        ge::posterior::pb_gibbs::write_splice_posterior_hyper(
            &args.out,
            post,
            &out.varmap,
            // geometry now travels on the result, not as a caller-supplied cap
            args.runtime.seed,
        )?;
        info!(
            "phase-1 splice posterior: {} sweeps; dims/gene β {:.2}, δ {:.2} (posterior); \
             {} gene(s) with an unidentified δ",
            post.n_kept,
            // Mean row-sum of each gate's PIP table — the dims a gene loads AS INFERRED,
            // per gate. Not `Σ_h (1 − π₀ₕ)`, which under the default IBP is the fixed
            // ladder and would echo `α` twice over; not a median, which on a monotone
            // ladder is the rate at dim H/2. Each gate gets its own because that is the
            // point of sampling them separately.
            mean_dims(&post.beta_pip, post.h),
            mean_dims(&post.delta_pip, post.h),
            post.delta_identified.iter().filter(|&&x| !x).count(),
        );
    }

    // On interrupt (Ctrl+C) `fit()` skips phase-2 + lineage, so the outputs below are
    // partial. gem has no heavy post-fit stage to skip (unlike bge's co-embed/ETM), so
    // it just writes what it has and exits — but flag that the result is un-projected.
    if ge::stop_flag().load(std::sync::atomic::Ordering::Relaxed) {
        log::warn!(
            "Interrupted — outputs are partial (cell embedding un-projected; \
             velocity/lineage skipped). Re-run without interrupting for full results."
        );
    }

    // NOTE: NO softmax co-embedding is written. (1) The gene↔cell co-embedding
    // (`{out}.feature_embedding.parquet`) is dropped: cell-type identity is carried by
    // a *few* high-contrast marker genes — a sparse, heavy-tailed distribution of gene
    // norms ‖β_g‖ — and a softmax barycenter that drops every gene onto the cell
    // manifold has to flatten that contrast, so the great majority of genes land with
    // no meaningful cell location: NOT EVERY GENE CAN BE CO-EMBEDDED. Gene↔cell
    // co-embedding and sharp cell clusters are the same degree of freedom pulling
    // opposite ways — we keep the sharp clusters. (2) No velocity "driver" co-embed
    // either: a per-gene velocity readout, if wanted, is the in-model δ_g (`--delta-l2`
    // → `{out}.delta_feature_embedding.parquet`), not a post-hoc average. The feature embedding
    // is keyed by *feature row* (`{gene}/count/{spliced|unspliced}`); the gene-keyed β_g
    // dictionary below is what marker-based `faba annotate` consumes.
    //
    // gem writes the EXPLICIT names (`{out}.cell_embedding.parquet` /
    // `{out}.feature_embedding.parquet`) rather than senna's `latent` / `dictionary` —
    // gem is not a topic model, so "latent"/"dictionary" said less than the tables are.
    // `faba {lineage, annotate}` read these names.
    /////////////
    // cell QC //
    /////////////
    // An OUTPUT filter, matching `senna bge`: every cell and edge still informs
    // the joint embedding and the feature dictionary; QC-failed cells are
    // dropped only from the per-cell tables, via the `cell_keep_idx` hook
    // `graph_embedding_util::eval` already provides (it subsets the row indices
    // and the barcodes together, so they cannot desync).
    let qc_keep: Option<Vec<usize>> = match args.qc.to_config() {
        Some(cfg) => {
            if cfg.feature_min_cells > 0 {
                log::warn!(
                    "--qc-feature-min-cells is ignored by gem (cell-only QC; the \
                     dictionary keeps all features)"
                );
            }
            let report = data_beans::qc_lib::compute_qc(unified.count_backend(), &cfg, None)
                .context("cell QC")?;
            let keep = report.emit_idx_unmasked();
            info!(
                "cell QC: {} / {} cells kept for OUTPUT ({} near-empty, {} MAD-outlier); \
                 training uses all of them",
                keep.len(),
                unified.n_cells(),
                report.near_empty.iter().filter(|&&e| e).count(),
                report.n_cells_dropped,
            );
            if let Some(path) = args.qc.qc_report.as_deref() {
                data_beans::qc_lib::write_qc_report(path, &unified.barcodes, &report)
                    .context("writing the QC report")?;
                info!("wrote {path}");
            }
            (keep.len() < unified.n_cells()).then_some(keep)
        }
        None => None,
    };

    let cpu = candle_util::candle_core::Device::Cpu;
    ge::save_outputs_named(
        &out.model,
        &ge::OutputContext {
            feature_names: &unified.feature_names,
            barcodes: &unified.barcodes,
            cell_keep_idx: qc_keep.as_deref(),
        },
        &args.out,
        ge::EmbeddingFileNames::EXPLICIT,
    )
    .context("save outputs")?;

    // Gene-keyed β_g dictionary. `save_outputs` writes the dictionary keyed by feature
    // row (`{gene}/count/spliced|unspliced`), which a gene-symbol marker set cannot
    // match. Read the per-gene `beta` Var directly and save it row-labeled by gene —
    // the spliced/mature gene program that `faba annotate --track spliced` pairs with
    // the cell latent θ. Every gene is in-model, so this table IS the whole gene axis.
    // Symmetric with the δ_g dictionary below.
    // β_g no longer needs to be snapshotted for the velocity posterior: the δ block
    // now carries β as a per-anchor offset refreshed EVERY sweep inside `fit`,
    // rather than against a one-time MAP copy taken out here.
    {
        let vars = out.varmap.data().lock().unwrap();
        if let Some(beta) = vars.get("beta") {
            let beta_t = beta.as_tensor().to_device(&cpu)?;
            let h = beta_t.dim(1)?;
            let flat: Vec<f32> = beta_t.flatten_all()?.to_vec1()?;
            let merged = Tensor::from_vec(flat, (gene_names.len(), h), &cpu)?;
            ge::save_embedding(
                &format!("{}.beta_feature_embedding.parquet", args.out),
                &merged,
                &gene_names,
                "feature",
            )
            .context("save β_g feature embedding")?;
            info!(
                "wrote {}.beta_feature_embedding.parquet (per-gene β_g; {} genes, all in-model)",
                args.out,
                gene_names.len(),
            );
        }
    }

    // Per-gene splice offset δ_g (`--delta-l2 > 0`): the nascent loading
    // (unspliced e_f = β_g + δ_g). Read the trained `delta` Var from the varmap
    // and save it row-labeled by gene — genes with large ‖δ_g‖ carry a distinct
    // nascent/velocity program; the L2 ridge shrinks the rest toward 0.
    if delta_l2 > 0.0 {
        let vars = out.varmap.data().lock().unwrap();
        if let Some(delta) = vars.get("delta") {
            let d_t = delta.as_tensor().to_device(&cpu)?;
            let h = d_t.dim(1)?;
            // genes whose ‖δ_g‖ is above ~0 (the L2 ridge shrinks but does NOT
            // sparsify, so this count is typically most genes — it is a coverage
            // readout, not a sparsity one).
            let per_gene_max: Vec<f32> = d_t.abs()?.max(1)?.to_vec1()?;
            let nz = per_gene_max.iter().filter(|&&x| x > 1e-6).count();
            let flat: Vec<f32> = d_t.flatten_all()?.to_vec1()?;
            let merged = Tensor::from_vec(flat, (gene_names.len(), h), &cpu)?;
            ge::save_embedding(
                &format!("{}.delta_feature_embedding.parquet", args.out),
                &merged,
                &gene_names,
                "feature",
            )
            .context("save δ_g feature embedding")?;
            info!(
                "wrote {}.delta_feature_embedding.parquet (δ_g; {}/{} genes with nonzero offset)",
                args.out, nz, n_genes,
            );
        }
    }

    // `{out}.gene_qc.parquet` and `{out}.projection_qc.json` are gone with the two-stage
    // fit. Every column they carried described the trained-vs-projected split — which gene
    // got an in-model estimate, how much evidence a projected one had, how well the two
    // frames were calibrated against each other. There is one frame now, so those columns
    // would be a constant `trained = true` beside all-NaN scan statistics.
    // Cell-axis velocity (β-sharing). The identity `latent` above is the RAW spliced θ.
    // The velocity is the EMBEDDING-SPACE operator v = P·θ (`velocity_operator`): the shift
    // that makes each cell's spliced prediction catch up to its nascent one, read off the
    // DENOISED dictionaries β_g (spliced) and δ_g (= β_u − β_s) — no raw U−S count
    // differencing. The per-cell Poisson increment δ_c is instead dominated by a
    // shrinkage-toward-origin common-mode (δ_c ≈ −0.5·θ, from fitting sparse unspliced
    // counts absolutely), so it is DEMOTED to `velocity_increment.parquet` (diagnostic).
    // Nascent state = θ + v is derivable, not written.
    let h = args.model.embedding_dim;
    let n = unified.barcodes.len();
    // One `cell × H` parquet writer, shared by the velocity outputs below.
    //
    // These do NOT go through `ge::save_outputs_named`, so the QC keep set has
    // to be applied here as well — otherwise `velocity.parquet` would keep every
    // cell while `cell_embedding.parquet` dropped some, and `faba lineage`
    // (which pairs them elementwise) would silently align the wrong rows.
    let write_cell = |suffix: &str, data: Vec<f32>| -> anyhow::Result<()> {
        let (t, names) = match qc_keep.as_deref() {
            Some(keep) => {
                let mut buf = Vec::with_capacity(keep.len() * h);
                let mut nm = Vec::with_capacity(keep.len());
                for &i in keep {
                    buf.extend_from_slice(&data[i * h..(i + 1) * h]);
                    nm.push(unified.barcodes[i].clone());
                }
                (Tensor::from_vec(buf, (keep.len(), h), &cpu)?, nm)
            }
            None => (
                Tensor::from_vec(data, (n, h), &cpu)?,
                unified.barcodes.clone(),
            ),
        };
        ge::save_embedding(
            &format!("{}.{suffix}.parquet", args.out),
            &t,
            &names,
            "cell",
        )
        .with_context(|| format!("save {suffix}"))?;
        Ok(())
    };
    let operator_velocity = {
        let vars = out.varmap.data().lock().unwrap();
        match (vars.get("beta"), vars.get("delta")) {
            (Some(beta), Some(delta)) => {
                let beta_t = beta.as_tensor().to_device(&cpu)?;
                let delta_t = delta.as_tensor().to_device(&cpu)?;
                let n_g = beta_t.dim(0)?;
                let beta_g = beta_t.flatten_all()?.to_vec1::<f32>()?;
                let delta_g = delta_t.flatten_all()?.to_vec1::<f32>()?;
                let theta = out
                    .model
                    .e_cell
                    .to_device(&cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                // λ=1e-2 (Gram-trace-scaled) conditions the h×h lin-solve; never an inverse.
                let p = ge::cell_projection::velocity_operator(&beta_g, &delta_g, n_g, h, 1e-2);
                // `v = P·θ` is already mean-zero: phase 2 gauge-fixes `θ` to `θ̄ = 0`
                // over the solved cells, and the operator is linear, so
                // `v̄ = P·θ̄ = 0`. (This used to subtract `P·θ̄` explicitly, back when
                // phase 2 left the common mode in `θ`.)
                Some(ge::cell_projection::apply_velocity_operator(
                    &theta, &p, n, h,
                ))
            }
            _ => None,
        }
    };
    match operator_velocity {
        Some(vel) => {
            write_cell("velocity", vel)?;
            info!(
                "wrote {}.velocity.parquet (embedding-space velocity v=P·θ; β_g+δ_g operator, no raw counts)",
                args.out
            );
            // The per-cell increment δ travels alongside the operator velocity.
            if let Some(delta_c) = &out.cell_velocity {
                write_cell("velocity_increment", delta_c.clone())?;
                info!(
                    "wrote {}.velocity_increment.parquet (analytic δ_c increment; \
                     shrinkage-prone (δ_c ≈ −0.5·θ), diagnostic only)",
                    args.out
                );
            }
        }
        // No δ_g dictionary (--delta-l2 = 0): fall back to the raw increment for velocity.parquet.
        None => {
            if let Some(velocity) = &out.cell_velocity {
                write_cell("velocity", velocity.clone())?;
                log::warn!(
                    "wrote {}.velocity.parquet from the raw increment δ_c (no δ_g dictionary for the \
                     operator — enable --delta-l2 for the fate-faithful embedding-space velocity)",
                    args.out
                );
            }
        }
    }

    // Lineage (experimental): dump the per-level pseudobulk states — identity θ_pb and
    // velocity δ_pb — so the structure can be inspected / scored and consumed by the
    // phase-2 cell lift. Only present when `--lineage-dag` ran on a β-sharing model.
    if let Some(pbv) = &out.pb_velocity {
        for (i, lvl) in pbv.iter().enumerate() {
            let np = lvl.n_pb;
            let pb_names: Vec<Box<str>> = (0..np)
                .map(|p| format!("pb_{i}_{p}").into_boxed_str())
                .collect();
            let th = candle_util::candle_core::Tensor::from_vec(lvl.theta.clone(), (np, h), &cpu)?;
            ge::save_embedding(
                &format!("{}.pb_theta_l{i}.parquet", args.out),
                &th,
                &pb_names,
                "pb",
            )
            .context("save pb theta")?;
            let dl = candle_util::candle_core::Tensor::from_vec(lvl.delta.clone(), (np, h), &cpu)?;
            ge::save_embedding(
                &format!("{}.pb_velocity_l{i}.parquet", args.out),
                &dl,
                &pb_names,
                "pb",
            )
            .context("save pb velocity")?;
        }
        info!("wrote pb-level θ/δ for {} level(s)", pbv.len());
    }

    // cell-lift (phase-2 cell-lineage): per-cell pseudotime τ_c + landmark ambiguity
    // (`{out}.dag_pseudotime.parquet`) and per-cell fate over the terminal pb nodes
    // (`{out}.dag_fate.parquet`). Evaluation-only — lifted from the finest pb trajectory
    // onto every cell. Present only when `--lineage-dag` produced a readout. These feed
    // `faba lineage --root-from-gem` as an informed backbone. The `dag_` prefix keeps
    // them distinct from `faba lineage`'s own `{out}.pseudotime.parquet` (Slingshot),
    // so the two can share an output prefix without clobbering each other.
    if let Some(lin) = &out.cell_lineage {
        use matrix_util::traits::IoOps;
        // pseudotime + ambiguity, two named columns, keyed on barcodes.
        let mut pt = Vec::with_capacity(n * 2);
        for c in 0..n {
            pt.push(lin.tau[c]);
            pt.push(lin.ambiguity[c]);
        }
        let pt_t = candle_util::candle_core::Tensor::from_vec(pt, (n, 2), &cpu)?;
        let pt_cols = [
            Box::<str>::from("pseudotime"),
            Box::<str>::from("ambiguity"),
        ];
        pt_t.to_parquet_with_names(
            &format!("{}.dag_pseudotime.parquet", args.out),
            (Some(&unified.barcodes), Some("cell")),
            Some(&pt_cols),
        )?;
        info!(
            "wrote {}.dag_pseudotime.parquet (per-cell τ + landmark ambiguity; pb level {})",
            args.out, lin.level
        );

        // fate: one column per terminal pb node (empty when no terminal exists,
        // e.g. a single-fate or edge-free trajectory — skip the write then).
        let k = lin.terminals.len();
        if k > 0 {
            let fate_t =
                candle_util::candle_core::Tensor::from_vec(lin.fate.clone(), (n, k), &cpu)?;
            let fate_cols: Vec<Box<str>> = lin
                .terminals
                .iter()
                .map(|t| format!("fate_pb{t}").into_boxed_str())
                .collect();
            fate_t.to_parquet_with_names(
                &format!("{}.dag_fate.parquet", args.out),
                (Some(&unified.barcodes), Some("cell")),
                Some(&fate_cols),
            )?;
            info!(
                "wrote {}.dag_fate.parquet (per-cell fate over {} terminal pb node(s))",
                args.out, k
            );
        }
    }

    // Unsupervised per-run structural stats (`{out}.lineage_qc.json`): the DESCRIPTIVE
    // structure (root/terminal counts, top-source reach, velocity coherence, placement
    // ambiguity) that characterizes the trajectory. No ground truth needed, and no coarse
    // one-word verdict — `--root-from-gem` reads `n_terminals` directly to skip a
    // structureless DAG.
    if let Some(qc) = &out.lineage_qc {
        // Through serde rather than a format string: these are all `{:.4}`
        // floats, and a non-finite one renders as a bare `NaN`, which is not
        // JSON. `root::from_gem` parses this file with `.ok()?`, so an invalid
        // write degrades silently to "no signal" instead of erroring — serde
        // emits `null` for a non-finite float, which parses.
        let json = serde_json::json!({
            "n_roots": qc.n_roots,
            "n_terminals": qc.n_terminals,
            "top_source_reach": qc.root_decisiveness,
            "velocity_coherence": qc.velocity_coherence,
            "mean_ambiguity": qc.mean_ambiguity,
            "refine_likelihood": qc.likelihood,
        });
        std::fs::write(
            format!("{}.lineage_qc.json", args.out),
            format!("{}\n", serde_json::to_string_pretty(&json)?),
        )
        .with_context(|| format!("writing {}.lineage_qc.json", args.out))?;
        info!(
            "lineage-DAG structure: {} root(s), {} terminal(s), top-source reach {:.2}, \
             velocity-coherence {:.2}, mean-ambiguity {:.2} → {}.lineage_qc.json",
            qc.n_roots,
            qc.n_terminals,
            qc.root_decisiveness,
            qc.velocity_coherence,
            qc.mean_ambiguity,
            args.out,
        );
    }

    // The posterior no longer runs here. It is phase-1's own sampler now
    // (`FitConfig::pb_posterior`), so it happens INSIDE `fit`, against the
    // pseudobulk side. gem's β-sharing feature side is not wired to it yet — see
    // the warning `fit` emits — so a `--posterior` on this path is currently a
    // no-op rather than a post-hoc pass over the finished fit.

    // Say what produced this prefix. gem's tables share names and shapes with
    // gem-encoder's while meaning something different — `cell_embedding.parquet`
    // is Euclidean here and a topic membership there — so a downstream step
    // handed only the prefix would otherwise have to guess. `latent` records
    // that these coordinates are NOT log θ: nothing downstream should `exp()`
    // them.
    let mut extra = serde_json::Map::new();
    extra.insert(
        "latent".into(),
        crate::manifest::Latent::Embedding.as_str().into(),
    );
    // Record the posterior tables when they were produced, so a consumer handed
    // only the prefix can tell a calibrated selection is available without
    // stat()-ing for filenames. Absent on a plain run, which is the honest signal.
    if let Some(plan) = posterior_plan {
        extra.insert(
            "posterior".into(),
            serde_json::json!({ "draws": plan.n_samples }),
        );
    }
    crate::manifest::write(&args.out, crate::manifest::RunKind::Embedding, extra)?;

    info!(
        "done (gem — raw spliced identity θ + raw velocity increment δ over the bge engine) — prefix '{}'",
        args.out
    );
    Ok(())
}

/// Split a gem feature row `{gene}/count/{spliced|unspliced}` into its gene key and
/// whether it is the unspliced track. Rows not matching that shape fall back to
/// `(whole name, spliced)` — defensive; genes-only input is all count rows.
/// Mean row-sum of a `[n_anchors × h]` PIP table: the expected number of dims an
/// anchor loads, as inferred. `0` for an empty table.
fn mean_dims(pip: &[f32], h: usize) -> f64 {
    let rows = (pip.len() / h.max(1)).max(1);
    f64::from(pip.iter().sum::<f32>()) / rows as f64
}

fn split_count_row(name: &str) -> (&str, bool) {
    match name.rsplit_once("/count/") {
        Some((gene, suffix)) => (gene, suffix == "unspliced"),
        None => (name, false),
    }
}

/// Intern each row's gene key (see [`split_count_row`]) to a dense gene id. Returns
/// `(row_to_gene, gene_names)`, `gene_names[gid]` the first-seen key for gene `gid`
/// (id order). Single source of the β-sharing gene-identity map — used by the HVG
/// gene filter (pre-subset) and [`build_splice_factor`] (post-subset).
fn intern_gene_keys(feature_names: &[Box<str>]) -> (Vec<u32>, Vec<Box<str>>) {
    let mut gene_ids: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut row_to_gene = Vec::with_capacity(feature_names.len());
    let mut gene_names: Vec<Box<str>> = Vec::new();
    for name in feature_names {
        let gene = split_count_row(name).0;
        // Borrow-first: only allocate a `Box<str>` key on a genuinely new gene.
        let gid = match gene_ids.get(gene) {
            Some(&gid) => gid,
            None => {
                let gid = gene_ids.len() as u32;
                gene_ids.insert(gene.into(), gid);
                gene_names.push(gene.into());
                gid
            }
        };
        row_to_gene.push(gid);
    }
    (row_to_gene, gene_names)
}

/// Build the per-gene β-sharing feature factorization + the id-ordered gene names.
/// Each row is `{gene}/count/{spliced|unspliced}`; rows sharing a `{gene}` key map to
/// one gene id (so both tracks embed as `β_g`), and the `unspliced` rows are flagged
/// so phase 2 can split each cell's edges (identity θ from spliced, velocity increment
/// δ from unspliced).
fn build_splice_factor(
    feature_names: &[Box<str>],
) -> (graph_embedding_util::FeatFactorSpec, Vec<Box<str>>) {
    let (row_to_gene, gene_names) = intern_gene_keys(feature_names);
    let unspliced_rows: Vec<bool> = feature_names.iter().map(|n| split_count_row(n).1).collect();
    (
        graph_embedding_util::FeatFactorSpec {
            row_to_gene,
            unspliced_rows,
        },
        gene_names,
    )
}

fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    // Fail on an ambiguous / empty gene spec before any I/O.
    args.genes()?;
    anyhow::ensure!(
        args.model.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.model.embedding_dim
    );
    Ok(())
}
