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

    run_gem_genes_bge(args, feature_kind, batch_files)
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

    // Full-backend gene identity, captured BEFORE any subsetting. `subset_features`
    // narrows the compact axis but never touches the backend, so these stay valid
    // across both fit passes and give the post-hoc projection the complement of
    // whatever survived: the `--n-hvg` remainder plus the feature-null drops.
    let backend_feature_names = unified
        .count_backend()
        .row_names()
        .context("backend row names")?;
    let (backend_row_to_gene, backend_gene_names) = intern_gene_keys(&backend_feature_names);
    let backend_unspliced_rows: Vec<bool> = backend_feature_names
        .iter()
        .map(|n| split_count_row(n).1)
        .collect();

    // Optional gene-level HVG feature filter (like `senna bge`): select the top-N
    // most variable GENES and drop the rest — dropping BOTH the spliced and
    // unspliced rows of a dropped gene together so the β-sharing factorization
    // stays aligned. `subset_features` narrows the dictionary/co-embed (removing
    // the low-detection "empty" genes that pile at the co-embed centre); the
    // uniform `hvg_weights` over the survivors then restricts the pb projection /
    // membership to those genes too. `None` (n_hvg = 0) keeps every gene.
    //
    // `--must-train-features` force-includes a curated panel on top of that cut, at
    // the GENE level (so both splice tracks of a kept gene come along). Loaded only
    // when the HVG cut is on (the softmax gate handles selection otherwise).
    //
    // The softmax feature gate is gem's selector now — a gene with no cell-state signal
    // sends its gate mass to null and contributes ≈0. The old ash-QC / LRT two-pass
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
        let keep_rows: Vec<usize> = (0..row_gene.len())
            .filter(|&r| keep_genes.contains(&(row_gene[r] as usize)))
            .collect();
        let before = unified.n_features();
        unified.subset_features(&keep_rows);
        info!(
            "HVG filter (--n-hvg {}): {} genes → {} kept ({} HVG + {} force-kept; \
             {} → {} feature rows)",
            args.collapse.n_hvg,
            n_genes,
            keep_genes.len(),
            keep_genes.len() - forced,
            forced,
            before,
            unified.n_features()
        );
    }

    // Compute device. gem does a single gated fit (the softmax gate selects
    // features during training), so there is no second pass to reconcile here.
    let dev = args
        .runtime
        .device
        .to_device(args.runtime.device_no)
        .context("candle device init")?;
    info!("compute device = {:?}", dev);

    // Build a `FitConfig` for the CURRENT feature axis of `unified`: the per-gene
    // β-sharing factor is derived from the live feature names, and the δ_g ridge /
    // HVG weights align to that axis. gem fits ONCE (the softmax gate is the
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
    let build_cfg = |unified: &UnifiedData| -> anyhow::Result<(ge::FitConfig, Vec<Box<str>>, f32)> {
        let (factor, gene_names) = build_splice_factor(&unified.feature_names);
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
        let hvg_weights = hvg_on.then(|| vec![1.0f32; unified.n_features()]);
        let cfg = ge::FitConfig {
            embedding_dim: args.model.embedding_dim,
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
            feature_network: None,
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
            // `null_fdr: 0` = restore ALL of them (no ash-null gate; the softmax gate
            // is now the selector). Self-disables when the trained axis is the backend
            // (the default `--n-hvg 0`): nothing is held out. Cell outputs unaffected.
            feature_projection: Some(ge::FeatureProjectionConfig {
                ridge: ge::DEFAULT_PROJECTION_RIDGE,
                calib_ridge: ge::DEFAULT_PROJECTION_CALIB_RIDGE,
                backend_row_to_gene: backend_row_to_gene.clone(),
                backend_unspliced_rows: backend_unspliced_rows.clone(),
                with_velocity: has_unspliced,
                null_fdr: 0.0,
            }),
            // `--nce-objective` (default softmax = InfoNCE: on gem's dense count data
            // the positive competing against its negatives in one distribution
            // separates cell types better than the per-pair logistic SGNS loss).
            nce_objective: args.model.nce_objective.to_ge(),
            // Per-gene softmax feature gate — the SuSiE variational spike-and-slab
            // single-effect, ALWAYS ON. Gates β_g (identity) AND, independently, δ_g
            // (velocity → velocity_selection); null absorber + categorical + Gaussian
            // effect KL at the fixed internal weight. Temperature is the one knob.
            softmax_gate: Some(ge::SoftmaxGateConfig {
                temperature: args.model.feature_softmax_temp,
            }),
        };
        Ok((cfg, gene_names, delta_l2))
    };

    // Single-pass gated fit — the softmax gate selects features DURING training (a
    // junk gene sends its gate mass to null → β̃_g ≈ 0), so there is no LRT null-call
    // or two-pass refit. The `--n-hvg` remainder (if any) is restored post-hoc.
    let (cfg, gene_names, delta_l2) = build_cfg(&unified)?;
    let out = ge::fit(&mut unified, cfg).context("ge::fit (genes bge)")?;
    let n_genes = gene_names.len();

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

    // Post-hoc projected genes: everything the trained axis never saw (the `--n-hvg`
    // remainder plus any feature-null drops), solved against the frozen pseudobulk
    // side. `None`/empty when the trained axis already covered the backend. A gene is
    // projected only when NONE of its rows were trained, so these names are disjoint
    // from `gene_names` and the dictionaries concatenate without a dedup pass.
    let proj = out
        .feature_projection
        .as_ref()
        .filter(|p| !p.gene_ids.is_empty());
    let projected_names: Vec<Box<str>> = proj
        .map(|p| {
            p.gene_ids
                .iter()
                .map(|&g| backend_gene_names[g as usize].clone())
                .collect()
        })
        .unwrap_or_default();
    let mut merged_gene_names = gene_names.clone();
    merged_gene_names.extend(projected_names.iter().cloned());
    if let Some(p) = proj {
        info!(
            "held-out gene projection: {} trained + {} projected = {} genes; \
             {:?} frame calibration on {} trained genes (cosine {:.3}, norm ratio {:.3}, R² {:.3})",
            n_genes,
            projected_names.len(),
            merged_gene_names.len(),
            p.calib.kind,
            p.calib.n_trained,
            p.calib.mean_cosine,
            p.calib.norm_ratio,
            p.calib.r2,
        );
    }

    // Gene-keyed β_g dictionary. `save_outputs` writes the dictionary keyed by feature
    // row (`{gene}/count/spliced|unspliced`), which a gene-symbol marker set cannot
    // match. Read the per-gene `beta` Var directly and save it row-labeled by gene —
    // the spliced/mature gene program that `faba annotate --track spliced` pairs with
    // the cell latent θ. Projected genes are appended, so a marker set sees the whole
    // backend. Symmetric with the δ_g dictionary below.
    let mut trained_norm2: Vec<f32> = Vec::new();
    {
        let vars = out.varmap.data().lock().unwrap();
        if let Some(beta) = vars.get("beta") {
            let beta_t = beta.as_tensor().to_device(&cpu)?;
            let h = beta_t.dim(1)?;
            let mut flat: Vec<f32> = beta_t.flatten_all()?.to_vec1()?;
            trained_norm2 = flat
                .chunks_exact(h)
                .map(|r| r.iter().map(|x| x * x).sum())
                .collect();
            if let Some(p) = proj {
                flat.extend_from_slice(&p.beta);
            }
            let merged = Tensor::from_vec(flat, (merged_gene_names.len(), h), &cpu)?;
            ge::save_embedding(
                &format!("{}.beta_feature_embedding.parquet", args.out),
                &merged,
                &merged_gene_names,
                "feature",
            )
            .context("save β_g feature embedding")?;
            info!(
                "wrote {}.beta_feature_embedding.parquet (per-gene β_g; {} genes, {} of them projected)",
                args.out,
                merged_gene_names.len(),
                projected_names.len()
            );
        }
    }

    // Per-gene splice offset δ_g (`--delta-l2 > 0`): the nascent loading
    // (unspliced e_f = β_g + δ_g). Read the trained `delta` Var from the varmap
    // and save it row-labeled by gene — genes with large ‖δ_g‖ carry a distinct
    // nascent/velocity program; the L2 ridge shrinks the rest toward 0.
    //
    // Projected genes get the analytic increment solved against their unspliced
    // pseudobulk edges, or a zero row when they have no unspliced track. Treat a
    // projected δ_g as low-confidence: the genes HVG dropped are exactly the
    // low-detection ones, so their unspliced signal is thin. Gate on
    // `n_detected_pb` in `{out}.gene_qc.parquet`.
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
            let mut flat: Vec<f32> = d_t.flatten_all()?.to_vec1()?;
            match proj.and_then(|p| p.delta.as_ref()) {
                Some(d) => flat.extend_from_slice(d),
                None => flat.resize(merged_gene_names.len() * h, 0.0),
            }
            let merged = Tensor::from_vec(flat, (merged_gene_names.len(), h), &cpu)?;
            ge::save_embedding(
                &format!("{}.delta_feature_embedding.parquet", args.out),
                &merged,
                &merged_gene_names,
                "feature",
            )
            .context("save δ_g feature embedding")?;
            info!(
                "wrote {}.delta_feature_embedding.parquet (δ_g; {}/{} trained genes with nonzero offset, \
                 {} projected)",
                args.out,
                nz,
                n_genes,
                projected_names.len()
            );
        }
    }

    // Per-gene QC: which genes were trained vs projected, and how much evidence
    // each projected gene actually had. Written separately so the dictionaries stay
    // plain `gene × H` tables that downstream `faba annotate` reads unchanged.
    // The LRT selection scan is retired (the gate is the selector), so trained genes
    // carry no per-gene scan stats — `gene_qc`'s LRT/deviance/detection columns stay
    // NaN for trained rows.
    if let Some(p) = proj {
        save_gene_qc(&merged_gene_names, &trained_norm2, p, &args.out)?;
        write_projection_qc_json(p, &args.out)?;
    }

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
fn split_count_row(name: &str) -> (&str, bool) {
    match name.rsplit_once("/count/") {
        Some((gene, suffix)) => (gene, suffix == "unspliced"),
        None => (name, false),
    }
}

/// Write `{out}.gene_qc.parquet` — one row per gene of the merged
/// `beta_feature_embedding`, in the same order:
///
/// * `trained` — 1 = fit by the model, 0 = projected post-hoc.
/// * `live` — 1 = the gene carries signal above the estimated null. At gem's shipped
///   settings this is 1 for **every** row: trained genes are live by construction, and
///   the projection null gate is off (`null_fdr: 0.0` — the softmax gate is the
///   selector, so the `--n-hvg` remainder is restored whole). It only discriminates if
///   that gate is re-enabled, where a gene called null is **zeroed** so its
///   `beta_feature_embedding` row is all-zero rather than a fabricated direction.
///   To find gated-off genes, read `norm2` instead — the gate parks them at `β̃_g ≈ 0`.
/// * `norm2` — `‖β_g‖²`.
/// * `n_detected_pb` — pseudobulk samples (summed over collapse levels) where the
///   gene reads above the column floor.
/// * `deviance` — Poisson deviance `D_fit` of the gene's solve.
/// * `lrt` — `D_null − D_fit`, the evidence that θ explains the gene at all. This
///   is the statistic `live` tests; rank projected markers by it.
///
/// The last three are **projected-gene only**: the held-out projection's own
/// goodness-of-fit, scored against the final trained frame. A trained gene has no
/// comparable per-gene scan — the softmax gate selects during training rather than
/// by a separate scored pass — so trained rows carry `NaN` there rather than a
/// fabricated `0`. Filter projected markers on `live`, rank them by `lrt`.
fn save_gene_qc(
    merged_gene_names: &[Box<str>],
    trained_norm2: &[f32],
    proj: &graph_embedding_util::FeatureProjection,
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_util::dmatrix_io::DMatrix;
    use matrix_util::traits::IoOps;
    let n_trained = trained_norm2.len();
    let mut m = DMatrix::<f32>::zeros(merged_gene_names.len(), 6);
    for g in 0..n_trained {
        m[(g, 0)] = 1.0;
        m[(g, 1)] = 1.0;
        m[(g, 2)] = trained_norm2[g];
        // No per-gene scan for a trained gene (the gate selects in-training), so the
        // projection-only columns stay NaN rather than a fabricated 0.
        m[(g, 3)] = f32::NAN;
        m[(g, 4)] = f32::NAN;
        m[(g, 5)] = f32::NAN;
    }
    let h = proj.beta.len() / proj.gene_ids.len().max(1);
    for i in 0..proj.gene_ids.len() {
        let g = n_trained + i;
        m[(g, 0)] = 0.0;
        m[(g, 1)] = f32::from(u8::from(proj.live[i]));
        m[(g, 2)] = proj.beta[i * h..(i + 1) * h].iter().map(|x| x * x).sum();
        m[(g, 3)] = proj.n_detected_pb[i] as f32;
        m[(g, 4)] = proj.deviance[i];
        m[(g, 5)] = proj.lrt[i];
    }
    let cols: Vec<Box<str>> = [
        "trained",
        "live",
        "norm2",
        "n_detected_pb",
        "deviance",
        "lrt",
    ]
    .iter()
    .map(|s| (*s).into())
    .collect();
    let path = format!("{out_prefix}.gene_qc.parquet");
    m.to_parquet_with_names(&path, (Some(merged_gene_names), Some("gene")), Some(&cols))
        .with_context(|| format!("writing {path}"))?;
    let n_live = proj.live.iter().filter(|&&l| l).count();
    info!(
        "wrote {path} ({} trained, {} projected of which {} live)",
        n_trained,
        proj.gene_ids.len(),
        n_live
    );
    Ok(())
}

/// Write `{out}.projection_qc.json` — how well the Poisson-MAP frame agreed with
/// the trained NCE frame before calibration, measured on the trained genes.
/// `mean_cosine` near 1 with `norm_ratio` near 1 means the two frames already
/// agreed and the `H×H` map was close to the identity. A low `mean_cosine` means
/// they genuinely disagree: distrust the projected genes rather than reading them.
fn write_projection_qc_json(
    proj: &graph_embedding_util::FeatureProjection,
    out_prefix: &str,
) -> anyhow::Result<()> {
    let c = &proj.calib;
    // Serde, not a format string — same reason as `lineage_qc.json` above: the
    // three floats can be non-finite, and `{:.4}` would emit a bare `NaN` that
    // no JSON parser accepts. It also escapes the `{:?}` calibration rendering
    // instead of splicing it raw between quotes.
    let json = serde_json::json!({
        "n_projected": proj.gene_ids.len(),
        "n_projected_live": proj.live.iter().filter(|&&l| l).count(),
        "n_trained_calibration": c.n_trained,
        "calibration": format!("{:?}", c.kind),
        "mean_cosine": c.mean_cosine,
        "norm_ratio": c.norm_ratio,
        "r2": c.r2,
    });
    let path = format!("{out_prefix}.projection_qc.json");
    std::fs::write(&path, format!("{}\n", serde_json::to_string_pretty(&json)?))
        .with_context(|| format!("writing {path}"))?;
    Ok(())
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
