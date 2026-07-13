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

use faba::gem::args::GemArgs;
use faba::gem::sample_id::{file_sample_id, longest_common_underscore_suffix};

/// Default ridge on the per-gene splice offset δ_g, applied automatically whenever
/// the input carries unspliced rows and the user did not set `--delta-l2`. Keeping a
/// mild ridge on by default means every spliced+unspliced gem run always emits a δ_g
/// dictionary (`{out}.delta_dictionary.parquet`) for downstream `faba annotate
/// --track velocity`, without over-shrinking. Matches the documented `--delta-l2`
/// range (0.01–1.0).
const DEFAULT_DELTA_L2: f32 = 1.0;

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
    // the GENE level (so both splice tracks of a kept gene come along). It is loaded
    // whether or not HVG is on, because it also exempts its genes from the
    // `--feature-null-fdr` drop below — the other gate a gene has to survive.
    let hvg_on = args.collapse.n_hvg > 0;
    let must_train = data_beans_alg::hvg::load_must_train(
        args.collapse.must_train_features.as_deref(),
        hvg_on || args.model.feature_null_fdr > 0.0,
    )?;

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
        let forced = if let Some(must_train) = must_train.as_ref() {
            let forced = must_train.resolve(&genes);
            let added = data_beans_alg::hvg::union_indices(&mut selected, &forced);
            info!(
                "--must-train-features: {added} gene(s) force-added on top of the HVG cut \
                 ({} of the {} matched were already HVGs)",
                forced.len() - added,
                forced.len()
            );
            added
        } else {
            0
        };

        let keep_genes: rustc_hash::FxHashSet<usize> = selected.into_iter().collect();
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

    // Compute device + a single shared Ctrl-C stop handler. `setup_stop_handler`
    // registers a SIGINT handler and a second registration panics, so it must run
    // once and be cloned into each pass's config (the feature-null refine fits twice).
    let dev = args
        .runtime
        .device
        .to_device(args.runtime.device_no)
        .context("candle device init")?;
    info!("compute device = {:?}", dev);

    // Build a `FitConfig` for the CURRENT feature axis of `unified`, so the same
    // builder serves pass 1 (full post-HVG axis) and the post-QC re-fit (null
    // features dropped): the per-gene β-sharing factor is rebuilt from the live
    // feature names, the δ_g ridge / HVG weights realign to the reduced axis, and
    // the Fisher cache self-invalidates on the name mismatch. Mirrors senna bge's
    // two-pass `build_config`. Returns the config plus the axis-derived gene names
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
            fisher_weights_cache: Some(
                format!("{}.fisher_weights.parquet", args.out).into_boxed_str(),
            ),
            feature_network: None,
            feature_embedding_l2: 0.0,
            weight_decay: 0.0,
            max_grad_norm: args.train.max_grad_norm,
            cell_weight_mult: None,
            phase1_cells_per_pb: args.collapse.phase1_cells_per_pb,
            feat_factor: Some(factor),
            delta_l2,
            lineage_dag: args.train.lineage_dag,
            dag_learnable: !args.train.fixed_dag,
            lineage_smooth: args.train.lineage_smooth,
            // Always on, no flag: geu projects whatever backend genes the trained
            // axis is missing. `--n-hvg 0` with no null-QC drops leaves nothing to
            // project, so this self-disables. Runs after phase 2 and reads only the
            // frozen pseudobulk side — the cell outputs are unaffected.
            feature_projection: Some(ge::FeatureProjectionConfig {
                ridge: ge::DEFAULT_PROJECTION_RIDGE,
                calib_ridge: ge::DEFAULT_PROJECTION_CALIB_RIDGE,
                backend_row_to_gene: backend_row_to_gene.clone(),
                backend_unspliced_rows: backend_unspliced_rows.clone(),
                with_velocity: has_unspliced,
                // Same EB null call the two-pass feature QC uses, on both ends:
                // only live trained genes calibrate the frame, and a projected
                // gene indistinguishable from the null is zeroed rather than
                // given a fabricated direction. Null genes never shaped the cell
                // embedding; resurrecting them as marker vectors would only add
                // noise to `faba annotate`.
                null_fdr: args.model.feature_null_fdr,
            }),
        };
        Ok((cfg, gene_names, delta_l2))
    };

    // Pass 1 — fit on the full (post-HVG) feature axis.
    let (cfg, mut gene_names, mut delta_l2) = build_cfg(&unified)?;
    let mut out = ge::fit(&mut unified, cfg).context("ge::fit (genes bge)")?;

    // Empirical-Bayes feature-null QC — the same shared engine call `senna bge` uses.
    // Each feature row's ‖e_feat‖² (its materialized β_g) is tested against an
    // estimated null (a gene the model never moved keeps ‖e_feat‖² ~ σ²·χ²_ν); rows
    // null at the FDR are the untrained low-detection background (the 'empty' genes
    // that pile at the co-embed centre) and are DROPPED, then the model re-fits on the
    // live axis (two-pass refine). β-sharing means a null gene drops both its tracks
    // together. The automatic, data-driven analog of the manual `--n-hvg` cut. Off
    // with `--feature-null-fdr 0`.
    if args.model.feature_null_fdr > 0.0 {
        let (n, h) = out.model.e_feat.dims2()?;
        let e_feat: Vec<f32> = out
            .model
            .e_feat
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let null = ge::null_call::embedding_null_call(&e_feat, n, h, args.model.feature_null_fdr);
        info!(
            "feature-null QC — σ̂²={:.4}, ν̂={:.1}/{}, π̂₀={:.2}; {} / {} rows null at FDR {} → {}.feature_qc.parquet",
            null.sigma2, null.eff_dof, h, null.pi0, n - null.n_live, n, args.model.feature_null_fdr, args.out,
        );
        save_feature_qc(&e_feat, n, h, &unified.feature_names, &null, &args.out)?;
        let mut live: Vec<usize> = (0..n).filter(|&i| null.live[i]).collect();

        // The all-null case is a degenerate-fit detector, so it has to read the RAW null
        // call — BEFORE any `--must-train-features` rescue. Rescuing first would make
        // `live` non-empty, the detector would never fire, and the refit would drop
        // every row EXCEPT the force-kept ones: the panel would eat the dictionary.
        if live.is_empty() {
            log::warn!(
                "feature-null QC flagged all {n} rows null (degenerate fit); keeping pass-1 fit."
            );
        } else {
            // `--must-train-features` outranks the null call as well as the HVG cut: a
            // gene the user named stays in the dictionary even when the model never moved
            // its β off init. Resolved gene-level again (not row-level) so a rescued gene
            // keeps BOTH its splice tracks — dropping one would break the β-sharing
            // pairing. The genes are still reported null in `{out}.feature_qc.parquet`.
            if let Some(must_train) = must_train.as_ref() {
                let (row_gene, genes) = intern_gene_keys(&unified.feature_names);
                let keep_genes: rustc_hash::FxHashSet<usize> =
                    must_train.resolve_quiet(&genes).into_iter().collect();
                let forced_rows: Vec<usize> = (0..row_gene.len())
                    .filter(|&r| keep_genes.contains(&(row_gene[r] as usize)))
                    .collect();
                let rescued = data_beans_alg::hvg::union_indices(&mut live, &forced_rows);
                if rescued > 0 {
                    log::warn!(
                        "--must-train-features: {rescued} feature row(s) were called null at FDR \
                         {} but kept anyway — those genes carry no signal in this data, so their \
                         β stays near its init. See {}.feature_qc.parquet.",
                        args.model.feature_null_fdr,
                        args.out
                    );
                }
            }

            if live.len() < n {
                info!(
                    "two-pass refine: dropping {} null feature rows, re-fitting on {} live rows.",
                    n - live.len(),
                    live.len()
                );
                unified.subset_features(&live);
                let (cfg2, gn2, dl2) = build_cfg(&unified)?;
                out = ge::fit(&mut unified, cfg2).context("ge::fit refit (feature-null)")?;
                gene_names = gn2;
                delta_l2 = dl2;
            }
        }
    }
    let n_genes = gene_names.len();

    // NOTE: NO softmax co-embedding is written. (1) The gene↔cell co-embedding
    // (`{out}.feature_embedding.parquet`) is dropped: cell-type identity is carried by
    // a *few* high-contrast marker genes — a sparse, heavy-tailed distribution of gene
    // norms ‖β_g‖ — and a softmax barycenter that drops every gene onto the cell
    // manifold has to flatten that contrast, so the great majority of genes land with
    // no meaningful cell location: NOT EVERY GENE CAN BE CO-EMBEDDED. Gene↔cell
    // co-embedding and sharp cell clusters are the same degree of freedom pulling
    // opposite ways — we keep the sharp clusters. (2) No velocity "driver" co-embed
    // either: a per-gene velocity readout, if wanted, is the in-model δ_g (`--delta-l2`
    // → `{out}.delta_dictionary.parquet`), not a post-hoc average. The feature embedding
    // is keyed by *feature row* (`{gene}/count/{spliced|unspliced}`); the gene-keyed β_g
    // dictionary below is what marker-based `faba annotate` consumes.
    //
    // gem writes the EXPLICIT names (`{out}.cell_embedding.parquet` /
    // `{out}.feature_embedding.parquet`) rather than senna's `latent` / `dictionary` —
    // gem is not a topic model, so "latent"/"dictionary" said less than the tables are.
    // `faba {lineage, annotate}` read these names.
    let cpu = candle_util::candle_core::Device::Cpu;
    ge::save_outputs_named(
        &out.model,
        &ge::OutputContext {
            feature_names: &unified.feature_names,
            barcodes: &unified.barcodes,
            cell_keep_idx: None,
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
                &format!("{}.beta_dictionary.parquet", args.out),
                &merged,
                &merged_gene_names,
                "gene",
            )
            .context("save β_g dictionary")?;
            info!(
                "wrote {}.beta_dictionary.parquet (per-gene β_g; {} genes, {} of them projected)",
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
                &format!("{}.delta_dictionary.parquet", args.out),
                &merged,
                &merged_gene_names,
                "gene",
            )
            .context("save δ_g dictionary")?;
            info!(
                "wrote {}.delta_dictionary.parquet (δ_g; {}/{} trained genes with nonzero offset, \
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
    if let Some(p) = proj {
        save_gene_qc(&merged_gene_names, &trained_norm2, p, &args.out)?;
        write_projection_qc_json(p, &args.out)?;
    }

    // Splice output (β-sharing): the identity `latent` above is the RAW spliced θ.
    // On the cell axis we also emit the RAW velocity increment δ (`velocity.parquet`),
    // same cell×H layout / barcodes as the latent — magnitude = speed, direction =
    // velocity, no post-hoc unit-norm. The nascent state is just θ + δ = latent +
    // velocity (derivable, not written). Per-gene velocity, if wanted, is the in-model
    // δ_g (`--delta-l2`, written to delta_dictionary.parquet) — not a post-hoc average.
    let h = args.model.embedding_dim;
    let n = unified.barcodes.len();
    if let Some(velocity) = &out.cell_velocity {
        let vel_t = candle_util::candle_core::Tensor::from_vec(velocity.clone(), (n, h), &cpu)?;
        ge::save_embedding(
            &format!("{}.velocity.parquet", args.out),
            &vel_t,
            &unified.barcodes,
            "cell",
        )
        .context("save velocity")?;
        info!(
            "wrote {}.velocity.parquet (raw cell velocity increment δ; ‖δ‖ = speed)",
            args.out
        );
    }

    // Lineage-DAG (experimental): dump the per-level pseudobulk states — identity
    // θ_pb, velocity δ_pb, and (learned-DAG) the learned DAG adjacency W — so the structure
    // can be inspected / scored and consumed by the phase-2 cell lift. Only present
    // when `--lineage-dag` ran on a β-sharing model.
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
            if let Some(w) = out
                .pb_dag_w
                .as_ref()
                .and_then(|d| d.get(i))
                .filter(|w| !w.is_empty())
            {
                let wt = candle_util::candle_core::Tensor::from_vec(w.clone(), (np, np), &cpu)?;
                ge::save_embedding(
                    &format!("{}.pb_dag_l{i}.parquet", args.out),
                    &wt,
                    &pb_names,
                    "pb",
                )
                .context("save pb dag W")?;
            }
        }
        info!(
            "wrote pb-level θ/δ{} for {} level(s)",
            if out.pb_dag_w.is_some() {
                " + learned DAG W"
            } else {
                ""
            },
            pbv.len()
        );
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
        let json = format!(
            "{{\n  \"n_roots\": {},\n  \"n_terminals\": {},\n  \"top_source_reach\": {:.4},\n  \
             \"velocity_coherence\": {:.4},\n  \"mean_ambiguity\": {:.4},\n  \
             \"refine_likelihood\": {:.4}\n}}\n",
            qc.n_roots,
            qc.n_terminals,
            qc.root_decisiveness,
            qc.velocity_coherence,
            qc.mean_ambiguity,
            qc.likelihood,
        );
        std::fs::write(format!("{}.lineage_qc.json", args.out), json)
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
/// `beta_dictionary`, in the same order:
///
/// * `trained` — 1 = fit by the model, 0 = projected post-hoc.
/// * `live` — 1 = the gene carries signal above the estimated null. Trained genes
///   reaching this point are live by construction (the two-pass QC already
///   dropped the null ones); a projected gene called null was **zeroed**, so its
///   `beta_dictionary` row is all-zero rather than a fabricated direction.
/// * `norm2` — `‖β_g‖²`.
/// * `n_detected_pb` — pseudobulk samples (summed over collapse levels) where the
///   gene reads above the column floor.
/// * `deviance` — Poisson deviance of the projection solve.
/// * `lrt` — `D_null − D_fit`, the evidence that θ explains the gene at all. This
///   is the statistic `live` tests; rank projected markers by it.
///
/// The last three are the projection's own goodness-of-fit and do not exist for a
/// trained gene, so those cells are `NaN` on trained rows rather than a fabricated
/// `0`. Filter projected markers on `live`, rank them by `lrt`.
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
    let json = format!(
        "{{\n  \"n_projected\": {},\n  \"n_projected_live\": {},\n  \
         \"n_trained_calibration\": {},\n  \"calibration\": \"{:?}\",\n  \
         \"mean_cosine\": {:.4},\n  \"norm_ratio\": {:.4},\n  \"r2\": {:.4}\n}}\n",
        proj.gene_ids.len(),
        proj.live.iter().filter(|&&l| l).count(),
        c.n_trained,
        c.kind,
        c.mean_cosine,
        c.norm_ratio,
        c.r2
    );
    let path = format!("{out_prefix}.projection_qc.json");
    std::fs::write(&path, json).with_context(|| format!("writing {path}"))?;
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

/// Write the feature-null QC report `{out}.feature_qc.parquet` (per feature row:
/// `norm2` = ‖e_feat‖², `live` = 1 kept / 0 dropped), the same schema `senna bge`
/// emits. `e_feat` is row-major `[n × h]`.
fn save_feature_qc(
    e_feat: &[f32],
    n: usize,
    h: usize,
    feature_names: &[Box<str>],
    null: &graph_embedding_util::null_call::NullCall,
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_util::dmatrix_io::DMatrix;
    use matrix_util::traits::IoOps;
    let mut m = DMatrix::<f32>::zeros(n, 2);
    for f in 0..n {
        m[(f, 0)] = e_feat[f * h..(f + 1) * h].iter().map(|&x| x * x).sum();
        m[(f, 1)] = f32::from(u8::from(null.live[f]));
    }
    let cols: Vec<Box<str>> = ["norm2", "live"].iter().map(|s| Box::from(*s)).collect();
    m.to_parquet_with_names(
        &format!("{out_prefix}.feature_qc.parquet"),
        (Some(feature_names), Some("feature")),
        Some(&cols),
    )?;
    Ok(())
}

fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    // Fail on an ambiguous / empty gene spec before any I/O.
    args.genes()?;
    anyhow::ensure!(
        args.model.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.model.embedding_dim
    );
    anyhow::ensure!(
        (0.0..1.0).contains(&args.model.feature_null_fdr),
        "--feature-null-fdr must be in [0, 1) (got {})",
        args.model.feature_null_fdr
    );
    Ok(())
}
