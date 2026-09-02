//! `senna bge` (Bipartite Graph Embedding) — thin clap + run-manifest
//! wrapper around the `graph-embedding-util` engine.
//!
//! Previously `senna gbe`; renamed to clarify that the bipartite (cell ×
//! feature) graph is *built* internally from expression counts. The
//! sibling `senna fne` (Feature Network Embedding) is the graph-input
//! companion that consumes an explicit feature-feature edge list.
//! `gbe` remains a clap alias for one release cycle.
//!
//! All algorithmic work lives in `graph_embedding_util`. This file
//! exists only to translate `BgeArgs` → `FitConfig` and write senna's
//! run manifest after training.
//!
//! `--feature-network` (SGC smoothing of `E_feat` through a feature-feature
//! edge list) was removed along with its implementation: it saw no practical
//! use, and its six flags dominated bge's surface. `senna topic` /
//! `masked-topic` keep their own, unrelated feature-network restriction.

use crate::embed_common::*;
use data_beans_alg::hvg::select_hvg_streaming;
use graph_embedding_util as ge;

pub(crate) mod args;
mod resolve_etm;
pub(crate) mod score;

pub use args::BgeArgs;
use args::MultiomeFile;
use resolve_etm::resolve_etm_topics;

/// One parsed `--multiome` file entry: `(optional modality label, file path)`.
pub fn fit_bge(args: &BgeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    anyhow::ensure!(
        args.pb_reference.is_none() || args.multiome.is_empty(),
        "a pb_reference and --multiome do not compose: multiome loads with union column \
         alignment, which gives no guarantee the carried pseudobulks stay contiguous at the \
         end — and their weights are applied by position. Drop --use-pb-reference for this \
         round and let it re-collapse."
    );
    // Reconcile --posterior with --mcmc/--jitter BEFORE any I/O, so a
    // contradictory pair fails in the first second rather than after the fit.
    let posterior_plan = args.posterior.resolve(args.seed)?;

    // Input files: positional (single-modality) OR --multiome modality groups.
    // Each --multiome occurrence is one group; comma-separated files within it.
    let is_multiome = !args.multiome.is_empty();

    // Parse each --multiome occurrence (one group) into its files, honoring an
    // optional `label=file` prefix that names the modality. The label (or, when
    // omitted, the within-group position `m{pos}`) namespaces that file's
    // features as `{name}/{label}` so distinct modalities stay on separate rows
    // (e.g. spliced vs unspliced `TSPAN6`), while the same modality across
    // samples (same label/position) still merges. Each `(label, file)` pair is
    // (Option<modality label>, file path).
    let multiome_groups: Vec<Vec<MultiomeFile>> = args
        .multiome
        .iter()
        .map(|s| {
            s.split(',')
                .map(|tok| match tok.split_once('=') {
                    Some((label, file)) if !label.is_empty() && !file.is_empty() => {
                        (Some(label.into()), file.into())
                    }
                    _ => (None, tok.into()),
                })
                .collect()
        })
        .collect();

    // Multiome mixes gene rows (RNA) and locus rows (ATAC peaks) on one axis,
    // so canonicalize per-name via `Mixed` (genes → gene rule, `chrX:s-e` →
    // locus rule).
    //
    // The rule is fixed rather than exposed: `--feature-name-delim` /
    // `--feature-name-exact` were CLI knobs whose defaults ('_', fuzzy) were the
    // only settings anyone used, and `_` is the separator every loader in this
    // workspace already writes (`ENSG…_TSPAN6`). `senna gem` and `senna predict`
    // still expose their own overrides where query-vs-reference name bridging is
    // an actual concern.
    let feature_kind = if is_multiome {
        ge::FeatureNameKind::Mixed
    } else {
        ge::FeatureNameKind::Gene { delim: '_' }
    };

    // Flatten groups to get the per-file slice passed to load_unified_data,
    // plus the parallel per-file modality suffix (label, else `m{within-group
    // position}`) used to namespace features as `{name}/{suffix}`.
    let data_files_flat: Vec<Box<str>>;
    let feature_suffix: Option<Vec<Box<str>>>;
    let data_files: &[Box<str>] = if is_multiome {
        data_files_flat = multiome_groups
            .iter()
            .flat_map(|g| g.iter().map(|(_, file)| file.clone()))
            .collect();
        feature_suffix = Some(
            multiome_groups
                .iter()
                .flat_map(|g| {
                    g.iter().enumerate().map(|(pos, (label, _))| {
                        label
                            .clone()
                            .unwrap_or_else(|| format!("m{pos}").into_boxed_str())
                    })
                })
                .collect(),
        );
        if multiome_groups.len() > 1 {
            let counts = multiome_groups
                .iter()
                .map(|g| g.len().to_string())
                .collect::<Vec<_>>()
                .join("+");
            info!(
                "--multiome: {} groups, {} total files ({})",
                multiome_groups.len(),
                data_files_flat.len(),
                counts
            );
        }
        if let Some(suf) = feature_suffix.as_ref() {
            info!(
                "--multiome: namespacing features as {{name}}/{{modality}} \n\
		 (per-file modality: {})",
                suf.iter()
                    .map(std::convert::AsRef::as_ref)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        &data_files_flat
    } else {
        feature_suffix = None;
        anyhow::ensure!(
            !args.data_files.is_empty(),
            "no input files: pass single-modality files positionally, or multiome \n\
             groups via `--multiome rna.zarr,atac.zarr [--multiome rna2.zarr,atac2.zarr ...]`"
        );
        &args.data_files
    };

    let effective_hvg =
        crate::hvg::resolve_multiome_with_hvg(is_multiome, data_files.len(), &args.hvg);
    let effective_multiome = effective_hvg.multiome;
    let column_alignment = if effective_multiome {
        data_beans::sparse_io_vector::ColumnAlignment::Union
    } else {
        data_beans::sparse_io_vector::ColumnAlignment::Disjoint
    };

    let batch_files = crate::senna_input::effective_batch_files(
        args.collapse.ignore_batch,
        args.batch_files.as_deref(),
    );

    let mut unified = ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: data_files.to_vec(),
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        feature_kind: Some(feature_kind.clone()),
        preload: args.preload_data,
        column_alignment,
        per_file_feature_suffix: feature_suffix,
        // senna uses disjoint barcodes per group; no per-file barcode suffix.
        ..Default::default()
    })?;

    // Carried pseudobulks: registered exactly like every other family — their
    // cell counts become column multiplicities on the count backend (weights
    // keyed on the PBREF_-tail layout, verified by name), so the collapse and
    // every downstream statistic treat a carried column as the cells it
    // stands for. The CELL axis of phase 1 sees them unweighted, as ~1k
    // prototype profiles among the new cells — a documented approximation;
    // the pb axis carries the properly weighted signal.
    if let Some(r) = args.pb_reference.as_ref() {
        let v = unified.count_backend_mut();
        let names = v.column_names()?;
        let w = crate::pb_reference::weights_for(&r.cell_counts, &names)?;
        v.register_column_multiplicity(&w)?;
        info!(
            "Column multiplicity: {} carried pseudobulks stand for {} cells",
            r.cell_counts.len(),
            r.cells_represented() as usize,
        );
    }

    // Guard barcode identity across groups: disjoint barcodes, so Union
    // loading never merges cells from different samples. No-op for one group.
    if is_multiome {
        let group_sizes: Vec<usize> = multiome_groups.iter().map(Vec::len).collect();
        ge::validate_multiome_groups(&group_sizes, &unified.barcodes, &unified.cell_modality)?;
    }

    /////////////////////////////
    // Cell QC (output filter) //
    /////////////////////////////
    // The `--qc` near-empty floor + MAD-outlier call is an OUTPUT filter: every
    // cell + edge still informs the joint embedding / feature dictionary, but
    // QC-failed cells are dropped from the archetypal analysis and all per-cell
    // outputs via a write-time `select_rows`. (The separate EB empty-droplet
    // call below, when `--cell-null-fdr > 0`, instead masks empties out of the
    // backend and re-fits.) Computed on the full-feature unified count backend,
    // so n_genes is the per-cell detected-feature count across all modalities.
    let qc_keep_idx: Option<Vec<usize>> = if let Some(cfg) = args.qc.to_config() {
        if cfg.feature_min_cells > 0 {
            log::warn!(
                "--qc-feature-min-cells is ignored by bge (cell-only QC; the \
                 dictionary keeps all features)"
            );
        }
        // Carried pseudobulks are processed outputs, not cells: they must
        // neither receive a QC verdict nor sit inside the MAD band statistics
        // (a few hundred smooth averages drag the robust center and can
        // guillotine every real cell as an "outlier" — measured 400 of 400).
        let exempt: Option<Vec<bool>> = args.pb_reference.as_ref().map(|r| {
            let n = unified.n_cells();
            let n_real = n.saturating_sub(r.cell_counts.len());
            (0..n).map(|c| c >= n_real).collect()
        });
        let report = data_beans::qc_lib::compute_qc_exempting(
            unified.count_backend(),
            &cfg,
            args.block_size,
            exempt.as_deref(),
        )?;
        let keep = report.emit_idx_unmasked();
        info!(
            "QC: {} / {} cells kept for output ({} near-empty, {} MAD-outlier dropped)",
            keep.len(),
            unified.n_cells(),
            report.near_empty.iter().filter(|&&e| e).count(),
            report.n_cells_dropped,
        );
        Some(keep)
    } else {
        None
    };

    // Carried pseudobulks train the model but are not cells; hold them out of
    // every per-cell artifact (bge does no column masking, so the qc indices
    // are already in original column order — the same space `exclude_carried`
    // composes on).
    let qc_keep_idx = crate::pb_reference::exclude_carried(
        args.pb_reference.as_ref(),
        unified.n_cells(),
        qc_keep_idx,
    );

    // HVG → projection weights (no longer subsets the feature axis).
    // Mirrors senna topic: HVG down-weights uninformative genes for the
    // random projection / pb sketching only; collapse + supergene
    // coarsening + training read all genes. Caller passes the weights
    // through `FitConfig.hvg_weights`.
    // Full-axis HVG weights (backend-row indexed, identity-aligned to the
    // current feature axis). Subset through `feature_to_backend_row` inside
    // `build_config` so the same vector serves pass 1 (full) and the post-QC
    // pass 2 (null features dropped). The feature network is rebuilt per pass
    // (its graph is aligned to the live feature-name axis), so it lives in the
    // closure rather than here.
    //
    // `--must-train-features` is a curated panel kept in the HVG-weighted set.
    let hvg_enabled = effective_hvg.selection_on();
    let must_train = crate::hvg::load_must_train(effective_hvg.must_train_file, hvg_enabled)?;
    let hvg_full: Option<Vec<f32>> = if hvg_enabled {
        let hvg = select_hvg_streaming(
            unified.count_backend(),
            (effective_hvg.n_hvg > 0).then_some(effective_hvg.n_hvg),
            effective_hvg.feature_list_file,
            must_train.as_ref(),
            args.block_size,
        )?;
        Some(hvg.row_weights(unified.n_features()))
    } else {
        None
    };

    // `--no-refine` is gbe-specific (the other subcommands always refine);
    // otherwise the shared `--pb-refine-*` flags drive RefineParams.
    let refine = if args.no_refine {
        None
    } else {
        Some(args.collapse.pb_refine.to_params())
    };

    // Assemble a `FitConfig` for the CURRENT feature AND cell axes of `unified`,
    // so the same builder serves pass 1 (full axis), the post-QC feature re-fit
    // (null features dropped) and the cell-empty re-fit (empties dropped): HVG
    // weights subset through `feature_to_backend_row`, the feature network
    // reloads against the live feature names, and the cell-indexed bridge weights
    // resolve against the live barcodes/cell axis. Everything else is
    // axis-independent and cloned in.
    let build_config = |unified: &ge::UnifiedData| -> anyhow::Result<ge::FitConfig> {
        let hvg_weights = hvg_full.as_ref().map(|w| {
            unified
                .feature_to_backend_row
                .iter()
                .map(|&i| w[i])
                .collect::<Vec<f32>>()
        });
        // Up-weight matched (multi-modality) cells in the cell-axis sampler so
        // they anchor the cross-modal alignment. No-op outside --multiome.
        let cell_weight_mult: Option<Vec<f32>> =
            if is_multiome && (args.bridge_weight - 1.0).abs() > f32::EPSILON {
                Some(
                    unified
                        .cell_modality
                        .iter()
                        .map(|&m| {
                            if m.count_ones() >= 2 {
                                args.bridge_weight
                            } else {
                                1.0
                            }
                        })
                        .collect(),
                )
            } else {
                None
            };
        Ok(ge::FitConfig {
            embedding_dim: args.embedding_dim,
            // Greedy batch correction against the carried reference, exactly
            // as in the other families — see `MultilevelParams::anchor_batches`.
            anchor_batches: args
                .pb_reference
                .is_some()
                .then(|| vec![crate::pb_reference::REFERENCE_BATCH.into()]),
            bulk_batches: args.collapse.mixture_batch.clone(),
            emit_finest_collapse: args.collapse.emit_pb_reference,
            num_levels: args.collapse.num_levels,
            sort_dim: args.collapse.sort_dim,
            knn_pb_samples: args.collapse.knn_cells,
            num_opt_iter: args.collapse.iter_opt,
            proj_dim: args.collapse.proj_dim,
            hvg_weights,
            refine: refine.clone(),
            epochs: args.epochs,
            batches_per_epoch: args.batches_per_epoch,
            batch_size: args.batch_size.unwrap_or(1024),
            gpu_mem_fraction: args.batch_size.is_none().then_some(args.gpu_mem_fraction),
            num_negatives: args.num_negatives,
            learning_rate: args.learning_rate,
            seed: args.seed,
            device: args.device.to_device(args.device_no)?,
            block_size: args.block_size,
            feature_embedding_l2: args.feature_embedding_l2,
            weight_decay: args.weight_decay,
            max_grad_norm: args.max_grad_norm,
            cell_weight_mult,
            phase1_cells_per_pb: args.phase1_cells_per_pb,
            // bge uses a free E_feat (no per-gene β-sharing factorization).
            feat_factor: None,
            // δ_g splice offset is gem-only (needs feat_factor); off for bge.
            delta_l2: 0.0,
            // Lineage-DAG is a gem-only (β-sharing) path; off for bge.
            lineage_dag: false,
            lineage_smooth: false,
            lineage_mst: false,
            joint_velocity: false,
            // `--posterior N` now means N retained sweeps of the phase-1
            // pseudobulk Gibbs, run inside `fit` and written back into the model
            // — not a post-hoc pass over the finished fit.
            pb_posterior_nested_delta: true,
            pb_posterior: posterior_plan.map(|plan| plan.pb_gibbs_config()),
            nce_objective: args.nce_objective.to_ge(),
            // Learned mixed-membership modules in front of ρ — ON by default for bge
            // (`--no-gene-modules` opts out): on held-out marrow cells they turned the
            // gain over the training-marginal null from negative to zero, raised the
            // per-cell rank agreement, and lost less under gene ablation.
            gene_modules: args.modules.resolve(Some(ge::DEFAULT_GENE_MODULES))?,
            // Per-(gene, dim) Bernoulli spike-and-slab feature gate, ALWAYS ON for bge
            // (inclusion KL against a learned π_h + Gaussian effect KL, at the fixed
            // internal weight). There is no null absorber and no simplex — that was the
            // retired softmax. Temperature is the one knob.
            feature_gate: Some(ge::FeatureGateConfig {
                temperature: args.feature_gate_temp,
                ibp_alpha: args.gate_ibp_alpha,
            }),
        })
    };

    // Single-pass gated fit over the full feature axis — the feature gate handles
    // feature selection during training (no post-hoc null-drop / refit).
    let cfg = build_config(&unified)?;
    let out = ge::fit(&mut unified, cfg)?;

    // Carried pseudobulks out, same contract as every other family: the
    // finest collapse level's evidence rates + per-column cell counts.
    let pb_reference_suffix = match out.finest_collapse.as_ref() {
        Some((finest, membership)) => crate::pb_reference::emit_if_requested(
            args.collapse.emit_pb_reference,
            &args.out,
            finest,
            Some(std::slice::from_ref(membership)),
            unified.count_backend().column_multiplicities(),
            &unified.count_backend().row_names()?,
            args.init_from.as_deref(),
            args.pb_reference.as_ref(),
        )?,
        None => None,
    };

    // No per-batch cell QC: it was removed because the per-batch debris cut
    // behaved incoherently across batches (near-identical depth distributions
    // produced 0%-vs-44% drops, guillotining real cells). The upfront `--qc` floor
    // is the only cell filter; bge fits on every cell that passes it.

    // If training was interrupted (Ctrl+C), `fit()` already skipped the heavy phase-2
    // per-cell projection, so the cell embedding is only partial. Skip the expensive
    // post-processing too (Leiden clustering + SIMBA co-embed + ETM) — it would grind
    // for minutes on an un-projected embedding — and write the raw partial outputs so
    // the run exits promptly with whatever it has.
    let interrupted = ge::stop_flag().load(std::sync::atomic::Ordering::Relaxed);
    // ETM topic layout only on a complete, non-interrupted run.
    let resolve_etm = !args.skip_etm && !interrupted;

    if interrupted {
        log::warn!(
            "Interrupted — skipping co-embedding, clustering, and ETM; writing raw partial \
             outputs (the cell embedding is un-projected). Re-run without interrupting for \
             full results."
        );
        ge::save_outputs_named(
            &out.model,
            &ge::OutputContext {
                feature_names: &unified.feature_names,
                barcodes: &unified.barcodes,
                cell_keep_idx: qc_keep_idx.as_deref(),
            },
            &args.out,
            ge::EmbeddingFileNames::SENNA_EMBEDDING,
        )?;
    } else {
        // The SIMBA-style co-embedding and the cluster-seeded ETM share ONE Leiden
        // clustering of the QC-kept cell embedding: the co-embed uses its median
        // cluster size as the temperature target, ETM uses the labels as topics —
        // so the embedding is clustered a single time. The co-embed re-embeds every
        // feature onto the cell manifold (gene = softmax-over-cells weighted average
        // of cell embeddings) and OVERRIDES {out}.feature_embedding.parquet (the raw
        // off-manifold ρ is not written). Cells are SIMBA's reference and are
        // unchanged. Post-hoc only — training (pseudobulk efficiency, phase-2
        // projection) is untouched.
        let cpu = candle_core::Device::Cpu;
        let e_feat_cpu = out.model.e_feat.to_device(&cpu)?; // [D, H] raw ρ
        let e_cell_cpu = match qc_keep_idx.as_deref() {
            Some(keep) => {
                let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
                let idx_t = candle_core::Tensor::from_vec(idx, keep.len(), &cpu)?;
                out.model.e_cell.to_device(&cpu)?.index_select(&idx_t, 0)?
            }
            None => out.model.e_cell.to_device(&cpu)?,
        };
        // Announce the post-training clustering + co-embed so the stretch after
        // "finalizing outputs" doesn't read as a hang (co-embed itself shows a bar).
        info!(
            "Post-training: clustering {} cells + SIMBA co-embedding {} features...",
            e_cell_cpu.dim(0)?,
            e_feat_cpu.dim(0)?
        );
        let (cell_labels, target_eff) = ge::cell_clusters(&e_cell_cpu, args.num_topics)?;

        // Every gene is trained + gated (no held-out projection), so the co-embed runs
        // directly on the trained ρ. The gate zeroes deselected genes' embeddings, and
        // the co-embed maps them onto the cell manifold like any other row.
        //
        // When the phase-1 posterior ran, each feature is additionally compressed
        // toward the origin by its `max_h PIP` — the co-embed has no other per-feature
        // quality signal. This is a radial scaling, NOT a fix for a centroid pile-up:
        // measured, 0.0% of genes sit within 0.1 cell-radii of the centroid (median
        // 0.803), so it creates a concentration rather than undoing one. See
        // `shrink_by_confidence`. `--no-pip-shrinkage` opts out.
        let confidence: Option<Vec<f32>> = (!args.no_pip_shrinkage)
            .then_some(out.pb_posterior.as_ref())
            .flatten()
            .map(|p| {
                p.pip
                    .chunks_exact(p.h)
                    .map(|row| row.iter().copied().fold(0f32, f32::max))
                    .collect()
            });
        ge::write_feature_coembedding(
            &args.out,
            &e_cell_cpu,
            &e_feat_cpu,
            &unified.feature_names,
            target_eff,
            confidence.as_deref(),
        )?;

        // Raw ρ, on BOTH paths. This is the model-axis loading that pairs with
        // the cell embedding in the Poisson rate `exp(ρ_g·z_n + a_g + b_n)` —
        // NOT interchangeable with the co-embed written just above, which is a
        // LOSSY derived view of it (a convex combination of cell embeddings;
        // ρ → co-embed is one-way). ρ used to survive only under `--skip-etm`,
        // where it borrowed the `dictionary` slot that the ETM path claims for
        // β, so a default run lost it entirely and rate-reconstruction consumers
        // (e.g. `senna deconvolve`) had to demand that flag. Purely additive.
        //
        // Under `--skip-etm` this DOES duplicate `dictionary` (same bytes, two
        // files). Kept deliberately: `dictionary`-as-ρ is what
        // `masked-topic --freeze-feature-embedding` and `annotate` already read.
        // The tidy end-state is to migrate those consumers onto `feature_loading`
        // and drop the alias, which is a wider change than this one.
        let rho_mat = Mat::from_tensor(&e_feat_cpu)?;
        let rho_h_names = axis_id_names("h", rho_mat.ncols());
        rho_mat.to_parquet_with_names(
            &format!("{}.feature_loading.parquet", args.out),
            (Some(&unified.feature_names), Some("gene")),
            Some(&rho_h_names),
        )?;

        // Output layout: the H-space cell embedding Z ALWAYS goes to
        // {out}.cell_embedding.parquet, on both paths. ETM resolved (default)
        // additionally emits the topic-model tables (latent = log θ,
        // dictionary = β); --skip-etm emits no latent at all and keeps
        // dictionary = ρ. So `latent` means log θ, unconditionally, and
        // downstream geometry reads cell_embedding via
        // `RunOutputs::geometry_latent` without having to know which flags ran.
        // The co-embedded feature_embedding is written above for both paths.
        if resolve_etm {
            resolve_etm_topics(
                &out.model,
                &unified.feature_names,
                &unified.barcodes,
                args,
                qc_keep_idx.as_deref(),
                &cell_labels,
            )?;
            // The --skip-etm branch writes these inside `save_outputs_named`.
            ge::write_module_tables(&args.out, &out.model, &unified.feature_names)?;
        } else {
            ge::save_outputs_named(
                &out.model,
                &ge::OutputContext {
                    feature_names: &unified.feature_names,
                    barcodes: &unified.barcodes,
                    cell_keep_idx: qc_keep_idx.as_deref(),
                },
                &args.out,
                ge::EmbeddingFileNames::SENNA_EMBEDDING,
            )?;
        }
    }

    let input: Vec<String> = data_files
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let batch: Vec<String> = args
        .batch_files
        .as_ref()
        .map(|v| v.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    let has_modules = out.model.modules.is_some();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        train_args: Some(crate::run_manifest::record_train_args(args)?),
        kind: crate::run_manifest::RunKind::Bge,
        prefix: &args.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        // With ETM resolved the dictionary is β (gene × topic); otherwise it IS ρ.
        //
        // ρ does NOT go to feature_embedding.parquet — that file is always the SIMBA co-embed (see
        // below, and `write_feature_coembedding` above). ρ lives on the model's own axis, not on
        // the cell manifold, so putting it there would hand `annotate-by-projection` an
        // off-manifold gene table and make its Euclidean nearest-centroid call ill-posed.
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_reference_suffix,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        // The SIMBA co-embed is written as feature_embedding.parquet in BOTH
        // the ETM and --skip-etm paths, so record it unconditionally (else a
        // skip-etm run's annotate-by-projection falls back to the raw-ρ
        // dictionary and ignores the co-embed file on disk).
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_loading_suffix: Some("feature_loading.parquet"),
        // Learned gene modules, when the run trained them; the composed row still
        // lives in `feature_loading`, so these are additive.
        module_membership_suffix: has_modules.then_some("module_membership.parquet"),
        module_dictionary_suffix: has_modules.then_some("module_dictionary.parquet"),
        // ETM resolved => `dictionary` holds the log-simplex β; --skip-etm => it is ρ.
        softmax_dictionary_suffix: resolve_etm.then_some("dictionary.parquet"),
        // Z always lands in cell_embedding.parquet — on BOTH the ETM and
        // --skip-etm paths — so every geometry consumer finds the H-space
        // embedding at one fixed name.
        cell_embedding_suffix: Some("cell_embedding.parquet"),
        default_colour_by: if resolve_etm { "topic" } else { "cluster" },
        // `latent` is log θ, so it exists only when the ETM actually resolved.
        has_latent: resolve_etm,
        velocity_suffix: None,
        velocity_factor_suffix: None,
        delta_feature_embedding_suffix: None,
        has_cell_to_pb: false,
    })?;

    // The posterior no longer runs here. It is phase-1's own sampler now
    // (`FitConfig::pb_posterior`), so it happens INSIDE `fit` — before phase 2,
    // before the co-embedding, and against the pseudobulk side rather than a
    // frozen full-cell side. The old guard here ("skip when interrupted, because
    // `e_cell` is un-projected") no longer applies: the pb Gibbs never reads
    // `e_cell`. Only its tables are written here, after the fit's own outputs.
    if let Some(post) = out.pb_posterior.as_ref() {
        ge::eval::write_pb_posterior_tables(&args.out, post, &unified.feature_names)?;
        ge::posterior::pb_gibbs::write_posterior_hyper_from_model(
            &args.out,
            post,
            &out.varmap,
            // geometry now travels on the result, not as a caller-supplied cap
            args.seed,
        )?;
        let worst_ess = post
            .sigma_diag
            .iter()
            .chain(&post.pi0_diag)
            .map(|d| f64::from(d.min_ess))
            .fold(f64::INFINITY, f64::min);
        info!(
            "phase-1 posterior: {} sweeps retained, per-dim σ₀² median {:.4}, \
             dims/feature {:.2} (posterior), worst hyper ESS {:.1}",
            post.n_kept,
            median_of(&post.sigma2),
            // Mean row-sum of the PIP table: the expected number of dims a feature
            // loads, AS INFERRED. Deliberately not `Σ_h (1 − π₀ₕ)`, which under the
            // default IBP is the fixed ladder and would merely echo the `α` that went
            // in; and not a median over π₀, which on a monotone ladder is just the rate
            // at dim H/2. This is the one number here the data actually determined, and
            // it is on the same scale as `α`, so the two can be read against each other.
            f64::from(post.pip.iter().sum::<f32>())
                / (post.pip.len() / post.h.max(1)).max(1) as f64,
            worst_ess
        );
        if worst_ess < 10.0 {
            log::warn!(
                "posterior hyper chains barely moved (min ESS {worst_ess:.1}) — the per-dim \
                 σ₀²/π₀ are close to their priors, so read the inclusion probabilities as \
                 weakly identified rather than as calibrated selection."
            );
        }
    }

    if resolve_etm {
        info!(
            "Done — outputs at {}.{{cell_embedding,latent,dictionary,feature_embedding,*_bias}}.parquet \
             (cell_embedding = Z, latent = log θ)",
            args.out
        );
    } else {
        info!(
            "Done — outputs at {}.{{cell_embedding,dictionary,feature_embedding,*_bias}}.parquet \
             (cell_embedding = Z; no latent — topics were not resolved)",
            args.out
        );
    }

    Ok(())
}

/// Median of a slice, for the one-line posterior summary. Empty ⇒ `NaN`, which
/// serializes and prints as such rather than silently reading as zero.
fn median_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[s.len() / 2]
}
