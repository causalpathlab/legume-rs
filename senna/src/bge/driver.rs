//! The shared driver behind `senna bge` and `senna gem`: everything from the
//! multilevel pseudobulk fit through the output writers and the run
//! manifest. `senna bge`'s own `fit_bge` (`bge/mod.rs`) resolves the
//! multiome layout, loads the data, and computes its HVG weights, then hands
//! off to [`fit_embed_family`] here; `senna gem` (`gem/run.rs`) does its own
//! (simpler) load and HVG pooling and hands off the same way. bge always
//! passes `EmbedPlan::tracks = None` (one track, every row its own gene);
//! gem passes its own row-grammar [`crate::gem::tracks::TrackPlan`], and
//! hooks its `{out}.feature_contrast.parquet` writer in through
//! [`EmbedPlan::after_fit`].
//!
//! [`EmbedKnobs`] is the flag surface both commands drive the fit with.
//! `GemArgs` now flattens the exact same `refine_weighting::CollapseArgs`
//! and `ge::FeatureModuleArgs` groups `BgeArgs` does (Task 5a), so both
//! `collapse` and `modules` are shared-by-reference fields here, as the
//! original sketch had them, and `build_config` reads the raw collapse
//! numbers (`num_levels`, `sort_dim`, `knn_cells`, `iter_opt`, `proj_dim`)
//! straight off whichever command's `collapse` this is — no more per-field
//! renaming. `bulk_batches` / `emit_pb_reference` / `refine` stay separate,
//! resolved fields: gem does not (yet) apply the carried-pb-reference /
//! mixture-batch policy those three encode (`GemArgs` has no
//! `pb_reference` / `init_from` surface and does not implement
//! `Updatable`), even though the shared struct's own flags now parse on
//! gem's surface too — so `GemArgs::knobs` still hardcodes them off, exactly
//! as before this task.

use crate::embed_common::*;
use crate::pb_reference::ReferenceInput;
use crate::run_manifest::RunKind;
use graph_embedding_util as ge;

/// Every driver flag both commands drive the fit with, borrowed from
/// whichever command's own `*Args` built this. Constructed by
/// `BgeArgs::knobs` / `GemArgs::knobs`.
pub(crate) struct EmbedKnobs<'a> {
    pub embedding_dim: usize,

    /// The shared random-projection + multilevel pseudobulk collapse
    /// pipeline (`--proj-dim`, `--sort-dim`, `--knn-cells`, `--num-levels`,
    /// `--iter-opt`, `--pb-refine-*`, `--mixture-batch`,
    /// `--emit-pb-reference` / `--no-emit-pb-reference`). Both commands
    /// flatten the identical struct now, so `build_config` reads its raw
    /// numbers straight off this reference either way.
    pub collapse: &'a crate::refine_weighting::CollapseArgs,
    /// Resolved separately from `collapse` (see the module doc): gem always
    /// passes `None` / `false` here regardless of what `--mixture-batch` /
    /// `--emit-pb-reference` parse to on its own surface.
    pub bulk_batches: Option<&'a [Box<str>]>,
    /// Carry the finest collapse level forward as `{out}.pb_reference.zarr.zip`
    /// (bge: `!--no-emit-pb-reference`; gem has no such flag yet, always
    /// `false`).
    pub emit_pb_reference: bool,
    /// BBKNN + DC-Poisson refinement params, already resolved by the
    /// caller; `None` disables it (bge: `--no-refine`; gem always refines).
    pub refine: Option<ge::RefineParams>,

    pub qc: &'a QcArgs,
    pub phase1_cells_per_pb: usize,
    pub modules_per_unit: usize,
    pub skip_etm: bool,
    pub num_topics: Option<usize>,
    pub epochs: usize,
    pub batches_per_epoch: Option<usize>,
    pub batch_size: Option<usize>,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub block_size: Option<usize>,
    pub seed: u64,
    pub device: &'a ComputeDevice,
    pub device_no: usize,

    /// The `--feature-modules` flag group. Both commands flatten it now, so
    /// this is always `Some`; kept `Option` because a caller that never
    /// resolves gene modules can still pass `None` explicitly.
    pub modules: Option<&'a ge::FeatureModuleArgs>,
    /// The module count this command trains when `--feature-modules` is not
    /// passed explicitly. Both commands pass `Some(DEFAULT_FEATURE_MODULES)`:
    /// the hierarchical phase 1 has no module-free mode (the hard gene
    /// partition it scores against is structural, not an optional layer —
    /// `ge::fit` errors "the hierarchical phase 1 needs a module count"
    /// without one), so `None` here is not a state either command can
    /// actually run in.
    pub default_feature_modules: Option<usize>,

    pub out: &'a str,
    pub batch_files: Option<&'a [Box<str>]>,
}

/// Everything [`fit_embed_family`] needs for one fit: the caller's already
/// -loaded data plus its resolved knobs.
pub(crate) struct EmbedPlan<'a> {
    pub kind: RunKind,
    pub knobs: EmbedKnobs<'a>,
    pub unified: ge::UnifiedData,
    /// The data files this run was loaded from (manifest `data.input`).
    pub data_files: Vec<Box<str>>,
    /// The resolved multiome layout, when this load had one (bge only).
    pub multiome: Option<crate::multiome_layout::RunMultiome>,
    /// Full-axis (current feature-axis-indexed) HVG projection weights.
    pub hvg_weights: Option<Vec<f32>>,
    /// Row structure of the feature axis, from [`crate::gem::tracks::assign_tracks`]:
    /// `senna gem`'s base count track plus any co-measured modality tracks.
    /// `None` (bge, always) = one track, every row its own gene.
    pub tracks: Option<crate::gem::tracks::TrackPlan>,
    /// Ridge on the per-track offsets (`FitConfig.offset_l2`); inert at one
    /// track. bge always passes `0.0`.
    pub offset_l2: f32,
    /// Rank of the per-track gene offsets (`FitConfig.offset_rank`); inert at
    /// one track. gem's `--offset-rank`; bge passes the LoRA default.
    pub offset_rank: usize,
    /// Gene rows given up front (`--{freeze,init,lora}-feature-embedding`),
    /// pinned or only started from; `None` = every row trains. On gem's
    /// tracked axis these are the base rows, ids on the gene axis.
    pub preset_features: Option<ge::PresetRows>,
    /// Given offsets on non-base tracks (gem only); empty for none.
    pub preset_offsets: Vec<ge::PresetOffsets>,
    /// The given table's rows that matched no feature, appended to the
    /// written ρ so the output is the full table.
    pub carried: Option<crate::carried_rows::CarriedRows>,
    pub pb_reference: Option<&'a ReferenceInput>,
    pub init_from: Option<&'a str>,
    pub train_args: crate::run_manifest::TrainArgsRecord,
    /// Called after the module tables are written and before the manifest.
    /// `senna gem` hooks its `{out}.feature_contrast.parquet` writer in
    /// here; bge always passes `None`.
    #[allow(clippy::type_complexity)]
    pub after_fit: Option<&'a dyn Fn(&FitArtifacts<'_>) -> anyhow::Result<()>>,
}

/// What [`EmbedPlan::after_fit`] sees.
pub(crate) struct FitArtifacts<'a> {
    pub out: &'a ge::FitOutput,
    pub unified: &'a ge::UnifiedData,
    /// Cell rows kept after QC, when QC ran. Feature-axis writers (the only
    /// kind `after_fit` has today) don't need it; kept for a future per-cell
    /// consumer, and to match every other writer in this module's own QC
    /// contract.
    #[allow(dead_code)]
    pub qc_keep: Option<&'a [usize]>,
    pub prefix: &'a str,
    /// `(track name, cell-encoder safetensors suffix)` for every track the
    /// encoder save just wrote, in [`ge::CellEncoders::iter`] order — empty
    /// when phase 2 placed cells by block SGD rather than a distilled
    /// encoder (`out.cell_encoder` was `None`). Not yet read by any
    /// `after_fit` hook; carried so a later manifest writer can record it.
    #[allow(dead_code)]
    pub track_encoders: Vec<(Box<str>, String)>,
}

/// Run the shared fit: multilevel pseudobulk collapse, phase-1/phase-2
/// training, post-training co-embed + (optional) ETM resolution, gene-module
/// tables, and the run manifest. Moved out of `senna bge`'s own `fit_bge`
/// (formerly `senna/src/bge/mod.rs` ~131-571) essentially unchanged, so
/// `senna bge` and `senna gem` run the exact same code from here down.
pub(crate) fn fit_embed_family(mut plan: EmbedPlan<'_>) -> anyhow::Result<()> {
    let knobs = &plan.knobs;

    // Assemble a `FitConfig` for the CURRENT feature AND cell axes of
    // `unified`. Kept as a closure (rather than inlined) even though this
    // task's callers run it once, matching `fit_bge`'s own shape from before
    // the extraction.
    let preset_features = plan.preset_features.take();
    let preset_offsets = std::mem::take(&mut plan.preset_offsets);
    let carried = plan.carried.take();
    let build_config = move |unified: &ge::UnifiedData| -> anyhow::Result<ge::FitConfig> {
        let hvg_weights = plan.hvg_weights.as_ref().map(|w| {
            unified
                .feature_to_backend_row
                .iter()
                .map(|&i| w[i])
                .collect::<Vec<f32>>()
        });
        let feature_modules = match knobs.modules {
            Some(m) => match m.resolve(knobs.default_feature_modules)? {
                Some(mut gm) => {
                    gm.parent = parent_modules(plan.init_from, &unified.feature_names)?;
                    Some(gm)
                }
                None => None,
            },
            None => None,
        };
        Ok(ge::FitConfig {
            embedding_dim: knobs.embedding_dim,
            // Greedy batch correction against the carried reference, exactly
            // as in the other families — see `MultilevelParams::anchor_batches`.
            anchor_batches: plan
                .pb_reference
                .is_some()
                .then(|| vec![crate::pb_reference::REFERENCE_BATCH.into()]),
            bulk_batches: knobs.bulk_batches.map(<[Box<str>]>::to_vec),
            emit_finest_collapse: knobs.emit_pb_reference,
            num_levels: knobs.collapse.num_levels,
            sort_dim: knobs.collapse.sort_dim,
            knn_pb_samples: knobs.collapse.knn_cells,
            num_opt_iter: knobs.collapse.iter_opt,
            proj_dim: knobs.collapse.proj_dim,
            hvg_weights,
            refine: knobs.refine.clone(),
            epochs: knobs.epochs,
            batches_per_epoch: knobs.batches_per_epoch,
            batch_size: knobs.batch_size.unwrap_or(1024),
            learning_rate: knobs.learning_rate,
            seed: knobs.seed,
            device: knobs.device.to_device(knobs.device_no)?,
            block_size: knobs.block_size,
            weight_decay: knobs.weight_decay,
            phase1_cells_per_pb: knobs.phase1_cells_per_pb,
            hier_units_per_step: knobs.batch_size.unwrap_or(256),
            hier_modules_per_unit: knobs.modules_per_unit,
            feature_modules,
            tracks: plan
                .tracks
                .as_ref()
                .map(crate::gem::tracks::TrackPlan::to_ge),
            offset_l2: plan.offset_l2,
            offset_rank: plan.offset_rank,
            preset_features,
            preset_offsets,
        })
    };

    // Single-pass fit over the full feature axis (no post-hoc null-drop / refit).
    let cfg = build_config(&plan.unified)?;
    let out = ge::fit(&mut plan.unified, cfg)?;

    // Carried pseudobulks out, same contract as every other family: the
    // finest collapse level's evidence rates + per-column cell counts.
    let pb_reference_suffix = match out.finest_collapse.as_ref() {
        Some((finest, membership)) => crate::pb_reference::emit_if_requested(
            knobs.emit_pb_reference,
            knobs.out,
            finest,
            Some(std::slice::from_ref(membership)),
            plan.unified.count_backend().column_multiplicities(),
            &plan.unified.count_backend().row_names()?,
            plan.init_from,
            plan.pb_reference,
        )?,
        None => None,
    };

    /////////////////////////////
    // Cell QC (output filter) //
    /////////////////////////////
    // Every cell + edge still informs the joint embedding / feature
    // dictionary; QC-failed cells are dropped from the archetypal analysis
    // and all per-cell outputs via a write-time `select_rows`.
    let qc_keep_idx: Option<Vec<usize>> = if let Some(cfg) = knobs.qc.to_config() {
        if cfg.feature_min_cells > 0 {
            log::warn!(
                "--qc-feature-min-cells is ignored (cell-only QC; the dictionary keeps all \
                 features)"
            );
        }
        // Carried pseudobulks are processed outputs, not cells: they must
        // neither receive a QC verdict nor sit inside the MAD band statistics.
        let exempt: Option<Vec<bool>> = plan.pb_reference.map(|r| {
            let n = plan.unified.n_cells();
            let n_real = n.saturating_sub(r.cell_counts.len());
            (0..n).map(|c| c >= n_real).collect()
        });
        let report = data_beans::qc_lib::compute_qc_exempting(
            plan.unified.count_backend(),
            &cfg,
            knobs.block_size,
            exempt.as_deref(),
        )?;
        let keep = report.emit_idx_unmasked();
        info!(
            "QC: {} / {} cells kept for output ({} near-empty, {} MAD-outlier dropped)",
            keep.len(),
            plan.unified.n_cells(),
            report.near_empty.iter().filter(|&&e| e).count(),
            report.n_cells_dropped,
        );
        Some(keep)
    } else {
        None
    };
    let qc_keep_idx = crate::pb_reference::exclude_carried(
        plan.pb_reference,
        plan.unified.n_cells(),
        qc_keep_idx,
    );

    // If training was interrupted (Ctrl+C), `fit()` already skipped the heavy phase-2
    // per-cell projection, so the cell embedding is only partial. Skip the expensive
    // post-processing too (Leiden clustering + SIMBA co-embed + ETM) — it would grind
    // for minutes on an un-projected embedding — and write the raw partial outputs so
    // the run exits promptly with whatever it has.
    let interrupted = ge::stop_flag().load(std::sync::atomic::Ordering::Relaxed);
    // ETM topic layout only on a complete, non-interrupted run.
    let resolve_etm = !knobs.skip_etm && !interrupted;

    // The map phase 2 placed the cells with, so `predict` places a query by the
    // same one. One self-contained file per map: the trunk plus its per-gene
    // mean. A one-track fit has exactly one, under the name `predict` reads;
    // further tracks get a file each, named by `encoder_suffix_for`. Written
    // here (before the interrupted/complete branch below) so `after_fit`
    // (complete runs only) can see which files were written.
    let mut track_encoders: Vec<(Box<str>, String)> = Vec::new();
    let cell_encoder_suffix = match out.cell_encoder.as_ref() {
        Some(encs) => {
            for te in encs.iter() {
                let suffix = crate::gem::tracks::encoder_suffix_for(te.track, &te.name);
                let path = format!("{}.{suffix}", knobs.out);
                te.encoder.save(&path)?;
                info!("Wrote the `{}` cell encoder to {path}", te.name);
                track_encoders.push((te.name.clone(), suffix));
            }
            Some("cell_encoder.safetensors")
        }
        None => None,
    };
    // Track 0's file keeps the `cell_encoder_suffix` slot above, so the
    // manifest's own `track_encoders` list is every track BEYOND it. Track 0
    // is always `encs.iter()`'s first entry (`TrackSpec::validate` requires
    // it to be present and a count track, and `CellEncoders::iter` is
    // ascending by track), so `skip(1)` is exact regardless of what track 0
    // happens to be named (`"base"` on bge's one-track axis, a real
    // modality/channel pair on gem's). Read off before `track_encoders` is
    // (maybe) moved into `FitArtifacts` below.
    let track_encoder_suffixes: Vec<(String, String)> = track_encoders
        .iter()
        .skip(1)
        .map(|(name, suf)| (name.to_string(), suf.clone()))
        .collect();

    // Raw ρ → {out}.feature_embedding.parquet, on EVERY path (complete or
    // interrupted). This is the model-axis embedding that pairs with the cell
    // embedding in the Poisson rate `exp(ρ_g·z_n + a_g + b_n)` — NOT
    // interchangeable with the SIMBA co-embed written on the complete path
    // below, which is a LOSSY derived view of it (a convex combination of cell
    // embeddings; ρ → co-embed is one-way). The manifest always names this
    // file, and the carry-through appends into it, so it must exist even when
    // the run was cut short.
    let cpu = candle_core::Device::Cpu;
    let e_feat_cpu = out.model.e_feat.to_device(&cpu)?; // [D, H] raw ρ
    ge::save_embedding(
        &format!("{}.feature_embedding.parquet", knobs.out),
        &e_feat_cpu,
        &plan.unified.feature_names,
        "feature",
    )?;
    // The learned-module tables, likewise on both paths; the composed ρ above
    // already carries them for every reader that does not care.
    ge::write_module_tables(knobs.out, &out.model, &plan.unified.feature_names)?;

    if interrupted {
        log::warn!(
            "Interrupted — skipping co-embedding, clustering, and ETM; writing raw partial \
             outputs (the cell embedding is un-projected). Re-run without interrupting for \
             full results."
        );
        ge::save_outputs_named(
            &out.model,
            &ge::OutputContext {
                feature_names: &plan.unified.feature_names,
                barcodes: &plan.unified.barcodes,
                cell_keep_idx: qc_keep_idx.as_deref(),
            },
            knobs.out,
            ge::EmbeddingFileNames::SENNA_EMBEDDING,
        )?;
    } else {
        // The SIMBA-style co-embedding and the cluster-seeded ETM share ONE Leiden
        // clustering of the QC-kept cell embedding: the co-embed uses its median
        // cluster size as the temperature target, ETM uses the labels as topics —
        // so the embedding is clustered a single time. The co-embed re-embeds every
        // feature onto the cell manifold (gene = softmax-over-cells weighted average
        // of cell embeddings) into {out}.feature_coembedding.parquet, beside the
        // raw off-manifold ρ above. Cells are SIMBA's reference and are
        // unchanged. Post-hoc only — training (pseudobulk efficiency, phase-2
        // projection) is untouched.
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
        let (cell_labels, target_eff) = ge::cell_clusters(&e_cell_cpu, knobs.num_topics)?;

        // Every gene is trained (no held-out projection), so the co-embed runs
        // directly on the trained ρ.
        ge::write_feature_coembedding(
            knobs.out,
            &e_cell_cpu,
            &e_feat_cpu,
            &plan.unified.feature_names,
            target_eff,
        )?;

        // Output layout: the H-space cell embedding Z ALWAYS goes to
        // {out}.cell_embedding.parquet, on both paths. ETM resolved (default)
        // additionally emits the topic-model tables (latent = log θ,
        // dictionary = β); --skip-etm emits neither.
        if resolve_etm {
            super::resolve_etm::resolve_etm_topics(
                &out.model,
                &plan.unified.feature_names,
                &plan.unified.barcodes,
                knobs.out,
                qc_keep_idx.as_deref(),
                &cell_labels,
            )?;
        } else {
            ge::save_outputs_named(
                &out.model,
                &ge::OutputContext {
                    feature_names: &plan.unified.feature_names,
                    barcodes: &plan.unified.barcodes,
                    cell_keep_idx: qc_keep_idx.as_deref(),
                },
                knobs.out,
                ge::EmbeddingFileNames::SENNA_EMBEDDING,
            )?;
        }
        if let Some(f) = plan.after_fit {
            f(&FitArtifacts {
                out: &out,
                unified: &plan.unified,
                qc_keep: qc_keep_idx.as_deref(),
                prefix: knobs.out,
                track_encoders,
            })?;
        }
    }

    let input: Vec<String> = plan
        .data_files
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let batch: Vec<String> = knobs
        .batch_files
        .map(|v| v.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    let has_modules = out.model.modules.is_some();
    // `after_fit` (gem's contrast-table writer) only ran in the non-interrupted
    // branch above; an interrupted run wrote no contrast tables, so the
    // manifest must not claim it did.
    let contrast_written = plan.after_fit.is_some() && !interrupted;
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        train_args: Some(plan.train_args),
        kind: plan.kind,
        prefix: knobs.out,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        // So `senna layout / plot / impute --from` can re-read these files the
        // way training did, instead of stacking the modalities as extra cells.
        data_multiome: plan.multiome,
        // With ETM resolved the dictionary is β (gene × topic); without it
        // there is none — ρ is `feature_embedding`, never this slot.
        dictionary_suffix: resolve_etm.then_some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_reference_suffix,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        // ρ on every path; the SIMBA co-embed only on a complete run — an
        // interrupted one skipped it, so the manifest must not name a file
        // that is not there.
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_coembedding_suffix: (!interrupted).then_some("feature_coembedding.parquet"),
        carried: carried.as_ref(),
        // Learned gene modules, when the run trained them; the composed row still
        // lives in `feature_embedding`, so these are additive.
        module_membership_suffix: has_modules.then_some(ge::transfer::MODULE_MEMBERSHIP_SUFFIX),
        module_dictionary_suffix: has_modules.then_some(ge::transfer::MODULE_DICTIONARY_SUFFIX),
        // ETM resolved => `dictionary` holds the log-simplex β; --skip-etm => it is ρ.
        softmax_dictionary_suffix: resolve_etm.then_some("dictionary.parquet"),
        // Z always lands in cell_embedding.parquet — on BOTH the ETM and
        // --skip-etm paths — so every geometry consumer finds the H-space
        // embedding at one fixed name.
        cell_embedding_suffix: Some("cell_embedding.parquet"),
        cell_encoder_suffix,
        feature_contrast_suffix: contrast_written.then_some("feature_contrast.parquet"),
        feature_contrast_bias_suffix: contrast_written.then_some("feature_contrast_bias.parquet"),
        track_encoder_suffixes,
        default_colour_by: if resolve_etm { "topic" } else { "cluster" },
        // `latent` is log θ, so it exists only when the ETM actually resolved.
        has_latent: resolve_etm,
        has_cell_to_pb: false,
        has_pb_tree: false,
    })?;

    // The phase-1 pseudobulk embeddings, with each pseudobulk's batch: the
    // geometry the dictionary was trained against. When the per-cell embedding
    // separates by batch, this table says whether the separation was already
    // there before phase 2.
    write_pb_embeddings(knobs.out, &out.pb_embeddings, &plan.unified.batch_names)?;
    if let Some(fold) = &out.batch_gene_fold {
        write_batch_gene_fold(knobs.out, fold, &plan.unified.feature_names)?;
    }

    if resolve_etm {
        info!(
            "Done — outputs at {}.{{cell_embedding,latent,dictionary,feature_embedding,feature_coembedding,*_bias}}.parquet \
             (cell_embedding = Z, feature_embedding = ρ, latent = log θ)",
            knobs.out
        );
    } else {
        info!(
            "Done — outputs at {}.{{cell_embedding,feature_embedding,feature_coembedding,*_bias}}.parquet \
             (cell_embedding = Z, feature_embedding = ρ; no latent — topics were not resolved)",
            knobs.out
        );
    }

    Ok(())
}

/// The parent run's module tables for `senna update`'s warm start, matched to
/// this fit's feature axis by exact name. `None` when there is no parent, or the
/// parent trained no modules (the fit then warm-starts from its own k-means, as a
/// fresh run would).
fn parent_modules(
    init_from: Option<&str>,
    feature_names: &[Box<str>],
) -> anyhow::Result<Option<ge::ParentModulesOwned>> {
    let Some(prefix) = init_from else {
        return Ok(None);
    };
    let parent = crate::bge::score::BgeEmbedding::open(prefix)?;
    let rho = parent.rho_matrix();
    let Some((pi, mu)) = parent.modules else {
        info!(
            "update: parent {prefix} trained no gene modules; warm-starting from this fit's own \
             profiles"
        );
        return Ok(None);
    };
    // The same flexible matcher `predict` aligns a query with, so a parent whose
    // names differ by case or suffix still matches.
    let remap = crate::topic::eval::build_gene_remap_with(
        &parent.gene_names,
        feature_names,
        &crate::topic::eval::QueryNameOpts::default(),
    );
    let n_matched = remap.new_to_train.iter().filter(|r| r.is_some()).count();
    info!(
        "update: carrying the parent's {}-module partition from {prefix}; {} of {} features \
         match the parent",
        mu.nrows(),
        n_matched,
        feature_names.len()
    );
    Ok(Some(ge::ParentModulesOwned {
        rho,
        pi,
        mu,
        row_to_parent: remap.new_to_train,
        knobs: ge::transfer::AlignKnobs::default(),
    }))
}

/// `{out}.batch_gene_fold.parquet`: the per-batch gene fold phase 2 divided each
/// batch's cell counts by, as `log δ_gb`, `[features × batches]`.
fn write_batch_gene_fold(
    out: &str,
    fold: &ge::fit::BatchGeneFold,
    feature_names: &[Box<str>],
) -> anyhow::Result<()> {
    let table = Mat::from_row_slice(fold.n_batches(), fold.n_features, &fold.delta)
        .map(f32::ln)
        .transpose();
    table.to_parquet_with_names(
        &format!("{out}.batch_gene_fold.parquet"),
        (Some(feature_names), Some("feature")),
        Some(&fold.batch_names),
    )?;
    info!("Wrote {out}.batch_gene_fold.parquet");
    Ok(())
}

/// `{out}.pb_embedding.parquet` (rows `l{level}:pb{i}`, columns `h0..`) and
/// `{out}.pb_batch.parquet` (level and batch name per row), every level stacked.
fn write_pb_embeddings(
    out: &str,
    levels: &[ge::fit::PbLevelEmbedding],
    batch_names: &[Box<str>],
) -> anyhow::Result<()> {
    use matrix_util::dmatrix_util::concatenate_vertical;
    use matrix_util::parquet::{write_named_table, Column};
    if levels.is_empty() {
        return Ok(());
    }
    let h = levels[0].e_pb.ncols();
    let table = concatenate_vertical(&levels.iter().map(|l| l.e_pb.clone()).collect::<Vec<_>>())?;
    let n = table.nrows();
    let mut rows: Vec<Box<str>> = Vec::with_capacity(n);
    let mut level_col: Vec<i32> = Vec::with_capacity(n);
    let mut batch_col: Vec<Box<str>> = Vec::with_capacity(n);
    for (level, l) in levels.iter().enumerate() {
        for i in 0..l.e_pb.nrows() {
            rows.push(format!("l{level}:pb{i}").into_boxed_str());
            level_col.push(level as i32);
            batch_col.push(match l.batch[i] {
                u32::MAX => Box::from(""),
                b => batch_names[b as usize].clone(),
            });
        }
    }
    table.to_parquet_with_names(
        &format!("{out}.pb_embedding.parquet"),
        (Some(&rows), Some("pb")),
        Some(&axis_id_names("h", h)),
    )?;
    write_named_table(
        &format!("{out}.pb_batch.parquet"),
        "pb",
        &rows,
        &[
            (Box::from("level"), Column::I32(&level_col)),
            (Box::from("batch"), Column::Str(&batch_col)),
        ],
    )?;
    info!(
        "Wrote {out}.pb_embedding.parquet / pb_batch.parquet ({n} pseudobulks over {} levels)",
        levels.len()
    );
    Ok(())
}

/// Default module count for the hierarchical phase 1's hard gene partition,
/// used by both `senna bge` and `senna gem` unless `--feature-modules` overrides
/// it — the engine has no module-free mode, so this is a shared policy
/// constant rather than an opt-in default.
const DEFAULT_FEATURE_MODULES: usize = 128;

impl super::BgeArgs {
    /// `embedding_dim` is the width resolved against a given feature table.
    pub(crate) fn knobs(&self, embedding_dim: usize) -> EmbedKnobs<'_> {
        EmbedKnobs {
            embedding_dim,
            collapse: &self.collapse,
            bulk_batches: self.collapse.mixture_batch.as_deref(),
            emit_pb_reference: self.collapse.emits_pb_reference(),
            // `--no-refine` is gbe-specific (the other subcommands always refine);
            // otherwise the shared `--pb-refine-*` flags drive RefineParams.
            refine: (!self.no_refine).then(|| self.collapse.pb_refine.to_params()),
            qc: &self.qc,
            phase1_cells_per_pb: self.phase1_cells_per_pb,
            modules_per_unit: self.modules_per_unit,
            skip_etm: self.skip_etm,
            num_topics: self.num_topics,
            epochs: self.epochs,
            batches_per_epoch: self.batches_per_epoch,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            weight_decay: self.weight_decay,
            block_size: self.block_size,
            seed: self.seed,
            device: &self.device,
            device_no: self.device_no,
            modules: Some(&self.modules),
            default_feature_modules: Some(DEFAULT_FEATURE_MODULES),
            out: &self.out,
            batch_files: self.batch_files.as_deref(),
        }
    }
}

impl crate::gem::args::GemArgs {
    /// `embedding_dim` is the width resolved against a given feature table.
    pub(crate) fn knobs(&self, embedding_dim: usize) -> EmbedKnobs<'_> {
        EmbedKnobs {
            embedding_dim,
            collapse: &self.collapse,
            // gem has no `senna update` / carried-pb-reference surface yet
            // (no `pb_reference` / `init_from` fields, no `Updatable` impl),
            // so these two stay off even though `--mixture-batch` /
            // `--emit-pb-reference` now parse on gem's shared `collapse`.
            bulk_batches: None,
            emit_pb_reference: false,
            refine: (!self.no_refine).then(|| self.collapse.pb_refine.to_params()),
            qc: &self.qc,
            phase1_cells_per_pb: self.phase1_cells_per_pb,
            modules_per_unit: self.modules_per_unit,
            skip_etm: self.skip_etm,
            num_topics: self.num_topics,
            epochs: self.epochs,
            // gem dropped `--batches-per-epoch` (Task 5a); always auto.
            batches_per_epoch: None,
            batch_size: self.batch_size,
            learning_rate: self.learning_rate,
            weight_decay: self.weight_decay,
            block_size: self.block_size,
            seed: self.seed,
            device: &self.device,
            device_no: self.device_no,
            modules: Some(&self.modules),
            // Same default as bge: the hierarchical phase 1 has no
            // module-free mode (see `EmbedKnobs::default_feature_modules`'s
            // doc), so this cannot be `None`.
            default_feature_modules: Some(DEFAULT_FEATURE_MODULES),
            out: &self.out,
            batch_files: self.batch_files.as_deref(),
        }
    }
}

#[cfg(test)]
#[path = "driver/tests.rs"]
mod tests;
