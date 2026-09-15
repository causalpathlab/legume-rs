//! The shared driver behind `senna bge` and `senna gem`: everything from the
//! multilevel pseudobulk fit through the output writers and the run
//! manifest. `senna bge`'s own `fit_bge` (`bge/mod.rs`) resolves the
//! multiome layout, loads the data, and computes its HVG weights, then hands
//! off to [`fit_embed_family`] here; `senna gem` (`gem/run.rs`) does its own
//! (simpler) load and HVG pooling and hands off the same way. gem is, for
//! now, bge run over every feature row — a later task adds spliced/unspliced
//! **tracks** on top of this same driver ([`EmbedPlan::tracks`]).
//!
//! [`EmbedKnobs`] is the flag surface both commands drive the fit with. The
//! sketch that first specified this module described it as a `collapse:
//! &refine_weighting::CollapseArgs` reference, mirroring `BgeArgs`'s own
//! field — but `gem::args::CollapseArgs` is a distinct, pre-existing type
//! (different flag names, no `--pb-refine-*` / `--mixture-batch` /
//! `--emit-pb-reference` surface of its own), and this task's brief is
//! explicit that gem's arg surface is not to be redesigned beyond its listed
//! field removals. So the collapse-pipeline knobs are flattened onto
//! [`EmbedKnobs`] directly as plain values instead of a shared struct
//! reference: `BgeArgs::knobs` reads them off its own `CollapseArgs`,
//! `GemArgs::knobs` off its own. Same story for `modules`: gem has no
//! `--gene-modules` flag yet, so it hands over `None` rather than a
//! `&ge::GeneModuleArgs` it doesn't have.

use crate::embed_common::*;
use crate::pb_reference::ReferenceInput;
use crate::run_manifest::RunKind;
use graph_embedding_util as ge;

/// Every driver flag both commands drive the fit with, borrowed from
/// whichever command's own `*Args` built this. Constructed by
/// `BgeArgs::knobs` / `GemArgs::knobs`.
pub(crate) struct EmbedKnobs<'a> {
    pub embedding_dim: usize,

    // Multilevel pseudobulk collapse (see the module doc for why this is
    // flattened rather than a shared `&CollapseArgs`).
    pub num_levels: usize,
    pub sort_dim: usize,
    pub knn_pb_samples: usize,
    pub num_opt_iter: usize,
    pub proj_dim: usize,
    pub bulk_batches: Option<&'a [Box<str>]>,
    /// Carry the finest collapse level forward as `{out}.pb_reference.zarr`
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

    /// The `--gene-modules` flag group, when this command has one. `None`
    /// for gem (not offered on its arg surface yet) unconditionally disables
    /// learned gene modules, matching what it always did.
    pub modules: Option<&'a ge::GeneModuleArgs>,
    /// The module count this command trains when `--gene-modules` is not
    /// passed explicitly (bge: `Some(DEFAULT_GENE_MODULES)`); meaningless
    /// when `modules` is `None`.
    pub default_gene_modules: Option<usize>,

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
    /// Reserved for the track-aware gem (a later task); `None` = one track.
    /// Neither read nor set to anything but `None` until that task wires a
    /// consumer — carried on the plan now so its shape doesn't change later.
    #[allow(dead_code)]
    pub tracks: Option<()>,
    /// Unused until tracks exist; carried through so the field exists on the
    /// plan before it has a consumer.
    #[allow(dead_code)]
    pub offset_l2: f32,
    pub pb_reference: Option<&'a ReferenceInput>,
    pub init_from: Option<&'a str>,
    pub train_args: crate::run_manifest::TrainArgsRecord,
    /// Called after the module tables are written and before the manifest.
    /// Always `None` in this task (both callers); reserved for the
    /// track-aware gem to hook a per-track output writer in here.
    #[allow(clippy::type_complexity)]
    pub after_fit: Option<&'a dyn Fn(&FitArtifacts<'_>) -> anyhow::Result<()>>,
}

/// What [`EmbedPlan::after_fit`] sees. Unused (no caller passes `after_fit`)
/// until the track-aware gem does; the fields are allowed dead for now.
#[allow(dead_code)]
pub(crate) struct FitArtifacts<'a> {
    pub out: &'a ge::FitOutput,
    pub unified: &'a ge::UnifiedData,
    pub qc_keep: Option<&'a [usize]>,
    pub prefix: &'a str,
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
    let build_config = |unified: &ge::UnifiedData| -> anyhow::Result<ge::FitConfig> {
        let hvg_weights = plan.hvg_weights.as_ref().map(|w| {
            unified
                .feature_to_backend_row
                .iter()
                .map(|&i| w[i])
                .collect::<Vec<f32>>()
        });
        let gene_modules = match knobs.modules {
            Some(m) => match m.resolve(knobs.default_gene_modules)? {
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
            num_levels: knobs.num_levels,
            sort_dim: knobs.sort_dim,
            knn_pb_samples: knobs.knn_pb_samples,
            num_opt_iter: knobs.num_opt_iter,
            proj_dim: knobs.proj_dim,
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
            gene_modules,
            tracks: None,
            offset_l2: 0.0,
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

        // Raw ρ, on BOTH paths. This is the model-axis loading that pairs with
        // the cell embedding in the Poisson rate `exp(ρ_g·z_n + a_g + b_n)` —
        // NOT interchangeable with the co-embed written just above, which is a
        // LOSSY derived view of it (a convex combination of cell embeddings;
        // ρ → co-embed is one-way).
        let rho_mat = Mat::from_tensor(&e_feat_cpu)?;
        let rho_h_names = axis_id_names("h", rho_mat.ncols());
        rho_mat.to_parquet_with_names(
            &format!("{}.feature_loading.parquet", knobs.out),
            (Some(&plan.unified.feature_names), Some("gene")),
            Some(&rho_h_names),
        )?;

        // Output layout: the H-space cell embedding Z ALWAYS goes to
        // {out}.cell_embedding.parquet, on both paths. ETM resolved (default)
        // additionally emits the topic-model tables (latent = log θ,
        // dictionary = β); --skip-etm emits no latent at all and keeps
        // dictionary = ρ.
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
        // The learned-module tables, on both paths; the composed ρ above already
        // carries them for every reader that does not care.
        ge::write_module_tables(knobs.out, &out.model, &plan.unified.feature_names)?;

        if let Some(f) = plan.after_fit {
            f(&FitArtifacts {
                out: &out,
                unified: &plan.unified,
                qc_keep: qc_keep_idx.as_deref(),
                prefix: knobs.out,
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
    // The map phase 2 placed the cells with, so `predict` places a query by the
    // same one. One self-contained file: the trunk plus its per-gene mean.
    let cell_encoder_suffix = match out.cell_encoder.as_ref() {
        Some(enc) => {
            let suffix = "cell_encoder.safetensors";
            let path = format!("{}.{suffix}", knobs.out);
            enc.save(&path)?;
            info!("Wrote the cell encoder to {path}");
            Some(suffix)
        }
        None => None,
    };
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
        // With ETM resolved the dictionary is β (gene × topic); otherwise it IS ρ.
        //
        // ρ does NOT go to feature_embedding.parquet — that file is always the SIMBA co-embed. ρ
        // lives on the model's own axis, not on the cell manifold, so putting it there would hand
        // `annotate-by-projection` an off-manifold gene table and make its Euclidean
        // nearest-centroid call ill-posed.
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_reference_suffix,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        // The SIMBA co-embed is written as feature_embedding.parquet in BOTH
        // the ETM and --skip-etm paths, so record it unconditionally.
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_loading_suffix: Some("feature_loading.parquet"),
        // Learned gene modules, when the run trained them; the composed row still
        // lives in `feature_loading`, so these are additive.
        module_membership_suffix: has_modules.then_some(ge::transfer::MODULE_MEMBERSHIP_SUFFIX),
        module_dictionary_suffix: has_modules.then_some(ge::transfer::MODULE_DICTIONARY_SUFFIX),
        // ETM resolved => `dictionary` holds the log-simplex β; --skip-etm => it is ρ.
        softmax_dictionary_suffix: resolve_etm.then_some("dictionary.parquet"),
        // Z always lands in cell_embedding.parquet — on BOTH the ETM and
        // --skip-etm paths — so every geometry consumer finds the H-space
        // embedding at one fixed name.
        cell_embedding_suffix: Some("cell_embedding.parquet"),
        cell_encoder_suffix,
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
            "Done — outputs at {}.{{cell_embedding,latent,dictionary,feature_embedding,*_bias}}.parquet \
             (cell_embedding = Z, latent = log θ)",
            knobs.out
        );
    } else {
        info!(
            "Done — outputs at {}.{{cell_embedding,dictionary,feature_embedding,*_bias}}.parquet \
             (cell_embedding = Z; no latent — topics were not resolved)",
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

/// Module count `senna bge` trains unless told otherwise — the policy is this
/// command's, so it lives here rather than in the shared flag group.
const DEFAULT_GENE_MODULES: usize = 128;

impl super::BgeArgs {
    pub(crate) fn knobs(&self) -> EmbedKnobs<'_> {
        EmbedKnobs {
            embedding_dim: self.embedding_dim,
            num_levels: self.collapse.num_levels,
            sort_dim: self.collapse.sort_dim,
            knn_pb_samples: self.collapse.knn_cells,
            num_opt_iter: self.collapse.iter_opt,
            proj_dim: self.collapse.proj_dim,
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
            default_gene_modules: Some(DEFAULT_GENE_MODULES),
            out: &self.out,
            batch_files: self.batch_files.as_deref(),
        }
    }
}

/// gem has no `--modules-per-unit` flag of its own yet (Task 5a's redesign);
/// this fixes it at the value `run_gem_genes_bge` always passed before this
/// driver existed.
const GEM_MODULES_PER_UNIT: usize = 8;

impl crate::gem::args::GemArgs {
    pub(crate) fn knobs(&self) -> EmbedKnobs<'_> {
        EmbedKnobs {
            embedding_dim: self.model.embedding_dim,
            num_levels: self.collapse.num_levels,
            sort_dim: self.collapse.sort_dim,
            knn_pb_samples: self.collapse.knn_pb,
            num_opt_iter: self.collapse.num_opt_iter,
            proj_dim: self.collapse.proj_dim,
            // gem has no `--mixture-batch` / carried-reference surface yet.
            bulk_batches: None,
            emit_pb_reference: false,
            // gem always refines (no `--no-refine` flag); geu's multilevel
            // collapse requires a refine spec (it surfaces the per-level
            // cell→pb maps phase-2 needs) — geu's defaults, same as a
            // `senna bge` run without `--no-refine`.
            refine: Some(ge::RefineParams::default()),
            qc: &self.qc,
            phase1_cells_per_pb: self.collapse.phase1_cells_per_pb,
            modules_per_unit: GEM_MODULES_PER_UNIT,
            // gem has no `--skip-etm` flag yet; resolve topics by default,
            // same as bge.
            skip_etm: false,
            num_topics: None,
            epochs: self.train.epochs,
            batches_per_epoch: self.train.batches_per_epoch,
            batch_size: self.train.batch_size,
            learning_rate: self.train.learning_rate,
            weight_decay: self.train.weight_decay,
            // gem has no `--block-size` flag of its own.
            block_size: None,
            seed: self.runtime.seed,
            device: &self.runtime.device,
            device_no: self.runtime.device_no,
            // Learned gene modules are not offered on gem's arg surface yet.
            modules: None,
            default_gene_modules: None,
            out: &self.out,
            batch_files: self.batch_files.as_deref(),
        }
    }
}

#[cfg(test)]
#[path = "driver/tests.rs"]
mod tests;
