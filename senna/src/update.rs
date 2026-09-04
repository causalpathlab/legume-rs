//! `senna update` — absorb new samples into a trained model by continuing its
//! training, rather than refitting the cohort from scratch.
//!
//! **This is a dispatcher, not a trainer.** It reconstructs the parent's own
//! fit call — recorded in its manifest as [`crate::run_manifest::TrainArgsRecord`]
//! — points it at `recorded ∪ new` data, turns on warm start, and hands the
//! whole thing to the unchanged family entry point. No new estimator, no second
//! training path to keep in sync with the first.
//!
//! **Why that is enough.** Every family already trains purely on pseudobulks:
//! `train_mixed` and `train_masked` consume only `&[CollapsedOut]`, and `svd`
//! runs `rsvd` on the pseudobulk posterior. Re-running the fit over the union
//! therefore gives PB-level retraining, *exact* cell-level replay of the old
//! cohort, and a batch adjustment where old and new cells are matched at **cell**
//! resolution by `collect_matched_stat_visitor` — stronger than anything that
//! works from stored pseudobulk summaries.
//!
//! What this costs is time: each round re-reads every previously absorbed cell,
//! so absorbing S samples one at a time is O(S²) in cell reads. That is the
//! known trade and the reason this is the correctness baseline rather than the
//! final word.
//!
//! **The partition is deliberately not inherited.** `--from` would pull the
//! parent's cell→pb membership along with its inputs, but
//! `align_cell_to_pb_to_cells` bails on any cell absent from the source — which
//! every new cell is. `from` is forced to `None`.

use crate::embed_common::*;
use crate::run_manifest::{RunKind, RunManifest};
use std::path::{Path, PathBuf};

/// The four things `update` changes about a recorded fit.
///
/// Applied by each family through [`Updatable`], implemented next to its own
/// arg struct so the field names stay where they are declared.
pub(crate) struct Rebase {
    /// Recorded inputs followed by the new ones.
    pub data_files: Vec<Box<str>>,
    /// Recorded batch files followed by the new ones; `None` when the cohort
    /// has no batch labels at all.
    pub batch_files: Option<Vec<Box<str>>>,
    /// Where the updated model is written.
    pub out: Box<str>,
    /// Parent prefix to warm-start from. Ignored by families without weights.
    pub init_from: Box<str>,
    /// Per-round epoch override; `None` keeps the recorded count.
    pub epochs: Option<usize>,
    /// The parent's own K and H. Growth resolves against these rather than
    /// against the recorded arguments, because a recorded `--embedding-dim 0`
    /// means "auto = 2K" — which would silently resize ρ the moment K grows.
    pub parent_topics: usize,
    pub parent_embedding_dim: Option<usize>,
    /// Capacity to add. Zero on both axes is an ordinary continue.
    pub growth: crate::topic::warm_start::Growth,
    /// The parent's carried pseudobulks when they are standing in for its
    /// cells; `None` means this round re-collapses.
    pub reference: Option<crate::pb_reference::ReferenceInput>,
}

/// A fit whose recorded arguments can be re-pointed at a larger cohort.
pub(crate) trait Updatable {
    fn rebase(&mut self, r: Rebase);
}

#[derive(Args, Debug)]
pub struct UpdateArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "New data files to absorb (.zarr or .h5)"
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        help = "Parent model prefix to continue training from",
        long_help = "A run prefix written by `senna topic / masked-topic / masked-sbp /\n\
                     masked-vae / vae / svd / bge / simba`."
    )]
    model: Box<str>,

    #[arg(
        short,
        long,
        required = true,
        help = "Output prefix for the updated model",
        long_help = "Must differ from --model. A trained model is a versioned\n\
                     artifact: write M_v2 and keep M_v1."
    )]
    out: Box<str>,

    #[arg(
        long,
        short,
        value_delimiter = ',',
        help = "Batch files for the NEW data, one per new data file",
        long_help = "Required when the parent had batch files, because the loader\n\
                     needs one batch file per data file across the whole cohort.\n\
                     The parent's recorded list is prepended automatically."
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        help = "Epochs for this round (default: the parent's recorded count)",
        long_help = "Continuing from trained weights usually needs fewer epochs than\n\
                     the original fit. Omit to reuse whatever the parent used."
    )]
    epochs: Option<usize>,

    #[arg(
        long,
        help = "Reuse the parent's carried pseudobulks instead of re-reading its cells",
        long_help = "Needs the parent to have been trained with --emit-pb-reference.\n\
                     \n\
                     Absorbing a sample normally re-collapses the whole cohort, so\n\
                     taking S samples one at a time re-reads every earlier cell each\n\
                     time. This substitutes the parent's stored pseudobulks for its\n\
                     cells, making a round cost the NEW data only.\n\
                     \n\
                     The trade is resolution: old-vs-new batch matching drops from\n\
                     cell level to pseudobulk level, and it is only a saving when a\n\
                     pseudobulk stands for many cells. `update` reports the ratio\n\
                     and says so when it does not.\n\
                     \n\
                     One consequence: the old cells are never loaded, so\n\
                     {out}.latent.parquet covers the NEW cells only. The parent's\n\
                     latent remains the record for everything absorbed earlier.\n\
                     \n\
                     Off by default — re-collapsing is the exact computation."
    )]
    use_pb_reference: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Grow the model by N topics while absorbing (topic / masked-* only)",
        long_help = "Replay alone cannot represent biology the parent has no topic for —\n\
                     the new cohort has to distort an existing topic to be explained.\n\
                     This adds capacity instead.\n\
                     \n\
                     Added topics start switched off (~0 mass) and the parent's keep\n\
                     their indices, so existing annotations stay valid. Off by default:\n\
                     K is part of every downstream artifact's identity."
    )]
    add_topics: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Grow the gene embedding ρ by N dimensions (masked family only)",
        long_help = "Widens H. Exactly function-preserving at step 0 — β is unchanged bit\n\
                     for bit — while the added subspace still receives gradient.\n\
                     \n\
                     Only the masked family has a ρ to widen; `topic`, `vae` and `svd`\n\
                     reject this."
    )]
    add_embedding_dim: usize,
}

/// Locate the parent's recorded input files.
///
/// The two candidate readings exist because the manifest is asymmetric.
/// *Outputs* are stored as bare basenames, so resolving them against the
/// manifest's directory is exactly right and a run directory can be moved.
/// *Data inputs* are stored *verbatim as the training command spelled them*
/// (`write_run_manifest` assigns `desc.data_input` unchanged) — i.e. relative
/// to the training **cwd**, which equals the manifest's directory only when the
/// run was written with a bare `-o prefix`.
///
/// So neither reading alone is correct: as-recorded breaks a moved or copied
/// run directory, and manifest-relative breaks `-o subdir/prefix`. Prefer the
/// path that exists, which succeeds wherever either would, and fall back to
/// as-recorded so the loader reports the name the user would recognise.
///
/// The real fix is to relativize data paths at write time the way outputs
/// already are; that changes every producer and every manifest on disk, so it
/// is deliberately not done here. `inherit_from` has the same gap.
fn recorded_paths(recorded: &[String], dir: &Path) -> Vec<Box<str>> {
    recorded
        .iter()
        .map(|s| {
            if Path::new(s).exists() {
                return s.as_str().into();
            }
            let rel = crate::run_manifest::resolve(dir, s);
            if rel.exists() {
                return rel.to_string_lossy().into_owned().into();
            }
            s.as_str().into()
        })
        .collect()
}

/// Recorded inputs followed by the new ones.
///
/// A repeat is rejected rather than deduplicated: passing a file the parent
/// already trained on means either the wrong file or the wrong parent, and
/// silently ignoring it would double a cohort's apparent growth in the log
/// while changing nothing. Both sides are compared after canonicalization so
/// two spellings of one path still count as a repeat.
fn union_inputs(recorded: Vec<Box<str>>, new: &[Box<str>]) -> anyhow::Result<Vec<Box<str>>> {
    let canon = |p: &str| std::fs::canonicalize(p).unwrap_or_else(|_| PathBuf::from(p));
    let seen: Vec<PathBuf> = recorded.iter().map(|r| canon(r)).collect();
    for n in new {
        anyhow::ensure!(
            !seen.contains(&canon(n)),
            "{n} is already part of the parent model's training data. `update` absorbs data \
             the model has NOT seen — check the file, or the --model prefix."
        );
    }
    let mut out = recorded;
    out.extend(new.iter().cloned());
    Ok(out)
}

/// Recorded batch files followed by the new ones, with the arity the loader
/// requires (one per data file) checked on both halves.
fn union_batches(
    recorded: Vec<Box<str>>,
    new: Option<&[Box<str>]>,
    n_new_data: usize,
) -> anyhow::Result<Option<Vec<Box<str>>>> {
    match (recorded.is_empty(), new) {
        (true, None) => Ok(None),
        (true, Some(_)) => anyhow::bail!(
            "--batch-files was given but the parent model was trained without batch labels. \
             Batch files are one-per-data-file across the whole cohort, so the parent's files \
             would have none. Re-train the parent with batch labels, or drop --batch-files."
        ),
        (false, None) => anyhow::bail!(
            "the parent model was trained with batch labels, so the new data needs them too \
             (one --batch-files entry per new data file). Without them the loader cannot pair \
             batch labels to cells."
        ),
        (false, Some(new)) => {
            anyhow::ensure!(
                new.len() == n_new_data,
                "--batch-files has {} entries but {n_new_data} new data file(s) were given; \
                 the loader needs exactly one batch file per data file.",
                new.len(),
            );
            let mut out = recorded;
            out.extend(new.iter().cloned());
            Ok(Some(out))
        }
    }
}

pub fn run_update(args: &UpdateArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.out.as_ref() != args.model.as_ref(),
        "update refuses to write in place: -o ({}) must differ from --model ({}). A trained \
         model is a versioned artifact — write M_v2 and keep M_v1.",
        args.out,
        args.model,
    );
    mkdir_parent(&args.out)?;

    // One load, and the kind comes from it. `resolve_run_kind` would parse the
    // same file only to return `.kind`, and its `{prefix}.model.json` fallback
    // cannot help here anyway: without a manifest there is no recorded fit to
    // replay, so a manifest-less prefix has to fail — with this message rather
    // than a bare io error one line later.
    let manifest_path = PathBuf::from(crate::run_manifest::default_path(&args.model));
    let (manifest, dir) = RunManifest::load(&manifest_path).map_err(|e| {
        anyhow::anyhow!(
            "{e}\n`senna update` replays the parent's recorded fit, which lives in its run \
             manifest. A prefix without one cannot be continued — re-train it, or drive the \
             family command directly with `--init-from {}`.",
            args.model,
        )
    })?;
    let kind = manifest.kind;

    // Either substitute the parent's carried pseudobulks for its cells, or
    // re-read the cells. The substitution is what turns a round from
    // "every cell ever absorbed" into "the new cells only".
    let reference = if args.use_pb_reference {
        anyhow::ensure!(
            kind != RunKind::Simba,
            "--use-pb-reference does not apply to a simba run: it trains on cells, never on \
             pseudobulks, so there is nothing to substitute. Drop the flag."
        );
        let r = crate::pb_reference::prepare(&args.model, &args.out)?.ok_or_else(|| {
            anyhow::anyhow!(
                "{} carries no pseudobulks, so there is nothing to substitute for its cells. \
                 Re-train it with --emit-pb-reference, or drop --use-pb-reference and let this \
                 round re-collapse.",
                args.model,
            )
        })?;
        Some(r)
    } else {
        None
    };

    let (data_files, batch_files) = if let Some(r) = reference.as_ref() {
        // The reference goes LAST: `weights_for` keys on that, and the loader
        // concatenates columns in file order.
        let mut d: Vec<Box<str>> = args.data_files.clone();
        d.push(r.backend.clone());
        let b = match args.batch_files.as_deref() {
            Some(new_b) => {
                anyhow::ensure!(
                    new_b.len() == args.data_files.len(),
                    "--batch-files has {} entries but {} new data file(s) were given",
                    new_b.len(),
                    args.data_files.len(),
                );
                let mut v = new_b.to_vec();
                v.push(r.batch_file.clone());
                Some(v)
            }
            None => anyhow::bail!(
                "--use-pb-reference needs --batch-files for the new data: the carried \
                 pseudobulks are their own batch, and the loader takes one batch file per \
                 data file."
            ),
        };
        (d, b)
    } else {
        (
            union_inputs(recorded_paths(&manifest.data.input, &dir), &args.data_files)?,
            union_batches(
                recorded_paths(&manifest.data.batch, &dir),
                args.batch_files.as_deref(),
                args.data_files.len(),
            )?,
        )
    };

    // Deliberately does not claim "warm-starting": `svd` has no weights, and
    // each arm below says what it actually does.
    if let Some(r) = reference.as_ref() {
        let (n_cols, n_cells) = (r.cell_counts.len() as f32, r.cells_represented());
        let ratio = n_cells / n_cols.max(1.0);
        info!(
            "update [{kind}]: reusing {} carried pseudobulks in place of {} cells ({ratio:.1} \
             cells each), plus {} new file(s)",
            n_cols as usize,
            n_cells as usize,
            args.data_files.len(),
        );
        // The saving is the ratio. Below ~2 the pseudobulks are near-singletons,
        // so this costs pseudobulk-level batch matching and buys almost nothing.
        if ratio < 2.0 {
            log::warn!(
                "carried pseudobulks hold {ratio:.1} cells each, so substituting them saves \
                 little while coarsening old-vs-new batch matching. Re-collapsing (drop \
                 --use-pb-reference) is likely the better trade at this scale."
            );
        }
    } else {
        info!(
            "update [{kind}]: continuing {} from {} recorded + {} new = {} data file(s)",
            args.model,
            manifest.data.input.len(),
            args.data_files.len(),
            data_files.len(),
        );
    }

    let growth = crate::topic::warm_start::Growth {
        add_topics: args.add_topics,
        add_embedding_dim: args.add_embedding_dim,
    };
    // Only the checkpointed families have anything to grow, and the parent's
    // own K / H are the base to grow from — the recorded arguments may say
    // `--embedding-dim 0`, meaning "auto", which would track the grown K.
    let (parent_topics, parent_embedding_dim) = if growth.is_none() {
        (0, None)
    } else {
        anyhow::ensure!(
            kind.is_masked_family() || kind == RunKind::Topic,
            "growth is not available for a '{kind}' run. `--add-topics` needs a checkpoint to \
             widen, which only topic and the masked family have."
        );
        // Rejected here rather than at warm start: only the masked args carry an
        // `add_embedding_dim`, so on a dense parent the flag would otherwise be
        // dropped on the floor and the user would sit through a full retrain
        // that did nothing they asked for.
        anyhow::ensure!(
            args.add_embedding_dim == 0 || kind.is_masked_family(),
            "--add-embedding-dim has no meaning for a '{kind}' run: its decoder has no per-gene \
             embedding ρ to widen. Use --add-topics to add capacity here, or the masked family \
             if you want a wider embedding."
        );
        let m = crate::topic::model_metadata::TopicModelMetadata::load(&args.model)?;
        info!(
            "growth: K {} → {}{}",
            m.n_topics,
            m.n_topics + growth.add_topics,
            match (m.embedding_dim, growth.add_embedding_dim) {
                (Some(h), a) if a > 0 => format!(", H {h} → {}", h + a),
                _ => String::new(),
            },
        );
        (m.n_topics, m.embedding_dim)
    };

    // A value, not a closure: the match arms are mutually exclusive, so exactly
    // one moves it and none of the vectors need cloning.
    let rebase = Rebase {
        data_files,
        batch_files,
        out: args.out.clone(),
        init_from: args.model.clone(),
        epochs: args.epochs,
        parent_topics,
        parent_embedding_dim,
        growth,
        reference,
    };

    match kind {
        RunKind::Topic => {
            let mut a: crate::topic::cmd::TopicArgs = manifest.train_args_as(&args.model)?;
            a.rebase(rebase);
            crate::topic::cmd::fit_topic_model(&a)
        }
        RunKind::Vae => {
            let mut a: crate::vae::VaeArgs = manifest.train_args_as(&args.model)?;
            a.rebase(rebase);
            crate::vae::fit_vae_model(&a)
        }
        // `svd` has no weights and no `--init-from`: this re-fits on the union
        // with the recorded configuration. Still worth routing here so one
        // command covers the cohort, but it is a refit, not a warm start.
        RunKind::Svd => {
            anyhow::ensure!(
                args.epochs.is_none(),
                "--epochs does not apply to an svd run: it has no training loop, only a \
                 randomized SVD of the pseudobulk matrix."
            );
            let mut a: crate::svd::SvdArgs = manifest.train_args_as(&args.model)?;
            a.rebase(rebase);
            info!("svd has no trainable weights — re-fitting on the union (not a warm start)");
            crate::svd::fit_svd(&a)
        }
        k if k.is_masked_family() => {
            use crate::topic::model_metadata::{masked_head_from_model_type, TopicModelMetadata};

            // `Itopic` covers both masked-topic and masked-sbp; only
            // `model_type` separates them.
            let metadata = TopicModelMetadata::load(&args.model)?;
            let head = masked_head_from_model_type(&metadata.model_type).ok_or_else(|| {
                anyhow::anyhow!(
                    "{}: manifest says '{kind}' but model_type is '{}', which is not a masked \
                     head. The two files disagree — check for a copied prefix.",
                    args.model,
                    metadata.model_type,
                )
            })?;

            let mut a: crate::masked_topic::MaskedTopicArgs =
                manifest.train_args_as(&args.model)?;
            a.rebase(rebase);
            // The three `fit_masked_*_model` wrappers exist for main.rs's
            // subcommand table and each immediately converts back to a head;
            // we already have one.
            crate::masked_topic::fit_masked_model(&a, head)
        }
        // Like `svd`: no weights to warm-start (the ETM is re-derived by
        // archetypal analysis each run), so this is a re-fit on the union —
        // O(new) when the parent carries a pb_reference.
        RunKind::Bge => {
            let mut a: crate::bge::BgeArgs = manifest.train_args_as(&args.model)?;
            a.rebase(rebase);
            info!("bge has no trainable checkpoint — re-fitting on the union (not a warm start)");
            crate::bge::fit_bge(&a)
        }
        // As `svd` and `bge`: the node tables are not a checkpoint to warm-start
        // (they are re-drawn per run), so this is a re-fit on the union.
        RunKind::Simba => {
            let mut a: crate::simba::SimbaArgs = manifest.train_args_as(&args.model)?;
            a.rebase(rebase);
            info!("simba has no trainable checkpoint — re-fitting on the union (not a warm start)");
            crate::simba::fit_simba(&a)
        }
        other => anyhow::bail!(
            "update does not support a '{other}' run. Supported: topic, masked-topic, \
             masked-sbp, masked-vae, vae, svd, bge, simba."
        ),
    }
}
