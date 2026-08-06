//! `senna update` — absorb a new batch into a trained masked model.
//!
//! This is the mutating counterpart to `senna probe`: probe measures whether an
//! update is worth making, `update` makes it. It is the first subcommand outside
//! the training runs to emit a `.safetensors` + `.model.json` pair.
//!
//! **The update already existed, unpersisted.** `counterfactual::refit_alpha`
//! refits the topic embeddings `α` with the encoder frozen and `θ` held fixed;
//! probe's *treatment arm* runs exactly that and then discards `α₁`. `update`
//! keeps it.
//!
//! **Governing constraint: `update` enacts precisely the intervention `probe`
//! measures** — same block (`α` only), same frozen encoder, same replay-style fit
//! set. Diverge from that and `benefit`/`forgetting` describe a different
//! operation than the one performed, and the decision layer stops meaning
//! anything.
//!
//! **λ is the replay ratio, not a Fisher penalty.** probe's arms already train on
//! `base ∪ extra`, so replay *is* the regularizer, and the ratio is the dial: larger
//! values protect old knowledge harder and yield less benefit.
//!
//! ⚠️ It does **not** currently reproduce probe's own sizing. probe's treatment arm
//! replays `|c_base| = (2/3)·n_calib − n_query/2` against `n_query/2`, so its effective
//! ratio depends on both set sizes (≈3.0 for a 3000/1000 split) — while `update` at 1.0
//! replays `n_new` against `n_new`. The governing constraint above is therefore honoured
//! for the *block refit* but not for the *replay mass*; matching them needs a shared
//! split policy, which `counterfactual::splits` should grow to own.
//!
//! **Never in place.** A model is a scientific artifact: `-o` must differ from
//! `--model`, and the parent is opened read-only.

use crate::counterfactual::{
    alpha_var, detached_copy, mean, rebuild_model, refit_alpha, row_norms_of_diff, score_cells,
    BankArgs, BankSource, CellBank, RefitCfg,
};
use crate::embed_common::*;
use crate::masked_topic::FeatureNameKindArg;
use crate::predict::{score_masked_backend, MaskedScoreArgs, MaskedScored};
use crate::topic::eval::QueryNameOpts;
use crate::topic::model_metadata::{
    load_feature_mean, load_shortlist_weights, masked_head_from_model_type, save_feature_mean,
    save_parameters, save_shortlist_weights, TopicModelMetadata,
};
use crate::topic::train_masked::{write_feature_embedding, write_masked_dictionary};
use log::info;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

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
        help = "Parent model prefix (output of `senna masked-topic/-vae/-sbp` -o)"
    )]
    model: Box<str>,

    #[arg(
        long,
        required = true,
        help = "Reference backend replayed alongside the new data",
        long_help = "Cells the parent model already explains. A fraction of them is replayed\n\
                     in the refit; that replay is what protects old knowledge, and its size\n\
                     is the λ knob. Required — an unregularized fine-tune should never be\n\
                     the accidental default."
    )]
    calibration: Box<str>,

    #[arg(short, long, required = true, help = "Output prefix for the NEW model")]
    out: Box<str>,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Replay mass as a multiple of the new batch size (λ knob)",
        long_help = "Include `ratio × n_new` reference cells in the refit, capped so a third of\n\
                     the reference stays unreplayed to measure the effect on. Larger values\n\
                     protect old knowledge harder and yield less benefit; 0 disables replay\n\
                     (unprotected, not recommended). NOTE this does not yet match probe's own\n\
                     replay sizing — see the module docs."
    )]
    replay_ratio: f64,

    #[arg(long, default_value_t = 100, help = "SGD steps for the refit")]
    steps: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "AdamW learning rate for the refit"
    )]
    lr: f64,

    #[arg(long, default_value_t = 42, help = "Seed for replay-cell sampling")]
    seed: u64,

    #[arg(long, default_value_t = 500, help = "Evaluation minibatch size")]
    minibatch_size: usize,

    #[arg(long, help = "Load all columns into memory before scoring")]
    preload_data: bool,
}

pub fn run_update(args: &UpdateArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.out.as_ref() != args.model.as_ref(),
        "update refuses to write in place: -o ({}) must differ from --model ({}). \
         A model is a versioned artifact — write M_v2, keep M.",
        args.out,
        args.model
    );
    mkdir_parent(&args.out)?;

    let metadata = TopicModelMetadata::load(&args.model)?;
    let head = masked_head_from_model_type(&metadata.model_type).ok_or_else(|| {
        anyhow::anyhow!(
            "update supports masked models only (masked-topic/-vae/-sbp); got '{}'",
            metadata.model_type
        )
    })?;
    let context_size = metadata
        .enc_context_size
        .ok_or_else(|| anyhow::anyhow!("update: metadata missing enc_context_size"))?;

    ////////////////////////////////////////
    // 1. Score both sides, encoder frozen //
    ////////////////////////////////////////

    let qopts = QueryNameOpts {
        kind: FeatureNameKindArg::Exact.resolve_or_gene(),
        suffix_delim: None,
        keep_suffix: None,
    };
    let scored = |files: &[Box<str>]| -> anyhow::Result<MaskedScored> {
        score_masked_backend(MaskedScoreArgs {
            model: &args.model,
            data_files: files,
            batch_files: None,
            preload: args.preload_data,
            minibatch_size: args.minibatch_size,
            query_name_opts: &qopts,
            metadata: &metadata,
            head,
        })
    };
    let cal = scored(std::slice::from_ref(&args.calibration))?;
    let new = scored(&args.data_files)?;
    let n_new = new.z_nk.nrows();
    let n_cal = cal.z_nk.nrows();
    anyhow::ensure!(n_new > 0, "update: the new batch has no cells");

    /////////////////////////////////////////////
    // 2. Rebuild ALL levels and load the parent //
    /////////////////////////////////////////////

    let dev = candle_core::Device::Cpu;
    let rebuilt = rebuild_model(&args.model, &metadata, &dev)?;
    let alpha = alpha_var(&rebuilt.parameters, &rebuilt.alpha_name)?;
    let alpha0 = detached_copy(alpha.as_tensor())?;

    // `load_feature_mean` already returns the gene axis, so no separate dictionary read.
    let (gene_names, feature_mean) = load_feature_mean(&args.model)?;
    let (_, shortlist) = load_shortlist_weights(&args.model)?;

    let bank = CellBank::build(BankArgs {
        calib: BankSource {
            data_vec: &cal.data_vec,
            z_nk: &cal.z_nk,
            gene_remap: cal.gene_remap.as_ref(),
        },
        query: BankSource {
            data_vec: &new.data_vec,
            z_nk: &new.z_nk,
            gene_remap: new.gene_remap.as_ref(),
        },
        context_size,
        feature_mean: &feature_mean,
        shortlist_weights: &shortlist,
        dev: &dev,
    })?;
    // The bank owns copies now; these hold a whole SparseIoVec each (the full matrix
    // under --preload-data) and are dead from here on. Peak RSS is in the refit below.
    drop(cal);
    drop(new);

    ///////////////////////////////////////
    // 3. Refit α on (replay ∪ new cells) //
    ///////////////////////////////////////

    // CellBank packs calibration first, then the new batch. Reserve a third of the
    // reference (mirroring probe's `c_eval`) so there is always an untouched slice to
    // report the realized effect on; replay is drawn from the remaining two thirds.
    let n_replay_cap = n_cal - (n_cal / 3).max(1);
    let n_wanted = (args.replay_ratio * n_new as f64).round() as usize;
    let n_replay = n_wanted.min(n_replay_cap);

    let mut cal_ids: Vec<usize> = (0..n_cal).collect();
    let mut rng = StdRng::seed_from_u64(args.seed);
    cal_ids.shuffle(&mut rng);
    let mut fit_ids: Vec<usize> = cal_ids[..n_replay].to_vec();
    fit_ids.extend(n_cal..n_cal + n_new);

    info!(
        "update: refitting α on {} cells ({} new + {} replayed of {} reference); {} steps, lr {}",
        fit_ids.len(),
        n_new,
        n_replay,
        n_cal,
        args.steps,
        args.lr
    );
    if n_replay < n_wanted {
        log::warn!(
            "--replay-ratio {:.2} wanted {n_wanted} replay cells but at most {n_replay_cap} are \
             available (of {n_cal} reference cells, a third is reserved unreplayed to measure the \
             effect on). The EFFECTIVE ratio is {:.2}, so old knowledge is protected less than \
             requested — supply a larger --calibration for a batch this size.",
            args.replay_ratio,
            n_replay as f64 / n_new as f64,
        );
    }
    if n_replay == 0 {
        log::warn!(
            "zero replay cells: this is an unprotected fine-tune and will forget the reference."
        );
    }

    // Two evaluation sets with DIFFERENT status, and the report must say which is which:
    //   held_ref — reference cells deliberately kept out of the refit: genuinely held out.
    //   all_new  — every new cell, all of which ARE in `fit_ids`: in-sample training fit.
    // The new side is in-sample by design — an update should train on all the data it is
    // given — so its number is optimistic and is labelled as such rather than fixed.
    let held_ref: Vec<usize> = cal_ids[n_replay..].to_vec();
    let all_new: Vec<usize> = (n_cal..n_cal + n_new).collect();
    let score = |ids: &[usize]| score_cells(&rebuilt.decoder, &bank, ids);

    let before_ref = score(&held_ref)?;
    let before_new = score(&all_new)?;

    refit_alpha(
        &rebuilt.decoder,
        &alpha,
        &bank,
        &fit_ids,
        &RefitCfg {
            steps: args.steps,
            lr: args.lr,
        },
    )?;

    // Equal-length slices, so mean-of-differences is the difference of means.
    let d_ref = mean(&score(&held_ref)?) - mean(&before_ref);
    let d_new = mean(&score(&all_new)?) - mean(&before_new);

    let moved = row_norms_of_diff(alpha.as_tensor(), &alpha0)?;
    info!(
        "update: per-topic ||α₁_k − α₀_k||: [{}]",
        moved
            .iter()
            .map(|v| format!("{v:.3e}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Both are ABSOLUTE changes against the parent, not probe's contrast. The parent is
    // not a stationary point of the per-cell scoring objective it is measured under (it
    // trained on masked-NB over pseudobulks), so *any* refit — including one on a batch
    // that adds nothing — converges the model toward that objective and improves both
    // numbers. probe cancels exactly that term with its control arm; `update` has one arm
    // and cannot. Read these as "what this update did"; read probe's benefit/forgetting
    // for "what this BATCH contributed".
    info!(
        "update: realized per-cell fit change — new batch {d_new:+.4e} (IN-SAMPLE, {} cells, all \
         trained on), held-out reference {d_ref:+.4e} ({} cells). Absolute vs the parent, not \
         probe's contrast: they include the batch-independent objective-convergence term.",
        all_new.len(),
        held_ref.len()
    );

    ///////////////////////////////////////////////////////
    // 4. Persist. Order matters: the dictionary is derived //
    ///////////////////////////////////////////////////////

    // The dictionary is `log_softmax_d(α·ρᵀ)`, and `probe`/`predict` score against
    // the PARQUET, not the safetensors. Regenerating it here is what keeps scoring
    // in sync with the weights — skip it and the new model silently scores as the
    // old one.
    write_masked_dictionary(&rebuilt.decoder, &gene_names, &args.out)?;
    // ρ is unchanged (that is the thesis), but the artifact must be self-contained
    // for `--freeze-feature-embedding` / `--warm-start-rho` to resolve against it.
    write_feature_embedding(rebuilt.decoder.feature_embeddings(), &gene_names, &args.out)?;
    // Gene axis is unchanged, so these carry over verbatim.
    save_shortlist_weights(&shortlist, &gene_names, &args.out)?;
    save_feature_mean(&feature_mean, &gene_names, &args.out)?;

    save_parameters(&rebuilt.parameters, &args.out)?;

    // `n_train_cells` becomes the running `N_absorbed` the net-gain ranking needs;
    // `theta_mean` is carried forward rather than recomputed (the masked path never
    // reads it, and recomputing needs a full encoder pass over the parent's cells).
    let mut new_meta = metadata.clone();
    new_meta.n_train_cells = Some(metadata.n_train_cells.unwrap_or(0) + n_new);
    new_meta.save(&args.out)?;

    info!(
        "update: wrote {} (n_train_cells {} → {})",
        args.out,
        metadata.n_train_cells.unwrap_or(0),
        new_meta.n_train_cells.unwrap_or(0)
    );
    info!("update: parent {} left untouched", args.model);
    Ok(())
}
