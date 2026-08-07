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
    CellBank, RefitCfg,
};
use crate::embed_common::*;
use crate::topic::masked_artifact::{write_masked_model, MaskedModel, WriteArgs};
use crate::topic::model_metadata::UpdateRecord;
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

/// How much reference data gets replayed alongside the new batch, and how much is
/// withheld to measure the effect on.
///
/// A third of the reference is reserved unreplayed — mirroring probe's `c_eval` — so
/// there is always an untouched slice to report against; replay is drawn from the
/// remaining two thirds. When the request exceeds that, it is **capped**, which silently
/// weakens the protection the caller asked for, so `capped()` gates a loud warning.
struct ReplayPlan {
    /// Cells the caller asked for: `ratio × n_new`.
    n_wanted: usize,
    /// The most that can be replayed while keeping the measurement reserve.
    n_replay_cap: usize,
    /// What is actually replayed.
    n_replay: usize,
}

impl ReplayPlan {
    fn new(n_cal: usize, n_new: usize, ratio: f64) -> Self {
        let n_replay_cap = n_cal - (n_cal / 3).max(1).min(n_cal);
        let n_wanted = (ratio.max(0.0) * n_new as f64).round() as usize;
        Self {
            n_wanted,
            n_replay_cap,
            n_replay: n_wanted.min(n_replay_cap),
        }
    }

    /// The request could not be honoured in full.
    fn capped(&self) -> bool {
        self.n_replay < self.n_wanted
    }

    /// Replay actually achieved, as a multiple of the new batch — what the caller gets,
    /// which is what the warning must report rather than the ratio they asked for.
    fn effective_ratio(&self, n_new: usize) -> f64 {
        if n_new == 0 {
            0.0
        } else {
            self.n_replay as f64 / n_new as f64
        }
    }
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

    let parent = MaskedModel::open(&args.model)?;

    ////////////////////////////////////////
    // 1. Score both sides, encoder frozen //
    ////////////////////////////////////////

    // `need_llik: false` — the refit uses only `z_nk`, and the per-cell predictive score
    // costs a second full pass over every column plus a dense reconstruction per block.
    let scored =
        |files: &[Box<str>]| parent.score(files, args.preload_data, args.minibatch_size, false);
    let cal = scored(std::slice::from_ref(&args.calibration))?;
    let new = scored(&args.data_files)?;
    let n_new = new.z_nk.nrows();
    let n_cal = cal.z_nk.nrows();
    anyhow::ensure!(n_new > 0, "update: the new batch has no cells");

    // Gene-name agreement. This is a sanity floor only — the identification question is
    // asked below, against ρ, once the model is rebuilt. `None` means the axes matched.
    let n_genes_model = parent.gene_names.len();
    // DISTINCT training genes. `build_gene_remap_with` is many-to-one — two query names can
    // canonicalize onto the same training gene — so a raw flatten would count a gene twice,
    // double-weight its ρ row in the Gram matrix, and let `n_genes_mapped` exceed the axis.
    let mapped_genes: Vec<usize> = match new.gene_remap.as_ref() {
        Some(r) => {
            let mut v: Vec<usize> = r.new_to_train.iter().flatten().copied().collect();
            v.sort_unstable();
            v.dedup();
            v
        }
        None => (0..n_genes_model).collect(),
    };
    let n_genes_mapped = mapped_genes.len();
    let overlap = n_genes_mapped as f64 / n_genes_model.max(1) as f64;
    // No fraction gate here. `build_remap` (predict.rs) already hard-fails below 10% on the
    // scoring call above, so a second, laxer threshold on this side was unreachable — a
    // `--min-gene-overlap` under 0.1 could never fire. The gate that *is* ours is the
    // identification check below, which asks the question a fraction cannot: whether the genes
    // this batch did map to span ρ well enough to pin α.
    //
    // ⚠️ Consequence worth knowing: a narrow-but-spanning panel (say 3% of genes covering all H
    // directions) is legitimate for an embedded dictionary and `alpha_conditioning` would accept
    // it — but `build_remap`'s shared 10% floor rejects it before we ever get here. Lifting that
    // for the embedded path is a separate change.

    /////////////////////////////////////////////
    // 2. Rebuild ALL levels and load the parent //
    /////////////////////////////////////////////

    let dev = candle_core::Device::Cpu;
    let rebuilt = rebuild_model(&args.model, &parent.metadata, &dev)?;
    let alpha = alpha_var(&rebuilt.parameters, &rebuilt.alpha_name)?;
    let alpha0 = detached_copy(alpha.as_tensor())?;

    // How well the batch pins α — the question a gene *fraction* cannot answer.
    // `None` = rank-deficient. Kept as an Option rather than an infinity because a non-finite
    // float serializes to `null` and then fails to deserialize — see `UpdateRecord`.
    let alpha_condition = alpha_conditioning(rebuilt.decoder.feature_embeddings(), &mapped_genes)?;
    info!(
        "update: {n_genes_mapped}/{n_genes_model} genes mapped ({:.1}%); cond(ρ_S) = {} over \
         the model's {} embedding dimensions",
        100.0 * overlap,
        alpha_condition.map_or_else(|| "rank-deficient".to_string(), |c| format!("{c:.3e}")),
        rebuilt.decoder.feature_embeddings().dim(1)?
    );

    let bank = CellBank::from_scored(&parent, &cal, &new, &dev)?;
    // The bank owns copies now; these hold a whole SparseIoVec each (the full matrix
    // under --preload-data) and are dead from here on. Peak RSS is in the refit below.
    drop(cal);
    drop(new);

    ///////////////////////////////////////
    // 3. Refit α on (replay ∪ new cells) //
    ///////////////////////////////////////

    // CellBank packs calibration first, then the new batch.
    let plan = ReplayPlan::new(n_cal, n_new, args.replay_ratio);

    let mut cal_ids: Vec<usize> = (0..n_cal).collect();
    let mut rng = StdRng::seed_from_u64(args.seed);
    cal_ids.shuffle(&mut rng);
    let mut fit_ids: Vec<usize> = cal_ids[..plan.n_replay].to_vec();
    fit_ids.extend(n_cal..n_cal + n_new);

    info!(
        "update: refitting α on {} cells ({} new + {} replayed of {} reference); {} steps, lr {}",
        fit_ids.len(),
        n_new,
        plan.n_replay,
        n_cal,
        args.steps,
        args.lr
    );
    if plan.capped() {
        log::warn!(
            "--replay-ratio {:.2} wanted {} replay cells but at most {} are available (of {n_cal} \
             reference cells, a third is reserved unreplayed to measure the effect on). The \
             EFFECTIVE ratio is {:.2}, so old knowledge is protected less than requested — supply \
             a larger --calibration for a batch this size.",
            args.replay_ratio,
            plan.n_wanted,
            plan.n_replay_cap,
            plan.effective_ratio(n_new),
        );
    }
    if plan.n_replay == 0 {
        log::warn!(
            "zero replay cells: this is an unprotected fine-tune and will forget the reference."
        );
    }
    // Replay cells carry the FULL gene axis, so they constrain α in every direction no matter
    // how narrow the new panel is. A rank-deficient ρ_S is therefore only fatal when nothing is
    // replayed — that is the one case where some direction of α is pinned by nobody and the
    // refit will move it arbitrarily, then write the result into β for every gene.
    if alpha_condition.is_none() {
        anyhow::ensure!(
            plan.n_replay > 0,
            "update refuses to write: ρ restricted to the {n_genes_mapped} mapped genes is \
             rank-deficient, so some directions of α are unconstrained by this batch — and with \
             zero replay nothing else constrains them either. The refit would move α arbitrarily \
             there and regenerate β for all {n_genes_model} genes from it. Raise --replay-ratio \
             above 0, or supply a batch covering more of the embedding space."
        );
        log::warn!(
            "ρ over the {n_genes_mapped} mapped genes is rank-deficient: this batch does not \
             constrain every direction of α. The {} replayed reference cells do, so the update is \
             identified — but what it learns in those directions comes from replay, not from the \
             new data.",
            plan.n_replay
        );
    }

    // Two evaluation sets with DIFFERENT status, and the report must say which is which:
    //   held_ref — reference cells deliberately kept out of the refit: genuinely held out.
    //   all_new  — every new cell, all of which ARE in `fit_ids`: in-sample training fit.
    // The new side is in-sample by design — an update should train on all the data it is
    // given — so its number is optimistic and is labelled as such rather than fixed.
    let held_ref: Vec<usize> = cal_ids[plan.n_replay..].to_vec();
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
    // `n_train_cells` becomes the running `N_absorbed` the net-gain ranking needs;
    // `theta_mean` is carried forward rather than recomputed (the masked path never
    // reads it, and recomputing needs a full encoder pass over the parent's cells).
    // The gene axis is unchanged by an α refit, so the parent's per-gene vectors carry
    // over verbatim.
    let mut new_meta = parent.metadata.clone();
    new_meta.n_train_cells = Some(parent.metadata.n_train_cells.unwrap_or(0) + n_new);
    // `n_train_cells` is a running total and loses the breakdown. The history keeps it, so a
    // child of a child can still say what each round did — and so absorbing the same batch
    // twice, or two children differing only in `--steps`, are visible on disk instead of
    // indistinguishable.
    new_meta.update_history.push(UpdateRecord {
        parent: args.model.clone(),
        data: args.data_files.clone(),
        calibration: args.calibration.clone(),
        n_new,
        n_replay: plan.n_replay,
        replay_ratio_requested: args.replay_ratio,
        replay_ratio_effective: plan.effective_ratio(n_new),
        steps: args.steps,
        lr: args.lr,
        seed: args.seed,
        n_genes_mapped,
        n_genes_model,
        alpha_condition,
    });
    write_masked_model(WriteArgs {
        out: &args.out,
        decoder: &rebuilt.decoder,
        parameters: &rebuilt.parameters,
        metadata: &new_meta,
        gene_names: &parent.gene_names,
        feature_mean: &parent.feature_mean,
        shortlist: &parent.shortlist,
    })?;

    // Without a manifest the child is invisible to every `RunManifest::load` consumer —
    // clustering, annotate, plot-topic, layouts, pseudotime, deconvolve — so it would load
    // in `predict`/`probe` and nowhere else. The child keeps its parent's `kind`: an
    // updated masked-topic model is still a masked-topic model.
    //
    // Suffixes are set to exactly what `update` writes. Notably `dictionary_empirical` is
    // absent: it is a per-pseudobulk empirical β and `update` runs no collapse, so there is
    // no cell→pseudobulk map to build one from. Consumers that prefer it fall back to the
    // factorized `dictionary.parquet` (`plot-topic` already does `dictionary_empirical
    // .or(dictionary)`), which is the same fallback `vae` and `joint-topic` rely on. That
    // does mean a parent and its child answer the gene-ranking question with different
    // estimators, so `--dictionary` should be passed explicitly when comparing them.
    let input: Vec<String> = args.data_files.iter().map(ToString::to_string).collect();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: crate::masked_topic::masked_run_kind(parent.head),
        prefix: &args.out,
        data_input: &input,
        data_batch: &[],
        data_input_null: &[],
        dictionary_suffix: Some("dictionary.parquet"),
        has_model: true,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_loading_suffix: None,
        softmax_dictionary_suffix: Some("dictionary.parquet"),
        cell_embedding_suffix: None,
        default_colour_by: "cluster",
        has_latent: false,
        has_cell_to_pb: false,
    })?;

    info!(
        "update: wrote {} (n_train_cells {} → {})",
        args.out,
        parent.metadata.n_train_cells.unwrap_or(0),
        new_meta.n_train_cells.unwrap_or(0)
    );
    info!("update: parent {} left untouched", args.model);
    Ok(())
}

/// How well the genes this batch observed pin `α`.
///
/// **Why this and not a gene-count fraction.** `α` is `[K,H]` and enters the likelihood only
/// through `α·ρᵀ`, so a batch constrains `α` exactly in the directions spanned by `ρ_S` — the
/// mapped-gene rows of ρ. Identification is therefore a question about **H**, not about **D**:
/// `|S| ≳ H` can suffice even when `|S| ≪ D`, because one `α_k` serves every gene and an
/// unobserved gene still gets `α_k·ρ_d`. A high gene-*fraction* floor is the right instrument for
/// a model with a free `[D,K]` dictionary, where an unobserved gene genuinely has nothing behind
/// it; importing it here would throw away the property the embedding exists to provide.
///
/// Returns `√(λ_max/λ_min)` of `ρ_Sᵀρ_S`. That Gram matrix is `[H,H]`, so this stays a ~128×128
/// symmetric eigenproblem however large D and `|S|` are. `INFINITY` means `ρ_S` is rank-deficient:
/// some direction of `α` is unconstrained by this batch.
fn alpha_conditioning(rho: &candle_core::Tensor, mapped: &[usize]) -> anyhow::Result<Option<f64>> {
    let h = rho.dim(1)?;
    if mapped.len() < h {
        // Fewer observed genes than embedding dimensions: rank-deficient by counting alone.
        return Ok(None);
    }
    let idx = candle_core::Tensor::from_vec(
        mapped.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        mapped.len(),
        rho.device(),
    )?;
    let rho_s = rho.index_select(&idx, 0)?;
    let gram = rho_s.t()?.matmul(&rho_s)?.flatten_all()?.to_vec1::<f32>()?;
    // The Gram matrix is symmetric, so nalgebra's column-major read of a row-major buffer is
    // the same matrix — no transpose needed.
    let g = nalgebra::DMatrix::<f64>::from_iterator(h, h, gram.iter().map(|&x| f64::from(x)));
    let eig = g.symmetric_eigenvalues();
    let (l_max, l_min) = eig
        .iter()
        .fold((f64::MIN, f64::MAX), |(a, b), &x| (a.max(x), b.min(x)));
    // Relative tolerance: a tiny or slightly negative eigenvalue is numerical rank deficiency.
    Ok(if l_max <= 0.0 || l_min <= l_max * 1e-12 {
        None
    } else {
        Some((l_max / l_min).sqrt())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_plan_honours_the_ratio_when_reference_is_ample() {
        // 3000 reference: 1000 reserved, 2000 replayable. 400 wanted fits easily.
        let p = ReplayPlan::new(3000, 400, 1.0);
        assert_eq!(p.n_wanted, 400);
        assert_eq!(p.n_replay, 400);
        assert!(!p.capped());
        assert!((p.effective_ratio(400) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn replay_plan_caps_and_reports_the_ratio_actually_achieved() {
        // The regression this guards: a 654-cell batch against 402 reference cells
        // silently got ~0.41 while reporting 1.00, i.e. far less protection than asked.
        let p = ReplayPlan::new(402, 654, 1.0);
        assert_eq!(p.n_replay_cap, 402 - 134);
        assert_eq!(p.n_wanted, 654);
        assert_eq!(p.n_replay, 268);
        assert!(
            p.capped(),
            "the request exceeded the cap and must be flagged"
        );
        let eff = p.effective_ratio(654);
        assert!((eff - 268.0 / 654.0).abs() < 1e-12);
        assert!(
            eff < 1.0,
            "effective ratio must reflect the cap, not the request"
        );
    }

    #[test]
    fn replay_plan_always_leaves_cells_to_measure_on() {
        // Whatever the ratio, the reserve survives — otherwise there is no held-out
        // reference slice and the realized-effect report is vacuous.
        for n_cal in [1usize, 2, 3, 10, 397, 5000] {
            for ratio in [0.0, 1.0, 10.0, 1e6] {
                let p = ReplayPlan::new(n_cal, 500, ratio);
                assert!(
                    p.n_replay < n_cal,
                    "n_cal={n_cal} ratio={ratio}: replay {} consumed the whole reference",
                    p.n_replay
                );
            }
        }
    }

    #[test]
    fn replay_plan_ratio_zero_disables_replay() {
        let p = ReplayPlan::new(900, 300, 0.0);
        assert_eq!(p.n_replay, 0);
        assert!(!p.capped(), "asking for nothing is not a capped request");
    }

    /// `[n_genes, h]` row-major ρ for tests.
    fn rho(rows: &[&[f32]]) -> candle_core::Tensor {
        let h = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        candle_core::Tensor::from_vec(flat, (rows.len(), h), &candle_core::Device::Cpu)
            .expect("tensor")
    }

    /// A rank-deficient batch must still produce a **readable** child model.
    ///
    /// The regression: `alpha_condition` used to be a bare `f64` holding `INFINITY` for that
    /// case. `serde_json` renders a non-finite float as `null`, and deserializing `null` into
    /// `f64` fails — so the child's `model.json` was written successfully and could then never
    /// be loaded again, breaking `predict`, `probe` and every later `update` on it. Rank
    /// deficiency is an expected outcome (it only aborts when replay is also zero), so this
    /// path is reachable in normal use.
    #[test]
    fn a_rank_deficient_update_still_round_trips_through_json() {
        let rec = UpdateRecord {
            parent: "parent".into(),
            data: vec!["new.zarr".into()],
            calibration: "ref.zarr".into(),
            n_new: 10,
            n_replay: 10,
            replay_ratio_requested: 1.0,
            replay_ratio_effective: 1.0,
            steps: 100,
            lr: 0.05,
            seed: 42,
            n_genes_mapped: 3,
            n_genes_model: 100,
            alpha_condition: None,
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        let back: UpdateRecord = serde_json::from_str(&json).expect(
            "a rank-deficient record must deserialize; a bare f64::INFINITY here serializes to \
             null and then fails to read, bricking the child model",
        );
        assert!(back.alpha_condition.is_none());

        // And the finite case must survive unchanged.
        let ok = UpdateRecord {
            alpha_condition: Some(15.5),
            ..rec
        };
        let back: UpdateRecord =
            serde_json::from_str(&serde_json::to_string(&ok).expect("serialize")).expect("read");
        assert!((back.alpha_condition.expect("finite") - 15.5).abs() < 1e-9);
    }

    /// The gene remap is many-to-one, so the mapped-gene list must be deduplicated: two query
    /// names canonicalizing onto one training gene would otherwise double-weight that ρ row and
    /// let `n_genes_mapped` exceed the model's axis.
    #[test]
    fn duplicate_remap_targets_collapse_to_distinct_genes() {
        let new_to_train = [Some(7usize), Some(7), Some(2), None, Some(2), Some(9)];
        let mut v: Vec<usize> = new_to_train.iter().flatten().copied().collect();
        v.sort_unstable();
        v.dedup();
        assert_eq!(v, vec![2, 7, 9], "distinct training genes only");
    }

    #[test]
    fn orthonormal_genes_are_perfectly_conditioned() {
        let r = rho(&[&[1., 0., 0.], &[0., 1., 0.], &[0., 0., 1.]]);
        let c = alpha_conditioning(&r, &[0, 1, 2])
            .expect("cond")
            .expect("full rank");
        assert!(
            (c - 1.0).abs() < 1e-6,
            "an orthonormal ρ_S pins every direction of α equally well; got {c}"
        );
    }

    #[test]
    fn too_few_genes_to_span_the_embedding_is_rank_deficient() {
        // The point of the check: with H=3, two genes cannot constrain α however many
        // CELLS the batch has. Identification is about H, not about D or N.
        let r = rho(&[&[1., 0., 0.], &[0., 1., 0.], &[0., 0., 1.]]);
        assert!(alpha_conditioning(&r, &[0, 1]).expect("cond").is_none());
    }

    #[test]
    fn genes_confined_to_a_subspace_are_rank_deficient() {
        // Enough genes by count, but they all lie in the first two embedding dimensions —
        // so α's third direction is unconstrained and a count-based floor would miss it.
        let r = rho(&[&[1., 0., 0.], &[0., 1., 0.], &[1., 1., 0.], &[2., -1., 0.]]);
        assert!(
            alpha_conditioning(&r, &[0, 1, 2, 3])
                .expect("cond")
                .is_none(),
            "four genes spanning a 2-D subspace must not read as identified"
        );
    }

    #[test]
    fn a_narrow_but_spanning_panel_is_accepted() {
        // The case the old fraction-based gate got wrong: 3 of 100 genes, but they span
        // the embedding, so α is identified and the update is legitimate.
        let mut rows: Vec<Vec<f32>> = vec![vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.]];
        rows.extend((0..97).map(|i| vec![i as f32, 0., 0.]));
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let c = alpha_conditioning(&rho(&refs), &[0, 1, 2]).expect("cond");
        assert!(c.is_some(), "3/100 genes that span H must be accepted");
    }
}
