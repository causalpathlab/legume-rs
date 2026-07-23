//! What the run says about itself: the per-epoch trace, the splice-ratio
//! consistency check, the two training-health warnings, and the manifest that
//! tells a reader what the artifacts mean.
//!
//! Everything here is a claim about whether the fit can be trusted, kept apart
//! from [`crate::gem_encoder::write`], which is the plumbing that puts tables on
//! disk.

use candle_util::decoder::gem_etm::{GemEtmDecoder, Track};
use candle_util::vae::masked_gem::GemScores;
use log::{info, warn};
use matrix_util::dmatrix_io::DMatrix as IoDMatrix;
use matrix_util::traits::IoOps;

use crate::gem_encoder::args::GemEncoderArgs;

/// Say plainly whether the run produced something trustworthy.
///
/// Two failure modes are invisible in a pooled likelihood and silently produce
/// junk velocity, so they get called out rather than left in the trace for
/// someone to notice later.
pub fn report_training_health(scores: &GemScores, delta_l2: f32) {
    if scores.delta_norm.is_empty() {
        return;
    }
    // Per-mode traces carry NaN for an epoch in which that mode was not drawn,
    // so read the last FINITE value — `last()` would hand every comparison a
    // NaN and quietly disable the checks below.
    let last = |v: &Vec<f32>| GemScores::last_finite(v);
    let delta_norm = last(&scores.delta_norm);
    info!(
        "final: llik/mature={:.4} llik/nascent={:.4} llik/mechanism={:.4} |δ|={:.4}",
        last(&scores.mature_llik),
        last(&scores.nascent_llik),
        last(&scores.mechanism_llik),
        delta_norm,
    );

    if delta_norm < 1e-3 {
        // With the ridge off, nothing in the objective is pulling δ down, so a
        // collapse is about the data (no nascent coverage) rather than a knob.
        let remedy = if delta_l2 > 0.0 {
            "Lower --delta-l2 (0 is the default), or check that the unspliced track actually has coverage."
        } else {
            "The ridge is already off, so check that the unspliced track actually has coverage."
        };
        warn!(
            "|δ| = {delta_norm:.2e}. The splice-ratio program has collapsed to ~zero.\n\
             The two dictionaries are effectively identical, \n\
	     which means the model kept no nascent/mature difference at all: \n\
	     velocity will be noise. \n\
	     {remedy}"
        );
    } else if let Some(slope) = trailing_slope(&scores.delta_norm) {
        // A FALLING |delta| means the ridge is pulling it down. It settles at an
        // equilibrium against the likelihood and reaches it from ABOVE, dipping
        // below its final value first, so a fall usually means the run stopped
        // mid-descent rather than that the ridge is too strong.
        //
        // With the ridge off there is no such equilibrium: |delta| grows, slowly
        // and without flattening (measured: still rising at epoch 300). So a
        // RISING trace is expected there and says nothing, which is why only the
        // falling direction is checked.
        let rel = slope / delta_norm.max(1e-9);
        if rel < -0.005 {
            let remedy = if delta_l2 > 0.0 {
                "Only lower --delta-l2 if it is still falling once the likelihood has plateaued."
            } else {
                "The ridge is off, so this is not a --delta-l2 problem."
            };
            warn!(
                "|δ| has not settled: still moving {:.2}% per epoch at the end of training (now {delta_norm:.4}). \n\
		 It approaches its equilibrium from above, \n\
		 so this is most likely too few epochs rather than too much ridge: \n\
		 re-run with more --epochs and compare before trusting the velocity. \n\
		 {remedy}",
                rel * 100.0
            );
        }
    }

    // Mechanism fit relative to the pooled mature fit. If predicting mature from
    // nascent alone is far worse than predicting it in general, the cross-track
    // path is not carrying information and the model has degenerated into two
    // topic models sharing rho.
    //
    // NOTE this does NOT detect delta collapse, and is not meant to: with
    // delta = 0 the model keeps fitting mature well and gives ground on the
    // nascent track instead (mature is the deeper track, so the count-weighted
    // likelihood prefers it). See `gem_mechanism_tests`. |delta| above is the
    // detector for that; this one catches a different failure.
    let (mech, pooled) = (last(&scores.mechanism_llik), last(&scores.mature_llik));
    if mech.is_finite() && pooled.is_finite() && mech < pooled - pooled.abs().max(1.0) {
        warn!(
            "the nascent→mature likelihood ({mech:.4}) trails the pooled mature likelihood \n\
             ({pooled:.4}) by more than its own scale. The model is imputing mature counts \n\
             from mature context rather than learning u → s; treat the velocity with suspicion \n\
             and consider lowering --mask-fraction so more context stays visible."
        );
    }
}

/// Least-squares slope per epoch over the trailing quarter of a trace (minimum
/// 8 points). `None` when the trace is too short to say anything.
fn trailing_slope(trace: &[f32]) -> Option<f32> {
    let n = trace.len();
    if n < 8 {
        return None;
    }
    let start = n - (n / 4).max(8).min(n);
    let tail = &trace[start..];
    let m = tail.len() as f32;
    let mean_x = (m - 1.0) / 2.0;
    let mean_y = tail.iter().sum::<f32>() / m;
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for (i, &y) in tail.iter().enumerate() {
        let dx = i as f32 - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    (den > 0.0).then(|| num / den)
}

/// Per-epoch trace. The two tracks and the mechanism-only fit are separate
/// columns because a pooled likelihood hides the failure mode this model has.
pub fn write_scores(scores: &GemScores, out: &str) -> anyhow::Result<()> {
    let n = scores.mature_llik.len();
    if n == 0 {
        return Ok(());
    }
    let mut m = IoDMatrix::<f32>::zeros(n, 4);
    for i in 0..n {
        m[(i, 0)] = scores.mature_llik[i];
        m[(i, 1)] = scores.nascent_llik[i];
        m[(i, 2)] = scores.mechanism_llik[i];
        m[(i, 3)] = scores.delta_norm[i];
    }
    let cols: Vec<Box<str>> = [
        "llik_mature",
        "llik_nascent",
        "llik_mechanism",
        "delta_norm",
    ]
    .iter()
    .map(|s| (*s).into())
    .collect();
    let rows: Vec<Box<str>> = (0..n)
        .map(|i| format!("epoch{i}").into_boxed_str())
        .collect();
    let path = format!("{out}.log_likelihood.parquet");
    m.to_parquet_with_names(&path, (Some(&rows), Some("epoch")), Some(&cols))?;
    info!("wrote {path}");
    Ok(())
}

/// Write `{out}.splice_ratio_qc.parquet` and report how well the learned
/// splice-ratio offset matches the one actually present in the counts.
///
/// # What `δ` is, precisely
///
/// `⟨α_t, δ_g⟩ = log β^s[t,g] − log β^u[t,g]`, and at steady state
/// `s* = (β_g/γ_g)·u`, so this quantity estimates
///
/// ```text
/// log( β_g / γ_g )     splicing rate  /  DEGRADATION rate
/// ```
///
/// Note what that is and is not. It is **not** a splicing rate: a gene scores
/// high here either because it is spliced quickly *or* because its mature mRNA
/// is stable and decays slowly, and this model cannot separate those two. It is
/// the steady-state mature:nascent ratio, no more. Read a large `δ_g` as "this
/// gene sits mature-heavy at equilibrium", not as "this gene is spliced fast".
///
/// # The check
///
/// The empirical counterpart is the observed per-gene `log(mature/nascent)`,
/// computed from **raw backend row means** — not from the pseudobulk posterior,
/// so the comparison is against the counts rather than against another smoothed
/// summary of them. If the two correlate, `δ` is tracking real splice kinetics.
/// If they do not, `δ` has become a free nuisance parameter: the fit will still
/// look healthy and the velocity will still be written, but it should not be
/// read as RNA velocity.
///
/// The two are not independent (both derive from the same counts), so this is a
/// consistency check rather than held-out validation. It is still the check that
/// fails loudly when `δ` stops meaning what the parameterization says.
pub fn write_splice_ratio_qc(
    decoder: &GemEtmDecoder,
    gene_names: &[Box<str>],
    map: &candle_util::data::indexed::GeneTrackMap,
    row_mean: &[f32],
    out: &str,
) -> anyhow::Result<()> {
    let log_s: Vec<Vec<f32>> = decoder.get_dictionary(Track::Mature)?.to_vec2()?;
    let log_u: Vec<Vec<f32>> = decoder.get_dictionary(Track::Nascent)?.to_vec2()?;
    let g = gene_names.len();
    let k = log_s.first().map_or(0, Vec::len);
    anyhow::ensure!(
        log_s.len() == g && log_u.len() == g,
        "dictionary has {} rows but there are {g} genes",
        log_s.len()
    );

    // Observed per-gene per-track rate, straight from the raw row means.
    let mut obs_u = vec![0f32; g];
    let mut obs_s = vec![0f32; g];
    for (r, (&gid, &is_nascent)) in map
        .row_to_gene
        .iter()
        .zip(map.row_is_nascent.iter())
        .enumerate()
    {
        let slot = if is_nascent { &mut obs_u } else { &mut obs_s };
        slot[gid as usize] += row_mean[r];
    }

    // Model estimate, averaged over factors.
    let model: Vec<f32> = (0..g)
        .map(|i| (0..k).map(|t| log_s[i][t] - log_u[i][t]).sum::<f32>() / k.max(1) as f32)
        .collect();
    let observed: Vec<f32> = (0..g)
        .map(|i| ((obs_s[i] + 1e-8) / (obs_u[i] + 1e-8)).ln())
        .collect();

    // Only genes with real coverage on BOTH tracks can speak to the ratio. A
    // gene with no nascent counts has an observed ratio set by the epsilon
    // rather than by biology, and including those would manufacture correlation.
    let usable: Vec<usize> = (0..g)
        .filter(|&i| obs_u[i] > 0.0 && obs_s[i] > 0.0)
        .collect();
    let r = pearson(
        &usable.iter().map(|&i| model[i]).collect::<Vec<_>>(),
        &usable.iter().map(|&i| observed[i]).collect::<Vec<_>>(),
    );

    let mut m = IoDMatrix::<f32>::zeros(g, 4);
    for i in 0..g {
        m[(i, 0)] = model[i];
        m[(i, 1)] = observed[i];
        m[(i, 2)] = obs_u[i];
        m[(i, 3)] = obs_s[i];
    }
    let cols: Vec<Box<str>> = [
        "log_splice_ratio_model",
        "log_splice_ratio_observed",
        "nascent_mean_obs",
        "mature_mean_obs",
    ]
    .iter()
    .map(|s| (*s).into())
    .collect();
    let path = format!("{out}.splice_ratio_qc.parquet");
    m.to_parquet_with_names(&path, (Some(gene_names), Some("gene")), Some(&cols))?;

    info!(
        "splice-ratio check: r = {r:.3} between the learned log(beta/gamma) and \n\
	 the observed log(mature/nascent), over {} of {g} genes with coverage \n\
	 on both tracks → {path}",
        usable.len()
    );
    if r.is_finite() && r < 0.2 {
        warn!(
            "the learned splice-ratio offset barely tracks the observed mature:nascent \
             ratio (r = {r:.3}). delta is behaving as a free nuisance parameter rather than \
             as log(splicing/degradation), so {out}.velocity.parquet should not be read as \
             RNA velocity on this dataset."
        );
    }
    Ok(())
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    if n < 2.0 {
        return f32::NAN;
    }
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut num, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x - ma, y - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da <= 0.0 || db <= 0.0 {
        return f32::NAN;
    }
    num / (da.sqrt() * db.sqrt())
}

/// Persist weights plus the architecture needed to rebuild the model, the `δ`
/// base convention, and the velocity offset that centring removed.
///
/// `delta_base` matters, and the trap here is subtler than it used to be. This
/// model now shares `faba gem`'s base — both store the **spliced** embedding —
/// but the two `δ`s still point in OPPOSITE directions, because gem derives its
/// other track by adding (`unspliced = β + δ_gem`) while this one subtracts
/// (`nascent = ρ − δ`):
///
/// ```text
/// gem:  δ_gem = unspliced − spliced
/// here: δ     = spliced   − unspliced      = −δ_gem
/// ```
///
/// The sign here is the ODE-natural one, so `⟨α_t, δ_g⟩ = log(β_g/γ_g)` reads
/// directly as mature-heavy-positive. Anything reading both files must negate
/// one; matching bases make it *look* safe to compare them raw, and it is not.
///
/// `velocity_common_mode` is the `[K]` per-axis offset subtracted from
/// **`{out}.velocity_factor.parquet`** (see
/// [`crate::gem_encoder::velocity::center_velocity`]); add it back to recover
/// the uncentred factor-space field.
///
/// It is **not** the offset for `{out}.velocity.parquet`, which has `H`
/// columns, not `K` — that file is the factor-space velocity projected through
/// `α`, and because centring and the projection are both linear the offset
/// there is `velocity_common_mode · α`, the decoder's topic embedding. Adding
/// a `K`-vector to an `H`-column table is either a shape error or a silent
/// wrong-axis broadcast, so the two are named apart here deliberately.
///
/// `"latent": "log-theta"` states the contract of `{out}.latent.parquet`:
/// `exp()` of a row is a probability vector summing to 1. It is written
/// explicitly because runs produced before 2026-07-21 stored **raw logits**
/// under the same `model_type`, and the two are indistinguishable by shape —
/// so a reader that guesses gets a plausible wrong θ rather than an error.
/// Files without this field are the old contract.
pub fn save_model_metadata(
    args: &GemEncoderArgs,
    n_genes: usize,
    velocity_common_mode: &[f32],
    parameters: &candle_util::candle_nn::VarMap,
) -> anyhow::Result<()> {
    parameters.save(format!("{}.safetensors", args.out))?;

    let mut extra = serde_json::Map::new();
    extra.insert("latent".into(), "log-theta".into());
    extra.insert("delta_base".into(), "unspliced".into());
    extra.insert("n_genes".into(), n_genes.into());
    extra.insert("n_latent".into(), args.n_latent.into());
    extra.insert("embedding_dim".into(), args.embedding_dim.into());
    extra.insert(
        "encoder_layers".into(),
        args.encoder_layers.iter().copied().collect(),
    );
    extra.insert("context_size".into(), args.context_size.into());
    extra.insert(
        "velocity_common_mode".into(),
        velocity_common_mode.iter().copied().collect(),
    );

    info!("wrote {}.safetensors", args.out);
    // `model_type` is stamped by the manifest writer from `RunKind::Topic`, so
    // this file cannot spell it differently. `write` logs the path it wrote.
    crate::manifest::write(&args.out, crate::manifest::RunKind::Topic, extra)
}
