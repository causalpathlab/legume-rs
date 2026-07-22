//! Does the `u + δ → s` parameterization actually recover the mechanism?
//!
//! Everything else in the test suite checks shapes, masking, and algebraic
//! identities. This file checks the claim the whole design rests on: that after
//! training, `⟨α_t, δ_g⟩` is the steady-state `log(β_g/γ_g)` of the
//! RNA-velocity ODE. Data is simulated FROM that ODE, so the ground truth is
//! known and recovery is measurable rather than asserted.

use candle_util::vae::masked_gem::*;
use candle_util::data::indexed::{GemIndexedArgs, GemIndexedData, GeneTrackMap};
use candle_util::decoder::gem_etm::{GemEtmDecoder, Track};
use candle_util::encoder::gem_encoder::{GemIndexedEncoder, GemIndexedEncoderArgs};
use candle_util::candle_core::{DType, Device};
use candle_util::candle_nn::{VarBuilder, VarMap};
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::sync::atomic::AtomicBool;

const G: usize = 120;
const N: usize = 150;
const K: usize = 2;
const H: usize = 8;

/// Row layout: `[0..G)` nascent, `[G..2G)` mature — one gene id per column pair.
fn track_map() -> GeneTrackMap {
    GeneTrackMap {
        row_to_gene: (0..G).chain(0..G).map(|g| g as u32).collect(),
        row_is_nascent: (0..G).map(|_| true).chain((0..G).map(|_| false)).collect(),
        n_genes: G,
    }
}

/// Simulated cohort plus the ground-truth quantity we hope to recover.
struct Sim {
    /// `[N, 2G]` counts, nascent block then mature block.
    counts: DMatrix<f32>,
    /// `log(β_g/γ_g)` — the per-gene steady-state ratio the model must find.
    log_ratio: Vec<f32>,
}

/// Generate from the ODE at steady state.
///
/// Each sample mixes two transcriptional programs. The nascent composition is
/// the mixture; the mature composition is that same mixture reweighted per gene
/// by `β_g/γ_g` and renormalized — exactly `s* = (β/γ)·u`. Counts are Poisson
/// around those compositions, with the mature track given a deeper library, as
/// in real data.
fn simulate(seed: u64) -> Sim {
    let mut rng = StdRng::seed_from_u64(seed);

    // Two programs over genes: each concentrates on its own half, with a floor
    // so no gene is structurally absent.
    let mut phi = vec![vec![0f32; G]; K];
    for (t, row) in phi.iter_mut().enumerate() {
        for (g, p) in row.iter_mut().enumerate() {
            let in_program = (g < G / 2) == (t == 0);
            *p = if in_program { 8.0 } else { 1.0 } * (0.5 + rng.random::<f32>());
        }
        let z: f32 = row.iter().sum();
        for p in row.iter_mut() {
            *p /= z;
        }
    }

    // Per-gene splicing/degradation ratio, log-normal around 1.
    let log_ratio: Vec<f32> = (0..G)
        .map(|_| (rng.random::<f32>() - 0.5) * 2.4)
        .collect();
    let ratio: Vec<f32> = log_ratio.iter().map(|x| x.exp()).collect();

    let mut counts = DMatrix::<f32>::zeros(N, 2 * G);
    for n in 0..N {
        // Mixture weight for this sample, spread across the simplex.
        let w = rng.random::<f32>();
        let theta = [w, 1.0 - w];

        let u_comp: Vec<f32> = (0..G)
            .map(|g| theta[0] * phi[0][g] + theta[1] * phi[1][g])
            .collect();
        // s* = (beta/gamma) * u, renormalized to a composition.
        let s_un: Vec<f32> = (0..G).map(|g| u_comp[g] * ratio[g]).collect();
        let s_z: f32 = s_un.iter().sum();
        let s_comp: Vec<f32> = s_un.iter().map(|x| x / s_z).collect();

        // Mature is the deeper track, as in real data.
        let (lib_u, lib_s) = (4_000.0f32, 20_000.0f32);
        for g in 0..G {
            counts[(n, g)] = poisson(&mut rng, u_comp[g] * lib_u);
            counts[(n, G + g)] = poisson(&mut rng, s_comp[g] * lib_s);
        }
    }

    Sim { counts, log_ratio }
}

/// Knuth's product method, adequate for the small means here.
fn poisson(rng: &mut StdRng, mean: f32) -> f32 {
    if mean <= 0.0 {
        return 0.0;
    }
    if mean > 30.0 {
        // Normal approximation keeps the loop bounded for the deep track.
        let (u1, u2): (f32, f32) = (rng.random::<f32>().max(1e-9), rng.random::<f32>());
        let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        return (mean + z * mean.sqrt()).max(0.0).round();
    }
    let l = (-mean).exp();
    let (mut k, mut p) = (0.0f32, 1.0f32);
    loop {
        p *= rng.random::<f32>();
        if p <= l {
            return k;
        }
        k += 1.0;
    }
}

struct Fitted {
    /// Per-gene model splice-ratio score — the factor-averaged `log β^s − log β^u`,
    /// which the design says is `log(β_g/γ_g)` up to an additive constant.
    splice_ratio: Vec<f32>,
    /// Final `u→s` (mechanism-mode) likelihood per scored position.
    mechanism_llik: f32,
    /// Final nascent-track likelihood per scored position.
    nascent_llik: f32,
    delta_norm: f32,
}

fn fit(sim: &Sim, epochs: usize, delta_l2: f32, seed_tag: &str) -> Fitted {
    let dev = Device::Cpu;
    let map = track_map();
    let weights = vec![1.0f32; G];

    let mut data = vec![GemIndexedData::from_dense(GemIndexedArgs {
        observed: &sim.counts,
        residual: None,
        adjusted: None,
        map: &map,
        context_size: G,
        gene_weights: &weights,
        nascent_mean: None,
        mature_mean: None,
    })
    .expect("loader")];

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let layers = vec![32usize, 32];
    let encoder = GemIndexedEncoder::new(
        GemIndexedEncoderArgs {
            n_genes: G,
            n_latent: K,
            embedding_dim: H,
            layers: &layers,
            latent_noise: false,
        },
        &varmap,
        vb.pp("enc"),
    )
    .expect("encoder");

    let decoders = vec![GemEtmDecoder::new(
        K,
        encoder.feature_embeddings().clone(),
        encoder.delta_embeddings().clone(),
        vb.pp("dec_0"),
    )
    .expect("decoder")];

    let opts = GemTrainOpts {
        // The simulation is compositional (s* = (beta/gamma)u renormalized), so
        // the depth-invariant likelihood is the matched one.
        likelihood: GemLikelihood::Multinomial,
        delta_l2,
        feature_embedding_l2: 0.0,
        ..Default::default()
    };
    let stop = AtomicBool::new(false);
    let scores = train_masked_gem(
        &mut data,
        &encoder,
        &decoders,
        &GemTrainConfig {
            parameters: &varmap,
            dev: &dev,
            epochs,
            minibatch_size: 60,
            learning_rate: 5e-2,
            weight_decay: 0.0,
            grad_clip: 5.0,
            stop: &stop,
        },
        &opts,
    )
    .unwrap_or_else(|e| panic!("{seed_tag}: training failed: {e}"));

    // Per-gene splice-ratio score, averaged over factors. `get_dictionary`
    // returns [G, K] log-probabilities.
    let log_s: Vec<Vec<f32>> = decoders[0]
        .get_dictionary(Track::Mature)
        .unwrap()
        .to_vec2()
        .unwrap();
    let log_u: Vec<Vec<f32>> = decoders[0]
        .get_dictionary(Track::Nascent)
        .unwrap()
        .to_vec2()
        .unwrap();
    let splice_ratio: Vec<f32> = (0..G)
        .map(|g| (0..K).map(|t| log_s[g][t] - log_u[g][t]).sum::<f32>() / K as f32)
        .collect();

    Fitted {
        splice_ratio,
        mechanism_llik: GemScores::last_finite(&scores.mechanism_llik),
        nascent_llik: GemScores::last_finite(&scores.nascent_llik),
        delta_norm: GemScores::last_finite(&scores.delta_norm),
    }
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let mut num = 0.0;
    let (mut da, mut db) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x - ma, y - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

/// **The load-bearing test.** Train on data generated from the velocity ODE and
/// check that the recovered splice-ratio program tracks the planted
/// `log(β_g/γ_g)`.
///
/// If this fails, `δ` is a free nuisance parameter rather than the splice-ratio
/// program, and every velocity the model reports is uninterpretable — no other
/// test in the suite would catch that.
#[test]
fn recovers_the_planted_splicing_to_degradation_ratio() {
    let sim = simulate(11);
    let fitted = fit(&sim, 180, 0.0, "recovery");

    let r = pearson(&fitted.splice_ratio, &sim.log_ratio);
    assert!(
        r > 0.6,
        "recovered splice-ratio program correlates only r={r:.3} with the planted \
         log(beta/gamma) (|delta|={:.4}). The u + delta -> s parameterization is not \
         recovering the mechanism.",
        fitted.delta_norm
    );
}

/// A ridge strong enough to pin `δ ≈ 0` must cost real fit — and this pins down
/// exactly **where** the cost lands, which is not where you would first guess.
///
/// With `δ = 0` the two dictionaries are forced equal, `β^s ≡ β^u`, so the model
/// cannot fit both compositions at once. It does not respond by fitting mature
/// worse: mature is the deeper track, so the count-weighted likelihood prefers
/// to satisfy it and give ground on nascent instead. Two consequences, both
/// asserted here:
///
/// 1. The **nascent** likelihood degrades, and the recovered ratio is destroyed.
/// 2. The **mature-side** metrics — including the `u→s` mechanism likelihood —
///    barely move, so they are NOT delta-collapse detectors. `|delta|` is.
///
/// `faba`'s `report_training_health` keys its delta warning off `|delta|` and
/// its trend precisely because of (2); if (2) ever stops holding, that warning
/// can be strengthened.
#[test]
fn crushing_delta_costs_nascent_fit_but_not_the_mature_side() {
    let sim = simulate(11);
    let free = fit(&sim, 170, 0.0, "free");
    let crushed = fit(&sim, 170, 1e4, "crushed");

    assert!(
        crushed.delta_norm < free.delta_norm * 0.5,
        "the large ridge did not actually shrink delta (free={:.4}, crushed={:.4})",
        free.delta_norm,
        crushed.delta_norm
    );

    // (1) the forced tie has to show up somewhere, and nascent is where.
    assert!(
        free.nascent_llik > crushed.nascent_llik,
        "delta carries no information about the nascent/mature split: \
         free nascent llik {:.4} did not beat crushed {:.4}",
        free.nascent_llik,
        crushed.nascent_llik
    );
    let (r_free, r_crushed) = (
        pearson(&free.splice_ratio, &sim.log_ratio),
        pearson(&crushed.splice_ratio, &sim.log_ratio),
    );
    assert!(
        r_free > r_crushed + 0.3,
        "crushing delta should destroy ratio recovery, but r went {r_free:.3} -> {r_crushed:.3}"
    );

    // (2) the mature side stays put — the documented blind spot.
    let gap = (free.mechanism_llik - crushed.mechanism_llik).abs();
    let scale = free.mechanism_llik.abs().max(1.0);
    assert!(
        gap < 0.25 * scale,
        "mechanism likelihood turned out to be sensitive to delta collapse \
         (free={:.3}, crushed={:.3}). If that is now reliably true, faba's \
         delta-collapse warning can key off it instead of |delta|.",
        free.mechanism_llik,
        crushed.mechanism_llik
    );
}
