//! Two properties of the GEM decoder's per-gene background and its topic axis.
//!
//! 1. The background is FROZEN at the data's gene marginal rather than learned.
//!    A learnable per-gene bias shifts all `K` topics equally on a gene, which is
//!    the shared abundance direction; leaving it free lets the optimizer
//!    reinstate that direction after any attempt to remove it, silently.
//!
//! 2. The topic MEAN of the nascent→mature difference must survive. Unlike the
//!    plain embedded-topic decoder — where centering `α` over the topic axis is
//!    what creates module competition — this decoder must NOT be centered: its
//!    estimand lives precisely in the direction centering annihilates. See the
//!    second test for the measurement.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_util::decoder::{GemEtmDecoder, Track};
use std::collections::HashMap;

const G: usize = 6;
const H: usize = 3;
const K: usize = 4;

fn embeddings(dev: &Device) -> (Tensor, Tensor) {
    let rho: Vec<f32> = (0..G * H).map(|i| (i % 7) as f32 * 0.3 - 0.6).collect();
    // Period 5 against a stride of H=3 keeps the δ rows genuinely gene-varying.
    let delta: Vec<f32> = (0..G * H).map(|i| (i % 5) as f32 * 0.2 - 0.2).collect();
    (
        Tensor::from_vec(rho, (G, H), dev).unwrap(),
        Tensor::from_vec(delta, (G, H), dev).unwrap(),
    )
}

/// A deliberately non-uniform gene marginal `π`, as `log π [1, G]`.
fn log_pi(dev: &Device) -> Tensor {
    let counts: Vec<f32> = (0..G).map(|g| (g + 1) as f32).collect();
    let total: f32 = counts.iter().sum();
    Tensor::from_vec(
        counts
            .iter()
            .map(|c| (c / total).ln())
            .collect::<Vec<f32>>(),
        (1, G),
        dev,
    )
    .unwrap()
}

/// Deterministic α with a non-zero mean archetype ᾱ, plus the `[2, G]`
/// dispersion rows, so the algebra below is exact rather than random.
fn decoder(dev: &Device) -> GemEtmDecoder {
    let (rho, delta) = embeddings(dev);
    let alpha: Vec<f32> = (0..K * H).map(|i| (i % 5) as f32 * 0.4 - 0.4).collect();
    let mut ts = HashMap::new();
    ts.insert(
        "dec.topic.embeddings".to_string(),
        Tensor::from_vec(alpha, (K, H), dev).unwrap(),
    );
    ts.insert(
        "dec.log_phi".to_string(),
        Tensor::from_vec(vec![0.693f32; 2 * G], (2, G), dev).unwrap(),
    );
    let vb = VarBuilder::from_tensors(ts, DType::F32, dev);
    GemEtmDecoder::new(K, rho, delta, log_pi(dev), vb.pp("dec")).unwrap()
}

/// The background must not be a trainable variable.
#[test]
fn the_gene_background_is_not_a_trainable_variable() {
    let dev = Device::Cpu;
    let (rho, delta) = embeddings(&dev);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let _dec = GemEtmDecoder::new(K, rho, delta, log_pi(&dev), vb.pp("dec")).unwrap();

    let trainable: Vec<String> = varmap.data().lock().unwrap().keys().cloned().collect();
    assert!(
        !trainable.iter().any(|n| n.contains("logit_bias")),
        "the per-gene background must be frozen, but a trainable `logit_bias` \
         was registered; trainable vars were {trainable:?}"
    );
}

/// `Σ_k (log β^mature_kg − log β^nascent_kg) = K⟨ᾱ, δ_g⟩ + const` — the topic
/// MEAN of the track difference, which is the gene-level splice program the
/// velocity readout averages over topics.
///
/// This is a guard against centering `α` over the topic axis here. Centering
/// sends `Σ_k (α_k − ᾱ)` to exactly 0 and so annihilates this quantity
/// identically — not approximately. It was measured: with `α` centered, the
/// end-to-end recovery of the planted `log(β_g/γ_g)` collapsed from the required
/// `r > 0.6` to `r = −0.096`, i.e. pure noise, while `‖δ‖` stayed non-zero so
/// nothing else looked wrong. Centering is the right mechanism for the plain
/// embedded-topic decoder, which has no `δ` track and no such estimand; it is
/// the wrong mechanism here.
#[test]
fn the_topic_mean_of_the_track_difference_is_not_annihilated() {
    let dev = Device::Cpu;
    let dec = decoder(&dev);

    let m: Vec<Vec<f32>> = dec
        .get_dictionary(Track::Mature)
        .unwrap()
        .to_vec2()
        .unwrap();
    let u: Vec<Vec<f32>> = dec
        .get_dictionary(Track::Nascent)
        .unwrap()
        .to_vec2()
        .unwrap();

    let topic_mean: Vec<f32> = (0..G)
        .map(|g| (0..K).map(|t| m[g][t] - u[g][t]).sum::<f32>() / K as f32)
        .collect();
    let lo = topic_mean.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = topic_mean.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        hi - lo > 1e-3,
        "the topic-mean track difference carries no gene-level signal (spread \
         {:.8} over {topic_mean:?}) — the splice program has been annihilated, \
         which is what centering α over the topic axis does here",
        hi - lo
    );
}
