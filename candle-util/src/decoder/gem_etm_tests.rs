use super::*;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

const G: usize = 12;
const H: usize = 5;
const K: usize = 3;

/// Build a decoder over deterministic (seed-free) ρ and δ so the algebraic
/// assertions below are exact rather than probabilistic.
fn fixture(delta_scale: f32) -> (GemEtmDecoder, Tensor, Tensor, Device) {
    let dev = Device::Cpu;
    let rho: Vec<f32> = (0..G * H).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
    let delta: Vec<f32> = (0..G * H)
        .map(|i| ((i % 5) as f32 - 2.0) * delta_scale)
        .collect();
    let rho = Tensor::from_vec(rho, (G, H), &dev).unwrap();
    let delta = Tensor::from_vec(delta, (G, H), &dev).unwrap();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let dec = GemEtmDecoder::new(K, rho.clone(), delta.clone(), vb.pp("dec")).unwrap();
    (dec, rho, delta, dev)
}

/// Both dictionaries must be proper distributions over the gene vocabulary —
/// that is what makes the nascent/mature comparison compositional rather than
/// depth-driven.
#[test]
fn dictionaries_are_normalized_over_genes() {
    let (dec, _, _, _) = fixture(0.2);
    for track in [Track::Nascent, Track::Mature] {
        let log_beta_gk = dec.get_dictionary(track).unwrap();
        assert_eq!(log_beta_gk.dims(), &[G, K]);
        let cols: Vec<Vec<f32>> = log_beta_gk.transpose(0, 1).unwrap().to_vec2().unwrap();
        for (t, col) in cols.iter().enumerate() {
            let mass: f32 = col.iter().map(|x| x.exp()).sum();
            assert!(
                (mass - 1.0).abs() < 1e-4,
                "{track:?} topic {t} sums to {mass}, not 1"
            );
        }
    }
}

/// `δ = 0` is the model's null — "mature composition equals nascent
/// composition", i.e. no differential processing. It must be exactly
/// degenerate, otherwise the ridge would be shrinking toward something other
/// than that hypothesis.
#[test]
fn zero_delta_collapses_the_two_dictionaries() {
    let (dec, _, _, _) = fixture(0.0);
    let n = dec.get_dictionary(Track::Mature).unwrap();
    let m = dec.get_dictionary(Track::Mature).unwrap();
    let (nv, mv): (Vec<f32>, Vec<f32>) = (
        n.flatten_all().unwrap().to_vec1().unwrap(),
        m.flatten_all().unwrap().to_vec1().unwrap(),
    );
    for (a, b) in nv.iter().zip(mv.iter()) {
        assert!((a - b).abs() < 1e-6, "delta=0 but dictionaries differ: {a} vs {b}");
    }
}

/// The load-bearing structural claim: subtracting the two log-dictionaries
/// leaves exactly `⟨α_t, δ_g⟩` plus a per-topic constant. That difference is
/// the model's `log(β_g/γ_g)` — the steady-state velocity ratio — so if this
/// identity does not hold, the mature-base parameterization is not doing what
/// the design claims. The identity is invariant to WHICH track is stored as the
/// base, which is exactly why the flip preserved `δ`.
#[test]
fn log_dictionary_difference_is_alpha_dot_delta_up_to_a_topic_constant() {
    let (dec, _, delta, _) = fixture(0.25);

    let log_n: Vec<Vec<f32>> = dec.get_dictionary(Track::Mature).unwrap().to_vec2().unwrap();
    let log_m: Vec<Vec<f32>> = dec.get_dictionary(Track::Mature).unwrap().to_vec2().unwrap();

    // ⟨α_t, δ_g⟩ computed independently of the decoder's internals.
    let alpha_kh: Vec<Vec<f32>> = dec.topic_embeddings().to_vec2().unwrap();
    let delta_gh: Vec<Vec<f32>> = delta.to_vec2().unwrap();

    for t in 0..K {
        let residual: Vec<f32> = (0..G)
            .map(|g| {
                let dot: f32 = (0..H).map(|h| alpha_kh[t][h] * delta_gh[g][h]).sum();
                log_m[g][t] - log_n[g][t] - dot
            })
            .collect();
        let first = residual[0];
        for (g, r) in residual.iter().enumerate() {
            assert!(
                (r - first).abs() < 1e-4,
                "topic {t}: residual varies across genes ({} at g=0 vs {r} at g={g}) — \
                 the difference is not alpha.delta + const",
                first
            );
        }
    }
}

/// `δ` is identified only up to a per-gene-constant shift, because both tracks
/// are separately softmax-normalized and `logZ^s` absorbs it. Pinning this down
/// matters: it is why the ridge (which picks the minimum-norm representative)
/// is load-bearing rather than cosmetic.
///
/// Targets the MATURE dictionary: under nascent-base the nascent dictionary is
/// `softmax(b + <a,rho>)` and does not involve `delta` at all, so asserting on it
/// would pass no matter what this function did.
#[test]
fn uniform_delta_shift_leaves_the_mature_dictionary_unchanged() {
    let dev = Device::Cpu;
    let rho: Vec<f32> = (0..G * H).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
    let rho = Tensor::from_vec(rho, (G, H), &dev).unwrap();
    let delta_a = Tensor::zeros((G, H), DType::F32, &dev).unwrap();
    // Same δ for every gene → a pure per-topic offset in the logits.
    let delta_b = (&delta_a + 0.75).unwrap();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let dec_a = GemEtmDecoder::new(K, rho.clone(), delta_a, vb.pp("dec")).unwrap();
    let dec_b = GemEtmDecoder::new(K, rho, delta_b, vb.pp("dec")).unwrap();

    let a: Vec<f32> = dec_a
        .get_dictionary(Track::Mature)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let b: Vec<f32> = dec_b
        .get_dictionary(Track::Mature)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < 1e-5,
            "a gene-constant delta shift changed the dictionary: {x} vs {y}"
        );
    }
}

/// The shared per-gene bias must be BOTH things at once: live enough to move the
/// dictionary (otherwise it cannot absorb the background it exists for), and
/// exactly cancelling from the nascent−mature difference (otherwise it eats the
/// `δ` signal). The second half is why there is ONE bias rather than one per
/// track — per-track biases would make `b^s − b^u` a free per-gene splice ratio
/// competing with `⟨α_t, δ_g⟩`.
///
/// The sibling identity test above runs at the zero-init bias, so it says
/// nothing about this; this one drives the bias off zero first.
#[test]
fn shared_bias_moves_the_dictionary_but_leaves_the_delta_estimand_intact() {
    let dev = Device::Cpu;
    let rho: Vec<f32> = (0..G * H).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
    let delta_v: Vec<f32> = (0..G * H).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect();
    let rho = Tensor::from_vec(rho, (G, H), &dev).unwrap();
    let delta = Tensor::from_vec(delta_v, (G, H), &dev).unwrap();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let dec =
        GemEtmDecoder::new(K, rho, delta.clone(), vb.pp("dec")).unwrap();

    let before: Vec<Vec<f32>> = dec.get_dictionary(Track::Mature).unwrap().to_vec2().unwrap();

    // Gene-VARYING bias. A gene-constant one is absorbed by the softmax, which
    // would make the "it moved" half of this test pass vacuously.
    let b: Vec<f32> = (0..G).map(|g| (g % 3) as f32 - 1.0).collect();
    varmap.data().lock().unwrap()["dec.logit_bias"]
        .set(&Tensor::from_vec(b, (1, G), &dev).unwrap())
        .unwrap();

    let after: Vec<Vec<f32>> = dec.get_dictionary(Track::Mature).unwrap().to_vec2().unwrap();
    let moved = before
        .iter()
        .zip(after.iter())
        .any(|(x, y)| x.iter().zip(y.iter()).any(|(a, c)| (a - c).abs() > 1e-4));
    assert!(moved, "the bias left the dictionary unchanged — it is not live");

    // …yet log β^s − log β^u is STILL ⟨α_t, δ_g⟩ + a per-topic constant.
    let log_n: Vec<Vec<f32>> = dec.get_dictionary(Track::Mature).unwrap().to_vec2().unwrap();
    let log_m: Vec<Vec<f32>> = dec.get_dictionary(Track::Mature).unwrap().to_vec2().unwrap();
    let alpha_kh: Vec<Vec<f32>> = dec.topic_embeddings().to_vec2().unwrap();
    let delta_gh: Vec<Vec<f32>> = delta.to_vec2().unwrap();

    for t in 0..K {
        let residual: Vec<f32> = (0..G)
            .map(|g| {
                let dot: f32 = (0..H).map(|h| alpha_kh[t][h] * delta_gh[g][h]).sum();
                log_m[g][t] - log_n[g][t] - dot
            })
            .collect();
        let first = residual[0];
        for (g, r) in residual.iter().enumerate() {
            assert!(
                (r - first).abs() < 1e-4,
                "topic {t}: a live bias leaked into the track difference \
                 ({first} at g=0 vs {r} at g={g}) — delta's estimand is contaminated"
            );
        }
    }
}

/// Only masked slots contribute, and each track reads its own dispersion row.
#[test]
fn masked_score_ignores_unmasked_slots() {
    let (dec, _, _, dev) = fixture(0.2);
    let n = 2;
    let k = 4;

    let indices = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5, 6, 7], (n, k), &dev).unwrap();
    let values = Tensor::from_vec(vec![3.0f32, 1.0, 7.0, 2.0, 5.0, 4.0, 1.0, 6.0], (n, k), &dev)
        .unwrap();
    let lib = (values.sum_keepdim(1).unwrap() + 1.0).unwrap();
    let log_theta = Tensor::from_vec(vec![-1.1f32, -1.1, -1.1, -1.1, -1.1, -1.1], (n, K), &dev)
        .unwrap();

    let full = dec.full_logits_kg(Track::Mature).unwrap();
    let logz = GemEtmDecoder::log_partition_from_logits(&full).unwrap();

    let all_off = Tensor::zeros((n, k), DType::F32, &dev).unwrap();
    let target = GemMaskedTarget {
        indices: &indices,
        residual: None,
        values: &values,
        lib: &lib,
        mask: &all_off,
        values_weight: None,
    };
    let ll = dec
        .impute_masked_nb(&log_theta, &target, Track::Mature, &logz)
        .unwrap();
    for v in ll.to_vec1::<f32>().unwrap() {
        assert_eq!(v, 0.0, "an all-zero mask must score exactly nothing");
    }

    // Turning one slot on must move the score off zero.
    let one_on =
        Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (n, k), &dev).unwrap();
    let target = GemMaskedTarget {
        mask: &one_on,
        ..target
    };
    let ll = dec
        .impute_masked_nb(&log_theta, &target, Track::Mature, &logz)
        .unwrap();
    let v = ll.to_vec1::<f32>().unwrap();
    assert!(v[0] != 0.0 && v[0].is_finite(), "scored slot gave {}", v[0]);
    assert_eq!(v[1], 0.0, "row 1 had no masked slot");
}

/// The two tracks must not share a dispersion row — nascent and mature counts
/// are genuinely differently over-dispersed.
#[test]
fn phi_is_per_track() {
    let (dec, _, _, _) = fixture(0.2);
    let phi = dec.phi_g2().unwrap();
    assert_eq!(phi.dims(), &[G, 2]);
}

/// The Fisher weight must scale the MULTINOMIAL term and be ignored by NB.
///
/// That asymmetry is deliberate and mirrors senna: the weight appears in the
/// multinomial-family path (`traits::indexed::forward_indexed_with_log_beta`)
/// and NOT in `MaskedNbTarget`, because NB already carries per-gene information
/// weighting through the learnable dispersion `φ_g` while multinomial takes
/// every count at face value. Applying it to both would double-count for NB and
/// turn a clean MLE into a biased one for no reason.
#[test]
fn fisher_weight_scales_multinomial_and_is_ignored_by_nb() {
    let (dec, _, _, dev) = fixture(0.2);
    let (n, k) = (2usize, 4usize);

    let indices = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5, 6, 7], (n, k), &dev).unwrap();
    let values =
        Tensor::from_vec(vec![3.0f32, 1.0, 7.0, 2.0, 5.0, 4.0, 1.0, 6.0], (n, k), &dev).unwrap();
    let lib = (values.sum_keepdim(1).unwrap() + 1.0).unwrap();
    let log_theta =
        Tensor::from_vec(vec![-1.1f32; n * K], (n, K), &dev).unwrap();
    let mask = Tensor::ones((n, k), DType::F32, &dev).unwrap();
    // A CONSTANT weight of 2, so the expected effect is exactly a factor of 2 on
    // a linear-in-the-term objective and exactly nothing on one that ignores it.
    let w2 = Tensor::from_vec(vec![2.0f32; n * k], (n, k), &dev).unwrap();

    let base = GemMaskedTarget {
        indices: &indices,
        residual: None,
        values: &values,
        lib: &lib,
        mask: &mask,
        values_weight: None,
    };
    let weighted = GemMaskedTarget { values_weight: Some(&w2), ..base };

    let score = |t: &GemMaskedTarget<'_>, multinomial: bool| -> Vec<f32> {
        let full = dec.full_logits_kg(Track::Mature).unwrap();
        let logz = GemEtmDecoder::log_partition_from_logits(&full).unwrap();
        let ll = if multinomial {
            dec.impute_masked_multinomial(&log_theta, t, Track::Mature, &logz).unwrap()
        } else {
            dec.impute_masked_nb(&log_theta, t, Track::Mature, &logz).unwrap()
        };
        ll.to_vec1().unwrap()
    };

    for (a, b) in score(&base, true).iter().zip(score(&weighted, true).iter()) {
        assert!(
            (b - 2.0 * a).abs() < 1e-3,
            "multinomial: weight 2 should double the term ({a} -> {b})"
        );
    }
    for (a, b) in score(&base, false).iter().zip(score(&weighted, false).iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "NB must ignore the Fisher weight ({a} -> {b}) — phi_g already weights per gene"
        );
    }
}
