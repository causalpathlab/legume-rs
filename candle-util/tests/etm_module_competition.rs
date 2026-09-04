//! Module-to-module competition on the feature side of the embedded topic
//! decoder.
//!
//! Each row of the `[K, D]` logit matrix is normalized independently
//! (`log_softmax` over the gene axis), so without intervention the `K` modules
//! are gradient-decoupled: "raise everyone on the abundant genes" is a direction
//! all `K` of them descend at once, and nothing opposes it. Centering `α` over
//! the topic axis removes that shared direction, making each gene's total
//! log-mass a conserved quantity so one module can only gain on a gene at
//! another's expense.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_util::decoder::EmbeddedNbTopicDecoder;
use std::collections::HashMap;

const D: usize = 6;
const H: usize = 3;
const K: usize = 4;

/// Deterministic ρ (no RNG) so the algebraic assertions are exact.
fn rho_rows() -> Vec<f32> {
    (0..D * H).map(|i| (i % 7) as f32 * 0.3 - 0.6).collect()
}

/// Deterministic α with a deliberately non-zero mean archetype ᾱ — that shared
/// direction is precisely what the centering must remove.
fn alpha_rows() -> Vec<f32> {
    (0..K * H).map(|i| (i % 5) as f32 * 0.4 - 0.4).collect()
}

fn decoder_with(alpha: Vec<f32>) -> EmbeddedNbTopicDecoder {
    decoder_with_background(alpha, vec![-(D as f32).ln(); D])
}

/// Same, with an explicit per-gene log-background `log π_g`.
fn decoder_with_background(alpha: Vec<f32>, log_pi: Vec<f32>) -> EmbeddedNbTopicDecoder {
    let dev = Device::Cpu;
    let rho = Tensor::from_vec(rho_rows(), (D, H), &dev).unwrap();
    let mut ts = HashMap::new();
    ts.insert(
        "dec.topic.embeddings".to_string(),
        Tensor::from_vec(alpha, (K, H), &dev).unwrap(),
    );
    ts.insert(
        "dec.log_phi".to_string(),
        Tensor::from_vec(vec![0.693f32; D], (1, D), &dev).unwrap(),
    );
    ts.insert(
        "dec.log_pi".to_string(),
        Tensor::from_vec(log_pi, (1, D), &dev).unwrap(),
    );
    let vb = VarBuilder::from_tensors(ts, DType::F32, &dev);
    EmbeddedNbTopicDecoder::new(K, rho, vb.pp("dec")).unwrap()
}

/// A gene abundant in every cell needs a home the centered topics cannot give
/// it. With `log π` pinned high on gene 0, every topic carries more mass on
/// gene 0 than under a uniform background, while the per-gene budget
/// `Σ_k log β_kg − K·log π_g` stays gene-constant so the competition is intact.
#[test]
fn pinned_background_gives_shared_abundance_a_home() {
    let mut log_pi = vec![-(D as f32).ln(); D];
    log_pi[0] += 3.0;
    let with = decoder_with_background(alpha_rows(), log_pi.clone())
        .get_dictionary()
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let without = decoder_with(alpha_rows())
        .get_dictionary()
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    for k in 0..K {
        assert!(
            with[0][k] > without[0][k] + 1.0,
            "topic {k} must raise gene 0 under the pinned background: {} vs {}",
            with[0][k],
            without[0][k]
        );
    }
    let budget: Vec<f32> = (0..D)
        .map(|g| (0..K).map(|k| with[g][k]).sum::<f32>() - K as f32 * log_pi[g])
        .collect();
    let spread = budget.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - budget.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        spread < 1e-4,
        "per-gene budget must stay conserved, spread {spread:.6}"
    );
}

/// Every gene carries the same total log-mass across the `K` modules, so a
/// module can only gain on a gene by taking from another. Uncentered, that total
/// is `K⟨ᾱ, ρ_g⟩ - Σ_k log Z_k`, whose first term tracks the shared abundance
/// direction and so varies from gene to gene.
#[test]
fn per_gene_log_mass_is_conserved_across_modules() {
    let log_beta_dk = decoder_with(alpha_rows()).get_dictionary().unwrap();
    assert_eq!(log_beta_dk.dims(), &[D, K]);

    let per_gene = log_beta_dk.sum(1).unwrap().to_vec1::<f32>().unwrap();
    let lo = per_gene.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = per_gene.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        hi - lo < 1e-4,
        "Σ_k log β_kg must not depend on the gene; spread was {:.6} over {:?}",
        hi - lo,
        per_gene
    );
}

/// One module's gain on a gene is every other module's loss on that same gene.
/// Pushing `α_k0` toward `ρ_g0` must raise module `k0` on gene `g0` and lower
/// every other module there. Uncentered, the off-diagonal response is exactly
/// zero — the modules do not see each other at all, which is the pathology.
#[test]
fn raising_one_module_on_a_gene_lowers_the_others() {
    const K0: usize = 1;
    const G0: usize = 0;
    const EPS: f32 = 0.5;

    let rho = rho_rows();
    let mut bumped = alpha_rows();
    for h in 0..H {
        bumped[K0 * H + h] += EPS * rho[G0 * H + h];
    }

    let logits = |a: Vec<f32>| -> Vec<f32> {
        decoder_with(a)
            .full_logits_kd()
            .unwrap()
            .i((.., G0))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let before = logits(alpha_rows());
    let after = logits(bumped);

    assert!(
        after[K0] > before[K0] + 1e-6,
        "module {K0} should gain on gene {G0}: {} -> {}",
        before[K0],
        after[K0]
    );
    for j in (0..K).filter(|&j| j != K0) {
        assert!(
            after[j] < before[j] - 1e-6,
            "module {j} must lose on gene {G0} when module {K0} gains, but went \
             {} -> {} (delta {:+.6}; a delta of exactly 0 means the modules are \
             gradient-decoupled)",
            before[j],
            after[j],
            after[j] - before[j]
        );
    }
}

/// The training likelihood must score the *same* `β` the dictionary reports.
/// Each row `n` observes one gene `g_n` alone, placed at a row-specific
/// position of a permuted index row, with its own `θ_n`; the multinomial head
/// then returns `log p_n = log Σ_t θ_nt β_{t,g_n}`. The full-vocab
/// probabilities under one shared `θ` must sum to one, and every row must
/// agree gene-wise with `θ_n · exp(get_dictionary)`. A rate built from the
/// raw `α` while the partition comes from the centered logits breaks both;
/// mixing rows or ignoring the index order breaks the second.
#[test]
fn masked_likelihood_scores_the_dictionary_it_writes() {
    use candle_util::decoder::MaskedNbTarget;
    let dev = Device::Cpu;
    let dec = decoder_with(alpha_rows());
    let full_kd = dec.full_logits_kd().unwrap();
    let beta_dk = dec.get_dictionary().unwrap().exp().unwrap();

    // Row n: a distinct θ_n, a permuted index row, a one-hot at gene n.
    let perm: Vec<u32> = (0..D as u32).map(|g| (g * 5 + 1) % D as u32).collect();
    let mut theta = vec![0f32; D * K];
    let mut values = vec![0f32; D * D];
    for n in 0..D {
        let raw: Vec<f32> = (0..K).map(|k| 1.0 + ((n * 3 + k * 7) % 5) as f32).collect();
        let z: f32 = raw.iter().sum();
        for k in 0..K {
            theta[n * K + k] = raw[k] / z;
        }
        let pos = perm.iter().position(|&g| g as usize == n).unwrap();
        values[n * D + pos] = 1.0;
    }
    let log_theta = Tensor::from_vec(theta.clone(), (D, K), &dev)
        .unwrap()
        .log()
        .unwrap();
    let indices = Tensor::from_vec(perm.clone(), (1, D), &dev)
        .unwrap()
        .broadcast_as((D, D))
        .unwrap()
        .contiguous()
        .unwrap();
    let values = Tensor::from_vec(values, (D, D), &dev).unwrap();
    let ones = Tensor::ones((D, D), DType::F32, &dev).unwrap();
    let lib = Tensor::ones((D, 1), DType::F32, &dev).unwrap();
    let target = MaskedNbTarget {
        indices: &indices,
        residual: None,
        values: &values,
        lib: &lib,
        mask: &ones,
    };
    let log_p = dec
        .impute_masked_multinomial(&log_theta, &target, &full_kd)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let beta = beta_dk.to_vec2::<f32>().unwrap(); // [D, K]
    for n in 0..D {
        let expected: f32 = (0..K).map(|k| theta[n * K + k] * beta[n][k]).sum();
        assert!(
            (log_p[n].exp() - expected).abs() < 1e-5,
            "row {n}: likelihood p={} but dictionary p={}",
            log_p[n].exp(),
            expected
        );
    }

    // Under one shared θ the full-vocab probabilities sum to one.
    let shared = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (1, K), &dev)
        .unwrap()
        .log()
        .unwrap()
        .broadcast_as((D, K))
        .unwrap()
        .contiguous()
        .unwrap();
    let total: f32 = dec
        .impute_masked_multinomial(&shared, &target, &full_kd)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|l| l.exp())
        .sum();
    assert!(
        (total - 1.0).abs() < 1e-4,
        "Σ_g p_g must be 1, got {total:.6}"
    );
}
