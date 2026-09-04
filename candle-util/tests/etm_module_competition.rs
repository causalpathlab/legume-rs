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
    let vb = VarBuilder::from_tensors(ts, DType::F32, &dev);
    EmbeddedNbTopicDecoder::new(K, rho, vb.pp("dec")).unwrap()
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
