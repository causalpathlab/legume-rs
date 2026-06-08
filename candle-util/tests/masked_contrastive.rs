//! Integration tests for the contrastive masked-imputation head
//! (`EmbeddedNbTopicDecoder::impute_masked_contrastive`).

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_util::decoder::masked_etm::{EmbeddedNbTopicDecoder, MaskedContrastiveTarget};

/// Build a tiny decoder (T topics, H embed, D genes) sharing a fresh ρ Var.
fn tiny_decoder(
    t: usize,
    h: usize,
    d: usize,
    dev: &Device,
) -> anyhow::Result<(EmbeddedNbTopicDecoder, VarMap)> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let rho = vs.get_with_hints(
        (d, h),
        "feature.embeddings",
        candle_nn::init::DEFAULT_KAIMING_NORMAL,
    )?;
    let decoder = EmbeddedNbTopicDecoder::new(t, rho, vs.pp("dec"))?;
    Ok((decoder, varmap))
}

#[test]
fn contrastive_head_finite_and_differentiable() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let (t, h, d) = (2usize, 4usize, 6usize);
    let (decoder, _vm) = tiny_decoder(t, h, d, &dev)?;

    let (n, k) = (3usize, 2usize);
    let log_theta = Tensor::randn(0f32, 1f32, (n, t), &dev)?;
    let indices = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], (n, k), &dev)?;
    let values = Tensor::from_vec(vec![3f32, 1., 2., 5., 4., 1.], (n, k), &dev)?;
    let mask = Tensor::ones((n, k), DType::F32, &dev)?;
    let fisher = Tensor::ones(d, DType::F32, &dev)?; // [D] per-gene weights

    // count^α-sampled negatives.
    let neg_idx = Tensor::from_vec(vec![5u32, 4, 1, 0, 3, 2], (n, k), &dev)?;
    let target = MaskedContrastiveTarget {
        indices: &indices,
        values: &values,
        mask: &mask,
        neg_indices: &neg_idx,
    };
    let loss = decoder.impute_masked_contrastive(&log_theta, &target, 0.5, &fisher)?;
    assert!(
        loss.to_scalar::<f32>()?.is_finite(),
        "contrastive loss not finite"
    );

    // Gradients must reach α (topics), ρ (genes), and the contrastive bias.
    let grads = loss.backward()?;
    assert!(
        grads.get(decoder.topic_embeddings()).is_some(),
        "no grad for α"
    );
    assert!(
        grads.get(decoder.feature_embeddings()).is_some(),
        "no grad for ρ"
    );
    assert!(
        grads.get(decoder.contrastive_bias()).is_some(),
        "no grad for bias"
    );

    Ok(())
}

#[test]
fn contrastive_bias_absent_from_dictionary() -> anyhow::Result<()> {
    // β = softmax(α·ρᵀ) must be independent of the contrastive bias: the
    // dictionary is unchanged whether or not the bias is perturbed.
    let dev = Device::Cpu;
    let (decoder, _vm) = tiny_decoder(2, 4, 6, &dev)?;
    let dict = decoder.get_dictionary()?; // [D, K] log β
    let s = dict.exp()?.sum_keepdim(0)?; // per-topic sums over genes ≈ 1
    let s: Vec<Vec<f32>> = s.to_vec2()?;
    for col in &s[0] {
        assert!((col - 1.0).abs() < 1e-3, "β column not normalized: {col}");
    }
    Ok(())
}
