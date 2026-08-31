use super::*;
use crate::traits::model::DecoderModuleT;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

/// `senna topic`'s default decoder. Its chunked likelihood exists purely
/// to save memory, so it must agree with the dense path at every chunk
/// width — including ones that do not divide the gene count.
#[test]
fn chunked_llik_matches_the_dense_path() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let (n, d, k) = (6usize, 29usize, 4usize);
    let dec = NbMixtureTopicDecoder::new(d, k, vb.pp("dec")).unwrap();

    // The topic decoders take log-simplex z, not a raw Gaussian.
    let raw = Tensor::rand(-1f32, 1f32, (n, k), &dev).unwrap();
    let log_z = candle_nn::ops::log_softmax(&raw, 1).unwrap();
    let x = Tensor::rand(0f32, 7f32, (n, d), &dev).unwrap();

    let (_, dense) = dec
        .forward_with_llik(&log_z, &x, &|_, _| unreachable!())
        .unwrap();
    let dense: Vec<f32> = dense.to_vec1().unwrap();

    for chunk in [1usize, 5, 29, 64] {
        let got: Vec<f32> = dec
            .llik_gene_chunked(&log_z, &x, chunk)
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (g, e)) in got.iter().zip(&dense).enumerate() {
            assert!(
                (g - e).abs() <= 1e-3 * e.abs().max(1.0),
                "chunk {chunk}, cell {i}: chunked {g} vs dense {e}"
            );
        }
    }
}

/// The ambient/rho mixing and its prior are per cell, not per gene, so
/// they must be applied exactly once however the genes are sliced.
#[test]
fn the_rho_prior_is_added_once_not_per_slice() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let (n, d, k) = (3usize, 16usize, 2usize);
    let mut dec = NbMixtureTopicDecoder::new(d, k, vb.pp("dec")).unwrap();
    dec.rho_prior_weight = 2.5;
    let raw = Tensor::rand(-1f32, 1f32, (n, k), &dev).unwrap();
    let log_z = candle_nn::ops::log_softmax(&raw, 1).unwrap();
    let x = Tensor::rand(0f32, 5f32, (n, d), &dev).unwrap();

    let one: Vec<f32> = dec
        .llik_gene_chunked(&log_z, &x, d)
        .unwrap()
        .to_vec1()
        .unwrap();
    let many: Vec<f32> = dec
        .llik_gene_chunked(&log_z, &x, 3)
        .unwrap()
        .to_vec1()
        .unwrap();
    for (a, b) in one.iter().zip(&many) {
        assert!(
            (a - b).abs() <= 1e-3 * a.abs().max(1.0),
            "slicing changed the prior: {a} vs {b}"
        );
    }
}
