//! The chunked likelihoods exist only to save memory, so each must agree with
//! the dense path it replaces — at every chunk width, including ones that do
//! not divide the gene count.

use super::*;
use crate::traits::model::DecoderModuleT;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

fn setup(d: usize, k: usize, n: usize) -> (VarMap, Device, Tensor, Tensor) {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let raw = Tensor::rand(-1f32, 1f32, (n, k), &dev).unwrap();
    // The topic decoders take log-simplex z, not a raw Gaussian.
    let log_z = candle_nn::ops::log_softmax(&raw, 1).unwrap();
    let x = Tensor::rand(0f32, 8f32, (n, d), &dev).unwrap();
    (varmap, dev, log_z, x)
}

fn assert_matches_dense<D: DecoderModuleT>(dec: &D, log_z: &Tensor, x: &Tensor, d: usize) {
    let (_, dense) = dec
        .forward_with_llik(log_z, x, &|_, _| unreachable!())
        .unwrap();
    let dense: Vec<f32> = dense.to_vec1().unwrap();
    assert!(dec.llik_is_gene_chunked(), "decoder must declare it chunks");
    for chunk in [1usize, 3, 7, d, d * 2] {
        let got: Vec<f32> = dec
            .llik_gene_chunked(log_z, x, chunk)
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

#[test]
fn multinom_chunked_llik_matches_the_dense_path() {
    let (d, k, n) = (23usize, 4usize, 5usize);
    let (varmap, dev, log_z, x) = setup(d, k, n);
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let dec = MultinomTopicDecoder::new(d, k, vb.pp("dec")).unwrap();
    assert_matches_dense(&dec, &log_z, &x, d);
}

/// The per-gene weights are narrowed alongside the genes; a slice must pick up
/// its own weights, not the first `len` of them.
#[test]
fn multinom_chunking_respects_feature_weights() {
    let (d, k, n) = (17usize, 3usize, 4usize);
    let (varmap, dev, log_z, x) = setup(d, k, n);
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let mut dec = MultinomTopicDecoder::new(d, k, vb.pp("dec")).unwrap();
    // Deliberately non-uniform, so mis-aligned narrowing cannot coincide.
    let w: Vec<f32> = (0..d).map(|i| 0.1 + (i as f32) / d as f32).collect();
    dec.attach_feature_weights(&w, &dev).unwrap();
    assert_matches_dense(&dec, &log_z, &x, d);
}

#[test]
fn nb_chunked_llik_matches_the_dense_path() {
    let (d, k, n) = (19usize, 3usize, 6usize);
    let (varmap, dev, log_z, x) = setup(d, k, n);
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let dec = NbTopicDecoder::new(d, k, vb.pp("dec")).unwrap();
    assert_matches_dense(&dec, &log_z, &x, d);
}

/// A cell with no counts has a zero library size; the sliced path must still
/// produce a finite score for it.
#[test]
fn an_empty_cell_scores_finitely() {
    let (d, k, n) = (12usize, 2usize, 2usize);
    let (varmap, dev, log_z, _) = setup(d, k, n);
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let dec = NbTopicDecoder::new(d, k, vb.pp("dec")).unwrap();
    let x = Tensor::zeros((n, d), DType::F32, &dev).unwrap();
    let got: Vec<f32> = dec
        .llik_gene_chunked(&log_z, &x, 5)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(got.iter().all(|v| v.is_finite()), "got {got:?}");
}

#[test]
fn gene_slices_tile_the_axis_exactly() {
    let slices: Vec<(usize, usize)> = crate::decoder::gene_slices(10, 4).collect();
    assert_eq!(slices, vec![(0, 4), (4, 4), (8, 2)]);
    assert_eq!(
        crate::decoder::gene_slices(0, 4).count(),
        0,
        "empty axis yields none"
    );
    assert_eq!(
        crate::decoder::gene_slices(5, 0).collect::<Vec<_>>(),
        (0..5).map(|i| (i, 1)).collect::<Vec<_>>(),
        "a zero chunk must not divide by zero"
    );
}
