use super::*;
use crate::loss::nb_log_likelihood;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

/// The chunked scorer exists only to save memory, so it has to agree with
/// the dense path it replaces — for every chunk width, including ones that
/// do not divide the gene count evenly.
#[test]
fn chunked_llik_matches_the_dense_path() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let (n, d, k) = (5usize, 23usize, 4usize);
    let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();

    let z = Tensor::rand(-1f32, 1f32, (n, k), &dev).unwrap();
    let x = Tensor::rand(0f32, 9f32, (n, d), &dev).unwrap();

    // The dense reference: exactly what `forward_with_llik` computes.
    let (_, dense) = dec
        .forward_with_llik(&z, &x, &|_, _| unreachable!())
        .unwrap();
    let dense: Vec<f32> = dense.to_vec1().unwrap();

    for chunk in [1usize, 2, 7, 23, 100] {
        let got: Vec<f32> = dec
            .llik_gene_chunked(&z, &x, chunk)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(got.len(), n, "chunk {chunk}");
        for (i, (g, e)) in got.iter().zip(&dense).enumerate() {
            assert!(
                (g - e).abs() <= 1e-3 * e.abs().max(1.0),
                "chunk {chunk}, cell {i}: chunked {g} vs dense {e}"
            );
        }
    }
}

/// A cell with no counts still has a well-defined score, and the streaming
/// max/sumexp must not produce NaN for it.
#[test]
fn an_empty_cell_scores_finitely() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let (n, d, k) = (2usize, 9usize, 3usize);
    let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();
    let z = Tensor::rand(-1f32, 1f32, (n, k), &dev).unwrap();
    let x = Tensor::zeros((n, d), DType::F32, &dev).unwrap();
    let got: Vec<f32> = dec.llik_gene_chunked(&z, &x, 4).unwrap().to_vec1().unwrap();
    assert!(got.iter().all(|v| v.is_finite()), "got {got:?}");
}

/// The reference against which the dense path itself is defined, so a
/// change to either is caught rather than silently agreed upon.
#[test]
fn the_dense_path_is_the_nb_likelihood_of_its_own_rate() {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let (n, d, k) = (3usize, 11usize, 2usize);
    let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();
    let z = Tensor::rand(-1f32, 1f32, (n, k), &dev).unwrap();
    let x = Tensor::rand(0f32, 4f32, (n, d), &dev).unwrap();

    let (pi, llik) = dec
        .forward_with_llik(&z, &x, &|_, _| unreachable!())
        .unwrap();
    let lib = x.sum_keepdim(1).unwrap();
    let mu = pi.broadcast_mul(&lib).unwrap();
    let want: Vec<f32> = nb_log_likelihood(&x, &mu, &dec.log_phi_1d)
        .unwrap()
        .to_vec1()
        .unwrap();
    let got: Vec<f32> = llik.to_vec1().unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{g} vs {w}");
    }
}
