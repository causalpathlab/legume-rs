//! Recovery tests for the NCE fit.
//!
//! The decisive one is [`test_recovers_a_known_low_rank_geometry`]: it plants a
//! sparse `U` and a dense `V`, generates whitened data from exactly the model's
//! own law, and asks whether the fit reproduces the `A`-invariant geometry
//! `V U'U V'`. Everything before this commit built instruments; this is the
//! first test that points them at the low-rank premise itself.

use super::*;
use crate::embedding::model::EmbedConfig;
use crate::embedding::whiten::WhitenedBlock;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

fn randn(r: usize, c: usize, rng: &mut SmallRng) -> DMatrix<f32> {
    DMatrix::from_fn(r, c, |_, _| {
        let v: f64 = StandardNormal.sample(rng);
        v as f32
    })
}

/// A sparse loading: `nnz` variants per program carry an effect, the rest zero.
fn sparse_u(p: usize, h: usize, nnz: usize, rng: &mut SmallRng) -> DMatrix<f32> {
    let mut u = DMatrix::<f32>::zeros(p, h);
    for col in 0..h {
        for j in sample(rng, p, nnz.min(p)) {
            let v: f64 = StandardNormal.sample(rng);
            u[(j, col)] = v as f32;
        }
    }
    u
}

struct Planted {
    blocks: Vec<WhitenedBlock>,
    geometry: DMatrix<f32>,
}

/// Generate whitened blocks from the model's own law at a chosen signal level.
fn plant(
    num_blocks: usize,
    k: usize,
    p: usize,
    t: usize,
    h: usize,
    snr: f32,
    seed: u64,
) -> Planted {
    let mut rng = SmallRng::seed_from_u64(seed);
    let v_true = randn(t, h, &mut rng);

    let mut blocks = Vec::with_capacity(num_blocks);
    let mut utu = DMatrix::<f32>::zeros(h, h);

    for bi in 0..num_blocks {
        let x_design = randn(k, p, &mut rng);
        let u_true = sparse_u(p, h, 3, &mut rng);
        utu += u_true.transpose() * &u_true;

        let mut mean = (&x_design * &u_true) * v_true.transpose();
        // Normalise the mean to the requested signal-to-noise, so the test is
        // about structure rather than about how big the numbers happen to be.
        let rms = (mean.iter().map(|v| v * v).sum::<f32>() / (k * t) as f32).sqrt();
        if rms > 0.0 {
            mean *= snr / rms;
        }

        let noise = randn(k, t, &mut rng);
        blocks.push(WhitenedBlock {
            block_idx: bi,
            snp_start: bi * p,
            num_snps: p,
            x_design,
            z_white: mean + noise,
            d_sq: vec![1.0; k],
        });
    }

    Planted {
        geometry: &v_true * utu * v_true.transpose(),
        blocks,
    }
}

fn unit_noise(blocks: &[WhitenedBlock]) -> NoiseModel {
    let d_sq: Vec<Vec<f32>> = blocks.iter().map(|b| b.d_sq.clone()).collect();
    NoiseModel::new(&d_sq, 1.0, 1.0, 1.0)
}

/// Correlation between strict upper triangles.
fn offdiag_corr(a: &DMatrix<f32>, b: &DMatrix<f32>) -> f32 {
    let t = a.nrows();
    let (mut va, mut vb) = (Vec::new(), Vec::new());
    for i in 0..t {
        for j in (i + 1)..t {
            va.push(a[(i, j)]);
            vb.push(b[(i, j)]);
        }
    }
    let n = va.len() as f32;
    let (ma, mb) = (va.iter().sum::<f32>() / n, vb.iter().sum::<f32>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..va.len() {
        let (da, db) = ((va[i] - ma) as f64, (vb[i] - mb) as f64);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    (sab / (saa * sbb).sqrt()) as f32
}

#[test]
fn test_recovers_a_known_low_rank_geometry() {
    let (t, h) = (10, 3);
    let planted = plant(6, 40, 50, t, h, 2.0, 7);
    let noise = unit_noise(&planted.blocks);

    let cfg = EmbedConfig {
        embedding_dim: h,
        num_negatives: 4,
        prior_inclusion: 0.05,
        learning_rate: 0.05,
        num_iterations: 400,
        grad_clip: Some(10.0),
        dense_arm: false,
        gauge_weight: 0.0,
        seed: 11,
    };
    let fit = train(&planted.blocks, &noise, &cfg, &Device::Cpu).unwrap();

    let corr = offdiag_corr(&fit.trait_geometry, &planted.geometry);
    let first = fit.loss_trace.first().copied().unwrap_or(0.0);
    let last = fit.loss_trace.last().copied().unwrap_or(0.0);
    println!(
        "geometry recovery corr = {corr:.3}; loss {first:.4} -> {last:.4}; offset {:+.4}",
        fit.offset
    );

    assert!(last < first, "loss should decrease: {first} -> {last}");
    assert!(
        corr > 0.6,
        "fitted V U'U V' should track the planted geometry, got {corr}"
    );
}

/// With no signal at all there is nothing to recover, and the fit must not
/// invent geometry out of noise.
#[test]
fn test_pure_noise_yields_no_geometry() {
    let (t, h) = (10, 3);
    let planted = plant(6, 40, 50, t, h, 0.0, 13);
    let noise = unit_noise(&planted.blocks);

    let cfg = EmbedConfig {
        embedding_dim: h,
        num_negatives: 4,
        prior_inclusion: 0.05,
        learning_rate: 0.05,
        num_iterations: 300,
        grad_clip: Some(10.0),
        dense_arm: false,
        gauge_weight: 0.0,
        seed: 17,
    };
    let fit = train(&planted.blocks, &noise, &cfg, &Device::Cpu).unwrap();

    let corr = offdiag_corr(&fit.trait_geometry, &planted.geometry);
    println!("null-data correlation with the (unused) planted geometry: {corr:.3}");
    assert!(
        corr.abs() < 0.6,
        "a null fit should not reproduce a geometry it never saw, got {corr}"
    );
}

/// Recovery should improve with signal strength — a monotone response is much
/// harder to fake than a single passing threshold.
#[test]
fn test_recovery_improves_with_signal() {
    let (t, h) = (10, 3);
    let mut corrs = Vec::new();
    for &snr in &[0.25f32, 1.0, 3.0] {
        let planted = plant(5, 35, 45, t, h, snr, 29);
        let noise = unit_noise(&planted.blocks);
        let cfg = EmbedConfig {
            embedding_dim: h,
            num_negatives: 4,
            prior_inclusion: 0.05,
            learning_rate: 0.05,
            num_iterations: 300,
            grad_clip: Some(10.0),
            dense_arm: false,
            gauge_weight: 0.0,
            seed: 31,
        };
        let fit = train(&planted.blocks, &noise, &cfg, &Device::Cpu).unwrap();
        let c = offdiag_corr(&fit.trait_geometry, &planted.geometry);
        println!("  snr {snr:>4}: corr {c:.3}");
        corrs.push(c);
    }

    assert!(
        corrs[2] > corrs[0],
        "recovery should improve with signal: {corrs:?}"
    );
    assert!(
        corrs[2] > 0.6,
        "strong signal should be well recovered, got {}",
        corrs[2]
    );
}

/// The offset is a calibration check, and what it detects here is *overfitting*,
/// not miscalibration of the score.
///
/// The score is exactly normalised, so at the truth the offset has nothing to
/// absorb. But `U` carries `p x H` free parameters per block against only `K`
/// eigen-coordinates of data, and when that ratio is poor the discriminator can
/// memorise the fixed positives — the positives are a fixed dataset while the
/// negatives are redrawn every step, so overfitting is one-sided and pushes the
/// offset up. Improving samples-per-parameter must pull it back down; if it did
/// not, the offset would be measuring something other than what is claimed.
#[test]
fn test_offset_tracks_samples_per_parameter() {
    let run = |k: usize, p: usize, seed: u64| -> (f32, f32) {
        let planted = plant(4, k, p, 8, 3, 1.0, seed);
        let noise = unit_noise(&planted.blocks);
        let cfg = EmbedConfig {
            embedding_dim: 3,
            num_negatives: 4,
            learning_rate: 0.05,
            num_iterations: 300,
            seed: 43,
            ..Default::default()
        };
        let fit = train(&planted.blocks, &noise, &cfg, &Device::Cpu).unwrap();
        (fit.offset, *fit.loss_trace.last().unwrap())
    };

    // Overparameterized: 40 params per block against 30 coordinates of data.
    let (offset_thin, loss_thin) = run(30, 40, 41);
    // Well-determined: 12 params per block against 240 coordinates.
    let (offset_rich, loss_rich) = run(240, 12, 41);

    println!(
        "offset: overparameterized {offset_thin:+.4} (loss {loss_thin:.4}), \
         well-determined {offset_rich:+.4} (loss {loss_rich:.4})"
    );

    assert!(
        offset_rich.abs() < offset_thin.abs(),
        "more data per parameter should shrink the offset: {offset_rich} vs {offset_thin}"
    );
    assert!(
        offset_rich.abs() < 0.5,
        "a well-determined fit should leave the offset near zero, got {offset_rich}"
    );
}

#[test]
fn test_fit_shapes_and_sparsity() {
    let (t, h, p) = (6, 4, 30);
    let planted = plant(3, 20, p, t, h, 1.0, 53);
    let noise = unit_noise(&planted.blocks);
    let cfg = EmbedConfig {
        embedding_dim: h,
        num_iterations: 50,
        seed: 59,
        ..Default::default()
    };
    let fit = train(&planted.blocks, &noise, &cfg, &Device::Cpu).unwrap();

    assert_eq!(fit.v_check.nrows(), t);
    assert_eq!(fit.v_check.ncols(), h);
    assert_eq!(fit.u_mean.len(), 3);
    assert_eq!(fit.u_pip.len(), 3);
    for u in &fit.u_mean {
        assert_eq!(u.nrows(), p);
        assert_eq!(u.ncols(), h);
        assert!(u.iter().all(|v| v.is_finite()));
    }
    for pip in &fit.u_pip {
        assert!(
            pip.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "PIPs must be probabilities"
        );
    }
}

#[test]
fn test_empty_block_list_is_an_error() {
    let noise = NoiseModel::new(&[], 1.0, 1.0, 1.0);
    let cfg = EmbedConfig::default();
    assert!(train(&[], &noise, &cfg, &Device::Cpu).is_err());
}
