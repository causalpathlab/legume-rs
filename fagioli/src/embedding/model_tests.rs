//! Unit tests for the score.

use super::*;
use candle_util::candle_core::DType;
use candle_util::candle_nn::VarMap;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

pub(super) fn rand_matrix(r: usize, c: usize, rng: &mut SmallRng) -> DMatrix<f32> {
    DMatrix::from_fn(r, c, |_, _| {
        let v: f64 = StandardNormal.sample(rng);
        v as f32
    })
}

pub(super) fn make_block(
    block_idx: usize,
    k: usize,
    p: usize,
    t: usize,
    rng: &mut SmallRng,
) -> WhitenedBlock {
    WhitenedBlock {
        block_idx,
        snp_start: block_idx * p,
        num_snps: p,
        x_design: rand_matrix(k, p, rng),
        z_white: rand_matrix(k, t, rng),
        d_sq: vec![1.0; k],
    }
}

fn unit_noise(blocks: &[WhitenedBlock]) -> NoiseModel {
    let d_sq: Vec<Vec<f32>> = blocks.iter().map(|b| b.d_sq.clone()).collect();
    // c = 1, tau = lambda ⇒ every scale is exactly 1.
    NoiseModel::new(&d_sq, 1.0, 1.0, 1.0)
}

fn build(blocks: &[WhitenedBlock], h: usize) -> (VarMap, EmbedModel) {
    let device = Device::Cpu;
    let noise = unit_noise(blocks);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let cfg = EmbedConfig {
        embedding_dim: h,
        ..Default::default()
    };
    let model = EmbedModel::new(&vb, blocks, &noise, cfg, &device).unwrap();
    (varmap, model)
}

#[test]
fn test_score_is_zero_for_a_zero_mean() {
    let mut rng = SmallRng::seed_from_u64(1);
    let blocks = vec![make_block(0, 12, 20, 4, &mut rng)];
    let (_vm, model) = build(&blocks, 3);

    let zero_mean = model.blocks[0].z_white.zeros_like().unwrap();
    let s = model
        .score(0, &model.blocks[0].z_white, &zero_mean)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // <z, 0> − ½‖0‖² = 0, plus a zero-initialised offset.
    assert!(s.iter().all(|v| v.abs() < 1e-6), "expected all-zero score");
}

/// The matched-filter property: for a fixed mean, the score is maximised when
/// the data equals that mean, and equals ½‖μ‖² there.
#[test]
fn test_score_peaks_when_data_equals_the_mean() {
    let mut rng = SmallRng::seed_from_u64(2);
    let blocks = vec![make_block(0, 8, 15, 3, &mut rng)];
    let (_vm, model) = build(&blocks, 2);
    let device = Device::Cpu;

    let mean_mat = rand_matrix(8, 3, &mut rng);
    let mean = {
        use matrix_util::traits::ConvertMatOps;
        mean_mat.to_tensor(&device).unwrap().contiguous().unwrap()
    };

    let at_mean = model.score(0, &mean, &mean).unwrap().to_vec1::<f32>().unwrap();
    let elsewhere = model
        .score(0, &model.blocks[0].z_white, &mean)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // At z = μ the score is ½‖μ‖² per coordinate.
    for ki in 0..8 {
        let want: f32 = 0.5 * (0..3).map(|t| mean_mat[(ki, t)].powi(2)).sum::<f32>();
        assert!(
            (at_mean[ki] - want).abs() < 1e-4,
            "coordinate {ki}: {} vs {want}",
            at_mean[ki]
        );
    }
    let sum_at: f32 = at_mean.iter().sum();
    let sum_else: f32 = elsewhere.iter().sum();
    assert!(
        sum_at > sum_else,
        "score should peak at the mean: {sum_at} vs {sum_else}"
    );
}

/// The noise scale must actually reweight the score: a coordinate with larger
/// noise should contribute proportionally less.
#[test]
fn test_score_downweights_noisier_coordinates() {
    let mut rng = SmallRng::seed_from_u64(3);
    let k = 6;
    let mut block = make_block(0, k, 10, 2, &mut rng);
    // Second half of the spectrum carries 4x the noise variance.
    block.d_sq = (0..k).map(|i| if i < k / 2 { 1.0 } else { 7.0 }).collect();
    let blocks = vec![block];

    let device = Device::Cpu;
    // c = 1, tau = 1, lambda = 1 ⇒ s² = (d²+1)/(d²+1) = 1 everywhere. Use a
    // mismatched lambda so the scales genuinely differ.
    let noise = NoiseModel::new(&[blocks[0].d_sq.clone()], 1.0, 1.0, 0.0);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = EmbedModel::new(
        &vb,
        &blocks,
        &noise,
        EmbedConfig {
            embedding_dim: 2,
            ..Default::default()
        },
        &device,
    )
    .unwrap();

    let ones = DMatrix::<f32>::from_element(k, 2, 1.0);
    let t = {
        use matrix_util::traits::ConvertMatOps;
        ones.to_tensor(&device).unwrap().contiguous().unwrap()
    };
    let s = model.score(0, &t, &t).unwrap().to_vec1::<f32>().unwrap();

    // s² = (d²+1)/d², so the high-d² coordinates are *less* noisy here and
    // should score higher for identical data.
    assert!(
        s[k - 1] > s[0],
        "coordinate weighting is not being applied: {:?}",
        s
    );
}

#[test]
fn test_trait_geometry_is_symmetric_and_sized_by_traits() {
    let mut rng = SmallRng::seed_from_u64(4);
    let t = 5;
    let blocks: Vec<WhitenedBlock> = (0..3)
        .map(|i| make_block(i, 10, 14, t, &mut rng))
        .collect();
    let (_vm, model) = build(&blocks, 3);

    let g = model.trait_geometry().unwrap();
    assert_eq!(g.nrows(), t);
    assert_eq!(g.ncols(), t);
    for i in 0..t {
        for j in 0..t {
            assert!(
                (g[(i, j)] - g[(j, i)]).abs() < 1e-6,
                "V U'U V' must be symmetric"
            );
        }
    }
}

#[test]
fn test_offset_starts_at_zero() {
    let mut rng = SmallRng::seed_from_u64(5);
    let blocks = vec![make_block(0, 6, 10, 3, &mut rng)];
    let (_vm, model) = build(&blocks, 2);
    assert!(model.offset_value().unwrap().abs() < 1e-9);
}
