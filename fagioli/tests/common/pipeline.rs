//! The `decompose -> calibrate -> whiten -> train` chain, in one place.
//!
//! Included with `#[path]` alongside `three_class.rs`.
#![allow(dead_code)]

use anyhow::Result;
use candle_util::candle_core::Device;
use fagioli::embedding::model::{EmbedConfig, UPrior};
use fagioli::embedding::noise::NoiseModel;
use fagioli::embedding::train::{train, EmbedFit};
use fagioli::embedding::whiten::whiten_blocks;
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::{decompose_blocks, SumstatInput};

/// Fit the NCE embedding, returning it with the SNP starts of the blocks that
/// were actually fitted.
///
/// The starts matter: `decompose_blocks` is a filter that can drop a block, and
/// `assemble_u` zips loadings against starts positionally — so taking them from
/// `input.blocks` would silently shift every later block onto the wrong
/// variants.
pub fn fit_embedding(
    input: &SumstatInput,
    embedding_dim: usize,
    seed: u64,
) -> Result<(EmbedFit, Vec<usize>)> {
    let bases = decompose_blocks(input);
    let report = calibrate_input(input, &bases).expect("calibration");
    let lambda = report.noise.lambda_white();
    let blocks = whiten_blocks(input, bases, None, lambda)?;
    let d_sq: Vec<Vec<f32>> = blocks.iter().map(|b| b.d_sq.clone()).collect();
    let noise = NoiseModel::new(&d_sq, report.noise.c, report.noise.tau, lambda);
    let starts: Vec<usize> = blocks.iter().map(|b| b.snp_start).collect();

    let fit = train(
        &blocks,
        &noise,
        &EmbedConfig {
            embedding_dim,
            num_negatives: 4,
            prior_inclusion: 0.02,
            u_prior: UPrior::SpikeSlab,
            num_components: 5,
            prior_alpha: 1.0,
            learning_rate: 0.05,
            num_iterations: 400,
            grad_clip: Some(10.0),
            dense_arm: false,
            gauge_weight: 0.0,
            seed,
        },
        &Device::Cpu,
    )?;
    Ok((fit, starts))
}
