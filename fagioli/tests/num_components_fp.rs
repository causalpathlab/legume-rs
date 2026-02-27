//! Simulation: does over-specifying L in SuSiE lead to false positives?
//!
//! Setup: n=500, p=200, 2 true causal SNPs (h²≈0.4), LD blocks of 10.
//! Sweep L ∈ {1, 2, 5, 10, 20, 50} for both CAVI and SGVB SuSiE.
//!
//! Run: cargo test -p fagioli --test num_components_fp -- --nocapture

use anyhow::Result;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::sgvb::cavi_susie::{cavi_susie, CaviSusieParams};
use fagioli::sgvb::{fit_block, BlockFitResult, FitConfig, ModelType};
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rand::prelude::*;
use rand_distr::Normal;
use std::collections::HashSet;

/// Generate X (n × p) with block-LD structure.
/// Within each block of `block_size` SNPs, pairwise r² ≈ 0.5.
fn generate_x_with_ld(n: usize, p: usize, block_size: usize, rng: &mut StdRng) -> DMatrix<f32> {
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let mut x = DMatrix::zeros(n, p);
    for i in 0..n {
        let mut j = 0;
        while j < p {
            let end = (j + block_size).min(p);
            let base: f32 = normal.sample(rng);
            for jj in j..end {
                x[(i, jj)] = 0.7 * base + (1.0 - 0.49f32).sqrt() * normal.sample(rng);
            }
            j = end;
        }
    }
    x.scale_columns_inplace();
    x
}

struct SimResult {
    l: usize,
    tp: usize,
    fp: usize,
    top5: Vec<(usize, f32, bool)>, // (idx, pip, is_causal)
}

fn count_discoveries(
    pip: &[f32],
    causal_set: &HashSet<usize>,
    threshold: f32,
) -> (usize, usize, Vec<(usize, f32, bool)>) {
    let mut tp = 0;
    let mut fp = 0;
    for (j, &p) in pip.iter().enumerate() {
        if p > threshold {
            if causal_set.contains(&j) {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    let mut sorted: Vec<(usize, f32)> = pip.iter().enumerate().map(|(j, &v)| (j, v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top5: Vec<(usize, f32, bool)> = sorted
        .iter()
        .take(5)
        .map(|&(j, v)| (j, v, causal_set.contains(&j)))
        .collect();
    (tp, fp, top5)
}

fn print_table(method: &str, results: &[SimResult], threshold: f32) {
    println!("\n  {} (PIP > {}):", method, threshold);
    println!(
        "  {:>4}  {:>3}  {:>3}  {:>36}",
        "L", "TP", "FP", "top 5 PIPs (idx:pip, *=causal)"
    );
    println!("  {:-<52}", "");
    for r in results {
        let top_str: String = r
            .top5
            .iter()
            .map(|(j, v, c)| {
                let marker = if *c { "*" } else { "" };
                format!("{}{}: {:.3}", j, marker, v)
            })
            .collect::<Vec<_>>()
            .join("  ");
        println!("  {:>4}  {:>3}  {:>3}  {}", r.l, r.tp, r.fp, top_str);
    }
}

#[test]
fn test_cavi_fp_vs_num_components() -> Result<()> {
    let n = 500;
    let p = 200;
    let causal = vec![42, 137];
    let causal_set: HashSet<usize> = causal.iter().copied().collect();
    let effect = 0.5f32;
    let noise_sd = 1.0f32;
    let threshold = 0.5;

    let mut rng = StdRng::seed_from_u64(12345);
    let x = generate_x_with_ld(n, p, 10, &mut rng);

    // y = effect * (x_42 + x_137) + noise
    let normal = Normal::new(0.0f32, noise_sd).unwrap();
    let mut y_data = vec![0.0f32; n];
    for i in 0..n {
        for &c in &causal {
            y_data[i] += effect * x[(i, c)];
        }
        y_data[i] += normal.sample(&mut rng);
    }

    let device = Device::Cpu;
    let x_data: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let x_t = Tensor::from_vec(x_data, (n, p), &device)?.to_dtype(DType::F32)?;
    let y_data_f64: Vec<f64> = y_data.iter().map(|&v| v as f64).collect();
    let y_t = Tensor::from_vec(y_data_f64, n, &device)?.to_dtype(DType::F32)?;

    let l_values = [1, 2, 5, 10, 20, 50];
    let mut results = Vec::new();

    for &l in &l_values {
        let params = CaviSusieParams {
            num_components: l,
            max_iter: 200,
            tol: 1e-6,
            prior_variance: 1.0,
            estimate_residual_variance: true,
        };

        let result = cavi_susie(&x_t, &y_t, &params)?;
        let pip_f32: Vec<f32> = result.pip.iter().map(|&v| v as f32).collect();
        let (tp, fp, top5) = count_discoveries(&pip_f32, &causal_set, threshold);
        results.push(SimResult { l, tp, fp, top5 });
    }

    println!("\n{:=<60}", "");
    println!(
        "  SuSiE FP vs L  (n={}, p={}, causal={:?}, effect={}, noise={})",
        n, p, causal, effect, noise_sd
    );
    println!("{:=<60}", "");
    print_table("CAVI SuSiE", &results, threshold);

    // Check: CAVI should produce 0 FP even at large L (proper shrinkage)
    let max_fp = results.iter().map(|r| r.fp).max().unwrap_or(0);
    println!("\n  Max FP across all L values: {} (CAVI)", max_fp);

    Ok(())
}

#[test]
fn test_sgvb_fp_vs_num_components() -> Result<()> {
    let n = 500;
    let p = 200;
    let causal = vec![42, 137];
    let causal_set: HashSet<usize> = causal.iter().copied().collect();
    let effect = 0.5f32;
    let noise_sd = 1.0f32;
    let threshold = 0.5;

    let mut rng = StdRng::seed_from_u64(12345);
    let x = generate_x_with_ld(n, p, 10, &mut rng);

    let normal = Normal::new(0.0f32, noise_sd).unwrap();
    let mut y = DMatrix::zeros(n, 1);
    for i in 0..n {
        let mut val = 0.0f32;
        for &c in &causal {
            val += effect * x[(i, c)];
        }
        val += normal.sample(&mut rng);
        y[(i, 0)] = val;
    }

    let device = Device::Cpu;
    let l_values = [1, 2, 5, 10, 20, 50];
    let mut results = Vec::new();

    for &l in &l_values {
        let config = FitConfig {
            model_type: ModelType::Susie,
            num_components: l,
            num_sgvb_samples: 20,
            learning_rate: 0.01,
            num_iterations: 500,
            batch_size: 1000,
            prior_vars: vec![0.2],
            elbo_window: 50,
            seed: 42,
            ml_block_size: 50,
            sigma2_inf: 0.0,
        };

        let result: BlockFitResult = fit_block(&x, &y, None, &config, &device)?;
        let pip: Vec<f32> = (0..p).map(|j| result.pip[(j, 0)]).collect();
        let (tp, fp, top5) = count_discoveries(&pip, &causal_set, threshold);
        results.push(SimResult { l, tp, fp, top5 });
    }

    println!("\n{:=<60}", "");
    println!(
        "  SuSiE FP vs L  (n={}, p={}, causal={:?}, effect={}, noise={})",
        n, p, causal, effect, noise_sd
    );
    println!("{:=<60}", "");
    print_table("SGVB SuSiE", &results, threshold);

    let max_fp = results.iter().map(|r| r.fp).max().unwrap_or(0);
    println!("\n  Max FP across all L values: {} (SGVB)", max_fp);

    Ok(())
}
