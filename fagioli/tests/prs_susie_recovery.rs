//! Simulation: does prs-susie recover true causal SNPs from summary statistics?
//!
//! Uses the same simulation infrastructure as `sim-sumstat`:
//! 1. Generate X with block-LD structure
//! 2. Sample causal effects via `sample_cell_type_genetic_effects`
//! 3. Compose phenotype, compute z-scores = X'y / sqrt(n)
//! 4. Ridge PRS: yhat = U inv(D+λI) V' z
//! 5. SuSiE fine-mapping: yhat ~ X * beta
//! 6. Assert causal SNPs have high PIPs
//!
//! Run: cargo test -p fagioli --test prs_susie_recovery -- --nocapture

use anyhow::Result;
use candle_util::candle_core::Device;
use candle_util::sgvb::cavi_susie::{cavi_susie, CaviSusieParams};
use fagioli::sgvb::{fit_block, fit_block_rss, BlockFitResult, FitConfig, ModelType, RssParams};
use fagioli::simulation::sample_cell_type_genetic_effects;
use fagioli::summary_stats::polygenic_score::{
    compute_all_polygenic_scores_ridge, compute_block_polygenic_scores_ridge,
};
use fagioli::summary_stats::LdBlock;
use matrix_util::traits::{ConvertMatOps, MatOps, SampleOps};
use nalgebra::DMatrix;
use rand::prelude::*;
use rand_distr::Normal;
use rustc_hash::FxHashSet as HashSet;

/// Generate X (n × p) with block-LD structure (same pattern as sim_sumstat).
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

/// Compute z-scores: z_j = X_j' y / sqrt(n) for each trait.
fn compute_zscores(x: &DMatrix<f32>, y: &DMatrix<f32>) -> DMatrix<f32> {
    let n = x.nrows();
    let n_sqrt = (n as f32).sqrt();
    x.tr_mul(y) / n_sqrt
}

/// Build uniform LD blocks from block_size.
fn make_blocks(p: usize, block_size: usize) -> Vec<LdBlock> {
    let mut blocks = Vec::new();
    let mut start = 0;
    let mut idx = 0;
    while start < p {
        let end = (start + block_size).min(p);
        blocks.push(LdBlock {
            block_idx: idx,
            snp_start: start,
            snp_end: end,
            chr: Box::from("1"),
            bp_start: start as u64 * 1000,
            bp_end: end as u64 * 1000,
        });
        start = end;
        idx += 1;
    }
    blocks
}

#[test]
fn test_prs_susie_recovers_causal_snps() -> Result<()> {
    let n = 500;
    let p = 200;
    let t = 1; // single trait for clarity
    let block_size = 20;
    let num_shared_causal = 2;
    let num_independent_causal = 0;
    let h2 = 0.4f32;

    let mut rng = StdRng::seed_from_u64(42);
    let x = generate_x_with_ld(n, p, block_size, &mut rng);
    let blocks = make_blocks(p, block_size);

    // Pick block 3 (SNPs 60-79) as the causal block
    let causal_block_idx = 3;
    let causal_block = &blocks[causal_block_idx];
    let block_m = causal_block.num_snps();

    // Sample causal effects using sim-sumstat's infrastructure
    let effects = sample_cell_type_genetic_effects(
        block_m,
        t,
        num_shared_causal,
        num_independent_causal,
        h2,
        42,
    )?;

    println!("\n{:=<60}", "");
    println!(
        "  PRS-SuSiE recovery test (n={}, p={}, T={}, h²={})",
        n, p, t, h2,
    );
    println!("{:=<60}", "");

    // Print causal SNPs (global indices)
    let causal_global: Vec<usize> = effects
        .shared_causal_indices
        .iter()
        .map(|&local| causal_block.snp_start + local)
        .collect();
    println!(
        "  Causal block: {} (SNPs {}-{})",
        causal_block_idx, causal_block.snp_start, causal_block.snp_end
    );
    println!("  Causal SNPs (global): {:?}", causal_global);
    let _causal_set: HashSet<usize> = causal_global.iter().copied().collect();

    // Compute genetic values: G = X_block * beta
    let x_block = x.columns(causal_block.snp_start, block_m).clone_owned();
    let mut g = DMatrix::zeros(n, t);
    for (j, &snp_idx) in effects.shared_causal_indices.iter().enumerate() {
        for i in 0..n {
            for tt in 0..t {
                g[(i, tt)] += x_block[(i, snp_idx)] * effects.shared_effect_sizes[(tt, j)];
            }
        }
    }

    // Compose phenotype: y = sqrt(h2)*std(G) + sqrt(1-h2)*noise
    g.scale_columns_inplace();
    g *= h2.sqrt();
    let mut noise = DMatrix::<f32>::rnorm(n, t);
    noise.scale_columns_inplace();
    noise *= (1.0 - h2).sqrt();
    let y = &g + &noise;

    // Compute z-scores
    let z = compute_zscores(&x, &y);

    // Show z-scores at causal SNPs
    for &c in &causal_global {
        println!("  z[{}] = {:.3}", c, z[(c, 0)]);
    }

    // ── Ridge PRS ─────────────────────────────────────────────────────────
    let lambda = 0.1f32;
    let yhat = compute_all_polygenic_scores_ridge(&x, &z, &blocks, lambda)?;

    let corr = correlation(&yhat.column(0), &y.column(0));
    println!("\n  Ridge PRS-y correlation: {:.4}", corr);
    assert!(
        corr > 0.2,
        "Ridge PRS should correlate with true y, got {:.4}",
        corr
    );

    // ── CAVI SuSiE: yhat ~ X_causal_block * beta ─────────────────────────
    // Fine-map only the causal block (mimicking per-block fitting)
    let device = Device::Cpu;
    let x_block_tensor = x_block.to_tensor(&device)?;
    let yhat_col: Vec<f32> = (0..n).map(|i| yhat[(i, 0)]).collect();
    let yhat_tensor = candle_util::candle_core::Tensor::from_slice(&yhat_col, (n,), &device)?;

    let params = CaviSusieParams {
        num_components: 10,
        max_iter: 200,
        tol: 1e-6,
        prior_variance: 0.5,
        estimate_residual_variance: true,
        prior_weights: None,
    };

    let cavi_result = cavi_susie(&x_block_tensor, &yhat_tensor, &params)?;
    let pip_cavi: Vec<f32> = cavi_result.pip.iter().map(|&v| v as f32).collect();

    println!(
        "\n  CAVI SuSiE top PIPs (within causal block, SNPs {}-{}):",
        causal_block.snp_start, causal_block.snp_end
    );
    print_top_pips(
        &pip_cavi,
        &effects.shared_causal_indices.iter().copied().collect(),
        10,
    );

    // Check: causal SNPs (local indices) should be in top-5
    let top5_local = top_k_indices(&pip_cavi, 5);
    let local_causal_set: HashSet<usize> = effects.shared_causal_indices.iter().copied().collect();
    let cavi_recovered: usize = local_causal_set
        .iter()
        .filter(|c| top5_local.contains(c))
        .count();

    println!(
        "\n  Causal SNPs in top-5 PIPs: {}/{}",
        cavi_recovered,
        local_causal_set.len(),
    );

    // ── SGVB SuSiE on same block ────────────────────────────────────────
    let y_block_mat = DMatrix::from_fn(n, 1, |i, _| yhat[(i, 0)]);
    let sgvb_config = FitConfig {
        model_type: ModelType::Susie,
        prior_type: fagioli::sgvb::PriorType::Single,
        num_components: 10,
        num_sgvb_samples: 20,
        learning_rate: 0.01,
        num_iterations: 500,
        batch_size: 1000,
        prior_vars: vec![0.5],
        elbo_window: 50,
        seed: 42,
        sigma2_inf: 0.0,
        prior_alpha: 1.0,
    };
    let sgvb_result: BlockFitResult =
        fit_block(&x_block, &y_block_mat, None, &sgvb_config, &device)?;
    let pip_sgvb: Vec<f32> = (0..block_m).map(|j| sgvb_result.pip[(j, 0)]).collect();

    println!("\n  SGVB SuSiE top PIPs:");
    print_top_pips(&pip_sgvb, &local_causal_set, 10);

    let top5_sgvb = top_k_indices(&pip_sgvb, 5);
    let sgvb_recovered: usize = local_causal_set
        .iter()
        .filter(|c| top5_sgvb.contains(c))
        .count();

    println!(
        "  SGVB causal in top-5: {}/{}",
        sgvb_recovered,
        local_causal_set.len(),
    );

    // ── RSS SGVB SuSiE (proper likelihood baseline) ─────────────────────
    let mut x_block_std = x_block.clone();
    x_block_std.scale_columns_inplace();
    let z_block = z.rows(causal_block.snp_start, block_m).clone_owned();
    let pip_rss = run_rss_sgvb(&x_block_std, &z_block, 10, &[0.5], 0.0, 42, &device)?;

    println!("\n  RSS SGVB top PIPs:");
    print_top_pips(&pip_rss, &local_causal_set, 10);

    let top5_rss = top_k_indices(&pip_rss, 5);
    let rss_recovered: usize = local_causal_set
        .iter()
        .filter(|c| top5_rss.contains(c))
        .count();

    // ── RSS CAVI SuSiE ────────────────────────────────────────────────────
    let pip_rss_cavi = run_rss_cavi(&x_block_std, &z_block, 10, 0.5, &device)?;

    println!("\n  RSS CAVI top PIPs:");
    print_top_pips(&pip_rss_cavi, &local_causal_set, 10);

    let top5_rss_cavi = top_k_indices(&pip_rss_cavi, 5);
    let rss_cavi_recovered: usize = local_causal_set
        .iter()
        .filter(|c| top5_rss_cavi.contains(c))
        .count();

    println!(
        "\n  Summary — causal in top-5: CAVI(PRS)={}, SGVB(PRS)={}, RSS-SGVB={}, RSS-CAVI={}  (of {})",
        cavi_recovered, sgvb_recovered, rss_recovered, rss_cavi_recovered, local_causal_set.len(),
    );

    assert!(
        cavi_recovered >= 1,
        "CAVI should recover ≥1 causal SNP in top-5, got {}",
        cavi_recovered,
    );
    assert!(
        sgvb_recovered >= 1,
        "SGVB should recover ≥1 causal SNP in top-5, got {}",
        sgvb_recovered,
    );

    Ok(())
}

#[test]
fn test_ridge_vs_truncation_prs() -> Result<()> {
    let n = 200;
    let p = 50;
    let mut rng = StdRng::seed_from_u64(99);
    let x = generate_x_with_ld(n, p, 10, &mut rng);
    let z = DMatrix::<f32>::rnorm(p, 1);

    let yhat_ridge = compute_block_polygenic_scores_ridge(&x, &z, 0.1)?;
    let yhat_trunc =
        fagioli::summary_stats::polygenic_score::compute_block_polygenic_scores(&x, &z)?;

    assert_eq!(yhat_ridge.nrows(), n);
    assert_eq!(yhat_ridge.ncols(), 1);

    // Correlated but not identical
    let corr = correlation(&yhat_ridge.column(0), &yhat_trunc.column(0));
    println!("Ridge vs truncation PRS correlation: {:.4}", corr);
    assert!(corr > 0.5);

    // Ridge with lambda≈0 should match truncation
    let yhat_ridge_0 = compute_block_polygenic_scores_ridge(&x, &z, 1e-8)?;
    let corr_0 = correlation(&yhat_ridge_0.column(0), &yhat_trunc.column(0));
    println!("Ridge(λ≈0) vs truncation correlation: {:.4}", corr_0);
    assert!(corr_0 > 0.95);

    Ok(())
}

/// n < p regime: more SNPs than individuals.
///
/// Ridge regularization is essential here — without it, D has many near-zero
/// singular values and the PRS explodes. Tests both CAVI and SGVB.
#[test]
fn test_prs_susie_n_less_than_p() -> Result<()> {
    let n = 100;
    let p = 500;
    let t = 1;
    let block_size = 50;
    let h2 = 0.4f32;
    let num_shared_causal = 2;

    let mut rng = StdRng::seed_from_u64(77);
    let x = generate_x_with_ld(n, p, block_size, &mut rng);
    let blocks = make_blocks(p, block_size);

    // Causal block 2 (SNPs 100-149)
    let causal_block_idx = 2;
    let causal_block = &blocks[causal_block_idx];
    let block_m = causal_block.num_snps();

    let effects = sample_cell_type_genetic_effects(block_m, t, num_shared_causal, 0, h2, 77)?;

    println!("\n{:=<60}", "");
    println!("  n < p test (n={}, p={}, T={}, h²={})", n, p, t, h2,);
    println!("{:=<60}", "");

    let causal_global: Vec<usize> = effects
        .shared_causal_indices
        .iter()
        .map(|&local| causal_block.snp_start + local)
        .collect();
    println!("  Causal SNPs (global): {:?}", causal_global);

    // Genetic values
    let x_block = x.columns(causal_block.snp_start, block_m).clone_owned();
    let mut g = DMatrix::zeros(n, t);
    for (j, &snp_idx) in effects.shared_causal_indices.iter().enumerate() {
        for i in 0..n {
            g[(i, 0)] += x_block[(i, snp_idx)] * effects.shared_effect_sizes[(0, j)];
        }
    }

    g.scale_columns_inplace();
    g *= h2.sqrt();
    let mut noise = DMatrix::<f32>::rnorm(n, t);
    noise.scale_columns_inplace();
    noise *= (1.0 - h2).sqrt();
    let y = &g + &noise;

    let z = compute_zscores(&x, &y);
    for &c in &causal_global {
        println!("  z[{}] = {:.3}", c, z[(c, 0)]);
    }

    // Ridge PRS — lambda matters more when n < p
    let lambda = 0.5f32;
    let yhat = compute_all_polygenic_scores_ridge(&x, &z, &blocks, lambda)?;
    let corr = correlation(&yhat.column(0), &y.column(0));
    println!("\n  Ridge PRS-y correlation (λ={}): {:.4}", lambda, corr);

    let device = Device::Cpu;
    let local_causal_set: HashSet<usize> = effects.shared_causal_indices.iter().copied().collect();

    // ── CAVI ────────────────────────────────────────────────────────────
    let x_block_tensor = x_block.to_tensor(&device)?;
    let yhat_col: Vec<f32> = (0..n).map(|i| yhat[(i, 0)]).collect();
    let yhat_tensor = candle_util::candle_core::Tensor::from_slice(&yhat_col, (n,), &device)?;

    let cavi_params = CaviSusieParams {
        num_components: 10,
        max_iter: 200,
        tol: 1e-6,
        prior_variance: 0.5,
        estimate_residual_variance: true,
        prior_weights: None,
    };
    let cavi_result = cavi_susie(&x_block_tensor, &yhat_tensor, &cavi_params)?;
    let pip_cavi: Vec<f32> = cavi_result.pip.iter().map(|&v| v as f32).collect();

    println!("\n  CAVI top PIPs (n<p):");
    print_top_pips(&pip_cavi, &local_causal_set, 10);

    let top5_cavi = top_k_indices(&pip_cavi, 5);
    let cavi_recovered = local_causal_set
        .iter()
        .filter(|c| top5_cavi.contains(c))
        .count();

    // ── SGVB ────────────────────────────────────────────────────────────
    let y_block_mat = DMatrix::from_fn(n, 1, |i, _| yhat[(i, 0)]);
    let sgvb_config = FitConfig {
        model_type: ModelType::Susie,
        prior_type: fagioli::sgvb::PriorType::Single,
        num_components: 10,
        num_sgvb_samples: 20,
        learning_rate: 0.01,
        num_iterations: 500,
        batch_size: 1000,
        prior_vars: vec![0.5],
        elbo_window: 50,
        seed: 77,
        sigma2_inf: 0.0,
        prior_alpha: 1.0,
    };
    let sgvb_result: BlockFitResult =
        fit_block(&x_block, &y_block_mat, None, &sgvb_config, &device)?;
    let pip_sgvb: Vec<f32> = (0..block_m).map(|j| sgvb_result.pip[(j, 0)]).collect();

    println!("\n  SGVB top PIPs (n<p):");
    print_top_pips(&pip_sgvb, &local_causal_set, 10);

    let top5_sgvb = top_k_indices(&pip_sgvb, 5);
    let sgvb_recovered = local_causal_set
        .iter()
        .filter(|c| top5_sgvb.contains(c))
        .count();

    // ── RSS SGVB ─────────────────────────────────────────────────────────
    let mut x_block_std = x_block.clone();
    x_block_std.scale_columns_inplace();
    let z_block = z.rows(causal_block.snp_start, block_m).clone_owned();
    let pip_rss = run_rss_sgvb(&x_block_std, &z_block, 10, &[0.5], 0.0, 77, &device)?;

    println!("\n  RSS SGVB top PIPs (n<p):");
    print_top_pips(&pip_rss, &local_causal_set, 10);

    let top5_rss = top_k_indices(&pip_rss, 5);
    let rss_recovered = local_causal_set
        .iter()
        .filter(|c| top5_rss.contains(c))
        .count();

    // ── RSS CAVI ─────────────────────────────────────────────────────────
    let pip_rss_cavi = run_rss_cavi(&x_block_std, &z_block, 10, 0.5, &device)?;

    println!("\n  RSS CAVI top PIPs (n<p):");
    print_top_pips(&pip_rss_cavi, &local_causal_set, 10);

    let top5_rss_cavi = top_k_indices(&pip_rss_cavi, 5);
    let rss_cavi_recovered = local_causal_set
        .iter()
        .filter(|c| top5_rss_cavi.contains(c))
        .count();

    println!(
        "\n  Causal in top-5: CAVI(PRS)={}, SGVB(PRS)={}, RSS-SGVB={}, RSS-CAVI={}  (of {})",
        cavi_recovered,
        sgvb_recovered,
        rss_recovered,
        rss_cavi_recovered,
        local_causal_set.len(),
    );

    assert!(
        cavi_recovered >= 1,
        "CAVI n<p: should recover ≥1 causal in top-5, got {}",
        cavi_recovered,
    );

    Ok(())
}

/// Polygenic background: sparse causal + dense infinitesimal effects.
///
/// h2_sparse=0.2 (2 causal SNPs) + h2_poly=0.3 (all SNPs contribute).
/// The polygenic signal acts as correlated noise for SuSiE fine-mapping.
/// This is the hard case — tests whether the sparse signal survives.
#[test]
fn test_prs_susie_with_polygenic_background() -> Result<()> {
    let n = 500;
    let p = 200;
    let t = 1;
    let block_size = 20;
    let h2_sparse = 0.2f32;
    let h2_poly = 0.3f32;
    let num_shared_causal = 2;

    let mut rng = StdRng::seed_from_u64(123);
    let x = generate_x_with_ld(n, p, block_size, &mut rng);
    let blocks = make_blocks(p, block_size);

    let causal_block_idx = 4;
    let causal_block = &blocks[causal_block_idx];
    let block_m = causal_block.num_snps();

    let effects =
        sample_cell_type_genetic_effects(block_m, t, num_shared_causal, 0, h2_sparse, 123)?;

    println!("\n{:=<60}", "");
    println!(
        "  Polygenic background (n={}, p={}, h2_sparse={}, h2_poly={})",
        n, p, h2_sparse, h2_poly,
    );
    println!("{:=<60}", "");

    let causal_global: Vec<usize> = effects
        .shared_causal_indices
        .iter()
        .map(|&local| causal_block.snp_start + local)
        .collect();
    println!("  Causal SNPs (global): {:?}", causal_global);

    // Sparse genetic values
    let x_block = x.columns(causal_block.snp_start, block_m).clone_owned();
    let mut g_sparse = DMatrix::zeros(n, t);
    for (j, &snp_idx) in effects.shared_causal_indices.iter().enumerate() {
        for i in 0..n {
            g_sparse[(i, 0)] += x_block[(i, snp_idx)] * effects.shared_effect_sizes[(0, j)];
        }
    }
    g_sparse.scale_columns_inplace();
    g_sparse *= h2_sparse.sqrt();

    // Polygenic effects: beta_j ~ N(0, 1/p) for all SNPs
    let normal_poly = Normal::new(0.0f32, (1.0 / p as f32).sqrt()).unwrap();
    let beta_poly = DMatrix::from_fn(p, t, |_, _| normal_poly.sample(&mut rng));
    let mut g_poly = &x * &beta_poly;
    g_poly.scale_columns_inplace();
    g_poly *= h2_poly.sqrt();

    // Noise
    let pve_noise = (1.0 - h2_sparse - h2_poly).max(0.0);
    let mut noise = DMatrix::<f32>::rnorm(n, t);
    noise.scale_columns_inplace();
    noise *= pve_noise.sqrt();

    let y = &g_sparse + &g_poly + &noise;

    let z = compute_zscores(&x, &y);
    for &c in &causal_global {
        println!("  z[{}] = {:.3}", c, z[(c, 0)]);
    }

    let lambda = 0.1f32;
    let yhat = compute_all_polygenic_scores_ridge(&x, &z, &blocks, lambda)?;
    let corr = correlation(&yhat.column(0), &y.column(0));
    println!(
        "\n  Ridge PRS-y correlation: {:.4} (h2_total={})",
        corr,
        h2_sparse + h2_poly,
    );

    let device = Device::Cpu;
    let local_causal_set: HashSet<usize> = effects.shared_causal_indices.iter().copied().collect();

    // ── CAVI (no polygenic correction) ──────────────────────────────────
    let x_block_tensor = x_block.to_tensor(&device)?;
    let yhat_col: Vec<f32> = (0..n).map(|i| yhat[(i, 0)]).collect();
    let yhat_tensor = candle_util::candle_core::Tensor::from_slice(&yhat_col, (n,), &device)?;

    let cavi_params = CaviSusieParams {
        num_components: 10,
        max_iter: 200,
        tol: 1e-6,
        prior_variance: 0.5,
        estimate_residual_variance: true,
        prior_weights: None,
    };
    let cavi_result = cavi_susie(&x_block_tensor, &yhat_tensor, &cavi_params)?;
    let pip_cavi: Vec<f32> = cavi_result.pip.iter().map(|&v| v as f32).collect();

    println!("\n  CAVI top PIPs (with polygenic background):");
    print_top_pips(&pip_cavi, &local_causal_set, 10);

    let top10_cavi = top_k_indices(&pip_cavi, 10);
    let cavi_recovered = local_causal_set
        .iter()
        .filter(|c| top10_cavi.contains(c))
        .count();

    // ── SGVB (no sigma2_inf) ────────────────────────────────────────────
    let y_block_mat = DMatrix::from_fn(n, 1, |i, _| yhat[(i, 0)]);
    let sgvb_config = FitConfig {
        model_type: ModelType::Susie,
        prior_type: fagioli::sgvb::PriorType::Single,
        num_components: 10,
        num_sgvb_samples: 20,
        learning_rate: 0.01,
        num_iterations: 500,
        batch_size: 1000,
        prior_vars: vec![0.5],
        elbo_window: 50,
        seed: 123,
        sigma2_inf: 0.0,
        prior_alpha: 1.0,
    };
    let sgvb_result: BlockFitResult =
        fit_block(&x_block, &y_block_mat, None, &sgvb_config, &device)?;
    let pip_sgvb: Vec<f32> = (0..block_m).map(|j| sgvb_result.pip[(j, 0)]).collect();

    println!("\n  SGVB top PIPs (no sigma2_inf):");
    print_top_pips(&pip_sgvb, &local_causal_set, 10);

    let top10_sgvb = top_k_indices(&pip_sgvb, 10);
    let sgvb_recovered = local_causal_set
        .iter()
        .filter(|c| top10_sgvb.contains(c))
        .count();

    // ── SGVB (with sigma2_inf for polygenic correction) ─────────────────
    let sgvb_config_inf = FitConfig {
        sigma2_inf: 1.0,
        seed: 124,
        ..sgvb_config.clone()
    };
    let sgvb_inf_result: BlockFitResult =
        fit_block(&x_block, &y_block_mat, None, &sgvb_config_inf, &device)?;
    let pip_sgvb_inf: Vec<f32> = (0..block_m).map(|j| sgvb_inf_result.pip[(j, 0)]).collect();

    println!("\n  SGVB top PIPs (with sigma2_inf=1.0):");
    print_top_pips(&pip_sgvb_inf, &local_causal_set, 10);

    let top10_sgvb_inf = top_k_indices(&pip_sgvb_inf, 10);
    let sgvb_inf_recovered = local_causal_set
        .iter()
        .filter(|c| top10_sgvb_inf.contains(c))
        .count();

    // ── RSS SGVB (no inf) ─────────────────────────────────────────────────
    let mut x_block_std = x_block.clone();
    x_block_std.scale_columns_inplace();
    let z_block = z.rows(causal_block.snp_start, block_m).clone_owned();
    let pip_rss = run_rss_sgvb(&x_block_std, &z_block, 10, &[0.5], 0.0, 123, &device)?;

    println!("\n  RSS SGVB top PIPs (no inf):");
    print_top_pips(&pip_rss, &local_causal_set, 10);

    let top10_rss = top_k_indices(&pip_rss, 10);
    let rss_recovered = local_causal_set
        .iter()
        .filter(|c| top10_rss.contains(c))
        .count();

    // ── RSS SGVB (with sigma2_inf) ──────────────────────────────────────
    let pip_rss_inf = run_rss_sgvb(&x_block_std, &z_block, 10, &[0.5], 1.0, 124, &device)?;

    println!("\n  RSS SGVB top PIPs (sigma2_inf=1.0):");
    print_top_pips(&pip_rss_inf, &local_causal_set, 10);

    let top10_rss_inf = top_k_indices(&pip_rss_inf, 10);
    let rss_inf_recovered = local_causal_set
        .iter()
        .filter(|c| top10_rss_inf.contains(c))
        .count();

    // ── RSS CAVI ─────────────────────────────────────────────────────────
    let pip_rss_cavi = run_rss_cavi(&x_block_std, &z_block, 10, 0.5, &device)?;

    println!("\n  RSS CAVI top PIPs (polygenic):");
    print_top_pips(&pip_rss_cavi, &local_causal_set, 10);

    let top10_rss_cavi = top_k_indices(&pip_rss_cavi, 10);
    let rss_cavi_recovered = local_causal_set
        .iter()
        .filter(|c| top10_rss_cavi.contains(c))
        .count();

    println!(
        "\n  Causal in top-10: CAVI(PRS)={}, SGVB(PRS)={}, SGVB+inf(PRS)={}, RSS-SGVB={}, RSS+inf={}, RSS-CAVI={}  (of {})",
        cavi_recovered, sgvb_recovered, sgvb_inf_recovered,
        rss_recovered, rss_inf_recovered, rss_cavi_recovered, local_causal_set.len(),
    );

    let best = cavi_recovered
        .max(sgvb_recovered)
        .max(sgvb_inf_recovered)
        .max(rss_recovered)
        .max(rss_inf_recovered)
        .max(rss_cavi_recovered);
    assert!(
        best >= 1,
        "At least one method should recover ≥1 causal in top-10 (polygenic)",
    );

    Ok(())
}

/// Run RSS CAVI SuSiE: project into eigenspace, then run CAVI on (X_tilde, y_tilde).
fn run_rss_cavi(
    x_block: &DMatrix<f32>,
    z_block: &DMatrix<f32>,
    num_components: usize,
    prior_variance: f64,
    device: &Device,
) -> Result<Vec<f32>> {
    use candle_util::sgvb::cavi_susie::{cavi_susie, CaviSusieParams};
    use fagioli::summary_stats::rss_svd::RssSvdNal;

    let n = x_block.nrows();
    let lambda = 0.1 / n as f64;

    let svd = RssSvdNal::from_genotypes(x_block, n, lambda)?;
    let y_tilde = svd.project_zscores(z_block); // (K, T)
    let x_tilde = svd.x_design(); // (K, p)

    // CAVI expects (n_obs, p) design and (n_obs,) response
    let x_tensor = x_tilde.to_tensor(device)?;
    let y_col: Vec<f32> = (0..y_tilde.nrows()).map(|i| y_tilde[(i, 0)]).collect();
    let y_tensor =
        candle_util::candle_core::Tensor::from_slice(&y_col, (y_tilde.nrows(),), device)?;

    let params = CaviSusieParams {
        num_components,
        max_iter: 200,
        tol: 1e-6,
        prior_variance,
        estimate_residual_variance: false, // fixed at 1.0 for RSS
        prior_weights: None,
    };

    let result = cavi_susie(&x_tensor, &y_tensor, &params)?;
    Ok(result.pip.iter().map(|&v| v as f32).collect())
}

/// Run RSS SGVB SuSiE on a standardized block + z-scores, return per-SNP PIPs.
fn run_rss_sgvb(
    x_block: &DMatrix<f32>,
    z_block: &DMatrix<f32>,
    num_components: usize,
    prior_vars: &[f32],
    sigma2_inf: f32,
    seed: u64,
    device: &Device,
) -> Result<Vec<f32>> {
    let p = x_block.ncols();
    let n = x_block.nrows();
    let config = FitConfig {
        model_type: ModelType::Susie,
        prior_type: fagioli::sgvb::PriorType::Single,
        num_components,
        num_sgvb_samples: 20,
        learning_rate: 0.01,
        num_iterations: 1000,
        batch_size: 1000,
        prior_vars: prior_vars.to_vec(),
        elbo_window: 50,
        seed,
        sigma2_inf,
        prior_alpha: 1.0,
    };
    let rss_params = RssParams {
        max_rank: n,
        lambda: 0.1 / n as f64,
        ldsc_intercept: false,
    };
    let result = fit_block_rss(x_block, z_block, &config, &rss_params, device)?;
    let best = result.best_result();
    Ok((0..p).map(|j| best.pip[(j, 0)]).collect())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn correlation(a: &nalgebra::DVectorView<f32>, b: &nalgebra::DVectorView<f32>) -> f32 {
    let n = a.len() as f32;
    let mean_a = a.sum() / n;
    let mean_b = b.sum() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for i in 0..a.len() {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    cov / (var_a * var_b).sqrt()
}

fn top_k_indices(pip: &[f32], k: usize) -> Vec<usize> {
    let mut sorted: Vec<(usize, f32)> = pip.iter().enumerate().map(|(j, &v)| (j, v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sorted.iter().take(k).map(|&(j, _)| j).collect()
}

fn print_top_pips(pip: &[f32], causal_set: &HashSet<usize>, k: usize) {
    let mut sorted: Vec<(usize, f32)> = pip.iter().enumerate().map(|(j, &v)| (j, v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(j, v) in sorted.iter().take(k) {
        let marker = if causal_set.contains(&j) {
            " *CAUSAL*"
        } else {
            ""
        };
        println!("    SNP {:>3}: PIP={:.4}{}", j, v, marker);
    }
}
