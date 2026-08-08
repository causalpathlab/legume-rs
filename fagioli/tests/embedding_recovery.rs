//! End-to-end: does the embedding recover planted structure from *simulated
//! GWAS summary statistics*, with real LD and the real whitening pipeline?
//!
//! The unit tests in `src/embedding/` generate data from the model's own law in
//! already-whitened coordinates. That tests the optimiser, not the premise.
//! Here nothing is whitened by hand:
//!
//! 1. genotypes with block LD, so `R` is far from the identity;
//! 2. a planted low-rank effect matrix `B = U V'` with sparse `U`;
//! 3. phenotypes `y_t = X β_t + ε` at a chosen `h²`;
//! 4. **marginal OLS z-scores**, exactly as a GWAS would report them;
//! 5. the real pipeline — eigenspace calibration, λ selection, whitening, NCE.
//!
//! Only the effect structure is planted. LD, the rSVD truncation, the ridge and
//! the calibration all have to work for the answer to come out right.
//!
//! `sim-sumstat` is deliberately *not* used as the generator. Its "shared
//! causal" variants draw effects independently per trait at shared loci
//! (`simulation/cell_type_effects.rs`), so `E[Cov_g] = 0`: it produces shared
//! loci, not shared genetics, and cannot make a nonzero genetic correlation to
//! recover.
//!
//! Run: cargo test -p fagioli --test embedding_recovery -- --nocapture

use anyhow::Result;
use candle_util::candle_core::Device;
use fagioli::embedding::diagnostics::to_correlation;
use fagioli::embedding::model::{EmbedConfig, UPrior};
use fagioli::embedding::noise::NoiseModel;
use fagioli::embedding::train::train;
use fagioli::embedding::whiten::whiten_blocks;
use fagioli::genotype::GenotypeMatrix;
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::{decompose_blocks, SumstatInput};
use fagioli::summary_stats::LdBlock;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};

const NUM_BLOCKS: usize = 8;
const SNPS_PER_BLOCK: usize = 120;
const NUM_INDIVIDUALS: usize = 800;
const NUM_TRAITS: usize = 12;
const NUM_PROGRAMS: usize = 3;
const CAUSAL_PER_PROGRAM: usize = 4;

/// Genotypes with block-correlated LD: within a block, SNPs load on a shared
/// latent haplotype, so `R` has genuine off-diagonal structure.
fn simulate_genotypes(rng: &mut SmallRng) -> DMatrix<f32> {
    let m = NUM_BLOCKS * SNPS_PER_BLOCK;
    let mut x = DMatrix::<f32>::zeros(NUM_INDIVIDUALS, m);

    for b in 0..NUM_BLOCKS {
        // A handful of latent haplotypes per block.
        let n_hap = 6;
        let hap = DMatrix::from_fn(NUM_INDIVIDUALS, n_hap, |_, _| {
            let v: f64 = StandardNormal.sample(rng);
            v as f32
        });
        for j in 0..SNPS_PER_BLOCK {
            let which = j % n_hap;
            // Decaying correlation with the anchor haplotype along the block.
            let rho = 0.85 - 0.4 * (j as f32 / SNPS_PER_BLOCK as f32);
            let col = b * SNPS_PER_BLOCK + j;
            for i in 0..NUM_INDIVIDUALS {
                let e: f64 = StandardNormal.sample(rng);
                let latent = rho * hap[(i, which)] + (1.0 - rho * rho).sqrt() * e as f32;
                // Coarse dosage, so this looks like genotypes rather than a
                // Gaussian: threshold into {0,1,2}.
                x[(i, col)] = if latent < -0.6 {
                    0.0
                } else if latent < 0.6 {
                    1.0
                } else {
                    2.0
                };
            }
        }
    }
    x
}

fn uniform_blocks() -> Vec<LdBlock> {
    (0..NUM_BLOCKS)
        .map(|b| LdBlock {
            block_idx: b,
            snp_start: b * SNPS_PER_BLOCK,
            snp_end: (b + 1) * SNPS_PER_BLOCK,
            chr: Box::from("chr1"),
            bp_start: (b * SNPS_PER_BLOCK * 1000) as u64,
            bp_end: ((b + 1) * SNPS_PER_BLOCK * 1000) as u64,
        })
        .collect()
}

fn dummy_geno(x: DMatrix<f32>) -> GenotypeMatrix {
    let m = x.ncols();
    GenotypeMatrix {
        individual_ids: (0..x.nrows()).map(|i| Box::from(format!("ind{i}"))).collect(),
        snp_ids: (0..m).map(|j| Box::from(format!("rs{j}"))).collect(),
        chromosomes: vec![Box::from("chr1"); m],
        positions: (0..m).map(|j| (j * 1000) as u64).collect(),
        allele1: vec![Box::from("A"); m],
        allele2: vec![Box::from("G"); m],
        genotypes: x,
    }
}

struct Planted {
    input: SumstatInput,
    /// True `V U'U V'`, the invariant geometry the fit should reproduce.
    geometry: DMatrix<f32>,
}

/// Plant `B = U V'`, simulate phenotypes at heritability `h2`, and report
/// marginal OLS z-scores. `h2 = 0` gives a pure null.
fn plant(h2: f32, seed: u64) -> Planted {
    let mut rng = SmallRng::seed_from_u64(seed);
    let x_raw = simulate_genotypes(&mut rng);
    let m = x_raw.ncols();

    // Standardized genotypes drive the phenotype, so effects are per-SD.
    let mut x = x_raw.clone();
    x.scale_columns_inplace();

    // Sparse variant loadings, dense trait loadings.
    let mut u = DMatrix::<f32>::zeros(m, NUM_PROGRAMS);
    for prog in 0..NUM_PROGRAMS {
        for j in rand::seq::index::sample(&mut rng, m, CAUSAL_PER_PROGRAM) {
            let v: f64 = StandardNormal.sample(&mut rng);
            u[(j, prog)] = v as f32;
        }
    }
    let v_true = DMatrix::from_fn(NUM_TRAITS, NUM_PROGRAMS, |_, _| {
        let val: f64 = StandardNormal.sample(&mut rng);
        val as f32
    });
    let b_true = &u * v_true.transpose(); // (M, T)

    // y_t = √h² · standardize(X β_t) + √(1-h²) · ε
    let g = &x * &b_true; // (N, T)
    let mut y = DMatrix::<f32>::zeros(NUM_INDIVIDUALS, NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let gt = g.column(t);
        let sd = (gt.iter().map(|v| v * v).sum::<f32>() / NUM_INDIVIDUALS as f32).sqrt();
        for i in 0..NUM_INDIVIDUALS {
            let e: f64 = StandardNormal.sample(&mut rng);
            let genetic = if sd > 0.0 && h2 > 0.0 {
                h2.sqrt() * gt[i] / sd
            } else {
                0.0
            };
            y[(i, t)] = genetic + (1.0 - h2).max(0.0).sqrt() * e as f32;
        }
    }

    // Marginal OLS z-scores, as a GWAS reports them: z_j = x_j'y / (sd · √n).
    let n = NUM_INDIVIDUALS as f32;
    let mut z = DMatrix::<f32>::zeros(m, NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let yt = y.column(t);
        let y_sd = (yt.iter().map(|v| v * v).sum::<f32>() / n).sqrt().max(1e-8);
        for j in 0..m {
            z[(j, t)] = x.column(j).dot(&yt) / (y_sd * n.sqrt());
        }
    }

    let utu = u.transpose() * &u;
    Planted {
        geometry: &v_true * utu * v_true.transpose(),
        input: SumstatInput {
            geno: dummy_geno(x_raw),
            zscores: z,
            blocks: uniform_blocks(),
            median_n: NUM_INDIVIDUALS as u64,
            max_rank: 80,
        },
    }
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

/// Run calibration + whitening + NCE, returning (geometry correlation, λ,
/// whiteness deviation, NCE offset).
fn run_pipeline(planted: &Planted, label: &str) -> Result<(f32, f64, f32, f32)> {
    let bases = decompose_blocks(&planted.input);
    let report =
        calibrate_input(&planted.input, &bases).expect("calibration should succeed");
    let lambda = report.noise.lambda_white();

    println!(
        "  [{label}] calibration: c={:.3}, τ={:.4} (λ_white, SE {:.4}), whiteness dev {:.3}{}",
        report.noise.c,
        report.noise.tau,
        report.lambda_se,
        report.deviation,
        if report.misspecified {
            "  <-- MISSPECIFIED"
        } else {
            ""
        },
    );

    // Ω = I: the traits here share one cohort by construction, but with no
    // cohort-specific noise, so overlap is the identity.
    let blocks = whiten_blocks(&planted.input, bases, None, lambda)?;
    let d_sq: Vec<Vec<f32>> = blocks.iter().map(|b| b.d_sq.clone()).collect();
    let noise = NoiseModel::new(&d_sq, report.noise.c, report.noise.tau, lambda);

    let cfg = EmbedConfig {
        embedding_dim: NUM_PROGRAMS,
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
        seed: 7,
    };
    let fit = train(&blocks, &noise, &cfg, &Device::Cpu)?;

    let corr = offdiag_corr(
        &to_correlation(&fit.trait_geometry),
        &to_correlation(&planted.geometry),
    );
    println!(
        "  [{label}] fit: geometry corr {:.3}, loss {:.4} -> {:.4}, offset {:+.3}",
        corr,
        fit.loss_trace.first().copied().unwrap_or(0.0),
        fit.loss_trace.last().copied().unwrap_or(0.0),
        fit.offset,
    );

    Ok((corr, lambda, report.deviation, fit.offset))
}

/// The headline: with real LD and real z-scores, does the fit recover the
/// planted trait geometry, and does it stay quiet on a null?
#[test]
fn test_embedding_recovers_planted_geometry_from_simulated_gwas() -> Result<()> {
    println!("\n=== signal (h² = 0.5) ===");
    let signal = plant(0.5, 20250807);
    let (corr_signal, lambda_s, dev_s, _) = run_pipeline(&signal, "signal")?;

    println!("\n=== null (h² = 0) ===");
    let null = plant(0.0, 20250807);
    let (corr_null, lambda_n, dev_n, _) = run_pipeline(&null, "null")?;

    println!(
        "\nsummary: signal corr {:.3} (λ {:.4}, dev {:.3}) vs null corr {:.3} (λ {:.4}, dev {:.3})\n",
        corr_signal, lambda_s, dev_s, corr_null, lambda_n, dev_n
    );

    assert!(
        corr_signal > corr_null.abs(),
        "signal should beat null: {corr_signal} vs {corr_null}"
    );
    assert!(
        corr_signal > 0.4,
        "planted geometry should be recovered from real z-scores, got {corr_signal}"
    );
    Ok(())
}

/// Recovery should strengthen with heritability. A monotone response over a
/// sweep is far harder to obtain by accident than one passing threshold.
#[test]
fn test_recovery_improves_with_heritability() -> Result<()> {
    let mut corrs = Vec::new();
    for &h2 in &[0.0f32, 0.2, 0.5, 0.8] {
        println!("\n=== h² = {h2} ===");
        let planted = plant(h2, 90210);
        let (corr, _, _, _) = run_pipeline(&planted, &format!("h2={h2}"))?;
        corrs.push(corr);
    }
    println!("\nrecovery vs h²: {corrs:?}\n");

    assert!(
        corrs[3] > corrs[0].abs(),
        "high h² should beat the null: {corrs:?}"
    );
    Ok(())
}
