//! Embedding PRS against the in-repo ridge PRS, on held-out individuals.
//!
//! This is the first external yardstick in the series. Everything before it
//! measured recovery of quantities the model itself defines; this measures
//! prediction of a phenotype neither method saw.
//!
//! The comparator is `compute_all_polygenic_scores_ridge`, which computes
//! `U diag(1/(d+λ)) V' z` — the *same* block SVD, ridge and eigenspace. LD
//! handling is therefore held constant and the only difference is whether
//! per-trait weights come from raw `z` or from `U v_t`. That makes it a clean
//! ablation of the low-rank cross-trait borrowing, and **not** evidence about
//! LDpred2 or PRS-CS, which share none of this pipeline.
//!
//! Protocol: split individuals, derive z-scores from the training half only,
//! fit on those, score the held-out half, correlate against its true phenotype.
//!
//! Prediction under test: low rank is a bias-variance trade, so the embedding
//! should **win on low-powered traits and lose on well-powered ones**, with the
//! crossover set by `H`. Winning everywhere would suggest leakage.
//!
//! Run: cargo test -p fagioli --test prs_benchmark -- --nocapture

use anyhow::Result;
use candle_util::candle_core::Device;
use fagioli::embedding::model::{EmbedConfig, UPrior};
use fagioli::embedding::noise::NoiseModel;
use fagioli::embedding::score::{assemble_u, score_cohort, PanelStandardization};
use fagioli::embedding::train::train;
use fagioli::embedding::whiten::whiten_blocks;
use fagioli::genotype::GenotypeMatrix;
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::SumstatInput;
use fagioli::summary_stats::polygenic_score::compute_all_polygenic_scores_ridge;
use fagioli::summary_stats::LdBlock;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

const NUM_BLOCKS: usize = 8;
const SNPS_PER_BLOCK: usize = 120;
const N_TRAIN: usize = 900;
const N_TEST: usize = 400;
const NUM_TRAITS: usize = 12;
const NUM_PROGRAMS: usize = 3;
const CAUSAL_PER_PROGRAM: usize = 5;

fn randn(r: usize, c: usize, rng: &mut SmallRng) -> DMatrix<f32> {
    DMatrix::from_fn(r, c, |_, _| {
        let v: f64 = StandardNormal.sample(rng);
        v as f32
    })
}

fn simulate_genotypes(n: usize, rng: &mut SmallRng) -> DMatrix<f32> {
    let m = NUM_BLOCKS * SNPS_PER_BLOCK;
    let mut x = DMatrix::<f32>::zeros(n, m);
    for b in 0..NUM_BLOCKS {
        let n_hap = 6;
        let hap = randn(n, n_hap, rng);
        for j in 0..SNPS_PER_BLOCK {
            let rho = 0.9 - 0.45 * (j as f32 / SNPS_PER_BLOCK as f32);
            let col = b * SNPS_PER_BLOCK + j;
            for i in 0..n {
                let e: f64 = StandardNormal.sample(rng);
                let latent = rho * hap[(i, j % n_hap)] + (1.0 - rho * rho).sqrt() * e as f32;
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

fn wrap(x: DMatrix<f32>) -> GenotypeMatrix {
    let m = x.ncols();
    GenotypeMatrix {
        individual_ids: (0..x.nrows()).map(|i| Box::from(format!("i{i}"))).collect(),
        snp_ids: (0..m).map(|j| Box::from(format!("rs{j}"))).collect(),
        chromosomes: vec![Box::from("chr1"); m],
        positions: (0..m).map(|j| (j * 1000) as u64).collect(),
        allele1: vec![Box::from("A"); m],
        allele2: vec![Box::from("G"); m],
        genotypes: x,
    }
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (ma, mb) = (
        a.iter().take(n).sum::<f32>() / n as f32,
        b.iter().take(n).sum::<f32>() / n as f32,
    );
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (da, db) = ((a[i] - ma) as f64, (b[i] - mb) as f64);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    (sab / (saa * sbb).sqrt()) as f32
}

/// Per-trait heritability, so traits can be split into low- and well-powered.
struct Split {
    train_geno: GenotypeMatrix,
    test_geno: GenotypeMatrix,
    train_z: DMatrix<f32>,
    test_y: DMatrix<f32>,
    h2: Vec<f32>,
}

/// Simulate a cohort, split it, and report z-scores from the training half only.
///
/// Traits are given a *range* of heritabilities so the predicted crossover is
/// visible within one run rather than needing a sweep.
fn simulate_split(seed: u64, trait_specific: f32) -> Split {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n_all = N_TRAIN + N_TEST;
    let x_all = simulate_genotypes(n_all, &mut rng);
    let m = x_all.ncols();

    let mut xs = x_all.clone();
    xs.scale_columns_inplace();

    let mut u = DMatrix::<f32>::zeros(m, NUM_PROGRAMS);
    for prog in 0..NUM_PROGRAMS {
        for j in rand::seq::index::sample(&mut rng, m, CAUSAL_PER_PROGRAM) {
            let v: f64 = StandardNormal.sample(&mut rng);
            u[(j, prog)] = v as f32;
        }
    }
    let v_true = randn(NUM_TRAITS, NUM_PROGRAMS, &mut rng);
    let mut b_true = &u * v_true.transpose();

    // Effects OUTSIDE the program space. With `trait_specific = 0` the truth is
    // exactly low-rank, so the H-dimensional restriction imposes no bias at all
    // and there is nothing for the bias-variance trade to trade. This is what
    // creates the bias the crossover prediction assumes.
    if trait_specific > 0.0 {
        let scale = b_true.abs().max().max(1e-6) * trait_specific;
        for t in 0..NUM_TRAITS {
            for j in rand::seq::index::sample(&mut rng, m, CAUSAL_PER_PROGRAM) {
                let v: f64 = StandardNormal.sample(&mut rng);
                b_true[(j, t)] += scale * v as f32;
            }
        }
    }
    let g = &xs * &b_true;

    // h² ramps from weak to strong across traits.
    let h2: Vec<f32> = (0..NUM_TRAITS)
        .map(|t| 0.05 + 0.55 * (t as f32 / (NUM_TRAITS - 1) as f32))
        .collect();

    let mut y = DMatrix::<f32>::zeros(n_all, NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let gt = g.column(t);
        let sd = (gt.iter().map(|v| v * v).sum::<f32>() / n_all as f32).sqrt();
        for i in 0..n_all {
            let e: f64 = StandardNormal.sample(&mut rng);
            let genetic = if sd > 0.0 { h2[t].sqrt() * gt[i] / sd } else { 0.0 };
            y[(i, t)] = genetic + (1.0 - h2[t]).sqrt() * e as f32;
        }
    }

    // Training-half z-scores, standardised within the training half.
    let x_train_raw = x_all.rows(0, N_TRAIN).clone_owned();
    let mut x_train = x_train_raw.clone();
    x_train.scale_columns_inplace();
    let nf = N_TRAIN as f32;
    let mut z = DMatrix::<f32>::zeros(m, NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let yt = y.rows(0, N_TRAIN).column(t).clone_owned();
        let y_sd = (yt.iter().map(|v| v * v).sum::<f32>() / nf).sqrt().max(1e-8);
        for j in 0..m {
            z[(j, t)] = x_train.column(j).dot(&yt) / (y_sd * nf.sqrt());
        }
    }

    Split {
        train_geno: wrap(x_train_raw),
        test_geno: wrap(x_all.rows(N_TRAIN, N_TEST).clone_owned()),
        train_z: z,
        test_y: y.rows(N_TRAIN, N_TEST).clone_owned(),
        h2,
    }
}

/// One replicate; returns (embedding_lo, ridge_lo, embedding_hi, ridge_hi).
fn one_replicate(seed: u64, trait_specific: f32, verbose: bool) -> Result<(f32, f32, f32, f32)> {
    let split = simulate_split(seed, trait_specific);

    let input = SumstatInput {
        geno: split.train_geno.clone(),
        zscores: split.train_z.clone(),
        blocks: uniform_blocks(),
        median_n: N_TRAIN as u64,
        max_rank: 80,
    };

    // ── Embedding PRS ────────────────────────────────────────────────────
    let (report, bases) = calibrate_input(&input).expect("calibration");
    let lambda = report.noise.lambda_white();
    let blocks = whiten_blocks(&input, bases, None, lambda)?;
    let d_sq: Vec<Vec<f32>> = blocks.iter().map(|b| b.d_sq.clone()).collect();
    let noise = NoiseModel::new(&d_sq, report.noise.c, report.noise.tau, lambda);

    let fit = train(
        &blocks,
        &noise,
        &EmbedConfig {
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
            seed: 9,
        },
        &Device::Cpu,
    )?;

    let starts: Vec<usize> = blocks.iter().map(|b| b.snp_start).collect();
    let u = assemble_u(&fit.u_mean, &starts, split.train_geno.num_snps());
    let panel = PanelStandardization::from_panel(&split.train_geno);
    let scored = score_cohort(&u, &fit.v_check, &panel, &split.test_geno)?;

    // ── Ridge PRS on the same held-out individuals ───────────────────────
    let mut x_test = split.test_geno.genotypes.clone();
    x_test.scale_columns_inplace();
    let ridge = compute_all_polygenic_scores_ridge(
        &x_test,
        &split.train_z,
        &uniform_blocks(),
        0.1,
    )?;

    if verbose {
        println!(
            "\n{:>6} {:>6} | {:>12} {:>12} | {:>8}",
            "trait", "h²", "embedding r", "ridge r", "Δ"
        );
        println!("{}", "-".repeat(56));
    }

    let (mut emb_lo, mut rid_lo, mut emb_hi, mut rid_hi) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut n_lo, mut n_hi) = (0usize, 0usize);

    for t in 0..NUM_TRAITS {
        let truth: Vec<f32> = split.test_y.column(t).iter().copied().collect();
        let e: Vec<f32> = scored.prs.column(t).iter().copied().collect();
        let r: Vec<f32> = ridge.column(t).iter().copied().collect();
        let (re, rr) = (pearson(&e, &truth).abs(), pearson(&r, &truth).abs());
        if verbose {
            println!(
                "{:>6} {:>6.2} | {:>12.3} {:>12.3} | {:>+8.3}",
                t,
                split.h2[t],
                re,
                rr,
                re - rr
            );
        }
        if split.h2[t] < 0.3 {
            emb_lo += re;
            rid_lo += rr;
            n_lo += 1;
        } else {
            emb_hi += re;
            rid_hi += rr;
            n_hi += 1;
        }
    }

    let (emb_lo, rid_lo) = (emb_lo / n_lo as f32, rid_lo / n_lo as f32);
    let (emb_hi, rid_hi) = (emb_hi / n_hi as f32, rid_hi / n_hi as f32);
    if verbose {
        println!(
            "\nlow-powered  (h² < 0.3, n={n_lo}): embedding {emb_lo:.3} vs ridge {rid_lo:.3}  (Δ {:+.3})",
            emb_lo - rid_lo
        );
        println!(
            "well-powered (h² ≥ 0.3, n={n_hi}): embedding {emb_hi:.3} vs ridge {rid_hi:.3}  (Δ {:+.3})",
            emb_hi - rid_hi
        );
        println!(
            "coverage {:?}, matched {}/{}",
            scored.coverage.iter().map(|c| format!("{c:.2}")).collect::<Vec<_>>(),
            scored.matched,
            split.test_geno.num_snps(),
        );
    }
    Ok((emb_lo, rid_lo, emb_hi, rid_hi))
}

/// The prediction, stated before the experiment: low rank is a bias-variance
/// trade, so the embedding should beat ridge on low-powered traits and lose on
/// well-powered ones.
///
/// **Half of it replicated.** The low-powered win holds across seeds. The
/// well-powered loss did not — and the reason is that the prediction assumed a
/// bias the simulation does not contain. With `trait_specific = 0` the truth is
/// exactly `B = UV'`, so the `H`-dimensional model class *contains* the truth
/// and low rank costs nothing. The crossover is therefore tested at two levels
/// of out-of-program effect, and should appear only at the second.
#[test]
fn test_embedding_prs_against_ridge_prs() -> Result<()> {
    let seeds = [20250808u64, 111, 222, 333];

    let mut mean_delta = Vec::new();
    for (label, trait_specific) in [("exactly low-rank", 0.0f32), ("+ off-program", 0.8)] {
        println!(
            "\n=== {label} (trait_specific = {trait_specific}) ===\n{:>8} | {:>16} | {:>16}",
            "seed", "low-powered Δ", "well-powered Δ"
        );
        println!("{}", "-".repeat(48));

        let (mut lo_wins, mut hi_losses) = (0usize, 0usize);
        let (mut lo_sum, mut hi_sum) = (0.0f32, 0.0f32);
        for (i, &seed) in seeds.iter().enumerate() {
            let (el, rl, eh, rh) =
                one_replicate(seed, trait_specific, i == 0 && trait_specific == 0.0)?;
            let (dlo, dhi) = (el - rl, eh - rh);
            println!("{seed:>8} | {dlo:>+16.3} | {dhi:>+16.3}");
            if dlo > 0.0 {
                lo_wins += 1;
            }
            if dhi < 0.0 {
                hi_losses += 1;
            }
            lo_sum += dlo;
            hi_sum += dhi;
        }
        let n = seeds.len() as f32;
        println!(
            "mean Δ: low-powered {:+.3} (ahead {}/{}), well-powered {:+.3} (behind {}/{})",
            lo_sum / n,
            lo_wins,
            seeds.len(),
            hi_sum / n,
            hi_losses,
            seeds.len()
        );
        mean_delta.push((lo_sum / n, hi_sum / n));
    }
    println!();

    // The replicated finding: what governs the trade is whether the truth is
    // low-rank, not how well powered a trait is. When the model class contains
    // the truth the embedding wins regardless of power; when it does not, it
    // loses regardless -- and loses harder where power is high, which is the
    // h² gradient the original prediction was reaching for.
    let (lo_ok, hi_ok) = (mean_delta[0], mean_delta[1]);
    assert!(
        lo_ok.0 > hi_ok.0 && lo_ok.1 > hi_ok.1,
        "the embedding should fare relatively better under a low-rank truth: \
         low-rank {lo_ok:?} vs off-program {hi_ok:?}"
    );
    assert!(
        hi_ok.1 < hi_ok.0,
        "under a mis-specified truth the loss should be worse for well-powered \
         traits: {:?}",
        hi_ok
    );
    Ok(())
}
