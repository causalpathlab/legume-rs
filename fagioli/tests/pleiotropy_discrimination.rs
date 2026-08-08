//! Does the embedding tell PLEIOTROPIC variants from TRAIT-SPECIFIC ones?
//!
//! Every experiment so far planted only program-mediated causal variants, so
//! every causal variant was pleiotropic and "causal vs null" was the only
//! contrast ever measured. That is not the headline claim. This plants three
//! classes and asks the question directly:
//!
//! - **pleiotropic** — acts through a program, so it moves every trait loading
//!   on that program;
//! - **trait-specific** — acts on exactly one trait, *outside* the program
//!   space, so no program can represent it;
//! - **null** — no effect.
//!
//! A caution about what a positive result would mean. The model routes all
//! effects through `H` programs, so a trait-specific variant has no program to
//! load on and should come back with small `‖u_g‖` almost by construction.
//! Separation would then reflect what the model *cannot represent* rather than
//! anything it detected. The informative comparison is therefore
//! trait-specific against **null**: if the model cannot separate those two
//! either, then trait-specific variants are simply invisible to it, and the
//! pleiotropic-vs-specific contrast is an artefact of that blindness.
//!
//! Run: cargo test -p fagioli --test pleiotropy_discrimination -- --nocapture
use anyhow::Result;
use candle_util::candle_core::Device;
use fagioli::embedding::model::{EmbedConfig, UPrior};
use fagioli::embedding::noise::NoiseModel;
use fagioli::embedding::train::{train, EmbedFit};
use fagioli::embedding::whiten::whiten_blocks;
use fagioli::genotype::GenotypeMatrix;
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::SumstatInput;
use fagioli::summary_stats::LdBlock;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rustc_hash::FxHashSet as HashSet;

const NUM_BLOCKS: usize = 6;
const SNPS_PER_BLOCK: usize = 150;
const NUM_INDIVIDUALS: usize = 800;
const NUM_TRAITS: usize = 10;
const NUM_PROGRAMS: usize = 3;
const CAUSAL_PER_PROGRAM: usize = 4;

fn randn(r: usize, c: usize, rng: &mut SmallRng) -> DMatrix<f32> {
    DMatrix::from_fn(r, c, |_, _| {
        let v: f64 = StandardNormal.sample(rng);
        v as f32
    })
}

fn simulate_genotypes(rng: &mut SmallRng) -> DMatrix<f32> {
    let m = NUM_BLOCKS * SNPS_PER_BLOCK;
    let mut x = DMatrix::<f32>::zeros(NUM_INDIVIDUALS, m);
    for b in 0..NUM_BLOCKS {
        let n_hap = 6;
        let hap = randn(NUM_INDIVIDUALS, n_hap, rng);
        for j in 0..SNPS_PER_BLOCK {
            let rho = 0.9 - 0.45 * (j as f32 / SNPS_PER_BLOCK as f32);
            let col = b * SNPS_PER_BLOCK + j;
            for i in 0..NUM_INDIVIDUALS {
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


fn auc(score: &[f32], positive: &HashSet<usize>) -> f32 {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap_or(std::cmp::Ordering::Equal));
    let (mut rank_sum, mut n_pos) = (0.0f64, 0usize);
    for (rank, &i) in idx.iter().enumerate() {
        if positive.contains(&i) {
            rank_sum += (rank + 1) as f64;
            n_pos += 1;
        }
    }
    let n_neg = score.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    ((rank_sum - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos * n_neg) as f64) as f32
}


fn variant_norms(fit: &EmbedFit) -> Vec<f32> {
    let mut out = Vec::new();
    for u in &fit.u_mean {
        for g in 0..u.nrows() {
            out.push((0..u.ncols()).map(|h| u[(g, h)].powi(2)).sum::<f32>().sqrt());
        }
    }
    out
}





/// Three planted classes.
struct Classes {
    input: SumstatInput,
    pleiotropic: HashSet<usize>,
    trait_specific: HashSet<usize>,
}

fn simulate_classes(h2: f32, seed: u64) -> Classes {
    let mut rng = SmallRng::seed_from_u64(seed);
    let x_raw = simulate_genotypes(&mut rng);
    let m = x_raw.ncols();
    let mut x = x_raw.clone();
    x.scale_columns_inplace();

    // Pleiotropic: through the programs.
    let mut u = DMatrix::<f32>::zeros(m, NUM_PROGRAMS);
    let mut pleiotropic = HashSet::default();
    for prog in 0..NUM_PROGRAMS {
        for j in rand::seq::index::sample(&mut rng, m, CAUSAL_PER_PROGRAM) {
            let v: f64 = StandardNormal.sample(&mut rng);
            u[(j, prog)] = v as f32;
            pleiotropic.insert(j);
        }
    }
    let v_true = randn(NUM_TRAITS, NUM_PROGRAMS, &mut rng);
    let mut b = &u * v_true.transpose();

    // Trait-specific: one trait each, outside the program space. Scaled to the
    // same typical magnitude as a pleiotropic entry so the contrast is about
    // structure and not about effect size.
    let scale = b.abs().max().max(1e-6);
    let mut trait_specific = HashSet::default();
    for t in 0..NUM_TRAITS {
        for j in rand::seq::index::sample(&mut rng, m, 2) {
            if pleiotropic.contains(&j) {
                continue;
            }
            let v: f64 = StandardNormal.sample(&mut rng);
            b[(j, t)] += scale * v as f32;
            trait_specific.insert(j);
        }
    }

    let g = &x * &b;
    let n = NUM_INDIVIDUALS as f32;
    let mut y = DMatrix::<f32>::zeros(NUM_INDIVIDUALS, NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let gt = g.column(t);
        let sd = (gt.iter().map(|v| v * v).sum::<f32>() / n).sqrt();
        for i in 0..NUM_INDIVIDUALS {
            let e: f64 = StandardNormal.sample(&mut rng);
            let genetic = if sd > 0.0 { h2.sqrt() * gt[i] / sd } else { 0.0 };
            y[(i, t)] = genetic + (1.0 - h2).sqrt() * e as f32;
        }
    }

    let mut z = DMatrix::<f32>::zeros(m, NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let yt = y.column(t);
        let y_sd = (yt.iter().map(|v| v * v).sum::<f32>() / n).sqrt().max(1e-8);
        for j in 0..m {
            z[(j, t)] = x.column(j).dot(&yt) / (y_sd * n.sqrt());
        }
    }

    let mut ld_score = vec![0.0f32; m];
    for bi in 0..NUM_BLOCKS {
        let s = bi * SNPS_PER_BLOCK;
        let xb = x.columns(s, SNPS_PER_BLOCK);
        let r = (xb.transpose() * xb) / n;
        for j in 0..SNPS_PER_BLOCK {
            ld_score[s + j] = (0..SNPS_PER_BLOCK).map(|k| r[(j, k)] * r[(j, k)]).sum();
        }
    }

    Classes {
        pleiotropic,
        trait_specific,
        input: SumstatInput {
            geno: wrap(x_raw),
            zscores: z,
            blocks: uniform_blocks(),
            median_n: NUM_INDIVIDUALS as u64,
            max_rank: 90,
        },
    }
}

/// AUC of `score` separating `pos` from `neg`, ignoring everything else.
fn auc_pair(score: &[f32], pos: &HashSet<usize>, neg: &HashSet<usize>) -> f32 {
    let subset: Vec<usize> = pos.iter().chain(neg.iter()).copied().collect();
    let s: Vec<f32> = subset.iter().map(|&i| score[i]).collect();
    let p: HashSet<usize> = subset
        .iter()
        .enumerate()
        .filter(|(_, &g)| pos.contains(&g))
        .map(|(i, _)| i)
        .collect();
    auc(&s, &p)
}

#[test]
fn test_pleiotropic_versus_trait_specific() -> Result<()> {
    println!(
        "\n{:>8} | {:>14} {:>14} {:>14}",
        "seed", "pleio vs null", "spec vs null", "pleio vs spec"
    );
    println!("{}", "-".repeat(58));

    let (mut a, mut b_, mut c) = (0.0f32, 0.0f32, 0.0f32);
    let seeds = [20250808u64, 111, 222];
    for &seed in &seeds {
        let cl = simulate_classes(0.4, seed);
        let all: HashSet<usize> = cl.pleiotropic.union(&cl.trait_specific).copied().collect();
        let null: HashSet<usize> = (0..cl.input.geno.num_snps())
            .filter(|j| !all.contains(j))
            .collect();
        let input = cl.input;
        let report = calibrate_input(&input).expect("calibration");
        let lambda = report.noise.lambda_white();
        let blocks = whiten_blocks(&input, None, lambda)?;
        let d_sq: Vec<Vec<f32>> = blocks.iter().map(|x| x.d_sq.clone()).collect();
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
                seed,
            },
            &Device::Cpu,
        )?;
        let norms = variant_norms(&fit);

        let pn = auc_pair(&norms, &cl.pleiotropic, &null);
        let sn = auc_pair(&norms, &cl.trait_specific, &null);
        let ps = auc_pair(&norms, &cl.pleiotropic, &cl.trait_specific);
        println!("{seed:>8} | {pn:>14.3} {sn:>14.3} {ps:>14.3}");
        a += pn;
        b_ += sn;
        c += ps;
    }
    let n = seeds.len() as f32;
    println!(
        "\nmean: pleio vs null {:.3}, trait-specific vs null {:.3}, pleio vs specific {:.3}\n",
        a / n,
        b_ / n,
        c / n
    );
    println!(
        "Read trait-specific vs null first. At 0.5 the model cannot see that class\n\
         at all, and any pleio-vs-specific separation is blindness rather than\n\
         discrimination.\n"
    );
    Ok(())
}
