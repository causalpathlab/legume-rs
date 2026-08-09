//! The three-class simulation shared by the pleiotropy experiments.
//!
//! Genotypes with block LD, then three planted classes of variant:
//!
//! - **pleiotropic** — acts through one program, so it moves every trait that
//!   loads on that program;
//! - **trait-specific** — acts on exactly one trait, *outside* the program
//!   space, so no program can represent it;
//! - **null** — no effect.
//!
//! [`simulate_blended`] is the same generator with one deliberate change, used
//! to measure the cost of the LD-free reading rather than assume it away.
//!
//! Included with `#[path]` rather than published, because integration tests
//! cannot depend on one another.
#![allow(dead_code)]

use fagioli::genotype::GenotypeMatrix;
use fagioli::summary_stats::common::SumstatInput;
use fagioli::summary_stats::LdBlock;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rustc_hash::FxHashSet as HashSet;

pub const NUM_BLOCKS: usize = 6;
pub const SNPS_PER_BLOCK: usize = 150;
pub const NUM_INDIVIDUALS: usize = 800;
pub const NUM_TRAITS: usize = 10;
pub const NUM_PROGRAMS: usize = 3;
pub const CAUSAL_PER_PROGRAM: usize = 4;
/// Haplotypes per block. SNP `j` loads on haplotype `j % N_HAPLOTYPES`, so two
/// SNPs in one block are in LD exactly when that index matches.
pub const N_HAPLOTYPES: usize = 6;

pub fn randn(r: usize, c: usize, rng: &mut SmallRng) -> DMatrix<f32> {
    DMatrix::from_fn(r, c, |_, _| {
        let v: f64 = StandardNormal.sample(rng);
        v as f32
    })
}

pub fn simulate_genotypes(rng: &mut SmallRng) -> DMatrix<f32> {
    let m = NUM_BLOCKS * SNPS_PER_BLOCK;
    let mut x = DMatrix::<f32>::zeros(NUM_INDIVIDUALS, m);
    for b in 0..NUM_BLOCKS {
        let n_hap = N_HAPLOTYPES;
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

pub fn uniform_blocks() -> Vec<LdBlock> {
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

pub fn wrap(x: DMatrix<f32>) -> GenotypeMatrix {
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

pub fn auc(score: &[f32], positive: &HashSet<usize>) -> f32 {
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

/// AUC of `score` separating `pos` from `neg`, ignoring everything else.
pub fn auc_pair(score: &[f32], pos: &HashSet<usize>, neg: &HashSet<usize>) -> f32 {
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

/// Three planted classes.
pub struct Classes {
    pub input: SumstatInput,
    pub pleiotropic: HashSet<usize>,
    pub trait_specific: HashSet<usize>,
    /// RMS of `‖β_g‖` within each causal class. The pleiotropic-vs-specific
    /// contrast is only about *structure* if these two agree — otherwise a
    /// norm-based score is reading total effect size instead.
    pub rms_norm_pleio: f32,
    pub rms_norm_specific: f32,
}

/// RMS row norm of `b` over the given rows.
pub fn rms_row_norm(b: &DMatrix<f32>, rows: &HashSet<usize>) -> f32 {
    if rows.is_empty() {
        return 0.0;
    }
    let acc: f32 = rows.iter().map(|&j| b.row(j).norm_squared()).sum();
    (acc / rows.len() as f32).sqrt()
}

pub fn simulate_classes(h2: f32, seed: u64) -> Classes {
    simulate_inner(h2, seed, 0.0)
}

/// The same truth, except every trait-specific variant is put into LD with a
/// pleiotropic one at correlation `rho`.
///
/// The LD-free reading rests on `z_g = R_gj · β_j Λ_N + e_g` — a scalar rescale
/// that preserves direction — which holds when a neighbourhood contains one
/// causal variant. Two causal variants with *different* trait profiles at the
/// same locus make the row a genuine mixture. This is the stated limit, made
/// measurable.
///
/// `rho` must stay below 1. At `rho = 1` the two genotype columns are
/// identical, so `z_j` and `z_src` are the *same row*: every summary-statistic
/// method scores exactly 0.5 by tie, including full RSS fine-mapping, and the
/// experiment measures nothing about any of them.
pub fn simulate_blended(h2: f32, seed: u64, rho: f32) -> Classes {
    simulate_inner(h2, seed, rho)
}

/// Variants whose effects span a *chosen number of traits*, for calibrating a
/// shape statistic instead of only classifying with one.
///
/// `counts[i]` variants are planted for each entry, each acting on exactly that
/// many randomly chosen traits with `N(0,1)` effects, all scaled to a common
/// `‖β_g‖` so the contrast is trait-count and not magnitude.
pub struct Graded {
    pub input: SumstatInput,
    /// One variant-index set per requested trait count.
    pub by_count: Vec<(usize, HashSet<usize>)>,
}

pub fn simulate_graded(h2: f32, seed: u64, counts: &[usize], per_count: usize) -> Graded {
    let mut rng = SmallRng::seed_from_u64(seed);
    let x_raw = simulate_genotypes(&mut rng);
    let m = x_raw.ncols();
    let mut x = x_raw.clone();
    x.scale_columns_inplace();

    let mut b = DMatrix::<f32>::zeros(m, NUM_TRAITS);
    let mut by_count: Vec<(usize, HashSet<usize>)> = Vec::new();
    let mut used: HashSet<usize> = HashSet::default();

    for &k in counts {
        let mut group: HashSet<usize> = HashSet::default();
        for j in rand::seq::index::sample(&mut rng, m, per_count * 2) {
            if group.len() >= per_count {
                break;
            }
            if used.contains(&j) {
                continue;
            }
            // Exactly k traits, equal expected magnitude per variant.
            let traits = rand::seq::index::sample(&mut rng, NUM_TRAITS, k);
            let mut row: Vec<f32> = vec![0.0; NUM_TRAITS];
            for t in traits {
                let v: f64 = StandardNormal.sample(&mut rng);
                row[t] = v as f32;
            }
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            for (t, v) in row.iter().enumerate() {
                b[(j, t)] = v / norm; // ‖β_g‖ = 1 for every planted variant
            }
            used.insert(j);
            group.insert(j);
        }
        by_count.push((k, group));
    }

    let z = marginal_zscores(&x, &b, h2, &mut rng);
    Graded {
        by_count,
        input: SumstatInput {
            geno: wrap(x_raw),
            zscores: z,
            blocks: uniform_blocks(),
            median_n: NUM_INDIVIDUALS as u64,
            max_rank: 90,
        },
    }
}

/// Phenotypes at heritability `h2`, then marginal-OLS z-scores.
fn marginal_zscores(
    x: &DMatrix<f32>,
    b: &DMatrix<f32>,
    h2: f32,
    rng: &mut SmallRng,
) -> DMatrix<f32> {
    let g = x * b;
    let n = NUM_INDIVIDUALS as f32;
    let mut y = DMatrix::<f32>::zeros(NUM_INDIVIDUALS, NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let gt = g.column(t);
        let sd = (gt.iter().map(|v| v * v).sum::<f32>() / n).sqrt();
        for i in 0..NUM_INDIVIDUALS {
            let e: f64 = StandardNormal.sample(rng);
            let genetic = if sd > 0.0 { h2.sqrt() * gt[i] / sd } else { 0.0 };
            y[(i, t)] = genetic + (1.0 - h2).sqrt() * e as f32;
        }
    }
    let mut z = DMatrix::<f32>::zeros(x.ncols(), NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let yt = y.column(t);
        let y_sd = (yt.iter().map(|v| v * v).sum::<f32>() / n).sqrt().max(1e-8);
        for j in 0..x.ncols() {
            z[(j, t)] = x.column(j).dot(&yt) / (y_sd * n.sqrt());
        }
    }
    z
}

fn simulate_inner(h2: f32, seed: u64, blend_rho: f32) -> Classes {
    let mut rng = SmallRng::seed_from_u64(seed);
    let x_raw = simulate_genotypes(&mut rng);
    let m = x_raw.ncols();
    let mut x = x_raw.clone();
    x.scale_columns_inplace();

    // Pleiotropic: through the programs.
    let mut u = DMatrix::<f32>::zeros(m, NUM_PROGRAMS);
    let mut pleiotropic = HashSet::default();
    let mut pleio_order: Vec<usize> = Vec::new();
    for prog in 0..NUM_PROGRAMS {
        for j in rand::seq::index::sample(&mut rng, m, CAUSAL_PER_PROGRAM) {
            let v: f64 = StandardNormal.sample(&mut rng);
            u[(j, prog)] = v as f32;
            if pleiotropic.insert(j) {
                pleio_order.push(j);
            }
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

    // Blended arm: mix each pleiotropic variant's genotype into its paired
    // trait-specific variant at correlation `rho`, so one marginal row carries
    // a weighted sum of both profiles. Re-standardized afterwards, so the mix
    // changes the correlation and not the scale.
    if blend_rho > 0.0 && !pleio_order.is_empty() {
        let mut targets: Vec<usize> = trait_specific.iter().copied().collect();
        targets.sort_unstable();
        let keep = (1.0 - blend_rho * blend_rho).max(0.0).sqrt();
        for (i, &j) in targets.iter().enumerate() {
            let src = pleio_order[i % pleio_order.len()];
            let mixed = x.column(src) * blend_rho + x.column(j) * keep;
            x.set_column(j, &mixed);
        }
        x.scale_columns_inplace();
    }

    let z = marginal_zscores(&x, &b, h2, &mut rng);

    let rms_norm_pleio = rms_row_norm(&b, &pleiotropic);
    let rms_norm_specific = rms_row_norm(&b, &trait_specific);

    Classes {
        rms_norm_pleio,
        rms_norm_specific,
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
