//! Two questions, one experiment.
//!
//! **A. Does the polygenic arm help identify sparse effects?** Sparse and dense
//! effects have different signatures: a sparse effect enters the *mean* as
//! `d̃_k V_R[g,k]` — first power of `d̃`, and coherent across coordinates — while
//! the dense component enters the *variance* as `d̃²_k`, incoherently. Without
//! somewhere for the incoherent part to go, the model must explain polygenic
//! background with sparse effects, and it understates the noise by
//! `(1 + d̃²_k σ²_d / s²_k)` — a factor that **grows with `d̃²`**. So the
//! prediction is not merely "PIPs inflate" but that they inflate *specifically
//! at variants loading on high-eigenvalue directions*, which is measurable as
//! `corr(‖u_g‖, LD score)`.
//!
//! **B. Do causal and non-causal embeddings differ characteristically?** Norm is
//! the difference by construction. Beyond it: a causal variant acts through one
//! program, so its `u_g` should be *anisotropic* across the `H` coordinates,
//! while a non-causal variant's residual is noise and therefore isotropic.
//! Measured as a participation ratio, `(Σ_h u²)² / Σ_h u⁴` — near 1 for a
//! variant on a single program, near `H` for noise.
//!
//! What is deliberately *not* claimed: that causal variants separate from their
//! tags. Under rSVD truncation, variants differing only in the null space of
//! `X̃` are exactly unidentified, so a causal variant and an `r² ≈ 1` tag share
//! the effect. That is ordinary fine-mapping non-identifiability; the credible
//! set differs, not the variant. Any apparent separation there would be a bug.
//!
//! Run: cargo test -p fagioli --test dense_arm_and_variant_embedding -- --nocapture

use anyhow::Result;
use candle_util::candle_core::Device;
use fagioli::embedding::model::{EmbedConfig, UPrior};
use fagioli::embedding::noise::NoiseModel;
use fagioli::embedding::train::{train, EmbedFit};
use fagioli::embedding::whiten::whiten_blocks;
use fagioli::genotype::GenotypeMatrix;
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::{decompose_blocks, SumstatInput};
use fagioli::summary_stats::LdBlock;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rustc_hash::FxHashSet as HashSet;

const NUM_BLOCKS: usize = 8;
const SNPS_PER_BLOCK: usize = 120;
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

/// Genotypes with block LD from latent haplotypes, thresholded to dosages.
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

fn dummy_geno(x: DMatrix<f32>) -> GenotypeMatrix {
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

struct Truth {
    input: SumstatInput,
    causal: HashSet<usize>,
    /// Within-block LD score per SNP, `ℓ_j = Σ_k r²_jk`.
    ld_score: Vec<f32>,
}

/// Sparse causal effects plus an optional polygenic background on every variant.
fn simulate(h2_sparse: f32, h2_polygenic: f32, seed: u64) -> Truth {
    let mut rng = SmallRng::seed_from_u64(seed);
    let x_raw = simulate_genotypes(&mut rng);
    let m = x_raw.ncols();
    let mut x = x_raw.clone();
    x.scale_columns_inplace();

    // Sparse: a few causal variants per program.
    let mut u = DMatrix::<f32>::zeros(m, NUM_PROGRAMS);
    let mut causal = HashSet::default();
    for prog in 0..NUM_PROGRAMS {
        for j in rand::seq::index::sample(&mut rng, m, CAUSAL_PER_PROGRAM) {
            let v: f64 = StandardNormal.sample(&mut rng);
            u[(j, prog)] = v as f32;
            causal.insert(j);
        }
    }
    let v_true = randn(NUM_TRAITS, NUM_PROGRAMS, &mut rng);
    let b_sparse = &u * v_true.transpose();

    // Polygenic: small effects on *every* variant, through the same programs,
    // so the background shares the trait geometry rather than being white.
    let u_dense = randn(m, NUM_PROGRAMS, &mut rng) * (1.0 / (m as f32).sqrt());
    let b_dense = &u_dense * v_true.transpose();

    let g_s = &x * &b_sparse;
    let g_d = &x * &b_dense;

    let mut y = DMatrix::<f32>::zeros(NUM_INDIVIDUALS, NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let norm = |c: nalgebra::DVectorView<f32>| {
            (c.iter().map(|v| v * v).sum::<f32>() / NUM_INDIVIDUALS as f32).sqrt()
        };
        let (sd_s, sd_d) = (norm(g_s.column(t)), norm(g_d.column(t)));
        let e_var = (1.0 - h2_sparse - h2_polygenic).max(0.0).sqrt();
        for i in 0..NUM_INDIVIDUALS {
            let e: f64 = StandardNormal.sample(&mut rng);
            let sparse = if sd_s > 0.0 {
                h2_sparse.sqrt() * g_s[(i, t)] / sd_s
            } else {
                0.0
            };
            let dense = if sd_d > 0.0 && h2_polygenic > 0.0 {
                h2_polygenic.sqrt() * g_d[(i, t)] / sd_d
            } else {
                0.0
            };
            y[(i, t)] = sparse + dense + e_var * e as f32;
        }
    }

    // Marginal OLS z-scores.
    let n = NUM_INDIVIDUALS as f32;
    let mut z = DMatrix::<f32>::zeros(m, NUM_TRAITS);
    for t in 0..NUM_TRAITS {
        let yt = y.column(t);
        let y_sd = (yt.iter().map(|v| v * v).sum::<f32>() / n).sqrt().max(1e-8);
        for j in 0..m {
            z[(j, t)] = x.column(j).dot(&yt) / (y_sd * n.sqrt());
        }
    }

    // Within-block LD scores from the standardized genotypes.
    let mut ld_score = vec![0.0f32; m];
    for b in 0..NUM_BLOCKS {
        let s = b * SNPS_PER_BLOCK;
        let xb = x.columns(s, SNPS_PER_BLOCK);
        let r = (xb.transpose() * xb) / n;
        for j in 0..SNPS_PER_BLOCK {
            ld_score[s + j] = (0..SNPS_PER_BLOCK).map(|k| r[(j, k)] * r[(j, k)]).sum();
        }
    }

    Truth {
        causal,
        ld_score,
        input: SumstatInput {
            geno: dummy_geno(x_raw),
            zscores: z,
            blocks: uniform_blocks(),
            median_n: NUM_INDIVIDUALS as u64,
            max_rank: 80,
        },
    }
}

/// Per-variant `‖u_g‖` in the global SNP ordering.
fn variant_norms(fit: &EmbedFit) -> Vec<f32> {
    let mut out = Vec::new();
    for u in &fit.u_mean {
        for g in 0..u.nrows() {
            out.push((0..u.ncols()).map(|h| u[(g, h)].powi(2)).sum::<f32>().sqrt());
        }
    }
    out
}

/// Participation ratio `(Σ_h u²)² / Σ_h u⁴`: 1 for a variant on one program,
/// `H` for isotropic noise.
fn participation_ratios(fit: &EmbedFit) -> Vec<f32> {
    let mut out = Vec::new();
    for u in &fit.u_mean {
        for g in 0..u.nrows() {
            let s2: f32 = (0..u.ncols()).map(|h| u[(g, h)].powi(2)).sum();
            let s4: f32 = (0..u.ncols()).map(|h| u[(g, h)].powi(4)).sum();
            out.push(if s4 > 0.0 { s2 * s2 / s4 } else { f32::NAN });
        }
    }
    out
}

/// AUC of `score` separating `positive` from the rest (Mann-Whitney).
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

/// AUC against an **LD-score-matched** control set, which is the honest
/// baseline: ranking by marginal evidence alone recovers tags, so beating an
/// unmatched control proves nothing.
fn auc_ld_matched(score: &[f32], positive: &HashSet<usize>, ld: &[f32]) -> f32 {
    let mut controls: HashSet<usize> = HashSet::default();
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|&a, &b| ld[a].partial_cmp(&ld[b]).unwrap_or(std::cmp::Ordering::Equal));
    let pos_of = |i: usize| order.iter().position(|&x| x == i).unwrap_or(0);
    for &c in positive {
        // Nearest-by-LD-score variants that are not themselves causal.
        let p = pos_of(c);
        for step in 1..order.len() {
            let mut placed = 0;
            for cand in [p.saturating_sub(step), (p + step).min(order.len() - 1)] {
                let g = order[cand];
                if !positive.contains(&g) && controls.insert(g) {
                    placed += 1;
                }
            }
            if placed > 0 {
                break;
            }
        }
    }
    let subset: Vec<usize> = positive.iter().chain(controls.iter()).copied().collect();
    let sub_scores: Vec<f32> = subset.iter().map(|&i| score[i]).collect();
    let sub_pos: HashSet<usize> = subset
        .iter()
        .enumerate()
        .filter(|(_, &g)| positive.contains(&g))
        .map(|(i, _)| i)
        .collect();
    auc(&sub_scores, &sub_pos)
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

fn mean_of(v: &[f32], keep: impl Fn(usize) -> bool) -> f32 {
    let sel: Vec<f32> = v
        .iter()
        .enumerate()
        .filter(|(i, x)| keep(*i) && x.is_finite())
        .map(|(_, x)| *x)
        .collect();
    sel.iter().sum::<f32>() / sel.len().max(1) as f32
}

struct Measured {
    auc: f32,
    auc_matched: f32,
    corr_ld: f32,
    pr_causal: f32,
    pr_other: f32,
}

fn run(truth: &Truth, dense_arm: bool, gauge_weight: f64) -> Result<Measured> {
    let bases = decompose_blocks(&truth.input);
    let report = calibrate_input(&truth.input, &bases).expect("calibration");
    let lambda = report.noise.lambda_white();
    let blocks = whiten_blocks(&truth.input, bases, None, lambda)?;
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
            dense_arm,
            gauge_weight,
            seed: 3,
        },
        &Device::Cpu,
    )?;

    let norms = variant_norms(&fit);
    let pr = participation_ratios(&fit);
    Ok(Measured {
        auc: auc(&norms, &truth.causal),
        auc_matched: auc_ld_matched(&norms, &truth.causal, &truth.ld_score),
        corr_ld: pearson(&norms, &truth.ld_score),
        pr_causal: mean_of(&pr, |i| truth.causal.contains(&i)),
        pr_other: mean_of(&pr, |i| !truth.causal.contains(&i)),
    })
}

#[test]
fn test_dense_arm_and_variant_embedding_characteristics() -> Result<()> {
    println!(
        "\n{:>10} {:>6} | {:>7} {:>9} {:>9} | {:>9} {:>9}",
        "polygenic", "arm", "AUC", "AUC(LDm)", "corr(u,ℓ)", "PR causal", "PR other"
    );
    println!("{}", "-".repeat(74));

    let mut corr_no_arm = Vec::new();
    let mut corr_arm = Vec::new();
    let mut auc_no_arm = Vec::new();
    let mut auc_arm = Vec::new();

    for &h2_poly in &[0.0f32, 0.2, 0.4] {
        let truth = simulate(0.3, h2_poly, 20250808);
        // Three arms, so the dense score term and the orthonormality gauge are
        // never varied together.
        for (label, dense, gauge) in [
            ("plain", false, 0.0),
            ("gauge", false, 10.0),
            ("dense", true, 10.0),
        ] {
            let m = run(&truth, dense, gauge)?;
            println!(
                "{:>10.1} {:>6} | {:>7.3} {:>9.3} {:>9.3} | {:>9.2} {:>9.2}",
                h2_poly,
                label,
                m.auc,
                m.auc_matched,
                m.corr_ld,
                m.pr_causal,
                m.pr_other
            );
            if dense {
                corr_arm.push(m.corr_ld.abs());
                auc_arm.push(m.auc_matched);
            } else if gauge == 0.0 {
                corr_no_arm.push(m.corr_ld.abs());
                auc_no_arm.push(m.auc_matched);
            }
        }
    }
    println!(
        "\nH = {NUM_PROGRAMS}, so a participation ratio of 1 means one program \
         and {NUM_PROGRAMS} means isotropic noise.\n"
    );

    // Question B: the embedding must carry causal information at all.
    assert!(
        auc_no_arm.iter().chain(auc_arm.iter()).any(|&a| a > 0.6),
        "‖u_g‖ should separate causal variants from LD-matched controls somewhere: \
         no-arm {auc_no_arm:?}, arm {auc_arm:?}"
    );
    Ok(())
}
