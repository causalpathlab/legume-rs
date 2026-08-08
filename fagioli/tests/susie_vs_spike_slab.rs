//! SuSiE against spike-slab for the variant loadings.
//!
//! The two differ in how selection is parameterised, not in the score. A
//! spike-slab gives every (variant, program) pair an independent Bernoulli
//! gate, so nothing forces variants to compete; SuSiE places `L` single-effect
//! components in each block, each a categorical over that block's variants, so
//! mass *must* compete and credible sets follow.
//!
//! Two things are measured, and they are not the same question:
//!
//! - **Detection.** LD-matched AUC on `‖u_g‖`. Whether a causal variant is
//!   found at all.
//! - **Concentration.** How much of a program's inclusion mass sits on its top
//!   few variants. This is what "credible set" means operationally, and it is
//!   where the categorical should pay off.
//!
//! The categorical spans one block, never the genome, which is what
//! `--max-block-snps` exists to bound. A run with unbounded blocks would put
//! thousands of categories under one softmax, and that is the regime where the
//! spike-slab's independent gates are the safer choice.
//!
//! Run: cargo test -p fagioli --test susie_vs_spike_slab -- --nocapture

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

struct Truth {
    input: SumstatInput,
    causal: HashSet<usize>,
    ld_score: Vec<f32>,
}

fn simulate(h2: f32, seed: u64) -> Truth {
    let mut rng = SmallRng::seed_from_u64(seed);
    let x_raw = simulate_genotypes(&mut rng);
    let m = x_raw.ncols();
    let mut x = x_raw.clone();
    x.scale_columns_inplace();

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
    let g = &x * (&u * v_true.transpose());

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
            geno: wrap(x_raw),
            zscores: z,
            blocks: uniform_blocks(),
            median_n: NUM_INDIVIDUALS as u64,
            max_rank: 90,
        },
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

/// AUC against LD-score-matched controls — the honest baseline, since ranking
/// by marginal evidence alone recovers tags.
fn auc_ld_matched(score: &[f32], positive: &HashSet<usize>, ld: &[f32]) -> f32 {
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|&a, &b| ld[a].partial_cmp(&ld[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut controls: HashSet<usize> = HashSet::default();
    for &c in positive {
        let p = order.iter().position(|&x| x == c).unwrap_or(0);
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

fn variant_norms(fit: &EmbedFit) -> Vec<f32> {
    let mut out = Vec::new();
    for u in &fit.u_mean {
        for g in 0..u.nrows() {
            out.push((0..u.ncols()).map(|h| u[(g, h)].powi(2)).sum::<f32>().sqrt());
        }
    }
    out
}

/// Fraction of a (block, program)'s inclusion mass carried by its top `k`
/// variants, averaged. High means selection concentrated; low means it is
/// spread, and no credible set of useful size exists.
fn top_k_mass(fit: &EmbedFit, k: usize) -> f32 {
    let (mut total, mut n) = (0.0f32, 0usize);
    for pip in &fit.u_pip {
        for h in 0..pip.ncols() {
            let mut col: Vec<f32> = (0..pip.nrows()).map(|g| pip[(g, h)]).collect();
            let sum: f32 = col.iter().sum();
            if sum <= 0.0 {
                continue;
            }
            col.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            total += col.iter().take(k).sum::<f32>() / sum;
            n += 1;
        }
    }
    total / n.max(1) as f32
}

struct Measured {
    auc_matched: f32,
    top4: f32,
    top20: f32,
    loss: f32,
}

fn run(truth: &Truth, u_prior: UPrior, seed: u64) -> Result<Measured> {
    let (report, bases) = calibrate_input(&truth.input).expect("calibration");
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
            u_prior,
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
    Ok(Measured {
        auc_matched: auc_ld_matched(&norms, &truth.causal, &truth.ld_score),
        top4: top_k_mass(&fit, 4),
        top20: top_k_mass(&fit, 20),
        loss: *fit.loss_trace.last().unwrap_or(&f32::NAN),
    })
}

#[test]
fn test_susie_against_spike_slab() -> Result<()> {
    println!(
        "\n{:>12} {:>6} | {:>9} | {:>8} {:>8} | {:>8}",
        "prior", "seed", "AUC(LDm)", "top-4", "top-20", "loss"
    );
    println!("{}", "-".repeat(64));

    let seeds = [20250808u64, 111, 222];
    let mut agg: Vec<(UPrior, f32, f32, f32)> = Vec::new();

    for &prior in &[UPrior::SpikeSlab, UPrior::Susie] {
        let (mut a, mut t4, mut t20) = (0.0f32, 0.0f32, 0.0f32);
        for &seed in &seeds {
            let truth = simulate(0.4, seed);
            let m = run(&truth, prior, seed)?;
            println!(
                "{:>12} {:>6} | {:>9.3} | {:>8.3} {:>8.3} | {:>8.4}",
                format!("{prior:?}"),
                seed,
                m.auc_matched,
                m.top4,
                m.top20,
                m.loss
            );
            a += m.auc_matched;
            t4 += m.top4;
            t20 += m.top20;
        }
        let n = seeds.len() as f32;
        agg.push((prior, a / n, t4 / n, t20 / n));
    }

    println!(
        "\nmean: spike-slab AUC {:.3}, top-4 {:.3}, top-20 {:.3}",
        agg[0].1, agg[0].2, agg[0].3
    );
    println!(
        "      susie      AUC {:.3}, top-4 {:.3}, top-20 {:.3}\n",
        agg[1].1, agg[1].2, agg[1].3
    );
    println!(
        "H = {NUM_PROGRAMS}, {} variants per block, {} causal per program.\n\
         top-k is the share of a (block, program)'s inclusion mass on its k largest\n\
         entries; 4 causal per program means a well-concentrated fit should put most\n\
         mass in the top few.\n",
        SNPS_PER_BLOCK, CAUSAL_PER_PROGRAM
    );

    // Both must find causal variants above an LD-matched baseline; a family
    // that cannot do that is not a candidate whatever its credible sets look like.
    assert!(
        agg[0].1 > 0.6 && agg[1].1 > 0.6,
        "both priors should detect causal variants: spike-slab {:.3}, susie {:.3}",
        agg[0].1,
        agg[1].1
    );
    Ok(())
}

/// Against the established summary-statistic method in the crate.
///
/// `fit_block_rss` is RSS fine-mapping in the Zhu & Stephens eigenspace with a
/// SuSiE variational family — the standard approach for this data type, and a
/// far stronger comparator than the ridge PRS used for prediction. It shares
/// the LD machinery with the embedding, so what differs is the model: RSS SuSiE
/// fine-maps **each trait independently**, while the embedding forces every
/// trait through `H` shared programs.
///
/// The expected shape follows from that difference, and is the same one the PRS
/// benchmark found: borrowing should help where a single trait is underpowered
/// and cost nothing to nothing where it is not. Per-variant detection is scored
/// by the best PIP across traits, so the two are compared on the same quantity.
#[test]
fn test_embedding_against_rss_susie() -> Result<()> {
    use fagioli::sgvb::{fit_block_rss, FitConfig, ModelType, PriorType, RssParams};

    println!(
        "\n{:>6} {:>8} | {:>16} {:>16}",
        "h²", "seed", "embedding AUC", "RSS SuSiE AUC"
    );
    println!("{}", "-".repeat(52));

    let (mut emb_sum, mut rss_sum, mut n) = (0.0f32, 0.0f32, 0usize);
    for &h2 in &[0.1f32, 0.4] {
        for &seed in &[20250808u64, 111] {
            let truth = simulate(h2, seed);
            let emb = run(&truth, UPrior::Susie, seed)?;

            // RSS SuSiE, per block, on the same z-scores and the same panel.
            let (report, _) = calibrate_input(&truth.input).expect("calibration");
            let lambda = report.noise.lambda_white();
            let config = FitConfig {
                model_type: ModelType::Susie,
                prior_type: PriorType::Single,
                num_components: 5,
                num_sgvb_samples: 10,
                learning_rate: 0.05,
                num_iterations: 300,
                batch_size: 256,
                prior_vars: vec![0.1],
                elbo_window: 50,
                seed,
                sigma2_inf: 0.0,
                prior_alpha: 1.0,
            };
            let rss_params = RssParams {
                max_rank: truth.input.max_rank,
                lambda,
                ldsc_intercept: false,
            };

            let mut rss_score = vec![0.0f32; truth.input.geno.num_snps()];
            for block in &truth.input.blocks {
                let bm = block.num_snps();
                let mut xb = truth
                    .input
                    .geno
                    .genotypes
                    .columns(block.snp_start, bm)
                    .clone_owned();
                xb.scale_columns_inplace();
                let zb = truth.input.zscores.rows(block.snp_start, bm).clone_owned();
                let detailed = fit_block_rss(&xb, &zb, &config, &rss_params, &Device::Cpu)?;
                let best = detailed.best_result();
                for g in 0..bm {
                    // Best PIP across traits, matching the embedding's ‖u_g‖,
                    // which is also a single per-variant number over traits.
                    rss_score[block.snp_start + g] = (0..best.pip.ncols())
                        .map(|t| best.pip[(g, t)])
                        .fold(0.0f32, f32::max);
                }
            }
            let rss_auc = auc_ld_matched(&rss_score, &truth.causal, &truth.ld_score);

            println!(
                "{:>6.1} {:>8} | {:>16.3} {:>16.3}",
                h2, seed, emb.auc_matched, rss_auc
            );
            emb_sum += emb.auc_matched;
            rss_sum += rss_auc;
            n += 1;
        }
    }

    println!(
        "\nmean over {n} runs: embedding {:.3}, RSS SuSiE {:.3} (Δ {:+.3})\n",
        emb_sum / n as f32,
        rss_sum / n as f32,
        (emb_sum - rss_sum) / n as f32
    );

    assert!(
        rss_sum / n as f32 > 0.5,
        "the RSS comparator should beat chance, got {:.3}",
        rss_sum / n as f32
    );
    Ok(())
}
