//! Reusing calibration's eigenbasis must change the whitening not at all.
//!
//! `calibrate_input` and `whiten_blocks` both decompose every block, with the
//! same standardisation and the same `max_rank`; only the ridge differs, and
//! the ridge is applied after the decomposition. The randomized SVD is seed-
//! pinned (`RSVD_SUBSPACE_SEED` in matrix-util), so the two decompositions were
//! bit-identical and one of them was waste.
//!
//! That is the argument. This is the check: whiten once from the handed-over
//! bases and once from an empty cache — which forces the old decompose-here
//! path — and require every entry of `X̃`, `ž` and `d²` to agree exactly. Not
//! approximately: a tolerance would hide exactly the drift this exists to
//! catch, since a re-decomposition that differed at all would differ in the
//! subspace, not in the last bit.
//!
//! Run: cargo test -p fagioli --test basis_reuse -- --nocapture
use anyhow::Result;
use fagioli::embedding::whiten::whiten_blocks;
use fagioli::genotype::GenotypeMatrix;
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::SumstatInput;
use fagioli::summary_stats::rss_svd::BlockEigenBases;
use fagioli::summary_stats::LdBlock;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

const NUM_BLOCKS: usize = 5;
const SNPS_PER_BLOCK: usize = 120;
const NUM_INDIVIDUALS: usize = 500;
const NUM_TRAITS: usize = 6;

/// Genotypes with block LD, z-scores with a little signal — enough that the
/// calibration lands on a non-trivial λ rather than the degenerate λ = 0.
fn simulate(seed: u64) -> SumstatInput {
    simulate_sized(seed, NUM_BLOCKS, SNPS_PER_BLOCK, NUM_INDIVIDUALS, 80)
}

fn simulate_sized(
    seed: u64,
    num_blocks: usize,
    snps_per_block: usize,
    num_individuals: usize,
    max_rank: usize,
) -> SumstatInput {
    let mut rng = SmallRng::seed_from_u64(seed);
    let m = num_blocks * snps_per_block;

    let mut x = DMatrix::<f32>::zeros(num_individuals, m);
    for b in 0..num_blocks {
        let n_hap = 5;
        let hap = DMatrix::from_fn(num_individuals, n_hap, |_, _| {
            let v: f64 = StandardNormal.sample(&mut rng);
            v as f32
        });
        for j in 0..snps_per_block {
            let rho = 0.9 - 0.45 * (j as f32 / snps_per_block as f32);
            for i in 0..num_individuals {
                let e: f64 = StandardNormal.sample(&mut rng);
                let latent = rho * hap[(i, j % n_hap)] + (1.0 - rho * rho).sqrt() * e as f32;
                x[(i, b * snps_per_block + j)] = if latent < -0.6 {
                    0.0
                } else if latent < 0.6 {
                    1.0
                } else {
                    2.0
                };
            }
        }
    }

    let zscores = DMatrix::from_fn(m, NUM_TRAITS, |_, _| {
        let v: f64 = StandardNormal.sample(&mut rng);
        v as f32 * 1.4
    });

    SumstatInput {
        geno: GenotypeMatrix {
            individual_ids: (0..num_individuals)
                .map(|i| Box::from(format!("i{i}")))
                .collect(),
            snp_ids: (0..m).map(|j| Box::from(format!("rs{j}"))).collect(),
            chromosomes: vec![Box::from("chr1"); m],
            positions: (0..m).map(|j| (j * 1000) as u64).collect(),
            allele1: vec![Box::from("A"); m],
            allele2: vec![Box::from("G"); m],
            genotypes: x,
        },
        zscores,
        blocks: (0..num_blocks)
            .map(|b| LdBlock {
                block_idx: b,
                snp_start: b * snps_per_block,
                snp_end: (b + 1) * snps_per_block,
                chr: Box::from("chr1"),
                bp_start: (b * snps_per_block * 1000) as u64,
                bp_end: ((b + 1) * snps_per_block * 1000) as u64,
            })
            .collect(),
        median_n: num_individuals as u64,
        max_rank,
    }
}

#[test]
fn test_reused_basis_reproduces_the_whitening_exactly() -> Result<()> {
    let input = simulate(20250808);

    let (report, bases) = calibrate_input(&input).expect("calibration");
    let lambda = report.noise.lambda_white();
    println!("λ_white = {lambda:.6}, {} bases cached", bases.num_cached());
    assert_eq!(
        bases.num_cached(),
        NUM_BLOCKS,
        "calibration must hand over one basis per block it decomposed",
    );

    let reused = whiten_blocks(&input, bases, None, lambda)?;
    // An empty cache is the pre-change path: whitening decomposes for itself.
    let fresh = whiten_blocks(&input, BlockEigenBases::empty(), None, lambda)?;

    assert_eq!(reused.len(), fresh.len(), "block count moved");
    assert!(!reused.is_empty(), "nothing was whitened, so nothing is tested");

    let mut compared = 0usize;
    for (r, f) in reused.iter().zip(fresh.iter()) {
        assert_eq!(r.block_idx, f.block_idx);
        assert_eq!(r.num_snps, f.num_snps);
        assert_eq!(r.d_sq, f.d_sq, "block {} spectrum differs", r.block_idx);
        assert_eq!(
            r.x_design.shape(),
            f.x_design.shape(),
            "block {} design shape differs",
            r.block_idx,
        );
        assert_eq!(
            r.z_white.shape(),
            f.z_white.shape(),
            "block {} response shape differs",
            r.block_idx,
        );
        assert!(
            r.x_design.iter().eq(f.x_design.iter()),
            "block {} design differs bitwise",
            r.block_idx,
        );
        assert!(
            r.z_white.iter().eq(f.z_white.iter()),
            "block {} whitened z differs bitwise",
            r.block_idx,
        );
        compared += r.x_design.len() + r.z_white.len() + r.d_sq.len();
    }

    println!("{compared} values identical across {} blocks", reused.len());
    Ok(())
}

/// What the reuse is worth, on a panel large enough for the decomposition to
/// dominate. Printed, not asserted: wall-clock on a shared machine is not a
/// property of the code.
#[test]
fn measure_whitening_cost_with_and_without_reuse() -> Result<()> {
    use std::time::Instant;

    println!(
        "\n{:>6} {:>6} {:>6} | {:>10} {:>10} {:>10} {:>8}",
        "blocks", "snps", "n", "calibrate", "whiten+", "whiten-", "saved"
    );
    println!("{}", "-".repeat(72));

    for &(nb, spb, n, rank) in &[(8usize, 400usize, 1000usize, 300usize), (16, 500, 1500, 400)] {
        let input = simulate_sized(20250808, nb, spb, n, rank);

        let t0 = Instant::now();
        let (report, bases) = calibrate_input(&input).expect("calibration");
        let t_cal = t0.elapsed().as_secs_f64();
        let lambda = report.noise.lambda_white();

        let t1 = Instant::now();
        let reused = whiten_blocks(&input, bases, None, lambda)?;
        let t_reused = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let fresh = whiten_blocks(&input, BlockEigenBases::empty(), None, lambda)?;
        let t_fresh = t2.elapsed().as_secs_f64();

        assert_eq!(reused.len(), fresh.len());
        let saved = (t_fresh - t_reused) / (t_cal + t_fresh);
        println!(
            "{nb:>6} {spb:>6} {n:>6} | {t_cal:>10.3} {t_reused:>10.3} {t_fresh:>10.3} \
             {:>7.1}%",
            saved * 100.0,
        );
    }
    println!(
        "\nwhiten+ reuses calibration's bases, whiten- decomposes again. `saved` is\n\
         the share of a calibrate-then-whiten pass that the reuse removes.\n"
    );
    Ok(())
}

/// A block the calibration could not *fit* must still be whitened. The two
/// stages filter on different thresholds, so the cache is allowed to miss, and
/// a miss has to cost a decomposition rather than a dropped block.
#[test]
fn test_empty_cache_still_whitens_every_block() -> Result<()> {
    let input = simulate(111);
    let (report, _) = calibrate_input(&input).expect("calibration");
    let lambda = report.noise.lambda_white();

    let fresh = whiten_blocks(&input, BlockEigenBases::empty(), None, lambda)?;
    assert_eq!(
        fresh.len(),
        NUM_BLOCKS,
        "an empty cache must fall back to decomposing, not to skipping",
    );

    // A short cache is the same situation: the tail is padded with misses.
    let short = whiten_blocks(&input, BlockEigenBases::from_slots(vec![None]), None, lambda)?;
    assert_eq!(short.len(), NUM_BLOCKS, "a short cache dropped blocks");
    Ok(())
}
