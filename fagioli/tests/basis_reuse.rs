//! One eigen-decomposition, two stages, and no difference in the answer.
//!
//! Calibration and whitening both need `R = V D² V'` per block, and they differ
//! only in what they do next: calibration fits the moment law on the
//! λ-independent projection `V'z` to derive τ, whitening applies the ridge
//! `λ = τ`. `decompose_blocks` therefore runs the randomized SVD once and lends
//! the result to both — borrowed for the fit, moved for the whitening.
//!
//! The risk in sharing is that the shared basis is not the basis a stage would
//! have computed for itself. [`test_shared_basis_matches_an_independent_one`]
//! rules that out by decomposing again, by hand, and requiring `X̃` and `ž` to
//! agree **exactly**. Not approximately: `rsvd` is seed-pinned
//! (`RSVD_SUBSPACE_SEED` in matrix-util), so a drift would be a drift in the
//! recovered subspace, and a tolerance would hide precisely what this is for.
//!
//! Run: cargo test -p fagioli --test basis_reuse -- --nocapture
use anyhow::Result;
use fagioli::embedding::whiten::whiten_blocks;
use fagioli::genotype::GenotypeMatrix;
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::{decompose_blocks, SumstatInput, MIN_BLOCK_SNPS};
use fagioli::summary_stats::rss_svd::RssEigenBasis;
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
fn test_shared_basis_matches_an_independent_one() -> Result<()> {
    let input = simulate(20250808);

    let bases = decompose_blocks(&input);
    let report = calibrate_input(&input, &bases).expect("calibration");
    let lambda = report.noise.lambda_white();
    println!("λ_white = {lambda:.6} over {} blocks", bases.len());

    let whitened = whiten_blocks(&input, bases, None, lambda)?;
    assert!(
        !whitened.is_empty(),
        "nothing was whitened, so nothing is tested"
    );

    let mut compared = 0usize;
    for w in &whitened {
        let block = &input.blocks[w.block_idx];

        // Decompose again, by hand, exactly as a stage would have for itself.
        let x_block = input.standardized_block(block);
        let basis = RssEigenBasis::from_genotypes(&x_block, input.max_rank)?;
        let d_sq = basis.singular_values_sq();
        let svd = basis.into_svd(lambda);
        let z_white = svd.project_zscores(&input.block_zscores(block));

        assert_eq!(w.d_sq, d_sq, "block {} spectrum differs", w.block_idx);
        // nalgebra's PartialEq compares shape then elements, so this is the
        // bitwise check and the shape check at once. `assert!` rather than
        // `assert_eq!` so a failure does not dump the whole matrix.
        assert!(
            *svd.x_design() == w.x_design,
            "block {} design differs bitwise",
            w.block_idx,
        );
        assert!(
            z_white == w.z_white,
            "block {} whitened z differs bitwise",
            w.block_idx,
        );
        compared += w.x_design.len() + w.z_white.len() + w.d_sq.len();
    }

    println!("{compared} values identical across {} blocks", whitened.len());
    Ok(())
}

/// The two stages must describe the same blocks. They used to filter on two
/// different constants in two different modules, which meant calibration could
/// pool a block that training never saw, and λ would then be fitted partly on
/// data the model was never shown.
#[test]
fn test_both_stages_see_the_same_blocks() -> Result<()> {
    let mut input = simulate(111);
    // One block too short to decompose, so the filter is actually exercised.
    input.blocks[2].snp_end = input.blocks[2].snp_start + MIN_BLOCK_SNPS - 1;

    let bases = decompose_blocks(&input);
    let decomposed: Vec<usize> = bases.blocks().iter().map(|b| b.block_idx).collect();
    assert_eq!(
        decomposed,
        vec![0, 1, 3, 4],
        "the short block should be the only one dropped",
    );

    let report = calibrate_input(&input, &bases).expect("calibration");
    let whitened = whiten_blocks(&input, bases, None, report.noise.lambda_white())?;

    let trained: Vec<usize> = whitened.iter().map(|b| b.block_idx).collect();
    assert_eq!(
        trained, decomposed,
        "every decomposed block must reach training, and no other",
    );
    Ok(())
}

/// Where the time goes. Ignored by default: wall-clock on a shared machine is
/// not a property of the code, and this allocates panels big enough to matter.
///
/// Run: cargo test -p fagioli --release --test basis_reuse -- --ignored --nocapture
#[test]
#[ignore = "wall-clock benchmark, not a correctness check"]
fn measure_where_the_time_goes() -> Result<()> {
    use std::time::Instant;

    println!(
        "\n{:>6} {:>6} {:>6} | {:>10} {:>10} {:>10} | {:>12}",
        "blocks", "snps", "n", "decompose", "calibrate", "whiten", "2nd pass cost"
    );
    println!("{}", "-".repeat(76));

    for &(nb, spb, n, rank) in &[(8usize, 400usize, 1000usize, 300usize), (16, 500, 1500, 400)] {
        let input = simulate_sized(20250808, nb, spb, n, rank);

        let t0 = Instant::now();
        let bases = decompose_blocks(&input);
        let t_decompose = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let report = calibrate_input(&input, &bases).expect("calibration");
        let t_calibrate = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let whitened = whiten_blocks(&input, bases, None, report.noise.lambda_white())?;
        let t_whiten = t2.elapsed().as_secs_f64();
        assert_eq!(whitened.len(), nb);

        let total = t_decompose + t_calibrate + t_whiten;
        println!(
            "{nb:>6} {spb:>6} {n:>6} | {t_decompose:>10.3} {t_calibrate:>10.3} {t_whiten:>10.3} \
             | {:>11.1}%",
            100.0 * t_decompose / total,
        );
    }
    println!(
        "\nThe last column is what a second decomposition would have added, as a\n\
         share of the pass — which is what running one per stage used to cost.\n"
    );
    Ok(())
}
