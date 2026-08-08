//! Full round trip through the actual binaries: simulate genotypes, simulate
//! correlated multi-trait GWAS, then ask whether the trait geometry can be
//! recovered from the summary statistics alone.
//!
//! Unlike `embedding_recovery`, nothing here is planted by the test. `Cov_g` is
//! whatever `sim-sumstat --num-genetic-factors` produced, read back from the
//! file it writes, and the z-scores arrive through the real BGZF reader with
//! real allele matching. This is the first test that exercises
//! `estimate_trait_geometry` -- the cross-trait three-term fit -- against a
//! *known* `r_g` on realistic LD.
//!
//! Run: cargo test -p fagioli --test sumstat_geometry_roundtrip -- --nocapture

use anyhow::Result;
use candle_util::candle_core::Device;
use fagioli::embedding::diagnostics::{estimate_trait_geometry, to_correlation};
use fagioli::embedding::model::EmbedConfig;
use fagioli::embedding::noise::NoiseModel;
use fagioli::embedding::train::train;
use fagioli::embedding::whiten::whiten_blocks;
use fagioli::genotype::{BedReader, GenotypeReader};
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::SumstatInput;
use fagioli::summary_stats::{create_uniform_blocks, read_sumstat_zscores_with_n};
use flate2::read::GzDecoder;
use nalgebra::DMatrix;
use std::io::{BufRead, BufReader};
use std::process::Command;

const NUM_INDIVIDUALS: usize = 900;
const NUM_SNPS: usize = 3000;
const NUM_TRAITS: usize = 10;
const NUM_FACTORS: usize = 2;
const BLOCK_SIZE: usize = 300;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_fagioli")
}

fn run(args: &[&str]) -> Result<()> {
    let out = Command::new(bin()).args(args).output()?;
    if !out.status.success() {
        anyhow::bail!(
            "fagioli {:?} failed:\n{}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

/// Read a `{prefix}.genetic_covariance.tsv.gz` written by `sim-sumstat`.
fn read_trait_matrix(path: &str) -> Result<DMatrix<f32>> {
    let f = std::fs::File::open(path)?;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for (i, line) in BufReader::new(GzDecoder::new(f)).lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // header
        }
        let vals: Vec<f32> = line
            .split('\t')
            .skip(1)
            .map(|v| v.parse::<f32>().unwrap_or(0.0))
            .collect();
        if !vals.is_empty() {
            rows.push(vals);
        }
    }
    let t = rows.len();
    anyhow::ensure!(t > 0, "empty trait matrix at {path}");
    Ok(DMatrix::from_fn(t, t, |i, j| rows[i][j]))
}

/// Correlation between strict upper triangles.
fn offdiag_corr(a: &DMatrix<f32>, b: &DMatrix<f32>) -> f32 {
    let t = a.nrows().min(b.nrows());
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

fn mean_abs_offdiag(m: &DMatrix<f32>) -> f32 {
    let t = m.nrows();
    let mut s = 0.0;
    let mut n = 0;
    for i in 0..t {
        for j in (i + 1)..t {
            s += m[(i, j)].abs();
            n += 1;
        }
    }
    s / n.max(1) as f32
}

struct Simulated {
    input: SumstatInput,
    /// True `r_g` in correlation form, straight from the simulator's own output.
    rg_true: DMatrix<f32>,
}

/// Run `sim-geno` then `sim-sumstat`, and read the results back through the
/// real readers.
fn simulate(dir: &std::path::Path, factors: usize, seed: u64) -> Result<Simulated> {
    simulate_dense(dir, factors, seed, 12, 6)
}

/// As `simulate`, but with the causal density under the caller's control.
fn simulate_dense(
    dir: &std::path::Path,
    factors: usize,
    seed: u64,
    shared_causal: usize,
    causal_blocks: usize,
) -> Result<Simulated> {
    let geno = dir.join("geno");
    let sim = dir.join(format!("sim_f{factors}_{shared_causal}_{causal_blocks}"));
    let geno_s = geno.to_str().unwrap();
    let sim_s = sim.to_str().unwrap();

    run(&[
        "sim-geno",
        "--num-individuals",
        &NUM_INDIVIDUALS.to_string(),
        "--num-snps",
        &NUM_SNPS.to_string(),
        "--chromosome",
        "1",
        "--seed",
        &seed.to_string(),
        "--output",
        geno_s,
    ])?;

    run(&[
        "sim-sumstat",
        "--bed-prefix",
        geno_s,
        "--chromosome",
        "1",
        "--num-traits",
        &NUM_TRAITS.to_string(),
        "--num-genetic-factors",
        &factors.to_string(),
        "--num-shared-causal",
        &shared_causal.to_string(),
        "--num-independent-causal",
        "0",
        "--num-causal-blocks",
        &causal_blocks.to_string(),
        "--h2-sparse",
        "0.5",
        "--min-block-snps",
        &BLOCK_SIZE.to_string(),
        "--max-block-snps",
        &BLOCK_SIZE.to_string(),
        "--seed",
        &seed.to_string(),
        "--output",
        sim_s,
    ])?;

    // Read the panel and the z-scores exactly as a fitter would.
    let mut reader = BedReader::new(geno_s)?;
    let geno_mat = reader.read(None, None)?;
    let (zscores, median_n) = read_sumstat_zscores_with_n(
        &format!("{sim_s}.sumstats.bed.gz"),
        &geno_mat.snp_ids,
        &geno_mat.chromosomes,
        &geno_mat.positions,
        &geno_mat.allele1,
        &geno_mat.allele2,
    )?;

    let blocks = create_uniform_blocks(
        geno_mat.num_snps(),
        BLOCK_SIZE,
        &geno_mat.positions,
        &geno_mat.chromosomes,
    );

    let cov_true = read_trait_matrix(&format!("{sim_s}.genetic_covariance.tsv.gz"))?;

    Ok(Simulated {
        rg_true: to_correlation(&cov_true),
        input: SumstatInput {
            geno: geno_mat,
            zscores,
            blocks,
            median_n,
            max_rank: 150,
        },
    })
}

/// Does the cross-trait three-term fit recover the `r_g` the simulator wrote?
#[test]
fn test_trait_geometry_recovers_simulated_rg() -> Result<()> {
    let dir = tempfile::tempdir()?;

    println!("\n=== correlated architecture (H = {NUM_FACTORS}) ===");
    let sim = simulate(dir.path(), NUM_FACTORS, 20250808)?;
    println!(
        "  truth: mean |r_g| = {:.3} over {} traits",
        mean_abs_offdiag(&sim.rg_true),
        sim.rg_true.nrows()
    );

    let geometry = estimate_trait_geometry(&sim.input).expect("geometry estimate");
    let corr = offdiag_corr(&geometry.genetic_correlation, &sim.rg_true);
    println!(
        "  estimated: mean |r_g| = {:.3}; correlation with truth = {:.3}",
        mean_abs_offdiag(&geometry.genetic_correlation),
        corr
    );

    // The null architecture as a control. NOTE: the estimate is known to report
    // a large spurious r_g here -- see test_rg_estimate_is_unusable_on_realistic_ld
    // for the cause. It is printed rather than asserted on until that is fixed.
    println!("\n=== uncorrelated architecture (H = 0) ===");
    let null = simulate(dir.path(), 0, 20250808)?;
    let null_geom = estimate_trait_geometry(&null.input).expect("geometry estimate");
    println!(
        "  truth mean |r_g| = {:.3}, estimated mean |r_g| = {:.3}",
        mean_abs_offdiag(&null.rg_true),
        mean_abs_offdiag(&null_geom.genetic_correlation),
    );

    println!();
    assert!(
        mean_abs_offdiag(&sim.rg_true) > 0.3,
        "the correlated architecture should actually be correlated, got {}",
        mean_abs_offdiag(&sim.rg_true)
    );
    assert!(
        corr > 0.3,
        "cross-trait fit should track the simulated r_g, got {corr}"
    );
    Ok(())
}

/// And does the embedding's own geometry track it, end to end?
#[test]
fn test_embedding_geometry_tracks_simulated_rg() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let sim = simulate(dir.path(), NUM_FACTORS, 424242)?;

    let report = calibrate_input(&sim.input).expect("calibration");
    let lambda = report.noise.lambda_white();
    println!(
        "\ncalibration: c={:.3}, τ={:.4} (SE {:.4}), whiteness dev {:.3}{}",
        report.noise.c,
        report.noise.tau,
        report.lambda_se,
        report.deviation,
        if report.misspecified {
            "  <-- alarm"
        } else {
            ""
        },
    );

    let blocks = whiten_blocks(&sim.input, None, lambda)?;
    let d_sq: Vec<Vec<f32>> = blocks.iter().map(|b| b.d_sq.clone()).collect();
    let noise = NoiseModel::new(&d_sq, report.noise.c, report.noise.tau, lambda);

    let fit = train(
        &blocks,
        &noise,
        &EmbedConfig {
            embedding_dim: NUM_FACTORS,
            num_negatives: 4,
            prior_inclusion: 0.02,
            learning_rate: 0.05,
            num_iterations: 300,
            grad_clip: Some(10.0),
            dense_arm: false,
            gauge_weight: 0.0,
            seed: 5,
        },
        &Device::Cpu,
    )?;

    let corr = offdiag_corr(&to_correlation(&fit.trait_geometry), &sim.rg_true);
    println!(
        "embedding: geometry vs simulated r_g = {:.3} (offset {:+.3}, loss {:.4} -> {:.4})\n",
        corr,
        fit.offset,
        fit.loss_trace.first().copied().unwrap_or(0.0),
        fit.loss_trace.last().copied().unwrap_or(0.0),
    );

    assert!(
        corr > 0.3,
        "the fitted geometry should track the simulated r_g, got {corr}"
    );
    Ok(())
}

/// Why the null architecture reports a large `mean |r_g|`: normalising a
/// near-zero covariance to correlation form divides noise by noise. This
/// separates the two by looking at the covariance scale and at whether the
/// entries are significant against their own block jackknife.
#[test]
fn test_null_rg_is_noise_not_signal() -> Result<()> {
    use fagioli::summary_stats::calibration::delete_a_group_se;

    let dir = tempfile::tempdir()?;

    for (label, factors) in [("correlated", NUM_FACTORS), ("null", 0)] {
        let sim = simulate(dir.path(), factors, 20250808)?;
        let geom = estimate_trait_geometry(&sim.input).expect("geometry");
        let t = sim.rg_true.nrows();

        // Genome-wide covariance, and its delete-a-group SE per entry.
        let mut sum = DMatrix::<f32>::zeros(t, t);
        for m in &geom.per_block {
            sum += &m.genetic_cov;
        }
        let sizes: Vec<usize> = geom.per_block.iter().map(|m| m.block_snps).collect();

        let mut n_sig = 0usize;
        let mut n_pairs = 0usize;
        let mut max_z: f32 = 0.0;
        for i in 0..t {
            for j in (i + 1)..t {
                let per_block: Vec<f32> =
                    geom.per_block.iter().map(|m| m.genetic_cov[(i, j)]).collect();
                let se = delete_a_group_se(&per_block, &sizes);
                let est = sum[(i, j)] / geom.per_block.len() as f32;
                let z = if se > 0.0 { est / se } else { 0.0 };
                max_z = max_z.max(z.abs());
                if z.abs() > 2.0 {
                    n_sig += 1;
                }
                n_pairs += 1;
            }
        }

        // Scale of the covariance itself, diagonal vs off-diagonal.
        let mean_diag: f32 = (0..t).map(|i| sum[(i, i)].abs()).sum::<f32>() / t as f32;
        let mean_off = mean_abs_offdiag(&sum);

        println!(
            "[{label:>10}] true |r_g| {:.3} | est |r_g| {:.3} | \
             cov: diag {:.3e}, offdiag {:.3e} (ratio {:.3}) | \
             significant pairs {n_sig}/{n_pairs}, max |z| {max_z:.2}",
            mean_abs_offdiag(&sim.rg_true),
            mean_abs_offdiag(&geom.genetic_correlation),
            mean_diag,
            mean_off,
            mean_off / mean_diag.max(1e-12),
        );
    }
    println!();
    Ok(())
}

/// Characterisation of a known defect: `estimate_trait_geometry` does **not**
/// work on realistic LD, and the sweep below records how it fails.
///
/// The genetic covariance is read off the `d⁴` coefficient of the three-term
/// eigenspace fit. On genotype-like spectra `corr(d⁴, d²) = 0.994`, so the
/// variance inflation factor for that coefficient is `1/(1 − 0.994²) ≈ 84`:
/// `d⁴` and `d²` trade off against each other with enormous variance while
/// only their *combination* is determined. On the diagonal the combination is
/// genuinely large, so the split hardly matters. Off the diagonal the truth is
/// zero, and the inflated noise in the `d⁴` coefficient is comparable to the
/// diagonal — which is exactly the spurious `r_g ≈ 0.6` seen here.
///
/// The block jackknife does not catch it. Every block shares the same
/// collinear design, so the inflated coefficient is *stable* across blocks and
/// the delete-a-group SE measures between-block variability rather than the
/// within-block collinearity variance. Entries therefore look significant
/// (`|z|` up to 4.6) while being pure artefact.
///
/// A first guess -- that this was a sparse-architecture artefact, since the
/// moment law assumes infinitesimal effects -- is ruled out below: the estimate
/// stays near 0.6 as the causal count goes 72 -> 2000 while the truth falls to
/// 0.02.
///
/// **When this is fixed the assertion here should be inverted.** Any fix has to
/// address the conditioning: orthogonalise `d⁴` against `d²` and report only
/// the identified combination, regularise the 3x3 solve, or estimate `r_g` by
/// cross-trait LDSC in SNP space instead.
#[test]
fn test_rg_estimate_is_unusable_on_realistic_ld() -> Result<()> {
    let dir = tempfile::tempdir()?;
    println!("\ncausal SNPs | true |r_g| | estimated |r_g|   (uncorrelated architecture)");

    let mut worst_gap = 0.0f32;
    for &(shared, blocks) in &[(12usize, 6usize), (60, 10), (200, 10)] {
        let sim = simulate_dense(dir.path(), 0, 20250808, shared, blocks)?;
        let geom = estimate_trait_geometry(&sim.input).expect("geometry");
        let (t_rg, e_rg) = (
            mean_abs_offdiag(&sim.rg_true),
            mean_abs_offdiag(&geom.genetic_correlation),
        );
        println!("{:>11} | {:>10.3} | {:>15.3}", shared * blocks, t_rg, e_rg);
        worst_gap = worst_gap.max(e_rg - t_rg);
    }
    println!();

    assert!(
        worst_gap > 0.3,
        "the spurious r_g appears to be gone (worst gap {worst_gap:.3}). If the \
         conditioning has been fixed, invert this assertion and re-enable the \
         recovery check in test_trait_geometry_recovers_simulated_rg."
    );
    Ok(())
}
