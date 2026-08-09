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
//! **These numbers do not reproduce run to run, and that is not this test's
//! doing.** `EmbedConfig::seed` reaches the NCE negatives but not the parameter
//! initialisation: `V̌` is drawn with `Init::Randn`, and candle's CPU backend
//! cannot be seeded — `Device::set_seed` errors outright there. Repeating the
//! sweep on one binary moves an AUC by up to ~0.12, which is the same size as
//! the seed-to-seed spread reported below. So read a single cell as indicative
//! and the trend across a column as the result. The offset column is the
//! exception and is stable to ~0.01, which is why the overfitting conclusion is
//! the firmer of the two.
//!
//! That caution turned out to be the finding. [`test_sweep_embedding_dim`]
//! varies `H` with everything else fixed, and the two columns move in opposite
//! directions: trait-specific-vs-null climbs from 0.66 to 0.86 as `H` goes 1 to
//! 20, while pleiotropic-vs-specific falls from its 0.65 peak at `H = 2` to
//! 0.42, i.e. through chance and out the other side. The `H` values where the
//! contrast looks best are exactly the ones where the model half-sees the
//! trait-specific class. So `‖u_g‖` is not a pleiotropy statistic at any `H`;
//! it is a detection statistic, and a good one — pleiotropic-vs-null holds
//! 0.70-0.86 across the whole sweep and needs no tuning.
//!
//! Run: cargo test -p fagioli --test pleiotropy_discrimination -- --nocapture
use anyhow::Result;
use candle_util::candle_core::Device;
use fagioli::embedding::model::{EmbedConfig, UPrior};
use fagioli::embedding::noise::NoiseModel;
use fagioli::embedding::train::{train, EmbedFit};
use fagioli::embedding::whiten::{whiten_blocks, WhitenedBlock};
use fagioli::summary_stats::calibration::calibrate_input;
use fagioli::summary_stats::common::decompose_blocks;
use rustc_hash::FxHashSet as HashSet;

#[path = "common/three_class.rs"]
mod three_class;
use three_class::{auc_pair, simulate_classes, NUM_PROGRAMS, NUM_TRAITS};

fn variant_norms(fit: &EmbedFit) -> Vec<f32> {
    let mut out = Vec::new();
    for u in &fit.u_mean {
        for g in 0..u.nrows() {
            out.push((0..u.ncols()).map(|h| u[(g, h)].powi(2)).sum::<f32>().sqrt());
        }
    }
    out
}


/// Everything upstream of the fit. None of it depends on `H`, so the sweep
/// below pays for it once per seed and varies only the program dimension.
struct Prepared {
    blocks: Vec<WhitenedBlock>,
    noise: NoiseModel,
    pleiotropic: HashSet<usize>,
    trait_specific: HashSet<usize>,
    null: HashSet<usize>,
    rms_norm_pleio: f32,
    rms_norm_specific: f32,
}

fn prepare(h2: f32, seed: u64) -> Result<Prepared> {
    let cl = simulate_classes(h2, seed);
    let all: HashSet<usize> = cl.pleiotropic.union(&cl.trait_specific).copied().collect();
    let null: HashSet<usize> = (0..cl.input.geno.num_snps())
        .filter(|j| !all.contains(j))
        .collect();

    let input = cl.input;
    let bases = decompose_blocks(&input);
    let report = calibrate_input(&input, &bases).expect("calibration");
    let lambda = report.noise.lambda_white();
    let blocks = whiten_blocks(&input, bases, None, lambda)?;
    let d_sq: Vec<Vec<f32>> = blocks.iter().map(|x| x.d_sq.clone()).collect();
    let noise = NoiseModel::new(&d_sq, report.noise.c, report.noise.tau, lambda);

    Ok(Prepared {
        blocks,
        noise,
        pleiotropic: cl.pleiotropic,
        trait_specific: cl.trait_specific,
        null,
        rms_norm_pleio: cl.rms_norm_pleio,
        rms_norm_specific: cl.rms_norm_specific,
    })
}

/// One fit's three AUCs and its NCE offset.
struct Scored {
    pleio_vs_null: f32,
    specific_vs_null: f32,
    pleio_vs_specific: f32,
    offset: f32,
}

/// Fit at program dimension `h`. Every other knob is fixed, so a difference
/// between two calls is a difference in `H` and nothing else.
fn fit_at(prep: &Prepared, h: usize, seed: u64) -> Result<Scored> {
    let fit = train(
        &prep.blocks,
        &prep.noise,
        &EmbedConfig {
            embedding_dim: h,
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
    Ok(Scored {
        pleio_vs_null: auc_pair(&norms, &prep.pleiotropic, &prep.null),
        specific_vs_null: auc_pair(&norms, &prep.trait_specific, &prep.null),
        pleio_vs_specific: auc_pair(&norms, &prep.pleiotropic, &prep.trait_specific),
        offset: fit.offset,
    })
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
        let prep = prepare(0.4, seed)?;
        let s = fit_at(&prep, NUM_PROGRAMS, seed)?;
        println!(
            "{seed:>8} | {:>14.3} {:>14.3} {:>14.3}",
            s.pleio_vs_null, s.specific_vs_null, s.pleio_vs_specific
        );
        a += s.pleio_vs_null;
        b_ += s.specific_vs_null;
        c += s.pleio_vs_specific;
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

/// Does `H` explain the weak pleiotropy result, and does a large `H` overfit?
///
/// Two questions, one sweep, because they share an instrument.
///
/// The first is the mechanism proposed when `pleio vs specific` came back at
/// 0.668: with `H = 3` against `T = 10` the programs span a 3-dimensional
/// subspace of trait space, a single-trait direction has a generic non-zero
/// projection onto it, and so an off-program effect is partly absorbed rather
/// than ignored. If that is the whole story then `H` is the knob, and the
/// contrast should move with it.
///
/// The second is whether picking `H` needs cross-validation at all. The NCE
/// offset is a validated overfitting instrument — it is not a free parameter,
/// since the score is exactly normalised, so what it measures is the model
/// memorising a fixed set of positives against negatives that are redrawn every
/// step (`+0.907 -> -0.027` as samples-per-parameter improved). Raising `H`
/// raises `p·H` parameters per block against a fixed `K·T` of data, so if the
/// offset stays flat across the sweep there is nothing to select against, and
/// if it climbs it says where to stop without any held-out data.
///
/// Run: cargo test -p fagioli --test pleiotropy_discrimination \
///        sweep_embedding_dim -- --nocapture
#[test]
fn test_sweep_embedding_dim() -> Result<()> {
    let seeds = [20250808u64, 111, 222, 333, 444];
    let dims = [1usize, 2, 3, 5, 8, 10, 15, 20];

    // Upstream of the fit and independent of H, so it is paid once per seed.
    let prepared: Vec<(u64, Prepared)> = seeds
        .iter()
        .map(|&s| Ok((s, prepare(0.4, s)?)))
        .collect::<Result<Vec<_>>>()?;

    let (rank, num_snps) = prepared
        .first()
        .and_then(|(_, p)| p.blocks.first())
        .map(|b| (b.rank(), b.num_snps))
        .unwrap_or((0, 0));
    println!(
        "\n{} traits, {} true programs, {} variants and {} eigen-coordinates per block.\n\
         U carries p*H = {}*H free rows against K*T = {} whitened observations.\n",
        NUM_TRAITS,
        NUM_PROGRAMS,
        num_snps,
        rank,
        num_snps,
        rank * NUM_TRAITS,
    );

    // ‖u_g‖ is a magnitude, so it can only be reading structure if the two
    // classes carry the same planted magnitude.
    let (mp, ms) = prepared.iter().fold((0.0f32, 0.0f32), |(p, s), (_, x)| {
        (p + x.rms_norm_pleio, s + x.rms_norm_specific)
    });
    let n_seeds = prepared.len() as f32;
    println!(
        "planted RMS ‖β_g‖: pleiotropic {:.3}, trait-specific {:.3} (ratio {:.2})\n",
        mp / n_seeds,
        ms / n_seeds,
        (ms / n_seeds) / (mp / n_seeds).max(1e-6),
    );
    println!(
        "{:>3} {:>9} | {:>13} {:>13} {:>15} | {:>14}",
        "H", "obs/param", "pleio vs null", "spec vs null", "pleio vs spec", "offset"
    );
    println!("{}", "-".repeat(80));

    // One row per H, holding that row's across-seed means.
    let mut by_dim: Vec<Scored> = Vec::with_capacity(dims.len());

    for &h in &dims {
        let scored: Vec<Scored> = prepared
            .iter()
            .map(|(seed, prep)| fit_at(prep, h, *seed))
            .collect::<Result<Vec<_>>>()?;
        let stat = |get: fn(&Scored) -> f32| mean_sd(&scored.iter().map(get).collect::<Vec<_>>());

        let obs_per_param = (rank * NUM_TRAITS) as f32 / (num_snps * h) as f32;
        let pn = stat(|s| s.pleio_vs_null);
        let sn = stat(|s| s.specific_vs_null);
        let ps = stat(|s| s.pleio_vs_specific);
        let off = stat(|s| s.offset);
        println!(
            "{h:>3} {obs_per_param:>9.2} | {:>13.3} {:>13.3} | {:>7.3} ±{:<6.3} | {:>+7.3} ±{:<5.3}",
            pn.0, sn.0, ps.0, ps.1, off.0, off.1,
        );
        by_dim.push(Scored {
            pleio_vs_null: pn.0,
            specific_vs_null: sn.0,
            pleio_vs_specific: ps.0,
            offset: off.0,
        });
    }

    println!(
        "\nobs/param is K*T per p*H, the ratio the offset was shown to track. A flat\n\
         offset would mean large H does not overfit and needs no held-out selection.\n\
         A climbing one names the ceiling instead, and does it without holding\n\
         anything out.\n"
    );

    let at = |h: usize| dims.iter().position(|&d| d == h).expect("dim in sweep");
    let (lo, mid, hi) = (at(1), at(3), at(20));

    // Characterisation, in the sense of f9175a8d: these assert what is *wrong*,
    // so that a fix trips them rather than passing silently.

    // The offset responds to H the way it responded to samples-per-parameter.
    // If this stops holding, either the instrument broke or the model stopped
    // overfitting, and both are worth stopping for.
    assert!(
        by_dim[hi].offset > by_dim[lo].offset + 0.3,
        "offset should climb with H: {:+.3} at H=1 against {:+.3} at H=20",
        by_dim[lo].offset,
        by_dim[hi].offset,
    );

    // Raising H past the truth does not rescue the pleiotropy contrast. This is
    // the claim the sweep was run to test, and it failed.
    assert!(
        by_dim[hi].pleio_vs_specific <= by_dim[mid].pleio_vs_specific,
        "more programs did not help the contrast, yet H=20 scored {:.3} against \
         {:.3} at the planted H=3",
        by_dim[hi].pleio_vs_specific,
        by_dim[mid].pleio_vs_specific,
    );

    // ...because the extra programs go to *representing* the trait-specific
    // class, not to setting it apart.
    assert!(
        by_dim[hi].specific_vs_null > by_dim[lo].specific_vs_null,
        "trait-specific detection should improve with H: {:.3} at H=1 against \
         {:.3} at H=20",
        by_dim[lo].specific_vs_null,
        by_dim[hi].specific_vs_null,
    );

    // Detection, unlike discrimination, barely depends on H at all.
    let pn_min = by_dim
        .iter()
        .map(|s| s.pleio_vs_null)
        .fold(f32::INFINITY, f32::min);
    assert!(
        pn_min > 0.6,
        "pleiotropic-vs-null should stay a good detector at every H, worst was {pn_min:.3}",
    );

    Ok(())
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

fn mean_sd(v: &[f32]) -> (f32, f32) {
    let m = mean(v);
    if v.len() < 2 {
        return (m, 0.0);
    }
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / (v.len() - 1) as f32;
    (m, var.sqrt())
}
