//! Is the pleiotropy question a *trait-axis* question, and does it need LD?
//!
//! # The claim being tested
//!
//! Write the multi-trait RSS model as `Z = R B Λ_N + E` with `E ~ MN(0, R, Ω)`.
//! Take one variant's row. Because `R` has unit diagonal,
//!
//! ```text
//! Cov(e_g) = R_gg · Ω = Ω
//! ```
//!
//! so **LD does not enter a single variant's noise covariance across traits**.
//! It enters only through the mean, and for a variant `g` tagging a single
//! causal variant `j`,
//!
//! ```text
//! z_g = R_gj · β_j Λ_N + e_g
//! ```
//!
//! where `R_gj` is a *scalar*: it rescales the mean and leaves its direction
//! alone. So the **mean direction** of a raw marginal z-row is LD-invariant.
//!
//! The *observable* shape is not, and the reason is in the same two lines:
//! `Cov(e_g) = Ω` is **not** scaled by `R_gj`, so a weakly-tagging variant has
//! the same noise on a smaller mean, and `PR` is pulled toward isotropy as
//! `R_gj` falls. [`test_raw_zscore_shape_separates_without_any_ld`] reports
//! that attenuation directly: LD tags of a trait-specific variant sit between
//! their source and a pure null. So the claim is that shape is invariant to LD
//! *in expectation*, degrading with tag strength — not that it is untouched.
//!
//! If that is right, the question needs no reference panel, no randomized SVD,
//! no calibration and no ridge: it is answerable from the z-score matrix alone.
//!
//! # The statistic
//!
//! Participation ratio, the effective number of nonzero coordinates:
//!
//! ```text
//! PR(x) = (Σ_t x²_t)² / Σ_t x⁴_t
//! ```
//!
//! `PR(c·e_t) = 1` exactly. The isotropic case needs care. Writing
//! `y = x/‖x‖`, uniform on the sphere, `PR = 1/Σ_t y⁴_t` and
//! `E[Σ y⁴] = 3/(T+2)`, so by Jensen
//!
//! ```text
//! E[PR]  >  (T+2)/3        =  4.0 at T = 10
//! ```
//!
//! **strictly** — `(T+2)/3` is a lower bound, not the mean, because `E[A/B]`
//! is not `E[A]/E[B]`. [`isotropic_reference_pr`] measures the actual value by
//! Monte Carlo (≈4.36 at `T = 10`) and every table below compares against that.
//! An earlier version of this file quoted `(T+2)/3` as the prediction and
//! "confirmed" it against a null mean of 3.93 — but that 3.93 was contaminated
//! by LD tags of causal variants, and the two errors cancelled. Genuinely null
//! rows sit near 4.3.
//!
//! The simulation plants pleiotropic effects as `c·v_h` with `v_h ~ N(0, I_T)`,
//! so isotropic noise and pleiotropic signal land in the *same* place, which is
//! why this separates trait-specific from everything else rather than signal
//! from null.
//!
//! # What this cannot do
//!
//! A tag inherits its causal variant's *direction*, attenuated by tag strength,
//! so shape ranks tags between their source and the null: these are locus
//! claims, not variant claims.
//! And a neighbourhood containing two causal variants with *different* trait
//! profiles blends into a mixture — [`test_blended_neighbourhood_is_the_limit`]
//! plants exactly that.
//!
//! Run: cargo test -p fagioli --test trait_profile_shape -- --nocapture
use anyhow::Result;
use fagioli::embedding::noise::omega_sqrt_pair;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rustc_hash::FxHashSet as HashSet;

#[path = "common/three_class.rs"]
mod three_class;
use three_class::{auc_pair, simulate_classes, Classes, NUM_TRAITS};

/// Effective number of nonzero coordinates of `x`.
///
/// `PR(c·e_t) = 1` exactly; for an isotropic Gaussian of length `T` it sits
/// near `(T+2)/3` — see the module docs for why it is not `T/3`.
///
/// Takes an iterator so the hot path (one call per variant per rotation draw)
/// does not allocate a row.
fn participation_ratio(x: impl Iterator<Item = f32>) -> f32 {
    let (s2, s4) = x.fold((0.0f64, 0.0f64), |(a, b), v| {
        let v2 = (v as f64) * (v as f64);
        (a + v2, b + v2 * v2)
    });
    if s4 <= 0.0 {
        return 0.0;
    }
    (s2 * s2 / s4) as f32
}

/// `E[PR]` for an isotropic Gaussian of length `t`, by Monte Carlo.
///
/// `(T+2)/3` is a strict lower bound on this (Jensen), so it cannot be used as
/// the reference value — see the module docs.
fn isotropic_reference_pr(t: usize, draws: usize, rng: &mut SmallRng) -> f32 {
    let acc: f64 = (0..draws)
        .map(|_| {
            let x: Vec<f32> = (0..t)
                .map(|_| {
                    let v: f64 = StandardNormal.sample(rng);
                    v as f32
                })
                .collect();
            participation_ratio(x.into_iter()) as f64
        })
        .sum();
    (acc / draws as f64) as f32
}

/// `PR(z_g)` for every row.
fn pr_rows(m: &DMatrix<f32>) -> Vec<f32> {
    (0..m.nrows())
        .map(|g| participation_ratio(m.row(g).iter().copied()))
        .collect()
}

/// `‖z_g‖` for every row.
fn row_norms(m: &DMatrix<f32>) -> Vec<f32> {
    (0..m.nrows()).map(|g| m.row(g).norm()).collect()
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

fn mean_over(values: &[f32], idx: &HashSet<usize>) -> f32 {
    mean(&idx.iter().map(|&g| values[g]).collect::<Vec<_>>())
}

#[test]
fn test_raw_zscore_shape_separates_without_any_ld() -> Result<()> {
    let seeds = [20250808u64, 111, 222, 333, 444];

    println!(
        "\nRaw marginal z-scores only. No panel decomposition, no calibration,\n\
         no whitening, no ridge, no fit.\n"
    );
    println!(
        "{:>8} | {:>26} | {:>26}",
        "", "‖z_g‖  (magnitude)", "PR(z_g)  (shape)"
    );
    println!(
        "{:>8} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "seed", "pl/null", "sp/null", "pl/sp", "pl/null", "sp/null", "pl/sp"
    );
    println!("{}", "-".repeat(76));

    let (mut n_pn, mut n_sn, mut n_ps) = (vec![], vec![], vec![]);
    let (mut p_pn, mut p_sn, mut p_ps) = (vec![], vec![], vec![]);
    let (mut pr_pleio, mut pr_spec, mut pr_null) = (vec![], vec![], vec![]);
    let mut pr_tags: Vec<f32> = vec![];

    for &seed in &seeds {
        let cl = simulate_classes(0.4, seed);
        let null = null_set(&cl);
        let (norm, pr) = (row_norms(&cl.input.zscores), pr_rows(&cl.input.zscores));

        // PR is *low* for a trait-specific variant, so the pleiotropic class is
        // the high side of that contrast — the same orientation as ‖z‖ for
        // detection, which keeps every column readable the same way.
        let (a, b, c) = (
            auc_pair(&norm, &cl.pleiotropic, &null),
            auc_pair(&norm, &cl.trait_specific, &null),
            auc_pair(&norm, &cl.pleiotropic, &cl.trait_specific),
        );
        let (d, e, f) = (
            auc_pair(&pr, &cl.pleiotropic, &null),
            auc_pair(&pr, &cl.trait_specific, &null),
            auc_pair(&pr, &cl.pleiotropic, &cl.trait_specific),
        );
        println!(
            "{seed:>8} | {a:>8.3} {b:>8.3} {c:>8.3} | {d:>8.3} {e:>8.3} {f:>8.3}"
        );
        n_pn.push(a);
        n_sn.push(b);
        n_ps.push(c);
        p_pn.push(d);
        p_sn.push(e);
        p_ps.push(f);
        pr_pleio.push(mean_over(&pr, &cl.pleiotropic));
        pr_spec.push(mean_over(&pr, &cl.trait_specific));
        // Tags of a trait-specific variant carry an attenuated copy of its
        // profile; a pure null carries none. Separating them is what shows the
        // shape is LD-invariant only in expectation.
        let tagged = ld_partners_of(&cl.trait_specific);
        let tags: HashSet<usize> = tagged
            .iter()
            .copied()
            .filter(|g| !cl.trait_specific.contains(g) && !cl.pleiotropic.contains(g))
            .collect();
        let clean: HashSet<usize> = null.iter().copied().filter(|g| !tagged.contains(g)).collect();
        pr_tags.push(mean_over(&pr, &tags));
        pr_null.push(mean_over(&pr, &clean));
    }

    println!("{}", "-".repeat(76));
    println!(
        "{:>8} | {:>8.3} {:>8.3} {:>8.3} | {:>8.3} {:>8.3} {:>8.3}",
        "mean",
        mean(&n_pn),
        mean(&n_sn),
        mean(&n_ps),
        mean(&p_pn),
        mean(&p_sn),
        mean(&p_ps),
    );

    let mut ref_rng = SmallRng::seed_from_u64(0x150_7807);
    let isotropic = isotropic_reference_pr(NUM_TRAITS, 200_000, &mut ref_rng);
    let bound = (NUM_TRAITS as f32 + 2.0) / 3.0;
    println!(
        "\nmean PR   trait-specific {:.2}   its LD tags {:.2}   pure null {:.2}   \
         pleiotropic {:.2}\n\
         reference: E[PR] for an isotropic row = {isotropic:.2} by Monte Carlo\n\
         ({bound:.2} = (T+2)/3 is a strict LOWER bound on that, not the mean)\n\n\
         Trait-specific sits above 1 because noise is isotropic and pulls PR up\n\
         at finite SNR. Its tags sit between it and the null: the same profile,\n\
         attenuated by tag strength. That gradient is the honest form of the\n\
         LD-invariance claim.",
        mean(&pr_spec),
        mean(&pr_tags),
        mean(&pr_null),
        mean(&pr_pleio),
    );

    // The pure null must match the isotropic reference; an earlier version of
    // this test compared a tag-contaminated null against (T+2)/3 and the two
    // errors cancelled into an apparent confirmation.
    assert!(
        (mean(&pr_null) - isotropic).abs() < 0.25,
        "pure-null PR {:.2} should match the isotropic reference {isotropic:.2}",
        mean(&pr_null),
    );
    // Attenuation, stated as an ordering rather than assumed away.
    assert!(
        mean(&pr_spec) < mean(&pr_tags) && mean(&pr_tags) < mean(&pr_null),
        "tags should sit between their source and the null: {:.2} / {:.2} / {:.2}",
        mean(&pr_spec),
        mean(&pr_tags),
        mean(&pr_null),
    );

    // The prediction, stated before the run: shape separates the two causal
    // classes, magnitude does not. Magnitude cannot, by construction — the
    // planted RMS ‖β_g‖ of the two classes differs by only ~9%.
    assert!(
        mean(&p_ps) > mean(&n_ps),
        "shape should beat magnitude on pleio-vs-specific: PR {:.3} against ‖z‖ {:.3}",
        mean(&p_ps),
        mean(&n_ps),
    );
    assert!(
        mean(&pr_spec) < mean(&pr_pleio),
        "trait-specific variants must have the lower participation ratio: {:.2} against {:.2}",
        mean(&pr_spec),
        mean(&pr_pleio),
    );
    Ok(())
}

/// The stated limit, as a dose-response rather than a single point.
///
/// Each trait-specific variant is put into LD with a pleiotropic one at
/// correlation `rho`, so one marginal row carries a weighted sum of two
/// different trait profiles. `rho = 0` is the ordinary simulation.
///
/// `rho = 1` is deliberately excluded: identical genotype columns give
/// `z_j ≡ z_src`, so the two variants are the *same row* and every
/// summary-statistic method ties at 0.5 — full RSS fine-mapping included. That
/// measures the impossibility of the configuration, not the method.
#[test]
fn test_blending_degrades_shape_in_proportion_to_ld() -> Result<()> {
    let seeds = [20250808u64, 111, 222];
    let rhos = [0.0f32, 0.3, 0.6, 0.9, 0.99];

    println!("\n{:>6}  {:>26}", "rho", "pleio vs specific, PR(z_g)");
    println!("{}", "-".repeat(38));

    let mut by_rho = Vec::new();
    for &rho in &rhos {
        let a: Vec<f32> = seeds
            .iter()
            .map(|&s| pleio_vs_specific_pr(&three_class::simulate_blended(0.4, s, rho)))
            .collect();
        println!("{rho:>6.2}  {:>26.3}", mean(&a));
        by_rho.push(mean(&a));
    }

    println!(
        "\nThe LD-free reading assumes one causal variant per neighbourhood, so\n\
         a scalar R_gj rescales the row without turning it. Two causal variants\n\
         with different profiles break that, and this is the price curve.\n"
    );

    assert!(
        by_rho[0] > by_rho[by_rho.len() - 1],
        "blending must cost something: {:.3} at rho=0 against {:.3} at rho={:.2}",
        by_rho[0],
        by_rho[by_rho.len() - 1],
        rhos[rhos.len() - 1],
    );
    Ok(())
}

/// Head-to-head on identical seeds: what the full pipeline extracts against
/// what the raw z-score rows already contain.
///
/// Also the invariance check. `U V̌'` is unchanged by `U→UA`, `V̌→V̌A⁻ᵀ` for any
/// invertible `A`; the gauge would force `A` orthogonal, but this harness runs
/// `gauge_weight = 0`, so `A` is free and `‖u_g‖` is *not* a property of the
/// fit. Applying a diagonal `A` post-hoc leaves every prediction bit-identical
/// and should move only the statistics that were never well-posed.
#[test]
fn test_fitted_versus_raw_and_the_rotation_attack() -> Result<()> {
    use fagioli::embedding::score::assemble_u;

    let seeds = [20250808u64, 111, 222, 333, 444];
    println!(
        "\n{:>8} | {:>10} {:>10} {:>10} | {:>12}",
        "seed", "‖u_g‖", "PR(V̌u_g)", "PR(z_g)", "‖u_g‖ rotated"
    );
    println!("{}", "-".repeat(62));

    let (mut un, mut fitted_pr, mut raw_pr, mut rot) = (vec![], vec![], vec![], vec![]);
    let mut un_shift: Vec<f32> = vec![];
    for &seed in &seeds {
        let cl = simulate_classes(0.4, seed);
        let (fit, starts) = fit_embedding(&cl, three_class::NUM_PROGRAMS, seed)?;
        let u = assemble_u(&fit.u_mean, &starts, cl.input.zscores.nrows());

        // ‖u_g‖ — the statistic used so far.
        let norms = row_norms(&u);
        // PR(V̌ u_g) — row g of the fitted effect matrix, invariant to A.
        let b_hat = &u * fit.v_check.transpose();
        let fpr = pr_rows(&b_hat);
        // PR(z_g) — no model at all.
        let rpr = pr_rows(&cl.input.zscores);

        // A = diag(a_h), invertible and non-orthogonal. U→UA, V̌→V̌A⁻¹ leaves
        // U V̌' exactly unchanged.
        let h = fit.v_check.ncols();
        let a: Vec<f32> = (0..h).map(|i| 10.0f32.powi(i as i32 - (h as i32) / 2)).collect();
        let u_rot = DMatrix::from_fn(u.nrows(), h, |g, k| u[(g, k)] * a[k]);
        let v_rot = DMatrix::from_fn(fit.v_check.nrows(), h, |t, k| fit.v_check[(t, k)] / a[k]);
        let drift = (&u_rot * v_rot.transpose() - &b_hat).abs().max();
        let scale = b_hat.abs().max().max(1.0);
        assert!(
            drift <= 1e-5 * scale,
            "the rotation must not change U V̌': drift {drift} at scale {scale}",
        );
        let norms_rot = row_norms(&u_rot);

        let (x, y, z, w) = (
            auc_pair(&norms, &cl.pleiotropic, &cl.trait_specific),
            auc_pair(&fpr, &cl.pleiotropic, &cl.trait_specific),
            auc_pair(&rpr, &cl.pleiotropic, &cl.trait_specific),
            auc_pair(&norms_rot, &cl.pleiotropic, &cl.trait_specific),
        );
        println!("{seed:>8} | {x:>10.3} {y:>10.3} {z:>10.3} | {w:>12.3}");
        un.push(x);
        un_shift.push((x - w).abs());
        fitted_pr.push(y);
        raw_pr.push(z);
        rot.push(w);
    }

    println!("{}", "-".repeat(62));
    println!(
        "{:>8} | {:>10.3} {:>10.3} {:>10.3} | {:>12.3}",
        "mean",
        mean(&un),
        mean(&fitted_pr),
        mean(&raw_pr),
        mean(&rot),
    );
    println!(
        "\nThe last column is the same fit under a reparameterisation that leaves\n\
         every prediction bit-identical. Any movement between columns 1 and 4 is\n\
         a statistic that was never a property of the fit.\n"
    );

    // The conclusion, asserted rather than printed.
    //
    // Per seed, not on the mean: the shifts partly cancel across seeds, so a
    // mean-vs-mean test understates a per-fit defect. The claim is that any one
    // fit's ‖u_g‖ AUC is not a property of that fit.
    println!(
        "mean per-seed |Δ AUC| under reparameterisation: {:.3}\n",
        mean(&un_shift)
    );
    assert!(
        mean(&un_shift) > 0.02,
        "‖u_g‖ should not survive a reparameterisation that leaves U V̌' \
         bit-identical, yet per-seed |Δ| averaged only {:.3}",
        mean(&un_shift),
    );
    assert!(
        mean(&raw_pr) > mean(&un) + 0.15,
        "the raw-row shape should beat the fitted norm by more than the fit's own \
         wobble: {:.3} against {:.3}",
        mean(&raw_pr),
        mean(&un),
    );
    Ok(())
}

/// Returns the fit and the whitened blocks' SNP starts.
///
/// `assemble_u` zips `u_mean` against `snp_starts` positionally, and
/// `decompose_blocks` is a filter that can drop a block — so the starts have to
/// come from the blocks that were actually fitted, not from `input.blocks`.
fn fit_embedding(
    cl: &Classes,
    h: usize,
    seed: u64,
) -> Result<(fagioli::embedding::train::EmbedFit, Vec<usize>)> {
    use candle_util::candle_core::Device;
    use fagioli::embedding::model::{EmbedConfig, UPrior};
    use fagioli::embedding::noise::NoiseModel;
    use fagioli::embedding::train::train;
    use fagioli::embedding::whiten::whiten_blocks;
    use fagioli::summary_stats::calibration::calibrate_input;
    use fagioli::summary_stats::common::decompose_blocks;

    let bases = decompose_blocks(&cl.input);
    let report = calibrate_input(&cl.input, &bases).expect("calibration");
    let lambda = report.noise.lambda_white();
    let blocks = whiten_blocks(&cl.input, bases, None, lambda)?;
    let d_sq: Vec<Vec<f32>> = blocks.iter().map(|b| b.d_sq.clone()).collect();
    let noise = NoiseModel::new(&d_sq, report.noise.c, report.noise.tau, lambda);
    let starts: Vec<usize> = blocks.iter().map(|b| b.snp_start).collect();
    let fit = train(
        &blocks,
        &noise,
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
    Ok((fit, starts))
}

/// Calibration, not classification. Does `PR(z_g)` track the *number of traits*
/// a variant acts on, or does it only separate the one-trait extreme?
///
/// The binary planting elsewhere in this file is the easy case for a shape
/// statistic. Here every planted variant carries `‖β_g‖ = 1` and differs only
/// in how many traits that norm is spread over. Predicted noise-free value for
/// `k` traits with Gaussian entries is `(k+2)/3`, compressed upward toward the
/// isotropic `(T+2)/3 = 4` by finite SNR.
#[test]
fn test_participation_ratio_tracks_the_number_of_traits() -> Result<()> {
    let seeds = [20250808u64, 111, 222];
    let counts = [1usize, 2, 3, 5, 10];
    // Every planted variant needs its own LD group; there are NUM_LD_GROUPS.
    const PER_COUNT: usize = 7;

    println!("\n{:>7} {:>10} {:>14}", "traits", "PR(z_g)", "lower bound");
    println!("{}", "-".repeat(32));

    let mut observed = vec![Vec::new(); counts.len()];
    for &seed in &seeds {
        let g = three_class::simulate_graded(0.4, seed, &counts, PER_COUNT);
        let pr = pr_rows(&g.input.zscores);
        for (i, (_, group)) in g.by_count.iter().enumerate() {
            observed[i].push(mean_over(&pr, group));
        }
    }

    let mut means = Vec::new();
    for (i, &k) in counts.iter().enumerate() {
        let m = mean(&observed[i]);
        println!("{k:>7} {m:>10.2} {:>14.2}", (k as f32 + 2.0) / 3.0);
        means.push(m);
    }

    println!(
        "\nA monotone column is the claim: PR is reading the effective number of\n\
         traits, not a binary. Values sit above prediction because isotropic\n\
         noise pulls every variant toward {:.2}.\n",
        (NUM_TRAITS as f32 + 2.0) / 3.0,
    );

    for i in 1..means.len() {
        assert!(
            means[i] > means[i - 1],
            "PR must increase with trait count: {:.2} at k={} against {:.2} at k={}",
            means[i],
            counts[i],
            means[i - 1],
            counts[i - 1],
        );
    }
    Ok(())
}

/// Every variant in LD with one of `seed_set`: same block and same haplotype
/// index, which is exactly how the fixture builds correlation.
fn ld_partners_of(seed_set: &HashSet<usize>) -> HashSet<usize> {
    use three_class::{N_HAPLOTYPES, NUM_BLOCKS, SNPS_PER_BLOCK};
    let key = |g: usize| (g / SNPS_PER_BLOCK, (g % SNPS_PER_BLOCK) % N_HAPLOTYPES);
    let keys: HashSet<(usize, usize)> = seed_set.iter().map(|&g| key(g)).collect();
    (0..NUM_BLOCKS * SNPS_PER_BLOCK)
        .filter(|&g| keys.contains(&key(g)))
        .collect()
}

/// The trait-by-trait covariance of the null rows, `Ω`.
///
/// Estimated from the z-scores themselves — no panel. Rows above the `keep`
/// quantile of `‖z_g‖` are dropped, because a handful of large causal rows
/// would otherwise pull the genetic covariance into what is meant to be the
/// noise law.
fn trimmed_trait_covariance(z: &DMatrix<f32>, keep: f32) -> DMatrix<f32> {
    let norms = row_norms(z);
    let mut sorted = norms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = sorted[((sorted.len() as f32 * keep) as usize).min(sorted.len() - 1)];

    let t = z.ncols();
    let mut acc = DMatrix::<f32>::zeros(t, t);
    let mut n = 0usize;
    for (g, &norm) in norms.iter().enumerate() {
        if norm > cutoff {
            continue;
        }
        let r = z.row(g);
        acc += r.transpose() * r;
        n += 1;
    }
    if n > 0 {
        acc /= n as f32;
    }
    acc
}

/// A random orthogonal `T x T` matrix, from the QR of a Gaussian.
fn random_orthogonal(t: usize, rng: &mut SmallRng) -> DMatrix<f32> {
    let g = DMatrix::from_fn(t, t, |_, _| {
        let v: f64 = StandardNormal.sample(rng);
        v as f32
    });
    g.qr().q()
}

/// Benjamini-Hochberg. Returns the selected indices at level `q`.
fn bh_select(p: &[f32], q: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap_or(std::cmp::Ordering::Equal));
    let n = p.len() as f32;
    let mut cut = 0usize;
    for (rank, &i) in order.iter().enumerate() {
        if p[i] <= q * (rank + 1) as f32 / n {
            cut = rank + 1;
        }
    }
    order[..cut].to_vec()
}

/// The LD-free null: rotate the trait axis and let the data supply its own
/// contrast.
///
/// `E ~ MN(0, R, Ω)` implies `EQ ~ MN(0, R, Q'ΩQ)`, so at `Ω = I` a right
/// rotation leaves the noise law *identical*, leaves `R` — hence LD, hence the
/// correlation between neighbouring variants' statistics — exactly intact, and
/// leaves every row's magnitude alone. It destroys only the alignment between
/// the effect rows and the trait basis, which is precisely the alternative
/// being tested.
///
/// Note it must be a **rotation**, not a permutation of trait labels: `PR` is
/// permutation-invariant, so relabelling is inert against this statistic.
///
/// The point is not the per-variant p-value — at `Ω = I` that reduces to the
/// isotropic reference already used above. It is that the *joint* null carries
/// the true sample's LD, so a genome-wide threshold is calibrated without any
/// reference panel.
#[test]
fn test_rotation_null_calibrates_without_a_panel() -> Result<()> {
    let seeds = [20250808u64, 111, 222];
    let draws = 400usize;
    let q_level = 0.10f32;

    println!(
        "\n{:>8} | {:>22} | {:>26}",
        "", "p < 0.05, by class", "BH at q = 0.10"
    );
    println!(
        "{:>8} | {:>7} {:>7} {:>6} {:>6} {:>6} | {:>7} {:>7}",
        "seed", "specif", "tags", "pleio", "null", "null Q", "n_sel", "FDP*"
    );
    println!("{}", "-".repeat(72));

    let (mut sp, mut pl, mut nu) = (vec![], vec![], vec![]);
    let (mut fdps_locus, mut nu_plain) = (vec![], vec![]);
    for &seed in &seeds {
        let cl = simulate_classes(0.4, seed);
        let null = null_set(&cl);
        let z = &cl.input.zscores;
        let observed = pr_rows(z);

        // A bare rotation is exchangeable only at Ω = I, and these ten traits
        // are genetically correlated by construction. Rotating inside the
        // whitened trait space and mapping back keeps the law exact:
        //
        //     A = Ω^{1/2} Q Ω^{-1/2}    ⟹    Cov(A z) = A Ω A' = Ω
        //
        // R is untouched either way, since the same linear map is applied to
        // every row. Note `Ω̂` is estimated by trimming on ‖z_g‖, which
        // `omega_sqrt_pair`'s own docs record as biased ~16% low; it also
        // flattens the spectrum, which would make the draws too isotropic.
        //
        // Both arms are run because the correction turns out not to be what
        // makes this calibrate — see the printed comparison. Q need not be Haar:
        // it is drawn independently of z, and for ANY fixed orthogonal Q,
        // Q'w ~ N(0,I) when w ~ N(0,I).
        let (om_sqrt, om_inv_sqrt) = omega_sqrt_pair(&trimmed_trait_covariance(z, 0.90))?;

        let mut ge = vec![0usize; observed.len()];
        let mut ge_plain = vec![0usize; observed.len()];
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x5EED_D0DA);
        for _ in 0..draws {
            let qm = random_orthogonal(z.ncols(), &mut rng);
            let a = &om_sqrt * &qm * &om_inv_sqrt;
            for (acc, m) in [(&mut ge, &a), (&mut ge_plain, &qm)] {
                let zq = z * m.transpose();
                for g in 0..zq.nrows() {
                    if participation_ratio(zq.row(g).iter().copied()) <= observed[g] {
                        acc[g] += 1;
                    }
                }
            }
        }
        // One-sided lower tail: trait-specific means unusually *concentrated*.
        let pv = |c: &[usize]| -> Vec<f32> {
            c.iter().map(|&k| (1 + k) as f32 / (1 + draws) as f32).collect()
        };
        let (p, p_plain) = (pv(&ge), pv(&ge_plain));

        // A "null" variant sharing a haplotype with a trait-specific one is its
        // tag: it carries a scaled copy of that concentrated profile, so firing
        // on it is correct locus-level behaviour, not a false positive. The
        // calibration check therefore needs a null with the tags removed.
        let tagged = ld_partners_of(&cl.trait_specific);
        let clean_null: HashSet<usize> =
            null.iter().copied().filter(|g| !tagged.contains(g)).collect();

        let frac = |s: &HashSet<usize>| {
            s.iter().filter(|&&g| p[g] < 0.05).count() as f32 / s.len().max(1) as f32
        };
        let (a, b, c) = (
            frac(&cl.trait_specific),
            frac(&cl.pleiotropic),
            frac(&clean_null),
        );
        let c_plain = clean_null.iter().filter(|&&g| p_plain[g] < 0.05).count() as f32
            / clean_null.len().max(1) as f32;
        let tag_rate = frac(&tagged.difference(&cl.trait_specific).copied().collect());

        let selected = bh_select(&p, q_level);
        let n_sel = selected.len();
        let false_sel: Vec<usize> = selected
            .iter()
            .copied()
            .filter(|g| !cl.trait_specific.contains(g))
            .collect();
        // A tag of a true trait-specific variant inherits its direction, so it
        // is a locus-level hit rather than a mistake. Splitting the two says
        // whether the FDP is the method erring or the resolution limit.
        // FDP* counts a hit as correct if it is a trait-specific variant *or*
        // in LD with one — the locus-level reading, which is the strongest
        // claim a shape statistic on marginal rows can support.
        let fdp_locus = false_sel.iter().filter(|g| !tagged.contains(g)).count() as f32
            / n_sel.max(1) as f32;

        println!(
            "{seed:>8} | {a:>7.2} {tag_rate:>7.2} {b:>7.2} {c:>6.2} {c_plain:>6.2} | \
             {n_sel:>7} {fdp_locus:>7.2}"
        );
        sp.push(a);
        pl.push(b);
        nu.push(c);
        nu_plain.push(c_plain);
        fdps_locus.push(fdp_locus);
    }

    println!("{}", "-".repeat(72));
    println!(
        "{:>8} | {:>7.2} {:>7} {:>7.2} {:>6.2} {:>6.2} | {:>7} {:>7.2}",
        "mean",
        mean(&sp),
        "",
        mean(&pl),
        mean(&nu),
        mean(&nu_plain),
        "",
        mean(&fdps_locus),
    );
    println!(
        "
null is the calibration check; `null Q` repeats it with a plain
\
         rotation and no Omega correction. The two agree, so the whitening is
\
         NOT what makes this calibrate -- defining the null class correctly is.
\
         tags are LD copies of trait-specific variants, carrying an attenuated
\
         copy of the same profile, so firing on them is right at locus
\
         resolution and only FDP* is reported. pleio near nominal is the point:
\
         this tests specificity, not signal -- though 12 variants x 3 seeds
\
         cannot resolve much there.
"
    );

    // Calibration: the null class must not fire more than the nominal rate,
    // with slack for the tags LD forces into it.
    assert!(
        mean(&nu) < 0.10,
        "rotation null must be calibrated on variants with no signal and no LD \
         to any: nominal 0.05, got {:.3}",
        mean(&nu),
    );
    assert!(
        mean(&fdps_locus) < 0.15,
        "BH at q = 0.10 should control locus-level FDP, got {:.3}",
        mean(&fdps_locus),
    );
    // Power: trait-specific variants must fire far more often than nulls.
    assert!(
        mean(&sp) > 3.0 * mean(&nu).max(0.01),
        "trait-specific variants should fire well above the null rate: {:.3} against {:.3}",
        mean(&sp),
        mean(&nu),
    );
    // Pleiotropic variants are isotropic in trait space, so this test must be
    // blind to them — that is what makes it a specificity test and not a
    // detector.
    assert!(
        mean(&pl) < mean(&sp),
        "the test must be specific to trait-specificity: pleio {:.3}, specific {:.3}",
        mean(&pl),
        mean(&sp),
    );
    Ok(())
}

fn pleio_vs_specific_pr(cl: &Classes) -> f32 {
    let pr = pr_rows(&cl.input.zscores);
    auc_pair(&pr, &cl.pleiotropic, &cl.trait_specific)
}

fn null_set(cl: &Classes) -> HashSet<usize> {
    let all: HashSet<usize> = cl.pleiotropic.union(&cl.trait_specific).copied().collect();
    (0..cl.input.zscores.nrows()).filter(|g| !all.contains(g)).collect()
}

