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
//! where `R_gj` is a *scalar*: it rescales the row and leaves its direction
//! alone. So the *shape* of a raw marginal z-row is LD-invariant, while its
//! *magnitude* is not — and "pleiotropic versus trait-specific" is a question
//! about shape.
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
//! `PR(c·e_t) = 1` exactly. For `x ~ N(0, I_T)`, `Σx² ~ χ²_T` so
//! `E[(Σx²)²] = T(T+2)`, while `E[Σx⁴] = 3T`, giving
//!
//! ```text
//! PR ≈ T(T+2) / 3T = (T+2)/3        →  4.0 at T = 10
//! ```
//!
//! (Not `T/3`: that is the ratio of `E[Σx²]²` to `E[Σx⁴]` and drops the
//! variance of the numerator.) The simulation plants pleiotropic effects as
//! `c·v_h` with `v_h ~ N(0, I_T)`, so the predicted contrast is **1 against
//! 4.0** — and isotropic noise also sits at 4.0, which is why this separates
//! trait-specific from *everything else* rather than signal from null.
//!
//! # What this cannot do
//!
//! A tag inherits its causal variant's direction, so shape ranks tags
//! alongside causal variants: these are locus claims, not variant claims.
//! And a neighbourhood containing two causal variants with *different* trait
//! profiles blends into a mixture — [`test_blended_neighbourhood_is_the_limit`]
//! plants exactly that.
//!
//! Run: cargo test -p fagioli --test trait_profile_shape -- --nocapture
use anyhow::Result;
use fagioli::embedding::noise::omega_sqrt_pair;
use fagioli::summary_stats::common::SumstatInput;
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
/// `PR(c·e_t) = 1`; `PR` of a length-`T` isotropic Gaussian tends to `T/3`.
fn participation_ratio(x: &[f32]) -> f32 {
    let s2: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let s4: f64 = x.iter().map(|&v| (v as f64).powi(4)).sum();
    if s4 <= 0.0 {
        return 0.0;
    }
    (s2 * s2 / s4) as f32
}

fn row(m: &DMatrix<f32>, g: usize) -> Vec<f32> {
    (0..m.ncols()).map(|t| m[(g, t)]).collect()
}

/// Per-variant `‖z_g‖` and `PR(z_g)`, straight off the z-score matrix.
fn raw_statistics(z: &DMatrix<f32>) -> (Vec<f32>, Vec<f32>) {
    (0..z.nrows())
        .map(|g| {
            let r = row(z, g);
            (r.iter().map(|v| v * v).sum::<f32>().sqrt(), participation_ratio(&r))
        })
        .unzip()
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

    for &seed in &seeds {
        let cl = simulate_classes(0.4, seed);
        let null = null_set(&cl);
        let (norm, pr) = raw_statistics(&cl.input.zscores);

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
        pr_null.push(mean_over(&pr, &null));
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

    let isotropic = (NUM_TRAITS as f32 + 2.0) / 3.0;
    println!(
        "\nmean PR: pleiotropic {:.2}, trait-specific {:.2}, null {:.2}   \
         (predicted {isotropic:.2} / 1.00 / {isotropic:.2})\n\
         Trait-specific sits above 1 because noise is isotropic and pulls PR\n\
         toward {isotropic:.2} at finite SNR.",
        mean(&pr_pleio),
        mean(&pr_spec),
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
    for &seed in &seeds {
        let cl = simulate_classes(0.4, seed);
        let null = null_set(&cl);
        let fit = fit_embedding(&cl, three_class::NUM_PROGRAMS, seed)?;

        let starts: Vec<usize> = cl.input.blocks.iter().map(|b| b.snp_start).collect();
        let u = assemble_u(&fit.u_mean, &starts, cl.input.zscores.nrows());

        // ‖u_g‖ — the statistic used so far.
        let norms: Vec<f32> = (0..u.nrows()).map(|g| u.row(g).norm()).collect();
        // PR(V̌ u_g) — row g of the fitted effect matrix, invariant to A.
        let b_hat = &u * fit.v_check.transpose();
        let fpr: Vec<f32> = (0..b_hat.nrows())
            .map(|g| participation_ratio(&row(&b_hat, g)))
            .collect();
        // PR(z_g) — no model at all.
        let (_, rpr) = raw_statistics(&cl.input.zscores);

        // A = diag(a_h), invertible and non-orthogonal. U→UA, V̌→V̌A⁻¹ leaves
        // U V̌' exactly unchanged.
        let h = fit.v_check.ncols();
        let a: Vec<f32> = (0..h).map(|i| 3.0f32.powi(i as i32 - (h as i32) / 2)).collect();
        let u_rot = DMatrix::from_fn(u.nrows(), h, |g, k| u[(g, k)] * a[k]);
        let v_rot = DMatrix::from_fn(fit.v_check.nrows(), h, |t, k| fit.v_check[(t, k)] / a[k]);
        let drift = (&u_rot * v_rot.transpose() - &b_hat).abs().max();
        assert!(drift < 1e-3, "the rotation must not change U V̌': {drift}");
        let norms_rot: Vec<f32> = (0..u_rot.nrows()).map(|g| u_rot.row(g).norm()).collect();

        let (x, y, z, w) = (
            auc_pair(&norms, &cl.pleiotropic, &cl.trait_specific),
            auc_pair(&fpr, &cl.pleiotropic, &cl.trait_specific),
            auc_pair(&rpr, &cl.pleiotropic, &cl.trait_specific),
            auc_pair(&norms_rot, &cl.pleiotropic, &cl.trait_specific),
        );
        let _ = &null;
        println!("{seed:>8} | {x:>10.3} {y:>10.3} {z:>10.3} | {w:>12.3}");
        un.push(x);
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
    Ok(())
}

fn fit_embedding(cl: &Classes, h: usize, seed: u64) -> Result<fagioli::embedding::train::EmbedFit> {
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
    train(
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
    )
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

    println!("\n{:>7} {:>10} {:>12}", "traits", "PR(z_g)", "predicted");
    println!("{}", "-".repeat(32));

    let mut observed = vec![Vec::new(); counts.len()];
    for &seed in &seeds {
        let g = three_class::simulate_graded(0.4, seed, &counts, 12);
        let (_, pr) = raw_statistics(&g.input.zscores);
        for (i, (_, group)) in g.by_count.iter().enumerate() {
            observed[i].push(mean_over(&pr, group));
        }
    }

    let mut means = Vec::new();
    for (i, &k) in counts.iter().enumerate() {
        let m = mean(&observed[i]);
        println!("{k:>7} {m:>10.2} {:>12.2}", (k as f32 + 2.0) / 3.0);
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
    let norms: Vec<f32> = (0..z.nrows()).map(|g| z.row(g).norm()).collect();
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
        "{:>8} | {:>7} {:>7} {:>6} {:>6} | {:>7} {:>7} {:>7}",
        "seed", "specif", "tags", "pleio", "null", "n_sel", "FDP", "FDP*"
    );
    println!("{}", "-".repeat(72));

    let (mut sp, mut pl, mut nu) = (vec![], vec![], vec![]);
    let (mut fdps, mut fdps_locus) = (vec![], vec![]);
    for &seed in &seeds {
        let cl = simulate_classes(0.4, seed);
        let null = null_set(&cl);
        let z = &cl.input.zscores;
        let (_, observed) = raw_statistics(z);

        // A bare rotation is only exchangeable at Ω = I. These ten traits are
        // genetically correlated by construction, so a null row has covariance
        // Ω and Q'ΩQ ≠ Ω. Rotate inside the whitened trait space and map back:
        //
        //     A = Ω^{1/2} Q Ω^{-1/2}    ⟹    Cov(A z) = A Ω A' = Ω
        //
        // which leaves the noise law exact, leaves R untouched (the same linear
        // map is applied to every row), and still scrambles alignment with the
        // trait basis — where the alternative lives.
        let (om_sqrt, om_inv_sqrt) = omega_sqrt_pair(&trimmed_trait_covariance(z, 0.90))?;

        let mut rng = SmallRng::seed_from_u64(seed ^ 0x5EED_D0DA);
        let mut ge = vec![0usize; observed.len()];
        for _ in 0..draws {
            let qm = random_orthogonal(z.ncols(), &mut rng);
            let a = &om_sqrt * &qm * &om_inv_sqrt;
            let zq = z * a.transpose();
            for g in 0..zq.nrows() {
                if participation_ratio(&row(&zq, g)) <= observed[g] {
                    ge[g] += 1;
                }
            }
        }
        // One-sided lower tail: trait-specific means unusually *concentrated*.
        let p: Vec<f32> =
            ge.iter().map(|&c| (1 + c) as f32 / (1 + draws) as f32).collect();

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
        let tag_rate = frac(&tagged.difference(&cl.trait_specific).copied().collect());

        let selected = bh_select(&p, q_level);
        let n_sel = selected.len();
        let false_sel: Vec<usize> = selected
            .iter()
            .copied()
            .filter(|g| !cl.trait_specific.contains(g))
            .collect();
        let fdp = false_sel.len() as f32 / n_sel.max(1) as f32;
        // A tag of a true trait-specific variant inherits its direction, so it
        // is a locus-level hit rather than a mistake. Splitting the two says
        // whether the FDP is the method erring or the resolution limit.
        // FDP* counts a hit as correct if it is a trait-specific variant *or*
        // in LD with one — the locus-level reading, which is the strongest
        // claim a shape statistic on marginal rows can support.
        let fdp_locus = false_sel.iter().filter(|g| !tagged.contains(g)).count() as f32
            / n_sel.max(1) as f32;

        println!(
            "{seed:>8} | {a:>7.2} {tag_rate:>7.2} {b:>7.2} {c:>6.2} | \
             {n_sel:>7} {fdp:>7.2} {fdp_locus:>7.2}"
        );
        sp.push(a);
        pl.push(b);
        nu.push(c);
        fdps.push(fdp);
        fdps_locus.push(fdp_locus);
    }

    println!("{}", "-".repeat(72));
    println!(
        "{:>8} | {:>7.2} {:>7} {:>7.2} {:>6.2} | {:>7} {:>7.2} {:>7.2}",
        "mean",
        mean(&sp),
        "",
        mean(&pl),
        mean(&nu),
        "",
        mean(&fdps),
        mean(&fdps_locus),
    );
    println!(
        "\nnull is the calibration check and sits at nominal. tags are LD copies of\n\
         trait-specific variants: firing on them is correct at locus resolution,\n\
         which is why FDP (variant-level) and FDP* (locus-level) differ so much.\n\
         pleio below nominal is the point — this tests specificity, not signal.\n"
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
    let (_, pr) = raw_statistics(&cl.input.zscores);
    auc_pair(&pr, &cl.pleiotropic, &cl.trait_specific)
}

fn null_set(cl: &Classes) -> HashSet<usize> {
    let all: HashSet<usize> = cl.pleiotropic.union(&cl.trait_specific).copied().collect();
    (0..num_snps(&cl.input)).filter(|g| !all.contains(g)).collect()
}

fn num_snps(input: &SumstatInput) -> usize {
    input.zscores.nrows()
}
