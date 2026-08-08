//! Tests for cohort scoring.
//!
//! The two that matter are the traps: scoring must use the *panel's*
//! standardisation rather than the new cohort's, and an allele flip must negate
//! the dosage rather than leave it alone. Both are silent failures — the
//! numbers come out plausible and wrong.

use super::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

fn geno(n: usize, m: usize, seed: u64) -> GenotypeMatrix {
    let mut rng = SmallRng::seed_from_u64(seed);
    let genotypes = DMatrix::from_fn(n, m, |_, _| {
        let v: f64 = StandardNormal.sample(&mut rng);
        if v < -0.5 {
            0.0
        } else if v < 0.5 {
            1.0
        } else {
            2.0
        }
    });
    GenotypeMatrix {
        individual_ids: (0..n).map(|i| Box::from(format!("i{i}"))).collect(),
        snp_ids: (0..m).map(|j| Box::from(format!("rs{j}"))).collect(),
        chromosomes: vec![Box::from("chr1"); m],
        positions: (0..m).map(|j| (j * 1000) as u64).collect(),
        allele1: vec![Box::from("A"); m],
        allele2: vec![Box::from("G"); m],
        genotypes,
    }
}

fn randn(r: usize, c: usize, seed: u64) -> DMatrix<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    DMatrix::from_fn(r, c, |_, _| {
        let v: f64 = StandardNormal.sample(&mut rng);
        v as f32
    })
}

#[test]
fn test_scoring_the_panel_itself_reproduces_the_design() {
    let (n, m, h, t) = (60, 30, 3, 4);
    let panel_geno = geno(n, m, 1);
    let panel = PanelStandardization::from_panel(&panel_geno);
    let u = randn(m, h, 2);
    let v = randn(t, h, 3);

    let out = score_cohort(&u, &v, &panel, &panel_geno).unwrap();

    assert_eq!(out.scores.nrows(), n);
    assert_eq!(out.scores.ncols(), h);
    assert_eq!(out.prs.ncols(), t);
    assert_eq!(out.matched, m);
    assert_eq!(out.flipped, 0);
    for c in &out.coverage {
        assert!((c - 1.0).abs() < 1e-4, "full coverage expected, got {c}");
    }

    // S must equal standardised-X times U exactly.
    let mut x = panel_geno.genotypes.clone();
    {
        use matrix_util::traits::MatOps;
        x.scale_columns_inplace();
    }
    let expect = &x * &u;
    let err = (&out.scores - &expect).abs().max();
    println!("‖S − X̃U‖_max = {err:.6}");
    assert!(err < 1e-3, "scores should match the design exactly: {err}");
}

/// The in-sample-MAF trap: a cohort with a different allele-frequency
/// distribution must still be standardised by the panel's constants.
#[test]
fn test_panel_constants_are_used_not_the_cohort_s() {
    let (n, m, h, t) = (80, 20, 2, 3);
    let panel_geno = geno(n, m, 5);
    let panel = PanelStandardization::from_panel(&panel_geno);
    let u = randn(m, h, 6);
    let v = randn(t, h, 7);

    // A cohort whose dosages are shifted: in-sample standardisation would
    // remove the shift, panel standardisation must retain it.
    let mut shifted = geno(n, m, 5);
    shifted.genotypes.iter_mut().for_each(|g| *g = (*g + 1.0).min(2.0));

    let base = score_cohort(&u, &v, &panel, &panel_geno).unwrap();
    let out = score_cohort(&u, &v, &panel, &shifted).unwrap();

    let diff = (&out.scores - &base.scores).abs().max();
    println!("score shift under a shifted cohort: {diff:.4}");
    assert!(
        diff > 1e-3,
        "panel standardisation must let a real dosage shift through, got {diff}"
    );
}

/// An allele flip has to negate the dosage. Getting this wrong silently
/// reverses the sign of that variant's contribution.
#[test]
fn test_allele_flip_negates_the_dosage() {
    let (n, m, h, t) = (40, 10, 2, 2);
    let panel_geno = geno(n, m, 11);
    let panel = PanelStandardization::from_panel(&panel_geno);
    let u = randn(m, h, 12);
    let v = randn(t, h, 13);

    let base = score_cohort(&u, &v, &panel, &panel_geno).unwrap();

    // Flip every variant's alleles AND its dosage coding: the two cancel, so
    // the scores must be unchanged.
    let mut flipped = panel_geno.clone();
    flipped.allele1 = vec![Box::from("G"); m];
    flipped.allele2 = vec![Box::from("A"); m];
    flipped.genotypes.iter_mut().for_each(|g| *g = 2.0 - *g);

    let out = score_cohort(&u, &v, &panel, &flipped).unwrap();
    assert_eq!(out.flipped, m, "every variant should register as flipped");

    let err = (&out.scores - &base.scores).abs().max();
    println!("‖S_flipped − S‖_max = {err:.6}");
    assert!(
        err < 1e-3,
        "flipping alleles and dosages together must cancel, got {err}"
    );
}

/// Strand-ambiguous variants must be dropped, not guessed at.
#[test]
fn test_ambiguous_alleles_are_dropped() {
    let (n, m, h, t) = (30, 8, 2, 2);
    let mut panel_geno = geno(n, m, 17);
    panel_geno.allele1 = vec![Box::from("A"); m];
    panel_geno.allele2 = vec![Box::from("T"); m]; // A/T is ambiguous
    let panel = PanelStandardization::from_panel(&panel_geno);
    let u = randn(m, h, 18);
    let v = randn(t, h, 19);

    let out = score_cohort(&u, &v, &panel, &panel_geno).unwrap();
    assert_eq!(out.matched, 0, "ambiguous variants must not be scored");
    assert_eq!(out.dropped, m);
}

/// Coverage is reported on effect mass, not variant count — with a sparse `U`
/// those give very different answers.
#[test]
fn test_coverage_is_effect_mass_not_variant_count() {
    let (n, m, h, t) = (40, 20, 1, 2);
    let panel_geno = geno(n, m, 23);
    let panel = PanelStandardization::from_panel(&panel_geno);

    // All the effect sits on one variant.
    let mut u = DMatrix::<f32>::zeros(m, h);
    u[(0, 0)] = 1.0;
    let v = randn(t, h, 24);

    // A cohort missing exactly that variant: 95% of variants, 0% of the mass.
    let mut partial = panel_geno.clone();
    partial.genotypes = partial.genotypes.columns(1, m - 1).clone_owned();
    partial.snp_ids = partial.snp_ids[1..].to_vec();
    partial.chromosomes = partial.chromosomes[1..].to_vec();
    partial.positions = partial.positions[1..].to_vec();
    partial.allele1 = partial.allele1[1..].to_vec();
    partial.allele2 = partial.allele2[1..].to_vec();

    let out = score_cohort(&u, &v, &panel, &partial).unwrap();
    println!(
        "matched {}/{} variants but coverage {:.3}",
        out.matched,
        m,
        out.coverage[0]
    );
    assert_eq!(out.matched, m - 1);
    assert!(
        out.coverage[0] < 1e-6,
        "coverage must follow effect mass, got {}",
        out.coverage[0]
    );
}

#[test]
fn test_assemble_u_places_blocks_in_snp_order() {
    let blocks = vec![
        DMatrix::from_element(3, 2, 1.0f32),
        DMatrix::from_element(4, 2, 2.0f32),
    ];
    let u = assemble_u(&blocks, &[0, 5], 10);
    assert_eq!(u.nrows(), 10);
    assert_eq!(u[(0, 0)], 1.0);
    assert_eq!(u[(2, 1)], 1.0);
    assert_eq!(u[(3, 0)], 0.0, "gap between blocks stays zero");
    assert_eq!(u[(5, 0)], 2.0);
    assert_eq!(u[(8, 1)], 2.0);
    assert_eq!(u[(9, 0)], 0.0);
}
