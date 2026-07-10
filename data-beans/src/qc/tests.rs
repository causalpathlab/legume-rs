use super::*;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

////////////////////////////////////////////////////////////////////
// 2-means cutoff suggestion                                       //
////////////////////////////////////////////////////////////////////

#[test]
fn bic_accepts_clear_bimodal() {
    // A large ambient peak (~5) + a real-cell mode (~300), well separated.
    let mut rng = StdRng::seed_from_u64(7);
    let amb = Normal::new(5.0_f64, 1.5).unwrap();
    let real = Normal::new(300.0_f64, 40.0).unwrap();
    let mut nnz: Vec<f32> = (0..2000)
        .map(|_| amb.sample(&mut rng).max(1.0) as f32)
        .collect();
    nnz.extend((0..400).map(|_| real.sample(&mut rng).max(1.0) as f32));
    let c = suggest_nnz_cutoff(&nnz).expect("clear bimodal → Some(cutoff)");
    assert!(
        c > 15 && c < 300,
        "cutoff {c} should sit in the ambient↔real valley"
    );
}

#[test]
fn bic_rejects_unimodal() {
    // A single Gaussian nnz cloud: the exact split still partitions it, but
    // the BIC guard must reject (the hard split fits the centre worse).
    let mut rng = StdRng::seed_from_u64(11);
    let one = Normal::new(60.0_f64, 12.0).unwrap();
    let nnz: Vec<f32> = (0..3000)
        .map(|_| one.sample(&mut rng).max(1.0) as f32)
        .collect();
    assert!(
        suggest_nnz_cutoff(&nnz).is_none(),
        "single-mode nnz → no cutoff (BIC favors 1 cluster)"
    );
}

#[test]
fn bic_rejects_discrete_lowcount_unimodal() {
    // Regression for the variance-floor defeat: a single unimodal LOW-COUNT
    // discrete cloud must NOT be split. The old per-component variance floor
    // (1e-9) let a near-constant k-means cluster earn a delta-spike density;
    // the homoscedastic (shared-variance) BIC bounds every component's
    // variance by the real within-cluster spread, so this is rejected.
    let mut rng = StdRng::seed_from_u64(3);
    let p = rand_distr::Poisson::new(6.0_f64).unwrap();
    let nnz: Vec<f32> = (0..5000)
        .map(|_| p.sample(&mut rng).max(1.0) as f32)
        .collect();
    assert!(
        suggest_nnz_cutoff(&nnz).is_none(),
        "unimodal discrete low-count nnz → no cutoff"
    );
}

#[test]
fn bic_accepts_lowcount_bimodal() {
    // Genuine bimodality is still found even when the ambient mode is a tight
    // low-count spike (the case the variance fix must not over-correct).
    let mut rng = StdRng::seed_from_u64(5);
    let amb = rand_distr::Poisson::new(2.0_f64).unwrap();
    let real = Normal::new(500.0_f64, 60.0).unwrap();
    let mut nnz: Vec<f32> = (0..6000)
        .map(|_| amb.sample(&mut rng).max(1.0) as f32)
        .collect();
    nnz.extend((0..800).map(|_| real.sample(&mut rng).max(1.0) as f32));
    let c = suggest_nnz_cutoff(&nnz).expect("low-count ambient + real mode → Some(cutoff)");
    assert!(c > 5 && c < 500, "cutoff {c} should land in the valley");
}

#[test]
fn cutoff_is_deterministic() {
    // The exact sweep has no RNG, so repeated calls on the same data are
    // bit-identical (no seeding / restarts needed).
    let mut rng = StdRng::seed_from_u64(9);
    let amb = Normal::new(4.0_f64, 1.0).unwrap();
    let real = Normal::new(200.0_f64, 30.0).unwrap();
    let mut nnz: Vec<f32> = (0..3000)
        .map(|_| amb.sample(&mut rng).max(1.0) as f32)
        .collect();
    nnz.extend((0..500).map(|_| real.sample(&mut rng).max(1.0) as f32));
    let a = suggest_nnz_cutoff(&nnz);
    let b = suggest_nnz_cutoff(&nnz);
    assert_eq!(a, b, "exact 1-D 2-means must be deterministic");
    assert!(a.is_some());
}

////////////////////////////////////////////////////////////////////
// generalized histogram rendering (used by `histogram` command)   //
////////////////////////////////////////////////////////////////////

#[test]
fn fmt_stat_is_integer_for_whole_values() {
    // nnz / sum are whole -> no decimal point; mean / sd keep 2 decimals.
    assert_eq!(fmt_stat(0.0), "0");
    assert_eq!(fmt_stat(5.0), "5");
    assert_eq!(fmt_stat(900.0), "900");
    assert_eq!(fmt_stat(0.3), "0.30");
    assert_eq!(fmt_stat(12.53), "12.53");
}

#[test]
fn log_histogram_tracks_exact_ranges_and_no_cutoff_when_zero() {
    // Counts are preserved, per-bin ranges are exact f32, and cutoff == 0
    // (the `histogram` command) marks no bin.
    let vals: Vec<f32> = vec![1.0, 1.0, 2.0, 100.0, 100.0, 100.0];
    let hist = create_log_histogram(&vals, 0);
    let total: usize = hist.iter().map(|b| b.count).sum();
    assert_eq!(total, vals.len());
    assert!(
        hist.iter().all(|b| !b.is_cutoff),
        "cutoff == 0 must not mark any bin"
    );
    // The 100-valued bin collapses to a single exact value.
    let big = hist
        .iter()
        .find(|b| b.count == 3)
        .expect("three 100s land in one bin");
    assert_eq!(big.val_min, 100.0);
    assert_eq!(big.val_max, 100.0);
}

#[test]
fn log_histogram_marks_cutoff_bin_when_positive() {
    let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 200.0, 300.0];
    let hist = create_log_histogram(&vals, 50);
    assert!(
        hist.iter().any(|b| b.is_cutoff),
        "a positive cutoff should mark exactly the first bin at/above it"
    );
}
