use super::*;
use crate::assoc::test_util::logit_line;

#[test]
fn rising_trend_detected_positive_effect() {
    let (k, n, x) = logit_line(-2.5, 5.0, 30, 60);
    let fit = association_test(&k, &n, &x, &GamArgs::default()).expect("fit");
    assert!(fit.p_value < 0.01, "rising trend p={}", fit.p_value);
    assert!(
        fit.effect > 0.0,
        "rising ⇒ positive effect, got {}",
        fit.effect
    );
}

#[test]
fn falling_trend_detected_negative_effect() {
    let (k, n, x) = logit_line(2.5, -5.0, 30, 60);
    let fit = association_test(&k, &n, &x, &GamArgs::default()).expect("fit");
    assert!(fit.p_value < 0.01, "falling trend p={}", fit.p_value);
    assert!(
        fit.effect < 0.0,
        "falling ⇒ negative effect, got {}",
        fit.effect
    );
}

#[test]
fn flat_rate_not_called() {
    // Constant rate 0.3 everywhere: full ≈ null ⇒ deviance diff ≈ 0 ⇒ p ≈ 1.
    let m = 60;
    let k = vec![9u32; m]; // 0.3 * 30
    let n = vec![30u32; m];
    let x: Vec<f32> = (0..m).map(|i| i as f32 / (m as f32 - 1.0)).collect();
    let fit = association_test(&k, &n, &x, &GamArgs::default()).expect("fit");
    assert!(
        fit.p_value > 0.2,
        "flat p should be large, got {}",
        fit.p_value
    );
    assert!(fit.stat.abs() < 1.0, "flat stat ≈ 0, got {}", fit.stat);
}

#[test]
fn quasi_binomial_is_more_conservative_under_overdispersion() {
    // Mild trend + strong alternating jitter ⇒ overdispersion the 5-knot spline
    // cannot absorb. The quasi-binomial F-test must not be sharper than the
    // Binomial LRT, and the estimated dispersion must exceed 1.
    let (m, cov) = (60usize, 20u32);
    let mut k = Vec::with_capacity(m);
    let n = vec![cov; m];
    let mut x = Vec::with_capacity(m);
    for i in 0..m {
        let xi = i as f64 / (m as f64 - 1.0);
        let base = 1.0 / (1.0 + (-(-0.8 + 1.6 * xi)).exp());
        let jitter = if i % 2 == 0 { 0.22 } else { -0.22 };
        let p = (base + jitter).clamp(0.02, 0.98);
        k.push((cov as f64 * p).round() as u32);
        x.push(xi as f32);
    }
    let binom = association_test(
        &k,
        &n,
        &x,
        &GamArgs {
            overdispersion: false,
            ..GamArgs::default()
        },
    )
    .expect("binom fit");
    let quasi = association_test(&k, &n, &x, &GamArgs::default()).expect("quasi fit");
    assert!(
        quasi.dispersion > 1.0,
        "overdispersion should be detected, φ={}",
        quasi.dispersion
    );
    assert!(
        quasi.p_value >= binom.p_value,
        "quasi p ({}) must be ≥ binomial p ({})",
        quasi.p_value,
        binom.p_value
    );
}

#[test]
fn too_few_observations_returns_none() {
    let k = vec![1u32, 2, 1];
    let n = vec![10u32, 10, 10];
    let x = vec![0.0f32, 0.5, 1.0];
    assert!(association_test(&k, &n, &x, &GamArgs::default()).is_none());
}

#[test]
fn no_pseudotime_spread_returns_none() {
    let k = vec![3u32; 20];
    let n = vec![10u32; 20];
    let x = vec![0.5f32; 20];
    assert!(association_test(&k, &n, &x, &GamArgs::default()).is_none());
}

#[test]
fn uncovered_cells_are_ignored() {
    // Half the cells have zero coverage; the covered half carries a clear rising
    // trend and must still be detected.
    let (mut k, mut n, x) = logit_line(-2.5, 5.0, 30, 60);
    for i in (0..k.len()).step_by(2) {
        k[i] = 0;
        n[i] = 0; // dropped inside association_test
    }
    let fit = association_test(&k, &n, &x, &GamArgs::default()).expect("fit");
    assert_eq!(fit.n_obs, 30, "only covered cells counted");
    assert!(
        fit.p_value < 0.05,
        "trend survives dropping uncovered, p={}",
        fit.p_value
    );
}

/// Regenerate the mgcv / tradeSeq comparison data consumed by
/// `faba/examples/gam_compare*.R`. Ignored by default; run on demand:
///   `GAM_COMPARE_DIR=/dir cargo test -p faba --bin faba \`
///     `assoc::gam::tests::dump_gam_compare_data -- --ignored --nocapture`
#[test]
#[ignore = "writes comparison TSVs on demand"]
fn dump_gam_compare_data() {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Binomial, Distribution};
    use std::io::Write;

    let dir = std::env::var("GAM_COMPARE_DIR")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    let n_cells = 150usize;
    let mut rng = SmallRng::seed_from_u64(20260706); // reproducible binomial noise
    let slopes = [
        -8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
    ];
    let covs = [10u32, 30u32];
    let t: Vec<f64> = (0..n_cells)
        .map(|c| c as f64 / (n_cells as f64 - 1.0))
        .collect();
    let tf: Vec<f32> = t.iter().map(|&v| v as f32).collect();

    let mut cells = std::fs::File::create(format!("{dir}/gam_cells.tsv")).unwrap();
    let mut ours = std::fs::File::create(format!("{dir}/gam_ours.tsv")).unwrap();
    writeln!(cells, "gene\tcell\tk\tn\tt").unwrap();
    writeln!(ours, "gene\tslope\tcov\tp_quasi\tp_binom\teffect").unwrap();
    let quasi = GamArgs::default();
    let binom = GamArgs {
        overdispersion: false,
        ..GamArgs::default()
    };

    let mut gi = 0usize;
    for &cov in &covs {
        for &b in &slopes {
            let a = -b / 2.0; // center the sigmoid in [0,1]
            let gene = format!("g{gi}_{cov}_{}", (b * 10.0) as i32);
            let nn = vec![cov; n_cells];
            let mut kk = Vec::with_capacity(n_cells);
            for (c, &tc) in t.iter().enumerate() {
                let p = 1.0 / (1.0 + (-(a + b * tc)).exp());
                let k = Binomial::new(cov as u64, p).unwrap().sample(&mut rng) as u32;
                kk.push(k);
                writeln!(cells, "{gene}\t{c}\t{k}\t{cov}\t{tc:.6}").unwrap();
            }
            let fq = association_test(&kk, &nn, &tf, &quasi).unwrap();
            let fb = association_test(&kk, &nn, &tf, &binom).unwrap();
            writeln!(
                ours,
                "{gene}\t{b}\t{cov}\t{:.6e}\t{:.6e}\t{:.4}",
                fq.p_value, fb.p_value, fq.effect
            )
            .unwrap();
            gi += 1;
        }
    }
    eprintln!("wrote {dir}/gam_cells.tsv ({gi} genes × {n_cells} cells) and {dir}/gam_ours.tsv");
}
