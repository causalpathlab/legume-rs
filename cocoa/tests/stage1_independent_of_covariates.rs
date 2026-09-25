//! Stage 1 (pseudobulks and the cell-state baseline) must depend on cell
//! state only: individual-level confounders enter stage 2 alone. So the
//! unadjusted contrast of a run with a covariate file must equal the
//! contrast of the same run without one.

mod common;
use common::*;

#[test]
fn covariate_file_does_not_change_stage_one() {
    let fx = build(&Spec {
        n_genes: 30,
        n_indv: 12,
        n_states: 3,
        cells_per_indv: 60,
        exposure_shifts_states: false,
    });
    let (plain, adjusted) = (fx.out("plain"), fx.out("adj"));
    diff(&fx, &fx.one_topic(), &fx.exposures(), &[], &plain);
    let conf = fx.covariates();
    diff(
        &fx,
        &fx.one_topic(),
        &fx.exposures(),
        &["--covariate-file", &conf],
        &adjusted,
    );

    let a = column(&format!("{plain}.contrast.parquet"), "contrast");
    let b = column(
        &format!("{adjusted}.contrast.parquet"),
        "contrast_unadjusted",
    );
    assert_eq!(a.len(), b.len());
    for (g, (x, y)) in a.iter().zip(&b).enumerate() {
        let same = (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-4;
        assert!(
            same,
            "gene {g}: the covariate file changed stage 1 ({x} vs {y})"
        );
    }
}
