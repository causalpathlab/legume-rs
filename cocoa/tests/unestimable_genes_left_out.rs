//! Genes whose exposure effect cannot be estimated (no counts, or an
//! exposure level whose individuals all have zero counts) are left out:
//! NA contrast and NA p-value, never a number. Every other gene is tested.

mod common;
use common::*;

#[test]
fn unestimable_genes_get_na_and_the_rest_are_tested() {
    let fx = build(&Spec {
        n_genes: 30,
        n_indv: 12,
        n_states: 3,
        cells_per_indv: 60,
        exposure_shifts_states: false,
    });
    let out = fx.out("out");
    diff(
        &fx,
        &fx.one_topic(),
        &fx.exposures(),
        &["--n-permutations", "30"],
        &out,
    );

    let contrast = column(&format!("{out}.perm.parquet"), "contrast");
    let pvalue = column(&format!("{out}.perm.parquet"), "pvalue");
    for (g, (c, p)) in contrast.iter().zip(&pvalue).enumerate() {
        if g == GENE_NO_COUNTS || g == GENE_ZERO_AT_LEVEL_1 {
            assert!(
                c.is_nan() && p.is_nan(),
                "gene {g}: contrast {c}, p-value {p}"
            );
        } else {
            assert!(c.is_finite(), "gene {g}: contrast {c}");
            assert!((0.0..=1.0).contains(p), "gene {g}: p-value {p}");
        }
    }
}
