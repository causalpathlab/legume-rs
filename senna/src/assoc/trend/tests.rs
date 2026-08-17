use super::*;
use crate::assoc::test_util::two_branch_panel as synth;

fn cfg() -> TrendConfig {
    TrendConfig {
        n_knots: 5,
        min_total_coverage: 30,
        min_cells: 10,
        overdispersion: true,
    }
}

#[test]
fn rising_branch_detected_flat_branch_not() {
    let (lin, sites) = synth();
    let res = run_trends(&sites, &lin, &cfg());
    let get = |site: usize, branch: usize| {
        res.iter()
            .find(|r| r.site == site && r.branch == branch)
            .unwrap()
    };
    // Site A rises in branch 0, flat in branch 1.
    let a0 = get(0, 0);
    assert!(a0.p_value < 0.05, "rising branch0 p={}", a0.p_value);
    assert!(a0.effect > 0.0, "rising ⇒ +effect, got {}", a0.effect);
    let a1 = get(0, 1);
    assert!(a1.p_value > 0.2, "flat branch1 p={}", a1.p_value);
}

#[test]
fn flat_site_never_called() {
    let (lin, sites) = synth();
    let res = run_trends(&sites, &lin, &cfg());
    for r in res.iter().filter(|r| r.site == 1) {
        assert!(
            r.p_value > 0.2,
            "flat site branch{} p={}",
            r.branch,
            r.p_value
        );
    }
}

#[test]
fn qc_drops_low_coverage_branches() {
    let (lin, sites) = synth();
    let strict = TrendConfig {
        min_cells: 1000,
        ..cfg()
    };
    assert!(
        run_trends(&sites, &lin, &strict).is_empty(),
        "min_cells beyond available ⇒ no results"
    );
}
