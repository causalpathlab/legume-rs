//! Stage 1 (which cells count, and their sums) must not depend on the
//! exposure labels: relabelling the individuals leaves the per-gene log mean
//! count, computed from the stage-1 sums, unchanged. A label-free stage 1 is
//! computed once and reused by every permutation draw.

mod common;
use common::*;

#[test]
fn relabelling_exposures_does_not_change_stage_one() {
    // exposure shifts cell states, so states are unevenly shared
    let fx = build(&Spec {
        n_genes: 30,
        n_indv: 12,
        n_states: 3,
        cells_per_indv: 60,
        exposure_shifts_states: true,
    });

    // rotate the labels across individuals
    let rows: Vec<(String, String)> = read_gz(&fx.exposures())
        .lines()
        .map(|l| {
            let mut f = l.split('\t');
            (f.next().unwrap().to_string(), f.next().unwrap().to_string())
        })
        .collect();
    let n = rows.len();
    let rotated: String = (0..n)
        .map(|i| format!("{}\t{}\n", rows[i].0, rows[(i + 1) % n].1))
        .collect();
    let rotated_file = fx.out("rotated.gz");
    write_gz(&rotated_file, &rotated);

    let (a, b) = (fx.out("a"), fx.out("b"));
    diff(&fx, &fx.states(), &fx.exposures(), &[], &a);
    diff(&fx, &fx.states(), &rotated_file, &[], &b);

    let a = column(&format!("{a}.contrast.parquet"), "log_mean");
    let b = column(&format!("{b}.contrast.parquet"), "log_mean");
    for (g, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(
            (x - y).abs() < 1e-5,
            "gene {g}: stage 1 depends on the exposure labels ({x} vs {y})"
        );
    }
}
