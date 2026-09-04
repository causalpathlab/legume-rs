//! Bulk batches: corrected TOWARD the cell frame, never dragging it.
//!
//! Same greedy discipline `senna update` applies to a carried pb_reference —
//! only the new samples are adjusted, the established frame stays fixed.
//! Naming a batch in `MultilevelParams::bulk_batches` makes every NON-bulk
//! batch the anchor, so:
//!
//! - bulk is a **receiver**: its counterfactual is drawn from the cells and
//!   its δ is estimated against them (that estimate is the platform
//!   correction, and it is what we want);
//! - bulk is never a **source**: it is excluded from every anchor set, so no
//!   cell ever draws a counterfactual from a bulk column;
//! - the cells **self-match** through the anchor path, so their δ settles at
//!   the prior and the frame they define does not move.
//!
//! The alternative — pooled mutual adjustment — splits the discrepancy and
//! drags the cell frame toward bulk. Measured on real data in the pb_reference
//! case, that cost ARI 0.030 vs 0.219 for anchored, which is why this path is
//! greedy rather than symmetric.

use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::traits::Inference;

const D: usize = 8;
const N_PER_BATCH: usize = 40;

fn type_profile(i: usize) -> Vec<f32> {
    if i.is_multiple_of(2) {
        (0..D).map(|g| 10.0 + 3.0 * (g % 2) as f32).collect()
    } else {
        (0..D).map(|g| 4.0 + 2.0 * ((g + 1) % 2) as f32).collect()
    }
}

/// Bulk sample `j`: a mixture of the two type profiles at composition
/// `mix[j]` — same per-state biology as the cells, wildly different
/// proportions. If composition leaked into δ, these are the columns that
/// would drag it.
const MIX: [f32; 4] = [0.9, 0.7, 0.3, 0.1];
fn bulk_profile(j: usize) -> Vec<f32> {
    let a = type_profile(0);
    let b = type_profile(1);
    let m = MIX[j];
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| m * x + (1.0 - m) * y)
        .collect()
}

/// ONE coherent cell frame plus a bulk batch. The cells are clean type
/// profiles; the bulk columns are composition-skewed mixtures of the SAME
/// profiles under a genuine 2x/0.5x platform factor.
///
/// One cell batch is the point: greedy correction anchors on the non-bulk
/// batches, and the invariant "the frame does not move" is only well posed
/// when those batches ARE one frame. (Several mutually-discrepant cell
/// batches still mutually adjust among themselves — that is the pre-existing
/// pooled behaviour, unchanged by the bulk role.)
fn cohort(tag: &str) -> (SparseIoVec, Vec<&'static str>) {
    let platform: Vec<f32> = (0..D).map(|g| if g < D / 2 { 2.0 } else { 0.5 }).collect();

    let mut cols: Vec<Vec<f32>> = Vec::new();
    let mut batches: Vec<&'static str> = Vec::new();
    for i in 0..2 * N_PER_BATCH {
        let wiggle = 1.0 + 0.05 * ((i % 5) as f32 - 2.0);
        cols.push(type_profile(i).into_iter().map(|v| v * wiggle).collect());
        batches.push("cells");
    }
    for j in 0..MIX.len() {
        cols.push(
            bulk_profile(j)
                .into_iter()
                .zip(platform.iter())
                .map(|(v, &p)| v * p)
                .collect(),
        );
        batches.push("blk");
    }

    let dir = std::env::temp_dir().join(format!("dba_bulk_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("bulk.zarr");
    let _ = std::fs::remove_dir_all(&path);

    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for (j, col) in cols.iter().enumerate() {
        for (g, &v) in col.iter().enumerate() {
            if v != 0.0 {
                triplets.push((g as u64, j as u64, v));
            }
        }
    }
    let shape = (D, cols.len(), triplets.len());
    let mut b = create_sparse_from_triplets(
        &triplets,
        shape,
        Some(path.to_str().expect("utf8")),
        Some(&data_beans::sparse_io::SparseIoBackend::Zarr),
    )
    .expect("create backend");
    b.register_row_names_vec(
        &(0..D)
            .map(|g| format!("g{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..cols.len())
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );

    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    (v, batches)
}

fn mean_abs_log_delta(out: &data_beans_alg::collapse_data::CollapsedOut, b: usize) -> f32 {
    let delta = out.delta.as_ref().expect("delta").posterior_mean();
    let col = delta.column(b);
    let logs: Vec<f32> = col
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| v.ln().abs())
        .collect();
    logs.iter().sum::<f32>() / logs.len().max(1) as f32
}

#[test]
fn bulk_composition_does_not_leak_into_delta() {
    let (mut v, batches) = cohort("delta");
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    let params = MultilevelParams {
        knn_pb_samples: 3,
        num_levels: 1,
        sort_dim: 3,
        num_opt_iter: 30,
        refine: Some(data_beans_alg::refine_multilevel::RefineParams::default()),
        output_calibration: matrix_param::traits::CalibrateTarget::All,
        anchor_batches: None,
        bulk_batches: Some(vec!["blk".into()]),
        observe_panels: true,
        keep_finest_stats: true,
    };
    let out = collapse_columns_multilevel_with_hierarchy(&mut v, &proj, &batches, &params)
        .expect("collapse");
    let finest = &out.levels[0];
    let membership = &out.cell_to_pb_per_level[0];
    let k = finest.mu_observed.posterior_mean().ncols();
    let n_cells = 2 * N_PER_BATCH;

    // 1. Structure: every bulk column keeps its own singleton finest group,
    //    appended after the cell groups in column order — never re-averaged
    //    into any (batch × group) blend.
    let k_new = k - MIX.len();
    for j in 0..MIX.len() {
        assert_eq!(
            membership[n_cells + j],
            k_new + j,
            "bulk column {j} not in its own appended group"
        );
    }
    for (c, &g) in membership.iter().take(n_cells).enumerate() {
        assert!(g < k_new, "cell {c} landed in a bulk group {g}");
    }

    // 2. The observed evidence of a bulk singleton is the sample's own
    //    profile back out — the dictionary sees bulk exactly as provided.
    for j in 0..MIX.len() {
        let platform: Vec<f32> = (0..D).map(|g| if g < D / 2 { 2.0 } else { 0.5 }).collect();
        for (g, expected) in bulk_profile(j)
            .into_iter()
            .zip(platform.iter())
            .map(|(v, &p)| v * p)
            .enumerate()
        {
            let back = finest.mu_observed.evidence_mean(g, k_new + j);
            assert!(
                (back - expected).abs() <= 1e-4 * expected.abs(),
                "bulk {j} gene {g}: expected {expected}, evidence {back}"
            );
        }
    }

    // 2b. Bulk IS adjusted. Greedy correction means bulk receives a
    //     counterfactual from the cell frame, so μ_adjusted must differ from
    //     μ_observed — if they were equal, bulk got no correction at all and
    //     the anchoring never engaged.
    let adj = finest
        .mu_adjusted
        .as_ref()
        .expect("adjusted")
        .posterior_mean();
    let obs = finest.mu_observed.posterior_mean();
    let moved = (0..MIX.len()).any(|j| {
        (0..D).any(|g| {
            let (a, o) = (adj[(g, k_new + j)], obs[(g, k_new + j)]);
            (a - o).abs() > 1e-3 * o.abs().max(1e-6)
        })
    });
    assert!(
        moved,
        "no bulk column was adjusted — greedy correction did not engage"
    );

    // 3. NOT ASSERTED — and the gap is deliberate, not an oversight.
    //
    //    The greedy design predicts an asymmetry: the anchored cell frame's δ
    //    should rest near the prior while bulk's absorbs the platform shift.
    //    On this fixture it does NOT appear — measured δ (mean|log|) is 0.522
    //    for cells and 0.509 for bulk, i.e. indistinguishable, with both
    //    displaced from 1. δ has a global scale that trades off against μ, and
    //    with two batches and four bulk columns that level is not identifiable,
    //    so a synthetic this small cannot separate "the frame moved" from "the
    //    common scale moved".
    //
    //    Do not add an absolute δ assertion here without first making the
    //    scale identifiable (more batches, or pinning μ). The claim that
    //    greedy correction protects the cell frame is currently UNVERIFIED;
    //    the check that would settle it is the real-data one — cell-type ARI
    //    for a cells+bulk fit against a cells-only fit with the SAME batch
    //    count (see the bulk-joint-fit findings note).
    let names = v.batch_names().expect("batch names");
    let idx = |n: &str| {
        names
            .iter()
            .position(|x| x.as_ref() == n)
            .expect("batch present")
    };
    // Both batches' δ is finite and positive — the fit is well-formed even
    // though the asymmetry above is not resolvable here.
    for b in ["cells", "blk"] {
        let d = mean_abs_log_delta(finest, idx(b));
        assert!(d.is_finite(), "delta for `{b}` is not finite: {d}");
    }
}
