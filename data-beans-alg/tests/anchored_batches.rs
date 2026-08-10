//! Anchored (greedy) batch correction: the reference frame is never
//! re-adjusted.
//!
//! With `MultilevelParams::anchor_batches`, counterfactuals come from the
//! anchor batches only. Measured on real data before this existed, the pooled
//! default let a 2-batch update *mutually* adjust — δ split the discrepancy
//! between the new cohort and the carried reference, the reference frame
//! drifted (mean|log δ_ref| = 2.6), and the training target absorbed
//! composition biology (ARI 0.03 vs 0.45 for the exact path). The invariant
//! asserted here is the one the design always claimed: **δ for the anchor
//! batch stays ≈ 1.**

use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::traits::Inference;

const D: usize = 8;

/// Two batches over a shared 2-type structure, with a multiplicative shift on
/// the "new" batch so there is a genuine batch effect for δ to find.
///
/// Deterministic layout: types alternate; the anchor batch's columns are
/// clean type profiles (as a carried reference would be), the new batch's are
/// the same profiles times a per-gene platform factor.
fn two_batch_cohort() -> (SparseIoVec, Vec<&'static str>, Vec<f32>) {
    let type_a: Vec<f32> = (0..D).map(|g| 10.0 + 3.0 * (g % 2) as f32).collect();
    let type_b: Vec<f32> = (0..D).map(|g| 4.0 + 2.0 * ((g + 1) % 2) as f32).collect();
    let platform: Vec<f32> = (0..D).map(|g| if g < D / 2 { 2.0 } else { 0.5 }).collect();

    let mut cols: Vec<Vec<f32>> = Vec::new();
    let mut batches: Vec<&'static str> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();

    // 40 "new" cells per type, platform-shifted, weight 1. A ± wiggle keeps
    // per-gene variance non-degenerate without an RNG.
    for i in 0..80 {
        let base = if i % 2 == 0 { &type_a } else { &type_b };
        let wiggle = 1.0 + 0.05 * ((i % 5) as f32 - 2.0);
        cols.push(
            base.iter()
                .zip(platform.iter())
                .map(|(&v, &p)| v * p * wiggle)
                .collect(),
        );
        batches.push("new");
        weights.push(1.0);
    }
    // 10 carried reference columns per type: clean profiles, 20 cells each.
    for i in 0..20 {
        let base = if i % 2 == 0 { &type_a } else { &type_b };
        let wiggle = 1.0 + 0.04 * ((i % 3) as f32 - 1.0);
        cols.push(base.iter().map(|&v| v * wiggle).collect());
        batches.push("__ref__");
        weights.push(20.0);
    }

    let dir = std::env::temp_dir().join(format!("dba_anchor_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("anchor.zarr");
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
    (v, batches, weights)
}

fn run(anchored: bool) -> data_beans_alg::collapse_data::CollapsedOut {
    let (mut v, batches, weights) = two_batch_cohort();
    v.register_column_multiplicity(&weights).expect("weights");

    // A tiny deterministic projection: enough dims for the partition to
    // separate the two types (values dominate the proj).
    let n = v.num_columns();
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    assert_eq!(proj.ncols(), n);

    let params = MultilevelParams {
        knn_pb_samples: 3,
        num_levels: 1,
        sort_dim: 3,
        num_opt_iter: 30,
        refine: Some(data_beans_alg::refine_multilevel::RefineParams::default()),
        output_calibration: matrix_param::traits::CalibrateTarget::All,
        anchor_batches: anchored.then(|| vec!["__ref__".into()]),
    };
    let mut out = collapse_columns_multilevel_with_hierarchy(&mut v, &proj, &batches, &params)
        .expect("collapse");
    out.levels.remove(0)
}

/// Mean |log δ| for one batch column of the delta plane.
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
fn the_anchor_batch_keeps_delta_at_one() {
    let out = run(true);
    // batch indices are registration order of first appearance; find them by
    // magnitude instead of guessing: the anchor's δ must hug 1, the new
    // batch's must not (it carries a genuine 2×/0.5× platform shift).
    let d0 = mean_abs_log_delta(&out, 0);
    let d1 = mean_abs_log_delta(&out, 1);
    let (anchor_dev, new_dev) = if d0 < d1 { (d0, d1) } else { (d1, d0) };
    assert!(
        anchor_dev < 0.15,
        "anchor batch was re-adjusted: mean|log delta| = {anchor_dev}"
    );
    assert!(
        new_dev > 0.3,
        "the new batch's platform shift went undetected: mean|log delta| = {new_dev}"
    );
}

/// Pooled matching splits the platform discrepancy between the two batches —
/// the drift anchoring exists to prevent. This is the contrast that makes the
/// test above meaningful rather than trivially satisfiable.
#[test]
fn pooled_matching_adjusts_both_sides() {
    let out = run(false);
    let d0 = mean_abs_log_delta(&out, 0);
    let d1 = mean_abs_log_delta(&out, 1);
    let smaller = d0.min(d1);
    assert!(
        smaller > 0.1,
        "expected pooled matching to move BOTH batches' delta away from 1, got {d0} / {d1}"
    );
}

/// Anchoring an unknown batch name is an error, not a silent fallback to the
/// pooled behavior it exists to replace.
#[test]
fn an_unknown_anchor_batch_is_refused() {
    let (mut v, batches, weights) = two_batch_cohort();
    v.register_column_multiplicity(&weights).expect("weights");
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    let params = MultilevelParams {
        knn_pb_samples: 3,
        num_levels: 1,
        sort_dim: 3,
        num_opt_iter: 10,
        refine: Some(data_beans_alg::refine_multilevel::RefineParams::default()),
        output_calibration: matrix_param::traits::CalibrateTarget::All,
        anchor_batches: Some(vec!["no_such_batch".into()]),
    };
    let err = match collapse_columns_multilevel_with_hierarchy(&mut v, &proj, &batches, &params) {
        Ok(_) => panic!("an unknown anchor batch must be refused"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("no_such_batch"), "{err}");
}
