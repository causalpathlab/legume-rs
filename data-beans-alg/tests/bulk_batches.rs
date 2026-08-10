//! Bulk batches: summaries that train the dictionary but never touch δ.
//!
//! A bulk RNA-seq sample is a mixture over cell states. Matching it against
//! single-state pb-samples — in either direction — lets δ absorb
//! *composition* rather than platform, which is exactly the failure that
//! sank the first `senna update` ("δ ate the biology"). The safeguard is
//! structural, not a threshold: `MultilevelParams::bulk_batches` columns are
//! singleton pb-samples and singleton finest groups (never re-averaged,
//! matching the append-only reference discipline), and
//! `bbknn_match_one_pbsamp` bars them from matching both as receiver and as
//! source. Their imputed sums stay zero, so δ for a bulk batch rests at its
//! prior no matter how skewed the sample's composition is.

use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::traits::Inference;

const D: usize = 8;
const N_PER_BATCH: usize = 40;

fn type_profile(i: usize) -> Vec<f32> {
    if i % 2 == 0 {
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

/// Two cell batches sharing the 2-type structure — "base" clean, "shift"
/// under a genuine 2×/0.5× platform factor — plus a bulk batch of
/// composition-skewed mixtures of the SAME profiles.
fn cohort(tag: &str) -> (SparseIoVec, Vec<&'static str>) {
    let platform: Vec<f32> = (0..D).map(|g| if g < D / 2 { 2.0 } else { 0.5 }).collect();

    let mut cols: Vec<Vec<f32>> = Vec::new();
    let mut batches: Vec<&'static str> = Vec::new();
    for i in 0..N_PER_BATCH {
        let wiggle = 1.0 + 0.05 * ((i % 5) as f32 - 2.0);
        cols.push(type_profile(i).into_iter().map(|v| v * wiggle).collect());
        batches.push("base");
    }
    for i in 0..N_PER_BATCH {
        let wiggle = 1.0 + 0.04 * ((i % 3) as f32 - 1.0);
        cols.push(
            type_profile(i)
                .into_iter()
                .zip(platform.iter())
                .map(|(v, &p)| v * p * wiggle)
                .collect(),
        );
        batches.push("shift");
    }
    for j in 0..MIX.len() {
        cols.push(bulk_profile(j));
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

fn mean_abs_log_delta(
    out: &data_beans_alg::collapse_data::CollapsedOut,
    b: usize,
) -> f32 {
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
    //    (`mu_adjusted` is NOT the plane to read for zero-imputed groups:
    //    with no matches its residual/γ denominators sit at their priors.)
    for j in 0..MIX.len() {
        for (g, expected) in bulk_profile(j).into_iter().enumerate() {
            let back = finest.mu_observed.evidence_mean(g, k_new + j);
            assert!(
                (back - expected).abs() <= 1e-4 * expected.abs(),
                "bulk {j} gene {g}: expected {expected}, evidence {back}"
            );
        }
    }

    // 3. δ: the cell batches' genuine platform shift is detected, while the
    //    bulk batch — whose columns range from 90:10 to 10:90 composition —
    //    stays at the prior. If bulk were matched (either direction), the
    //    mixture-vs-component discrepancy would blow its δ off 1.
    let names = v.batch_names().expect("batch names");
    let idx = |n: &str| {
        names
            .iter()
            .position(|x| x.as_ref() == n)
            .expect("batch present")
    };
    let d_shift = mean_abs_log_delta(finest, idx("shift"));
    let d_bulk = mean_abs_log_delta(finest, idx("blk"));
    assert!(
        d_shift > 0.3,
        "platform shift on the cell batches went undetected: {d_shift}"
    );
    assert!(
        d_bulk < 0.15,
        "bulk composition leaked into delta: mean|log delta| = {d_bulk}"
    );
}
