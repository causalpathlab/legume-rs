//! Pooled (un-anchored) batch adjustment must land every batch's pseudobulks in
//! ONE common frame: the same cell type from two batches should have the same
//! `mu_adjusted` profile, up to matching noise. That is what every downstream
//! consumer assumes when it trains on `mu_adjusted` and later corrects cells with
//! `delta` (or `mu_residual`) relative to it.
//!
//! Planted: two batches over the same 2-type biology, batch "shifted" carrying a
//! per-gene platform factor (2× on the first half of the genes, 0.5× on the
//! second), batch "clean" carrying none.

use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_hierarchy, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::traits::Inference;

const D: usize = 8;
const PER_BATCH: usize = 120;

fn type_profile(i: usize) -> Vec<f32> {
    if i.is_multiple_of(2) {
        (0..D).map(|g| 10.0 + 3.0 * (g % 2) as f32).collect()
    } else {
        (0..D).map(|g| 4.0 + 2.0 * ((g + 1) % 2) as f32).collect()
    }
}

fn platform(g: usize) -> f32 {
    if g < D / 2 {
        2.0
    } else {
        0.5
    }
}

/// Columns `0..PER_BATCH` are the shifted batch, the next `PER_BATCH` the clean
/// one; types alternate by column parity in both.
///
/// The backend lives in a directory of its own, returned as a guard: the two
/// tests in this file run concurrently in one process, and a path keyed by
/// process id had them deleting and rebuilding the SAME zarr under each other
/// (a ~50% failure at `create backend`, never in serial).
fn cohort() -> (SparseIoVec, Vec<&'static str>, tempfile::TempDir) {
    let mut cols: Vec<Vec<f32>> = Vec::new();
    let mut batches: Vec<&'static str> = Vec::new();
    for i in 0..PER_BATCH {
        let wiggle = 1.0 + 0.05 * ((i % 5) as f32 - 2.0);
        cols.push(
            type_profile(i)
                .into_iter()
                .enumerate()
                .map(|(g, v)| v * platform(g) * wiggle)
                .collect(),
        );
        batches.push("shifted");
    }
    for i in 0..PER_BATCH {
        let wiggle = 1.0 + 0.05 * ((i % 5) as f32 - 2.0);
        cols.push(type_profile(i).into_iter().map(|v| v * wiggle).collect());
        batches.push("clean");
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("frame.zarr");
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for (j, col) in cols.iter().enumerate() {
        for (g, &v) in col.iter().enumerate() {
            triplets.push((g as u64, j as u64, v));
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
    (v, batches, dir)
}

/// Per-gene ratio `mu_adjusted(shifted-batch pbs) / mu_adjusted(clean-batch pbs)`
/// for one planted type, averaged over batch-pure pseudobulks.
/// Run the pooled collapse once; returns the finest level and the collapse's
/// batch names in δ-column order.
fn run_pooled() -> (
    data_beans_alg::collapse_data::MultilevelCollapseOut,
    Vec<Box<str>>,
) {
    // `_dir` keeps the backend's directory alive until the collapse is done.
    let (mut v, batches, _dir) = cohort();
    let corrected = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    // Real batches rarely hash together (most finest groups are batch-pure on
    // multi-study data); a batch indicator as the first sketch dimension keeps
    // that property here so each batch has its own pseudobulks to compare.
    // Cross-batch matching itself runs within each batch's own index, which this
    // constant-per-batch row does not disturb.
    let mut proj = nalgebra::DMatrix::<f32>::zeros(corrected.nrows() + 1, corrected.ncols());
    for c in 0..corrected.ncols() {
        proj[(0, c)] = if c < PER_BATCH { 10.0 } else { -10.0 };
    }
    proj.rows_mut(1, corrected.nrows()).copy_from(&corrected);
    let params = MultilevelParams {
        knn_pb_samples: 3,
        num_levels: 1,
        sort_dim: 4,
        num_opt_iter: 30,
        // No refinement sweeps: the hashed partition stands, so the batch
        // indicator above keeps every pseudobulk batch-pure.
        refine: Some(data_beans_alg::refine_multilevel::RefineParams {
            num_gibbs: 0,
            num_greedy: 0,
            ..Default::default()
        }),
        output_calibration: matrix_param::traits::CalibrateTarget::All,
        anchor_batches: None,
        bulk_batches: None,
        observe_panels: true,
        keep_finest_stats: false,
    };
    let out = collapse_columns_multilevel_with_hierarchy(&mut v, &proj, &batches, &params)
        .expect("collapse");
    let names = v.batch_names().expect("batch names");
    (out, names)
}

fn ratio_from(out: &data_beans_alg::collapse_data::MultilevelCollapseOut, ty: usize) -> Vec<f32> {
    let level = &out.levels[0];
    let cell_to_pb = &out.cell_to_pb_per_level[0];
    let adj = level
        .mu_adjusted
        .as_ref()
        .expect("two batches ⇒ mu_adjusted")
        .posterior_mean();
    let n_pb = adj.ncols();

    // Which (batch, type) each pb holds; keep only pure pbs.
    let mut count = vec![[[0usize; 2]; 2]; n_pb]; // [pb][batch][type]
    for (c, &p) in cell_to_pb.iter().enumerate() {
        let b = usize::from(c >= PER_BATCH);
        let t = (c % PER_BATCH) % 2;
        count[p][b][t] += 1;
    }
    let mut mean = [vec![0f32; D], vec![0f32; D]];
    let mut n = [0usize; 2];
    for (p, cnt) in count.iter().enumerate() {
        let tot: usize = cnt.iter().flatten().sum();
        for b in 0..2 {
            if cnt[b][ty] == tot && tot > 0 {
                for g in 0..D {
                    mean[b][g] += adj[(g, p)];
                }
                n[b] += 1;
            }
        }
    }
    assert!(
        n[0] >= 1 && n[1] >= 1,
        "need batch-pure pseudobulks of type {ty} in both batches, got {n:?}"
    );
    (0..D)
        .map(|g| (mean[0][g] / n[0] as f32) / (mean[1][g] / n[1] as f32))
        .collect()
}

/// The invariant every consumer of `mu_adjusted` relies on: after pooled
/// adjustment the two batches' pseudobulks of the same type coincide. If the
/// adjustment instead moved each batch into the OTHER batch's frame, the ratio
/// would be the inverse platform factor (0.5 on the first half, 2 on the second).
#[test]
fn pooled_adjustment_lands_both_batches_in_one_frame() {
    let (out, _) = run_pooled();
    for ty in 0..2 {
        let ratio = ratio_from(&out, ty);
        let worst = ratio.iter().map(|r| r.ln().abs()).fold(0f32, f32::max);
        let swapped: Vec<f32> = (0..D).map(|g| 1.0 / platform(g)).collect();
        assert!(
            worst < 0.3f32.ln().abs().min(0.3),
            "type {ty}: mu_adjusted(shifted)/mu_adjusted(clean) per gene = {ratio:?} — \
             not a common frame (|log ratio| up to {worst:.2}); the inverse platform \
             factor would be {swapped:?}"
        );
    }
}

/// `δ` carries the whole platform factor between the batches and is pinned to
/// geometric mean 1 per gene, so neither batch is "the" reference.
#[test]
fn delta_carries_the_platform_factor_symmetrically() {
    let (out, names) = run_pooled();
    let delta = out.levels[0]
        .delta
        .as_ref()
        .expect("two batches ⇒ delta")
        .posterior_mean();
    let shifted = names.iter().position(|n| n.as_ref() == "shifted").unwrap();
    let clean = names.iter().position(|n| n.as_ref() == "clean").unwrap();
    for g in 0..D {
        let ratio = delta[(g, shifted)] / delta[(g, clean)];
        assert!(
            (ratio / platform(g)).ln().abs() < 0.3f32.ln().abs().min(0.3),
            "gene {g}: delta ratio {ratio:.3} vs planted platform {}",
            platform(g)
        );
        let gm = (delta[(g, shifted)] * delta[(g, clean)]).sqrt();
        assert!(
            (gm - 1.0).abs() < 0.1,
            "gene {g}: geometric mean of delta = {gm:.3}, want 1"
        );
    }
}
