//! A `--from` run must reproduce the source run's finest partition exactly.
//!
//! The inherited membership need not agree with this run's marginal hash
//! leaves — a source whose high bits came from within-node residuals crosses
//! them by design — so the pb-samples must be built from the inherited labels
//! themselves, not from fresh marginal codes that are then modal-voted.
use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_partition, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::traits::Inference;
use rustc_hash::FxHashMap;

const D: usize = 8;
const N: usize = 96;

fn cohort() -> (SparseIoVec, Vec<&'static str>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("inherit.zarr");
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for j in 0..N {
        let wiggle = 1.0 + 0.05 * ((j % 5) as f32 - 2.0);
        for g in 0..D {
            let v = if j.is_multiple_of(2) {
                10.0 + 3.0 * (g % 2) as f32
            } else {
                4.0 + 2.0 * ((g + 1) % 2) as f32
            };
            triplets.push((g as u64, j as u64, v * wiggle));
        }
    }
    let mut b = create_sparse_from_triplets(
        &triplets,
        (D, N, triplets.len()),
        Some(path.to_str().expect("utf8")),
        Some(&data_beans::sparse_io::SparseIoBackend::Zarr),
    )
    .expect("backend");
    b.register_row_names_vec(
        &(0..D)
            .map(|g| format!("g{g}").into())
            .collect::<Vec<Box<str>>>(),
    );
    b.register_column_names_vec(
        &(0..N)
            .map(|c| format!("c{c}").into())
            .collect::<Vec<Box<str>>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    (v, vec!["one"; N], dir)
}

/// Same partition up to relabelling: the two labelings induce the same
/// equivalence on cells.
fn same_partition(a: &[usize], b: &[usize]) -> bool {
    let mut ab: FxHashMap<usize, usize> = FxHashMap::default();
    let mut ba: FxHashMap<usize, usize> = FxHashMap::default();
    a.iter()
        .zip(b)
        .all(|(&x, &y)| *ab.entry(x).or_insert(y) == y && *ba.entry(y).or_insert(x) == x)
}

#[test]
fn inherited_partition_is_reproduced_exactly() {
    let (mut v, batches, _dir) = cohort();
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    let mut params = MultilevelParams::new(4);
    params.num_levels = 1;
    params.sort_dim = 3;
    // Five groups that ignore the marginal leaves entirely.
    let inherited: Vec<Vec<usize>> = vec![(0..N).map(|c| (c * 7919) % 5).collect()];
    let out =
        collapse_columns_multilevel_with_partition(&mut v, &proj, &batches, &params, &inherited)
            .expect("collapse");
    assert_eq!(out.levels.len(), 1);
    assert_eq!(out.cell_to_pb_per_level.len(), 1);
    assert!(
        same_partition(&inherited[0], &out.cell_to_pb_per_level[0]),
        "the finest partition must be the inherited one, not a modal vote over marginal leaves"
    );
    assert_eq!(out.levels[0].mu_observed.posterior_mean().ncols(), 5);
}
