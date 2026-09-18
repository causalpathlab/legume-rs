//! CNV strata: in-collapse hard partition — expression grouping cannot
//! mix donor-private clones with the mixable residual, and δ is estimated
//! only on matched (typically stratum-0) mass.
use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::collapse_data::{collapse_columns_multilevel_with_strata, MultilevelParams};
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::traits::Inference;
use rustc_hash::FxHashSet;

const D: usize = 8;
const NOVEL: [usize; 2] = [6, 7];

fn diploid(i: usize) -> Vec<f32> {
    let wiggle = 1.0 + 0.05 * ((i % 5) as f32 - 2.0);
    (0..D)
        .map(|g| (10.0 + 3.0 * (g % 2) as f32) * wiggle)
        .collect()
}

fn clone_profile(i: usize) -> Vec<f32> {
    let wiggle = 1.0 + 0.05 * ((i % 5) as f32 - 2.0);
    (0..D)
        .map(|g| {
            let base = if NOVEL.contains(&g) { 30.0 } else { 2.0 };
            base * wiggle
        })
        .collect()
}

fn cohort(
    tag: &str,
) -> (
    SparseIoVec,
    Vec<&'static str>,
    Vec<usize>,
    tempfile::TempDir,
) {
    // 40 diploid A, 40 clone A, 40 diploid B (1.5× platform).
    let mut cols: Vec<Vec<f32>> = Vec::new();
    let mut batches: Vec<&'static str> = Vec::new();
    let mut stratum: Vec<usize> = Vec::new();
    for i in 0..40 {
        cols.push(diploid(i));
        batches.push("A");
        stratum.push(0);
    }
    for i in 0..40 {
        cols.push(clone_profile(i));
        batches.push("A");
        stratum.push(1);
    }
    for i in 0..40 {
        cols.push(diploid(i).into_iter().map(|v| v * 1.5).collect());
        batches.push("B");
        stratum.push(0);
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("strata_{tag}.zarr"));
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for (j, col) in cols.iter().enumerate() {
        for (g, &v) in col.iter().enumerate() {
            if v != 0.0 {
                triplets.push((g as u64, j as u64, v));
            }
        }
    }
    let mut b = create_sparse_from_triplets(
        &triplets,
        (D, cols.len(), triplets.len()),
        Some(path.to_str().expect("utf8")),
        Some(&data_beans::sparse_io::SparseIoBackend::Zarr),
    )
    .expect("backend");
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
    (v, batches, stratum, dir)
}

fn params() -> MultilevelParams {
    MultilevelParams {
        knn_pb_samples: 3,
        num_levels: 1,
        sort_dim: 3,
        num_opt_iter: 20,
        refine: Some(data_beans_alg::refine_multilevel::RefineParams::default()),
        output_calibration: matrix_param::traits::CalibrateTarget::All,
        anchor_batches: None,
        bulk_batches: None,
        observe_panels: true,
        keep_finest_stats: false,
        pb_tree: None,
        strata: None,
    }
}

fn pb_sets(c2g: &[usize], stratum: &[usize]) -> (FxHashSet<usize>, FxHashSet<usize>) {
    let mut bucket = FxHashSet::default();
    let mut clone = FxHashSet::default();
    for (c, &s) in stratum.iter().enumerate() {
        if s == 0 {
            bucket.insert(c2g[c]);
        } else {
            clone.insert(c2g[c]);
        }
    }
    (bucket, clone)
}

/// Mean of `mat[(g, p)]` over genes in `genes` and pb-samples in `pbs`.
fn gene_mean(mat: &nalgebra::DMatrix<f32>, pbs: &FxHashSet<usize>, genes: &[usize]) -> f32 {
    if pbs.is_empty() {
        return 0.0;
    }
    pbs.iter()
        .map(|&p| genes.iter().map(|&g| mat[(g, p)]).sum::<f32>() / genes.len() as f32)
        .sum::<f32>()
        / pbs.len() as f32
}

fn novel_over_house(mat: &nalgebra::DMatrix<f32>, pbs: &FxHashSet<usize>) -> f32 {
    let house: Vec<usize> = (0..6).collect();
    gene_mean(mat, pbs, &NOVEL) / gene_mean(mat, pbs, &house).max(1e-3)
}

#[test]
fn private_clone_cells_never_share_a_pb_sample_with_the_bucket() {
    let (mut v, batches, stratum, _dir) = cohort("noshare");
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    let out = collapse_columns_multilevel_with_strata(&mut v, &proj, &batches, &params(), &stratum)
        .expect("collapse");
    let c2g = &out.cell_to_pb_per_level[0];
    let mut bucket: FxHashSet<usize> = FxHashSet::default();
    let mut clone: FxHashSet<usize> = FxHashSet::default();
    for (c, &s) in stratum.iter().enumerate() {
        if s == 0 {
            bucket.insert(c2g[c]);
        } else {
            clone.insert(c2g[c]);
        }
    }
    assert!(
        bucket.is_disjoint(&clone),
        "stratum 0 and clone 1 share a pb-sample: {:?}",
        bucket.intersection(&clone).collect::<Vec<_>>()
    );
}

#[test]
fn mixable_bucket_still_estimates_delta_across_donors() {
    let (mut v, batches, stratum, _dir) = cohort("delta");
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    let out = collapse_columns_multilevel_with_strata(&mut v, &proj, &batches, &params(), &stratum)
        .expect("collapse");
    let delta = out.levels[0]
        .delta
        .as_ref()
        .expect("stratum 0 has two batches so δ must exist");
    assert_eq!(delta.ncols(), 2);
    // Platform 1.5× on B vs A is a real batch effect: mean |log δ| > 0.
    let mean = delta.posterior_mean();
    let mut abs_log = 0.0f32;
    let mut n = 0.0f32;
    for g in 0..D {
        for b in 0..2 {
            abs_log += mean[(g, b)].max(1e-6).ln().abs();
            n += 1.0;
        }
    }
    assert!(abs_log / n > 0.02, "δ too close to 1: {}", abs_log / n);
}

#[test]
fn private_program_in_mu_adjusted_residual_is_batch_fold() {
    let (mut v, batches, stratum, _dir) = cohort("novel");
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    let out = collapse_columns_multilevel_with_strata(&mut v, &proj, &batches, &params(), &stratum)
        .expect("collapse");
    let (bucket_pb, clone_pb) = pb_sets(&out.cell_to_pb_per_level[0], &stratum);
    assert!(!clone_pb.is_empty());

    let mu_adj = out.levels[0]
        .mu_adjusted
        .as_ref()
        .expect("batched collapse emits mu_adjusted")
        .posterior_mean();
    let clone_ratio = novel_over_house(mu_adj, &clone_pb);
    let bucket_ratio = novel_over_house(mu_adj, &bucket_pb);
    assert!(
        clone_ratio > 3.0 * bucket_ratio.max(0.1),
        "mu_adjusted lost the private program: clone_ratio={clone_ratio} bucket_ratio={bucket_ratio}"
    );
    // Planted novel/house ≈ 15; under strata we recover most of it in μ.
    assert!(
        clone_ratio > 8.0,
        "mu_adjusted clone novel/house too weak: {clone_ratio}"
    );

    // δ is estimated on the bucket only; genes 6/7 are quiet there, so the
    // batch ratio on those genes must not dwarf the housekeeping genes.
    let delta = out.levels[0].delta.as_ref().unwrap().posterior_mean();
    let batch_names = v.batch_names().expect("batches registered");
    let batch_a = batch_names
        .iter()
        .position(|n| n.as_ref() == "A")
        .expect("batch A");
    let ratio = |g: usize| delta[(g, 0)].max(1e-6) / delta[(g, 1)].max(1e-6);
    let house_log = (0..6).map(ratio).fold(0.0f32, |a, r| a + r.ln().abs()) / 6.0;
    let novel_log = (NOVEL
        .iter()
        .copied()
        .map(ratio)
        .fold(0.0f32, |a, r| a + r.ln().abs()))
        / 2.0;
    assert!(
        novel_log < house_log + 1.0,
        "private genes absorbed as δ: novel={novel_log} house={house_log}"
    );

    // Unmatched clone pbs: mu_residual ≈ δ_A and flat across genes — that
    // flatness is why svd / Residual adjust without dividing out the CN.
    let resid = out.levels[0]
        .mu_residual
        .as_ref()
        .expect("batched collapse emits mu_residual")
        .posterior_mean();
    let resid_nh = novel_over_house(resid, &clone_pb);
    assert!(
        (resid_nh - 1.0).abs() < 0.25,
        "clone mu_residual should be flat across genes (novel/hk={resid_nh})"
    );
    let mut abs_rel = 0.0f32;
    let mut n = 0.0f32;
    for &p in &clone_pb {
        for g in 0..D {
            let r = resid[(g, p)].max(1e-6);
            let d = delta[(g, batch_a)].max(1e-6);
            abs_rel += (r / d).ln().abs();
            n += 1.0;
        }
    }
    assert!(
        abs_rel / n < 0.15,
        "clone mu_residual should track batch-A δ (mean |log resid/δ|={})",
        abs_rel / n
    );
}

/// Same cohort without strata: documents the failure the strata path fixes.
/// Cross-donor matching lets the private program leak into δ and into
/// `mu_residual` (then divided out of cells), while `mu_adjusted` dilutes it.
#[test]
fn without_strata_private_program_leaks_into_delta_and_residual() {
    let (mut v, batches, stratum, _dir) = cohort("nostrata");
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    let out = data_beans_alg::collapse_data::collapse_columns_multilevel_with_hierarchy(
        &mut v,
        &proj,
        &batches,
        &params(),
    )
    .expect("collapse");
    // Label clone pbs by the planted stratum of their cells (collapse itself
    // is free to mix; we still know which cells are the clone).
    let (bucket_pb, clone_pb) = pb_sets(&out.cell_to_pb_per_level[0], &stratum);
    // Without strata the partitions may overlap — take cells that are clone
    // only for the "clone" arm of the contrast.
    let pure_clone: FxHashSet<usize> = clone_pb.difference(&bucket_pb).copied().collect();
    let pbs = if pure_clone.is_empty() {
        &clone_pb
    } else {
        &pure_clone
    };

    let mu_adj = out.levels[0].mu_adjusted.as_ref().unwrap().posterior_mean();
    let resid = out.levels[0].mu_residual.as_ref().unwrap().posterior_mean();
    let delta = out.levels[0].delta.as_ref().unwrap().posterior_mean();

    let adj_nh = novel_over_house(mu_adj, pbs);
    let resid_nh = novel_over_house(resid, pbs);
    // Diluted vs planted ~15; residual carries a large novel/hk (program that
    // would be divided out of cells).
    assert!(
        adj_nh < 8.0,
        "expected diluted mu_adjusted without strata, got novel/hk={adj_nh}"
    );
    assert!(
        resid_nh > 2.0,
        "expected private program in mu_residual without strata, got novel/hk={resid_nh}"
    );

    // δ is no longer gene-flat: housekeeping tracks platform while novel
    // genes pick up a different A/B fold (program leaked into δ).
    let ratio = |g: usize| delta[(g, 0)].max(1e-6) / delta[(g, 1)].max(1e-6);
    let house_r = (0..6).map(ratio).fold(0.0f32, |a, r| a + r.ln()) / 6.0;
    let novel_r = (NOVEL
        .iter()
        .copied()
        .map(ratio)
        .fold(0.0f32, |a, r| a + r.ln()))
        / 2.0;
    assert!(
        (house_r - novel_r).abs() > 0.3,
        "expected δ A/B to disagree on novel vs housekeeping without strata: \
         log_house={house_r} log_novel={novel_r}"
    );
}

#[test]
fn a_single_stratum_matches_unstratified_hierarchy() {
    let (mut v, batches, _, _dir) = cohort("one");
    let n = v.num_columns();
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    let all_zero = vec![0usize; n];
    let p = params();
    let out_strata =
        collapse_columns_multilevel_with_strata(&mut v, &proj, &batches, &p, &all_zero)
            .expect("strata");
    let out_none = data_beans_alg::collapse_data::collapse_columns_multilevel_with_hierarchy(
        &mut v, &proj, &batches, &p,
    )
    .expect("none");
    let map = assert_partitions_equal_up_to_relabel(
        &out_none.cell_to_pb_per_level[0],
        &out_strata.cell_to_pb_per_level[0],
    );
    assert_collapsed_means_agree(&out_none.levels[0], &out_strata.levels[0], &map, 1e-6);
}

#[test]
fn strata_none_path_still_produces_delta() {
    let (mut v, batches, _, _dir) = cohort("none");
    let n = v.num_columns();
    let proj = v
        .project_columns_with_batch_correction(4, None, Some(&batches))
        .expect("proj")
        .proj;
    let mut p = params();
    p.strata = None;
    let out_none = data_beans_alg::collapse_data::collapse_columns_multilevel_with_hierarchy(
        &mut v, &proj, &batches, &p,
    )
    .expect("collapse");
    let all_zero = vec![0usize; n];
    let out_zero = collapse_columns_multilevel_with_strata(&mut v, &proj, &batches, &p, &all_zero)
        .expect("all-zero strata");
    let map = assert_partitions_equal_up_to_relabel(
        &out_none.cell_to_pb_per_level[0],
        &out_zero.cell_to_pb_per_level[0],
    );
    assert_collapsed_means_agree(&out_none.levels[0], &out_zero.levels[0], &map, 1e-6);
}

/// Same cell→group partition after independent compacting. Returns `a_group → b_group`.
fn assert_partitions_equal_up_to_relabel(
    a: &[usize],
    b: &[usize],
) -> std::collections::HashMap<usize, usize> {
    assert_eq!(a.len(), b.len());
    let mut map_ab = std::collections::HashMap::new();
    let mut map_ba = std::collections::HashMap::new();
    for (&x, &y) in a.iter().zip(b.iter()) {
        match map_ab.insert(x, y) {
            Some(prev) if prev != y => {
                panic!("partition mismatch: {x} maps to both {prev} and {y}")
            }
            _ => {}
        }
        match map_ba.insert(y, x) {
            Some(prev) if prev != x => {
                panic!("partition mismatch: {y} maps to both {prev} and {x}")
            }
            _ => {}
        }
    }
    map_ab
}

fn assert_collapsed_means_agree(
    a: &data_beans_alg::collapse_data::CollapsedOut,
    b: &data_beans_alg::collapse_data::CollapsedOut,
    a_to_b: &std::collections::HashMap<usize, usize>,
    tol: f32,
) {
    let align = |m: &nalgebra::DMatrix<f32>| -> nalgebra::DMatrix<f32> {
        // Reorder `m`'s columns (b's labeling) into a's group order.
        let mut out = nalgebra::DMatrix::zeros(m.nrows(), m.ncols());
        for (&ga, &gb) in a_to_b {
            out.column_mut(ga).copy_from(&m.column(gb));
        }
        out
    };
    let check = |name: &str,
                 xa: Option<&matrix_param::dmatrix_gamma::GammaMatrix>,
                 xb: Option<&matrix_param::dmatrix_gamma::GammaMatrix>| {
        match (xa, xb) {
            (None, None) => {}
            (Some(ga), Some(gb)) => {
                let ma = ga.posterior_mean();
                let mb = align(gb.posterior_mean());
                assert_eq!(ma.shape(), mb.shape(), "{name} shape");
                let mut max_abs = 0.0f32;
                for g in 0..ma.nrows() {
                    for c in 0..ma.ncols() {
                        max_abs = max_abs.max((ma[(g, c)] - mb[(g, c)]).abs());
                    }
                }
                assert!(
                    max_abs < tol,
                    "{name} posterior means disagree: max |Δ|={max_abs} (tol={tol})"
                );
            }
            _ => panic!("{name}: one side missing"),
        }
    };
    check("mu_observed", Some(&a.mu_observed), Some(&b.mu_observed));
    check(
        "mu_adjusted",
        a.mu_adjusted.as_ref(),
        b.mu_adjusted.as_ref(),
    );
    check(
        "mu_residual",
        a.mu_residual.as_ref(),
        b.mu_residual.as_ref(),
    );
    // δ is [genes × batches] — no group axis; compare directly.
    match (a.delta.as_ref(), b.delta.as_ref()) {
        (None, None) => {}
        (Some(da), Some(db)) => {
            let ma = da.posterior_mean();
            let mb = db.posterior_mean();
            assert_eq!(ma.shape(), mb.shape(), "delta shape");
            let mut max_abs = 0.0f32;
            for g in 0..ma.nrows() {
                for c in 0..ma.ncols() {
                    max_abs = max_abs.max((ma[(g, c)] - mb[(g, c)]).abs());
                }
            }
            assert!(
                max_abs < tol,
                "delta posterior means disagree: max |Δ|={max_abs} (tol={tol})"
            );
        }
        _ => panic!("delta: one side missing"),
    }
}
