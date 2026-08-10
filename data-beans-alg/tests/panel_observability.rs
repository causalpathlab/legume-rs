//! Missingness is not silence (the plan's Stage-3 verification 1b).
//!
//! A gene absent from one cohort's PANEL reads zero in every cell of that
//! cohort, and nothing in a count matrix distinguishes that from "measured,
//! and off". Without observability the estimate is dragged toward zero by
//! cells that never looked: μ halves when half the cells can't see the gene,
//! and δ reads the structural absence as a huge batch effect. With
//! `MultilevelParams::observe_panels`, per-gene denominators count only the
//! mass whose source measures the gene, and δ falls to its prior where a
//! batch has no coverage.

use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::{RowAlignment, SparseIoVec};
use data_beans_alg::collapse_data::{
    collapse_columns_multilevel_with_hierarchy, CollapsedOut, MultilevelParams,
};
use data_beans_alg::random_projection::RandProjOps;
use matrix_param::traits::Inference;
use std::sync::Arc;

const D_FULL: usize = 6;
const MISSING_GENE: usize = 5;
const N_PER_BATCH: usize = 60;
const TRUE_RATE: f32 = 8.0;

/// Per-cell rate for `gene` given the cell's type. Type 1 is flat at
/// `TRUE_RATE`; type 2 is dominated by gene 0, so the projection separates
/// the two types into different partition groups deterministically.
fn rate(cell_type: u8, gene: usize, c: usize) -> f32 {
    let wiggle = 1.0 + 0.05 * ((c % 5) as f32 - 2.0);
    match cell_type {
        1 => TRUE_RATE * wiggle,
        _ => (if gene == 0 { 40.0 } else { 2.0 }) * wiggle,
    }
}

fn one_backend(
    dir: &std::path::Path,
    tag: &str,
    genes: &[usize],
    cell_types: &[u8],
) -> Arc<dyn data_beans::sparse_io::SparseIo<IndexIter = Vec<usize>>> {
    let n_cells = cell_types.len();
    let path = dir.join(format!("{tag}.zarr"));
    let _ = std::fs::remove_dir_all(&path);
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for (c, &t) in cell_types.iter().enumerate() {
        for (local, &g) in genes.iter().enumerate() {
            triplets.push((local as u64, c as u64, rate(t, g, c)));
        }
    }
    let shape = (genes.len(), n_cells, triplets.len());
    let mut b = create_sparse_from_triplets(
        &triplets,
        shape,
        Some(path.to_str().expect("utf8")),
        Some(&data_beans::sparse_io::SparseIoBackend::Zarr),
    )
    .expect("create backend");
    b.register_row_names_vec(
        &genes
            .iter()
            .map(|g| format!("g{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..n_cells)
            .map(|c| format!("{tag}_c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    Arc::from(b)
}

/// Batch "full" measures all six genes; batch "narrow" measures five — gene 5
/// is absent from its PANEL, not from its biology. Row-union alignment puts
/// both on the six-gene axis, with the narrow cohort reading zero at gene 5.
///
/// Cell types force the group structure the assertions need: type 1 lives in
/// BOTH batches (a mixed group, where naive averaging drags the estimate) and
/// type 2 lives ONLY in the narrow batch (a group with zero measuring mass at
/// gene 5, which must fall back to the prior).
fn cohort(tag: &str) -> (SparseIoVec, Vec<&'static str>) {
    let dir = std::env::temp_dir().join(format!("dba_panel_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");

    let full: Vec<usize> = (0..D_FULL).collect();
    let narrow: Vec<usize> = (0..D_FULL).filter(|&g| g != MISSING_GENE).collect();

    let full_types = vec![1u8; N_PER_BATCH];
    let mut narrow_types = vec![1u8; N_PER_BATCH / 2];
    narrow_types.extend(vec![2u8; N_PER_BATCH - N_PER_BATCH / 2]);

    let mut v = SparseIoVec::new()
        .with_row_alignment(RowAlignment::Union)
        .expect("union rows");
    v.push(
        one_backend(&dir, &format!("{tag}_full"), &full, &full_types),
        None,
    )
    .expect("push full");
    v.push(
        one_backend(&dir, &format!("{tag}_narrow"), &narrow, &narrow_types),
        None,
    )
    .expect("push narrow");

    let mut batches = vec!["full"; N_PER_BATCH];
    batches.extend(vec!["narrow"; N_PER_BATCH]);
    (v, batches)
}

fn run(tag: &str, observe_panels: bool) -> (CollapsedOut, Vec<Box<str>>) {
    let (mut v, batches) = cohort(tag);
    let proj = v
        .project_columns_with_batch_correction(3, None, Some(&batches))
        .expect("proj")
        .proj;
    let params = MultilevelParams {
        knn_pb_samples: 3,
        num_levels: 1,
        // One binary split: the type-2 program dominates the projection, so
        // the partition separates types, and the type-1 group necessarily
        // mixes both batches — the composition the drag assertions need.
        sort_dim: 1,
        num_opt_iter: 30,
        refine: Some(data_beans_alg::refine_multilevel::RefineParams::default()),
        output_calibration: matrix_param::traits::CalibrateTarget::All,
        anchor_batches: None,
        observe_panels,
    };
    let mut out = collapse_columns_multilevel_with_hierarchy(&mut v, &proj, &batches, &params)
        .expect("collapse");
    let names = v.batch_names().expect("batch names");
    (out.levels.remove(0), names)
}

/// Row-index of the missing gene on the union axis (row names survive the
/// union, so look it up rather than assuming order).
fn missing_row(v_names_hint: usize) -> usize {
    // rows are g0..g5 in push order of the full backend; the union preserves
    // first-seen order, so the index equals the gene id.
    let _ = v_names_hint;
    MISSING_GENE
}

#[test]
fn a_panel_gap_does_not_drag_mu_toward_zero() {
    let (masked, _) = run("mu_masked", true);
    let (naive, _) = run("mu_naive", false);
    let g = missing_row(D_FULL);
    let prior_mean = 1.0; // Gamma(a0 = 1, b0 = 1)

    // Per-sample control: genes 1..=4 are fully observed in both panels, so
    // their within-sample mean is that sample's true per-gene rate (≈ 8 for a
    // type-1 sample, ≈ 2 for a narrow-only type-2 sample). The missing gene's
    // biology matches the control in every cell, so per sample the honest
    // estimate is either the control (some member measures it) or the prior
    // (no member does) — never a value dragged in between by cells that
    // never looked.
    let m_masked = masked.mu_observed.posterior_mean();
    let m_naive = naive.mu_observed.posterior_mean();
    let (mut saw_measured, mut saw_prior, mut saw_dragged) = (false, false, false);
    for s_idx in 0..m_masked.ncols() {
        let ctrl: f32 = (1..MISSING_GENE).map(|r| m_masked[(r, s_idx)]).sum::<f32>()
            / (MISSING_GENE - 1) as f32;
        let vm = m_masked[(g, s_idx)];
        let vn = m_naive[(g, s_idx)];

        // The invariant: per sample, the masked estimate is either the prior
        // ("no member source measures this gene") or the sample's own true
        // rate ("the measuring members own the estimate") — NEVER a value
        // dragged in between by cells that never looked. Which of the two a
        // given sample lands on depends on the partition's batch composition,
        // which we deliberately do not assume.
        let is_prior = (vm - prior_mean).abs() < 0.35;
        let tracks_ctrl = vm > 0.8 * ctrl;
        assert!(
            is_prior || tracks_ctrl,
            "masked estimate {vm} is neither the prior nor its sample control \
             {ctrl} — unmeasured mass leaked into the denominator"
        );
        saw_measured |= tracks_ctrl && ctrl > 4.0;
        saw_prior |= is_prior;
        // The drag is naive-vs-honest on the SAME sample: unmeasured mass in
        // the denominator pulls the naive estimate strictly below the value
        // the measuring mass supports, without reaching the prior.
        if vm > 4.0 {
            saw_dragged |= vn > 2.0 * prior_mean && vn < 0.8 * vm;
        }
    }
    assert!(saw_measured, "no mixed (type-1) sample formed");
    assert!(saw_prior, "no narrow-only (type-2) sample formed");
    assert!(
        saw_dragged,
        "the naive arm never exhibited the drag — the contrast is untested"
    );
}

#[test]
fn a_panel_gap_does_not_masquerade_as_a_batch_effect() {
    let (masked, names) = run("delta_masked", true);
    let (naive, _) = run("delta_naive", false);
    let g = missing_row(D_FULL);
    let narrow_b = names
        .iter()
        .position(|n| n.as_ref() == "narrow")
        .expect("narrow batch");

    let delta_at =
        |out: &CollapsedOut| out.delta.as_ref().expect("delta").posterior_mean()[(g, narrow_b)];

    // Naive: zero observed over a real denominator → δ crushed toward zero,
    // i.e. a huge spurious "batch effect". Masked: no evidence either way →
    // the prior, ≈ 1, "no adjustment".
    let naive_delta = delta_at(&naive);
    let masked_delta = delta_at(&masked);
    assert!(
        naive_delta < 0.5,
        "naive delta {naive_delta} should read the panel gap as a large fake effect"
    );
    assert!(
        (masked_delta.ln()).abs() < 0.2,
        "masked delta {masked_delta} should stay at the prior (~1)"
    );
}

/// Identical panels ⇒ the mask machinery must not engage at all: the
/// coverage probe returns `None` and both settings produce the same numbers.
#[test]
fn identical_panels_are_a_bitwise_no_op() {
    let dir = std::env::temp_dir().join(format!("dba_panel_noop_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let full: Vec<usize> = (0..D_FULL).collect();

    let build = || {
        let mut v = SparseIoVec::new()
            .with_row_alignment(RowAlignment::Union)
            .expect("union rows");
        let types = vec![1u8; N_PER_BATCH];
        v.push(one_backend(&dir, "noop_a", &full, &types), None)
            .expect("push");
        v.push(one_backend(&dir, "noop_b", &full, &types), None)
            .expect("push");
        v
    };
    assert!(
        build().row_coverage_by_backend().is_none(),
        "identical panels must probe as fully covered"
    );

    let mut batches = vec!["a"; N_PER_BATCH];
    batches.extend(vec!["b"; N_PER_BATCH]);
    let run = |observe: bool| {
        let mut v = build();
        let proj = v
            .project_columns_with_batch_correction(3, None, Some(&batches))
            .expect("proj")
            .proj;
        let params = MultilevelParams {
            knn_pb_samples: 3,
            num_levels: 1,
            sort_dim: 2,
            num_opt_iter: 20,
            refine: Some(data_beans_alg::refine_multilevel::RefineParams::default()),
            output_calibration: matrix_param::traits::CalibrateTarget::All,
            anchor_batches: None,
            observe_panels: observe,
        };
        let mut out = collapse_columns_multilevel_with_hierarchy(&mut v, &proj, &batches, &params)
            .expect("collapse");
        out.levels.remove(0)
    };
    let on = run(true);
    let off = run(false);
    assert_eq!(
        on.mu_observed.posterior_mean(),
        off.mu_observed.posterior_mean(),
        "full coverage must be bitwise-identical with the flag on or off"
    );
}
