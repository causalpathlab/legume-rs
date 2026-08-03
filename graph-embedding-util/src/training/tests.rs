//! Unit tests for the velocity-drift SEM residual.

use super::{sem_penalty, PbSemTerm, TrainingParams};
use crate::fit::lineage::PbLineageLevel;
use candle_util::candle_core::{Device, Tensor, Var};

/// One edge 0→1, parent velocity v̂₀ = (1,0), step s = 1. The residual is
/// `e₁ − e₀ − v̂₀`, so the penalty vanishes exactly when `e₁ = e₀ + (1,0)` and is
/// positive otherwise.
fn one_edge_level() -> PbLineageLevel {
    PbLineageLevel {
        n_pb: 2,
        edges: vec![(0, 1, 1.0)],
        velocity: vec![1.0, 0.0, /*n0*/ 0.0, 0.0 /*n1*/],
    }
}

#[test]
fn sem_penalty_zero_at_consistency() {
    let dev = Device::Cpu;
    let h = 2;
    let term = PbSemTerm::new(&one_edge_level(), h, 1.0, 1.0, &dev)
        .unwrap()
        .unwrap();

    // e₁ = e₀ + s·v̂₀ = (0.5,0.3) + (1,0) = (1.5,0.3) → residual 0.
    let consistent = Tensor::from_vec(vec![0.5f32, 0.3, 1.5, 0.3], (2, h), &dev).unwrap();
    let pen0 = sem_penalty(&consistent, &term)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(pen0 < 1e-6, "penalty should vanish at consistency ({pen0})");

    // Off-consistency (e₁ not a velocity-step ahead) → strictly positive.
    let bad = Tensor::from_vec(vec![0.5f32, 0.3, 0.5, 0.3], (2, h), &dev).unwrap();
    let pen1 = sem_penalty(&bad, &term)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        pen1 > 1e-3,
        "penalty should be positive off-consistency ({pen1})"
    );
}

#[test]
fn sem_penalty_gradient_step_reduces() {
    let dev = Device::Cpu;
    let h = 2;
    let term = PbSemTerm::new(&one_edge_level(), h, 1.0, 1.0, &dev)
        .unwrap()
        .unwrap();

    // Start off-consistency; one gradient-descent step on e_cell must lower it.
    let var =
        Var::from_tensor(&Tensor::from_vec(vec![0.5f32, 0.3, 0.5, 0.3], (2, h), &dev).unwrap())
            .unwrap();
    let before = sem_penalty(var.as_tensor(), &term)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    let loss = sem_penalty(var.as_tensor(), &term).unwrap();
    let grads = loss.backward().unwrap();
    let g = grads.get(var.as_tensor()).unwrap();
    let stepped = (var.as_tensor() - (g * 0.1).unwrap()).unwrap();
    let after = sem_penalty(&stepped, &term)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    assert!(
        after < before,
        "gradient step should reduce the SEM penalty ({before} -> {after})"
    );
}

/// A level with no oriented edges produces no term (nothing to penalize).
#[test]
fn empty_level_yields_no_term() {
    let dev = Device::Cpu;
    let level = PbLineageLevel {
        n_pb: 3,
        edges: vec![],
        velocity: vec![0.0; 6],
    };
    assert!(PbSemTerm::new(&level, 2, 1.0, 1.0, &dev).unwrap().is_none());
}

/// A level with a δ-less node still builds a term (the θ-pseudotime fallback keeps
/// it in the graph) — exercised via the fixed-KNN path in `fit`.
#[test]
fn sem_term_survives_multi_edge_level() {
    let dev = Device::Cpu;
    let level = PbLineageLevel {
        n_pb: 3,
        edges: vec![(0, 1, 1.0), (1, 2, 1.0)],
        velocity: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    assert!(PbSemTerm::new(&level, 2, 1.0, 1.0, &dev).unwrap().is_some());
}

/// The gate KL is a prior over GLOBAL parameters, so the share of the objective
/// it occupies must not move when a THROUGHPUT knob moves.
///
/// Before this, the weight was `GATE_KL_WEIGHT / batch_size`, so
/// `--batch-size 4096` cut the prior's share to a quarter and `64` raised it
/// 16x — a flag chosen for memory silently retuning how sparse the learned
/// feature sets came out. Nothing caught it because no test anywhere pinned
/// gate sparsity or KL magnitude.
///
/// Calls [`gate_kl_weight_for`], the SAME function the training loop applies,
/// rather than re-deriving the formula here: an earlier version of this test
/// recomputed the weight in its own closure and therefore could not fail when
/// the loop's formula changed.
#[test]
fn gate_kl_weight_is_invariant_to_batch_size_and_step_budget() {
    use crate::model::{GATE_KL_REF_UNITS, GATE_KL_WEIGHT};
    use crate::training::{gate_kl_weight_for, resolve_batches_per_epoch};

    const MAX_AXIS_UNITS: usize = 8_192;

    let params = |batch_size, batches_per_epoch| TrainingParams {
        epochs: 1,
        batches_per_epoch,
        batch_size,
        num_negatives: 4,
        seed: 42,
        objective: crate::loss::NceObjective::Softmax,
        feature_embedding_l2: 0.0,
        max_grad_norm: 0.0,
        delta_l2: 0.0,
    };

    let sizes = [1usize, 64, 512, 1024, 4096, 65_536];
    let base = gate_kl_weight_for(&params(1024, None));

    for bs in sizes {
        let got = gate_kl_weight_for(&params(bs, None));
        assert!(
            (got / base - 1.0).abs() < 1e-12,
            "--batch-size {bs} moved the gate-KL weight: {got} vs {base}"
        );
        // The auto budget must stay positive at every size, or the epoch is skipped.
        assert!(resolve_batches_per_epoch(&params(bs, None), MAX_AXIS_UNITS) >= 1);
    }
    // An explicit step budget must not move it either.
    assert!((gate_kl_weight_for(&params(1024, Some(7))) / base - 1.0).abs() < 1e-12);
    assert_eq!(
        resolve_batches_per_epoch(&params(1024, Some(7)), MAX_AXIS_UNITS),
        7
    );

    // THE LOAD-BEARING CLAIM: at the default `--batch-size 1024`, which both
    // `senna bge` and `faba gem` carry, the weight equals the `λ/batch_size`
    // this replaced. That is the whole reason it lands as a correctness fix
    // rather than a re-tune, so it is pinned rather than argued.
    let historical = GATE_KL_WEIGHT / 1024.0;
    assert!(
        (base - historical).abs() < 1e-15,
        "not behaviour-preserving at the default: {base} vs {historical}"
    );
    // Pin the level, so re-tuning it is a reviewed edit rather than drift.
    assert!((base - GATE_KL_WEIGHT / GATE_KL_REF_UNITS).abs() < 1e-15);
}
