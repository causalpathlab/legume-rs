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

/// The gate KL is a prior over GLOBAL parameters, so its weight must not move
/// when a THROUGHPUT knob moves. It used to be `GATE_KL_WEIGHT / batch_size`,
/// making the prior's share of the objective scale `1/B`: `--batch-size 4096`
/// cut it to a quarter and `64` raised it 16x, so a flag chosen for memory
/// silently retuned how sparse the learned feature sets came out.
///
/// Two earlier versions of this test were vacuous — one recomputed the formula
/// in its own closure, the next called a wrapper that ignored its argument. So
/// this one does not try to prove invariance by calling anything: the weight is
/// now a `const` with no runtime inputs at all, and invariance is a property of
/// the type system rather than of a value. What is left worth pinning is the
/// LEVEL, which is the part a refactor could silently move.
#[test]
fn gate_kl_step_weight_is_pinned_to_the_historical_level() {
    use crate::model::{GATE_KL_REF_UNITS, GATE_KL_STEP_WEIGHT, GATE_KL_WEIGHT};

    // THE LOAD-BEARING CLAIM: at the default `--batch-size 1024`, which both
    // `senna bge` and `senna gem` carry, this equals the `λ/batch_size` it
    // replaced. That is the whole reason the change was behaviour-preserving,
    // so it is pinned rather than argued.
    assert!((GATE_KL_STEP_WEIGHT - GATE_KL_WEIGHT / 1024.0).abs() < 1e-15);
    assert!((GATE_KL_STEP_WEIGHT - GATE_KL_WEIGHT / GATE_KL_REF_UNITS).abs() < 1e-15);
    // The reference IS both CLIs' default `--batch-size`; moving it is a
    // re-tune, not a refactor. Compile-time, since both sides are consts.
    const _: () = assert!(GATE_KL_REF_UNITS == 1024.0);

    // The general helper must agree with the constant at the units geu uses.
    assert!(
        (crate::model::gate_kl_step_weight(GATE_KL_WEIGHT, 1) - GATE_KL_STEP_WEIGHT).abs() < 1e-15
    );
}

/// `resolve_batches_per_epoch` is what turns `--batch-size` into a step budget;
/// pin the auto and explicit paths so a change there is deliberate.
#[test]
fn batches_per_epoch_resolves_auto_and_explicit() {
    use crate::training::resolve_batches_per_epoch;

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

    assert_eq!(resolve_batches_per_epoch(&params(1024, None), 8_192), 8);
    assert_eq!(resolve_batches_per_epoch(&params(1024, Some(7)), 8_192), 7);
    // Never zero, or the epoch would run no steps at all.
    assert!(resolve_batches_per_epoch(&params(65_536, None), 8_192) >= 1);
}
