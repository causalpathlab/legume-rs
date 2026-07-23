//! Unit tests for the velocity-drift SEM residual.

use super::{sem_penalty, PbSemTerm};
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
