//! Unit tests for the velocity-drift SEM residual and the learnable pb-DAG term.

use super::{acyclicity_series, sem_penalty, PbDagParams, PbDagTerm, PbSemTerm};
use crate::fit::lineage::PbLineageLevel;
use crate::fit::projection::PbLevelVelocity;
use candle_util::candle_core::{Device, Tensor, Var};
use candle_util::candle_nn::VarMap;

fn eye(p: usize, dev: &Device) -> Tensor {
    let v: Vec<f32> = (0..p * p).map(|i| f32::from(i / p == i % p)).collect();
    Tensor::from_vec(v, (p, p), dev).unwrap()
}

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

// ---- learned-DAG: learnable pb-DAG ----

/// A strictly-upper-triangular (acyclic) `W` has ~zero acyclicity; a 2-cycle is
/// strictly positive.
#[test]
fn acyclicity_zero_for_dag_positive_for_cycle() {
    let dev = Device::Cpu;
    // DAG: 0→1, 0→2, 1→2.
    let dag = Tensor::from_vec(
        vec![0.0f32, 0.5, 0.3, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0],
        (3, 3),
        &dev,
    )
    .unwrap();
    let h_dag = acyclicity_series(&dag, &eye(3, &dev), 6, 3)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(h_dag.abs() < 1e-6, "DAG acyclicity should be ~0 ({h_dag})");

    // 2-cycle: 0↔1.
    let cyc = Tensor::from_vec(vec![0.0f32, 0.6, 0.7, 0.0], (2, 2), &dev).unwrap();
    let h_cyc = acyclicity_series(&cyc, &eye(2, &dev), 6, 2)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        h_cyc > 1e-3,
        "cycle acyclicity should be positive ({h_cyc})"
    );
}

/// The learnable term builds (registering a zero-initialized `W`), and its loss is
/// finite, non-negative, and differentiable in both `e_cell` and `W`.
#[test]
fn dag_term_builds_and_loss_is_differentiable() {
    let dev = Device::Cpu;
    let h = 3;
    let vel = PbLevelVelocity {
        n_pb: 3,
        theta: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        delta: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };
    let vm = VarMap::new();
    let term = PbDagTerm::new(
        &vel,
        h,
        &PbDagParams::default(),
        "dag_test_w",
        &vm,
        &dev,
        None,
    )
    .unwrap()
    .unwrap();
    // W starts at zero (clean DAGMA start).
    assert!(term.w_dense().unwrap().iter().all(|&x| x == 0.0));

    let e = Var::from_tensor(
        &Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            (3, h),
            &dev,
        )
        .unwrap(),
    )
    .unwrap();
    let loss = term.dag_loss(e.as_tensor()).unwrap();
    let val = loss.to_scalar::<f32>().unwrap();
    assert!(
        val.is_finite() && val >= 0.0,
        "dag loss finite non-neg ({val})"
    );
    // Gradients flow to both the embedding and the W var.
    let grads = loss.backward().unwrap();
    assert!(grads.get(e.as_tensor()).is_some(), "no grad to e_cell");
    let w_var = vm.data().lock().unwrap().get("dag_test_w").unwrap().clone();
    assert!(grads.get(w_var.as_tensor()).is_some(), "no grad to W");
}
