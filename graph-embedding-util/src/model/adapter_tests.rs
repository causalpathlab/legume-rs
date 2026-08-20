//! Contract of the adapted feature side: `e_feat[g] = (rho[g] · W) (+ residual[g])`.
//!
//! `rho` is a fixed dictionary from another training run; `W` (and the
//! optional per-feature residual) are the only gene-side parameters. The
//! tests pin the composition against a dense reference, the parameter
//! surface (no `[n_features, H]` free Var may exist), the gate composing
//! after the composition, and the materialized snapshot's idempotence.

use super::*;
use candle_util::candle_core::Var;

fn dev() -> Device {
    Device::Cpu
}

fn small_rho() -> nalgebra::DMatrix<f32> {
    // [4 features x 3 source dims], every entry distinct.
    nalgebra::DMatrix::from_fn(4, 3, |r, c| (r * 3 + c) as f32 * 0.5 - 2.0)
}

fn build(residual: bool) -> (JointEmbedModel, VarMap) {
    let rho = small_rho();
    let vm = VarMap::new();
    let m = JointEmbedModel::new_adapted(
        AdapterInit {
            n_cells: 5,
            embedding_dim: 2,
            rho: &rho,
            b_feat: &[0.0; 4],
            b_cell: &[0.0; 5],
            seed: 2026,
            residual,
        },
        &vm,
        &dev(),
    )
    .unwrap();
    (m, vm)
}

fn var(vm: &VarMap, name: &str) -> Var {
    vm.data()
        .lock()
        .unwrap()
        .get(name)
        .cloned()
        .unwrap_or_else(|| panic!("missing var {name}"))
}

/// The gathered rows must equal the dense `rho · W` reference exactly,
/// including after `W` moves, and the residual must add on top when enabled.
#[test]
fn adapter_gather_matches_the_dense_composition() {
    let (m, vm) = build(true);
    let rho = small_rho();

    // Move W and the residual off their init so the test cannot pass by
    // accident of zeros.
    let w_var = var(&vm, "adapter_w");
    let w_new = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), &dev()).unwrap();
    w_var.set(&w_new).unwrap();
    let r_var = var(&vm, "adapter_residual");
    let r_new = Tensor::from_slice(&[10.0f32; 8], (4, 2), &dev()).unwrap();
    r_var.set(&r_new).unwrap();

    let idx = Tensor::from_slice(&[0u32, 2, 3], 3, &dev()).unwrap();
    let got = crate::loss::feat::gather_feature_rows(&m, &idx)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    let w_ref = [[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]];
    for (out_row, &g) in got.iter().zip([0usize, 2, 3].iter()) {
        for c in 0..2 {
            let want: f32 = (0..3).map(|k| rho[(g, k)] * w_ref[k][c]).sum::<f32>() + 10.0;
            assert!(
                (out_row[c] - want).abs() < 1e-6,
                "row {g} col {c}: got {} want {want}",
                out_row[c]
            );
        }
    }
}

/// Only `W` (and, when asked for, the residual) may be trainable on the gene
/// side: an adapted model must not register a free `[n_features, H]` e_feat
/// Var, which is the whole point of the parameterization.
#[test]
fn adapter_registers_only_w_and_optionally_the_residual() {
    let (_m, vm) = build(false);
    let keys: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    assert!(keys.iter().any(|k| k == "adapter_w"), "{keys:?}");
    assert!(
        !keys.iter().any(|k| k == "e_feat"),
        "an adapted model must not own a free e_feat Var: {keys:?}"
    );
    assert!(
        !keys.iter().any(|k| k == "adapter_residual"),
        "residual off => no residual Var: {keys:?}"
    );

    let (_m, vm) = build(true);
    let keys: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    assert!(keys.iter().any(|k| k == "adapter_residual"), "{keys:?}");
    // Zero-init: training starts exactly at rho . W.
    let r = var(&vm, "adapter_residual");
    assert!(r
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|&v| v == 0.0));
}

/// An installed pip mask gates the COMPOSED rows, exactly as it gates a free
/// model's rows: pip 0 rows come back zero, pip 1 rows come back unchanged.
#[test]
fn adapter_pip_gate_masks_the_composed_rows() {
    let (mut m, _vm) = build(false);
    // Deterministic mask: pip is exactly 0 or 1 per entry.
    let pip =
        Tensor::from_slice(&[1.0f32, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], (4, 2), &dev()).unwrap();
    m.install_gate_pip(GateKind::Identity, &pip).unwrap();
    m.resample_gate_mask().unwrap();

    let idx = Tensor::from_slice(&[0u32, 1, 2, 3], 4, &dev()).unwrap();
    let gated = crate::loss::feat::gather_feature_rows(&m, &idx)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    m.clear_gate_mask();
    let raw = crate::loss::feat::gather_feature_rows(&m, &idx)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    let pip_host = [[1.0f32, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    for g in 0..4 {
        for c in 0..2 {
            let want = raw[g][c] * pip_host[g][c];
            assert!(
                (gated[g][c] - want).abs() < 1e-6,
                "row {g} col {c}: got {} want {want}",
                gated[g][c]
            );
        }
    }
}

/// `materialize_e_feat` must write the current composition into the `e_feat`
/// field (the snapshot every output reader consumes), and calling it twice
/// must not change the answer.
#[test]
fn adapter_materialize_is_composed_and_idempotent() {
    let (mut m, vm) = build(false);
    let rho = small_rho();
    let w_var = var(&vm, "adapter_w");
    let w_new = Tensor::from_slice(&[0.0f32, 2.0, 1.0, 0.0, 0.5, 0.5], (3, 2), &dev()).unwrap();
    w_var.set(&w_new).unwrap();

    m.materialize_e_feat().unwrap();
    let once = m.e_feat.to_vec2::<f32>().unwrap();
    m.materialize_e_feat().unwrap();
    let twice = m.e_feat.to_vec2::<f32>().unwrap();
    assert_eq!(once, twice, "materialize must be idempotent");

    let w_ref = [[0.0f32, 2.0], [1.0, 0.0], [0.5, 0.5]];
    for g in 0..4 {
        for c in 0..2 {
            let want: f32 = (0..3).map(|k| rho[(g, k)] * w_ref[k][c]).sum();
            assert!(
                (once[g][c] - want).abs() < 1e-6,
                "row {g} col {c}: got {} want {want}",
                once[g][c]
            );
        }
    }
}
