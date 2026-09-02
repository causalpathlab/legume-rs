//! Tests of the learned-module parameterization.

use super::*;
use candle_util::candle_core::{Device, Var};
use candle_util::nn::sparsemax;

fn dev() -> Device {
    Device::Cpu
}

fn build(labels: Option<&[u32]>, own_mass: f32) -> (JointEmbedModel, VarMap) {
    let vm = VarMap::new();
    let m = JointEmbedModel::new_with_modules(
        ModuleInit {
            n_features: 6,
            n_cells: 3,
            embedding_dim: 4,
            n_modules: 3,
            init_labels: labels,
            init_own_mass: own_mass,
            b_feat: &[0f32; 6],
            b_cell: &[0f32; 3],
            seed: 9,
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
        .unwrap_or_else(|| panic!("{name} not registered"))
        .clone()
}

#[test]
fn own_mass_formula_holds_under_sparsemax() {
    for &(p, m) in &[(0.9f32, 3usize), (0.6, 8), (0.5, 128), (1.0, 16)] {
        let kappa = module_logit_for_own_mass(p, m);
        let mut row = vec![0f32; m];
        row[0] = kappa;
        let t = Tensor::from_vec(row, (1, m), &dev()).unwrap();
        let out: Vec<Vec<f32>> = sparsemax(&t).unwrap().to_vec2().unwrap();
        assert!(
            (out[0][0] - p).abs() < 1e-5,
            "M={m}: own mass {} for target {p} (κ={kappa})",
            out[0][0]
        );
        let rest: f32 = out[0][1..].iter().sum();
        assert!((rest - (1.0 - p)).abs() < 1e-5);
    }
}

#[test]
fn registers_exactly_the_module_vars_and_no_free_e_feat() {
    let (_, vm) = build(Some(&[0, 0, 1, 1, 2, 2]), 0.9);
    let names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    for want in [
        MODULE_LOGITS_VAR_NAME,
        MODULE_MU_VAR_NAME,
        MODULE_RESIDUAL_VAR_NAME,
        MODULE_BIAS_VAR_NAME,
        "e_cell",
        "b_feat",
        "b_cell",
    ] {
        assert!(
            names.iter().any(|n| n == want),
            "{want} missing from {names:?}"
        );
    }
    assert!(
        !names.iter().any(|n| n == E_FEAT_VAR_NAME),
        "a module model must not register a free e_feat Var"
    );
}

#[test]
fn warm_start_labels_set_the_membership() {
    let labels = [0u32, 0, 1, 1, 2, 2];
    let (m, _) = build(Some(&labels), 0.9);
    let pi: Vec<Vec<f32>> = m.module_membership().unwrap().unwrap().to_vec2().unwrap();
    for (g, &l) in labels.iter().enumerate() {
        assert!(
            (pi[g][l as usize] - 0.9).abs() < 1e-5,
            "row {g}: {:?}",
            pi[g]
        );
    }
}

#[test]
fn gather_matches_the_dense_composition() {
    let (m, _) = build(Some(&[0, 1, 2, 0, 1, 2]), 0.8);
    let full = m.modules.as_ref().unwrap().compose().unwrap();
    let idx = Tensor::from_vec(vec![4u32, 1, 5], 3, &dev()).unwrap();
    let rows = crate::loss::gather_feature_rows(&m, &idx).unwrap();
    let want = full.index_select(&idx, 0).unwrap();
    let a: Vec<f32> = rows.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-5);
    }
}

#[test]
fn materialize_composes_and_is_idempotent() {
    let (mut m, vm) = build(Some(&[0, 1, 2, 0, 1, 2]), 0.8);
    // Move μ so the snapshot taken at construction is stale.
    let mu = var(&vm, MODULE_MU_VAR_NAME);
    mu.set(&(mu.as_tensor() * 3.0).unwrap()).unwrap();
    m.materialize_e_feat().unwrap();
    let once: Vec<f32> = m.e_feat.flatten_all().unwrap().to_vec1().unwrap();
    let want: Vec<f32> = m
        .modules
        .as_ref()
        .unwrap()
        .compose()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(once, want);
    m.materialize_e_feat().unwrap();
    let twice: Vec<f32> = m.e_feat.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(once, twice);
}

#[test]
fn ridge_lands_on_the_residual() {
    let (m, vm) = build(Some(&[0, 1, 2, 0, 1, 2]), 0.8);
    let r = var(&vm, MODULE_RESIDUAL_VAR_NAME);
    r.set(&Tensor::ones((6, 4), candle_util::candle_core::DType::F32, &dev()).unwrap())
        .unwrap();
    let pen = m
        .feature_ridge(0.5)
        .unwrap()
        .expect("module model has a ridge");
    let grads = pen.backward().unwrap();
    assert!(
        grads.get(r.as_tensor()).is_some(),
        "ridge gradient must reach the residual"
    );
    let v: f32 = pen.to_scalar().unwrap();
    assert!(
        (v - 0.5 * 4.0).abs() < 1e-5,
        "λ·mean_g‖r_g‖² = 0.5·4, got {v}"
    );
}

#[test]
fn frozen_membership_receives_no_gradient() {
    let (m, vm) = build(Some(&[0, 1, 2, 0, 1, 2]), 0.8);
    let modules = m.modules.as_ref().unwrap();
    let logits = var(&vm, MODULE_LOGITS_VAR_NAME);
    let mu = var(&vm, MODULE_MU_VAR_NAME);
    let idx = Tensor::from_vec(vec![0u32, 3], 2, &dev()).unwrap();

    modules.set_frozen(true);
    let loss = crate::loss::gather_feature_rows(&m, &idx)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    assert!(
        grads.get(logits.as_tensor()).is_none(),
        "frozen logits must not get a gradient"
    );
    assert!(
        grads.get(mu.as_tensor()).is_some(),
        "μ trains during the warm-up"
    );

    modules.set_frozen(false);
    let loss = crate::loss::gather_feature_rows(&m, &idx)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    assert!(
        grads.get(logits.as_tensor()).is_some(),
        "released logits train"
    );
}

/// Every sharing head must carry the module tables: a head without them gathers
/// from `e_feat`, a detached snapshot, and trains a feature side nobody sees.
#[test]
fn every_shared_head_carries_the_modules() {
    let (m, vm) = build(Some(&[0, 1, 2, 0, 1, 2]), 0.8);
    let head = JointEmbedModel::new_sharing_features(
        ShareFeaturesArgs {
            n_cells: 5,
            embedding_dim: 4,
            shared_e_feat: m.e_feat.clone(),
            shared_b_feat: m.b_feat.clone(),
            e_cell_init: None,
            b_cell_init: &[0f32; 5],
            var_prefix: "pb_l0",
            seed: 3,
            shared_s_feat: None,
            shared_e_feat_raw: None,
            shared_e_feat_logstd: None,
            shared_gate_ibp_bias: None,
            gate: None,
            shared_modules: m.modules.clone(),
        },
        &vm,
        &dev(),
    )
    .unwrap();
    let mu = var(&vm, MODULE_MU_VAR_NAME);
    let idx = Tensor::from_vec(vec![2u32], 1, &dev()).unwrap();
    let loss = crate::loss::gather_feature_rows(&head, &idx)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    assert!(
        grads.get(mu.as_tensor()).is_some(),
        "the head's gather must reach the shared μ"
    );
    // And the frozen flag is one cell shared by both.
    m.modules.as_ref().unwrap().set_frozen(true);
    assert!(head.modules.as_ref().unwrap().is_frozen());
}

/// The gate sits on the composed ρ: with a gate enabled, the gather still reaches
/// μ and the residual through the gate multiplier, and the KL prices the LIVE
/// composition.
#[test]
fn gate_rides_on_the_composed_rows() {
    let (mut m, vm) = build(Some(&[0, 1, 2, 0, 1, 2]), 0.8);
    m.enable_feature_gate(
        FeatureGateSpec {
            temperature: 1.0,
            ibp_alpha: None,
        },
        &vm,
        &dev(),
    )
    .unwrap();
    let mu = var(&vm, MODULE_MU_VAR_NAME);
    let s = var(&vm, "s_feat");
    let idx = Tensor::from_vec(vec![1u32, 4], 2, &dev()).unwrap();
    let loss = crate::loss::gather_feature_rows(&m, &idx)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    assert!(grads.get(mu.as_tensor()).is_some());
    assert!(grads.get(s.as_tensor()).is_some());
    let kl = m.gate_kl().unwrap().expect("gated model has a KL");
    let grads = kl.backward().unwrap();
    assert!(
        grads.get(mu.as_tensor()).is_some(),
        "the effect KL must see the live μ"
    );
}
