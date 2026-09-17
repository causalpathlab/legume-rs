//! Contract of the anchored feature side: `e_feat[g] = base[g] + u_g · v` on
//! the anchored rows and `base[g]` alone elsewhere, with `base` the model's
//! own trained Var (its anchored rows are pinned by the caller's restore) and
//! the factors registered beside it under the shared LoRA names.

use super::*;
use candle_util::candle_core::Var;
use candle_util::candle_nn::{AdamW, Optimizer, ParamsAdamW};
use candle_util::lora::factor_names;

fn dev() -> Device {
    Device::Cpu
}

/// Four genes, the first and third anchored.
fn build() -> (JointEmbedModel, VarMap) {
    let e = nalgebra::DMatrix::from_fn(4, 2, |r, c| (r * 2 + c) as f32);
    let vm = VarMap::new();
    let m = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features: 4,
            n_cells: 3,
            embedding_dim: 2,
            seed: 7,
        },
        &ModelInit {
            e_feat: Some(&e),
            e_cell: None,
            b_feat: &[0.0; 4],
            b_cell: &[0.0; 3],
        },
        &vm,
        &dev(),
    )
    .unwrap()
    .with_lora(&vm, &dev(), 1, &[0, 2], 7)
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

#[test]
fn anchored_rows_carry_the_residual_and_free_rows_do_not() {
    let (m, vm) = build();
    let (u_name, v_name) = factor_names(E_FEAT_VAR_NAME);
    // v starts at zero: the composition is the base exactly.
    let rows = m.lora.as_ref().unwrap().compose().unwrap();
    assert_eq!(
        rows.to_vec2::<f32>().unwrap(),
        m.e_feat.to_vec2::<f32>().unwrap()
    );

    // Move v; only the anchored rows may change, by u_g · v.
    var(&vm, &v_name)
        .set(&Tensor::from_slice(&[10.0f32, 20.0], (1, 2), &dev()).unwrap())
        .unwrap();
    let u = var(&vm, &u_name).to_vec2::<f32>().unwrap();
    assert_eq!(u[1], vec![0.0], "an unanchored row has a zero factor");
    assert_ne!(u[0], vec![0.0], "an anchored row draws its factor");
    let idx = Tensor::from_slice(&[0u32, 1, 2, 3], 4, &dev()).unwrap();
    let got = crate::loss::feat::gather_feature_rows(&m, &idx)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let base = m.e_feat.to_vec2::<f32>().unwrap();
    for g in 0..4 {
        let shift = if g == 0 || g == 2 { u[g][0] } else { 0.0 };
        for c in 0..2 {
            let want = base[g][c] + shift * [10.0, 20.0][c];
            assert!(
                (got[g][c] - want).abs() < 1e-5,
                "row {g} col {c}: {} vs {want}",
                got[g][c]
            );
        }
    }
    // Materializing snapshots the same composition, and again is a no-op.
    let mut m = m;
    m.materialize_e_feat().unwrap();
    let once = m.e_feat.to_vec2::<f32>().unwrap();
    assert_eq!(once, got);
    m.materialize_e_feat().unwrap();
    assert_eq!(m.e_feat.to_vec2::<f32>().unwrap(), once);
}

#[test]
fn the_factors_are_registered_beside_the_base_and_nothing_else_is_added() {
    let (_, vm) = build();
    let mut names: Vec<String> = vm.data().lock().unwrap().keys().cloned().collect();
    names.sort();
    let (u_name, v_name) = factor_names(E_FEAT_VAR_NAME);
    assert_eq!(
        names,
        vec![
            "b_cell".to_string(),
            "b_feat".to_string(),
            "e_cell".to_string(),
            E_FEAT_VAR_NAME.to_string(),
            u_name,
            v_name
        ]
    );
}

/// The mask is part of the composition, so an unanchored row's factor gets no
/// gradient and an optimizer that decays every parameter keeps it at zero;
/// the anchored rows' factors move.
#[test]
fn an_optimizer_step_leaves_the_unanchored_factors_at_zero() {
    let (m, vm) = build();
    let (u_name, v_name) = factor_names(E_FEAT_VAR_NAME);
    var(&vm, &v_name)
        .set(&Tensor::from_slice(&[1.0f32, 1.0], (1, 2), &dev()).unwrap())
        .unwrap();
    let mut opt = AdamW::new(
        vm.all_vars(),
        ParamsAdamW {
            lr: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    let before = var(&vm, &u_name).to_vec2::<f32>().unwrap();
    let loss = m
        .lora
        .as_ref()
        .unwrap()
        .compose()
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    opt.backward_step(&loss).unwrap();
    let after = var(&vm, &u_name).to_vec2::<f32>().unwrap();
    assert_eq!(after[1], vec![0.0]);
    assert_eq!(after[3], vec![0.0]);
    assert_ne!(after[0], before[0]);
    assert_ne!(after[2], before[2]);
}

/// The ridge is the residual's summed row norm² over the anchored rows.
#[test]
fn the_ridge_is_the_anchored_residuals_summed_row_norm() {
    let (m, vm) = build();
    let (u_name, v_name) = factor_names(E_FEAT_VAR_NAME);
    var(&vm, &v_name)
        .set(&Tensor::from_slice(&[3.0f32, 4.0], (1, 2), &dev()).unwrap())
        .unwrap();
    let u = var(&vm, &u_name).to_vec2::<f32>().unwrap();
    let want: f32 = [0usize, 2].iter().map(|&g| u[g][0] * u[g][0] * 25.0).sum();
    let got = m
        .lora
        .as_ref()
        .unwrap()
        .ridge()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!((got - want).abs() < 1e-4 * want.max(1.0), "{got} vs {want}");
}
