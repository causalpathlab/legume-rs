use super::{factor_names, fold, LoraFactors, LoraPlus};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};

fn set(vm: &VarMap, name: &str, v: Vec<f32>, shape: (usize, usize)) {
    let data = vm.data().lock().unwrap();
    data[name]
        .set(&Tensor::from_vec(v, shape, &Device::Cpu).unwrap())
        .unwrap();
}

fn factors(n: usize, h: usize, rank: usize) -> (VarMap, LoraFactors) {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    let f = LoraFactors::new(n, h, rank, vb).unwrap();
    (vm, f)
}

/// `v` is zero at start, so the residual is nothing; rank 0 is refused.
#[test]
fn the_residual_starts_at_nothing_and_rank_zero_is_refused() {
    let (_vm, f) = factors(4, 3, 2);
    assert_eq!(f.rank(), 2);
    let r = f
        .residual()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(r.iter().all(|x| *x == 0.0));
    let u = f.u.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(u.iter().any(|x| *x != 0.0), "u is drawn, not zero");
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    assert!(LoraFactors::new(4, 3, 0, vb).is_err());
}

/// Gathered rows and the projected residual agree with `u·v` formed by hand.
#[test]
fn every_read_agrees_with_the_dense_residual() {
    let (n, h, rank) = (5, 3, 2);
    let (vm, f) = factors(n, h, rank);
    let u: Vec<f32> = (0..n * rank).map(|i| (i as f32 - 4.0) * 0.1).collect();
    let v: Vec<f32> = (0..rank * h).map(|i| 0.5 - i as f32 * 0.2).collect();
    set(&vm, super::U_VAR_NAME, u.clone(), (n, rank));
    set(&vm, super::V_VAR_NAME, v.clone(), (rank, h));
    let mut dense = vec![0f32; n * h];
    for g in 0..n {
        for k in 0..h {
            for j in 0..rank {
                dense[g * h + k] += u[g * rank + j] * v[j * h + k];
            }
        }
    }
    let got = f
        .residual()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (a, b) in got.iter().zip(&dense) {
        assert!((a - b).abs() < 1e-6, "residual {a} vs {b}");
    }
    let ids = Tensor::from_vec(vec![4u32, 1], 2, &Device::Cpu).unwrap();
    let rows = f
        .residual_rows(&ids)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want: Vec<f32> = [4usize, 1]
        .iter()
        .flat_map(|&g| dense[g * h..(g + 1) * h].to_vec())
        .collect();
    for (a, b) in rows.iter().zip(&want) {
        assert!((a - b).abs() < 1e-6, "rows {a} vs {b}");
    }
    let q = Tensor::from_vec(vec![1.0f32, -2.0, 0.5], (h, 1), &Device::Cpu).unwrap();
    let proj = f
        .project_dims(&q)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for g in 0..n {
        let want = dense[g * h] - 2.0 * dense[g * h + 1] + 0.5 * dense[g * h + 2];
        assert!(
            (proj[g] - want).abs() < 1e-5,
            "project {} vs {want}",
            proj[g]
        );
    }
}

/// Folding writes `base + u·v` into the base slot and removes the factors; a
/// map with no factors is untouched; the names follow the prefix.
#[test]
fn fold_leaves_a_plain_table_behind() {
    assert_eq!(
        factor_names("enc.feature"),
        (
            "enc.feature.lora_u".to_string(),
            "enc.feature.lora_v".to_string()
        )
    );
    let (n, h, rank) = (3, 2, 1);
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
    let _base = vb
        .get_with_hints((n, h), "table", candle_nn::Init::Const(1.0))
        .unwrap();
    let _f = LoraFactors::new(n, h, rank, vb.pp("t")).unwrap();
    set(&vm, "t.lora_u", vec![1.0, 2.0, 3.0], (n, rank));
    set(&vm, "t.lora_v", vec![0.5, -0.5], (rank, h));
    fold(&vm, "table", "t").unwrap();
    let tbl = vm.data().lock().unwrap();
    assert_eq!(tbl.len(), 1);
    let folded = tbl["table"]
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(folded, vec![1.5, 0.5, 2.0, 0.0, 2.5, -0.5]);
    drop(tbl);
    fold(&vm, "table", "t").unwrap();
    assert_eq!(vm.data().lock().unwrap().len(), 1);
}

/// The LoRA+ group holds exactly `v`, at the scaled rate, and a step through
/// it moves `v` alone; an unknown name is an error.
#[test]
fn the_lora_plus_group_steps_v_alone_at_the_scaled_rate() {
    let (vm, f) = factors(3, 2, 1);
    let plus = LoraPlus {
        v_var: "lora_v",
        lr_ratio: 4.0,
        ridge: 0.0,
    };
    let mut adam = plus.optimizer(&vm, 0.1).unwrap();
    assert!((adam.learning_rate() - 0.4).abs() < 1e-6);
    let u_before = f.u.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    // A loss with a nonzero gradient on both factors.
    let loss = (f.residual().unwrap() + 1.0)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    adam.step(&grads).unwrap();
    let v = f.v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(v.iter().any(|x| *x != 0.0), "v moved");
    assert_eq!(
        f.u.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        u_before,
        "u is not in the group"
    );
    assert!(LoraPlus {
        v_var: "nope",
        lr_ratio: 1.0,
        ridge: 0.0
    }
    .optimizer(&vm, 0.1)
    .is_err());
}

/// A pinned residual draws `u` on the pinned rows only, masks its gradient
/// to them, starts `v` at zero, and steps `v` faster by the ratio.
#[test]
fn a_pinned_residual_moves_only_the_pinned_rows_and_v_faster() {
    use super::PinnedLora;
    let dev = Device::Cpu;
    let l = PinnedLora::new(4, 3, 1, &[0, 2], 7, &dev).unwrap();
    let u =
        l.u.as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
    assert!(u[0] != 0.0 && u[1] == 0.0 && u[2] != 0.0 && u[3] == 0.0);
    assert_eq!(
        l.u_mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![1., 0., 1., 0.]
    );
    assert!(l
        .residual()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    let other = PinnedLora::new(4, 3, 1, &[0, 2], 8, &dev).unwrap();
    assert_ne!(
        u,
        other
            .u
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        "seeded"
    );
    assert!(PinnedLora::new(4, 3, 1, &[4], 7, &dev).is_err());
    // Move v off zero so u gets a gradient too, then step from a loss that
    // reaches every row of u.
    l.v.set(&Tensor::from_vec(vec![0.5f32, -0.5, 1.0], (1, 3), &dev).unwrap())
        .unwrap();
    let mut opt = l.optimizers(0.1, 4.0, &dev).unwrap();
    let loss = (l.residual().unwrap() + 1.0)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    l.step(&mut opt, &grads).unwrap();
    let u2 =
        l.u.as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
    assert!(u2[0] != u[0] && u2[2] != u[2], "pinned rows' factors moved");
    assert!(
        u2[1] == 0.0 && u2[3] == 0.0,
        "free rows' factors are masked"
    );
    let v2 =
        l.v.as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
    assert!((v2[0] - 0.5).abs() > 0.3, "v took the larger step: {v2:?}");
}

/// The Gram-form ridge is the dense residual's summed row norm², and it
/// reaches both factors.
#[test]
fn the_ridge_equals_the_dense_residuals_mean_row_norm() {
    use super::PinnedLora;
    let dev = Device::Cpu;
    let l = PinnedLora::new(5, 3, 2, &[0, 2, 4], 3, &dev).unwrap();
    l.v.set(&Tensor::from_vec(vec![0.5f32, -0.5, 1.0, 0.2, 0.1, -0.3], (2, 3), &dev).unwrap())
        .unwrap();
    let dense = l
        .residual()
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let ridge = l.ridge().unwrap().to_scalar::<f32>().unwrap();
    assert!((ridge - dense).abs() < 1e-5, "{ridge} vs {dense}");
    assert_eq!(l.n_pinned, 3);
    let grads = l.ridge().unwrap().backward().unwrap();
    assert!(
        grads.get(&l.u).is_some() && grads.get(&l.v).is_some(),
        "the ridge reaches both factors"
    );
}
