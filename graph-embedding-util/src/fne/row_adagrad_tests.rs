use super::*;
use candle_util::candle_core::{DType, Device, Tensor, Var};

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn one_row_adagrad_step_matches_the_closed_form() {
    let dev = Device::Cpu;
    let var =
        Var::from_tensor(&Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (3, 2), &dev).unwrap())
            .unwrap();
    let grad = Tensor::from_vec(vec![1f32, -1., 0., 0., 2., 2.], (3, 2), &dev).unwrap();
    let mut opt = RowAdagrad::new(3, 0.1, &dev).unwrap();
    opt.step(&var, &grad).unwrap();
    // acc = mean_d(grad²) = [1, 0, 4]; std = sqrt(acc) + 1e-10
    // row0: p − 0.1·g/1 = [0.9, 2.1]; row2: p − 0.1·2/2 = [4.9, 5.9]
    let p = var.as_tensor().to_vec2::<f32>().unwrap();
    assert!(approx(p[0][0], 0.9, 1e-6) && approx(p[0][1], 2.1, 1e-6));
    assert!(approx(p[2][0], 4.9, 1e-6) && approx(p[2][1], 5.9, 1e-6));
    let acc = opt.accumulator().to_vec1::<f32>().unwrap();
    assert!(approx(acc[0], 1.0, 1e-6) && approx(acc[2], 4.0, 1e-6));
    // A second identical step divides by sqrt(2·acc).
    opt.step(&var, &grad).unwrap();
    let p = var.as_tensor().to_vec2::<f32>().unwrap();
    assert!(approx(p[0][0], 0.9 - 0.1 / 2f32.sqrt(), 1e-6));
    assert!(approx(p[2][1], 5.9 - 0.1 * 2.0 / 8f32.sqrt(), 1e-6));
}

#[test]
fn rows_with_zero_gradient_are_left_unchanged_and_keep_a_zero_accumulator() {
    let dev = Device::Cpu;
    let var = Var::from_tensor(
        &Tensor::from_vec(vec![1f32, 2., 3.3333, 4.25, 5., 6.], (3, 2), &dev).unwrap(),
    )
    .unwrap();
    let before = var.as_tensor().to_vec2::<f32>().unwrap();
    let grad = Tensor::from_vec(vec![1f32, -1., 0., 0., 2., 2.], (3, 2), &dev).unwrap();
    let mut opt = RowAdagrad::new(3, 0.1, &dev).unwrap();
    opt.step(&var, &grad).unwrap();
    opt.step(&var, &grad).unwrap();
    let after = var.as_tensor().to_vec2::<f32>().unwrap();
    assert_eq!(after[1], before[1], "bit-identical untouched row");
    assert_eq!(opt.accumulator().to_vec1::<f32>().unwrap()[1], 0.0);
    assert_ne!(after[0], before[0]);
}

#[test]
fn the_dense_gradient_of_index_select_sums_duplicate_rows_like_a_coalesced_sparse_update() {
    let dev = Device::Cpu;
    let var = Var::from_tensor(&Tensor::ones((6, 2), DType::F32, &dev).unwrap()).unwrap();
    let idx = Tensor::from_vec(vec![2u32, 2, 5], 3, &dev).unwrap();
    let w = Tensor::from_vec(vec![1f32, 1., 2., 2., 3., 3.], (3, 2), &dev).unwrap();
    let loss = var
        .as_tensor()
        .index_select(&idx, 0)
        .unwrap()
        .mul(&w)
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    let g = grads
        .get(&var)
        .expect("dense gradient on the table")
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(g.len(), 6);
    assert_eq!(g[2], vec![3.0, 3.0], "row 2 appears twice: 1 + 2");
    assert_eq!(g[5], vec![3.0, 3.0]);
    for r in [0, 1, 3, 4] {
        assert_eq!(g[r], vec![0.0, 0.0], "untouched row {r} is exactly zero");
    }
}

/// The accumulator is updated from the gradient tensor. If that update kept
/// its autograd history, the accumulator would hold a reference to every
/// step's gradient storage for the whole run (measured before the fix: RSS
/// grew ~650 MB per epoch on a 3000 × 2000 toy). Observable here with a
/// gradient whose history reaches a variable: after the step, nothing
/// reachable from the accumulator may lead back to it.
#[test]
fn the_accumulator_does_not_retain_the_gradients_autograd_graph() {
    let dev = Device::Cpu;
    let var =
        Var::from_tensor(&Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (3, 2), &dev).unwrap())
            .unwrap();
    let source = Var::from_tensor(
        &Tensor::from_vec(vec![0.5f32, -1., 0., 0., 2., 1.], (3, 2), &dev).unwrap(),
    )
    .unwrap();
    // A "gradient" with a live history back to `source`.
    let g = source.as_tensor().affine(2.0, 0.0).unwrap();
    assert!(
        g.sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .get(&source)
            .is_some(),
        "premise: the fake gradient is graph-attached"
    );
    let mut opt = RowAdagrad::new(3, 0.1, &dev).unwrap();
    opt.step(&var, &g).unwrap();
    let reach = opt.accumulator().sum_all().unwrap().backward().unwrap();
    assert!(
        reach.get(&source).is_none(),
        "the accumulator retains the gradient's graph"
    );
}
