//! Tests for the row gather with a fast backward: the forward equals
//! `index_select`, the row-wise index-add equals `index_add` including
//! duplicate ids, and the gradients through both match candle's own.

use super::{gather_rows, index_add_rows};
use candle_core::{Device, Tensor, Var};

fn dev() -> Device {
    Device::Cpu
}

fn table() -> Tensor {
    Tensor::from_vec(
        (0..24).map(|i| i as f32 * 0.5 - 3.0).collect(),
        (6, 4),
        &dev(),
    )
    .unwrap()
}

fn ids() -> Tensor {
    Tensor::from_vec(vec![5u32, 0, 2, 5, 5, 1, 0], 7, &dev()).unwrap()
}

#[test]
fn gather_rows_matches_index_select() {
    let t = table();
    let a = gather_rows(&t, &ids()).unwrap().to_vec2::<f32>().unwrap();
    let b = t.index_select(&ids(), 0).unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(a, b);
}

#[test]
fn index_add_rows_matches_index_add_with_duplicate_ids() {
    let dst = table();
    let src =
        Tensor::from_vec((0..28).map(|i| (i as f32).sin()).collect(), (7, 4), &dev()).unwrap();
    let a = index_add_rows(&dst, &ids(), &src)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let b = dst
        .index_add(&ids(), &src, 0)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    for (ra, rb) in a.iter().zip(&b) {
        for (x, y) in ra.iter().zip(rb) {
            assert!((x - y).abs() < 1e-5, "{a:?} vs {b:?}");
        }
    }
}

#[test]
fn gather_rows_gradient_matches_index_select_gradient() {
    let w = Tensor::from_vec(
        (0..28).map(|i| (i as f32 * 0.37).cos()).collect(),
        (7, 4),
        &dev(),
    )
    .unwrap();
    let t1 = Var::from_tensor(&table()).unwrap();
    let g1 = (gather_rows(t1.as_tensor(), &ids()).unwrap() * &w)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    let t2 = Var::from_tensor(&table()).unwrap();
    let g2 = (t2.as_tensor().index_select(&ids(), 0).unwrap() * &w)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    let a = g1.get(&t1).unwrap().to_vec2::<f32>().unwrap();
    let b = g2.get(&t2).unwrap().to_vec2::<f32>().unwrap();
    for (ra, rb) in a.iter().zip(&b) {
        for (x, y) in ra.iter().zip(rb) {
            assert!((x - y).abs() < 1e-5, "gradient {a:?} vs {b:?}");
        }
    }
    // Rows never gathered (3, 4) get exactly zero.
    assert!(a[3].iter().all(|&x| x == 0.0) && a[4].iter().all(|&x| x == 0.0));
}

#[test]
fn index_add_rows_gradients_reach_dst_and_src() {
    let dst = Var::from_tensor(&table()).unwrap();
    let src =
        Var::from_tensor(&Tensor::ones((7, 4), candle_core::DType::F32, &dev()).unwrap()).unwrap();
    let c = Tensor::from_vec((0..24).map(|i| i as f32).collect(), (6, 4), &dev()).unwrap();
    let grads = (index_add_rows(dst.as_tensor(), &ids(), src.as_tensor()).unwrap() * &c)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(
        grads.get(&dst).unwrap().to_vec2::<f32>().unwrap(),
        c.to_vec2::<f32>().unwrap()
    );
    let gs = grads.get(&src).unwrap().to_vec2::<f32>().unwrap();
    let want = c.index_select(&ids(), 0).unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(
        gs, want,
        "src gradient is the upstream gradient gathered at ids"
    );
}
