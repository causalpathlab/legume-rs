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

#[test]
fn an_id_past_the_table_is_rejected() {
    let dst = table(); // [6, 4]
    let ids = Tensor::from_vec(vec![0u32, 6], 2, &dev()).unwrap();
    let src = Tensor::ones((2, 4), candle_core::DType::F32, &dev()).unwrap();
    let err = index_add_rows(&dst, &ids, &src).unwrap_err().to_string();
    assert!(
        err.contains("out of range"),
        "expected an out-of-range error, got: {err}"
    );
}

#[test]
fn a_transposed_source_adds_the_same_as_a_contiguous_one() {
    // The backward of a gather can arrive as a transposed view; the scatter
    // must take it as it is.
    let dst = Tensor::zeros((6, 4), candle_core::DType::F32, &dev()).unwrap();
    let src_t = Tensor::from_vec((0..28).map(|i| i as f32).collect(), (4, 7), &dev()).unwrap();
    let src_view = src_t.t().unwrap(); // [7, 4], not contiguous
    assert!(!src_view.is_contiguous());
    let a = index_add_rows(&dst, &ids(), &src_view)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let b = index_add_rows(&dst, &ids(), &src_view.contiguous().unwrap())
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn a_wide_table_with_many_duplicate_ids_matches_index_add() {
    // Large enough to take the parallel path; ids pile onto a few rows so
    // every block sees duplicates.
    let (d, h, n) = (50usize, 128usize, 4000usize);
    let dst = Tensor::zeros((d, h), candle_core::DType::F32, &dev()).unwrap();
    let ids: Vec<u32> = (0..n).map(|i| ((i * 7919) % 13) as u32 + 3).collect();
    let ids_t = Tensor::from_vec(ids, n, &dev()).unwrap();
    let src = Tensor::from_vec(
        (0..n * h)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.25)
            .collect(),
        (n, h),
        &dev(),
    )
    .unwrap();
    let a = index_add_rows(&dst, &ids_t, &src).unwrap();
    let b = dst.index_add(&ids_t, &src, 0).unwrap();
    let diff = (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-3, "max |diff| = {diff}");
}
/// Elementwise results must not depend on how many threads ran them: the
/// same op on the same bytes, once on a one-thread pool and once on the
/// process pool, is bit-identical. Guards the threaded CPU maps.
#[test]
fn cpu_elementwise_results_do_not_depend_on_the_thread_count() {
    let n = 1 << 18; // above any parallel threshold
    let x = Tensor::from_vec(
        (0..n)
            .map(|i| ((i * 7919) % 1000) as f32 * 0.001 - 0.5)
            .collect(),
        (256, 1024),
        &dev(),
    )
    .unwrap();
    let gate = Tensor::from_vec(
        (0..256).map(|i| i as f32 * 0.01).collect(),
        (256, 1),
        &dev(),
    )
    .unwrap();
    let run = || -> Vec<Vec<f32>> {
        let e = x.exp().unwrap();
        let g = x.broadcast_mul(&gate).unwrap();
        let a = x.affine(3.0, -1.0).unwrap();
        let c = x
            .gt(0.0)
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap();
        let s = x.sum_keepdim(1).unwrap();
        let t = x.sum_all().unwrap().reshape((1, 1)).unwrap();
        [e, g, a, c, s, t]
            .iter()
            .map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            .collect()
    };
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let many = rayon::ThreadPoolBuilder::new()
        .num_threads(7)
        .build()
        .unwrap();
    let a = one.install(run);
    let b = many.install(run);
    let c = run();
    for (i, ((a, b), c)) in a.iter().zip(&b).zip(&c).enumerate() {
        assert!(
            a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits()),
            "op {i}: 1 vs 7 threads differ"
        );
        assert!(
            a.iter().zip(c).all(|(p, q)| p.to_bits() == q.to_bits()),
            "op {i}: 1 thread vs the process pool differ"
        );
    }
}
