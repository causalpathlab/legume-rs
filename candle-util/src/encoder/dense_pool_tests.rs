use super::*;
use crate::encoder::scatter_pool::{
    attention_scores_from_vector, pool_by_scatter, query_over_features,
};
use crate::value_transform::anscombe_lite;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops;

const D: usize = 9;
const H: usize = 4;
const N: usize = 3;
const K: usize = 3;

/// A deterministic spread of values, so a failure is reproducible.
fn ramp(n: usize, a: f32, b: f32) -> Vec<f32> {
    (0..n)
        .map(|i| a + b * ((i as f32 * 0.7).sin() + 0.3 * (i as f32 * 0.11).cos()))
        .collect()
}

/// `1 [N, 1]` where a row has any visible gene, `0` where it has none — the
/// encoder's `has_visible_n1`, which both paths apply after pooling.
fn has_visible(visible: &Tensor) -> Tensor {
    visible
        .sum_keepdim(1)
        .unwrap()
        .gt(0.0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
}

struct Fixture {
    /// `[N, D]` the dense gate over every gene, zeros included.
    gate_nd: Tensor,
    /// `[N, K]` the planted support of each row.
    idx_nk: Tensor,
    query: Tensor,
    features: std::sync::Arc<FeatureEmbedding>,
    dev: Device,
}

/// Three rows, each with a planted `K`-gene support, and a `[D, H]` table.
///
/// The supports are disjoint sets of distinct ids: the dense path has one
/// weight per gene per row, so this is the shape on which it can be asked to
/// agree with the slot path at all.
fn fixture() -> Fixture {
    let dev = Device::Cpu;
    let values = Tensor::from_vec(ramp(N * D, 5.0, 4.0), (N, D), &dev).unwrap();
    let gate_nd = anscombe_lite(&values, None, None).unwrap();
    let ids: Vec<u32> = vec![
        1, 3, 5, // row 0
        0, 2, 6, // row 1
        4, 7, 8, // row 2
    ];
    let idx_nk = Tensor::from_vec(ids, (N, K), &dev).unwrap();
    let query = Tensor::from_vec(ramp(H, 0.1, 0.9), (1, H), &dev).unwrap();
    let table = Tensor::from_vec(ramp(D * H, 0.0, 1.0), (D, H), &dev).unwrap();
    Fixture {
        gate_nd,
        idx_nk,
        query,
        features: FeatureEmbedding::fixed(table),
        dev,
    }
}

/// `[N, D]` visible mask that is 1 exactly on `idx_nk`.
fn visible_on_support(f: &Fixture) -> Tensor {
    let ids: Vec<Vec<u32>> = f.idx_nk.to_vec2().unwrap();
    let mut buf = vec![0f32; N * D];
    for (r, row) in ids.iter().enumerate() {
        for &g in row {
            buf[r * D + g as usize] = 1.0;
        }
    }
    Tensor::from_vec(buf, (N, D), &f.dev).unwrap()
}

/// The dense pool: `(attn [N, D], pool [N, H])`.
fn dense_path(f: &Fixture, visible_nd: &Tensor) -> (Tensor, Tensor) {
    let rq_d = query_over_features(&f.features, &f.query).unwrap();
    let scores =
        attention_scores_dense(&f.gate_nd, &rq_d, Some(visible_nd), 1.0 / (H as f64).sqrt())
            .unwrap();
    let attn = ops::softmax(&scores, 1).unwrap();
    let pooled = pool_dense(&attn, &f.gate_nd, &f.features).unwrap();
    let pooled = pooled.broadcast_mul(&has_visible(visible_nd)).unwrap();
    (attn, pooled)
}

/// Stage 1's pool over the planted support, every slot visible.
fn scatter_path(f: &Fixture) -> Tensor {
    let gate_nk = f
        .gate_nd
        .gather(&f.idx_nk.to_dtype(DType::U32).unwrap(), 1)
        .unwrap();
    let visible_nk = Tensor::ones((N, K), DType::F32, &f.dev).unwrap();
    let rq_d = query_over_features(&f.features, &f.query).unwrap();
    let scores = attention_scores_from_vector(
        &gate_nk,
        &f.idx_nk,
        &rq_d,
        &visible_nk,
        1.0 / (H as f64).sqrt(),
    )
    .unwrap();
    let attn = ops::softmax(&scores, 1).unwrap();
    pool_by_scatter(&attn, &gate_nk, &f.idx_nk, &f.features).unwrap()
}

fn assert_close(got: &Tensor, want: &Tensor, tol: f32, what: &str) {
    let g: Vec<Vec<f32>> = got.to_vec2().unwrap();
    let w: Vec<Vec<f32>> = want.to_vec2().unwrap();
    assert_eq!(g.len(), w.len(), "{what}: row count");
    for (i, (gr, wr)) in g.iter().zip(&w).enumerate() {
        assert_eq!(gr.len(), wr.len(), "{what}: row {i} width");
        for (j, (a, b)) in gr.iter().zip(wr).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "{what}: [{i},{j}] {a} vs {b} (tol {tol})"
            );
        }
    }
}

/// THE bridge between stage 1 and stage 2.
///
/// Hide every gene outside a row's planted support and the dense softmax is
/// the slot softmax: the same table, the same query, the same gate values,
/// summed in a different order. If this drifts, the window-free encoder is not
/// the same pooling operator as the one the equivalence tests pinned to the
/// block path — it is a different model wearing its name.
#[test]
fn dense_pool_equals_the_scatter_pool_on_a_planted_support() {
    let f = fixture();
    let visible_nd = visible_on_support(&f);
    let (_, dense) = dense_path(&f, &visible_nd);
    let scatter = scatter_path(&f);
    assert_close(&dense, &scatter, 1e-5, "dense vs scatter pool");
}

/// A hidden gene gets exactly no say: no attention mass, and nothing of its
/// gate reaches the pool.
#[test]
fn a_hidden_gene_has_zero_attention_and_zero_weight() {
    let f = fixture();
    let visible_nd = visible_on_support(&f);
    let (attn, _) = dense_path(&f, &visible_nd);
    let a: Vec<Vec<f32>> = attn.to_vec2().unwrap();
    let v: Vec<Vec<f32>> = visible_nd.to_vec2().unwrap();
    for (r, (arow, vrow)) in a.iter().zip(&v).enumerate() {
        let mut seen = 0f32;
        for (g, (&w, &vis)) in arow.iter().zip(vrow).enumerate() {
            if vis == 0.0 {
                assert_eq!(w, 0.0, "row {r} gene {g} is hidden but carries weight {w}");
            } else {
                seen += w;
            }
        }
        assert!(
            (seen - 1.0).abs() < 1e-5,
            "row {r}: the visible genes must carry all the mass, got {seen}"
        );
    }
    // And the weight that multiplies ρ is zero there too, so no hidden gene's
    // value can leak into the pool through the gate.
    let w_nd = (&attn * &f.gate_nd).unwrap().to_vec2::<f32>().unwrap();
    for (r, (wrow, vrow)) in w_nd.iter().zip(&v).enumerate() {
        for (g, (&w, &vis)) in wrow.iter().zip(vrow).enumerate() {
            if vis == 0.0 {
                assert_eq!(w, 0.0, "row {r} gene {g}: hidden weight {w}");
            }
        }
    }
}

/// A row with nothing visible pools to exact zero, not to a uniform average
/// over the genes it was told not to look at.
#[test]
fn a_fully_hidden_row_pools_to_zero() {
    let f = fixture();
    let mut buf: Vec<f32> = visible_on_support(&f)
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for slot in buf.iter_mut().take(D) {
        *slot = 0.0; // row 0 sees nothing
    }
    let visible_nd = Tensor::from_vec(buf, (N, D), &f.dev).unwrap();
    let (_, pooled) = dense_path(&f, &visible_nd);
    let p: Vec<Vec<f32>> = pooled.to_vec2().unwrap();
    for (h, &x) in p[0].iter().enumerate() {
        assert_eq!(x, 0.0, "fully hidden row: pool[0,{h}] = {x}");
    }
    assert!(
        p[1].iter().any(|x| x.abs() > 1e-6),
        "a row that can still see must not be zeroed"
    );
}
