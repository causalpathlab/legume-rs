use super::*;
use crate::batched_dot::{batched_matvec_shared, batched_weighted_sum};
use crate::value_transform::anscombe_lite;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops;

const D: usize = 7;
const H: usize = 4;
const N: usize = 3;
const K: usize = 4;

/// A deterministic spread of values, so a failure is reproducible.
fn ramp(n: usize, a: f32, b: f32) -> Vec<f32> {
    (0..n)
        .map(|i| a + b * ((i as f32 * 0.7).sin() + 0.3 * (i as f32 * 0.11).cos()))
        .collect()
}

/// The shared fixture: a repeated id inside one cell, a cell with two masked
/// slots, and a fully masked cell — the three cases an exact re-association
/// can silently get wrong.
struct Fixture {
    idx: Tensor,
    gate: Tensor,
    visible: Tensor,
    query: Tensor,
    table: Tensor,
    dev: Device,
}

fn fixture() -> Fixture {
    let dev = Device::Cpu;
    // cell 0: id 2 appears in two slots, everything visible
    // cell 1: two masked slots
    // cell 2: fully masked
    let ids: Vec<u32> = vec![2, 5, 2, 0, 1, 3, 6, 4, 0, 1, 2, 3];
    let idx = Tensor::from_vec(ids, (N, K), &dev).unwrap();
    let values = Tensor::from_vec(
        vec![
            4.0f32, 9.0, 1.0, 16.0, 2.0, 25.0, 7.0, 3.0, 11.0, 5.0, 8.0, 6.0,
        ],
        (N, K),
        &dev,
    )
    .unwrap();
    let gate = anscombe_lite(&values, None, None).unwrap();
    let visible = Tensor::from_vec(
        vec![
            1.0f32, 1.0, 1.0, 1.0, // cell 0
            1.0, 0.0, 1.0, 0.0, // cell 1
            0.0, 0.0, 0.0, 0.0, // cell 2
        ],
        (N, K),
        &dev,
    )
    .unwrap();
    let query = Tensor::from_vec(ramp(H, 0.1, 0.9), (1, H), &dev).unwrap();
    let table = Tensor::from_vec(ramp(D * H, 0.0, 1.0), (D, H), &dev).unwrap();
    Fixture {
        idx,
        gate,
        visible,
        query,
        table,
        dev,
    }
}

/// `1 [N, 1]` where a row has any visible slot, `0` where it has none — the
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

/// The pooling as the encoder wrote it before this module existed: gather one
/// row per slot into `[N, K, H]`, gate it, score it, softmax, weighted sum.
fn block_path(f: &Fixture, features: &FeatureEmbedding) -> (Tensor, Tensor) {
    let e_nkh = features
        .gather(&f.idx.flatten_all().unwrap())
        .unwrap()
        .reshape((N, K, H))
        .unwrap();
    let content = e_nkh.broadcast_mul(&f.gate.unsqueeze(2).unwrap()).unwrap();
    let scale = (H as f64).sqrt();
    let scores = batched_matvec_shared(&content, &f.query)
        .unwrap()
        .affine(1.0 / scale, 0.0)
        .unwrap();
    let neg_inf = f
        .visible
        .affine(-1.0, 1.0)
        .unwrap()
        .affine(-1e9, 0.0)
        .unwrap();
    let attn = ops::softmax(&(scores + neg_inf).unwrap(), 1).unwrap();
    let pooled = batched_weighted_sum(&attn, &content).unwrap();
    let pooled = pooled.broadcast_mul(&has_visible(&f.visible)).unwrap();
    (attn, pooled)
}

/// The re-associated path: a matvec, a gather from a vector, a scatter, a gemm.
fn scatter_path(f: &Fixture, features: &FeatureEmbedding) -> (Tensor, Tensor) {
    let rq_d = query_over_features(features, &f.query).unwrap();
    let scores =
        attention_scores_from_vector(&f.gate, &f.idx, &rq_d, &f.visible, 1.0 / (H as f64).sqrt())
            .unwrap();
    let attn = ops::softmax(&scores, 1).unwrap();
    let pooled = pool_by_scatter(&attn, &f.gate, &f.idx, features).unwrap();
    let pooled = pooled.broadcast_mul(&has_visible(&f.visible)).unwrap();
    (attn, pooled)
}

fn assert_close(got: &Tensor, want: &Tensor, tol: f32, what: &str) {
    let g: Vec<Vec<f32>> = got.to_vec2().unwrap();
    let w: Vec<Vec<f32>> = want.to_vec2().unwrap();
    assert_eq!(g.len(), w.len(), "{what}: row count");
    for (i, (gr, wr)) in g.iter().zip(&w).enumerate() {
        for (j, (a, b)) in gr.iter().zip(wr).enumerate() {
            assert!(
                (a - b).abs() < tol,
                "{what}: [{i}, {j}] {a} vs {b} (tol {tol})"
            );
        }
    }
}

/// The identity itself: on a fixture with a repeated id, masked slots and a
/// fully masked cell, the re-associated pool equals the block pool.
#[test]
fn equals_the_block_path_to_tolerance() {
    let f = fixture();
    let features = FeatureEmbedding::fixed(f.table.clone());
    let (attn_old, pooled_old) = block_path(&f, &features);
    let (attn_new, pooled_new) = scatter_path(&f, &features);
    assert_close(&attn_new, &attn_old, 1e-6, "attention");
    assert_close(&pooled_new, &pooled_old, 1e-5, "pool");
    assert_eq!(pooled_new.dims(), &[N, H]);
    let _ = &f.dev;
}

/// Both paths, on the same parameters, with `Σ pool²` as the loss. Returned as
/// `(block grads, scatter grads)` from two separate backward passes, so a
/// parameter's two gradients are read from the same graph it appears in.
fn grads_both_ways(
    f: &Fixture,
    features: &FeatureEmbedding,
) -> (
    candle_core::backprop::GradStore,
    candle_core::backprop::GradStore,
) {
    let (_, pooled_old) = block_path(f, features);
    let g_old = pooled_old
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    let (_, pooled_new) = scatter_path(f, features);
    let g_new = pooled_new
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    (g_old, g_new)
}

fn assert_grads_agree(
    g_old: &candle_core::backprop::GradStore,
    g_new: &candle_core::backprop::GradStore,
    var: &candle_core::Var,
    what: &str,
) {
    let a = g_old
        .get(var)
        .unwrap_or_else(|| panic!("{what}: the block path gave no gradient"));
    let b = g_new
        .get(var)
        .unwrap_or_else(|| panic!("{what}: the scatter path gave no gradient"));
    assert!(
        a.abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            > 1e-6,
        "{what}: the reference gradient is all zero, so this proves nothing"
    );
    assert_close(a, b, 1e-4, what);
}

/// No detach anywhere: every parameter must receive the same gradient from
/// either summation order — the table (or the module pair that composes it),
/// the query, and the value gate.
#[test]
fn gradients_agree() {
    let base = fixture();
    let dev = base.dev.clone();

    // Free table.
    let table = candle_core::Var::from_tensor(&base.table).unwrap();
    let query = candle_core::Var::from_tensor(&base.query).unwrap();
    let gate = candle_core::Var::from_tensor(&base.gate).unwrap();
    let f = Fixture {
        idx: base.idx.clone(),
        gate: gate.as_tensor().clone(),
        visible: base.visible.clone(),
        query: query.as_tensor().clone(),
        table: table.as_tensor().clone(),
        dev: dev.clone(),
    };
    let free = FeatureEmbedding::Free(table.as_tensor().clone());
    let (g_old, g_new) = grads_both_ways(&f, &free);
    assert_grads_agree(&g_old, &g_new, &table, "free table");
    assert_grads_agree(&g_old, &g_new, &query, "query, free table");
    assert_grads_agree(&g_old, &g_new, &gate, "gate, free table");

    // Composed feature side: the gradient has to reach both the membership
    // logits and the shared dictionary the same way on either path.
    let m = 3usize;
    let logits = candle_core::Var::from_tensor(
        &Tensor::from_vec(ramp(D * m, 0.0, 1.0), (D, m), &dev).unwrap(),
    )
    .unwrap();
    let mu = candle_core::Var::from_tensor(
        &Tensor::from_vec(ramp(m * H, 0.0, 1.0), (m, H), &dev).unwrap(),
    )
    .unwrap();
    let composed = FeatureEmbedding::Composed {
        logits: logits.as_tensor().clone(),
        mu: mu.as_tensor().clone(),
    };
    let (g_old, g_new) = grads_both_ways(&f, &composed);
    assert_grads_agree(&g_old, &g_new, &logits, "membership logits");
    assert_grads_agree(&g_old, &g_new, &mu, "module dictionary");
    assert_grads_agree(&g_old, &g_new, &query, "query, composed");
    assert_grads_agree(&g_old, &g_new, &gate, "gate, composed");
}

/// Attention weights and per-gene weights for the fixture, straight from the
/// pieces the encoder wires together.
fn parts(f: &Fixture, features: &FeatureEmbedding) -> (Tensor, Tensor, Tensor, Tensor) {
    let rq_d = query_over_features(features, &f.query).unwrap();
    let scores =
        attention_scores_from_vector(&f.gate, &f.idx, &rq_d, &f.visible, 1.0 / (H as f64).sqrt())
            .unwrap();
    let attn = ops::softmax(&scores, 1).unwrap();
    let (w_nk, w_nd, pool) = pool_parts(&attn, &f.gate, &f.idx, features).unwrap();
    (attn, w_nk, w_nd, pool)
}

/// A masked slot must leave with no attention and no weight — otherwise the
/// encoder pools the value it is being asked to impute.
#[test]
fn a_masked_slot_has_zero_attention_and_zero_weight() {
    let f = fixture();
    let features = FeatureEmbedding::fixed(f.table.clone());
    let (attn, w_nk, w_nd, _) = parts(&f, &features);
    let attn: Vec<Vec<f32>> = attn.to_vec2().unwrap();
    let w_nk: Vec<Vec<f32>> = w_nk.to_vec2().unwrap();
    let w_nd: Vec<Vec<f32>> = w_nd.to_vec2().unwrap();
    // Cell 1 has slots 1 and 3 masked; they hold genes 3 and 4, which it holds
    // nowhere else, so the scattered weight on those genes is zero too.
    for (slot, gene) in [(1usize, 3usize), (3, 4)] {
        assert!(
            attn[1][slot] < 1e-12,
            "masked slot {slot} kept attention {}",
            attn[1][slot]
        );
        assert!(
            w_nk[1][slot].abs() < 1e-12,
            "masked slot {slot} kept weight {}",
            w_nk[1][slot]
        );
        assert!(
            w_nd[1][gene].abs() < 1e-12,
            "masked gene {gene} kept weight {}",
            w_nd[1][gene]
        );
    }
    // The visible slots of that row still carry the whole softmax.
    assert!(
        (attn[1][0] + attn[1][2] - 1.0).abs() < 1e-6,
        "{:?}",
        attn[1]
    );
}

/// A cell with nothing visible must pool to an exact zero row, not to the
/// uniform average over padding that an all-`-1e9` softmax would give. That is
/// what the encoder's `has_visible` multiply is for, and both paths take it.
#[test]
fn a_fully_masked_cell_pools_to_zero() {
    let f = fixture();
    let features = FeatureEmbedding::fixed(f.table.clone());
    let (_, pooled) = scatter_path(&f, &features);
    let pooled: Vec<Vec<f32>> = pooled.to_vec2().unwrap();
    for (j, v) in pooled[2].iter().enumerate() {
        assert_eq!(*v, 0.0, "fully masked cell has a non-zero pool at {j}: {v}");
    }
    // and the rows that do see something are not zero, so this is a property
    // of the mask rather than of the whole fixture.
    assert!(pooled[0].iter().any(|v| v.abs() > 1e-6), "{:?}", pooled[0]);
}

/// The one place an exact re-association can fail silently: a gene named by
/// two slots of the same cell. The scatter must ADD the two weights.
#[test]
fn repeated_ids_accumulate_in_the_scatter() {
    let f = fixture();
    let features = FeatureEmbedding::fixed(f.table.clone());
    let (_, w_nk, w_nd, _) = parts(&f, &features);
    let w_nk: Vec<Vec<f32>> = w_nk.to_vec2().unwrap();
    let w_nd: Vec<Vec<f32>> = w_nd.to_vec2().unwrap();
    // Cell 0 names gene 2 in slots 0 and 2.
    let both = w_nk[0][0] + w_nk[0][2];
    assert!(
        w_nk[0][0] > 1e-6 && w_nk[0][2] > 1e-6,
        "both slots must carry weight for this to test anything: {:?}",
        w_nk[0]
    );
    assert!(
        (w_nd[0][2] - both).abs() < 1e-6,
        "repeated gene got {} instead of the sum {both}",
        w_nd[0][2]
    );
}

/// The point of the whole module: nothing formed here is `N·K·H`. Checked on
/// the shapes themselves at two very different `K`, so growing the context
/// cannot quietly reintroduce the block.
#[test]
fn no_intermediate_scales_with_k_times_h() {
    let dev = Device::Cpu;
    let (n, h, d) = (2usize, 8usize, 16usize);
    let table = Tensor::from_vec(ramp(d * h, 0.0, 1.0), (d, h), &dev).unwrap();
    let features = FeatureEmbedding::fixed(table);
    let query = Tensor::from_vec(ramp(h, 0.1, 0.9), (1, h), &dev).unwrap();
    for k in [64usize, 4096] {
        let ids: Vec<u32> = (0..n * k).map(|i| ((i * 7) % d) as u32).collect();
        let idx = Tensor::from_vec(ids, (n, k), &dev).unwrap();
        let gate = Tensor::from_vec(ramp(n * k, 2.0, 0.5), (n, k), &dev).unwrap();
        let visible = Tensor::ones((n, k), DType::F32, &dev).unwrap();
        let rq_d = query_over_features(&features, &query).unwrap();
        let scores =
            attention_scores_from_vector(&gate, &idx, &rq_d, &visible, 1.0 / (h as f64).sqrt())
                .unwrap();
        let attn = ops::softmax(&scores, 1).unwrap();
        let (w_nk, w_nd, pool) = pool_parts(&attn, &gate, &idx, &features).unwrap();

        let budget = (n * k).max(n * d).max(d * h).max(n * h);
        for (what, t) in [
            ("rq_d", &rq_d),
            ("scores", &scores),
            ("w_nk", &w_nk),
            ("W_nd", &w_nd),
            ("pool", &pool),
        ] {
            assert!(
                t.elem_count() <= budget,
                "K = {k}: {what} is {:?} = {} elements, over the budget {budget} \
                 (the block would be {})",
                t.dims(),
                t.elem_count(),
                n * k * h
            );
        }
        assert_eq!(pool.dims(), &[n, h]);
    }
}

/// One affine, not two.
///
/// [`masked_scores`] is the additive visibility mask both scorers used to
/// spell out inline as `(1 − v)` then `·(−1e9)`. A softmax is not forgiving
/// about where `−1e9` lands, so the single affine has to agree with the old
/// pair BIT for bit on a 0/1 mask, not merely to tolerance.
#[test]
fn masked_scores_equals_the_two_affine_form() {
    let f = fixture();
    let scores = Tensor::from_vec(ramp(N * K, 0.0, 2.0), (N, K), &f.dev).unwrap();
    // The form both `attention_scores_from_vector` and `attention_scores_dense`
    // carried before the helper existed, kept here as the reference.
    let neg_inf = f
        .visible
        .affine(-1.0, 1.0)
        .unwrap()
        .affine(-1e9, 0.0)
        .unwrap();
    let reference: Vec<Vec<f32>> = (&scores + neg_inf).unwrap().to_vec2().unwrap();
    let got: Vec<Vec<f32>> = masked_scores(&scores, &f.visible)
        .unwrap()
        .to_vec2()
        .unwrap();
    assert_eq!(got, reference, "the one-affine mask must be bit-identical");
}
