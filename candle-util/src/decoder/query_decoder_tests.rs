//! Tests for the query decoder: shapes, the masking contract (a masked or
//! padded slot is never read), rows with nothing visible read nothing,
//! permutation invariance over the context slots, the gradient contract (a
//! query's read depends on the gate and embedding of the slots it attends to
//! and on no other row), the projection-through-ρ identity, and a host
//! reference on a tiny instance.

use super::{QueryDecoder, QueryInput};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

const H: usize = 4;
const D: usize = 7;
const N: usize = 2;
const V: usize = 5;
const Q: usize = 3;
const RANK: usize = 3;

fn dev() -> Device {
    Device::Cpu
}

/// The feature side these tests read through: a fixed table.
fn features() -> std::sync::Arc<crate::feature_embedding::FeatureEmbedding> {
    crate::feature_embedding::FeatureEmbedding::fixed(rho())
}

fn rho() -> Tensor {
    let v: Vec<f32> = (0..D * H)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
        .collect();
    Tensor::from_vec(v, (D, H), &dev()).unwrap()
}

/// Context gene ids per slot; slot 2 of row 0 and slot 3 of row 1 are pads
/// (index 0, gate 0, not visible).
fn indices() -> Tensor {
    Tensor::from_vec(vec![3u32, 5, 0, 1, 6, 2, 4, 6, 0, 1], (N, V), &dev()).unwrap()
}

/// The Anscombe gate per slot (what the encoder multiplies ρ by).
fn gate() -> Tensor {
    #[rustfmt::skip]
    let v: Vec<f32> = vec![
        1.5, 0.7, 0.0, 2.2, 0.4,
        0.9, 1.1, 0.3, 0.0, 1.8,
    ];
    Tensor::from_vec(v, (N, V), &dev()).unwrap()
}

/// Row 0 sees slots {0, 1, 3}; row 1 sees {0, 2, 4}.
fn visible() -> Tensor {
    #[rustfmt::skip]
    let v: Vec<f32> = vec![
        1.0, 1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0, 1.0,
    ];
    Tensor::from_vec(v, (N, V), &dev()).unwrap()
}

fn query_ids() -> Tensor {
    Tensor::from_vec(vec![0u32, 3, 6, 1, 1, 5], (N, Q), &dev()).unwrap()
}

fn input<'a>(
    indices: &'a Tensor,
    gate: &'a Tensor,
    visible: &'a Tensor,
    query_ids: &'a Tensor,
) -> QueryInput<'a> {
    QueryInput {
        indices,
        gate,
        visible,
        query_ids,
    }
}

fn decoder(varmap: &VarMap) -> QueryDecoder {
    let vb = VarBuilder::from_varmap(varmap, DType::F32, &dev());
    QueryDecoder::new(H, RANK, vb.pp("dec.query")).unwrap()
}

#[test]
fn shapes_and_attention_rows_sum_to_one_over_the_visible_slots() {
    let varmap = VarMap::new();
    let d = decoder(&varmap);
    let (idx, g, vis, q) = (indices(), gate(), visible(), query_ids());
    let out = d.forward(&features(), &input(&idx, &g, &vis, &q)).unwrap();
    assert_eq!(out.residual.dims(), &[N, Q]);
    assert_eq!(out.attention.dims(), &[N, Q, V]);
    let a = out.attention.to_vec3::<f32>().unwrap();
    let vis = vis.to_vec2::<f32>().unwrap();
    for (n, row) in a.iter().enumerate() {
        for (qi, weights) in row.iter().enumerate() {
            let s: f32 = weights.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "row {n} query {qi}: attention sums to {s}"
            );
            for (v, &w) in weights.iter().enumerate() {
                if vis[n][v] == 0.0 {
                    assert!(w < 1e-6, "row {n} query {qi} read masked slot {v}: {w}");
                }
            }
        }
    }
}

#[test]
fn a_row_with_nothing_visible_reads_nothing() {
    let varmap = VarMap::new();
    let d = decoder(&varmap);
    let (idx, g, q) = (indices(), gate(), query_ids());
    let none = Tensor::zeros((N, V), DType::F32, &dev()).unwrap();
    let out = d.forward(&features(), &input(&idx, &g, &none, &q)).unwrap();
    let r = out.residual.to_vec2::<f32>().unwrap();
    assert!(
        r.iter().flatten().all(|&x| x == 0.0),
        "residual {r:?} must be exactly zero"
    );
}

#[test]
fn the_read_is_invariant_to_the_order_of_the_context_slots() {
    let varmap = VarMap::new();
    let d = decoder(&varmap);
    let (idx, g, vis, q) = (indices(), gate(), visible(), query_ids());
    let a = d.forward(&features(), &input(&idx, &g, &vis, &q)).unwrap();
    let perm = Tensor::from_vec((0..V as u32).rev().collect::<Vec<_>>(), V, &dev()).unwrap();
    let (idx_p, g_p, vis_p) = (
        idx.index_select(&perm, 1).unwrap(),
        g.index_select(&perm, 1).unwrap(),
        vis.index_select(&perm, 1).unwrap(),
    );
    let b = d
        .forward(&features(), &input(&idx_p, &g_p, &vis_p, &q))
        .unwrap();
    let ra = a.residual.to_vec2::<f32>().unwrap();
    let rb = b.residual.to_vec2::<f32>().unwrap();
    for n in 0..N {
        for qi in 0..Q {
            assert!(
                (ra[n][qi] - rb[n][qi]).abs() < 1e-5,
                "row {n} query {qi}: {} vs {}",
                ra[n][qi],
                rb[n][qi]
            );
        }
    }
}

#[test]
fn a_query_reads_the_visible_slots_of_its_own_row_only() {
    let varmap = VarMap::new();
    let d = decoder(&varmap);
    let (idx, vis, q) = (indices(), visible(), query_ids());
    let g = Var::from_tensor(&gate()).unwrap();
    let rho_var = Var::from_tensor(&rho()).unwrap();
    let out = d
        .forward(
            &crate::feature_embedding::FeatureEmbedding::fixed(rho_var.as_tensor().clone()),
            &input(&idx, g.as_tensor(), &vis, &q),
        )
        .unwrap();
    // Row 0, query 0's residual against every slot's gate and against ρ.
    let target = out.residual.get(0).unwrap().get(0).unwrap();
    let grads = target.backward().unwrap();
    let gg = grads
        .get(&g)
        .expect("the gate receives gradient")
        .to_vec2::<f32>()
        .unwrap();
    let visv = vis.to_vec2::<f32>().unwrap();
    for v in 0..V {
        assert_eq!(
            gg[1][v], 0.0,
            "row 1 slot {v} got gradient from a row-0 query"
        );
        if visv[0][v] == 0.0 {
            assert_eq!(gg[0][v], 0.0, "masked slot {v} of row 0 got gradient");
        } else {
            assert!(gg[0][v] != 0.0, "visible slot {v} of row 0 got no gradient");
        }
    }
    let gr = grads
        .get(&rho_var)
        .expect("ρ receives gradient")
        .to_vec2::<f32>()
        .unwrap();
    // Row 0's visible genes {3, 5, 1} and its query gene 0 move; gene 4 (row 1
    // only) and gene 2 (row 1 only) do not.
    for gene in [3usize, 5, 1, 0] {
        assert!(
            gr[gene].iter().any(|&x| x != 0.0),
            "gene {gene} got no gradient"
        );
    }
    for gene in [4usize, 2] {
        assert!(
            gr[gene].iter().all(|&x| x == 0.0),
            "gene {gene} got gradient from another row"
        );
    }
}

/// Projecting ρ once and gathering equals projecting the gathered, gated
/// tokens: `W (a·ρ_g) = a · (W ρ)_g`. Checked through the whole forward against
/// a token-level reference computed on the host.
#[test]
fn the_read_matches_a_host_reference_on_gated_tokens() {
    let h = 2usize;
    let eye = Tensor::eye(h, DType::F32, &dev()).unwrap();
    let mut ts = HashMap::new();
    ts.insert("dec.query.q.weight".to_string(), eye.clone());
    ts.insert("dec.query.k.weight".to_string(), eye.clone());
    ts.insert("dec.query.v.weight".to_string(), eye);
    ts.insert(
        "dec.query.mask".to_string(),
        Tensor::from_vec(vec![0.5f32, -0.25], (1, h), &dev()).unwrap(),
    );
    ts.insert(
        "dec.query.out.weight".to_string(),
        Tensor::from_vec(vec![1.0f32, 1.0], (1, h), &dev()).unwrap(),
    );
    ts.insert(
        "dec.query.out.bias".to_string(),
        Tensor::zeros(1, DType::F32, &dev()).unwrap(),
    );
    let vb = VarBuilder::from_tensors(ts, DType::F32, &dev());
    let d = QueryDecoder::new(h, h, vb.pp("dec.query")).unwrap();

    // Three genes; the context holds genes 2, 0, 1 with gates 0.5, 2.0, 1.0; slot 2 masked.
    let rho_h: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 2.0, -1.0];
    let rho = Tensor::from_vec(rho_h.clone(), (3, h), &dev()).unwrap();
    let idx = Tensor::from_vec(vec![2u32, 0, 1], (1, 3), &dev()).unwrap();
    let g = Tensor::from_vec(vec![0.5f32, 2.0, 1.0], (1, 3), &dev()).unwrap();
    let vis = Tensor::from_vec(vec![1.0f32, 1.0, 0.0], (1, 3), &dev()).unwrap();
    let q = Tensor::from_vec(vec![1u32], (1, 1), &dev()).unwrap();
    let out = d
        .forward(
            &crate::feature_embedding::FeatureEmbedding::fixed(rho.clone()),
            &input(&idx, &g, &vis, &q),
        )
        .unwrap();
    let got = out.residual.to_vec2::<f32>().unwrap()[0][0];

    // Host: tokens = gate · ρ_idx; query = ρ_1 + mask = [0.5, 0.75].
    let tok = |slot: usize| -> [f64; 2] {
        let (gene, gate) = ([2usize, 0, 1][slot], [0.5f64, 2.0, 1.0][slot]);
        [
            gate * f64::from(rho_h[gene * 2]),
            gate * f64::from(rho_h[gene * 2 + 1]),
        ]
    };
    let qv = [0.5f64, 0.75];
    let s: Vec<f64> = (0..2)
        .map(|v| (qv[0] * tok(v)[0] + qv[1] * tok(v)[1]) / (h as f64).sqrt())
        .collect();
    let m = s[0].max(s[1]);
    let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = w.iter().sum();
    let read: Vec<f64> = (0..2)
        .map(|d_| (0..2).map(|v| w[v] / z * tok(v)[d_]).sum())
        .collect();
    let want = (read[0] + read[1]) as f32;
    assert!((got - want).abs() < 1e-5, "residual {got} vs host {want}");
}
