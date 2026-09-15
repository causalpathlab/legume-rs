//! The pair head is symmetric in its two codes, degenerates to one linear map
//! at a single expert, and gates with a distribution over experts.

use super::*;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn head(
    code_dim: usize,
    out_dim: usize,
    n_experts: usize,
    extra_dim: usize,
) -> (SymmetricPairHead, VarMap) {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let head = SymmetricPairHead::new(
        SymmetricPairHeadArgs {
            code_dim,
            out_dim,
            n_experts,
            extra_dim,
        },
        vb,
    )
    .unwrap();
    (head, varmap)
}

fn codes(rows: usize, cols: usize, seed: f32) -> Tensor {
    let v: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.8)
        .collect();
    Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap()
}

#[test]
fn the_pair_code_is_symmetric_in_its_endpoints() {
    let (head, _) = head(6, 4, 3, 2);
    let a = codes(5, 6, 0.1);
    let b = codes(5, 6, 2.3);
    let x = codes(5, 2, 4.4);
    let flat = |t: Tensor| -> Vec<f32> { t.flatten_all().unwrap().to_vec1().unwrap() };
    let ab = flat(head.forward(&a, &b, Some(&x)).unwrap());
    let ba = flat(head.forward(&b, &a, Some(&x)).unwrap());
    assert_eq!(ab, ba);
    // The extra features are the caller's contract, in either direction.
    assert!(head.forward(&a, &b, None).is_err());
    let (bare, _) = crate::encoder::pair_head::tests::head(6, 4, 3, 0);
    assert!(bare.forward(&a, &b, Some(&x)).is_err());
}

#[test]
fn one_expert_is_a_plain_linear_head() {
    let (head, varmap) = head(3, 2, 1, 0);
    let a = codes(4, 3, 0.5);
    let b = codes(4, 3, 1.5);
    let z: Vec<Vec<f32>> = head.forward(&a, &b, None).unwrap().to_vec2().unwrap();

    // By hand: f = [(a + b)/2 ‖ a ⊙ b], z = f · Wᵀ + c.
    let vars = varmap.data().lock().unwrap();
    let w: Vec<Vec<f32>> = vars["experts.weight"].as_tensor().to_vec2().unwrap();
    let c: Vec<f32> = vars["experts.bias"].as_tensor().to_vec1().unwrap();
    assert!(
        !vars.contains_key("gate.weight"),
        "a single expert needs no gate"
    );
    let a: Vec<Vec<f32>> = a.to_vec2().unwrap();
    let b: Vec<Vec<f32>> = b.to_vec2().unwrap();
    for r in 0..4 {
        let f: Vec<f32> = (0..3)
            .map(|j| (a[r][j] + b[r][j]) / 2.0)
            .chain((0..3).map(|j| a[r][j] * b[r][j]))
            .collect();
        for o in 0..2 {
            let want: f32 = f.iter().zip(&w[o]).map(|(x, y)| x * y).sum::<f32>() + c[o];
            assert!(
                (z[r][o] - want).abs() < 1e-5,
                "row {r} out {o}: {} vs {want}",
                z[r][o]
            );
        }
    }
}

#[test]
fn the_gate_is_a_distribution_over_experts_and_the_output_has_the_pair_shape() {
    let (head, _) = head(5, 3, 4, 1);
    let a = codes(7, 5, 0.2);
    let b = codes(7, 5, 0.9);
    let x = codes(7, 1, 3.3);
    let f = head.features(&a, &b, Some(&x)).unwrap();
    assert_eq!(f.dims(), &[7, 11]);
    let pi: Vec<Vec<f32>> = head.gate(&f).unwrap().to_vec2().unwrap();
    for row in &pi {
        assert_eq!(row.len(), 4);
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(row.iter().all(|&p| p >= 0.0));
    }
    assert_eq!(head.forward(&a, &b, Some(&x)).unwrap().dims(), &[7, 3]);
    assert_eq!(head.n_experts(), 4);
    assert_eq!(head.out_dim(), 3);
    assert_eq!(head.code_dim(), 5);
    assert_eq!(head.extra_dim(), 1);
}
