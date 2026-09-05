//! Tests for the dense masked-imputation heads: the `[N, D]` mixture rate
//! sums to one, the dense NB / multinomial heads match an element-wise
//! reference, and they agree with the indexed heads at the same positions.

use super::*;
use candle_core::{DType, Device};
use std::collections::HashMap;

const D: usize = 6;
const H: usize = 3;
const K: usize = 4;
const N: usize = 3;

fn dev() -> Device {
    Device::Cpu
}

fn rho() -> Tensor {
    let rows: Vec<f32> = (0..D * H)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
        .collect();
    Tensor::from_vec(rows, (D, H), &dev()).unwrap()
}

fn alpha() -> Tensor {
    let rows: Vec<f32> = (0..K * H)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.25)
        .collect();
    Tensor::from_vec(rows, (K, H), &dev()).unwrap()
}

/// A decoder with deterministic `α`, `φ` and a non-uniform pinned background.
fn decoder() -> EmbeddedNbTopicDecoder {
    let log_phi: Vec<f32> = (0..D).map(|g| 0.2 * g as f32 - 0.5).collect();
    let log_pi: Vec<f32> = (0..D).map(|g| -((g + 1) as f32).ln()).collect();
    let mut ts = HashMap::new();
    ts.insert("dec.topic.embeddings".to_string(), alpha());
    ts.insert(
        "dec.log_phi".to_string(),
        Tensor::from_vec(log_phi, (1, D), &dev()).unwrap(),
    );
    ts.insert(
        "dec.log_pi".to_string(),
        Tensor::from_vec(log_pi, (1, D), &dev()).unwrap(),
    );
    let vb = VarBuilder::from_tensors(ts, DType::F32, &dev());
    EmbeddedNbTopicDecoder::new(K, rho(), vb.pp("dec")).unwrap()
}

fn log_theta() -> Tensor {
    let logits: Vec<f32> = (0..N * K).map(|i| ((i * 3 % 7) as f32) * 0.6).collect();
    let t = Tensor::from_vec(logits, (N, K), &dev()).unwrap();
    ops::log_softmax(&t, 1).unwrap()
}

/// Counts with zeros in every row.
fn values() -> Tensor {
    #[rustfmt::skip]
    let v: Vec<f32> = vec![
        3.0, 0.0, 1.0, 0.0, 5.0, 0.0,
        0.0, 2.0, 0.0, 0.0, 0.0, 7.0,
        1.0, 1.0, 0.0, 4.0, 0.0, 0.0,
    ];
    Tensor::from_vec(v, (N, D), &dev()).unwrap()
}

/// 1 = scored. Scores zero-count genes and leaves some nonzero ones out.
fn mask() -> Tensor {
    #[rustfmt::skip]
    let m: Vec<f32> = vec![
        0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
    ];
    Tensor::from_vec(m, (N, D), &dev()).unwrap()
}

fn residual() -> Tensor {
    let r: Vec<f32> = (0..N * D)
        .map(|i| 0.5 + ((i * 5 % 9) as f32) * 0.1)
        .collect();
    Tensor::from_vec(r, (N, D), &dev()).unwrap()
}

fn lib() -> Tensor {
    Tensor::from_vec(vec![9.0f32, 10.0, 7.0], (N, 1), &dev()).unwrap()
}

fn to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_vec2().unwrap()
}

fn to_vec1(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn dense_mixture_rate_rows_sum_to_one() {
    let dec = decoder();
    let full_kd = dec.full_logits_kd().unwrap();
    let p = dec.mixture_rate_nd(&log_theta(), &full_kd).unwrap();
    assert_eq!(p.dims(), &[N, D]);
    for (n, row) in to_vec2(&p).iter().enumerate() {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "row {n} sums to {s}");
    }
}

/// The dense heads compose `μ = residual · ℓ · p` and the mask exactly as an
/// element-by-element reference does. The reference calls the same NB element
/// function on `[1, 1]` tensors, so this pins the composition, not the lgamma.
#[test]
fn dense_heads_match_an_elementwise_reference() {
    let dec = decoder();
    let full_kd = dec.full_logits_kd().unwrap();
    let p = to_vec2(&dec.mixture_rate_nd(&log_theta(), &full_kd).unwrap());
    let (y, m, r) = (to_vec2(&values()), to_vec2(&mask()), to_vec2(&residual()));
    let l = to_vec1(&lib());
    let log_phi = to_vec1(dec.log_phi());

    let (values, residual, lib, mask) = (values(), residual(), lib(), mask());
    let target = MaskedDenseTarget {
        values: &values,
        residual: Some(&residual),
        lib: &lib,
        mask: &mask,
        log_residual: None,
    };
    let nb = to_vec1(
        &dec.impute_dense_nb(&log_theta(), &target, &full_kd)
            .unwrap(),
    );
    let mn = to_vec1(
        &dec.impute_dense_multinomial(&log_theta(), &target, &full_kd)
            .unwrap(),
    );

    let one = |v: f32| Tensor::new(&[[v]], &dev()).unwrap();
    for n in 0..N {
        let mut nb_ref = 0f32;
        let mut mn_ref = 0f32;
        for g in 0..D {
            if m[n][g] == 0.0 {
                continue;
            }
            let mu = r[n][g] * l[n] * p[n][g];
            let elem = nb_log_likelihood_elem(&one(y[n][g]), &one(mu), &one(log_phi[g])).unwrap();
            nb_ref += to_vec1(&elem)[0];
            mn_ref += y[n][g] * (p[n][g] + 1e-20).ln();
        }
        assert!(
            (nb[n] - nb_ref).abs() < 1e-3,
            "row {n}: dense NB {} vs reference {nb_ref}",
            nb[n]
        );
        assert!(
            (mn[n] - mn_ref).abs() < 1e-4,
            "row {n}: dense multinomial {} vs reference {mn_ref}",
            mn[n]
        );
    }
}

/// Handing the indexed heads every gene of a row, in any order, must give the
/// dense heads' number: the two layouts score one likelihood.
#[test]
fn dense_and_indexed_heads_agree_on_the_same_positions() {
    let dec = decoder();
    let full_kd = dec.full_logits_kd().unwrap();
    let perms: Vec<Vec<u32>> = vec![
        vec![5, 0, 3, 1, 4, 2],
        vec![2, 4, 0, 5, 1, 3],
        vec![0, 1, 2, 3, 4, 5],
    ];
    let gather = |t: &Tensor| -> Tensor {
        let v = to_vec2(t);
        let mut g = Vec::with_capacity(N * D);
        for (n, p) in perms.iter().enumerate() {
            for &j in p {
                g.push(v[n][j as usize]);
            }
        }
        Tensor::from_vec(g, (N, D), &dev()).unwrap()
    };
    let indices = Tensor::from_vec(perms.concat(), (N, D), &dev()).unwrap();
    let (values, residual, lib, mask) = (values(), residual(), lib(), mask());
    let (vals_k, mask_k, res_k) = (gather(&values), gather(&mask), gather(&residual));

    let indexed = MaskedNbTarget {
        indices: &indices,
        residual: Some(&res_k),
        values: &vals_k,
        lib: &lib,
        mask: &mask_k,
    };
    let dense = MaskedDenseTarget {
        values: &values,
        residual: Some(&residual),
        lib: &lib,
        mask: &mask,
        log_residual: None,
    };

    let nb_i = to_vec1(
        &dec.impute_masked_nb(&log_theta(), &indexed, &full_kd)
            .unwrap(),
    );
    let nb_d = to_vec1(&dec.impute_dense_nb(&log_theta(), &dense, &full_kd).unwrap());
    let mn_i = to_vec1(
        &dec.impute_masked_multinomial(&log_theta(), &indexed, &full_kd)
            .unwrap(),
    );
    let mn_d = to_vec1(
        &dec.impute_dense_multinomial(&log_theta(), &dense, &full_kd)
            .unwrap(),
    );
    for n in 0..N {
        assert!(
            (nb_i[n] - nb_d[n]).abs() < 1e-3,
            "row {n}: indexed NB {} vs dense {}",
            nb_i[n],
            nb_d[n]
        );
        assert!(
            (mn_i[n] - mn_d[n]).abs() < 1e-4,
            "row {n}: indexed multinomial {} vs dense {}",
            mn_i[n],
            mn_d[n]
        );
    }
}

//////////////////////////////
// Query-decoder residual    //
//////////////////////////////

/// `log_residual` multiplies the rate: a uniform residual `c` on the NB head is
/// the same as scaling the library by `exp(c)`, and a residual set on one
/// gene changes the likelihood only where that gene is scored.
#[test]
fn a_log_residual_scales_the_nb_rate_and_only_where_it_is_set() {
    let dec = decoder();
    let full_kd = dec.full_logits_kd().unwrap();
    let (values, mask, lib) = (values(), mask(), lib());
    let c = 0.7f64;
    let uniform = Tensor::full(c as f32, (N, D), &dev()).unwrap();
    let with_res = dec
        .impute_dense_nb(
            &log_theta(),
            &MaskedDenseTarget {
                values: &values,
                residual: None,
                lib: &lib,
                mask: &mask,
                log_residual: Some(&uniform),
            },
            &full_kd,
        )
        .unwrap();
    let scaled_lib = lib.affine(c.exp(), 0.0).unwrap();
    let with_lib = dec
        .impute_dense_nb(
            &log_theta(),
            &MaskedDenseTarget {
                values: &values,
                residual: None,
                lib: &scaled_lib,
                mask: &mask,
                log_residual: None,
            },
            &full_kd,
        )
        .unwrap();
    for (a, b) in to_vec1(&with_res).iter().zip(to_vec1(&with_lib)) {
        assert!(
            (a - b).abs() < 1e-4,
            "uniform residual {a} vs scaled library {b}"
        );
    }

    // Residual only on gene 2. Row 0 scores gene 2, so its llik moves; with a
    // mask that leaves gene 2 out, nothing moves.
    let mut one = vec![0f32; N * D];
    one[2] = 1.5;
    one[D + 2] = -0.8;
    one[2 * D + 2] = 0.3;
    let one = Tensor::from_vec(one, (N, D), &dev()).unwrap();
    let base = dec
        .impute_dense_nb(
            &log_theta(),
            &MaskedDenseTarget {
                values: &values,
                residual: None,
                lib: &lib,
                mask: &mask,
                log_residual: None,
            },
            &full_kd,
        )
        .unwrap();
    let moved = dec
        .impute_dense_nb(
            &log_theta(),
            &MaskedDenseTarget {
                values: &values,
                residual: None,
                lib: &lib,
                mask: &mask,
                log_residual: Some(&one),
            },
            &full_kd,
        )
        .unwrap();
    for (n, (a, b)) in to_vec1(&base).iter().zip(to_vec1(&moved)).enumerate() {
        assert!(
            (a - b).abs() > 1e-4,
            "row {n} scores gene 2 but did not move: {a} vs {b}"
        );
    }
    let mut m = to_vec2(&mask);
    for row in &mut m {
        row[2] = 0.0;
    }
    let mask_no2 = Tensor::from_vec(m.concat(), (N, D), &dev()).unwrap();
    let base2 = dec
        .impute_dense_nb(
            &log_theta(),
            &MaskedDenseTarget {
                values: &values,
                residual: None,
                lib: &lib,
                mask: &mask_no2,
                log_residual: None,
            },
            &full_kd,
        )
        .unwrap();
    let same = dec
        .impute_dense_nb(
            &log_theta(),
            &MaskedDenseTarget {
                values: &values,
                residual: None,
                lib: &lib,
                mask: &mask_no2,
                log_residual: Some(&one),
            },
            &full_kd,
        )
        .unwrap();
    for (a, b) in to_vec1(&base2).iter().zip(to_vec1(&same)) {
        assert!(
            (a - b).abs() < 1e-5,
            "gene 2 unscored but the llik moved: {a} vs {b}"
        );
    }
}

/// The multinomial rate is renormalized after the residual, so a uniform
/// residual leaves it unchanged and a one-gene residual moves the other
/// genes' probabilities down.
#[test]
fn the_multinomial_renormalizes_after_the_residual() {
    let dec = decoder();
    let full_kd = dec.full_logits_kd().unwrap();
    let (values, mask, lib) = (values(), mask(), lib());
    let uniform = Tensor::full(0.9f32, (N, D), &dev()).unwrap();
    let base = dec
        .impute_dense_multinomial(
            &log_theta(),
            &MaskedDenseTarget {
                values: &values,
                residual: None,
                lib: &lib,
                mask: &mask,
                log_residual: None,
            },
            &full_kd,
        )
        .unwrap();
    let same = dec
        .impute_dense_multinomial(
            &log_theta(),
            &MaskedDenseTarget {
                values: &values,
                residual: None,
                lib: &lib,
                mask: &mask,
                log_residual: Some(&uniform),
            },
            &full_kd,
        )
        .unwrap();
    for (a, b) in to_vec1(&base).iter().zip(to_vec1(&same)) {
        assert!(
            (a - b).abs() < 1e-4,
            "uniform residual changed the multinomial: {a} vs {b}"
        );
    }
    // Raise gene 4 in every row; row 0 has count 5 there and gene 4 is unscored
    // in row 0, so row 0's scored genes lose mass and its llik must fall.
    let mut one = vec![0f32; N * D];
    for n in 0..N {
        one[n * D + 4] = 2.0;
    }
    let one = Tensor::from_vec(one, (N, D), &dev()).unwrap();
    let moved = dec
        .impute_dense_multinomial(
            &log_theta(),
            &MaskedDenseTarget {
                values: &values,
                residual: None,
                lib: &lib,
                mask: &mask,
                log_residual: Some(&one),
            },
            &full_kd,
        )
        .unwrap();
    let (b, m) = (to_vec1(&base), to_vec1(&moved));
    assert!(
        m[0] < b[0] - 1e-4,
        "row 0: unscored gene 4 up must pull scored mass down ({} vs {})",
        m[0],
        b[0]
    );
}
