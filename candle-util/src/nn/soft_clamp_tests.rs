use super::*;
use candle_core::{DType, Device, Tensor, Var};

/// The whole point: a value driven past the bound must still carry gradient
/// back, where a hard `clamp` returns exactly zero and the unit is frozen for
/// the rest of training.
#[test]
fn gradient_survives_beyond_the_bound_where_a_hard_clamp_is_zero() {
    let dev = Device::Cpu;
    let c = MASKED_LOGIT_CLAMP;

    for &x in &[9.0f32, 12.0, 20.0] {
        let v = Var::from_tensor(&Tensor::new(&[x], &dev).unwrap()).unwrap();

        let soft = soft_clamp(v.as_tensor(), c).unwrap().sum_all().unwrap();
        let g_soft = soft.backward().unwrap();
        let g_soft = g_soft.get(&v).unwrap().to_vec1::<f32>().unwrap()[0];

        let hard = v.as_tensor().clamp(-c, c).unwrap().sum_all().unwrap();
        let g_hard = hard.backward().unwrap();
        let g_hard = g_hard.get(&v).unwrap().to_vec1::<f32>().unwrap()[0];

        assert_eq!(g_hard, 0.0, "hard clamp is expected to be dead at x={x}");
        assert!(
            g_soft > 0.0,
            "soft clamp must stay trainable at x={x}, got gradient {g_soft}"
        );
    }
}

/// It must still be a bound — the output never leaves `(−c, c)`, which is what
/// the downstream `exp` / softmax relies on.
#[test]
fn output_stays_inside_the_bound() {
    let dev = Device::Cpu;
    let c = MASKED_LOGIT_CLAMP;
    let x = Tensor::new(&[-1e4f32, -50.0, -8.0, 0.0, 8.0, 50.0, 1e4], &dev).unwrap();
    let y: Vec<f32> = soft_clamp(&x, c)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    for v in y {
        assert!(v.abs() <= c as f32, "{v} escaped the bound {c}");
    }
}

/// Near zero it must be close to the identity, so ordinary logits are not
/// distorted by the guard.
#[test]
fn near_identity_well_inside_the_bound() {
    let dev = Device::Cpu;
    let x = Tensor::new(&[-1.0f32, -0.25, 0.0, 0.25, 1.0], &dev).unwrap();
    let y: Vec<f32> = soft_clamp(&x, MASKED_LOGIT_CLAMP).unwrap().to_vec1().unwrap();
    for (a, b) in [-1.0f32, -0.25, 0.0, 0.25, 1.0].iter().zip(y.iter()) {
        assert!((a - b).abs() < 0.01, "{a} distorted to {b}");
    }
}
