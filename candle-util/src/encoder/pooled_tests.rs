//! The shared pooled-gene trunk: the same var names the masked encoder has
//! always registered, an optional visible mask that defaults to every gene,
//! and a frozen table that stays put while the trunk trains.

use super::*;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};

const D: usize = 9;
const H: usize = 4;
const N: usize = 3;
const OUT: usize = 5;

fn dev() -> Device {
    Device::Cpu
}

fn ramp(n: usize, a: f32, b: f32) -> Vec<f32> {
    (0..n)
        .map(|i| a + b * ((i as f32 * 0.7).sin() + 0.3 * (i as f32 * 0.11).cos()))
        .collect()
}

fn table() -> Tensor {
    Tensor::from_vec(ramp(D * H, 0.0, 1.0), (D, H), &dev()).unwrap()
}

fn counts() -> Tensor {
    Tensor::from_vec(ramp(N * D, 5.0, 4.0), (N, D), &dev()).unwrap()
}

fn build(prefix: &str) -> (PooledGeneEncoder, VarMap) {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev());
    let enc = PooledGeneEncoder::new(
        FeatureEmbedding::fixed(table()),
        PooledGeneEncoderArgs {
            layers: &[6],
            out_dim: OUT,
            attn_pool: true,
            in_dim_extra: 0,
        },
        &varmap,
        vb.pp(prefix),
    )
    .unwrap();
    (enc, varmap)
}

/// The contract with every masked checkpoint on disk: the trunk's parameters
/// keep the names the masked encoder registered before it delegated here.
#[test]
fn the_trunk_registers_the_masked_encoders_var_names() {
    let (_, varmap) = build("enc");
    let mut names: Vec<String> = varmap.data().lock().unwrap().keys().cloned().collect();
    names.sort();
    let want = [
        "enc.attn.query",
        "enc.nn.enc.bn_z.bias",
        "enc.nn.enc.bn_z.running_mean",
        "enc.nn.enc.bn_z.running_var",
        "enc.nn.enc.bn_z.weight",
        "enc.nn.enc.fc.relu_linear_stack.0.bias",
        "enc.nn.enc.fc.relu_linear_stack.0.weight",
        "enc.nn.enc.z.mean.bias",
        "enc.nn.enc.z.mean.weight",
    ];
    assert_eq!(names, want.map(String::from).to_vec());
}

/// `visible: None` is every gene visible.
#[test]
fn no_visible_mask_means_every_gene_is_visible() {
    let (enc, _) = build("enc");
    let x = counts();
    let ones = Tensor::ones((N, D), DType::F32, &dev()).unwrap();
    let a = enc.pool(&x, None, None, None).unwrap();
    let b = enc.pool(&x, None, None, Some(&ones)).unwrap();
    assert_eq!(a.dims(), &[N, H]);
    let (a, b): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (a.to_vec2().unwrap(), b.to_vec2().unwrap());
    for n in 0..N {
        for h in 0..H {
            assert!(
                (a[n][h] - b[n][h]).abs() < 1e-6,
                "[{n}, {h}]: {} vs {}",
                a[n][h],
                b[n][h]
            );
        }
    }
    // And a hidden gene changes the pool: the mask is read.
    let mut m = vec![1f32; N * D];
    m[0] = 0.0;
    let masked = Tensor::from_vec(m, (N, D), &dev()).unwrap();
    let c: Vec<Vec<f32>> = enc
        .pool(&x, None, None, Some(&masked))
        .unwrap()
        .to_vec2()
        .unwrap();
    assert!(
        (0..H).any(|h| (c[0][h] - a[0][h]).abs() > 1e-6),
        "row 0 ignored its mask"
    );
    for h in 0..H {
        assert!((c[1][h] - a[1][h]).abs() < 1e-6, "row 1 must be untouched");
    }
}

/// The forward is raw `[N, out]`: no clamp, no simplex, finite.
#[test]
fn the_forward_is_a_raw_projection() {
    let (enc, _) = build("enc");
    let z = enc.forward(&counts(), None, None, None, false).unwrap();
    assert_eq!(z.dims(), &[N, OUT]);
    assert!(z
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

/// One optimizer step moves the query, the FC and the head, and a fixed table
/// is not a parameter at all — the shape a distillation onto a frozen
/// dictionary needs.
#[test]
fn a_step_trains_the_trunk_and_never_the_fixed_table() {
    let (enc, varmap) = build("enc");
    let table_before: Vec<f32> = enc
        .features()
        .full()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let vars = varmap.all_vars();
    assert!(
        vars.iter().all(|v| v.dims() != [D, H]),
        "the fixed table registered a var"
    );
    let before: Vec<Vec<f32>> = vars
        .iter()
        .map(|v| v.flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    let mut adam = candle_nn::AdamW::new(
        vars.clone(),
        candle_nn::ParamsAdamW {
            lr: 1e-2,
            ..Default::default()
        },
    )
    .unwrap();
    let loss = enc
        .forward(&counts(), None, None, None, true)
        .unwrap()
        .sqr()
        .unwrap()
        .mean_all()
        .unwrap();
    adam.backward_step(&loss).unwrap();
    let names: Vec<String> = varmap.data().lock().unwrap().keys().cloned().collect();
    let mut moved = 0usize;
    for (v, b) in vars.iter().zip(&before) {
        let after: Vec<f32> = v.flatten_all().unwrap().to_vec1().unwrap();
        if after != *b {
            moved += 1;
        }
    }
    // query, fc weight+bias, head weight+bias, bn affine: everything with a
    // gradient; the BN running stats move in train mode too.
    assert!(
        moved >= 5,
        "only {moved} of {} vars moved ({names:?})",
        vars.len()
    );
    let table_after: Vec<f32> = enc
        .features()
        .full()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(table_before, table_after);
}
