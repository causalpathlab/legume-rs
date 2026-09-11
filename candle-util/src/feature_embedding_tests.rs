//! The feature side as a mixture of shared modules.
//!
//! Every property here is one the rest of the design leans on: growth assumes a
//! uniform row lands at the dictionary centroid, recycling assumes a dead module
//! is exactly zero mass and receives no gradient, and the module-space encoder
//! assumes composing then gathering equals gathering then composing.

use super::FeatureEmbedding;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn build(d: usize, m: usize, h: usize) -> (VarMap, FeatureEmbedding) {
    let dev = Device::Cpu;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let fe = FeatureEmbedding::new(d, m, h, vb).expect("feature embedding");
    (vm, fe)
}

fn set(vm: &VarMap, name: &str, v: Vec<f32>, shape: (usize, usize)) {
    let dev = Device::Cpu;
    let data = vm.data().lock().unwrap();
    data[name]
        .set(&Tensor::from_vec(v, shape, &dev).unwrap())
        .unwrap();
}

/// The initialization every new feature gets, and the one growth relies on: a
/// flat row is uniform over the whole support, so the composed row is the
/// centroid of the dictionary and every module can still earn the feature.
#[test]
fn a_flat_row_is_uniform_over_the_full_support() {
    let (vm, fe) = build(2, 4, 3);
    set(&vm, "modules.logits", vec![0.0; 8], (2, 4));
    let pi = fe.membership().unwrap().unwrap().to_vec2::<f32>().unwrap();
    for row in &pi {
        for v in row {
            assert!((v - 0.25).abs() < 1e-6, "expected uniform, got {row:?}");
        }
    }
}

/// A module no feature puts mass on is exactly zero, not merely small. Recycling
/// keys on this, and it is what a softmax could not give.
#[test]
fn an_unused_module_is_exactly_zero() {
    let (vm, fe) = build(1, 3, 2);
    set(&vm, "modules.logits", vec![5.0, 0.0, 0.0], (1, 3));
    let pi = fe.membership().unwrap().unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(pi[0][1], 0.0, "row {:?}", pi[0]);
    assert_eq!(pi[0][2], 0.0, "row {:?}", pi[0]);
    assert!((pi[0][0] - 1.0).abs() < 1e-6);
}

/// Gathering the rows a minibatch touches must equal composing the whole table
/// and then selecting from it. The module-space encoder is only valid because
/// these two agree.
#[test]
fn gathering_agrees_with_composing_the_whole_table() {
    let (_vm, fe) = build(5, 3, 4);
    let ids = Tensor::from_vec(vec![3u32, 0, 3], 3, &Device::Cpu).unwrap();
    let gathered = fe.gather(&ids).unwrap().to_vec2::<f32>().unwrap();
    let full = fe.full().unwrap().to_vec2::<f32>().unwrap();
    for (row, &g) in gathered.iter().zip([3usize, 0, 3].iter()) {
        for (a, b) in row.iter().zip(&full[g]) {
            assert!((a - b).abs() < 1e-6, "{row:?} vs {:?}", full[g]);
        }
    }
}

/// A feature whose mass lands on one module reproduces that module's vector, so
/// a one-hot membership is a faithful limit rather than an approximation.
#[test]
fn a_feature_on_one_module_reproduces_its_vector() {
    let (vm, fe) = build(1, 2, 3);
    set(&vm, "modules.logits", vec![9.0, 0.0], (1, 2));
    set(
        &vm,
        "modules.mu",
        vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        (2, 3),
    );
    let row = fe.full().unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(row[0], vec![1.0, 2.0, 3.0]);
}

/// Both parameters have to receive gradient, or the feature side cannot learn.
/// The second half is the dead-module hazard: a module outside every support
/// gets nothing, which is why capacity is added by splitting and never by
/// appending.
#[test]
fn gradient_reaches_both_parameters_but_not_an_unused_module() {
    let (vm, fe) = build(2, 3, 2);
    // Feature 0 sits entirely on module 0, feature 1 entirely on module 1;
    // module 2 is outside both supports.
    set(
        &vm,
        "modules.logits",
        vec![9.0, 0.0, 0.0, 0.0, 9.0, 0.0],
        (2, 3),
    );
    let loss = fe.full().unwrap().sqr().unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();

    let data = vm.data().lock().unwrap();
    let mu_g = grads.get(&data["modules.mu"]).expect("mu gets gradient");
    let rows: Vec<Vec<f32>> = mu_g.to_vec2().unwrap();
    assert!(rows[0].iter().any(|v| *v != 0.0), "module 0 must train");
    assert!(rows[1].iter().any(|v| *v != 0.0), "module 1 must train");
    assert!(
        rows[2].iter().all(|v| *v == 0.0),
        "a module outside every support receives nothing: {:?}",
        rows[2]
    );
    assert!(
        grads.get(&data["modules.logits"]).is_some(),
        "membership must train"
    );
}

/// One module means one shared row for every feature. Degenerate, but it should
/// be exactly that rather than a shape error or a silent NaN.
#[test]
fn a_single_module_gives_every_feature_the_same_row() {
    let (vm, fe) = build(3, 1, 2);
    set(&vm, "modules.mu", vec![0.5, -0.5], (1, 2));
    let rows = fe.full().unwrap().to_vec2::<f32>().unwrap();
    for row in &rows {
        // Approximately, not exactly: sparsemax of one coordinate reaches 1.0
        // through a sort and a cumulative sum, so the composed row carries a
        // rounding of that arithmetic.
        for (got, want) in row.iter().zip([0.5f32, -0.5]) {
            assert!((got - want).abs() < 1e-6, "{row:?}");
        }
    }
}

/// A flat membership gives every feature the same composed row. That is the
/// right state for a feature the axis gains later, since the model around it
/// is trained — but it is a degenerate place to START a fit from, so the
/// initialization has to separate features.
#[test]
fn initialization_separates_the_features() {
    let (_vm, fe) = build(8, 4, 3);
    let rows = fe.full().unwrap().to_vec2::<f32>().unwrap();
    assert!(
        rows.iter().any(|r| r != &rows[0]),
        "every feature composed the same row: {:?}",
        &rows[..2]
    );
    // Still near-uniform, though: the separation is a symmetry break, not a
    // prior about which features belong together.
    let pi = fe.membership().unwrap().unwrap().to_vec2::<f32>().unwrap();
    for row in &pi {
        assert!(
            row.iter().all(|v| *v > 0.0),
            "initialization must leave every module able to earn a feature: {row:?}"
        );
    }
}
