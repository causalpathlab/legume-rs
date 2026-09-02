//! The gradient-flow contract of the module parameterization, checked against the
//! real loss functions rather than described: which Var each term reaches, and
//! the analytic gradient of the exact term.

use candle_util::candle_core::{DType, Device, Tensor, Var};
use candle_util::candle_nn::VarMap;
use candle_util::nn::sparsemax;
use graph_embedding_util::loss::{
    gather_feature_rows, module_balance_prior, module_softmax_loss, nce_loss_identity, EdgeBatch,
    NceObjective,
};
use graph_embedding_util::model::{
    FeatureGateSpec, JointEmbedModel, ModuleInit, ModuleWarmStart, MODULE_BIAS_VAR_NAME,
    MODULE_LOGITS_VAR_NAME, MODULE_MU_VAR_NAME, MODULE_RESIDUAL_VAR_NAME,
};

const D: usize = 8;
const M: usize = 3;
const H: usize = 4;
const N: usize = 5;

fn dev() -> Device {
    Device::Cpu
}

/// A module model with a hard (one-hot) warm start unless `own_mass < 1`.
fn model(own_mass: f32, vm: &VarMap) -> JointEmbedModel {
    let labels: Vec<u32> = (0..D as u32).map(|g| g % M as u32).collect();
    JointEmbedModel::new_with_modules(
        ModuleInit {
            n_features: D,
            n_cells: N,
            embedding_dim: H,
            n_modules: M,
            warm: ModuleWarmStart::Labels {
                labels: &labels,
                own_mass,
            },
            b_feat: &[0f32; D],
            b_cell: &[0f32; N],
            seed: 1,
        },
        vm,
        &dev(),
    )
    .unwrap()
}

fn var(vm: &VarMap, name: &str) -> Var {
    vm.data().lock().unwrap().get(name).unwrap().clone()
}

fn grad_norm(grads: &candle_util::candle_core::backprop::GradStore, v: &Var) -> Option<f32> {
    grads.get(v.as_tensor()).map(|g| {
        g.sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt()
    })
}

/// Within-module NCE, hard membership: negatives share the positive's module, so
/// μ cancels and receives no gradient; r, a_g, e_c, b_c do; the logits do not
/// (a one-hot sparsemax row has a zero Jacobian).
#[test]
fn within_module_nce_reaches_residual_and_cells_but_not_mu() {
    let vm = VarMap::new();
    let m = model(1.0, &vm);
    // Feature 0 (module 0) positive; negatives 3 and 6 are the other module-0 members.
    let batch = EdgeBatch {
        coarse_cells: vec![2],
        fine_feats: vec![0],
        neg_feats: vec![3, 6],
        n_negatives: 2,
    };
    // Perturb the residual so rows differ (zero-init would make the loss flat).
    let r = var(&vm, MODULE_RESIDUAL_VAR_NAME);
    r.set(&Tensor::rand(-0.5f32, 0.5, (D, H), &dev()).unwrap())
        .unwrap();
    let loss = nce_loss_identity(&m, batch, NceObjective::Softmax, &dev()).unwrap();
    let grads = loss.backward().unwrap();
    let mu = grad_norm(&grads, &var(&vm, MODULE_MU_VAR_NAME)).unwrap_or(0.0);
    assert!(
        mu < 1e-5,
        "μ must cancel within a hard module, got ‖∇μ‖ = {mu}"
    );
    assert!(grad_norm(&grads, &var(&vm, MODULE_RESIDUAL_VAR_NAME)).unwrap() > 1e-4);
    assert!(grad_norm(&grads, &var(&vm, "b_feat")).unwrap() > 1e-4);
    assert!(grad_norm(&grads, &var(&vm, "e_cell")).unwrap() > 1e-4);
    // b_cell is shared by the positive and its negatives, so the softmax NCE is
    // invariant to it: its gradient is exactly zero even though it is in the graph.
    assert!(grad_norm(&grads, &var(&vm, "b_cell")).unwrap_or(0.0) < 1e-6);
    // One-hot rows have a zero sparsemax Jacobian EXCEPT at an exact support tie
    // (the warm start puts the other logits at 0 = τ), where candle's clamp
    // backward passes a boundary gradient. Small, and gone once anything moves.
    let lg = grad_norm(&grads, &var(&vm, MODULE_LOGITS_VAR_NAME)).unwrap_or(0.0);
    assert!(
        lg < 1e-2,
        "hard rows: only a boundary-tie gradient is expected, got {lg}"
    );
}

/// Mixed membership with negatives whose membership rows EQUAL the positive's
/// still cancels μ exactly: the module vector drops out of every score difference.
#[test]
fn identical_membership_rows_cancel_mu_even_when_mixed() {
    let vm = VarMap::new();
    let m = model(0.7, &vm);
    // 0, 3, 6 all carry the same warm-start row (0.7, 0.15, 0.15).
    let batch = EdgeBatch {
        coarse_cells: vec![2],
        fine_feats: vec![0],
        neg_feats: vec![3, 6],
        n_negatives: 2,
    };
    let loss = nce_loss_identity(&m, batch, NceObjective::Softmax, &dev()).unwrap();
    let grads = loss.backward().unwrap();
    let mu = grad_norm(&grads, &var(&vm, MODULE_MU_VAR_NAME)).unwrap_or(0.0);
    assert!(mu < 1e-5, "identical rows must cancel μ, got {mu}");
}

/// Mixed membership with negatives whose rows DIFFER from the positive's (module
/// 0 also holds genes whose main module is 1 or 2, at weight 0.15): the NCE now
/// reaches μ and the logits.
#[test]
fn within_module_nce_reaches_mu_and_logits_under_mixed_membership() {
    let vm = VarMap::new();
    let m = model(0.7, &vm);
    let batch = EdgeBatch {
        coarse_cells: vec![2],
        fine_feats: vec![0],
        neg_feats: vec![1, 2],
        n_negatives: 2,
    };
    let loss = nce_loss_identity(&m, batch, NceObjective::Softmax, &dev()).unwrap();
    let grads = loss.backward().unwrap();
    assert!(grad_norm(&grads, &var(&vm, MODULE_MU_VAR_NAME)).unwrap() > 1e-4);
    assert!(grad_norm(&grads, &var(&vm, MODULE_LOGITS_VAR_NAME)).unwrap() > 1e-4);
    // Frozen: the logits drop out of the graph, μ still trains.
    m.modules.as_ref().unwrap().set_frozen(true);
    let batch = EdgeBatch {
        coarse_cells: vec![2],
        fine_feats: vec![0],
        neg_feats: vec![1, 2],
        n_negatives: 2,
    };
    let loss = nce_loss_identity(&m, batch, NceObjective::Softmax, &dev()).unwrap();
    let grads = loss.backward().unwrap();
    assert!(grads
        .get(var(&vm, MODULE_LOGITS_VAR_NAME).as_tensor())
        .is_none());
    assert!(grad_norm(&grads, &var(&vm, MODULE_MU_VAR_NAME)).unwrap() > 1e-4);
}

/// Exact module term: reaches μ, b_m and e_c; never the logits, r, a_g, b_c. Its
/// gradient on the module bias is exactly mean_c (q_cm − p_cm).
#[test]
fn exact_term_reaches_mu_bias_cells_with_q_minus_p_and_nothing_else() {
    let vm = VarMap::new();
    let m = model(0.7, &vm);
    let modules = m.modules.as_ref().unwrap();
    let u = 3usize;
    let x = Tensor::rand(0f32, 5.0, (u, D), &dev()).unwrap();
    let pi = modules.membership().unwrap();
    let x_cm = x.matmul(&pi).unwrap();
    let e_units = m.e_cell.narrow(0, 0, u).unwrap();
    let loss = module_softmax_loss(&e_units, &modules.mu, &modules.b_module, &x_cm).unwrap();
    let grads = loss.backward().unwrap();
    assert!(grad_norm(&grads, &var(&vm, MODULE_MU_VAR_NAME)).unwrap() > 1e-5);
    assert!(grad_norm(&grads, &var(&vm, "e_cell")).unwrap() > 1e-5);
    for absent in [
        MODULE_LOGITS_VAR_NAME,
        MODULE_RESIDUAL_VAR_NAME,
        "b_feat",
        "b_cell",
    ] {
        assert!(
            grads.get(var(&vm, absent).as_tensor()).is_none(),
            "{absent} must not be reached by the exact term"
        );
    }
    // Analytic check on b_m: ∂L/∂b_m = mean_c (q_cm − p_cm).
    let s = e_units
        .matmul(&modules.mu.t().unwrap())
        .unwrap()
        .broadcast_add(&modules.b_module)
        .unwrap();
    let q = candle_util::candle_nn::ops::softmax(&s, 1).unwrap();
    let p = x_cm.broadcast_div(&x_cm.sum_keepdim(1).unwrap()).unwrap();
    let want: Vec<f32> = (q - p).unwrap().mean(0).unwrap().to_vec1().unwrap();
    let got: Vec<f32> = grads
        .get(var(&vm, MODULE_BIAS_VAR_NAME).as_tensor())
        .unwrap()
        .to_vec1()
        .unwrap();
    for (a, b) in want.iter().zip(&got) {
        assert!(
            (a - b).abs() < 1e-5,
            "∂L/∂b_m: analytic {a} vs autograd {b}"
        );
    }
}

/// Balance prior reaches only the logits; the ridge only the residual; the gate
/// KL reaches the gate tables and, through the live composition, μ.
#[test]
fn priors_reach_their_own_tables() {
    let vm = VarMap::new();
    let mut m = model(0.7, &vm);
    let pi = m.modules.as_ref().unwrap().membership().unwrap();
    let grads = module_balance_prior(&pi).unwrap().backward().unwrap();
    assert!(grad_norm(&grads, &var(&vm, MODULE_LOGITS_VAR_NAME)).unwrap() > 1e-6);
    assert!(grads
        .get(var(&vm, MODULE_MU_VAR_NAME).as_tensor())
        .is_none());

    var(&vm, MODULE_RESIDUAL_VAR_NAME)
        .set(&Tensor::ones((D, H), DType::F32, &dev()).unwrap())
        .unwrap();
    let grads = m.feature_ridge(0.1).unwrap().unwrap().backward().unwrap();
    assert!(grad_norm(&grads, &var(&vm, MODULE_RESIDUAL_VAR_NAME)).unwrap() > 1e-6);
    assert!(grads
        .get(var(&vm, MODULE_MU_VAR_NAME).as_tensor())
        .is_none());
    assert!(grads
        .get(var(&vm, MODULE_LOGITS_VAR_NAME).as_tensor())
        .is_none());

    m.enable_feature_gate(
        FeatureGateSpec {
            temperature: 1.0,
            ibp_alpha: None,
        },
        &vm,
        &dev(),
    )
    .unwrap();
    let grads = m.gate_kl().unwrap().unwrap().backward().unwrap();
    assert!(grad_norm(&grads, &var(&vm, "s_feat")).unwrap() > 1e-8);
    assert!(grad_norm(&grads, &var(&vm, "e_feat_logstd")).unwrap() > 1e-8);
    assert!(grad_norm(&grads, &var(&vm, MODULE_MU_VAR_NAME)).unwrap() > 1e-8);
    // And the gated gather still composes the live row.
    let idx = Tensor::from_vec(vec![1u32], 1, &dev()).unwrap();
    let rows = gather_feature_rows(&m, &idx).unwrap();
    assert_eq!(rows.dims(), &[1, H]);
    let _ = sparsemax(&m.modules.as_ref().unwrap().logits).unwrap();
}
