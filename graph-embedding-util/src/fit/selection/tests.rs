//! Tests for the selection pass's mask plumbing.
//!
//! Every defect these cover shipped to `main` with `cargo build` and `cargo test`
//! green, because no test touched `install_gate_pip`, `share_gate_masks`,
//! `resample_gate_mask`, `clear_gate_mask`, or the pip branch of
//! `gathered_gate_weights`. Compilation cannot see any of them: the fields are plain
//! `Option`s and `Arc`s, so dropping one is a runtime behaviour change, not a type
//! error.

use super::*;
use crate::model::{FactoredInit, ModelArgs, ModelInit};
use candle_util::candle_core::DType;
use candle_util::candle_nn::{VarBuilder, VarMap};

fn dev() -> Device {
    Device::Cpu
}

/// A `[rows, H]` pip table on the test device. `install_gate_pip` takes a tensor
/// because the composite fit uploads once and shares it across heads; tests have no
/// such concern, so they build one here.
fn pip_tensor(pip: &[f32], rows: usize, h: usize) -> Tensor {
    Tensor::from_slice(pip, (rows, h), &dev()).unwrap()
}

/// A free model plus `n_heads` sharing heads, all gated by the same `pip`, wired the
/// way `install_selection` wires them.
fn free_with_heads(
    n_features: usize,
    h: usize,
    n_heads: usize,
    pip: &[f32],
    vm: &VarMap,
) -> (JointEmbedModel, Vec<JointEmbedModel>) {
    let mut m = JointEmbedModel::new_with_init(
        ModelArgs {
            n_features,
            n_cells: 4,
            embedding_dim: h,
            seed: 7,
        },
        &ModelInit {
            e_feat: None,
            e_cell: None,
            b_feat: &vec![0f32; n_features],
            b_cell: &[0f32; 4],
        },
        vm,
        &dev(),
    )
    .unwrap();
    let mut heads: Vec<JointEmbedModel> = (0..n_heads)
        .map(|i| {
            JointEmbedModel::new_sharing_features(
                crate::model::ShareFeaturesArgs {
                    n_cells: 3,
                    embedding_dim: h,
                    shared_e_feat: m.e_feat.clone(),
                    shared_b_feat: m.b_feat.clone(),
                    e_cell_init: None,
                    b_cell_init: &[0f32; 3],
                    var_prefix: &format!("pb_l{i}"),
                    seed: 5,
                    shared_s_feat: None,
                    shared_e_feat_raw: None,
                    shared_e_feat_logstd: None,
                    shared_gate_pi_logit: None,
                    gate: None,
                },
                vm,
                &dev(),
            )
            .unwrap()
        })
        .collect();
    m.install_gate_pip(GateKind::Identity, &pip_tensor(pip, n_features, h))
        .unwrap();
    let cell = m.gate_mask_cell();
    for head in &mut heads {
        head.install_gate_pip(GateKind::Identity, &pip_tensor(pip, n_features, h))
            .unwrap();
        head.share_gate_masks(&cell, None);
    }
    (m, heads)
}

/// A factored model with velocity, plus one sharing head, both carrying an identity
/// and a velocity `pip`. `install_selection` is the thing under test.
fn factored_with_head(
    n_genes: usize,
    h: usize,
    vm: &VarMap,
) -> (JointEmbedModel, Vec<JointEmbedModel>) {
    let row_to_gene: Vec<u32> = (0..n_genes as u32).flat_map(|g| [g, g]).collect();
    let unspliced: Vec<bool> = (0..2 * n_genes).map(|r| r % 2 == 1).collect();
    let n_features = row_to_gene.len();
    let vs = VarBuilder::from_varmap(vm, DType::F32, &dev());
    let m = JointEmbedModel::new_factored(
        FactoredInit {
            n_features,
            n_cells: 4,
            embedding_dim: h,
            n_genes,
            row_to_gene: &row_to_gene,
            b_feat: &vec![0f32; n_features],
            b_cell: &[0f32; 4],
            seed: 9,
            unspliced_rows: Some(&unspliced),
        },
        vm,
        vs,
        &dev(),
    )
    .unwrap();
    let head = m.new_sharing_factor(3, "pb_l0", vm, &dev(), 5).unwrap();
    (m, vec![head])
}

/// A `SpliceGibbsResult` carrying only the fields `install_selection` reads; every
/// other field is left at a zero/empty value it never looks at.
fn splice_result(
    n_genes: usize,
    h: usize,
    beta: f32,
    delta: f32,
    ident: bool,
) -> SpliceGibbsResult {
    use crate::posterior::pb_gibbs::PartitionGeometry;
    SpliceGibbsResult {
        beta_pip: vec![beta; n_genes * h],
        beta_mean: vec![0.0; n_genes * h],
        delta_pip: vec![delta; n_genes * h],
        delta_mean: vec![0.0; n_genes * h],
        delta_identified: vec![ident; n_genes],
        mean_pb: vec![],
        mean_b_feat: vec![0.0; n_genes],
        mean_b_pb: vec![],
        beta_sigma2: vec![1.0; h],
        beta_pi0: vec![0.1; h],
        delta_sigma2: vec![1.0; h],
        delta_pi0: vec![0.1; h],
        partition: PartitionGeometry {
            pb_entries: 0,
            pb_scale: 1.0,
            feat_entries: 0,
            feat_scale: 1.0,
        },
        beta_sigma_diag: vec![],
        beta_pi0_diag: vec![],
        delta_sigma_diag: vec![],
        delta_pi0_diag: vec![],
        h,
        n_kept: 0,
    }
}

/// THE δ-vs-β test. `factored_feat_rows` looks up a weight table from a `Tensor` of
/// logits it cannot identify at the call site, which is the entire reason `GateKind`
/// exists. Handing δ the β table is a silent ~9× error on real data (measured means
/// 0.014 vs 0.125 on rep2_wt), not a rounding one — and nothing about it fails to
/// compile.
#[test]
fn the_two_gates_return_distinct_tables() {
    let (n_genes, h) = (4usize, 3usize);
    let vm = VarMap::new();
    let (mut m, mut heads) = factored_with_head(n_genes, h, &vm);
    let splice = splice_result(n_genes, h, 0.75, 0.25, true);
    install_selection(
        &mut m,
        &mut heads,
        None,
        Some(&splice),
        n_genes * 2,
        h,
        &dev(),
    )
    .unwrap();

    let rows = candle_util::candle_core::Tensor::from_vec(
        (0..n_genes as u32).collect::<Vec<_>>(),
        n_genes,
        &dev(),
    )
    .unwrap();
    let beta_w = m
        .gathered_gate_weights(GateKind::Identity, None, &rows)
        .unwrap()
        .expect("identity pip installed");
    let delta_w = m
        .gathered_gate_weights(GateKind::Velocity, None, &rows)
        .unwrap()
        .expect("velocity pip installed");
    let b: Vec<f32> = beta_w.flatten_all().unwrap().to_vec1().unwrap();
    let d: Vec<f32> = delta_w.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        b.iter().all(|x| (x - 0.75).abs() < 1e-6),
        "identity gate must read beta_pip, got {b:?}"
    );
    assert!(
        d.iter().all(|x| (x - 0.25).abs() < 1e-6),
        "velocity gate must read its OWN delta_pip, not beta's: got {d:?}"
    );
}

/// A gene whose δ is UNIDENTIFIED must be masked off outright, not gated at its prior
/// rate. The sampler leaves such a gene's `delta_pip` finite and plausible — it masks
/// it only at parquet-write time — so a NaN scan here (which is what shipped) is a
/// no-op that lets a prior draw into the fit.
#[test]
fn an_unidentified_delta_is_masked_off_not_gated_at_the_prior() {
    let (n_genes, h) = (4usize, 3usize);
    let vm = VarMap::new();
    let (mut m, mut heads) = factored_with_head(n_genes, h, &vm);
    // delta_pip is a perfectly finite 0.4 everywhere, and NOTHING is identified.
    let splice = splice_result(n_genes, h, 0.9, 0.4, false);
    assert!(
        splice.delta_pip.iter().all(|x| x.is_finite()),
        "precondition: the sampler does not emit NaN here, which is why scanning for \
         it was a no-op"
    );
    install_selection(
        &mut m,
        &mut heads,
        None,
        Some(&splice),
        n_genes * 2,
        h,
        &dev(),
    )
    .unwrap();
    let rows = candle_util::candle_core::Tensor::from_vec(
        (0..n_genes as u32).collect::<Vec<_>>(),
        n_genes,
        &dev(),
    )
    .unwrap();
    let d: Vec<f32> = m
        .gathered_gate_weights(GateKind::Velocity, None, &rows)
        .unwrap()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        d.iter().all(|x| *x == 0.0),
        "an unidentified δ must be gated OFF, got {d:?}"
    );
}

/// Every axis must see the SAME δ draw. `install_gate_pip` mints a fresh `Arc` for the
/// velocity cell each call, and `resample_gate_mask` runs on `axes[0]` only — so a
/// caller that shares just the identity cell leaves every other axis gating δ by the
/// mean while `axes[0]` gates it by a draw, i.e. the axes train against different
/// sub-models within one epoch.
#[test]
fn one_draw_reaches_every_axis_on_both_gates() {
    let (n_genes, h) = (6usize, 4usize);
    let vm = VarMap::new();
    let (mut m, mut heads) = factored_with_head(n_genes, h, &vm);
    let splice = splice_result(n_genes, h, 0.5, 0.5, true);
    install_selection(
        &mut m,
        &mut heads,
        None,
        Some(&splice),
        n_genes * 2,
        h,
        &dev(),
    )
    .unwrap();
    m.resample_gate_mask().unwrap();

    let read = |model: &JointEmbedModel, kind: GateKind| -> Vec<f32> {
        let rows = candle_util::candle_core::Tensor::from_vec(
            (0..n_genes as u32).collect::<Vec<_>>(),
            n_genes,
            &dev(),
        )
        .unwrap();
        model
            .gathered_gate_weights(kind, None, &rows)
            .unwrap()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    for kind in [GateKind::Identity, GateKind::Velocity] {
        let want = read(&m, kind);
        assert!(
            want.iter().all(|x| *x == 0.0 || *x == 1.0),
            "a draw must be binary, got {want:?}"
        );
        assert!(
            want.contains(&0.0) && want.contains(&1.0),
            "a pip of 0.5 over {} entries should not draw all-same; if this flakes the \
             draw is not random",
            n_genes * h
        );
        for (i, head) in heads.iter().enumerate() {
            assert_eq!(
                read(head, kind),
                want,
                "head {i} must see axes[0]'s {kind:?} draw, not its own"
            );
        }
    }
}

/// The draw must actually CHANGE the weights, and clearing it must restore the mean.
/// A regex once ate the `resample_gate_mask()` call outright: `pub` methods do not warn
/// when unused, so the gate ran deterministic while the log announced that it was not.
#[test]
fn a_draw_changes_the_weights_and_clearing_restores_the_mean() {
    let (n_features, h) = (64usize, 4usize);
    let vm = VarMap::new();
    let pip = vec![0.5f32; n_features * h];
    let (m, heads) = free_with_heads(n_features, h, 2, &pip, &vm);

    let all = |model: &JointEmbedModel| -> Vec<f32> {
        model
            .gate_weights(GateKind::Identity, &model.e_feat)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let mean = all(&m);
    assert!(
        mean.iter().all(|x| (x - 0.5).abs() < 1e-6),
        "before any draw the gate is the mean pip"
    );

    m.resample_gate_mask().unwrap();
    let drawn = all(&m);
    assert!(
        drawn.iter().all(|x| *x == 0.0 || *x == 1.0),
        "after a draw the gate is binary"
    );
    assert!(
        drawn != mean,
        "the draw must actually change what the model multiplies by"
    );
    for head in &heads {
        assert_eq!(all(head), drawn, "heads share the draw");
    }

    m.clear_gate_mask();
    assert_eq!(
        all(&m),
        mean,
        "clearing reverts to E[z] = pip, which is the dictionary the fit implies"
    );
    for head in &heads {
        assert_eq!(all(head), mean, "clearing reaches the heads too");
    }
}

/// A `pip` of exactly 0 is a permanent mask: it can never draw a 1, so that
/// coordinate's loading never trains and the shipped dictionary carries an exact zero.
#[test]
fn a_zero_pip_never_draws_on() {
    let (n_features, h) = (32usize, 4usize);
    let vm = VarMap::new();
    let mut pip = vec![1.0f32; n_features * h];
    for (i, p) in pip.iter_mut().enumerate() {
        if i % 2 == 0 {
            *p = 0.0;
        }
    }
    let (m, _) = free_with_heads(n_features, h, 1, &pip, &vm);
    for _ in 0..20 {
        m.resample_gate_mask().unwrap();
        let w: Vec<f32> = m
            .gate_weights(GateKind::Identity, &m.e_feat)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, x) in w.iter().enumerate() {
            let want = if i % 2 == 0 { 0.0 } else { 1.0 };
            assert_eq!(*x, want, "entry {i}: pip {} drew {x}", pip[i]);
        }
    }
}

/// An UNGATED model carrying only an installed `pip` must ship the masked dictionary.
///
/// `gather_feature_rows` takes the pip branch whether or not the learned gate exists,
/// so such a model trains under the mask. `materialize_e_feat` used to require the
/// learned gate's Vars to be present before it baked anything in, so the dictionary
/// went out UNMASKED — phase 2 then projected every cell against a feature side the
/// fit had never used. Nothing about that fails to compile, and no arm of it is
/// unreachable: `senna bge --posterior` reaches it whenever the gate is off.
#[test]
fn an_ungated_model_ships_the_mask_it_trained_under() {
    let (n_features, h) = (8usize, 3usize);
    let vm = VarMap::new();
    let mut pip = vec![1.0f32; n_features * h];
    for (i, p) in pip.iter_mut().enumerate() {
        if i % 3 == 0 {
            *p = 0.0;
        }
    }
    let (mut m, _) = free_with_heads(n_features, h, 1, &pip, &vm);
    assert!(
        m.gate.is_none() && m.s_feat.is_none(),
        "precondition: no LEARNED gate, only a pip"
    );
    let raw: Vec<f32> = m
        .e_feat_raw
        .as_ref()
        .expect("installing a pip pins the raw Var")
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    m.materialize_e_feat().unwrap();
    let got: Vec<f32> = m.e_feat.flatten_all().unwrap().to_vec1().unwrap();
    for (i, (g, r)) in got.iter().zip(&raw).enumerate() {
        let want = r * pip[i];
        assert!(
            (g - want).abs() < 1e-6,
            "entry {i}: dictionary {g} != raw {r} x pip {}",
            pip[i]
        );
    }

    // And it is IDEMPOTENT: materializing again must not gate the already-gated table.
    m.materialize_e_feat().unwrap();
    let twice: Vec<f32> = m.e_feat.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(
        got, twice,
        "a second materialize must not apply the pip twice"
    );
}

/// Under a `pip`, `feature_selection` must emit NOTHING rather than the pip again.
/// Training stops reading the learned logits the moment a pip is installed, so their
/// `α` is frozen at its init — and `gate_weights` would hand back the pip itself,
/// making `feature_selection.parquet` a byte copy of `feature_pip.parquet`.
#[test]
fn a_pip_suppresses_the_learned_selection_table() {
    let (n_genes, h) = (4usize, 3usize);
    let vm = VarMap::new();
    let (mut m, mut heads) = factored_with_head(n_genes, h, &vm);
    m.enable_feature_gate(
        crate::model::FeatureGateSpec { temperature: 1.0 },
        &vm,
        &dev(),
    )
    .unwrap();
    assert!(
        m.feature_selection().unwrap().is_some(),
        "with only the learned gate the table is the readout"
    );
    let splice = splice_result(n_genes, h, 0.5, 0.5, true);
    install_selection(
        &mut m,
        &mut heads,
        None,
        Some(&splice),
        n_genes * 2,
        h,
        &dev(),
    )
    .unwrap();
    assert!(
        m.feature_selection().unwrap().is_none(),
        "once a pip selects, feature_pip is the table to read"
    );
    assert!(
        m.velocity_selection().unwrap().is_none(),
        "same for the velocity gate"
    );
}
