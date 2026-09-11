//! Growing a checkpoint must not change what the model computes.
//!
//! The two properties are in tension, which is the whole reason the fill rule
//! is per-tensor rather than "pad with zeros":
//!
//! - `β = softmax_g(α·ρᵀ)` must be **unchanged** at step 0, so `α`'s new
//!   columns are zero.
//! - The added subspace must still **receive gradient**, so `ρ`'s new columns
//!   are *not* zero. Zeroing both would make `∂β/∂α_new ∝ ρ_new = 0` and
//!   `∂β/∂ρ_new ∝ α_new = 0` — capacity that can never learn anything.
//!
//! A change that satisfies only the first is easy to make by accident, and
//! silently gives a model that trains as if the flag had done nothing.

use candle_core::{DType, Device, Tensor};
use candle_util::grow::{grow_tensor, load_grown, AxisRemap, GrowthDims, NEW_TOPIC_LOGIT_BIAS};

const K_OLD: usize = 4;
const K_NEW: usize = 7;
const H_OLD: usize = 12;
const H_NEW: usize = 18;
const D: usize = 40;

fn dims() -> GrowthDims<'static> {
    GrowthDims {
        k_old: K_OLD,
        k_new: K_NEW,
        h_old: H_OLD,
        h_new: H_NEW,
        gene_axis: None,
    }
}

/// No K/H growth; only the gene axis changes.
fn gene_dims(new_to_old: &[Option<usize>], n_old: usize) -> GrowthDims<'_> {
    GrowthDims {
        k_old: K_OLD,
        k_new: K_OLD,
        h_old: H_OLD,
        h_new: H_OLD,
        gene_axis: Some(AxisRemap { new_to_old, n_old }),
    }
}

/// A tensor whose entries are distinct and non-zero, so a value that survives
/// the copy is distinguishable from a value that was filled in.
fn ramp(shape: &[usize], dev: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
    Tensor::from_vec(v, shape, dev).expect("ramp")
}

fn to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .to_vec2()
        .expect("vec2")
}

#[test]
fn alpha_keeps_its_values_and_zeroes_the_new_embedding_columns() {
    let dev = Device::Cpu;
    let saved = ramp(&[K_OLD, H_OLD], &dev);
    let fresh = ramp(&[K_OLD, H_NEW], &dev).affine(-1.0, 0.0).expect("neg");

    let out = grow_tensor("dec_0.topic.embeddings", &fresh, &saved, &dims()).expect("grow");
    assert_eq!(out.dims(), &[K_OLD, H_NEW]);

    let (o, s) = (to_vec2(&out), to_vec2(&saved));
    for k in 0..K_OLD {
        for h in 0..H_OLD {
            assert_eq!(o[k][h], s[k][h], "checkpoint value lost at ({k},{h})");
        }
        for (h, &v) in o[k].iter().enumerate().skip(H_OLD) {
            assert_eq!(v, 0.0, "new alpha column ({k},{h}) must be zero");
        }
    }
}

/// The counterpart, and the one that is easy to get wrong: `ρ`'s new columns
/// must keep their random init or the added subspace is dead on arrival.
#[test]
fn rho_keeps_its_fresh_init_in_the_new_columns() {
    let dev = Device::Cpu;
    let saved = ramp(&[D, H_OLD], &dev);
    let fresh = ramp(&[D, H_NEW], &dev).affine(-1.0, 0.0).expect("neg");

    let out = grow_tensor("enc.feature.embeddings", &fresh, &saved, &dims()).expect("grow");
    let (o, f, s) = (to_vec2(&out), to_vec2(&fresh), to_vec2(&saved));
    for d in 0..D {
        for h in 0..H_OLD {
            assert_eq!(o[d][h], s[d][h], "checkpoint value lost at ({d},{h})");
        }
        for h in H_OLD..H_NEW {
            assert_eq!(
                o[d][h], f[d][h],
                "rho's new column ({d},{h}) must keep its init"
            );
            assert_ne!(o[d][h], 0.0, "a zeroed rho column is a dead subspace");
        }
    }
}

#[test]
fn added_topics_start_switched_off_at_the_encoder() {
    let dev = Device::Cpu;
    let l2 = 5usize;

    let w = grow_tensor(
        "enc.nn.enc.z.mean.weight",
        &ramp(&[K_NEW, l2], &dev),
        &ramp(&[K_OLD, l2], &dev),
        &dims(),
    )
    .expect("grow weight");
    for (k, row) in to_vec2(&w).iter().enumerate().skip(K_OLD) {
        assert!(
            row.iter().all(|&x| x == 0.0),
            "added topic {k} must start with zero weights, got {row:?}"
        );
    }
    let _ = l2;

    let b = grow_tensor(
        "enc.nn.enc.z.mean.bias",
        &ramp(&[K_NEW], &dev),
        &ramp(&[K_OLD], &dev),
        &dims(),
    )
    .expect("grow bias");
    let bv: Vec<f32> = b.to_vec1().expect("vec1");
    for (k, v) in bv.iter().enumerate().take(K_NEW).skip(K_OLD) {
        assert!(
            (f64::from(*v) - NEW_TOPIC_LOGIT_BIAS).abs() < 1e-6,
            "added topic {k} bias {v} should be the off-switch constant"
        );
    }
    // With zero weights the pre-activation IS the bias, so this is the mass the
    // topic starts with relative to a parent topic sitting at a logit of ~0.
    assert!(
        NEW_TOPIC_LOGIT_BIAS.exp() < 1e-3,
        "the off-switch must actually switch off"
    );
}

/// Growth must not become a way to paper over an architecture change: a
/// widened hidden layer also grows a tensor, and has to stay an error.
#[test]
fn an_unrelated_shape_change_is_rejected() {
    let dev = Device::Cpu;
    // 5 → 9 is neither the K growth (4 → 7) nor the H growth (12 → 18).
    let err = grow_tensor(
        "enc.nn.enc.fc.relu_linear_stack.1.weight",
        &ramp(&[9, 64], &dev),
        &ramp(&[5, 64], &dev),
        &dims(),
    )
    .expect_err("an unexplained growth must not be silently zero-padded");
    let msg = err.to_string();
    assert!(
        msg.contains("architecture change"),
        "should say what it is refusing: {msg}"
    );
}

/// Grow one tensor onto the axis `[old 2, unseen, old 0]`: four saved rows
/// become three, one of them a feature the checkpoint never had. Returns the
/// grown tensor alongside it and the checkpoint, with the part both callers
/// share already asserted.
fn onto_a_changed_gene_axis(name: &str, width: usize) -> (Tensor, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let dev = Device::Cpu;
    let saved = ramp(&[4, width], &dev);
    let fresh = ramp(&[3, width], &dev).affine(-1.0, 0.0).expect("neg");
    let remap = [Some(2), None, Some(0)];
    let out = grow_tensor(name, &fresh, &saved, &gene_dims(&remap, 4)).expect("grow");
    let (o, s) = (to_vec2(&out), to_vec2(&saved));
    assert_eq!(o[0], s[2], "a known feature keeps its row exactly");
    assert_eq!(o[2], s[0]);
    (out, o, s)
}

/// A gene the checkpoint knew keeps its ρ row wherever it now sits; one it
/// never saw starts at the mean of the rows it did have.
#[test]
fn rho_rows_follow_the_gene_by_name_and_new_genes_start_at_the_mean() {
    let (_, o, s) = onto_a_changed_gene_axis("enc.feature.embeddings", H_OLD);
    for h in 0..H_OLD {
        let mean = (0..4).map(|d| s[d][h]).sum::<f32>() / 4.0;
        assert!(
            (o[1][h] - mean).abs() < 1e-5,
            "new gene column {h}: {} vs {mean}",
            o[1][h]
        );
    }
}

/// The encoder's gene-keyed input weight is gathered on its COLUMN axis, and a
/// new gene's column is zero so the encoder's output does not move.
#[test]
fn a_gene_keyed_weight_gathers_its_columns_and_zeroes_the_new_gene() {
    let dev = Device::Cpu;
    let hidden = 3usize;
    let saved = ramp(&[hidden, 4], &dev);
    let remap = [Some(3), Some(1), None];
    let fresh = ramp(&[hidden, 3], &dev);

    let out = grow_tensor(
        "enc.nn.enc.fc.relu_linear_stack.0.weight",
        &fresh,
        &saved,
        &gene_dims(&remap, 4),
    )
    .expect("grow");
    let (o, s) = (to_vec2(&out), to_vec2(&saved));
    for r in 0..hidden {
        assert_eq!(o[r][0], s[r][3]);
        assert_eq!(o[r][1], s[r][1]);
        assert_eq!(o[r][2], 0.0, "new gene's input weight must be zero");
    }
}

/// Same gene count, different order: the shapes match, but a positional copy
/// would key every gene to the wrong row. The loader must gather anyway, and
/// a tensor without a gene axis is left alone.
#[test]
fn a_same_length_permutation_is_gathered_not_copied() {
    use candle_nn::VarMap;
    let dev = Device::Cpu;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("ckpt.safetensors");
    let path = path.to_str().expect("utf8");

    let src = VarMap::new();
    let (d, h) = (3usize, 2usize);
    src.get(
        (1, d),
        "dec_0.log_phi",
        candle_nn::Init::Const(0.0),
        DType::F32,
        &dev,
    )
    .expect("var");
    src.get(
        (K_OLD, h),
        "dec_0.topic.embeddings",
        candle_nn::Init::Const(0.0),
        DType::F32,
        &dev,
    )
    .expect("var");
    {
        let data = src.data().lock().expect("lock");
        data["dec_0.log_phi"]
            .set(&ramp(&[1, d], &dev))
            .expect("set");
        data["dec_0.topic.embeddings"]
            .set(&ramp(&[K_OLD, h], &dev))
            .expect("set");
    }
    src.save(path).expect("save");

    let dst = VarMap::new();
    dst.get(
        (1, d),
        "dec_0.log_phi",
        candle_nn::Init::Const(0.0),
        DType::F32,
        &dev,
    )
    .expect("var");
    dst.get(
        (K_OLD, h),
        "dec_0.topic.embeddings",
        candle_nn::Init::Const(0.0),
        DType::F32,
        &dev,
    )
    .expect("var");
    let remap = [Some(2), Some(0), Some(1)];
    let dims = GrowthDims {
        k_old: K_OLD,
        k_new: K_OLD,
        h_old: h,
        h_new: h,
        gene_axis: Some(AxisRemap {
            new_to_old: &remap,
            n_old: d,
        }),
    };
    load_grown(&dst, path, &dims).expect("load");

    let data = dst.data().lock().expect("lock");
    let phi = to_vec2(data["dec_0.log_phi"].as_tensor());
    assert_eq!(
        phi[0],
        vec![3.0, 1.0, 2.0],
        "gathered onto the new gene order"
    );
    let alpha = to_vec2(data["dec_0.topic.embeddings"].as_tensor());
    assert_eq!(
        alpha,
        to_vec2(&ramp(&[K_OLD, h], &dev)),
        "no gene axis: copied as is"
    );
}

/// The membership table is gene-keyed, so it follows the axis the way `ρ` does
/// — but it must NOT take `ρ`'s fill. Sparsemax has zero Jacobian outside the
/// support, so a feature's set of modules can only shrink: whatever support it
/// starts with is the most it will ever have. Given the checkpoint's mean
/// logits, a new feature would inherit that vector's support — one module, for
/// a trained spread — and be locked out of the rest of the dictionary before
/// its first step. A flat row is the only start that leaves every module
/// reachable, and it composes the dictionary's centroid.
#[test]
fn a_new_features_membership_starts_flat_so_every_module_stays_reachable() {
    use candle_util::nn::layers::sparsemax;
    let m = 5usize;
    // The checkpoint's logits are spread the way a trained table's are, so the
    // mean of them has a support of ONE module: taking it would foreclose the
    // other four for this feature permanently.
    let (out, _, _) = onto_a_changed_gene_axis("enc.modules.logits", m);

    let pi = to_vec2(&sparsemax(&out).expect("sparsemax"));
    let uniform = 1.0 / m as f32;
    for (j, &p) in pi[1].iter().enumerate() {
        assert!(
            (p - uniform).abs() < 1e-6,
            "a new feature starts at {p} on module {j}, not the uniform {uniform}: \
             every module it is not on is foreclosed for good",
        );
    }
}
