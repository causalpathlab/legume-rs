use super::*;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

const G: usize = 16;
const H: usize = 6;
const K_LATENT: usize = 4;

fn build(dev: &Device) -> (GemIndexedEncoder, VarMap) {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let layers = vec![8usize, 8];
    let enc = GemIndexedEncoder::new(
        GemIndexedEncoderArgs {
            n_genes: G,
            n_latent: K_LATENT,
            embedding_dim: H,
            layers: &layers,
            latent_noise: false,
        },
        &varmap,
        vb.pp("enc"),
    )
    .unwrap();
    (enc, varmap)
}

/// Two cells × four genes, with per-track values and visibility supplied by the
/// caller so each test states exactly the pattern it is probing.
struct Case {
    idx: Tensor,
    nas: Tensor,
    mat: Tensor,
    nas_vis: Tensor,
    mat_vis: Tensor,
}

fn case(dev: &Device, nas: &[f32], mat: &[f32], nas_vis: &[f32], mat_vis: &[f32]) -> Case {
    let (n, k) = (2usize, 4usize);
    Case {
        idx: Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5, 6, 7], (n, k), dev).unwrap(),
        nas: Tensor::from_vec(nas.to_vec(), (n, k), dev).unwrap(),
        mat: Tensor::from_vec(mat.to_vec(), (n, k), dev).unwrap(),
        nas_vis: Tensor::from_vec(nas_vis.to_vec(), (n, k), dev).unwrap(),
        mat_vis: Tensor::from_vec(mat_vis.to_vec(), (n, k), dev).unwrap(),
    }
}

fn logits(enc: &GemIndexedEncoder, c: &Case) -> Vec<Vec<f32>> {
    let input = GemEncoderInput {
        gene_indices: &c.idx,
        nascent_observed: &c.nas,
        mature_observed: &c.mat,
        nascent_residual: None,
        mature_residual: None,
        nascent_mean: None,
        mature_mean: None,
        nascent_visible: &c.nas_vis,
        mature_visible: &c.mat_vis,
    };
    // `train = false` so BatchNorm uses running stats and the forward is a pure
    // function of the input — otherwise a two-row batch would couple the rows.
    enc.logits(&input, false).unwrap().to_vec2().unwrap()
}

#[test]
fn logits_have_the_latent_shape_and_are_finite() {
    let dev = Device::Cpu;
    let (enc, _vm) = build(&dev);
    let c = case(
        &dev,
        &[3.0, 0.0, 5.0, 1.0, 2.0, 4.0, 0.0, 6.0],
        &[9.0, 2.0, 1.0, 4.0, 7.0, 3.0, 8.0, 2.0],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let z = logits(&enc, &c);
    assert_eq!(z.len(), 2);
    assert_eq!(z[0].len(), K_LATENT);
    for row in &z {
        for v in row {
            assert!(v.is_finite(), "non-finite logit {v}");
            assert!(v.abs() <= 8.0 + 1e-4, "logit {v} escaped the clamp");
        }
    }
}

/// The central safety property of the masked objective: a slot the encoder is
/// not shown must not influence the latent at all. If it did, the masked
/// imputation target would be visible to its own predictor and every
/// likelihood the model reports would be inflated.
#[test]
fn hidden_slot_values_cannot_reach_the_latent() {
    let dev = Device::Cpu;
    let (enc, _vm) = build(&dev);

    let nas_vis = [1.0f32, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let mat_vis = [1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];

    // Slot (0,2) is nascent-hidden and slot (0,1) is mature-hidden; perturb
    // exactly those and nothing else.
    let base_n = [3.0f32, 1.0, 5.0, 1.0, 2.0, 4.0, 9.0, 6.0];
    let mut pert_n = base_n;
    pert_n[2] = 999.0;
    let base_m = [9.0f32, 2.0, 1.0, 4.0, 7.0, 3.0, 8.0, 2.0];
    let mut pert_m = base_m;
    pert_m[1] = 999.0;

    let a = logits(&enc, &case(&dev, &base_n, &base_m, &nas_vis, &mat_vis));
    let b = logits(&enc, &case(&dev, &pert_n, &pert_m, &nas_vis, &mat_vis));

    for (ra, rb) in a.iter().zip(b.iter()) {
        for (x, y) in ra.iter().zip(rb.iter()) {
            assert!(
                (x - y).abs() < 1e-5,
                "a hidden slot changed the latent ({x} vs {y}) — the mask leaks"
            );
        }
    }
}

/// The complement of the leak test: a *visible* slot must actually matter.
/// Without this, the leak test above would pass trivially on an encoder that
/// ignores its input entirely.
#[test]
fn visible_slot_values_do_reach_the_latent() {
    let dev = Device::Cpu;
    let (enc, _vm) = build(&dev);
    let vis = [1.0f32; 8];

    let base_n = [3.0f32, 1.0, 5.0, 1.0, 2.0, 4.0, 9.0, 6.0];
    let mut pert_n = base_n;
    pert_n[0] = 400.0;
    let mat = [9.0f32, 2.0, 1.0, 4.0, 7.0, 3.0, 8.0, 2.0];

    let a = logits(&enc, &case(&dev, &base_n, &mat, &vis, &vis));
    let b = logits(&enc, &case(&dev, &pert_n, &mat, &vis, &vis));

    let moved = a[0]
        .iter()
        .zip(b[0].iter())
        .any(|(x, y)| (x - y).abs() > 1e-5);
    assert!(
        moved,
        "changing a visible nascent value left the latent untouched"
    );
}

/// In the `u→s` mode the mature track is entirely hidden, so the latent must be
/// a function of nascent alone. This is what makes the `z_u` inference pass —
/// and hence the velocity — well defined.
#[test]
fn nascent_to_mature_mode_ignores_mature_entirely() {
    let dev = Device::Cpu;
    let (enc, _vm) = build(&dev);
    let all_on = [1.0f32; 8];
    let all_off = [0.0f32; 8];
    let nas = [3.0f32, 1.0, 5.0, 1.0, 2.0, 4.0, 9.0, 6.0];

    let a = logits(
        &enc,
        &case(
            &dev,
            &nas,
            &[9.0, 2.0, 1.0, 4.0, 7.0, 3.0, 8.0, 2.0],
            &all_on,
            &all_off,
        ),
    );
    let b = logits(
        &enc,
        &case(
            &dev,
            &nas,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &all_on,
            &all_off,
        ),
    );

    for (ra, rb) in a.iter().zip(b.iter()) {
        for (x, y) in ra.iter().zip(rb.iter()) {
            assert!(
                (x - y).abs() < 1e-5,
                "mature values changed the latent while mature was fully hidden"
            );
        }
    }
}

/// Hiding one track entirely must leave the latent a function of the OTHER.
///
/// In `s→u` mode nascent is hidden and mature is fully visible, so the latent
/// has to still depend on mature. Because the tracks are pooled independently
/// and concatenated, `p^u` goes to zero while `p^s` is untouched. This is the
/// property the deleted attention path did NOT have — there both terms were
/// functions of the nascent slots, so hiding nascent gave every cell in the
/// batch one identical latent and ~30 % of each epoch trained against a
/// constant. It is asserted here so a future pooling change cannot quietly
/// reintroduce that.
#[test]
fn sum_pooling_still_depends_on_mature_when_nascent_is_hidden() {
    let dev = Device::Cpu;
    let (enc, _vm) = build(&dev);
    let all_on = [1.0f32; 8];
    let all_off = [0.0f32; 8];
    let nas = [3.0f32, 1.0, 5.0, 1.0, 2.0, 4.0, 9.0, 6.0];

    let a = logits(
        &enc,
        &case(
            &dev,
            &nas,
            &[9.0, 2.0, 1.0, 4.0, 7.0, 3.0, 8.0, 2.0],
            &all_off,
            &all_on,
        ),
    );
    let b = logits(&enc, &case(&dev, &nas, &[1.0; 8], &all_off, &all_on));
    let moved = a
        .iter()
        .zip(b.iter())
        .any(|(ra, rb)| ra.iter().zip(rb.iter()).any(|(x, y)| (x - y).abs() > 1e-5));
    assert!(
        moved,
        "nascent hidden + mature visible: the latent ignored mature ({a:?} vs {b:?})"
    );
}

/// A cell with nothing visible has no content to pool. Its pooled vector is
/// zeroed explicitly, so the latent is the bias-driven empty-cell
/// representation rather than an average over padding genes.
#[test]
fn fully_hidden_cells_share_one_defined_representation() {
    let dev = Device::Cpu;
    let (enc, _vm) = build(&dev);
    let off = [0.0f32; 8];
    // Two cells with completely different counts, both fully hidden.
    let z = logits(
        &enc,
        &case(
            &dev,
            &[3.0, 1.0, 5.0, 1.0, 900.0, 400.0, 9.0, 6.0],
            &[9.0, 2.0, 1.0, 4.0, 700.0, 300.0, 8.0, 2.0],
            &off,
            &off,
        ),
    );
    for (x, y) in z[0].iter().zip(z[1].iter()) {
        assert!(
            (x - y).abs() < 1e-5,
            "two fully-hidden cells got different latents ({x} vs {y})"
        );
        assert!(x.is_finite(), "fully-hidden latent is not finite");
    }
}

/// δ is zero-initialized, so at construction the mature embedding equals the
/// nascent one — training starts exactly at the "no differential processing"
/// null that the ridge shrinks toward.
#[test]
fn delta_starts_at_zero() {
    let dev = Device::Cpu;
    let (enc, _vm) = build(&dev);
    let d: Vec<f32> = enc
        .delta_embeddings()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(d.len(), G * H);
    assert!(d.iter().all(|&x| x == 0.0), "delta must be zero-init");
}

/// The bound must hold, but strictly — a value AT the bound would mean a hard
/// clamp had slipped back in, and with it the zero-gradient region that froze
/// the encoder (99.6 % of cells pinned their first stick logit on a six-sample
/// fit).
#[test]
fn logits_stay_strictly_inside_the_bound() {
    let dev = Device::Cpu;
    // Extreme pre-activations, far outside the bound.
    // The realistic operating range: pre-activations come from a Linear on
    // batch-normed features, so they sit in the low tens. (Past |x| ~ 9c, f32
    // tanh reaches exactly 1 and the gradient underflows again — a real limit,
    // documented on `soft_clamp`, an order of magnitude past where the hard
    // clamp bit.)
    let x = Tensor::from_vec(
        vec![-40.0f32, -16.0, -8.0, 0.0, 8.0, 16.0, 40.0],
        (1, 7),
        &dev,
    )
    .unwrap();
    let y: Vec<f32> = soft_clamp(&x, 8.0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    for v in &y {
        assert!(v.is_finite(), "soft clamp produced {v}");
        assert!(
            v.abs() < 8.0,
            "value {v} reached the bound exactly — that is the zero-gradient state"
        );
    }
    // Strictly increasing: the bound must not destroy ordering information.
    for w in y.windows(2) {
        assert!(w[1] > w[0], "soft clamp is not strictly monotone: {w:?}");
    }
    // Near zero it should be ~identity, so the ordinary operating range is
    // untouched.
    assert!((y[3] - 0.0).abs() < 1e-6);
}

/// Extreme inputs that a hard clamp would map to the SAME value must stay
/// distinguishable, otherwise two very different cells collapse onto one latent.
#[test]
fn saturated_inputs_remain_distinguishable() {
    let dev = Device::Cpu;
    let x = Tensor::from_vec(vec![9.0f32, 40.0], (1, 2), &dev).unwrap();
    let y: Vec<f32> = soft_clamp(&x, 8.0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        y[1] > y[0],
        "9 and 40 both mapped to {:?} — a hard clamp would do this",
        y
    );
}
