//! The two GEM tracks must not share a dispersion row.
//!
//! `phi_is_per_track` asserts only `phi.dims() == [G, 2]`. A shape assertion
//! cannot see the two tracks *reading* the same row, which is what that test's
//! doc comment actually claims, and nothing else in the crate covered it either:
//! collapsing `log_phi_track`'s `Track::Mature` arm from row 1 to row 0 — a
//! one-character change that silently destroys per-track dispersion — passed the
//! entire suite. This test closes that hole behaviourally.
//!
//! The isolation trick is `delta = 0`. Both tracks then read the same gene
//! embedding, so the dictionaries collapse and the mixture rate is identical;
//! dispersion is the only remaining thing that can move an NB score. Any
//! difference between the two tracks' scores is therefore attributable to `phi`
//! alone.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_util::decoder::{GemEtmDecoder, GemMaskedTarget, Track};

const G: usize = 8;
const H: usize = 3;
const TOPICS: usize = 2;
const N: usize = 2;
const TOPK: usize = 4;

/// Score the same masked target under both tracks, with the two dispersion rows
/// set far apart and `delta = 0` so nothing else can differ.
fn track_scores(dev: &Device) -> (f32, f32) {
    let rho: Vec<f32> = (0..G * H).map(|i| (i % 7) as f32 * 0.3 - 0.6).collect();
    let rho = Tensor::from_vec(rho, (G, H), dev).unwrap();
    let delta = Tensor::zeros((G, H), DType::F32, dev).unwrap();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let dec = GemEtmDecoder::new(
        TOPICS,
        rho,
        delta,
        GemEtmDecoder::uniform_log_pi(G, dev).unwrap(),
        vb.pp("dec"),
    )
    .unwrap();

    // Row 0 (nascent) and row 1 (mature) deliberately far apart. `log_phi` is
    // still a registered Var — only the background was frozen — so it can be
    // driven off its constant init through the varmap.
    let mut rows = vec![(0.5f32).ln(); G];
    rows.extend(std::iter::repeat_n(8.0f32.ln(), G));
    varmap.data().lock().unwrap()["dec.log_phi"]
        .set(&Tensor::from_vec(rows, (2, G), dev).unwrap())
        .unwrap();

    let indices = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5, 6, 7], (N, TOPK), dev).unwrap();
    let values = Tensor::from_vec(
        vec![3.0f32, 1.0, 7.0, 2.0, 5.0, 4.0, 1.0, 6.0],
        (N, TOPK),
        dev,
    )
    .unwrap();
    let lib = (values.sum_keepdim(1).unwrap() + 1.0).unwrap();
    let mask = Tensor::ones((N, TOPK), DType::F32, dev).unwrap();
    let log_theta = Tensor::full(-(TOPICS as f64).ln() as f32, (N, TOPICS), dev).unwrap();

    let score = |track: Track| -> f32 {
        let full = dec.full_logits_kg(track).unwrap();
        let logz = GemEtmDecoder::log_partition_from_logits(&full).unwrap();
        let target = GemMaskedTarget {
            indices: &indices,
            residual: None,
            values: &values,
            lib: &lib,
            mask: &mask,
            values_weight: None,
        };
        dec.impute_masked_nb(&log_theta, &target, track, &logz)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };

    (score(Track::Nascent), score(Track::Mature))
}

/// With `delta = 0` the two tracks agree on everything except dispersion, so
/// giving the two `phi` rows different values must move the two tracks' NB
/// scores apart. If it does not, both tracks are reading the same row.
#[test]
fn the_two_tracks_do_not_share_a_dispersion_row() {
    let (nascent, mature) = track_scores(&Device::Cpu);

    assert!(
        (nascent - mature).abs() > 1e-3,
        "nascent and mature scored identically ({nascent} vs {mature}) with the \
         two phi rows set far apart and delta = 0 — the tracks are sharing a \
         dispersion row, so phi is not per-track"
    );
}
