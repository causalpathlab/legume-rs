use super::*;
use crate::data::indexed::{GemSample, GeneTrackMap};
use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

const G: usize = 24;
const H: usize = 4;
const K: usize = 3;

fn map_of(n_genes: usize) -> GeneTrackMap {
    GeneTrackMap {
        row_to_gene: (0..2 * n_genes).map(|r| (r / 2) as u32).collect(),
        row_is_nascent: (0..2 * n_genes).map(|r| r % 2 == 0).collect(),
        n_genes,
    }
}

/// A decoder whose dictionary is KNOWN: each topic loads a disjoint third of
/// the genes, so `θ` is recoverable from counts by inspection.
fn planted_decoder(dev: &Device) -> (GemEtmDecoder, VarMap) {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

    // ρ: gene g gets a large value on the H-slot of its own block.
    let mut rho = vec![0f32; G * H];
    for g in 0..G {
        rho[g * H + (g % K)] = 3.0;
    }
    let rho_t = Tensor::from_vec(rho, (G, H), dev).unwrap();
    let delta_t = Tensor::zeros((G, H), DType::F32, dev).unwrap();
    let dec = GemEtmDecoder::new(K, rho_t, delta_t, vb.pp("dec")).unwrap();

    // α: topic t reads H-slot t, so β_t concentrates on genes with g % K == t.
    let mut alpha = vec![0f32; K * H];
    for t in 0..K {
        alpha[t * H + t] = 1.0;
    }
    {
        let vars = varmap.data().lock().unwrap();
        let a = vars
            .get("dec.topic.embeddings")
            .expect("decoder registers alpha as `topic.embeddings`");
        a.set(&Tensor::from_vec(alpha, (K, H), dev).unwrap()).unwrap();
    }
    (dec, varmap)
}

/// One cell whose counts come ONLY from topic `t`'s genes, on both tracks.
fn cell_on_topic(t: usize) -> GemSample {
    let genes: Vec<u32> = (0..G as u32).collect();
    let vals: Vec<f32> = (0..G).map(|g| if g % K == t { 20.0 } else { 0.0 }).collect();
    GemSample { genes, nascent: vals.clone(), mature: vals }
}

/// The decisive test for the post-hoc claim: fitting θ against the FROZEN
/// dictionary must move it TOWARD the truth, from a deliberately wrong start.
///
/// This is the whole basis for recovering a cell-level delta without a latent
/// delta in the model. If a warm-started sampler cannot improve on its
/// initialization, `Δθ = θ^u − θ^s` is measuring the initialization, not the
/// cell — which is exactly the failure the old two-pass `Δz` had.
#[test]
fn fit_theta_to_track_moves_toward_the_planted_topic() {
    let dev = Device::Cpu;
    let (dec, _vm) = planted_decoder(&dev);
    let map = map_of(G);

    // Three cells, one per topic.
    let samples: Vec<GemSample> = (0..K).map(cell_on_topic).collect();
    let data = GemIndexedData::from_samples(samples, &map, G, None, None, None).unwrap();
    let mb = data.minibatch_ordered(0, K, &dev).unwrap();

    // Deliberately wrong start: uniform logits, so every cell begins at 1/K.
    let init = Tensor::zeros((K, K), DType::F32, &dev).unwrap();
    let fitted = fit_theta_to_track(
        &dec,
        &mb,
        &init,
        Track::Mature,
        &ThetaFitConfig::default(),
    )
    .unwrap();

    let th: Vec<Vec<f32>> = fitted.exp().unwrap().to_vec2().unwrap();
    for (cell, row) in th.iter().enumerate() {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "row {cell} must stay on the simplex, sums to {s}");
        // It must be closer to its own topic than a uniform start was.
        assert!(
            row[cell] > 1.0 / K as f32,
            "cell {cell} was generated from topic {cell}; the fit left it at \
             {:.3}, no better than the uniform {:.3} it started from",
            row[cell],
            1.0 / K as f32
        );
    }
}

/// ESS must not wander off the simplex or emit NaNs even when a track is empty
/// for a cell — the nascent track is routinely all-zero for real cells.
#[test]
fn fit_theta_survives_a_cell_with_no_counts_on_the_track() {
    let dev = Device::Cpu;
    let (dec, _vm) = planted_decoder(&dev);
    let map = map_of(G);
    let empty = GemSample {
        genes: (0..G as u32).collect(),
        nascent: vec![0.0; G],
        mature: vec![0.0; G],
    };
    let data =
        GemIndexedData::from_samples(vec![cell_on_topic(0), empty], &map, G, None, None, None)
            .unwrap();
    let mb = data.minibatch_ordered(0, 2, &dev).unwrap();
    let init = Tensor::zeros((2, K), DType::F32, &dev).unwrap();
    let fitted = fit_theta_to_track(
        &dec,
        &mb,
        &init,
        Track::Nascent,
        &ThetaFitConfig::default(),
    )
    .unwrap();
    let th: Vec<Vec<f32>> = fitted.exp().unwrap().to_vec2().unwrap();
    for (i, row) in th.iter().enumerate() {
        assert!(row.iter().all(|v| v.is_finite()), "row {i} has non-finite θ");
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "row {i} sums to {s}");
    }
}
