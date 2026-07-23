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
        a.set(&Tensor::from_vec(alpha, (K, H), dev).unwrap())
            .unwrap();
    }
    (dec, varmap)
}

/// One cell whose counts come ONLY from topic `t`'s genes, on both tracks.
fn cell_on_topic(t: usize) -> GemSample {
    let genes: Vec<u32> = (0..G as u32).collect();
    let vals: Vec<f32> = (0..G)
        .map(|g| if g % K == t { 20.0 } else { 0.0 })
        .collect();
    GemSample {
        genes,
        nascent: vals.clone(),
        mature: vals,
    }
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
    let fitted =
        fit_theta_to_track(&dec, &mb, &init, Track::Mature, &ThetaFitConfig::default()).unwrap();

    let th: Vec<Vec<f32>> = fitted.exp().unwrap().to_vec2().unwrap();
    for (cell, row) in th.iter().enumerate() {
        let s: f32 = row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-3,
            "row {cell} must stay on the simplex, sums to {s}"
        );
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

/// The retained-mean must still be a composition. Averaging θ over states keeps
/// it on the simplex by convexity; averaging `z` and mapping afterwards would
/// not be the posterior mean at all, so this pins which space the average
/// happens in.
#[test]
fn fit_theta_posterior_mean_stays_on_the_simplex() {
    let dev = Device::Cpu;
    let (dec, _vm) = planted_decoder(&dev);
    let map = map_of(G);
    let samples: Vec<GemSample> = (0..K).map(cell_on_topic).collect();
    let data = GemIndexedData::from_samples(samples, &map, G, None, None, None).unwrap();
    let mb = data.minibatch_ordered(0, K, &dev).unwrap();
    let init = Tensor::zeros((K, K), DType::F32, &dev).unwrap();

    let (log_theta, sd) =
        fit_theta_posterior(&dec, &mb, &init, Track::Mature, &ThetaFitConfig::default()).unwrap();

    assert_eq!(
        sd.dims(),
        log_theta.dims(),
        "SD must match the mean's shape"
    );
    let th: Vec<Vec<f32>> = log_theta.exp().unwrap().to_vec2().unwrap();
    for (cell, row) in th.iter().enumerate() {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "row {cell} sums to {s}, not 1");
        assert!(
            row.iter().all(|v| v.is_finite()),
            "row {cell} has non-finite θ"
        );
    }
    let sd_v: Vec<f32> = sd.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        sd_v.iter().all(|v| v.is_finite() && *v >= 0.0),
        "SD must be finite and non-negative"
    );
}

/// `n_keep = 1` must reproduce the old single-final-draw behaviour exactly, so
/// the change is opt-out rather than a fork. A one-state average has no spread
/// by construction, which is the observable that proves nothing was retained.
#[test]
fn fit_theta_posterior_with_one_retained_state_has_zero_spread() {
    let dev = Device::Cpu;
    let (dec, _vm) = planted_decoder(&dev);
    let map = map_of(G);
    let data =
        GemIndexedData::from_samples(vec![cell_on_topic(0)], &map, G, None, None, None).unwrap();
    let mb = data.minibatch_ordered(0, 1, &dev).unwrap();
    let init = Tensor::zeros((1, K), DType::F32, &dev).unwrap();

    let cfg = ThetaFitConfig {
        n_keep: 1,
        ..ThetaFitConfig::default()
    };
    let (_, sd) = fit_theta_posterior(&dec, &mb, &init, Track::Mature, &cfg).unwrap();

    let sd_v: Vec<f32> = sd.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        sd_v.iter().all(|v| *v == 0.0),
        "one retained state cannot have spread, got {sd_v:?}"
    );
}

/// The geometric center must be CLOSED. Averaging log θ lands off the simplex —
/// unlike the arithmetic mean, which stays on it for free — so if the closure
/// were dropped the rows would silently stop being probabilities and every
/// downstream `exp(latent)` would read a mis-scaled θ.
#[test]
fn fit_theta_posterior_geometric_center_is_closed() {
    let dev = Device::Cpu;
    let (dec, _vm) = planted_decoder(&dev);
    let map = map_of(G);
    let samples: Vec<GemSample> = (0..K).map(cell_on_topic).collect();
    let data = GemIndexedData::from_samples(samples, &map, G, None, None, None).unwrap();
    let mb = data.minibatch_ordered(0, K, &dev).unwrap();
    let init = Tensor::zeros((K, K), DType::F32, &dev).unwrap();

    let cfg = ThetaFitConfig {
        mean: ThetaMean::Geometric,
        ..ThetaFitConfig::default()
    };
    let (log_theta, sd) = fit_theta_posterior(&dec, &mb, &init, Track::Mature, &cfg).unwrap();

    let th: Vec<Vec<f32>> = log_theta.exp().unwrap().to_vec2().unwrap();
    for (cell, row) in th.iter().enumerate() {
        let s: f32 = row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-3,
            "geometric row {cell} sums to {s}, not 1"
        );
        assert!(
            row.iter().all(|v| v.is_finite()),
            "geometric row {cell} has non-finite θ"
        );
    }
    // The log-scale branch never clamps, so a −inf here would mean `log_softmax`
    // stopped being the safe map this branch relies on.
    let lt: Vec<f32> = log_theta.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        lt.iter().all(|v| v.is_finite()),
        "log θ must stay finite without a clamp"
    );
    let sd_v: Vec<f32> = sd.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        sd_v.iter().all(|v| v.is_finite() && *v >= 0.0),
        "SD must be finite, non-negative"
    );
}

/// The spread must track how much the counts actually pin the cell. A cell with
/// no counts on the track has a FLAT likelihood, so ESS wanders the prior; a
/// cell with 20 counts on every gene of one topic is pinned. If the SD does not
/// separate those two it is measuring the sampler rather than the evidence, and
/// is useless as the "is this Δθ real?" check it exists to be.
///
/// Statistical, not structural — the CPU backend cannot seed `randn`
/// (`Device::set_seed` errors there), so the margin is deliberately wide.
#[test]
fn fit_theta_posterior_spread_is_wider_when_the_counts_are_uninformative() {
    let dev = Device::Cpu;
    let (dec, _vm) = planted_decoder(&dev);
    let map = map_of(G);
    let empty = GemSample {
        genes: (0..G as u32).collect(),
        nascent: vec![0.0; G],
        mature: vec![0.0; G],
    };
    // Row 0 pinned by counts, row 1 unconstrained.
    let data =
        GemIndexedData::from_samples(vec![cell_on_topic(0), empty], &map, G, None, None, None)
            .unwrap();
    let mb = data.minibatch_ordered(0, 2, &dev).unwrap();
    let init = Tensor::zeros((2, K), DType::F32, &dev).unwrap();

    let (_, sd) =
        fit_theta_posterior(&dec, &mb, &init, Track::Mature, &ThetaFitConfig::default()).unwrap();

    let sd_rows: Vec<Vec<f32>> = sd.to_vec2().unwrap();
    let mean_of = |r: &Vec<f32>| r.iter().sum::<f32>() / r.len() as f32;
    let (pinned, floating) = (mean_of(&sd_rows[0]), mean_of(&sd_rows[1]));
    assert!(
        floating > pinned,
        "a cell with no counts must read wider than one pinned by counts, \
         got floating={floating:.4} vs pinned={pinned:.4}"
    );
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
    let fitted =
        fit_theta_to_track(&dec, &mb, &init, Track::Nascent, &ThetaFitConfig::default()).unwrap();
    let th: Vec<Vec<f32>> = fitted.exp().unwrap().to_vec2().unwrap();
    for (i, row) in th.iter().enumerate() {
        assert!(
            row.iter().all(|v| v.is_finite()),
            "row {i} has non-finite θ"
        );
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "row {i} sums to {s}");
    }
}
