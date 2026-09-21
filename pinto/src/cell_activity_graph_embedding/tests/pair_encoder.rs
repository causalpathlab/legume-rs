//! The encoder is checked against the exact solver it amortizes: on planted
//! data its pair placement and its self-pair (the cell placement) land where
//! the Newton solve lands, it is symmetric end to end, a saved encoder
//! reproduces itself, and empty rows stay at the origin.

use super::fixture::*;
use crate::cell_activity_graph_embedding::pair_projection::encoder::{
    CellRow, PairEncoder, PairEncoderSpec,
};
use crate::cell_activity_graph_embedding::pair_projection::{
    project_pairs, PairDictionary, PairProjectionArgs, PairSolver,
};
use crate::util::common::*;
use crate::util::feature_axis::FeatureAxis;
use legume_numeric::candle::candle_core::Device;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N_TRAIN_CELLS: usize = 150;
const RIDGE: f32 = 1.0;

/// A planted cell: its exact expected counts and the latent they came from.
type PlantedCell = (Vec<(u32, f32)>, Vec<f32>);

/// Cells whose latents sit near one of three centres, with their exact
/// expected counts, as `(global profile, planted latent)`.
fn planted_cells(n: usize, seed: u64) -> Vec<PlantedCell> {
    let e = dictionary_matrix();
    let (b, _) = abundances();
    let centres = [
        [0.7f32, -0.3, 0.2, 0.0],
        [-0.4, 0.5, 0.0, 0.3],
        [0.1, 0.1, -0.6, -0.4],
    ];
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|c| {
            let centre = centres[c % 3];
            let theta: Vec<f32> = centre
                .iter()
                .map(|&v| v + 0.15 * (rng.random::<f32>() - 0.5))
                .collect();
            let beta = 0.5 * (rng.random::<f32>() - 0.5);
            (counts_from(&e, &b, &theta, beta), theta)
        })
        .collect()
}

fn dictionary() -> PairDictionary {
    let e = dictionary_matrix();
    let (_, totals) = abundances();
    PairDictionary::new(&e, &totals, N_CELLS).expect("dictionary")
}

fn corpus_of(dict: &PairDictionary, cells: &[PlantedCell]) -> Vec<CellRow> {
    cells
        .iter()
        .map(|(profile, _)| CellRow::from_profile(dict, profile))
        .collect()
}

fn random_pairs(n_cells: usize, n_pairs: usize, seed: u64) -> Vec<(u32, u32)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_pairs)
        .map(|_| {
            let u = rng.random_range(0..n_cells as u32);
            let mut v = rng.random_range(0..n_cells as u32);
            while v == u {
                v = rng.random_range(0..n_cells as u32);
            }
            (u.min(v), u.max(v))
        })
        .collect()
}

fn spec(epochs: usize) -> PairEncoderSpec {
    PairEncoderSpec {
        trunk_width: 32,
        n_experts: 4,
        epochs,
        batch: 128,
        ridge: RIDGE,
    }
}

fn exact(dict: &PairDictionary, obs: &[(u32, f32)]) -> (Vec<f32>, f32) {
    let (theta, beta, _) = dict.solve(obs, RIDGE);
    (theta, beta)
}

/// A trained encoder on the planted cells, with the training pairs.
struct Trained {
    dict: PairDictionary,
    cells: Vec<PlantedCell>,
    corpus: Vec<CellRow>,
    edges: Vec<(u32, u32)>,
    enc: PairEncoder,
}

/// Trained once for every test that reads it; the encoder is never mutated
/// after training.
fn trained() -> &'static Trained {
    static TRAINED: std::sync::OnceLock<Trained> = std::sync::OnceLock::new();
    TRAINED.get_or_init(|| {
        let dict = dictionary();
        let cells = planted_cells(N_TRAIN_CELLS, 1);
        let corpus = corpus_of(&dict, &cells);
        let edges = random_pairs(N_TRAIN_CELLS, 1500, 2);
        let enc = PairEncoder::build(&dict, &corpus, &spec(20), 7, &Device::Cpu).unwrap();
        enc.train(&corpus, &edges, &spec(20), 7).unwrap();
        Trained {
            dict,
            cells,
            corpus,
            edges,
            enc,
        }
    })
}

#[test]
fn encoder_recovers_the_exact_solvers_pair_latent() {
    let Trained {
        dict,
        cells,
        corpus,
        enc,
        ..
    } = trained();
    // Pairs the encoder never trained on, over the same cells.
    let held_out = random_pairs(N_TRAIN_CELLS, 200, 99);
    let out = enc.encode_all(corpus, &held_out, 64, 64).unwrap();
    let (mut cos_sum, mut nll_enc, mut nll_exact) = (0f32, 0f64, 0f64);
    for (i, &(u, v)) in held_out.iter().enumerate() {
        let obs = pooled(&cells[u as usize].0, &cells[v as usize].0);
        let (theta, _) = exact(dict, &obs);
        let z: Vec<f32> = out.pairs.latent.row(i).iter().copied().collect();
        cos_sum += cosine(&z, &theta);
        // Every feature is active on this fixture, so global ids are positions.
        nll_enc += f64::from(dict.nll(&obs, &z));
        nll_exact += f64::from(dict.nll(&obs, &theta));
    }
    let mean_cos = cos_sum / held_out.len() as f32;
    let ratio = nll_enc / nll_exact;
    eprintln!("pair encoder vs exact: mean cosine {mean_cos:.4}, NLL ratio {ratio:.5}");
    assert!(mean_cos > 0.95, "mean cosine to the exact MAP {mean_cos}");
    assert!(ratio < 1.02, "encoder NLL is {ratio} of the exact one");
}

#[test]
fn self_pair_recovers_the_cells_placement() {
    let Trained {
        dict,
        cells,
        corpus,
        edges,
        enc,
    } = trained();
    let out = enc.encode_all(corpus, &edges[..1], 64, 64).unwrap();
    let mut cos_sum = 0f32;
    let mut beta_gap = 0f32;
    for (c, (profile, _)) in cells.iter().enumerate() {
        let doubled: Vec<(u32, f32)> = profile.iter().map(|&(g, n)| (g, 2.0 * n)).collect();
        let (theta, beta) = exact(dict, &doubled);
        let z: Vec<f32> = out.cells.latent.row(c).iter().copied().collect();
        cos_sum += cosine(&z, &theta);
        // The oracle's intercept is for the doubled depth; the cell's own is
        // `ln 2` below it.
        beta_gap += (out.cells.bias[c] - (beta - 2.0f32.ln())).abs();
    }
    let mean_cos = cos_sum / cells.len() as f32;
    let mean_gap = beta_gap / cells.len() as f32;
    eprintln!("self-pair vs exact: mean cosine {mean_cos:.4}, intercept gap {mean_gap:.4}");
    assert!(
        mean_cos > 0.95,
        "mean cosine of the self-pair to the exact MAP {mean_cos}"
    );
    assert!(
        mean_gap < 0.1,
        "cell intercepts off by {mean_gap} on average"
    );
}

#[test]
fn pair_code_is_symmetric_end_to_end() {
    let Trained {
        corpus, edges, enc, ..
    } = trained();
    let flipped: Vec<(u32, u32)> = edges.iter().map(|&(u, v)| (v, u)).collect();
    let a = enc.encode_all(corpus, edges, 100, 64).unwrap();
    let b = enc.encode_all(corpus, &flipped, 100, 64).unwrap();
    assert_eq!(a.pairs.latent.as_slice(), b.pairs.latent.as_slice());
    assert_eq!(a.pairs.bias, b.pairs.bias);
}

#[test]
fn blocks_do_not_move_a_cell() {
    let Trained {
        corpus, edges, enc, ..
    } = trained();
    let whole = enc
        .encode_all(corpus, &edges[..1], 64, corpus.len())
        .unwrap();
    let thirds = enc
        .encode_all(corpus, &edges[..1], 64, corpus.len() / 3)
        .unwrap();
    for c in 0..corpus.len() {
        let a: Vec<f32> = whole.cells.latent.row(c).iter().copied().collect();
        let b: Vec<f32> = thirds.cells.latent.row(c).iter().copied().collect();
        assert!(cosine(&a, &b) > 0.999, "cell {c} moved with the block size");
    }
}

#[test]
fn empty_endpoints_land_on_the_origin() {
    let Trained {
        dict,
        corpus,
        edges,
        enc,
        ..
    } = trained();
    // Two cells with no counts at all.
    let mut corpus = corpus.clone();
    corpus.push(CellRow::from_profile(dict, &[]));
    corpus.push(CellRow::from_profile(dict, &[]));
    let n = corpus.len() as u32;
    let probe = vec![(n - 2, n - 1), edges[0]];
    let out = enc.encode_all(&corpus, &probe, 64, 64).unwrap();
    assert!(out.pairs.latent.row(0).iter().all(|&v| v == 0.0));
    assert_eq!(out.pairs.bias[0], 0.0);
    assert!(out.pairs.latent.row(1).iter().any(|&v| v != 0.0));
    for c in [n - 2, n - 1] {
        assert!(out.cells.latent.row(c as usize).iter().all(|&v| v == 0.0));
        assert_eq!(out.cells.bias[c as usize], 0.0);
    }
}

#[test]
fn save_load_round_trip_is_byte_identical() {
    let Trained {
        dict,
        corpus,
        edges,
        enc,
        ..
    } = trained();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("enc.safetensors");
    let path = path.to_str().unwrap();
    enc.save(path).unwrap();
    let back = PairEncoder::load(dict, path, &Device::Cpu).unwrap();
    assert_eq!(back.trunk_width(), 32);
    assert_eq!(back.n_experts(), 4);
    let a = enc.encode_all(corpus, edges, 100, 64).unwrap();
    let b = back.encode_all(corpus, edges, 100, 64).unwrap();
    assert_eq!(a.pairs.latent.as_slice(), b.pairs.latent.as_slice());
    assert_eq!(a.cells.latent.as_slice(), b.cells.latent.as_slice());
    assert_eq!(a.pairs.bias, b.pairs.bias);
    assert_eq!(a.cells.bias, b.cells.bias);
}

/// The pooled global profile of two cells.
fn pooled(a: &[(u32, f32)], b: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let mut dense = vec![0f32; N_FEATURES];
    for &(g, n) in a.iter().chain(b) {
        dense[g as usize] += n;
    }
    dense
        .into_iter()
        .enumerate()
        .filter(|&(_, n)| n > 0.0)
        .map(|(g, n)| (g as u32, n))
        .collect()
}

////////////////////////////////////////
// Through `project_pairs`, from disk //
////////////////////////////////////////

/// A sparse matrix of the planted cells, one row per feature.
fn planted_data(dir: &tempfile::TempDir, cells: &[PlantedCell]) -> anyhow::Result<SparseIoVec> {
    let triplets: Vec<(u64, u64, f32)> = cells
        .iter()
        .enumerate()
        .flat_map(|(c, (profile, _))| profile.iter().map(move |&(g, n)| (g as u64, c as u64, n)))
        .collect();
    let path = dir.path().join("planted.zarr");
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (N_FEATURES, cells.len(), triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )?;
    let rows: Vec<Box<str>> = (0..N_FEATURES).map(|g| format!("G{g}").into()).collect();
    backend.register_row_names_vec(&rows);
    let names: Vec<Box<str>> = (0..cells.len()).map(|c| format!("c{c}").into()).collect();
    backend.register_column_names_vec(&names);
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(backend), None)?;
    Ok(v)
}

#[test]
fn the_saved_encoder_reproduces_the_runs_pairs_and_cells() {
    let dir = tempfile::tempdir().unwrap();
    let cells = planted_cells(60, 5);
    let data = planted_data(&dir, &cells).unwrap();
    let e = dictionary_matrix();
    let (_, totals) = abundances();
    let rows: Vec<Box<str>> = (0..N_FEATURES).map(|g| format!("G{g}").into()).collect();
    let axis = FeatureAxis::resolve_or_identity(&rows).unwrap();
    let edges = random_pairs(60, 300, 3);
    let saved = dir.path().join("run.pair_encoder.safetensors");
    let saved = saved.to_str().unwrap();
    let spec = spec(15);
    let fitted = project_pairs(
        &data,
        &edges,
        &e,
        None,
        &PairProjectionArgs {
            solver: PairSolver::TrainEncoder {
                spec: &spec,
                dev: &Device::Cpu,
                save_to: saved,
            },
            seed: 11,
            pair_block: 64,
            eval_features: None,
            score_pairs: true,
        },
        &axis,
        &totals,
    )
    .unwrap();
    assert_eq!(fitted.cells.latent.nrows(), 60);
    assert_eq!(fitted.scores.len(), edges.len());
    assert!(std::path::Path::new(saved).is_file());

    let loaded = project_pairs(
        &data,
        &edges,
        &e,
        None,
        &PairProjectionArgs {
            solver: PairSolver::LoadEncoder {
                path: saved,
                dev: &Device::Cpu,
            },
            seed: 11,
            pair_block: 64,
            eval_features: None,
            score_pairs: false,
        },
        &axis,
        &totals,
    )
    .unwrap();
    assert_eq!(fitted.latent.as_slice(), loaded.latent.as_slice());
    assert_eq!(
        fitted.cells.latent.as_slice(),
        loaded.cells.latent.as_slice()
    );
    assert!(loaded.scores.is_empty());
}

#[test]
fn the_certificate_finishes_what_an_untrained_encoder_gets_wrong() {
    // An encoder that never trained places pairs from a random map. The
    // certificate bounds every placement's excess over the optimum, the rows
    // it puts far out are finished exactly, and so the result as a whole
    // must sit within a few nats per pair of the optimum regardless.
    let dir = tempfile::tempdir().unwrap();
    let cells = planted_cells(60, 21);
    let data = planted_data(&dir, &cells).unwrap();
    let e = dictionary_matrix();
    let (_, totals) = abundances();
    let rows: Vec<Box<str>> = (0..N_FEATURES).map(|g| format!("G{g}").into()).collect();
    let axis = FeatureAxis::resolve_or_identity(&rows).unwrap();
    let edges = random_pairs(60, 300, 4);
    let saved = dir.path().join("untrained.pair_encoder.safetensors");
    let spec = spec(0);
    let out = project_pairs(
        &data,
        &edges,
        &e,
        None,
        &PairProjectionArgs {
            solver: PairSolver::TrainEncoder {
                spec: &spec,
                dev: &Device::Cpu,
                save_to: saved.to_str().unwrap(),
            },
            seed: 5,
            pair_block: 64,
            eval_features: None,
            score_pairs: false,
        },
        &axis,
        &totals,
    )
    .unwrap();
    let dict = PairDictionary::new(&e, &totals, 60).unwrap();
    let (mut got, mut best) = (0f64, 0f64);
    for (i, &(u, v)) in edges.iter().enumerate() {
        let obs = pooled(&cells[u as usize].0, &cells[v as usize].0);
        let (theta, _) = exact(&dict, &obs);
        let z: Vec<f32> = out.latent.row(i).iter().copied().collect();
        got += f64::from(dict.nll(&obs, &z));
        best += f64::from(dict.nll(&obs, &theta));
    }
    let ratio = got / best;
    assert!(
        ratio < 1.005,
        "placements sit {ratio} of the optimum's likelihood"
    );
}
