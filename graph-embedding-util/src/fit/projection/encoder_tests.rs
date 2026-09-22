//! The distilled encoder path: the sub-pseudobulk aggregate, the seeded
//! hold-out, the exact conditional intercept, and that the distillation fits a
//! planted target on held-out pseudobulks.

use super::*;
use legume_numeric::candle::candle_core::Device;

fn aggregate_rows(rows: &[FoldedRow], members: &[usize], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; d];
    aggregate_into(rows, members, &mut out);
    out
}

const D: usize = 12;
const H: usize = 3;

fn rows(n: usize) -> Vec<FoldedRow> {
    (0..n)
        .map(|i| {
            let feats: Vec<u32> = (0..D as u32)
                .filter(|g| !(g + i as u32).is_multiple_of(3))
                .collect();
            let counts: Vec<f32> = feats
                .iter()
                .map(|&g| (1 + (g as usize * 7 + i * 3) % 5) as f32)
                .collect();
            FoldedRow::new(feats, counts)
        })
        .collect()
}

#[test]
fn the_aggregate_is_the_mean_of_the_members_folded_counts() {
    let r = rows(5);
    let out = aggregate_rows(&r, &[1, 3, 4], D);
    for (g, &got) in out.iter().enumerate() {
        let want: f32 = [1usize, 3, 4]
            .iter()
            .map(|&i| {
                r[i].feats
                    .iter()
                    .zip(&r[i].counts)
                    .find(|(&f, _)| f as usize == g)
                    .map_or(0.0, |(_, &c)| c)
            })
            .sum::<f32>()
            / 3.0;
        assert!((got - want).abs() < 1e-6, "gene {g}: {got} vs {want}");
    }
    assert_eq!(aggregate_rows(&r, &[], D), vec![0.0; D]);
}

#[test]
fn the_holdout_is_seeded_disjoint_and_a_tenth() {
    let (train, held) = split_holdout(40, 0.1, 7);
    assert_eq!(held.len(), 4);
    assert_eq!(train.len(), 36);
    assert!(train.iter().all(|p| !held.contains(p)));
    assert_eq!(split_holdout(40, 0.1, 7), (train.clone(), held.clone()));
    assert_ne!(split_holdout(40, 0.1, 8).1, held);
    // Too few to hold any out: everything trains.
    let (t, h) = split_holdout(5, 0.1, 1);
    assert!(h.is_empty() && t.len() == 5);
}

/// `c = ln(total) − logsumexp_f(θ·e_f + b_f)` makes the expected total count
/// under the cell's rates equal the observed total, exactly.
#[test]
fn the_intercept_matches_the_total_count() {
    let dev = Device::Cpu;
    let feat: Vec<f32> = (0..D * H)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.05)
        .collect();
    let b: Vec<f32> = (0..D).map(|g| -((g + 1) as f32).ln()).collect();
    let dict = FrozenDict::new(&feat, &b, H, &dev).unwrap();
    let theta = Tensor::from_vec(vec![0.3f32, -0.2, 0.5, -0.4, 0.1, 0.2], (2, H), &dev).unwrap();
    let totals = [37.0f32, 1200.0];
    let c = null_intercept(&dict, &theta, &totals).unwrap();
    let th: Vec<Vec<f32>> = theta.to_vec2().unwrap();
    for n in 0..2 {
        let expected_total: f64 = (0..D)
            .map(|g| {
                let s: f32 = (0..H).map(|k| th[n][k] * feat[g * H + k]).sum::<f32>() + b[g] + c[n];
                f64::from(s).exp()
            })
            .sum();
        assert!(
            ((expected_total - f64::from(totals[n])) / f64::from(totals[n])).abs() < 1e-4,
            "row {n}: {expected_total} vs {}",
            totals[n]
        );
    }
}

/// Planted: every pseudobulk's target row is a fixed linear map of its member
/// cells' mean composition. The distilled encoder must predict held-out
/// pseudobulks far better than the zero predictor.
#[test]
fn the_distillation_fits_a_planted_target_on_held_out_pseudobulks() {
    let dev = Device::Cpu;
    // Enough pseudobulks for several steps per pass at the production budget.
    let n_pb = 600;
    let per_pb = 5;
    let n_cells = n_pb * per_pb;
    // Cells of pseudobulk p share a composition profile that varies with p.
    let mut folded = Vec::with_capacity(n_cells);
    let mut cell_to_pb = Vec::with_capacity(n_cells);
    for p in 0..n_pb {
        for c in 0..per_pb {
            let feats: Vec<u32> = (0..D as u32).collect();
            let counts: Vec<f32> = (0..D)
                .map(|g| {
                    let base = 3.0 + 2.5 * ((p as f32 * 0.37 + g as f32 * 0.9).sin());
                    let jitter = 1.0 + 0.3 * (((c * 5 + g * 3 + p) % 7) as f32 / 7.0 - 0.5);
                    (base * jitter).max(0.0).round()
                })
                .collect();
            folded.push(FoldedRow::new(feats, counts));
            cell_to_pb.push(p);
        }
    }
    // A fixed [H, D] map of the composition.
    let a: Vec<f32> = (0..H * D)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.3)
        .collect();
    let mut e_pb = nalgebra::DMatrix::<f32>::zeros(n_pb, H);
    for p in 0..n_pb {
        let members: Vec<usize> = (0..n_cells).filter(|&i| cell_to_pb[i] == p).collect();
        let row = aggregate_rows(&folded, &members, D);
        let z: f32 = row.iter().sum::<f32>().max(1e-6);
        for k in 0..H {
            e_pb[(p, k)] = (0..D).map(|g| a[k * D + g] * row[g] / z).sum::<f32>() * 10.0;
        }
    }
    let feat: Vec<f32> = (0..D * H)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
        .collect();
    let b = vec![0f32; D];
    let dict = FrozenDict::new(&feat, &b, H, &dev).unwrap();
    let mean = gene_mean(&folded, D);
    let groups = members_by_pb(&cell_to_pb, &(0..n_cells as u32).collect::<Vec<_>>(), n_pb);
    let levels = vec![DistillTargets {
        e_pb: &e_pb,
        groups: &groups,
    }];
    let (encoder, report) = distill(dict, &mean, &levels, &folded, 3, &dev).unwrap();
    assert!(report.held_out_mse.is_finite());
    assert!(
        report.held_out_mse < 0.5 * report.held_out_target_var,
        "held-out MSE {} vs target variance {}",
        report.held_out_mse,
        report.held_out_target_var
    );
    assert!(
        report.held_out_cosine > 0.8,
        "cosine {}",
        report.held_out_cosine
    );

    // The likelihood refinement starts from the distilled map and lowers the
    // cells' own per-count NLL — the objective it trains is the one reported.
    // Forced by `refine`'s signature only: it now takes the fit's encoder SET
    // and one dictionary per track, which on this one-track axis is this trunk
    // and one dictionary. No assertion below changed.
    let encs = CellEncoders::new(
        vec![TrackEncoder {
            track: 0,
            name: "t0".into(),
            rows: (0..D as u32).collect(),
            encoder,
        }],
        D,
    );
    let dicts = vec![FrozenDict::new(&feat, &b, H, &dev).unwrap()];
    let refined = refine(
        &encs,
        &dicts,
        &[std::borrow::Cow::Borrowed(folded.as_slice())],
        1.0,
        3,
        &dev,
    )
    .unwrap();
    let encoder = &encs.iter()[0].encoder;
    assert_eq!(refined.n_cells, n_cells);
    assert!(refined.nll_per_count_before.is_finite() && refined.nll_per_count_after.is_finite());
    assert!(
        refined.nll_per_count_after < refined.nll_per_count_before,
        "NLL/count {} → {}",
        refined.nll_per_count_before,
        refined.nll_per_count_after
    );

    // Saved and reloaded on the same dictionary, the trunk places the same
    // rows at the same points: one estimator, both halves.
    let path = std::env::temp_dir().join(format!(
        "cell_enc_roundtrip_{}.safetensors",
        std::process::id()
    ));
    let path = path.to_string_lossy().to_string();
    encoder.save(&path).unwrap();
    let again = CellEncoder::load(&feat, &b, H, &path, &dev).unwrap();
    assert_eq!(
        again.feature_mean().unwrap(),
        encoder.feature_mean().unwrap()
    );
    std::fs::remove_file(&path).ok();
    let nodes: Vec<(u32, &[u32], &[f32])> = folded[..7]
        .iter()
        .enumerate()
        .map(|(i, r)| (i as u32, r.feats.as_slice(), r.counts.as_slice()))
        .collect();
    let a = encoder.encode_edges(&nodes).unwrap();
    let b2 = again.encode_edges(&nodes).unwrap();
    assert_eq!(a.theta.len(), 7 * H);
    for (x, y) in a.theta.iter().zip(&b2.theta) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
    for (x, y) in a.b_node.iter().zip(&b2.b_node) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
    // And the encoded rows are not all alike: the map is informative.
    let first: Vec<f32> = a.theta[..H].to_vec();
    assert!(a.theta[H..]
        .chunks(H)
        .any(|r| r.iter().zip(&first).any(|(p, q)| (p - q).abs() > 1e-3)));

    // The gauge shift folds into the head exactly: every placement moves by
    // −shift and nothing else, and it survives the save/load round trip.
    let shift = vec![0.5f32, -1.25, 2.0];
    encoder.shift_output(&shift).unwrap();
    let shifted = encoder.encode_edges(&nodes).unwrap();
    for (row, orig) in shifted.theta.chunks(H).zip(a.theta.chunks(H)) {
        for k in 0..H {
            assert!(
                (row[k] - (orig[k] - shift[k])).abs() < 1e-5,
                "{} vs {}",
                row[k],
                orig[k] - shift[k]
            );
        }
    }
    encoder.save(&path).unwrap();
    let again = CellEncoder::load(&feat, &b, H, &path, &dev).unwrap();
    std::fs::remove_file(&path).ok();
    let reloaded = again.encode_edges(&nodes).unwrap();
    for (x, y) in reloaded.theta.iter().zip(&shifted.theta) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
}

////////////////////////////////////
// The one-track fixture (parity) //
////////////////////////////////////

/// Owned pieces of the one-track `project_cells` fixture, so the `(id, feats,
/// counts)` tuples can borrow them.
struct OneTrackFixture {
    feat: Vec<f32>,
    b_feat: Vec<f32>,
    edges: Vec<(Vec<u32>, Vec<f32>)>,
    cell_to_pb: Vec<usize>,
    e_pb: nalgebra::DMatrix<f32>,
}

const FIX_PB: usize = 12;
const FIX_PER_PB: usize = 4;
const FIX_CELLS: usize = FIX_PB * FIX_PER_PB;

/// A planted one-track fixture: `FIX_PB` pseudobulks of `FIX_PER_PB` cells each,
/// every cell dense over `D` features, and a pseudobulk table that is a fixed
/// linear map of the members' mean composition.
fn one_track_fixture() -> OneTrackFixture {
    let mut edges = Vec::with_capacity(FIX_CELLS);
    let mut cell_to_pb = Vec::with_capacity(FIX_CELLS);
    for p in 0..FIX_PB {
        for c in 0..FIX_PER_PB {
            let feats: Vec<u32> = (0..D as u32)
                .filter(|g| !(*g as usize + c).is_multiple_of(5))
                .collect();
            let counts: Vec<f32> = feats
                .iter()
                .map(|&g| {
                    let base = 1.5 + 0.9 * ((p as f32 * 0.41 + g as f32 * 0.77).sin());
                    let jitter =
                        1.0 + 0.25 * (((c * 3 + g as usize * 5 + p) % 7) as f32 / 7.0 - 0.5);
                    (base * jitter).max(1.0).round()
                })
                .collect();
            edges.push((feats, counts));
            cell_to_pb.push(p);
        }
    }
    let folded: Vec<FoldedRow> = edges
        .iter()
        .map(|(f, c)| FoldedRow::new(f.clone(), c.clone()))
        .collect();
    let a: Vec<f32> = (0..H * D)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1)
        .collect();
    let mut e_pb = nalgebra::DMatrix::<f32>::zeros(FIX_PB, H);
    for p in 0..FIX_PB {
        let members: Vec<usize> = (0..FIX_CELLS).filter(|&i| cell_to_pb[i] == p).collect();
        let row = aggregate_rows(&folded, &members, D);
        let z: f32 = row.iter().sum::<f32>().max(1e-6);
        for k in 0..H {
            e_pb[(p, k)] = (0..D).map(|g| a[k * D + g] * row[g] / z).sum::<f32>();
        }
    }
    OneTrackFixture {
        feat: (0..D * H)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.2)
            .collect(),
        b_feat: (0..D).map(|g| -((g + 1) as f32).ln() * 0.1).collect(),
        edges,
        cell_to_pb,
        e_pb,
    }
}

/// `project_cells` on the one-track fixture, returning `(θ, b_cell)`.
fn one_track_project(fx: &OneTrackFixture) -> (Vec<f32>, Vec<f32>) {
    let dev = Device::Cpu;
    let cells: Vec<(u32, &[u32], &[f32])> = fx
        .edges
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let input = Phase2Input {
        feat: &fx.feat,
        b_feat: &fx.b_feat,
        h: H,
        n_cells: FIX_CELLS,
        lambda: 1.0,
        dev: &dev,
        label: "Phase 2",
        gauge_fix: true,
    };
    let levels = vec![DistillLevel {
        e_pb: &fx.e_pb,
        cell_to_pb: &fx.cell_to_pb,
    }];
    let spec = DistillSpec {
        levels: &levels,
        seed: 20_260_914,
    };
    let (out, encs) =
        project_cells(&input, &cells, None, &spec, &TrackSpec::base(D), None).unwrap();
    // One count track, so exactly one encoder — the file `senna bge` persists.
    assert_eq!(encs.iter().len(), 1);
    assert!(encs.single().is_some());
    (out.theta, out.b_cell)
}

/// The parity guard for the one-track (`senna bge`) path.
///
/// The numbers below were taken from `project_cells` on this same fixture —
/// the encoder's placement with the intercept exact at it, no per-cell solve
/// after it — as the mean of six runs, one process each.
///
/// The bar is 5e-4, not exact: the global gradient-norm clip sums the
/// per-parameter squares in `GradStore`'s hash-map order, which differs between
/// processes, so the clip factor — and through the refinement, the answer — can
/// move run to run *within one build*. The six runs behind the snapshot agreed
/// far inside the bar, and re-seeding track 0 off `spec.seed` moves θ[0] by
/// well over it.
#[test]
fn one_track_project_cells_matches_the_previous_output() {
    #[rustfmt::skip]
    const THETA: [f32; FIX_CELLS * H] = [
        0.203109, -0.107541, -0.157314, 0.058433, -0.086704, -0.152246,
        -0.045752, 0.085753, 0.015765, -0.041326, 0.176028, -0.036660,
        0.267939, -0.087062, -0.182925, -0.042972, -0.019994, -0.176514,
        -0.092915, 0.123936, -0.005518, -0.110193, 0.279293, 0.002435,
        0.265573, -0.143911, -0.158012, -0.065789, -0.076130, -0.149596,
        0.020219, 0.033865, -0.057489, -0.110193, 0.279293, 0.002435,
        0.194284, -0.131810, -0.127327, -0.076847, -0.010759, -0.093634,
        -0.038931, 0.031786, -0.009169, -0.077517, 0.295738, 0.015932,
        0.085314, -0.109521, -0.066885, -0.051909, 0.071052, -0.047744,
        -0.032488, 0.015211, 0.018998, -0.070097, 0.266701, 0.016663,
        0.085314, -0.109521, -0.066885, -0.057143, 0.030019, -0.030547,
        -0.063254, -0.002902, 0.119968, -0.030610, 0.238321, 0.028054,
        0.042879, -0.045137, -0.017436, -0.045018, -0.025343, 0.031198,
        -0.089959, -0.099387, 0.170223, -0.023186, 0.219146, 0.025290,
        0.043764, -0.066149, -0.046222, -0.045018, -0.025343, 0.031198,
        -0.092799, -0.135982, 0.186704, -0.068617, 0.196431, 0.028556,
        0.043764, -0.066149, -0.046222, -0.006413, -0.065482, 0.052580,
        -0.022780, -0.145474, 0.204507, -0.042115, 0.042647, 0.124450,
        0.081346, -0.078643, -0.011180, -0.002283, -0.054526, 0.030870,
        -0.022780, -0.145474, 0.204507, -0.014857, -0.005274, 0.159811,
        0.120790, -0.088061, -0.165305, -0.037615, -0.076657, 0.019209,
        -0.003736, -0.126952, 0.181781, -0.005667, -0.014257, 0.058296,
        0.085639, -0.045470, -0.133344, -0.021665, -0.069696, -0.042657,
        -0.040255, -0.105655, 0.193101, -0.005667, -0.014257, 0.058296,
    ];
    #[rustfmt::skip]
    const B_CELL: [f32; FIX_CELLS] = [
        0.375882, 0.442689, 0.501585, 0.364092, 0.433011, 0.492288,
        0.430064, 0.467982, 0.302100, 0.553641, 0.445128, 0.467982,
        0.379358, 0.496944, 0.443856, 0.347053, 0.317504, 0.372328,
        0.507619, 0.352778, 0.317504, 0.440672, 0.383090, 0.363420,
        0.244760, 0.383526, 0.311893, 0.431103, 0.163324, 0.383526,
        0.379372, 0.294358, 0.163324, 0.319245, 0.314993, 0.240584,
        0.166783, 0.244607, 0.314993, 0.318938, 0.154058, 0.316019,
        0.317954, 0.387527, 0.236552, 0.314083, 0.315606, 0.387527,
    ];
    let (theta, b_cell) = one_track_project(&one_track_fixture());
    assert_eq!(theta.len(), THETA.len());
    assert_eq!(b_cell.len(), B_CELL.len());
    for (i, (&got, &want)) in theta.iter().zip(&THETA).enumerate() {
        assert!(
            (got - want).abs() < 5e-4,
            "θ[{i}] drifted from the pre-change output: {got} vs {want}"
        );
    }
    for (i, (&got, &want)) in b_cell.iter().zip(&B_CELL).enumerate() {
        assert!(
            (got - want).abs() < 5e-4,
            "b_cell[{i}] drifted from the pre-change output: {got} vs {want}"
        );
    }
}

////////////////////////
// Per-track plumbing //
////////////////////////

/// A count track over rows `0..6` and another over `6..12`, both naming the
/// same six genes.
fn two_track_spec() -> TrackSpec {
    TrackSpec {
        track_of_row: (0..D).map(|r| (r / 6) as u32).collect(),
        gene_of_row: (0..D).map(|r| (r % 6) as u32).collect(),
        tracks: vec![
            crate::fit::config::TrackInfo {
                name: "t0".into(),
                is_count: true,
            },
            crate::fit::config::TrackInfo {
                name: "m1/t1".into(),
                is_count: true,
            },
        ],
    }
}

/// Two count tracks over rows `0..4` / `4..8` and a NON-count track over
/// `8..12`.
fn three_track_spec() -> TrackSpec {
    TrackSpec {
        track_of_row: (0..D).map(|r| (r / 4) as u32).collect(),
        gene_of_row: (0..D).map(|r| (r % 4) as u32).collect(),
        tracks: vec![
            crate::fit::config::TrackInfo {
                name: "t0".into(),
                is_count: true,
            },
            crate::fit::config::TrackInfo {
                name: "t1".into(),
                is_count: true,
            },
            crate::fit::config::TrackInfo {
                name: "t2".into(),
                is_count: false,
            },
        ],
    }
}

fn dict_feat() -> (Vec<f32>, Vec<f32>) {
    (
        (0..D * H)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1)
            .collect(),
        (0..D).map(|g| -((g + 1) as f32).ln() * 0.1).collect(),
    )
}

#[test]
fn for_rows_over_all_rows_equals_new() {
    let dev = Device::Cpu;
    let (feat, b) = dict_feat();
    let all: Vec<u32> = (0..D as u32).collect();
    let a = FrozenDict::new(&feat, &b, H, &dev).unwrap();
    let c = FrozenDict::for_rows(&feat, &b, H, &all, &dev).unwrap();
    assert_eq!(a.d(), D);
    assert_eq!(c.d(), D);
    assert_eq!(
        a.e_hd.to_vec2::<f32>().unwrap(),
        c.e_hd.to_vec2::<f32>().unwrap()
    );
    assert_eq!(
        a.b_1d.to_vec2::<f32>().unwrap(),
        c.b_1d.to_vec2::<f32>().unwrap()
    );
    // A subset takes exactly those rows, in the order given.
    let some = FrozenDict::for_rows(&feat, &b, H, &[7, 2], &dev).unwrap();
    assert_eq!(some.d(), 2);
    let e = some.e_hd.to_vec2::<f32>().unwrap();
    for (k, col) in e.iter().enumerate() {
        assert_eq!(col[0], feat[7 * H + k]);
        assert_eq!(col[1], feat[2 * H + k]);
    }
    assert_eq!(some.b_1d.to_vec2::<f32>().unwrap()[0], vec![b[7], b[2]]);
}

#[test]
fn split_rows_by_track_relabels_to_local_ids_and_borrows_for_one_track() {
    let r = rows(4);
    // One track: the whole axis, borrowed — nothing is copied at T = 1.
    let one = split_rows_by_track(&r, &TrackSpec::base(D));
    assert_eq!(one.len(), 1);
    assert!(matches!(one[0], std::borrow::Cow::Borrowed(_)));
    for (a, b) in one[0].iter().zip(&r) {
        assert_eq!(a.feats, b.feats);
        assert_eq!(a.counts, b.counts);
    }
    // Two tracks: each row keeps its own track's edges, relabelled to `0..6`.
    let spec = two_track_spec();
    spec.validate(D).unwrap();
    let two = split_rows_by_track(&r, &spec);
    assert_eq!(two.len(), 2);
    assert!(matches!(two[0], std::borrow::Cow::Owned(_)));
    for (i, row) in r.iter().enumerate() {
        for (t, track) in two.iter().enumerate() {
            let want: Vec<(u32, f32)> = row
                .feats
                .iter()
                .zip(&row.counts)
                .filter(|(&f, _)| (f as usize) / 6 == t)
                .map(|(&f, &c)| (f - 6 * t as u32, c))
                .collect();
            let got: Vec<(u32, f32)> = track[i]
                .feats
                .iter()
                .zip(&track[i].counts)
                .map(|(&f, &c)| (f, c))
                .collect();
            assert_eq!(got, want, "row {i}, track {t}");
            assert!(got.iter().all(|&(f, _)| (f as usize) < 6));
        }
    }
}

/// An encoder whose head is a constant: the weight is zeroed and the bias
/// planted, so its output is `bias` for any input. That makes the combine rule
/// readable off the placements.
fn planted_encoder(rows: &[u32], bias: &[f32], dev: &Device) -> CellEncoder {
    let (feat, b) = dict_feat();
    let dict = FrozenDict::for_rows(&feat, &b, H, rows, dev).unwrap();
    let mean = vec![1f32; rows.len()];
    let enc = CellEncoder::build(dict, &mean, dev).unwrap();
    seed_trunk(&enc.varmap, 11).unwrap();
    {
        let vars = enc.varmap.data().lock().unwrap();
        let w = vars.get("cell_enc.nn.enc.z.mean.weight").unwrap();
        w.set(&w.as_tensor().zeros_like().unwrap()).unwrap();
        let bs = vars.get("cell_enc.nn.enc.z.mean.bias").unwrap();
        bs.set(&Tensor::from_slice(bias, H, dev).unwrap()).unwrap();
    }
    enc
}

fn planted_pair(dev: &Device) -> CellEncoders {
    let spec = two_track_spec();
    CellEncoders::new(
        vec![
            TrackEncoder {
                track: 0,
                name: spec.tracks[0].name.clone(),
                rows: spec.rows_of_track(0),
                encoder: planted_encoder(&spec.rows_of_track(0), &[1.0, 2.0, 3.0], dev),
            },
            TrackEncoder {
                track: 1,
                name: spec.tracks[1].name.clone(),
                rows: spec.rows_of_track(1),
                encoder: planted_encoder(&spec.rows_of_track(1), &[-1.0, 0.0, 5.0], dev),
            },
        ],
        D,
    )
}

/// Nodes: one with counts on both tracks, one on track 0 only, one on track 1
/// only.
fn combine_nodes() -> Vec<(Vec<u32>, Vec<f32>)> {
    vec![
        (vec![0, 1, 7, 8], vec![2.0, 3.0, 4.0, 5.0]),
        (vec![0, 2], vec![6.0, 1.0]),
        (vec![9, 10], vec![3.0, 2.0]),
    ]
}

#[test]
fn two_count_tracks_combine_by_mean_and_an_absent_track_falls_back() {
    let dev = Device::Cpu;
    let encs = planted_pair(&dev);
    assert!(encs.single().is_none());
    let owned = combine_nodes();
    let nodes: Vec<(u32, &[u32], &[f32])> = owned
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let out = encs.encode_edges(&nodes).unwrap();
    let want = [
        [0.0f32, 1.0, 4.0], // both tracks: the mean of the two heads
        [1.0, 2.0, 3.0],    // track 1 absent: track 0's head alone
        [-1.0, 0.0, 5.0],   // track 0 absent: track 1's head alone
    ];
    for (n, row) in out.theta.chunks(H).enumerate() {
        for k in 0..H {
            assert!(
                (row[k] - want[n][k]).abs() < 1e-5,
                "node {n}, dim {k}: {} vs {}",
                row[k],
                want[n][k]
            );
        }
    }
    // The reported intercept is track 0's; the node with no track-0 counts sits
    // at the score clamp's floor.
    assert_eq!(out.b_node.len(), 3);
    assert_eq!(out.b_node[2], -(crate::cell_projection::SCORE_CLAMP as f32));
    assert!(out.b_node[0].is_finite() && out.b_node[1].is_finite());
}

#[test]
fn shift_output_on_every_encoder_shifts_the_combined_placement_by_minus_shift() {
    let dev = Device::Cpu;
    let encs = planted_pair(&dev);
    let owned = combine_nodes();
    let nodes: Vec<(u32, &[u32], &[f32])> = owned
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let before = encs.encode_edges(&nodes).unwrap();
    let shift = [0.5f32, -1.25, 2.0];
    encs.shift_output(&shift).unwrap();
    let after = encs.encode_edges(&nodes).unwrap();
    for (n, (a, b)) in after
        .theta
        .chunks(H)
        .zip(before.theta.chunks(H))
        .enumerate()
    {
        for k in 0..H {
            assert!(
                (a[k] - (b[k] - shift[k])).abs() < 1e-5,
                "node {n}, dim {k}: {} vs {}",
                a[k],
                b[k] - shift[k]
            );
        }
    }
}

#[test]
fn save_and_load_round_trip_per_track() {
    let dev = Device::Cpu;
    let spec = two_track_spec();
    let encs = planted_pair(&dev);
    let dir = std::env::temp_dir();
    let paths: Vec<(u32, String)> = (0..2u32)
        .map(|t| {
            let p = dir
                .join(format!(
                    "cell_enc_track{t}_{}.safetensors",
                    std::process::id()
                ))
                .to_string_lossy()
                .to_string();
            encs.iter()[t as usize].encoder.save(&p).unwrap();
            (t, p)
        })
        .collect();
    let (feat, b) = dict_feat();
    let again = CellEncoders::load(&feat, &b, H, &spec, &paths, &dev).unwrap();
    for (_, p) in &paths {
        std::fs::remove_file(p).ok();
    }
    assert_eq!(again.iter().len(), 2);
    assert_eq!(&*again.iter()[1].name, "m1/t1");
    assert_eq!(again.iter()[1].rows, spec.rows_of_track(1));
    let owned = combine_nodes();
    let nodes: Vec<(u32, &[u32], &[f32])> = owned
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let a = encs.encode_edges(&nodes).unwrap();
    let c = again.encode_edges(&nodes).unwrap();
    for (x, y) in a.theta.iter().zip(&c.theta) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
    for (x, y) in a.b_node.iter().zip(&c.b_node) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
}

#[test]
fn refine_lowers_the_summed_nll_with_a_third_non_count_track_present() {
    let dev = Device::Cpu;
    let spec = three_track_spec();
    spec.validate(D).unwrap();
    let (feat, b) = dict_feat();
    let folded = rows(64);
    let by_track = split_rows_by_track(&folded, &spec);
    assert_eq!(by_track.len(), 3);
    let dicts: Vec<FrozenDict> = (0..3)
        .map(|t| FrozenDict::for_rows(&feat, &b, H, &spec.rows_of_track(t), &dev).unwrap())
        .collect();
    let build = || {
        CellEncoders::new(
            spec.count_tracks()
                .iter()
                .map(|&t| TrackEncoder {
                    track: t as u32,
                    name: spec.tracks[t].name.clone(),
                    rows: spec.rows_of_track(t),
                    encoder: {
                        let rows_t = spec.rows_of_track(t);
                        let dict = FrozenDict::for_rows(&feat, &b, H, &rows_t, &dev).unwrap();
                        let mean = gene_mean(&by_track[t], rows_t.len());
                        let enc = CellEncoder::build(dict, &mean, &dev).unwrap();
                        seed_trunk(&enc.varmap, 5).unwrap();
                        enc
                    },
                })
                .collect(),
            D,
        )
    };
    // `nll_per_count` divides by the counts of every track scored, so the
    // comparisons below are on the NLL itself, reconstructed from the reported
    // per-count value and the counts that went into it.
    let counts_of = |bt: &[Cow<'_, [FoldedRow]>]| -> f64 {
        bt.iter()
            .flat_map(|t| t.iter())
            .flat_map(|r| r.counts.iter())
            .map(|&v| f64::from(v))
            .sum()
    };
    let nll_of = |st: &RefineStats, bt: &[Cow<'_, [FoldedRow]>]| -> f64 {
        f64::from(st.nll_per_count_before) * counts_of(bt)
    };

    // Two count tracks and one that only contributes its likelihood term.
    let all = build();
    assert_eq!(all.iter().len(), 2);
    let three = refine(&all, &dicts, &by_track, 1.0, 9, &dev).unwrap();
    assert_eq!(three.n_cells, 64);
    assert!(
        three.nll_per_count_after < three.nll_per_count_before,
        "NLL/count {} → {}",
        three.nll_per_count_before,
        three.nll_per_count_after
    );

    // The same two count tracks alone.
    let two = refine(&build(), &dicts[..2], &by_track[..2], 1.0, 9, &dev).unwrap();
    let (n3, n2) = (nll_of(&three, &by_track), nll_of(&two, &by_track[..2]));
    assert!(
        n3 > n2 + 1.0,
        "the third track added nothing to the summed NLL: {n3} vs {n2}"
    );

    // …and an ABSENT third track — present in the spec, no counts on it —
    // contributes exactly zero, so it puts the NLL back where two tracks had it.
    let empty: Vec<FoldedRow> = (0..64).map(|_| FoldedRow::new(vec![], vec![])).collect();
    let absent = [by_track[0].clone(), by_track[1].clone(), Cow::Owned(empty)];
    let none = refine(&build(), &dicts, &absent, 1.0, 9, &dev).unwrap();
    let n0 = nll_of(&none, &absent);
    assert!(
        (n0 - n2).abs() < 1e-2,
        "an absent track must contribute exactly 0: {n0} vs {n2}"
    );
}

///////////////////////////////////////////////////
// One COUNT track is not the same as one track  //
///////////////////////////////////////////////////

/// A count track over rows `0..6` and a NON-count track over `6..12`: one
/// encoder, but the axis is wider than the track it reads.
fn count_plus_non_count_spec() -> TrackSpec {
    TrackSpec {
        track_of_row: (0..D).map(|r| (r / 6) as u32).collect(),
        gene_of_row: (0..D).map(|r| (r % 6) as u32).collect(),
        tracks: vec![
            crate::fit::config::TrackInfo {
                name: "t0".into(),
                is_count: true,
            },
            crate::fit::config::TrackInfo {
                name: "t1".into(),
                is_count: false,
            },
        ],
    }
}

/// One encoder is NOT one track. With a non-count track present the single
/// encoder reads only its own `D_0 < F` rows, so the warm start must go through
/// the split rows: handing it the unsplit global rows indexes a `[n, D_0]`
/// block at a global feature id and blows up.
#[test]
fn one_count_track_beside_a_non_count_track_does_not_take_the_one_track_shortcut() {
    let dev = Device::Cpu;
    let fx = one_track_fixture();
    let spec = count_plus_non_count_spec();
    spec.validate(D).unwrap();
    let cells: Vec<(u32, &[u32], &[f32])> = fx
        .edges
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let input = Phase2Input {
        feat: &fx.feat,
        b_feat: &fx.b_feat,
        h: H,
        n_cells: FIX_CELLS,
        lambda: 1.0,
        dev: &dev,
        label: "Phase 2",
        gauge_fix: true,
    };
    let levels = vec![DistillLevel {
        e_pb: &fx.e_pb,
        cell_to_pb: &fx.cell_to_pb,
    }];
    let distill_spec = DistillSpec {
        levels: &levels,
        seed: 20_260_914,
    };
    let (out, encs) = project_cells(&input, &cells, None, &distill_spec, &spec, None).unwrap();
    // Exactly one encoder, on the count track's rows alone.
    assert_eq!(encs.iter().len(), 1);
    assert_eq!(encs.iter()[0].rows, spec.rows_of_track(0));
    assert_eq!(out.theta.len(), FIX_CELLS * H);
    assert!(out.theta.iter().all(|v| v.is_finite()));
    assert!(out.b_cell.iter().all(|v| v.is_finite()));
    // The non-count track gets its own intercept, exact at the cell's θ.
    assert_eq!(out.other_intercepts.len(), 1);
    assert_eq!(out.other_intercepts[0].len(), FIX_CELLS);
}

#[test]
fn encode_edges_with_one_encoder_that_does_not_span_the_axis_reads_only_its_rows() {
    let dev = Device::Cpu;
    let spec = count_plus_non_count_spec();
    let rows0 = spec.rows_of_track(0);
    let encs = CellEncoders::new(
        vec![TrackEncoder {
            track: 0,
            name: spec.tracks[0].name.clone(),
            rows: rows0.clone(),
            encoder: planted_encoder(&rows0, &[1.0, 2.0, 3.0], &dev),
        }],
        D,
    );
    assert!(encs.single().is_some());
    let owned = combine_nodes();
    let nodes: Vec<(u32, &[u32], &[f32])> = owned
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let out = encs.encode_edges(&nodes).unwrap();
    assert_eq!(out.theta.len(), 3 * H);
    // Nodes 0 and 1 have counts on the count track: the planted head.
    for n in 0..2 {
        for k in 0..H {
            assert!(
                (out.theta[n * H + k] - (k + 1) as f32).abs() < 1e-5,
                "node {n}, dim {k}: {}",
                out.theta[n * H + k]
            );
        }
    }
    // Node 2's counts are all on the non-count track, so nothing reads it: it
    // stays at the origin and reports the score clamp's floor.
    assert!(out.theta[2 * H..].iter().all(|v| v.abs() < 1e-6));
    assert_eq!(out.b_node[2], -(crate::cell_projection::SCORE_CLAMP as f32));
}

#[test]
fn load_refuses_a_path_list_that_is_not_exactly_the_count_tracks() {
    let dev = Device::Cpu;
    let spec = two_track_spec();
    let encs = planted_pair(&dev);
    let dir = std::env::temp_dir();
    let paths: Vec<(u32, String)> = (0..2u32)
        .map(|t| {
            let p = dir
                .join(format!(
                    "cell_enc_load_guard_{t}_{}.safetensors",
                    std::process::id()
                ))
                .to_string_lossy()
                .to_string();
            encs.iter()[t as usize].encoder.save(&p).unwrap();
            (t, p)
        })
        .collect();
    let (feat, b) = dict_feat();
    // Both count tracks: fine.
    CellEncoders::load(&feat, &b, H, &spec, &paths, &dev).unwrap();
    // One path for a two-count-track axis would silently place by track 0 alone.
    let err = CellEncoders::load(&feat, &b, H, &spec, &paths[..1], &dev)
        .map(|_| ())
        .unwrap_err();
    assert!(
        format!("{err}").contains("count track"),
        "unhelpful message: {err}"
    );
    // A list that skips track 0 would report another track's intercept as its own.
    let err = CellEncoders::load(&feat, &b, H, &spec, &paths[1..], &dev)
        .map(|_| ())
        .unwrap_err();
    assert!(
        format!("{err}").contains("count track"),
        "unhelpful message: {err}"
    );
    // A duplicate is not a cover either.
    let dup = vec![paths[0].clone(), paths[0].clone()];
    let err = CellEncoders::load(&feat, &b, H, &spec, &dup, &dev)
        .map(|_| ())
        .unwrap_err();
    assert!(
        format!("{err}").contains("count track"),
        "unhelpful message: {err}"
    );
    for (_, p) in &paths {
        std::fs::remove_file(p).ok();
    }
}

/// The refinement step is a dense `[step × D]` block with a backward pass, so
/// it answers to the same activation budget as every other phase-2 block: an
/// ordinary axis keeps the full step, a very wide one (a multiome axis with
/// every peak) shrinks it instead of running the device out of memory.
#[test]
fn refine_step_answers_to_the_block_budget() {
    assert_eq!(refine_cells_per_step(30_000), REFINE_CELLS_PER_STEP);
    let wide = 2_000_000;
    let step = refine_cells_per_step(wide);
    assert!(
        step < REFINE_CELLS_PER_STEP,
        "step {step} on {wide} features"
    );
    assert_eq!(step, block_sgd::block_cells(wide));
    assert!(refine_cells_per_step(usize::MAX / 1024) >= 1);
}

/// Phase 2 on a collapsed axis. Every cell's intercept is the FULL axis'
/// closed form at its `θ` (the collapse is exact for the likelihood), and the
/// saved encoder, which carries the collapse, places the run's own full-axis
/// cells where phase 2 put them: one estimator, both halves.
#[test]
fn phase_2_on_a_collapsed_axis_is_exact_on_the_full_axis() {
    let dev = Device::Cpu;
    let mut fx = one_track_fixture();
    // Rows 6..9 and 9..12 are module-only: each module's rows share one row.
    let module_only: Vec<bool> = (0..D).map(|g| g >= 6).collect();
    let labels: Vec<u32> = (0..D as u32)
        .map(|g| match g {
            0..6 => g,
            6..9 => 6,
            _ => 7,
        })
        .collect();
    for g in 6..D {
        let src = if g < 9 { 6 } else { 9 };
        for k in 0..H {
            fx.feat[g * H + k] = fx.feat[src * H + k];
        }
    }
    let collapse = RowCollapse::from_modules(&module_only, &labels).unwrap();
    assert_eq!(collapse.n_rows, 8);

    let cells: Vec<(u32, &[u32], &[f32])> = fx
        .edges
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let input = Phase2Input {
        feat: &fx.feat,
        b_feat: &fx.b_feat,
        h: H,
        n_cells: FIX_CELLS,
        lambda: 1.0,
        dev: &dev,
        label: "Phase 2",
        gauge_fix: true,
    };
    let levels = vec![DistillLevel {
        e_pb: &fx.e_pb,
        cell_to_pb: &fx.cell_to_pb,
    }];
    let spec = DistillSpec {
        levels: &levels,
        seed: 20_260_922,
    };
    let (out, encs) = project_cells(
        &input,
        &cells,
        None,
        &spec,
        &TrackSpec::base(D),
        Some(&collapse),
    )
    .unwrap();
    assert_eq!(out.theta.len(), FIX_CELLS * H);
    let tm = &out.gauge.theta_mean;
    let theta = |i: usize| -> Vec<f32> { (0..H).map(|k| out.theta[i * H + k] + tm[k]).collect() };

    for (i, (_, counts)) in fx.edges.iter().enumerate() {
        let th = theta(i);
        let scores: Vec<f64> = (0..D)
            .map(|g| {
                let s: f32 = (0..H).map(|k| th[k] * fx.feat[g * H + k]).sum::<f32>() + fx.b_feat[g];
                f64::from(s)
            })
            .collect();
        let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = mx + scores.iter().map(|s| (s - mx).exp()).sum::<f64>().ln();
        let n: f64 = counts.iter().map(|&c| f64::from(c)).sum();
        let want = n.ln() - lse;
        assert!(
            (f64::from(out.b_cell[i]) - want).abs() < 1e-4,
            "cell {i}: intercept {} vs the full axis' {want}",
            out.b_cell[i]
        );
    }

    let enc = encs.single().expect("one count track, one encoder");
    let path = std::env::temp_dir().join(format!(
        "cell_enc_phase2_collapsed_{}.safetensors",
        std::process::id()
    ));
    let path = path.to_string_lossy().to_string();
    enc.save(&path).unwrap();
    let again = CellEncoder::load(&fx.feat, &fx.b_feat, H, &path, &dev).unwrap();
    std::fs::remove_file(&path).ok();
    let placed = again.encode_edges(&cells).unwrap();
    for i in 0..FIX_CELLS {
        for (k, want) in theta(i).into_iter().enumerate() {
            let got = placed.theta[i * H + k];
            assert!(
                (got - want).abs() < 1e-5,
                "cell {i} θ[{k}]: {got} vs phase 2's {want}"
            );
        }
    }
}

/// An encoder trained on the collapsed axis, saved with its row map and
/// reloaded against the FULL dictionary, places full-axis cells exactly as the
/// in-memory encoder places the same cells already collapsed.
#[test]
fn a_collapsed_encoder_round_trips_onto_the_full_axis() {
    let dev = Device::Cpu;
    // Full axis: rows 0, 1 residual; rows 2..6 module-only in two modules.
    let module_only = [false, false, true, true, true, true];
    let labels = [0u32, 1, 5, 5, 7, 7];
    let collapse = RowCollapse::from_modules(&module_only, &labels).unwrap();
    let h = 3;
    let mut feat = vec![0f32; 6 * h];
    for (g, row) in feat.chunks_mut(h).enumerate() {
        let src = if module_only[g] {
            labels[g] as usize
        } else {
            g
        };
        for (k, x) in row.iter_mut().enumerate() {
            *x = ((src * 5 + k * 3) % 7) as f32 * 0.1 - 0.3;
        }
    }
    let b = vec![0.2f32, -0.1, -1.0, -0.4, -0.7, -1.3];
    let (rf, rb) = collapse.reduce_dictionary(&feat, &b, h);
    let mean_red = vec![1.0f32; collapse.n_rows];
    let mut enc =
        CellEncoder::build(FrozenDict::new(&rf, &rb, h, &dev).unwrap(), &mean_red, &dev).unwrap();
    enc.collapse = Some(collapse.clone());

    let path = std::env::temp_dir().join(format!(
        "cell_enc_collapsed_{}.safetensors",
        std::process::id()
    ));
    let path = path.to_string_lossy().to_string();
    enc.save(&path).unwrap();
    let again = CellEncoder::load(&feat, &b, h, &path, &dev).unwrap();
    std::fs::remove_file(&path).ok();

    // Full-axis cells through the reloaded encoder …
    let cells: Vec<(Vec<u32>, Vec<f32>)> = vec![
        (vec![0, 2, 3, 5], vec![2.0, 1.0, 4.0, 3.0]),
        (vec![1, 4], vec![5.0, 1.0]),
    ];
    let full: Vec<(u32, &[u32], &[f32])> = cells
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    let a = again.encode_edges(&full).unwrap();
    // … match the same cells collapsed by hand through the in-memory encoder.
    let reduced: Vec<(Vec<u32>, Vec<f32>)> = cells
        .iter()
        .map(|(f, c)| collapse.reduce_edges(f, c))
        .collect();
    let red: Vec<(u32, &[u32], &[f32])> = reduced
        .iter()
        .enumerate()
        .map(|(i, (f, c))| (i as u32, f.as_slice(), c.as_slice()))
        .collect();
    enc.collapse = None;
    let bx = enc.encode_edges(&red).unwrap();
    for (x, y) in a.theta.iter().zip(&bx.theta) {
        assert!((x - y).abs() < 1e-5, "θ {x} vs {y}");
    }
    for (x, y) in a.b_node.iter().zip(&bx.b_node) {
        assert!((x - y).abs() < 1e-5, "intercept {x} vs {y}");
    }
}
