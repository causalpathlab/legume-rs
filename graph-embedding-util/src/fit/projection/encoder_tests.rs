//! The distilled encoder path: the sub-pseudobulk aggregate, the seeded
//! hold-out, the exact conditional intercept, and that the distillation fits a
//! planted target on held-out pseudobulks.

use super::*;
use candle_util::candle_core::Device;

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
    let (out, encs) = project_cells(&input, &cells, None, &spec, &TrackSpec::base(D)).unwrap();
    // One count track, so exactly one encoder — the file `senna bge` persists.
    assert_eq!(encs.iter().len(), 1);
    assert!(encs.single().is_some());
    (out.theta, out.b_cell)
}

/// The parity guard for the one-track (`senna bge`) path.
///
/// The numbers below were taken from `project_cells` on this same fixture with
/// the code as it stood BEFORE the per-track rewrite (single `FrozenDict::new`,
/// one `PooledGeneEncoder`, the single-dictionary `refine`), as the mean of six
/// runs of that build.
///
/// The bar is 5e-4, not exact, and the reason is not this change: the global
/// gradient-norm clip sums the per-parameter squares in `GradStore`'s hash-map
/// order, which differs between processes, so the clip factor — and through
/// fifty epochs of SGD, the answer — moves run to run *within one build*. That
/// spread was measured at 1.6e-4 (max over six runs, values spanning ±0.7);
/// with the clip disabled the same six runs agreed to the bit. The snapshot is
/// the mean of those six, so 5e-4 sits ~6x above the worst deviation from it —
/// and 3x BELOW what the smallest real change reaches: re-seeding track 0 off
/// `spec.seed` already moves θ[0] by 1.5e-3.
#[test]
fn one_track_project_cells_matches_the_previous_output() {
    #[rustfmt::skip]
    const THETA: [f32; FIX_CELLS * H] = [
        0.415465, 0.197483, -0.161280, 0.078608, -0.070422, -0.702394,
        -0.063581, 0.142593, -0.038534, -0.014298, 0.412886, -0.173098,
        0.476744, 0.466121, -0.258836, -0.060034, 0.114163, -0.579790,
        -0.146075, 0.122002, -0.215837, -0.081836, 0.543518, -0.141621,
        0.475339, 0.658283, 0.029077, -0.128021, -0.089705, -0.554376,
        0.117690, 0.223137, -0.175178, -0.081836, 0.543518, -0.141621,
        0.367344, 0.404782, 0.024488, -0.157214, -0.021566, -0.422413,
        -0.105511, -0.046066, -0.146974, -0.023166, 0.671513, 0.102777,
        0.281262, 0.153771, 0.129035, -0.028430, 0.150919, -0.146246,
        -0.125012, -0.150155, -0.096700, -0.077601, 0.598665, 0.044016,
        0.281262, 0.153771, 0.129035, -0.051901, 0.027350, -0.086648,
        -0.172001, -0.210862, 0.017108, 0.014058, 0.489938, 0.191971,
        0.058501, -0.273539, 0.102896, -0.228285, -0.289475, -0.047525,
        -0.436371, -0.619823, 0.004610, 0.036538, 0.425799, 0.300043,
        0.152790, -0.070007, 0.132742, -0.228285, -0.289475, -0.047525,
        -0.390827, -0.619072, 0.108628, -0.138582, 0.256469, 0.350371,
        0.152790, -0.070007, 0.132742, -0.145479, -0.621181, -0.065321,
        -0.078132, -0.393741, 0.370869, -0.061412, -0.040242, 0.364270,
        0.240890, -0.278171, 0.189121, -0.126900, -0.525292, -0.106238,
        -0.078132, -0.393741, 0.370869, 0.025259, 0.004374, 0.515326,
        0.386110, -0.158149, 0.036904, -0.270283, -0.519122, -0.176567,
        0.037664, -0.239856, 0.496499, -0.006318, 0.013277, 0.200861,
        0.199471, -0.157492, -0.053043, -0.137731, -0.414818, -0.338569,
        -0.148210, -0.225632, 0.331219, -0.006318, 0.013277, 0.200861,
    ];
    #[rustfmt::skip]
    const B_CELL: [f32; FIX_CELLS] = [
        0.365279, 0.347813, 0.495867, 0.326731, 0.382886, 0.417894,
        0.409558, 0.415506, 0.274180, 0.497274, 0.432204, 0.415506,
        0.377130, 0.455120, 0.435345, 0.297668, 0.328201, 0.365830,
        0.498418, 0.298518, 0.328201, 0.441726, 0.367414, 0.348246,
        0.231170, 0.358406, 0.224674, 0.426644, 0.168407, 0.358406,
        0.288082, 0.279452, 0.168407, 0.259073, 0.246233, 0.219340,
        0.140509, 0.203507, 0.246233, 0.281543, 0.151683, 0.268983,
        0.253660, 0.383563, 0.243059, 0.282474, 0.273670, 0.383563,
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
    let (out, encs) = project_cells(&input, &cells, None, &distill_spec, &spec).unwrap();
    // Exactly one encoder, on the count track's rows alone.
    assert_eq!(encs.iter().len(), 1);
    assert_eq!(encs.iter()[0].rows, spec.rows_of_track(0));
    assert_eq!(out.theta.len(), FIX_CELLS * H);
    assert!(out.theta.iter().all(|v| v.is_finite()));
    assert!(out.b_cell.iter().all(|v| v.is_finite()));
    // The non-count track got its own intercept from the polish.
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
