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
}
