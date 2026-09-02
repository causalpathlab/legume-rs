//! Unit tests for the cell-block phase-2 Poisson SGD.
//!
//! The planted-parameter tests mirror `crate::cell_projection`'s
//! `irls_recovers_planted_cell` / `increment_recovers_planted_velocity`, but
//! generate counts under the **full-partition** likelihood this solver actually
//! optimizes: every feature gets its Poisson rate, so a cell's "observed" edge set
//! is every feature with a nonzero rate rather than a hand-picked subset.
//!
//! Every recovery assertion adds the reported [`GaugeShift`] back before
//! comparing. The solver deliberately returns latents mean-centred (see
//! `GaugeShift`), so `θ_c + θ̄` is the quantity that must match the planted value —
//! which makes these a check on the gauge bookkeeping as well as on the fit. All of
//! them plant **several** cells: with a single cell the mean is that cell and
//! centring is trivially degenerate.

use super::edges::block_cells;
use super::*;
use crate::fit::batch_fold::BatchGeneFold;
use candle_util::candle_core::Device;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// `latent + mean` — undo the gauge fix for comparison against a planted value.
fn ungauge(latent: &[f32], mean: &[f32]) -> Vec<f32> {
    latent.iter().zip(mean).map(|(a, b)| a + b).collect()
}

/// Deterministic frozen dictionary: `e_f` scaled to the magnitude real fits carry
/// (`‖β_g‖ ≈ 0.013`), so the learning-rate auto-scaling is exercised on the scale
/// it was designed for rather than on a convenient `O(1)` one.
fn dictionary(n_feat: usize, h: usize, scale: f32) -> (Vec<f32>, Vec<f32>) {
    let mut e = vec![0f32; n_feat * h];
    let mut b = vec![0f32; n_feat];
    for f in 0..n_feat {
        for k in 0..h {
            e[f * h + k] = ((((f * 7 + k * 13) % 11) as f32 / 11.0) - 0.5) * scale;
        }
        b[f] = (((f * 5) % 7) as f32 / 7.0) - 0.3;
    }
    (e, b)
}

/// Noiseless Poisson rates for one cell at latent `theta` and intercept `b_c`.
fn rates(e: &[f32], b: &[f32], h: usize, theta: &[f32], b_c: f32) -> Vec<(u32, f32)> {
    (0..b.len())
        .map(|f| {
            let ef = &e[f * h..(f + 1) * h];
            let s: f32 = ef.iter().zip(theta).map(|(a, t)| a * t).sum::<f32>() + b[f] + b_c;
            (f as u32, s.exp())
        })
        .collect()
}

/// Run `project_cells` over one pass (no splice split, no fold) for a set of planted cells.
fn project(e: &[f32], b: &[f32], h: usize, per_cell: &[Vec<(u32, f32)>], lambda: f64) -> Phase2Out {
    project_with(e, b, h, per_cell, lambda, None)
}

/// Three planted latents; counts at the noiseless full-partition rate. Each cell's
/// `θ_c + θ̄` must recover its planted direction.
#[test]
fn recovers_planted_cells() {
    let (h, n_feat) = (8, 300);
    let (e, b) = dictionary(n_feat, h, 0.4);
    let planted = [
        [0.8f32, -0.6, 0.4, 0.2, -0.3, 0.5, 0.1, -0.4],
        [-0.5f32, 0.7, -0.2, 0.6, 0.1, -0.4, 0.3, 0.2],
        [0.2f32, 0.1, -0.7, -0.3, 0.5, 0.2, -0.6, 0.4],
    ];
    let per_cell: Vec<_> = planted
        .iter()
        .enumerate()
        .map(|(i, t)| rates(&e, &b, h, t, 0.4 + 0.1 * i as f32))
        .collect();

    let out = project(&e, &b, h, &per_cell, 1e-3);
    assert!(out.velocity.is_none(), "no splice mask ⇒ no velocity pass");
    for (i, want) in planted.iter().enumerate() {
        let got = ungauge(&out.theta[i * h..(i + 1) * h], &out.gauge.theta_mean);
        let c = cos(&got, want);
        assert!(c > 0.97, "cell {i} misaligned (cos={c:.3})");
    }
}

/// The returned latents really are mean-zero — that is what the co-embedding needs,
/// and what the caller relies on when folding the shift into `b_feat`.
#[test]
fn latents_come_back_gauge_fixed() {
    let (h, n_feat) = (6, 240);
    let (e, b) = dictionary(n_feat, h, 0.4);
    let planted = [
        [1.0f32, -0.5, 0.3, 0.0, 0.2, -0.1],
        [-0.4f32, 0.9, -0.2, 0.5, -0.6, 0.3],
        [0.3f32, 0.2, 0.8, -0.4, 0.1, 0.5],
    ];
    let per_cell: Vec<_> = planted.iter().map(|t| rates(&e, &b, h, t, 0.2)).collect();

    let out = project(&e, &b, h, &per_cell, 1e-3);
    let drift = norm(&mean_rows(&out.theta, h));
    assert!(
        drift < 1e-4,
        "latents are not mean-zero (‖mean‖={drift:.2e})"
    );
    assert!(
        norm(&out.gauge.theta_mean) > 1e-3,
        "a non-trivial gauge shift should have been reported"
    );
}

/// Cells with different planted latents must stay distinguishable — a block solves
/// them jointly, so this is the test that per-cell separability survives the block
/// formulation (nothing leaks between rows).
#[test]
fn block_keeps_cells_independent() {
    let (h, n_feat) = (6, 240);
    let (e, b) = dictionary(n_feat, h, 0.4);
    let planted = [
        [1.0f32, -0.5, 0.3, 0.0, 0.2, -0.1],
        [-0.4f32, 0.9, -0.2, 0.5, -0.6, 0.3],
        [0.3f32, 0.2, 0.8, -0.4, 0.1, 0.5],
    ];
    let per_cell: Vec<_> = planted
        .iter()
        .enumerate()
        .map(|(i, t)| rates(&e, &b, h, t, -0.3 + 0.2 * i as f32))
        .collect();

    let out = project(&e, &b, h, &per_cell, 1e-3);
    for (i, want) in planted.iter().enumerate() {
        let got = ungauge(&out.theta[i * h..(i + 1) * h], &out.gauge.theta_mean);
        let c = cos(&got, want);
        assert!(c > 0.95, "cell {i} bled into its block-mates (cos={c:.3})");
    }
}

/// A cell the samplers never saw keeps the zero row. After the gauge fix the origin
/// IS the population mean, so that is the right "no information" position rather
/// than an arbitrary corner — and the empty-droplet QC still reads `‖θ‖ = 0` for it.
#[test]
fn unseen_cell_stays_at_the_origin() {
    let (h, n_feat) = (4, 120);
    let (e, b) = dictionary(n_feat, h, 0.4);
    let planted = [[0.5f32, -0.4, 0.3, 0.2], [-0.3f32, 0.6, 0.1, -0.5]];
    let seen: Vec<_> = planted.iter().map(|t| rates(&e, &b, h, t, 0.1)).collect();
    let feats: Vec<Vec<u32>> = seen
        .iter()
        .map(|c| c.iter().map(|&(f, _)| f).collect())
        .collect();
    let counts: Vec<Vec<f32>> = seen
        .iter()
        .map(|c| c.iter().map(|&(_, n)| n).collect())
        .collect();
    // Cells 0 and 1 are present; cell 2 exists on the axis but carries no edges.
    let cells: Vec<(u32, &[u32], &[f32])> = (0..2)
        .map(|i| (i as u32, feats[i].as_slice(), counts[i].as_slice()))
        .collect();
    let out = project_cells(
        &Phase2Input {
            feat: &e,
            b_feat: &b,
            h,
            n_cells: 3,
            lambda: 1e-3,
            dev: &Device::Cpu,
            label: "Phase 2",
            gauge_fix: true,
            joint: false,
        },
        &cells,
        None,
        None,
    )
    .unwrap();
    assert_eq!(out.theta.len(), 3 * h);
    assert!(
        out.theta[2 * h..].iter().all(|x| *x == 0.0),
        "unseen cell moved off the origin"
    );
    assert_eq!(out.b_cell[2], 0.0);
}

/// Splice path: spliced rows carry `θ*`, unspliced rows carry `θ* + δ*`. The
/// identity pass must recover θ* from the spliced half and the velocity pass δ*
/// from the unspliced half with θ held fixed — each up to its own gauge shift.
#[test]
fn splice_pass_recovers_theta_and_delta() {
    let (h, n_half) = (6, 200);
    let n_feat = n_half * 2; // rows [0, n_half) spliced, [n_half, 2·n_half) unspliced
    let unspliced_rows: Vec<bool> = (0..n_feat).map(|f| f >= n_half).collect();
    let (e, b) = dictionary(n_feat, h, 0.4);

    let theta_star = [
        [0.6f32, -0.4, 0.3, 0.2, -0.5, 0.1],
        [-0.2f32, 0.7, -0.4, 0.5, 0.2, -0.3],
        [0.4f32, 0.1, 0.6, -0.3, -0.1, 0.5],
    ];
    let delta_star = [
        [0.25f32, -0.3, 0.15, 0.2, 0.1, -0.2],
        [-0.15f32, 0.2, 0.3, -0.1, -0.25, 0.1],
        [0.1f32, 0.15, -0.2, 0.3, 0.2, -0.15],
    ];

    let mut feats: Vec<Vec<u32>> = Vec::new();
    let mut counts: Vec<Vec<f32>> = Vec::new();
    for (t, d) in theta_star.iter().zip(&delta_star) {
        let nascent: Vec<f32> = t.iter().zip(d).map(|(a, c)| a + c).collect();
        let (mut fi, mut ci) = (Vec::new(), Vec::new());
        for f in 0..n_feat {
            let ef = &e[f * h..(f + 1) * h];
            let latent: &[f32] = if unspliced_rows[f] { &nascent } else { t };
            let s: f32 = ef.iter().zip(latent).map(|(a, x)| a * x).sum::<f32>() + b[f] + 0.2;
            fi.push(f as u32);
            ci.push(s.exp());
        }
        feats.push(fi);
        counts.push(ci);
    }
    let cells: Vec<(u32, &[u32], &[f32])> = (0..3)
        .map(|i| (i as u32, feats[i].as_slice(), counts[i].as_slice()))
        .collect();

    let out = project_cells(
        &Phase2Input {
            feat: &e,
            b_feat: &b,
            h,
            n_cells: 3,
            lambda: 1e-3,
            dev: &Device::Cpu,
            label: "Phase 2",
            gauge_fix: true,
            joint: false,
        },
        &cells,
        None,
        Some(&unspliced_rows),
    )
    .unwrap();

    let vel = out.velocity.as_ref().expect("splice mask ⇒ velocity pass");
    for i in 0..3 {
        let th = ungauge(&out.theta[i * h..(i + 1) * h], &out.gauge.theta_mean);
        let dl = ungauge(&vel[i * h..(i + 1) * h], &out.gauge.delta_mean);
        let ct = cos(&th, &theta_star[i]);
        let cd = cos(&dl, &delta_star[i]);
        assert!(ct > 0.97, "cell {i} identity misaligned (cos={ct:.3})");
        assert!(cd > 0.95, "cell {i} velocity misaligned (cos={cd:.3})");
    }
}

/// `block_cells` must keep a block's activations inside the budget for feature
/// counts spanning the range a real fit sees (a few hundred HVGs to every gene in
/// the annotation), and must never round down to zero.
#[test]
fn block_sizing_respects_the_activation_budget() {
    for f in [1usize, 100, 5_000, 58_651, 500_000, 5_000_000] {
        let bc = block_cells(f);
        assert!(bc >= 1, "F={f}: block size collapsed to zero");
        assert!(
            bc * f * 4 * LIVE_BLOCK_TENSORS <= BLOCK_ACTIVATION_BYTES || bc == 1,
            "F={f}: block of {bc} exceeds the activation budget"
        );
    }
}

/// Gate folding is exact for a feature whose `e_f` is identically zero: its score is
/// `β_f + c` however `Θ` moves, so pulling it out of the matmul and into the scalar
/// partition mass must not change the answer.
#[test]
fn gate_folding_is_exact_for_zero_rows() {
    let (h, n_live) = (5, 150);
    let (mut e, mut b) = dictionary(n_live, h, 0.4);
    let planted = [
        [0.7f32, -0.5, 0.4, 0.2, -0.3],
        [-0.3f32, 0.6, -0.2, 0.5, 0.1],
    ];
    let live_edges: Vec<_> = planted.iter().map(|t| rates(&e, &b, h, t, 0.3)).collect();

    // Reference: live rows only.
    let ref_out = project(&e, &b, h, &live_edges, 1e-3);

    // Same problem plus 40 identically-zero rows, which the fold must remove.
    let n_dead = 40;
    e.extend(std::iter::repeat_n(0f32, n_dead * h));
    b.extend((0..n_dead).map(|d| (d as f32 % 5.0) / 5.0 - 0.4));
    let mut edges = live_edges;
    for cell in edges.iter_mut() {
        for d in 0..n_dead {
            let f = n_live + d;
            cell.push((f as u32, (b[f] + 0.3).exp()));
        }
    }
    let fold_out = project(&e, &b, h, &edges, 1e-3);

    for i in 0..planted.len() {
        let c = cos(
            &ref_out.theta[i * h..(i + 1) * h],
            &fold_out.theta[i * h..(i + 1) * h],
        );
        assert!(c > 0.999, "gate folding moved cell {i} (cos={c:.4})");
    }
}

/// Joint θ+δ (`joint: true`): the SAME planted splice problem as
/// `splice_pass_recovers_theta_and_delta`, but θ and δ are estimated **together** in one
/// solve (θ pulled by both tracks). Both must still recover their planted directions.
#[test]
fn joint_recovers_theta_and_delta() {
    let (h, n_half) = (6, 200);
    let n_feat = n_half * 2; // rows [0, n_half) spliced, [n_half, 2·n_half) unspliced
    let unspliced_rows: Vec<bool> = (0..n_feat).map(|f| f >= n_half).collect();
    let (e, b) = dictionary(n_feat, h, 0.4);

    let theta_star = [
        [0.6f32, -0.4, 0.3, 0.2, -0.5, 0.1],
        [-0.2f32, 0.7, -0.4, 0.5, 0.2, -0.3],
        [0.4f32, 0.1, 0.6, -0.3, -0.1, 0.5],
    ];
    let delta_star = [
        [0.25f32, -0.3, 0.15, 0.2, 0.1, -0.2],
        [-0.15f32, 0.2, 0.3, -0.1, -0.25, 0.1],
        [0.1f32, 0.15, -0.2, 0.3, 0.2, -0.15],
    ];

    let mut feats: Vec<Vec<u32>> = Vec::new();
    let mut counts: Vec<Vec<f32>> = Vec::new();
    for (t, d) in theta_star.iter().zip(&delta_star) {
        let nascent: Vec<f32> = t.iter().zip(d).map(|(a, c)| a + c).collect();
        let (mut fi, mut ci) = (Vec::new(), Vec::new());
        for f in 0..n_feat {
            let ef = &e[f * h..(f + 1) * h];
            let latent: &[f32] = if unspliced_rows[f] { &nascent } else { t };
            let s: f32 = ef.iter().zip(latent).map(|(a, x)| a * x).sum::<f32>() + b[f] + 0.2;
            fi.push(f as u32);
            ci.push(s.exp());
        }
        feats.push(fi);
        counts.push(ci);
    }
    let cells: Vec<(u32, &[u32], &[f32])> = (0..3)
        .map(|i| (i as u32, feats[i].as_slice(), counts[i].as_slice()))
        .collect();

    let out = project_cells(
        &Phase2Input {
            feat: &e,
            b_feat: &b,
            h,
            n_cells: 3,
            lambda: 1e-3,
            dev: &Device::Cpu,
            label: "Phase 2",
            gauge_fix: true,
            joint: true,
        },
        &cells,
        None,
        Some(&unspliced_rows),
    )
    .unwrap();

    let vel = out.velocity.as_ref().expect("splice mask ⇒ velocity pass");
    for i in 0..3 {
        let th = ungauge(&out.theta[i * h..(i + 1) * h], &out.gauge.theta_mean);
        let dl = ungauge(&vel[i * h..(i + 1) * h], &out.gauge.delta_mean);
        let ct = cos(&th, &theta_star[i]);
        let cd = cos(&dl, &delta_star[i]);
        assert!(
            ct > 0.97,
            "joint cell {i} identity misaligned (cos={ct:.3})"
        );
        assert!(
            cd > 0.95,
            "joint cell {i} velocity misaligned (cos={cd:.3})"
        );
    }
}

/// The streaming entry point must agree with the one-shot one **exactly**, not
/// approximately. `project_prepared` skips `project_cells`'s partition/edge setup in
/// favour of a `PassDict` the caller already holds, so this is the test that the
/// dictionary carries the same feature partition, the edge remap still lands on the
/// same live ids, and the scatter still writes the same rows.
#[test]
fn streaming_entry_matches_the_one_shot_one() {
    let (h, n_feat) = (6, 240);
    let (e, b) = dictionary(n_feat, h, 0.4);
    let planted = [
        [0.7f32, -0.5, 0.4, 0.2, -0.3, 0.1],
        [-0.3f32, 0.6, -0.2, 0.5, 0.1, -0.4],
        [0.2f32, 0.1, 0.5, -0.3, 0.4, 0.2],
    ];
    let per_cell: Vec<_> = planted
        .iter()
        .enumerate()
        .map(|(i, t)| rates(&e, &b, h, t, 0.2 + 0.1 * i as f32))
        .collect();
    let feats: Vec<Vec<u32>> = per_cell
        .iter()
        .map(|c| c.iter().map(|&(f, _)| f).collect())
        .collect();
    let counts: Vec<Vec<f32>> = per_cell
        .iter()
        .map(|c| c.iter().map(|&(_, n)| n).collect())
        .collect();
    let nodes: Vec<(u32, &[u32], &[f32])> = (0..3)
        .map(|i| (i as u32, feats[i].as_slice(), counts[i].as_slice()))
        .collect();

    // `gauge_fix: false` — the frozen projection never re-centres (`b_feat` is frozen,
    // so re-gauging θ alone would break its correspondence with the dictionary).
    let input = Phase2Input {
        feat: &e,
        b_feat: &b,
        h,
        n_cells: 3,
        lambda: 1e-3,
        dev: &Device::Cpu,
        label: "Projection",
        gauge_fix: false,
        joint: false,
    };

    let one_shot = project_cells(&input, &nodes, None, None).expect("one-shot");

    let dict = PassDict::build(
        &DictSpec {
            feat: &e,
            b_feat: &b,
            h,
            lambda: 1e-3,
            dev: &Device::Cpu,
            label: "Projection",
            pass: "nodes",
        },
        (0..n_feat as u32).collect(),
    )
    .expect("dict");
    let streamed = project_prepared(&input, &dict, &nodes, &indicatif::ProgressBar::hidden())
        .expect("streamed");

    assert_eq!(
        one_shot.theta, streamed.theta,
        "θ differs between entry points"
    );
    assert_eq!(
        one_shot.b_cell, streamed.b_cell,
        "b_node differs between entry points"
    );
}

/// The group size a streaming caller is handed must be a whole number of blocks.
///
/// That is the whole reason it is the engine's number and not the caller's: blocks
/// are cut at multiples of `Bc` from the start of each group, so a group cut here
/// reproduces the block partition of a single call over the whole query — and
/// therefore its numbers exactly. A group size that is *not* a multiple leaves a
/// short block at the end of every group, which both wastes steps and moves the
/// answer.
#[test]
fn the_offered_group_size_is_a_whole_number_of_blocks() {
    for f in [1usize, 100, 5_000, 58_651] {
        let (h, n_feat) = (4, f.min(400));
        let (e, b) = dictionary(n_feat, h, 0.4);
        // The dict's own partition is what sizes its blocks, so vary that rather than
        // building an enormous dictionary just to move `Bc`.
        let rows: Vec<u32> = (0..n_feat as u32).collect();
        let dict = PassDict::build(
            &DictSpec {
                feat: &e,
                b_feat: &b,
                h,
                lambda: 1.0,
                dev: &Device::Cpu,
                label: "Projection",
                pass: "nodes",
            },
            rows,
        )
        .unwrap();
        let bc = block_cells(n_feat);
        let g = dict.group_nodes();
        assert!(
            g >= bc,
            "F={n_feat}: group of {g} is under one block of {bc}"
        );
        assert_eq!(
            g % bc,
            0,
            "F={n_feat}: group of {g} is not whole blocks of {bc}"
        );
    }
}

////////////////////////////
// Per-batch gene fold     //
////////////////////////////

/// A planted per-feature batch fold on batch 1: powers of two in
/// `{¼, ½, 1, 2, 4}`, exact in f32 so `(y·δ)/δ == y` bit for bit.
fn planted_fold(n_feat: usize) -> Vec<f32> {
    (0..n_feat)
        .map(|f| 2f32.powi(((f * 3) % 5) as i32 - 2))
        .collect()
}

/// A two-batch fold table: batch 0 unfolded, batch 1 with the planted fold.
fn fold_table(n_feat: usize) -> BatchGeneFold {
    let mut delta = vec![1f32; n_feat]; // batch 0: no fold
    delta.extend(planted_fold(n_feat));
    BatchGeneFold {
        delta,
        n_features: n_feat,
        batch_names: vec!["a".into(), "b".into()],
    }
}

/// Two batches, three planted latents each. Batch 1's counts carry a per-feature
/// fold on top of the shared dictionary. Dividing by the matching fold must
/// recover every planted direction in BOTH batches; solving raw must fail on
/// batch 1 — that is what shows the fold is doing the work.
#[test]
fn batch_fold_recovers_a_shared_latent_across_batches() {
    let (h, n_feat) = (8, 300);
    let (e, b) = dictionary(n_feat, h, 0.4);
    let fold = planted_fold(n_feat);
    let planted = [
        [0.8f32, -0.6, 0.4, 0.2, -0.3, 0.5, 0.1, -0.4],
        [-0.5f32, 0.7, -0.2, 0.6, 0.1, -0.4, 0.3, 0.2],
        [0.2f32, 0.1, -0.7, -0.3, 0.5, 0.2, -0.6, 0.4],
    ];
    // Cells 0..3 in batch 0, 3..6 in batch 1 (the same latents again).
    let mut per_cell: Vec<Vec<(u32, f32)>> = Vec::new();
    let mut cell_to_batch: Vec<u32> = Vec::new();
    for batch in 0..2u32 {
        for (i, t) in planted.iter().enumerate() {
            let mut r = rates(&e, &b, h, t, 0.4 + 0.1 * i as f32);
            if batch == 1 {
                for (f, x) in r.iter_mut() {
                    *x *= fold[*f as usize];
                }
            }
            per_cell.push(r);
            cell_to_batch.push(batch);
        }
    }
    let table = fold_table(n_feat);
    let cbf = CellBatchFold {
        fold: &table,
        cell_to_batch: &cell_to_batch,
    };

    let with = project_with(&e, &b, h, &per_cell, 1e-3, Some(cbf));
    let without = project_with(&e, &b, h, &per_cell, 1e-3, None);
    let cos_of = |out: &Phase2Out, i: usize| {
        let got = ungauge(&out.theta[i * h..(i + 1) * h], &out.gauge.theta_mean);
        cos(&got, &planted[i % 3])
    };
    for i in 0..6 {
        let c = cos_of(&with, i);
        assert!(c > 0.97, "with fold: cell {i} misaligned (cos={c:.3})");
    }
    let worst_without = (3..6).map(|i| cos_of(&without, i)).fold(1.0f32, f32::min);
    assert!(
        worst_without < 0.97,
        "without the fold batch 1 must NOT pass the recovery bar (worst cos={worst_without:.3}), \
         or the test has no teeth"
    );
}

/// The contract that makes the divide batch-invariant under misfit: a cell whose
/// counts are exactly a batch's fold of another cell's counts is the SAME problem
/// after the divide, so it solves to the identical latent and intercept — bit for
/// bit, misfit or not. (An offset on the rate weights each batch's misfit by its
/// own fold and does not have this property.)
#[test]
fn a_folded_copy_of_a_cell_solves_to_the_identical_latent() {
    let (h, n_feat) = (6, 120);
    let (e, b) = dictionary(n_feat, h, 0.3);
    let fold = planted_fold(n_feat);
    // Counts that the dictionary cannot reproduce exactly: a planted profile plus
    // an off-model bump on every third feature.
    let mut base: Vec<(u32, f32)> = rates(&e, &b, h, &[0.4, -0.2, 0.5, 0.1, -0.6, 0.3], 0.3);
    for (f, x) in base.iter_mut() {
        if *f % 3 == 0 {
            *x *= 3.0;
        }
    }
    let folded: Vec<(u32, f32)> = base
        .iter()
        .map(|&(f, x)| (f, x * fold[f as usize]))
        .collect();
    let other = rates(&e, &b, h, &[-0.3, 0.6, -0.1, 0.4, 0.2, -0.5], 0.2);
    let per_cell = vec![base, folded, other];
    let cell_to_batch = vec![0u32, 1, 0];
    let table = fold_table(n_feat);
    let out = project_with(
        &e,
        &b,
        h,
        &per_cell,
        1e-3,
        Some(CellBatchFold {
            fold: &table,
            cell_to_batch: &cell_to_batch,
        }),
    );
    assert_eq!(
        &out.theta[..h],
        &out.theta[h..2 * h],
        "folded copy must solve identically"
    );
    assert_eq!(out.b_cell[0], out.b_cell[1], "and to the same intercept");
    // Sanity: the third cell is a different problem.
    assert_ne!(&out.theta[..h], &out.theta[2 * h..3 * h]);
}

/// A single batch with an all-ones fold row is the same problem as no fold at
/// all — same block partition, same numbers, byte for byte.
#[test]
fn unit_fold_is_byte_identical_to_none() {
    let (h, n_feat) = (6, 120);
    let (e, b) = dictionary(n_feat, h, 0.3);
    let per_cell: Vec<_> = (0..5)
        .map(|i| {
            let t: Vec<f32> = (0..h)
                .map(|k| (((i * 3 + k * 5) % 7) as f32 / 7.0) - 0.5)
                .collect();
            rates(&e, &b, h, &t, 0.2 + 0.05 * i as f32)
        })
        .collect();
    let ones = BatchGeneFold {
        delta: vec![1f32; n_feat],
        n_features: n_feat,
        batch_names: vec!["only".into()],
    };
    let cell_to_batch = vec![0u32; 5];
    let a = project_with(&e, &b, h, &per_cell, 1e-3, None);
    let z = project_with(
        &e,
        &b,
        h,
        &per_cell,
        1e-3,
        Some(CellBatchFold {
            fold: &ones,
            cell_to_batch: &cell_to_batch,
        }),
    );
    assert_eq!(a.theta, z.theta);
    assert_eq!(a.b_cell, z.b_cell);
    assert_eq!(a.gauge.theta_mean, z.gauge.theta_mean);
}

/// `project` with an optional batch fold.
fn project_with(
    e: &[f32],
    b: &[f32],
    h: usize,
    per_cell: &[Vec<(u32, f32)>],
    lambda: f64,
    fold: Option<CellBatchFold>,
) -> Phase2Out {
    let feats: Vec<Vec<u32>> = per_cell
        .iter()
        .map(|c| c.iter().map(|&(f, _)| f).collect())
        .collect();
    let counts: Vec<Vec<f32>> = per_cell
        .iter()
        .map(|c| c.iter().map(|&(_, n)| n).collect())
        .collect();
    let cells: Vec<(u32, &[u32], &[f32])> = (0..per_cell.len())
        .map(|i| (i as u32, feats[i].as_slice(), counts[i].as_slice()))
        .collect();
    project_cells(
        &Phase2Input {
            feat: e,
            b_feat: b,
            h,
            n_cells: per_cell.len(),
            lambda,
            dev: &Device::Cpu,
            label: "Phase 2",
            gauge_fix: true,
            joint: false,
        },
        &cells,
        fold,
        None,
    )
    .expect("phase-2 SGD")
}
