use super::*;
use crate::data::Triplet;
use crate::fit::config::{TrackInfo, TrackSpec};
use crate::fit::hier::units::UnitTable;
use std::sync::atomic::AtomicBool;

fn t(cell: u32, feature: u32, count: f32) -> Triplet {
    Triplet {
        cell,
        feature,
        count,
    }
}

/// Two planted programs: units 0..10 count genes 0..10, units 10..20 count
/// genes 10..20 (with a little of the other). Training must separate the
/// two unit groups and put each gene's row on its program's side.
#[test]
fn planted_programs_separate_units_and_genes() {
    let mut trip = Vec::new();
    for u in 0..20u32 {
        let own = if u < 10 { 0..10u32 } else { 10..20u32 };
        let other = if u < 10 { 10..20u32 } else { 0..10u32 };
        for g in own {
            trip.push(t(u, g, 20.0 + (g % 3) as f32));
        }
        for g in other {
            trip.push(t(u, g, 1.0));
        }
    }
    let units = UnitTable::from_pseudobulks_and_cells(&[&trip], &[20], &[], None, 20);
    let labels: Vec<u32> = (0..20u32).map(|g| if g < 10 { 0 } else { 1 }).collect();
    let cfg = HierConfig {
        n_modules: 2,
        epochs: 200,
        units_per_step: 8,
        modules_per_unit: 2,
        lr: 0.1,
        weight_decay: 0.0,
        seed: 3,
        offset_l2: 0.0,
    };
    let stop = AtomicBool::new(false);
    let out = train(&units, &labels, 4, &cfg, &stop).unwrap();
    assert_eq!(out.rho.nrows(), 20);
    let cos = |a: &[f32], b: &[f32]| {
        let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        d / (a.iter().map(|x| x * x).sum::<f32>().sqrt()
            * b.iter().map(|x| x * x).sum::<f32>().sqrt())
        .max(1e-12)
    };
    // unit 0 and unit 1 are alike; unit 0 and unit 15 are not
    let e = |u: usize| out.e_u.row(u).iter().copied().collect::<Vec<_>>();
    assert!(cos(&e(0), &e(1)) > cos(&e(0), &e(15)) + 0.3);
    // gene 3 scores higher against unit 0 than against unit 15
    let r = |g: usize| out.rho.row(g).iter().copied().collect::<Vec<_>>();
    let score =
        |g: usize, u: usize| r(g).iter().zip(e(u)).map(|(a, b)| a * b).sum::<f32>() + out.b_feat[g];
    assert!(score(3, 0) > score(3, 15));
    assert!(out.final_loss_per_unit.is_finite());
}

#[test]
fn the_stop_flag_ends_training_early_with_finite_output() {
    let trip = vec![t(0, 0, 3.0), t(0, 1, 1.0), t(1, 1, 4.0)];
    let units = UnitTable::from_pseudobulks_and_cells(&[&trip], &[2], &[], None, 2);
    let cfg = HierConfig {
        n_modules: 1,
        epochs: 1000,
        units_per_step: 2,
        modules_per_unit: 1,
        lr: 0.1,
        weight_decay: 0.0,
        seed: 1,
        offset_l2: 0.0,
    };
    let stop = AtomicBool::new(true);
    let out = train(&units, &[0, 0], 2, &cfg, &stop).unwrap();
    assert!(out.rho.iter().all(|v| v.is_finite()));
}

/// A unit with a non-zero composition on a track gets pair weights that sum to
/// 1 across every module it lands in ON THAT TRACK; a (unit, track) with an
/// all-zero composition is dropped from the plan entirely.
#[test]
fn draw_plan_weights_sum_to_one_per_unit() {
    let (n_m, n_t) = (3usize, 2usize);
    // (unit 0, track 0): [0.5, 0.5, 0.0]; (unit 0, track 1): [0.0, 0.0, 1.0];
    // (unit 1, track 0): all-zero (nothing counted there); (unit 1, track 1): [0.25, 0.75, 0.0]
    let um = UnitModules {
        n_tracks: n_t,
        n_modules: n_m,
        q: vec![
            0.5, 0.5, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, //
            0.25, 0.75, 0.0,
        ],
        n_um: vec![0.0; 2 * n_t * n_m],
        by_module: vec![Vec::new(), Vec::new()],
    };
    let mut rng = StdRng::seed_from_u64(7);
    let pickers = module_pickers(&um, n_m);
    assert_eq!(pickers.len(), 2 * n_t);
    let plan = draw_plan(&[0, 1], &pickers, n_m, n_t, 10, &mut rng);
    let mut sum_by_unit_track = std::collections::HashMap::new();
    for ((t, _), pairs) in &plan.pairs_by_module {
        for &(u, w) in pairs {
            *sum_by_unit_track.entry((u, *t)).or_insert(0.0f32) += w;
        }
    }
    for key in [(0u32, 0u32), (0, 1), (1, 1)] {
        let got = sum_by_unit_track.get(&key).copied().unwrap_or(0.0);
        assert!((got - 1.0).abs() < 1e-6, "{key:?} summed to {got}");
    }
    assert!(!sum_by_unit_track.contains_key(&(1, 0)));
}

/// The base track spec is what the untracked constructor builds, so the same
/// fixture trained through either one must come out identical — every table,
/// exactly, not merely close.
#[test]
fn single_track_output_is_identical_through_both_constructors() {
    let trip = vec![
        t(0, 0, 5.0),
        t(0, 1, 2.0),
        t(0, 3, 1.0),
        t(1, 1, 4.0),
        t(1, 2, 3.0),
        t(2, 0, 1.0),
        t(2, 3, 6.0),
    ];
    let cfg = HierConfig {
        n_modules: 2,
        epochs: 25,
        units_per_step: 2,
        modules_per_unit: 2,
        lr: 0.1,
        weight_decay: 0.0,
        seed: 17,
        offset_l2: 0.0,
    };
    let labels = vec![0u32, 0, 1, 1];
    let stop = AtomicBool::new(false);
    let plain = UnitTable::from_pseudobulks_and_cells(&[&trip], &[3], &[], None, 4);
    let tracked = UnitTable::from_pseudobulks_and_cells_tracked(
        &[&trip],
        &[3],
        &[],
        None,
        4,
        TrackSpec::base(4),
    );
    let a = train(&plain, &labels, 3, &cfg, &stop).unwrap();
    let b = train(&tracked, &labels, 3, &cfg, &stop).unwrap();
    assert_eq!(a.e_u, b.e_u);
    assert_eq!(a.rho, b.rho);
    assert_eq!(a.b_feat, b.b_feat);
}

/// Two tracks of the same 20 genes, with the programs SWAPPED on track 1:
/// units 0..10 count genes 0..10 on track 0 and genes 10..20 on track 1. A
/// gene's track-1 row must therefore sit on the other unit group's side, and
/// the move from its track-0 row must point along the planted contrast between
/// the two unit groups' embeddings.
#[test]
fn planted_two_track_programs() {
    let (n_g, n_u) = (20u32, 20u32);
    let mut trip = Vec::new();
    for u in 0..n_u {
        let own = if u < 10 { 0..10u32 } else { 10..20u32 };
        let other = if u < 10 { 10..20u32 } else { 0..10u32 };
        for g in own.clone() {
            trip.push(t(u, g, 20.0 + (g % 3) as f32));
        }
        for g in other.clone() {
            trip.push(t(u, g, 1.0));
            // track 1 swaps the two programs: rows n_g..2·n_g are the same genes
            trip.push(t(u, g + n_g, 20.0 + (g % 3) as f32));
        }
        for g in own {
            trip.push(t(u, g + n_g, 1.0));
        }
    }
    let n_features = 2 * n_g as usize;
    let tracks = TrackSpec {
        track_of_row: (0..n_features)
            .map(|r| (r >= n_g as usize) as u32)
            .collect(),
        gene_of_row: (0..n_features).map(|r| (r % n_g as usize) as u32).collect(),
        tracks: vec![
            TrackInfo {
                name: "t0".into(),
                is_count: true,
            },
            TrackInfo {
                name: "t1".into(),
                is_count: true,
            },
        ],
    };
    let units = UnitTable::from_pseudobulks_and_cells_tracked(
        &[&trip],
        &[n_u as usize],
        &[],
        None,
        n_features,
        tracks,
    );
    let labels: Vec<u32> = (0..n_g).map(|g| u32::from(g >= 10)).collect();
    let cfg = HierConfig {
        n_modules: 2,
        epochs: 200,
        units_per_step: 8,
        modules_per_unit: 2,
        lr: 0.1,
        weight_decay: 0.0,
        seed: 3,
        offset_l2: 0.01,
    };
    let stop = AtomicBool::new(false);
    let out = train(&units, &labels, 4, &cfg, &stop).unwrap();
    assert_eq!(out.rho.nrows(), n_features);
    assert_eq!(out.b_feat.len(), n_features);

    // The planted contrast: where the second unit group sits, minus the first.
    let group_mean = |lo: usize, hi: usize| -> Vec<f32> {
        (0..4)
            .map(|k| (lo..hi).map(|u| out.e_u[(u, k)]).sum::<f32>() / (hi - lo) as f32)
            .collect()
    };
    let (a, b) = (group_mean(0, 10), group_mean(10, 20));
    let planted: Vec<f32> = b.iter().zip(&a).map(|(x, y)| x - y).collect();
    let cos = |x: &[f32], y: &[f32]| {
        let d: f32 = x.iter().zip(y).map(|(p, q)| p * q).sum();
        d / (x.iter().map(|p| p * p).sum::<f32>().sqrt()
            * y.iter().map(|q| q * q).sum::<f32>().sqrt())
        .max(1e-12)
    };
    for gene in [0usize, 3, 7] {
        let shift: Vec<f32> = (0..4)
            .map(|k| out.rho[(gene + n_g as usize, k)] - out.rho[(gene, k)])
            .collect();
        let c = cos(&shift, &planted);
        assert!(c > 0.5, "gene {gene}: track-1 shift cosine {c}");
    }
}

/// `HierConfig::offset_l2` is a per-EPOCH weight. A step carries `1/S` of it, so
/// the ridge pulls equally hard over an epoch whatever the batch size — without
/// this, halving `units_per_step` would silently double the penalty and inflate
/// every offset row's Adagrad accumulator twice as fast.
#[test]
fn the_ridge_weight_is_spread_over_the_epochs_steps() {
    assert_eq!(per_step_offset_l2(0.8, 1), 0.8);
    assert_eq!(per_step_offset_l2(0.8, 4), 0.2);
    // 20 steps of the 4-step weight is 5 epochs' worth, not 20.
    assert!((per_step_offset_l2(0.8, 4) * 4.0 - 0.8).abs() < 1e-7);
    // A degenerate epoch (no unit, so no step) must not divide by zero.
    assert_eq!(per_step_offset_l2(0.8, 0), 0.8);
    assert_eq!(per_step_offset_l2(0.0, 7), 0.0);
}
