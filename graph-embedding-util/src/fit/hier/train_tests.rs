use super::*;
use crate::data::Triplet;
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
    };
    let stop = AtomicBool::new(true);
    let out = train(&units, &[0, 0], 2, &cfg, &stop).unwrap();
    assert!(out.rho.iter().all(|v| v.is_finite()));
}

/// A unit with a non-zero composition gets pair weights that sum to 1 across
/// every module it lands in; a unit with an all-zero composition (no counted
/// genes at all) is dropped from the plan entirely.
#[test]
fn draw_plan_weights_sum_to_one_per_unit() {
    let n_m = 3usize;
    // unit 0: composition [0.5, 0.5, 0.0]; unit 1: all-zero (e.g. an empty pb row)
    let um = UnitModules {
        q: vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        n_um: vec![0.0; 2 * n_m],
        by_module: vec![Vec::new(), Vec::new()],
    };
    let mut rng = StdRng::seed_from_u64(7);
    let plan = draw_plan(&[0, 1], &um, n_m, 10, &mut rng);
    let mut sum_by_unit = std::collections::HashMap::new();
    for (_, pairs) in &plan.pairs_by_module {
        for &(u, w) in pairs {
            *sum_by_unit.entry(u).or_insert(0.0f32) += w;
        }
    }
    assert!((sum_by_unit.get(&0).copied().unwrap_or(0.0) - 1.0).abs() < 1e-6);
    assert!(!sum_by_unit.contains_key(&1));
}
