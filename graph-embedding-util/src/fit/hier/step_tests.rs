use super::*;
use crate::data::Triplet;
use crate::fit::hier::partition::{Partition, UnitModules};
use crate::fit::hier::units::UnitTable;

fn t(cell: u32, feature: u32, count: f32) -> Triplet {
    Triplet {
        cell,
        feature,
        count,
    }
}

/// Three units, two modules, six genes, H = 2 — every quantity small enough
/// to difference numerically.
fn fixture() -> (UnitTable, Partition, UnitModules, HierParams) {
    let l0 = vec![
        t(0, 0, 4.0),
        t(0, 1, 1.0),
        t(0, 3, 2.0),
        t(1, 2, 3.0),
        t(1, 4, 5.0),
        t(1, 5, 1.0),
        t(2, 0, 1.0),
        t(2, 2, 1.0),
        t(2, 3, 6.0),
        t(2, 5, 2.0),
    ];
    let units = UnitTable::from_pseudobulks_and_cells(&[&l0], &[3], &[], None, 6);
    let part = Partition::from_labels(&[0, 0, 1, 0, 1, 1], 2);
    let um = UnitModules::new(&units, &part);
    let params = HierParams::new(3, 2, 6, 2, 11);
    (units, part, um, params)
}

fn plan_all() -> StepPlan {
    // every unit paired with every module it has counts in, at full weight → deterministic
    StepPlan {
        units: vec![0, 1, 2],
        pairs_by_module: vec![(0, vec![(0, 1.0), (2, 1.0)]), (1, vec![(1, 1.0), (2, 1.0)])],
    }
}

fn total_loss(
    p: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    plan: &StepPlan,
) -> f64 {
    let (s, _) = loss_and_grads(p, units, um, part, plan);
    s.loss_module + s.loss_gene
}

#[test]
fn analytic_gradients_match_finite_differences() {
    let (units, part, um, mut p) = fixture();
    let plan = plan_all();
    let (_, g) = loss_and_grads(&p, &units, &um, &part, &plan);
    assert_eq!(g.r.len(), 6);
    assert_eq!(g.b_g.len(), 6);
    let eps = 1e-3f32;
    let fd = |p: &mut HierParams, get: &dyn Fn(&mut HierParams) -> &mut f32| -> f64 {
        let x0 = *get(p);
        *get(p) = x0 + eps;
        let lp = total_loss(p, &units, &um, &part, &plan);
        *get(p) = x0 - eps;
        let lm = total_loss(p, &units, &um, &part, &plan);
        *get(p) = x0;
        (lp - lm) / (2.0 * eps as f64)
    };
    let close = |a: f64, b: f32| (a - f64::from(b)).abs() < 2e-3 * (1.0 + a.abs());
    for u in 0..3 {
        for k in 0..2 {
            let n = fd(&mut p, &|p| &mut p.e_u[u * 2 + k]);
            assert!(
                close(n, g.e_u[u * 2 + k]),
                "e_u[{u},{k}] fd {n} vs {}",
                g.e_u[u * 2 + k]
            );
        }
    }
    for m in 0..2 {
        for k in 0..2 {
            let n = fd(&mut p, &|p| &mut p.mu[m * 2 + k]);
            assert!(
                close(n, g.mu[m * 2 + k]),
                "mu[{m},{k}] fd {n} vs {}",
                g.mu[m * 2 + k]
            );
        }
        let n = fd(&mut p, &|p| &mut p.b_m[m]);
        assert!(close(n, g.b_m[m]), "b_m[{m}] fd {n} vs {}", g.b_m[m]);
    }
    for &(gene, ref row) in &g.r {
        for k in 0..2 {
            let n = fd(&mut p, &|p| &mut p.r[gene as usize * 2 + k]);
            assert!(close(n, row[k]), "r[{gene},{k}] fd {n} vs {}", row[k]);
        }
    }
    for &(gene, gb) in &g.b_g {
        let n = fd(&mut p, &|p| &mut p.b_g[gene as usize]);
        assert!(close(n, gb), "b_g[{gene}] fd {n} vs {gb}");
    }
}

#[test]
fn every_touched_gene_row_appears_once_and_untouched_genes_do_not() {
    let (units, part, um, p) = fixture();
    let plan = StepPlan {
        units: vec![0],
        pairs_by_module: vec![(0, vec![(0, 1.0)])],
    };
    let (s, g) = loss_and_grads(&p, &units, &um, &part, &plan);
    assert_eq!(s.n_pairs, 1);
    let mut genes: Vec<u32> = g.r.iter().map(|(g, _)| *g).collect();
    genes.sort_unstable();
    assert_eq!(genes, vec![0, 1, 3]); // module 0's members
}

#[test]
fn pair_weight_scales_the_gene_level_term() {
    let (units, part, um, p) = fixture();
    let plan_full = StepPlan {
        units: vec![0],
        pairs_by_module: vec![(0, vec![(0, 1.0)])],
    };
    let plan_half = StepPlan {
        units: vec![0],
        pairs_by_module: vec![(0, vec![(0, 0.5)])],
    };
    let (s_full, g_full) = loss_and_grads(&p, &units, &um, &part, &plan_full);
    let (s_half, g_half) = loss_and_grads(&p, &units, &um, &part, &plan_half);
    assert!((s_half.loss_gene - 0.5 * s_full.loss_gene).abs() < 1e-6);
    assert_eq!(s_full.loss_module, s_half.loss_module);
    assert_eq!(g_full.r.len(), g_half.r.len());
    for (a, b) in g_full.r.iter().zip(&g_half.r) {
        assert_eq!(a.0, b.0);
        for k in 0..2 {
            assert!(
                (b.1[k] - 0.5 * a.1[k]).abs() < 1e-6,
                "gene {} k {}: full {} vs half {}",
                a.0,
                k,
                a.1[k],
                b.1[k]
            );
        }
    }
}

#[test]
fn twenty_steps_on_the_full_plan_lower_the_loss() {
    let (units, part, um, mut p) = fixture();
    let plan = plan_all();
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(3, 0.2),
        mu: RowAdagrad::new(2, 0.2),
        r: RowAdagrad::new(6, 0.2),
    };
    let before = total_loss(&p, &units, &um, &part, &plan);
    for _ in 0..20 {
        let (_, g) = loss_and_grads(&p, &units, &um, &part, &plan);
        apply(&mut p, &mut opt, &g, &plan, 0.0);
    }
    let after = total_loss(&p, &units, &um, &part, &plan);
    assert!(after < before * 0.9, "{before} → {after}");
}

#[test]
fn apply_with_weight_decay_shrinks_rows_and_leaves_biases() {
    let (_units, _part, _um, mut p) = fixture();
    let plan = plan_all();
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(3, 0.2),
        mu: RowAdagrad::new(2, 0.2),
        r: RowAdagrad::new(6, 0.2),
    };
    let e_u_before = p.e_u.clone();
    let mu_before = p.mu.clone();
    let b_m_before = p.b_m.clone();
    let r_before = p.r.clone();
    let b_g_before = p.b_g.clone();

    // Gradient zeroed by construction, shaped exactly for `plan`: all three
    // units at the module level, all six genes at the gene level (module 0
    // owns {0,1,3}, module 1 owns {2,4,5}).
    let zero_grads = Grads {
        e_u: vec![0.0; plan.units.len() * p.h],
        mu: vec![0.0; 2 * p.h],
        b_m: vec![0.0; 2],
        r: (0..6u32).map(|g| (g, vec![0.0; p.h])).collect(),
        b_g: (0..6u32).map(|g| (g, 0.0)).collect(),
    };

    let wd = 0.5f32;
    apply(&mut p, &mut opt, &zero_grads, &plan, wd);

    let f = 1.0 - 0.2f32 * wd; // 0.9
    for (a, b) in e_u_before.iter().zip(&p.e_u) {
        assert!((b - a * f).abs() < 1e-6, "{a} -> {b}, expected {}", a * f);
    }
    for (a, b) in mu_before.iter().zip(&p.mu) {
        assert!((b - a * f).abs() < 1e-6, "{a} -> {b}, expected {}", a * f);
    }
    for (a, b) in r_before.iter().zip(&p.r) {
        assert!((b - a * f).abs() < 1e-6, "{a} -> {b}, expected {}", a * f);
    }
    assert_eq!(p.b_m, b_m_before);
    assert_eq!(p.b_g, b_g_before);
}

#[test]
#[ignore = "timing; run by hand with --ignored --nocapture"]
fn production_shape_step_time() {
    use std::time::Instant;
    let (u, m, d, h) = (256usize, 128usize, 34_000usize, 128usize);
    // synthetic units: 2000 counted genes each, spread over modules
    let labels: Vec<u32> = (0..d as u32).map(|g| g % m as u32).collect();
    let part = Partition::from_labels(&labels, m);
    let mut trip = Vec::new();
    for uu in 0..u as u32 {
        for j in 0..2000u32 {
            trip.push(t(
                uu,
                (uu * 7919 + j * 104_729) % d as u32,
                1.0 + (j % 5) as f32,
            ));
        }
    }
    let units = UnitTable::from_pseudobulks_and_cells(&[&trip], &[u], &[], None, d);
    let um = UnitModules::new(&units, &part);
    let mut p = HierParams::new(u, m, d, h, 1);
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(u, 0.1),
        mu: RowAdagrad::new(m, 0.1),
        r: RowAdagrad::new(d, 0.1),
    };
    let plan = StepPlan {
        units: (0..u as u32).collect(),
        pairs_by_module: (0..m as u32)
            .map(|mm| {
                (
                    mm,
                    (0..u as u32)
                        .filter(|x| (x + mm) % 16 == 0)
                        .map(|x| (x, 0.125))
                        .collect(),
                )
            })
            .collect(), // K = 8 pairs per unit
    };
    let t0 = Instant::now();
    for _ in 0..10 {
        let (_, g) = loss_and_grads(&p, &units, &um, &part, &plan);
        apply(&mut p, &mut opt, &g, &plan, 0.0);
    }
    eprintln!(
        "step (B=256, M=128, D=34k, H=128, K=8): {:.1} ms",
        t0.elapsed().as_secs_f64() * 100.0
    );
}
