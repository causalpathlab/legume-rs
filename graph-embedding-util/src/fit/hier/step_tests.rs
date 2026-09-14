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
    // every unit paired with every module it has counts in → deterministic
    StepPlan {
        units: vec![0, 1, 2],
        pairs_by_module: vec![(0, vec![0, 2]), (1, vec![1, 2])],
    }
}

fn total_loss(
    p: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    plan: &StepPlan,
) -> f64 {
    let (s, _) = loss_and_grads(p, units, um, part, plan, 1);
    s.loss_module + s.loss_gene
}

#[test]
fn analytic_gradients_match_finite_differences() {
    let (units, part, um, mut p) = fixture();
    let plan = plan_all();
    let (_, g) = loss_and_grads(&p, &units, &um, &part, &plan, 1);
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
        pairs_by_module: vec![(0, vec![0])],
    };
    let (s, g) = loss_and_grads(&p, &units, &um, &part, &plan, 1);
    assert_eq!(s.n_pairs, 1);
    let mut genes: Vec<u32> = g.r.iter().map(|(g, _)| *g).collect();
    genes.sort_unstable();
    assert_eq!(genes, vec![0, 1, 3]); // module 0's members
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
        let (_, g) = loss_and_grads(&p, &units, &um, &part, &plan, 1);
        apply(&mut p, &mut opt, &g, &plan, 0.0);
    }
    let after = total_loss(&p, &units, &um, &part, &plan);
    assert!(after < before * 0.9, "{before} → {after}");
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
    let p = HierParams::new(u, m, d, h, 1);
    let plan = StepPlan {
        units: (0..u as u32).collect(),
        pairs_by_module: (0..m as u32)
            .map(|mm| (mm, (0..u as u32).filter(|x| (x + mm) % 16 == 0).collect()))
            .collect(), // K = 8 pairs per unit
    };
    let t0 = Instant::now();
    for _ in 0..10 {
        let _ = loss_and_grads(&p, &units, &um, &part, &plan, 8);
    }
    eprintln!(
        "step (B=256, M=128, D=34k, H=128, K=8): {:.1} ms",
        t0.elapsed().as_secs_f64() * 100.0
    );
}
