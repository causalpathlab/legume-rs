use super::*;
use crate::data::Triplet;
use crate::fit::config::{TrackInfo, TrackSpec};
use crate::fit::hier::partition::{Partition, TrackSupport, UnitModules};
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
fn fixture() -> (UnitTable, Partition, UnitModules, TrackSupport, HierParams) {
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
    let sup = TrackSupport::new(&units.tracks, &part);
    (units, part, um, sup, params)
}

fn plan_all() -> StepPlan {
    // every unit paired with every module it has counts in, at full weight → deterministic
    StepPlan {
        units: vec![0, 1, 2],
        pairs_by_module: vec![
            ((0, 0), vec![(0, 1.0), (2, 1.0)]),
            ((0, 1), vec![(1, 1.0), (2, 1.0)]),
        ],
    }
}

fn total_loss(
    p: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    sup: &TrackSupport,
    plan: &StepPlan,
) -> f64 {
    let (s, _) = loss_and_grads(p, units, um, part, sup, plan, 0.0);
    s.loss_module + s.loss_gene
}

#[test]
fn analytic_gradients_match_finite_differences() {
    let (units, part, um, sup, mut p) = fixture();
    let plan = plan_all();
    let (_, g) = loss_and_grads(&p, &units, &um, &part, &sup, &plan, 0.0);
    assert_eq!(g.r.len(), 6);
    assert_eq!(g.b_g.len(), 6);
    let eps = 1e-3f32;
    let fd = |p: &mut HierParams, get: &dyn Fn(&mut HierParams) -> &mut f32| -> f64 {
        let x0 = *get(p);
        *get(p) = x0 + eps;
        let lp = total_loss(p, &units, &um, &part, &sup, &plan);
        *get(p) = x0 - eps;
        let lm = total_loss(p, &units, &um, &part, &sup, &plan);
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
        for (k, &want) in row.iter().enumerate() {
            let n = fd(&mut p, &|p| &mut p.r[gene as usize * 2 + k]);
            assert!(close(n, want), "r[{gene},{k}] fd {n} vs {want}");
        }
    }
    for &(gene, gb) in &g.b_g {
        let n = fd(&mut p, &|p| &mut p.b_g[gene as usize]);
        assert!(close(n, gb), "b_g[{gene}] fd {n} vs {gb}");
    }
}

#[test]
fn extreme_softmax_still_reports_a_finite_loss() {
    let (units, part, um, sup, mut p) = fixture();
    let plan = plan_all();
    // Scale e_u to force an extreme (near-degenerate) softmax, underflowing some
    // p toward 0 for a target with q > 0.
    for x in &mut p.e_u {
        *x *= 1e4;
    }
    let (s, _) = loss_and_grads(&p, &units, &um, &part, &sup, &plan, 0.0);
    assert!(s.loss_module.is_finite(), "loss_module: {}", s.loss_module);
    assert!(s.loss_gene.is_finite(), "loss_gene: {}", s.loss_gene);
}

#[test]
fn every_touched_gene_row_appears_once_and_untouched_genes_do_not() {
    let (units, part, um, sup, p) = fixture();
    let plan = StepPlan {
        units: vec![0],
        pairs_by_module: vec![((0, 0), vec![(0, 1.0)])],
    };
    let (_, g) = loss_and_grads(&p, &units, &um, &part, &sup, &plan, 0.0);
    let mut genes: Vec<u32> = g.r.iter().map(|(g, _)| *g).collect();
    genes.sort_unstable();
    assert_eq!(genes, vec![0, 1, 3]); // module 0's members
}

#[test]
fn pair_weight_scales_the_gene_level_term() {
    let (units, part, um, sup, p) = fixture();
    let plan_full = StepPlan {
        units: vec![0],
        pairs_by_module: vec![((0, 0), vec![(0, 1.0)])],
    };
    let plan_half = StepPlan {
        units: vec![0],
        pairs_by_module: vec![((0, 0), vec![(0, 0.5)])],
    };
    let (s_full, g_full) = loss_and_grads(&p, &units, &um, &part, &sup, &plan_full, 0.0);
    let (s_half, g_half) = loss_and_grads(&p, &units, &um, &part, &sup, &plan_half, 0.0);
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
    let (units, part, um, sup, mut p) = fixture();
    let plan = plan_all();
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(3, 0.2),
        mu: RowAdagrad::new(2, 0.2),
        r: RowAdagrad::new(6, 0.2),
        offsets: Vec::new(),
    };
    let before = total_loss(&p, &units, &um, &part, &sup, &plan);
    for _ in 0..20 {
        let (_, g) = loss_and_grads(&p, &units, &um, &part, &sup, &plan, 0.0);
        apply(&mut p, &mut opt, &g, &plan, 0.0);
    }
    let after = total_loss(&p, &units, &um, &part, &sup, &plan);
    assert!(after < before * 0.9, "{before} → {after}");
}

#[test]
fn apply_with_weight_decay_shrinks_rows_and_leaves_biases() {
    let (_units, _part, _um, _sup, mut p) = fixture();
    let plan = plan_all();
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(3, 0.2),
        mu: RowAdagrad::new(2, 0.2),
        r: RowAdagrad::new(6, 0.2),
        offsets: Vec::new(),
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
        offsets: Vec::new(),
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
    let sup = TrackSupport::new(&units.tracks, &part);
    let mut p = HierParams::new(u, m, d, h, 1);
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(u, 0.1),
        mu: RowAdagrad::new(m, 0.1),
        r: RowAdagrad::new(d, 0.1),
        offsets: Vec::new(),
    };
    let plan = StepPlan {
        units: (0..u as u32).collect(),
        pairs_by_module: (0..m as u32)
            .map(|mm| {
                (
                    (0, mm),
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
        let (_, g) = loss_and_grads(&p, &units, &um, &part, &sup, &plan, 0.0);
        apply(&mut p, &mut opt, &g, &plan, 0.0);
    }
    eprintln!(
        "step (B=256, M=128, D=34k, H=128, K=8): {:.1} ms",
        t0.elapsed().as_secs_f64() * 100.0
    );
}
///////////////////////////////////////
// The two-track fixture and its tests //
///////////////////////////////////////

/// Ridge strength the multi-track tests differentiate through.
const OFFSET_L2: f32 = 0.3;

/// Four units on a TWO-track axis over seven genes in three modules
/// (0: {0, 2, 3}, 1: {1, 4}, 2: {5, 6}). Rows 0..7 are genes 0..7 on track 0;
/// rows 7..11 are genes {0, 3, 5, 6} on track 1, so track 1's support puts
/// every interesting case in one fixture: gene 2 is a member of module 0, which
/// track 1 IS scored in, but track 1 has no row for it; module 1 is entirely
/// outside track 1's support; module 2 is fully supported. Units 0, 1 and 2
/// count on both tracks; unit 3 counts on track 0 only.
fn fixture_tracks() -> (UnitTable, Partition, UnitModules, TrackSupport, HierParams) {
    let l0 = vec![
        // track 0, rows 0..7 ≡ genes 0..7
        t(0, 0, 4.0),
        t(0, 1, 1.0),
        t(0, 3, 2.0),
        t(0, 5, 1.5),
        t(1, 2, 3.0),
        t(1, 4, 5.0),
        t(1, 6, 1.0),
        t(2, 0, 1.0),
        t(2, 2, 1.0),
        t(2, 3, 6.0),
        t(2, 5, 2.0),
        t(2, 6, 1.0),
        t(3, 1, 2.0),
        t(3, 4, 3.0),
        // track 1, row 7 = gene 0, 8 = gene 3, 9 = gene 5, 10 = gene 6
        t(0, 7, 2.0),
        t(0, 9, 3.0),
        t(1, 8, 1.0),
        t(1, 10, 4.0),
        t(2, 7, 1.0),
        t(2, 8, 2.0),
        t(2, 10, 2.0),
    ];
    let tracks = TrackSpec {
        track_of_row: vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        gene_of_row: vec![0, 1, 2, 3, 4, 5, 6, 0, 3, 5, 6],
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
    let units = UnitTable::from_pseudobulks_and_cells_tracked(&[&l0], &[4], &[], None, 11, tracks);
    let part = Partition::from_labels(&[0, 1, 0, 0, 1, 2, 2], 3);
    let um = UnitModules::new(&units, &part);
    let sup = TrackSupport::new(&units.tracks, &part);
    let mut params = HierParams::new_tracked(4, 3, 7, 2, 2, 11);
    // The offsets start at zero, where BOTH the ridge term and its gradient
    // vanish; move them off zero so the differences below can see them.
    let off = &mut params.offsets[0];
    for (i, x) in off.d_mu.iter_mut().enumerate() {
        *x = 0.07 * (i as f32 + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    for (i, x) in off.d_b_m.iter_mut().enumerate() {
        *x = 0.05 - 0.03 * i as f32;
    }
    for (i, x) in off.d_r.iter_mut().enumerate() {
        *x = 0.04 * ((i % 5) as f32 - 2.0);
    }
    for (i, x) in off.d_b_g.iter_mut().enumerate() {
        *x = 0.02 * ((i % 3) as f32 - 1.0);
    }
    (units, part, um, sup, params)
}

/// Genes track 1 has no row for.
const OFF_TRACK_GENES: [usize; 3] = [1, 2, 4];

/// Every `(track, module)` in the track's SUPPORT that a unit has mass in, at
/// full weight. `(1, 1)` is absent: module 1 = {1, 4} and track 1 has a row for
/// neither, so it is outside track 1's support entirely.
fn plan_all_tracks() -> StepPlan {
    StepPlan {
        units: vec![0, 1, 2, 3],
        pairs_by_module: vec![
            ((0, 0), vec![(0, 1.0), (1, 1.0), (2, 1.0)]),
            ((0, 1), vec![(0, 1.0), (1, 1.0), (3, 1.0)]),
            ((0, 2), vec![(0, 1.0), (1, 1.0), (2, 1.0)]),
            ((1, 0), vec![(0, 1.0), (1, 1.0), (2, 1.0)]),
            ((1, 2), vec![(0, 1.0), (1, 1.0), (2, 1.0)]),
        ],
    }
}

fn total_loss_tracks(
    p: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    sup: &TrackSupport,
    plan: &StepPlan,
) -> f64 {
    let (s, _) = loss_and_grads(p, units, um, part, sup, plan, OFFSET_L2);
    s.loss_module + s.loss_gene + s.loss_ridge
}

#[test]
fn the_support_is_the_tracks_own_rows() {
    let (units, part, _um, sup, _p) = fixture_tracks();
    assert!(sup.is_full(0), "track 0 has a row for every gene");
    assert!(!sup.is_full(1));
    // module 1 = {1, 4}: track 1 has neither, so it is not scored there
    assert_eq!(sup.modules_of(1), &[0, 2]);
    // module 0 = {0, 2, 3}: track 1 has rows for genes 0 and 3, at slots 0 and 2
    assert_eq!(sup.slots_of(1, 0), &[0, 2]);
    assert_eq!(sup.local_of(1, 0), &[0, u32::MAX, 1]);
    // module 2 = {5, 6}: both
    assert_eq!(sup.slots_of(1, 2), &[0, 1]);
    assert!(sup.slots_of(1, 1).is_empty());
    let _ = (units, part);
}

#[test]
fn analytic_gradients_match_finite_differences_with_tracks() {
    let (units, part, um, sup, mut p) = fixture_tracks();
    let plan = plan_all_tracks();
    let (_, g) = loss_and_grads(&p, &units, &um, &part, &sup, &plan, OFFSET_L2);
    let eps = 1e-3f32;
    let fd = |p: &mut HierParams, get: &dyn Fn(&mut HierParams) -> &mut f32| -> f64 {
        let x0 = *get(p);
        *get(p) = x0 + eps;
        let lp = total_loss_tracks(p, &units, &um, &part, &sup, &plan);
        *get(p) = x0 - eps;
        let lm = total_loss_tracks(p, &units, &um, &part, &sup, &plan);
        *get(p) = x0;
        (lp - lm) / (2.0 * eps as f64)
    };
    let close = |a: f64, b: f32| (a - f64::from(b)).abs() < 2e-3 * (1.0 + a.abs());

    // base tables
    for u in 0..4 {
        for k in 0..2 {
            let n = fd(&mut p, &|p| &mut p.e_u[u * 2 + k]);
            assert!(
                close(n, g.e_u[u * 2 + k]),
                "e_u[{u},{k}] fd {n} vs {}",
                g.e_u[u * 2 + k]
            );
        }
    }
    for m in 0..3 {
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
    for (gene, row) in &g.r {
        for (k, &want) in row.iter().enumerate() {
            let n = fd(&mut p, &|p| &mut p.r[*gene as usize * 2 + k]);
            assert!(close(n, want), "r[{gene},{k}] fd {n} vs {want}");
        }
    }
    for &(gene, gb) in &g.b_g {
        let n = fd(&mut p, &|p| &mut p.b_g[gene as usize]);
        assert!(close(n, gb), "b_g[{gene}] fd {n} vs {gb}");
    }

    // track 1's offset tables, ridge included
    let og = &g.offsets[0];
    for m in 0..3 {
        for k in 0..2 {
            let n = fd(&mut p, &|p| &mut p.offsets[0].d_mu[m * 2 + k]);
            assert!(
                close(n, og.mu[m * 2 + k]),
                "d_mu[{m},{k}] fd {n} vs {}",
                og.mu[m * 2 + k]
            );
        }
        let n = fd(&mut p, &|p| &mut p.offsets[0].d_b_m[m]);
        assert!(close(n, og.b_m[m]), "d_b_m[{m}] fd {n} vs {}", og.b_m[m]);
    }
    for gene in 0..7 {
        for k in 0..2 {
            let want = og.r[gene * 2 + k];
            let n = fd(&mut p, &|p| &mut p.offsets[0].d_r[gene * 2 + k]);
            assert!(close(n, want), "d_r[{gene},{k}] fd {n} vs {want}");
        }
        let n = fd(&mut p, &|p| &mut p.offsets[0].d_b_g[gene]);
        assert!(
            close(n, og.b_g[gene]),
            "d_b_g[{gene}] fd {n} vs {}",
            og.b_g[gene]
        );
    }
}

/// A track's rows ARE its feature axis: a gene with no row on track `t` is
/// outside that track's axis, not a permanent negative in it. Nothing on that
/// track may move it — not the base tables, not the offsets — and the module it
/// sits in gets no track-`t` term either when NO gene of that module is on the
/// track.
#[test]
fn a_gene_without_a_row_on_a_track_gets_no_gradient_from_that_track() {
    let (units, part, um, sup, p) = fixture_tracks();
    // Only track 1's groups, so every gradient below is track 1's alone.
    let track1_only = StepPlan {
        units: plan_all_tracks().units,
        pairs_by_module: plan_all_tracks()
            .pairs_by_module
            .into_iter()
            .filter(|&((tr, _), _)| tr == 1)
            .collect(),
    };
    // No ridge, so an offset row's only possible gradient is the data term.
    let (_, g) = loss_and_grads(&p, &units, &um, &part, &sup, &track1_only, 0.0);

    let base: std::collections::HashMap<u32, Vec<f32>> =
        g.r.iter().map(|(gene, row)| (*gene, row.clone())).collect();
    let base_b: std::collections::HashMap<u32, f32> = g.b_g.iter().copied().collect();
    for gene in OFF_TRACK_GENES {
        let k = gene as u32;
        assert!(
            base.get(&k).is_none_or(|row| row.iter().all(|&x| x == 0.0)),
            "gene {gene} took a base gradient from a track it has no row on: {:?}",
            base.get(&k)
        );
        assert_eq!(base_b.get(&k).copied().unwrap_or(0.0), 0.0, "b_g[{gene}]");
        let og = &g.offsets[0];
        assert!(
            og.r[gene * 2..gene * 2 + 2].iter().all(|&x| x == 0.0),
            "gene {gene} took an offset gradient: {:?}",
            &og.r[gene * 2..gene * 2 + 2]
        );
        assert_eq!(og.b_g[gene], 0.0, "d_b_g[{gene}]");
    }
    // Module 1 is outside track 1's support: no module-level term either.
    let og = &g.offsets[0];
    for k in 0..2 {
        assert_eq!(og.mu[2 + k], 0.0, "d_mu[module 1, {k}]");
    }
    assert_eq!(og.b_m[1], 0.0);
    // …while the supported genes and modules DO move, so this is not vacuous.
    assert_eq!(og.r.len(), 7 * 2, "the offset gene table is dense [G × H]");
    for gene in [0usize, 3, 5, 6] {
        assert!(
            og.r[gene * 2..gene * 2 + 2].iter().any(|&x| x != 0.0),
            "gene {gene} is on track 1 and must take a gradient"
        );
    }
    assert!(og.mu[0..2].iter().any(|&x| x != 0.0));
    assert!(og.mu[4..6].iter().any(|&x| x != 0.0));
}

#[test]
fn a_unit_absent_on_a_track_contributes_nothing_there() {
    let (units, part, um, sup, p) = fixture_tracks();
    assert_eq!(units.total_of(3, 1), 0.0);
    assert_eq!(units.weight_of(3, 1), 0.0);
    assert!(
        um.by_module[3].iter().all(|&((tr, _), _)| tr == 0),
        "no track-1 bucket for a unit with no counts there"
    );
    let plan = plan_all_tracks();
    let (_, g) = loss_and_grads(&p, &units, &um, &part, &sup, &plan, OFFSET_L2);
    // Move that unit's embedding: track 1 cannot see it (weight 0 ⇒ a zero
    // delta1 row, and no gene-level pair), so every track-1 offset gradient is
    // untouched while the base module table moves.
    let mut q = fixture_tracks().4;
    for k in 0..2 {
        q.e_u[3 * 2 + k] += 0.9;
    }
    let (_, g2) = loss_and_grads(&q, &units, &um, &part, &sup, &plan, OFFSET_L2);
    assert_eq!(g.offsets[0].mu, g2.offsets[0].mu);
    assert_eq!(g.offsets[0].b_m, g2.offsets[0].b_m);
    assert_eq!(g.offsets[0].r, g2.offsets[0].r);
    assert_eq!(g.offsets[0].b_g, g2.offsets[0].b_g);
    assert_ne!(g.mu, g2.mu, "the base table does see that unit on track 0");
}

#[test]
fn base_tables_get_the_sum_over_tracks_and_a_gene_row_appears_once() {
    let (units, part, um, sup, p) = fixture_tracks();
    let plan = plan_all_tracks();
    let (_, g) = loss_and_grads(&p, &units, &um, &part, &sup, &plan, 0.0);
    let mut keys: Vec<u32> = g.r.iter().map(|&(gene, _)| gene).collect();
    keys.sort_unstable();
    let mut uniq = keys.clone();
    uniq.dedup();
    assert_eq!(keys, uniq, "a gene row appears at most once in g.r");
    assert_eq!(uniq, vec![0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(g.b_g.len(), g.r.len());

    // The same plan restricted to one track at a time; the base tables must be
    // the sum of the two.
    let only = |tr: u32| StepPlan {
        units: plan.units.clone(),
        pairs_by_module: plan_all_tracks()
            .pairs_by_module
            .into_iter()
            .filter(|&((t, _), _)| t == tr)
            .collect(),
    };
    let per_track = |pl: &StepPlan| -> (Vec<Vec<f32>>, Vec<f32>) {
        let (_, gg) = loss_and_grads(&p, &units, &um, &part, &sup, pl, 0.0);
        let mut rows = vec![vec![0f32; 2]; 7];
        let mut bias = vec![0f32; 7];
        for (&(gene, ref row), &(_, gb)) in gg.r.iter().zip(&gg.b_g) {
            rows[gene as usize].clone_from(row);
            bias[gene as usize] = gb;
        }
        (rows, bias)
    };
    let (r0, b0) = per_track(&only(0));
    let (r1, b1) = per_track(&only(1));
    for (&(gene, ref row), &(_, gb)) in g.r.iter().zip(&g.b_g) {
        let i = gene as usize;
        for k in 0..2 {
            let want = r0[i][k] + r1[i][k];
            assert!(
                (row[k] - want).abs() < 1e-6,
                "r[{gene},{k}] {} vs {want}",
                row[k]
            );
        }
        assert!((gb - (b0[i] + b1[i])).abs() < 1e-6, "b_g[{gene}]");
    }
    // …and the second summand is not vacuous: every gene ON track 1's support
    // moves the base gradient relative to track 0 alone, and no gene off it does.
    let full: std::collections::HashMap<u32, Vec<f32>> =
        g.r.iter().map(|(gene, row)| (*gene, row.clone())).collect();
    for gene in [0usize, 3, 5, 6] {
        assert!(
            r1[gene].iter().any(|&x| x != 0.0),
            "gene {gene}: track 1 contributes nothing"
        );
        assert!(
            full[&(gene as u32)]
                .iter()
                .zip(&r0[gene])
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "gene {gene}: the base gradient is track 0's alone"
        );
    }
    for gene in OFF_TRACK_GENES {
        assert!(
            r1[gene].iter().all(|&x| x == 0.0),
            "gene {gene} is off track 1"
        );
        assert_eq!(full[&(gene as u32)], r0[gene]);
    }
}

/// The base gene tables must not depend on the order the plan lists its groups.
/// In particular a module can be claimed first by a RESTRICTED track, and the
/// members that track has no row for then arrive on a later group — module 0's
/// gene 2 here, which track 1 lacks.
#[test]
fn a_module_claimed_by_a_restricted_track_first_still_collects_every_gene() {
    let (units, part, um, sup, p) = fixture_tracks();
    let mut reordered = plan_all_tracks();
    reordered.pairs_by_module.reverse(); // track 1's groups claim their modules first
    assert_eq!(reordered.pairs_by_module[0].0, (1, 2));
    let by_gene = |pl: &StepPlan| -> (Vec<Vec<f32>>, Vec<f32>) {
        let (_, gg) = loss_and_grads(&p, &units, &um, &part, &sup, pl, 0.0);
        let mut rows = vec![Vec::new(); 7];
        let mut bias = vec![f32::NAN; 7];
        for (&(gene, ref row), &(_, gb)) in gg.r.iter().zip(&gg.b_g) {
            assert!(rows[gene as usize].is_empty(), "gene {gene} appears twice");
            rows[gene as usize].clone_from(row);
            bias[gene as usize] = gb;
        }
        (rows, bias)
    };
    let (r_sorted, b_sorted) = by_gene(&plan_all_tracks());
    let (r_rev, b_rev) = by_gene(&reordered);
    for gene in 0..7usize {
        assert_eq!(
            r_sorted[gene].len(),
            2,
            "gene {gene} is missing from the sorted plan's gradient"
        );
        assert_eq!(
            r_rev[gene].len(),
            2,
            "gene {gene} was dropped when a restricted track claimed its module first"
        );
        for k in 0..2 {
            assert!(
                (r_sorted[gene][k] - r_rev[gene][k]).abs() < 1e-6,
                "r[{gene},{k}] {} vs {}",
                r_sorted[gene][k],
                r_rev[gene][k]
            );
        }
        assert!((b_sorted[gene] - b_rev[gene]).abs() < 1e-6, "b_g[{gene}]");
    }
}

#[test]
fn offset_ridge_is_mean_row_norm() {
    let (units, part, um, sup, p) = fixture_tracks();
    // No unit, no pair: the data term is exactly zero and only the ridge is left.
    let empty = StepPlan {
        units: Vec::new(),
        pairs_by_module: Vec::new(),
    };
    let o = &p.offsets[0];
    let sq = |v: &[f32]| v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
    let per_epoch = f64::from(OFFSET_L2) * (sq(&o.d_mu) / 3.0 + sq(&o.d_r) / 7.0);
    // `offset_l2` is a per-EPOCH weight: one step carries `λ/S` of it, so an
    // epoch of `S` steps sums to exactly the per-epoch figure whatever `S` is.
    // `S == 1` is the one-step-per-epoch case and keeps the plain numbers.
    for steps_per_epoch in [1usize, 4] {
        let lam = crate::fit::hier::train::per_step_offset_l2(OFFSET_L2, steps_per_epoch);
        let (s, g) = loss_and_grads(&p, &units, &um, &part, &sup, &empty, lam);
        assert_eq!(s.loss_module, 0.0);
        assert_eq!(s.loss_gene, 0.0);
        let want = f64::from(lam) * (sq(&o.d_mu) / 3.0 + sq(&o.d_r) / 7.0);
        assert!(
            (s.loss_ridge - want).abs() < 1e-9,
            "S={steps_per_epoch}: {} vs {want}",
            s.loss_ridge
        );
        assert!(
            (s.loss_ridge * steps_per_epoch as f64 - per_epoch).abs() < 1e-6,
            "S={steps_per_epoch}: an epoch sums to {} not {per_epoch}",
            s.loss_ridge * steps_per_epoch as f64
        );
        let og = &g.offsets[0];
        assert_eq!(og.r.len(), 7 * 2, "the ridge covers every gene row");
        for i in 0..o.d_mu.len() {
            let want = 2.0 * lam * o.d_mu[i] / 3.0;
            assert!(
                (og.mu[i] - want).abs() < 1e-7,
                "S={steps_per_epoch}: d_mu[{i}] {} vs {want}",
                og.mu[i]
            );
        }
        for (i, &got) in og.r.iter().enumerate() {
            let want = 2.0 * lam * o.d_r[i] / 7.0;
            assert!(
                (got - want).abs() < 1e-7,
                "S={steps_per_epoch}: d_r[{}, {}] {got} vs {want}",
                i / 2,
                i % 2
            );
        }
        assert!(
            og.b_m.iter().all(|&x| x == 0.0),
            "module biases are unpenalised"
        );
        assert!(
            og.b_g.iter().all(|&x| x == 0.0),
            "gene biases are unpenalised"
        );
    }
}

/// The offset tables are shrunk by their own exact ridge, already summed into
/// `Grads::offsets`, so `apply` must never also decay them — that would be a
/// second, undeclared penalty on top of it. The base tables still decay.
#[test]
fn apply_never_decays_the_offset_tables() {
    let (_units, _part, _um, _sup, mut p) = fixture_tracks();
    let plan = plan_all_tracks();
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(4, 0.2),
        mu: RowAdagrad::new(3, 0.2),
        r: RowAdagrad::new(7, 0.2),
        offsets: vec![(RowAdagrad::new(3, 0.2), RowAdagrad::new(7, 0.2))],
    };
    let before = p.offsets[0].clone();
    let mu_before = p.mu.clone();
    // Every gradient zero: only weight decay can move anything.
    let zero = Grads {
        e_u: vec![0.0; plan.units.len() * p.h],
        mu: vec![0.0; 3 * p.h],
        b_m: vec![0.0; 3],
        r: (0..7u32).map(|g| (g, vec![0.0; p.h])).collect(),
        b_g: (0..7u32).map(|g| (g, 0.0)).collect(),
        offsets: vec![TrackGrads {
            mu: vec![0.0; 3 * p.h],
            b_m: vec![0.0; 3],
            r: vec![0.0; 7 * p.h],
            b_g: vec![0.0; 7],
        }],
    };
    apply(&mut p, &mut opt, &zero, &plan, 0.5);
    assert_eq!(p.offsets[0].d_mu, before.d_mu);
    assert_eq!(p.offsets[0].d_b_m, before.d_b_m);
    assert_eq!(p.offsets[0].d_r, before.d_r);
    assert_eq!(p.offsets[0].d_b_g, before.d_b_g);
    assert_ne!(p.mu, mu_before, "the base tables still decay");
}

#[test]
#[ignore = "timing; run by hand with --ignored --nocapture"]
fn production_shape_step_time_two_tracks() {
    use std::time::Instant;
    let (u, m, d, h) = (256usize, 128usize, 34_000usize, 128usize);
    // The same shape as `production_shape_step_time`, over two tracks that both
    // cover every gene — the unrestricted path on both, so this is the upper
    // bound the support rule can never exceed.
    let labels: Vec<u32> = (0..d as u32).map(|g| g % m as u32).collect();
    let part = Partition::from_labels(&labels, m);
    let mut trip = Vec::new();
    for uu in 0..u as u32 {
        for j in 0..2000u32 {
            let gene = (uu * 7919 + j * 104_729) % d as u32;
            trip.push(t(uu, gene, 1.0 + (j % 5) as f32));
            trip.push(t(uu, gene + d as u32, 1.0 + (j % 3) as f32));
        }
    }
    let tracks = TrackSpec {
        track_of_row: (0..2 * d).map(|r| u32::from(r >= d)).collect(),
        gene_of_row: (0..2 * d).map(|r| (r % d) as u32).collect(),
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
    let units =
        UnitTable::from_pseudobulks_and_cells_tracked(&[&trip], &[u], &[], None, 2 * d, tracks);
    let um = UnitModules::new(&units, &part);
    let sup = TrackSupport::new(&units.tracks, &part);
    let mut p = HierParams::new_tracked(u, m, d, 2, h, 1);
    let mut opt = Optimizers {
        e_u: RowAdagrad::new(u, 0.1),
        mu: RowAdagrad::new(m, 0.1),
        r: RowAdagrad::new(d, 0.1),
        offsets: vec![(RowAdagrad::new(m, 0.1), RowAdagrad::new(d, 0.1))],
    };
    let plan = StepPlan {
        units: (0..u as u32).collect(),
        pairs_by_module: (0..2u32)
            .flat_map(|tr| {
                (0..m as u32).map(move |mm| {
                    (
                        (tr, mm),
                        (0..u as u32)
                            .filter(|x| (x + mm) % 16 == 0)
                            .map(|x| (x, 0.125))
                            .collect(),
                    )
                })
            })
            .collect(), // K = 8 pairs per unit per track
    };
    let t0 = Instant::now();
    for _ in 0..10 {
        let (_, g) = loss_and_grads(&p, &units, &um, &part, &sup, &plan, 0.1);
        apply(&mut p, &mut opt, &g, &plan, 0.0);
    }
    eprintln!(
        "step T=2 (B=256, M=128, D=34k genes × 2 tracks, H=128, K=8): {:.1} ms",
        t0.elapsed().as_secs_f64() * 100.0
    );
}
