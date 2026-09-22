use super::*;
use crate::data::Triplet;
use crate::fit::config::{TrackInfo, TrackSpec};
use crate::fit::hier::params::{HostOffset, PresetGenes, PresetMode, PresetOffsets};
use crate::fit::hier::partition::{Partition, TrackSupport, UnitModules};
use crate::fit::hier::units::UnitTable;
use crate::LoraSpec;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::candle::convert::to_host;

fn t(cell: u32, feature: u32, count: f32) -> Triplet {
    Triplet {
        cell,
        feature,
        count,
    }
}

const OFFSET_L2: f32 = 0.3;

////////////////////////////////////////////////////////////////////////
// An independent reference: the module docs' formula in f64 loops    //
////////////////////////////////////////////////////////////////////////

struct Host {
    h: usize,
    e_u: Vec<f32>,
    mu: Vec<f32>,
    b_m: Vec<f32>,
    r: Vec<f32>,
    b_g: Vec<f32>,
    /// Per non-base track, with the dense gene offset `δ₀ + u·V` third.
    offsets: Vec<HostOffset>,
    /// Per non-base track, the trained part `u·V` alone: what the ridge sees.
    resid: Vec<Vec<f32>>,
}

fn host(p: &HierParams) -> Host {
    Host {
        h: p.h,
        e_u: to_host(p.e_u.as_tensor()).unwrap(),
        mu: to_host(p.mu.as_tensor()).unwrap(),
        b_m: to_host(p.b_m.as_tensor()).unwrap(),
        r: to_host(p.r.as_tensor()).unwrap(),
        b_g: to_host(p.b_g.as_tensor()).unwrap(),
        offsets: p
            .offsets
            .iter()
            .map(|o| {
                (
                    to_host(o.d_mu.as_tensor()).unwrap(),
                    to_host(o.d_b_m.as_tensor()).unwrap(),
                    o.delta_host().unwrap(),
                    to_host(o.d_b_g.as_tensor()).unwrap(),
                )
            })
            .collect(),
        resid: p
            .offsets
            .iter()
            .map(|o| to_host(&o.d_r.residual().unwrap()).unwrap())
            .collect(),
    }
}

fn log_softmax_f64(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + scores.iter().map(|s| (s - m).exp()).sum::<f64>().ln();
    scores.iter().map(|s| s - lse).collect()
}

/// `L = Σ_u Σ_t w [L₁ + Σ_k (c_k/K) L₂] + ridge`, written from the formula
/// with no shared code.
fn reference_loss(
    p: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    sup: &TrackSupport,
    plan: &StepPlan,
    offset_l2: f32,
) -> f64 {
    let hp = host(p);
    let h = hp.h;
    let n_m = part.n_modules();
    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    };
    let mut loss = 0f64;
    for t in 0..units.n_tracks() {
        let off = t.checked_sub(1).map(|i| &hp.offsets[i]);
        let mu_row = |m: usize| -> Vec<f32> {
            (0..h)
                .map(|k| hp.mu[m * h + k] + off.map_or(0.0, |o| o.0[m * h + k]))
                .collect()
        };
        let r_row = |g: usize| -> Vec<f32> {
            (0..h)
                .map(|k| hp.r[g * h + k] + off.map_or(0.0, |o| o.2[g * h + k]))
                .collect()
        };
        let mods: Vec<usize> = if sup.is_full(t) {
            (0..n_m).collect()
        } else {
            sup.modules_of(t).iter().map(|&m| m as usize).collect()
        };
        for &u in &plan.units {
            let u = u as usize;
            let w = f64::from(units.weight_of(u, t));
            let e = &hp.e_u[u * h..(u + 1) * h];
            let scores: Vec<f64> = mods
                .iter()
                .map(|&m| dot(e, &mu_row(m)) + f64::from(hp.b_m[m] + off.map_or(0.0, |o| o.1[m])))
                .collect();
            let logp = log_softmax_f64(&scores);
            for (j, &m) in mods.iter().enumerate() {
                let q = f64::from(um.q[um.idx(u, t, m)]);
                loss -= w * q * logp[j];
            }
        }
        for ((tt, m), pairs) in &plan.pairs_by_module {
            if *tt as usize != t {
                continue;
            }
            let m = *m as usize;
            let members = &part.members[m];
            let genes: Vec<usize> = if sup.is_full(t) {
                members.iter().map(|&g| g as usize).collect()
            } else {
                sup.slots_of(t, m)
                    .iter()
                    .map(|&j| members[j as usize] as usize)
                    .collect()
            };
            for &(u, wt) in pairs {
                let u = u as usize;
                let scale = f64::from(units.weight_of(u, t)) * f64::from(wt);
                let e = &hp.e_u[u * h..(u + 1) * h];
                let scores: Vec<f64> = genes
                    .iter()
                    .map(|&g| {
                        dot(e, &r_row(g)) + f64::from(hp.b_g[g] + off.map_or(0.0, |o| o.3[g]))
                    })
                    .collect();
                let logp = log_softmax_f64(&scores);
                let n_um = f64::from(um.n_um[um.idx(u, t, m)]);
                let counts = um.by_module[u]
                    .iter()
                    .find(|((tr, k), _)| *tr as usize == t && *k as usize == m)
                    .map(|(_, v)| v.as_slice())
                    .unwrap_or(&[]);
                for &(slot, c) in counts {
                    let g = members[slot as usize] as usize;
                    let j = genes
                        .iter()
                        .position(|&x| x == g)
                        .expect("a count sits on a scored gene");
                    loss -= scale * f64::from(c) / n_um * logp[j];
                }
            }
        }
    }
    let n_g = hp.b_g.len() as f64;
    for ((d_mu, _, _, _), resid) in hp.offsets.iter().zip(&hp.resid) {
        let sq = |v: &[f32]| v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
        loss += f64::from(offset_l2) * (sq(d_mu) / n_m as f64 + sq(resid) / n_g);
    }
    loss
}

//////////////
// Fixtures //
//////////////

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
    let params = HierParams::new(3, 2, 6, 2, 11, &Device::Cpu).unwrap();
    let sup = TrackSupport::new(&units.tracks, &part);
    (units, part, um, sup, params)
}

fn plan_all() -> StepPlan {
    StepPlan {
        units: vec![0, 1, 2],
        pairs_by_module: vec![
            ((0, 0), vec![(0, 1.0), (2, 1.0)]),
            ((0, 1), vec![(1, 1.0), (2, 1.0)]),
        ],
    }
}

/// Four units, three modules, seven genes on two tracks; track 1 has rows for
/// genes {0, 3, 5, 6} only, so module 1 = {1, 4} is outside its support. The
/// offsets start off zero so the ridge and its gradient are visible; the gene
/// offset is rank 1 at H = 2.
fn fixture_tracks() -> (UnitTable, Partition, UnitModules, TrackSupport, HierParams) {
    let l0 = vec![
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
    let params = HierParams::new_tracked(4, 3, 7, 2, 2, 1, 11, &Device::Cpu).unwrap();
    let off = &params.offsets[0];
    let set = |v: &Var, f: &dyn Fn(usize) -> f32| {
        let n: usize = v.dims().iter().product();
        let data: Vec<f32> = (0..n).map(f).collect();
        v.set(&Tensor::from_vec(data, v.dims(), &Device::Cpu).unwrap())
            .unwrap();
    };
    set(&off.d_mu, &|i| {
        0.07 * (i as f32 + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 }
    });
    set(&off.d_b_m, &|i| 0.05 - 0.03 * i as f32);
    set(&off.d_r.u, &|i| 0.04 * ((i % 5) as f32 - 2.0));
    set(&off.d_r.v, &|i| if i == 0 { 0.3 } else { -0.2 });
    set(&off.d_b_g, &|i| 0.02 * ((i % 3) as f32 - 1.0));
    (units, part, um, sup, params)
}

/// Genes track 1 has no row for.
const OFF_TRACK_GENES: [usize; 3] = [1, 2, 4];

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

fn total(
    p: &HierParams,
    units: &UnitTable,
    um: &UnitModules,
    part: &Partition,
    sup: &TrackSupport,
    plan: &StepPlan,
    l2: f32,
) -> (f64, Tensor) {
    let ctx = StepCtx {
        units,
        um,
        part,
        sup,
        axis: 0,
    };
    let (s, loss) = step_loss(p, &ctx, plan, l2, 0.0).unwrap();
    (s.loss_module + s.loss_gene + s.loss_ridge, loss)
}

/// Central difference of the loss in one entry of `var`.
fn finite_difference(var: &Var, flat: usize, eps: f32, loss_at: &dyn Fn() -> f64) -> f64 {
    let dims = var.dims().to_vec();
    let base = var
        .as_tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let bump = |d: f32| -> f64 {
        let mut v = base.clone();
        v[flat] += d;
        var.set(&Tensor::from_vec(v, dims.as_slice(), &Device::Cpu).unwrap())
            .unwrap();
        loss_at()
    };
    let plus = bump(eps);
    let minus = bump(-eps);
    bump(0.0);
    (plus - minus) / (2.0 * f64::from(eps))
}

fn grad_of(grads: &GradStore, v: &Var) -> Vec<f32> {
    match grads.get(v) {
        Some(g) => g.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        None => vec![0.0; v.dims().iter().product()],
    }
}

///////////
// Tests //
///////////

#[test]
fn the_step_loss_matches_the_f64_reference_on_one_track() {
    let (units, part, um, sup, p) = fixture();
    let plan = plan_all();
    let (got, _) = total(&p, &units, &um, &part, &sup, &plan, 0.0);
    let want = reference_loss(&p, &units, &um, &part, &sup, &plan, 0.0);
    assert!(
        (got - want).abs() < 1e-4 * (1.0 + want.abs()),
        "{got} vs {want}"
    );
}

#[test]
fn the_step_loss_matches_the_f64_reference_with_tracks_and_ridge() {
    let (units, part, um, sup, p) = fixture_tracks();
    let plan = plan_all_tracks();
    let (got, _) = total(&p, &units, &um, &part, &sup, &plan, OFFSET_L2);
    let want = reference_loss(&p, &units, &um, &part, &sup, &plan, OFFSET_L2);
    assert!(
        (got - want).abs() < 1e-4 * (1.0 + want.abs()),
        "{got} vs {want}"
    );
    let (s, _) = step_loss(
        &p,
        &StepCtx {
            units: &units,
            um: &um,
            part: &part,
            sup: &sup,
            axis: 0,
        },
        &plan,
        OFFSET_L2,
        0.0,
    )
    .unwrap();
    assert!(s.loss_ridge > 0.0, "the moved offsets carry a ridge");
}

/// Autograd against central differences of the same loss, on every table.
#[test]
fn autograd_matches_finite_differences_with_tracks() {
    let (units, part, um, sup, p) = fixture_tracks();
    let plan = plan_all_tracks();
    let (_, loss) = total(&p, &units, &um, &part, &sup, &plan, OFFSET_L2);
    let grads = loss.backward().unwrap();
    let loss_at = || total(&p, &units, &um, &part, &sup, &plan, OFFSET_L2).0;
    let o = &p.offsets[0];
    let checks: Vec<(&str, &Var, usize)> = vec![
        ("e_u", &p.e_u, 3),
        ("mu", &p.mu, 4),
        ("b_m", &p.b_m, 1),
        ("r", &p.r, 7),
        ("b_g", &p.b_g, 5),
        ("d_mu", &o.d_mu, 2),
        ("d_b_m", &o.d_b_m, 0),
        ("u", &o.d_r.u, 6),
        ("V", &o.d_r.v, 1),
        ("d_b_g", &o.d_b_g, 3),
    ];
    for (name, var, flat) in checks {
        let analytic = f64::from(grad_of(&grads, var)[flat]);
        let numeric = finite_difference(var, flat, 1e-3, &loss_at);
        assert!(
            (analytic - numeric).abs() < 2e-3 * (1.0 + analytic.abs()),
            "{name}[{flat}]: autograd {analytic} vs finite difference {numeric}"
        );
    }
}

#[test]
fn a_gene_without_a_row_on_a_track_gets_no_gradient_from_that_track() {
    let (units, part, um, sup, p) = fixture_tracks();
    let plan = plan_all_tracks();
    let (_, loss) = total(&p, &units, &um, &part, &sup, &plan, 0.0);
    let grads = loss.backward().unwrap();
    let h = p.h;
    let d_u = grad_of(&grads, &p.offsets[0].d_r.u);
    let d_b_g = grad_of(&grads, &p.offsets[0].d_b_g);
    for &g in &OFF_TRACK_GENES {
        assert_eq!(d_u[g], 0.0, "δ row factor {g} moved");
        assert_eq!(d_b_g[g], 0.0);
    }
    // Module 1 = {1, 4} is outside track 1's support: its offset takes nothing.
    let d_mu = grad_of(&grads, &p.offsets[0].d_mu);
    assert!(d_mu[h..2 * h].iter().all(|&x| x == 0.0));
    // The base rows of those genes still learn from track 0.
    let r = grad_of(&grads, &p.r);
    assert!(OFF_TRACK_GENES
        .iter()
        .any(|&g| r[g * h..(g + 1) * h].iter().any(|&x| x != 0.0)));
}

#[test]
fn pair_weight_scales_the_gene_level_term() {
    let (units, part, um, sup, p) = fixture();
    let one = plan_all();
    let mut half = plan_all();
    for (_, pairs) in &mut half.pairs_by_module {
        for pr in pairs.iter_mut() {
            pr.1 = 0.5;
        }
    }
    let (a, _) = step_loss(
        &p,
        &StepCtx {
            units: &units,
            um: &um,
            part: &part,
            sup: &sup,
            axis: 0,
        },
        &one,
        0.0,
        0.0,
    )
    .unwrap();
    let (b, _) = step_loss(
        &p,
        &StepCtx {
            units: &units,
            um: &um,
            part: &part,
            sup: &sup,
            axis: 0,
        },
        &half,
        0.0,
        0.0,
    )
    .unwrap();
    assert!((b.loss_gene - 0.5 * a.loss_gene).abs() < 1e-5);
    assert!((b.loss_module - a.loss_module).abs() < 1e-6);
}

#[test]
fn twenty_steps_on_the_full_plan_lower_the_loss() {
    let (units, part, um, sup, mut p) = fixture();
    let plan = plan_all();
    let mut opt = Optimizers::new(&p, 0.2).unwrap();
    let before = total(&p, &units, &um, &part, &sup, &plan, 0.0).0;
    for _ in 0..20 {
        let (_, loss) = total(&p, &units, &um, &part, &sup, &plan, 0.0);
        let grads = loss.backward().unwrap();
        apply(&mut p, &mut opt, &grads, 0.2, 0.0).unwrap();
    }
    let after = total(&p, &units, &um, &part, &sup, &plan, 0.0).0;
    assert!(after < before * 0.9, "{before} → {after}");
}

/// Decay reaches the rows a step touched and no bias; a row the step never
/// scored keeps its value; the offset tables never decay.
#[test]
fn weight_decay_shrinks_touched_rows_only_and_never_the_offsets() {
    let (units, part, um, sup, mut p) = fixture_tracks();
    // A plan that scores modules 0 and 2 only, leaving module 1's genes {1, 4} untouched.
    let plan = StepPlan {
        units: vec![0, 1, 2],
        pairs_by_module: vec![
            ((0, 0), vec![(0, 1.0), (1, 1.0), (2, 1.0)]),
            ((0, 2), vec![(0, 1.0), (1, 1.0), (2, 1.0)]),
        ],
    };
    // A negligible optimizer rate, so the decay is all that moves a row; the
    // decay factor itself comes from the rate handed to `apply`.
    let mut opt = Optimizers::new(&p, 1e-7).unwrap();
    let r0 = to_host(p.r.as_tensor()).unwrap();
    let b0 = to_host(p.b_g.as_tensor()).unwrap();
    let e0 = to_host(p.e_u.as_tensor()).unwrap();
    let u0 = to_host(p.offsets[0].d_r.u.as_tensor()).unwrap();
    let v0 = to_host(p.offsets[0].d_r.v.as_tensor()).unwrap();
    let (_, loss) = total(&p, &units, &um, &part, &sup, &plan, OFFSET_L2);
    let grads = loss.backward().unwrap();
    apply(&mut p, &mut opt, &grads, 0.5, 0.2).unwrap();
    let h = p.h;
    let r1 = to_host(p.r.as_tensor()).unwrap();
    for g in [0usize, 2, 3, 5, 6] {
        for k in 0..h {
            assert!(
                (r1[g * h + k] - 0.9 * r0[g * h + k]).abs() < 1e-5,
                "touched row {g} decays"
            );
        }
    }
    for g in [1usize, 4] {
        for k in 0..h {
            assert_eq!(
                r1[g * h + k],
                r0[g * h + k],
                "untouched row {g} keeps its value"
            );
        }
    }
    for (a, b) in to_host(p.b_g.as_tensor()).unwrap().iter().zip(&b0) {
        assert!((a - b).abs() < 1e-5, "biases never decay");
    }
    let e1 = to_host(p.e_u.as_tensor()).unwrap();
    for k in 0..h {
        assert!((e1[k] - 0.9 * e0[k]).abs() < 1e-5);
        assert_eq!(e1[3 * h + k], e0[3 * h + k], "unit 3 was not in the plan");
    }
    let o = &p.offsets[0];
    for (a, b) in to_host(o.d_r.u.as_tensor()).unwrap().iter().zip(&u0) {
        assert!((a - b).abs() < 1e-5, "offsets never decay");
    }
    for (a, b) in to_host(o.d_r.v.as_tensor()).unwrap().iter().zip(&v0) {
        assert!((a - b).abs() < 1e-5, "offsets never decay");
    }
}

/// A pinned row takes no step, its bias does, and the dictionary stays put.
#[test]
fn pinned_rows_hold_while_their_biases_train() {
    let (units, part, um, sup, mut p) = fixture();
    let given = PresetGenes {
        ids: vec![0, 3],
        rows: vec![0.5, -0.5, 0.25, 0.75],
        mode: PresetMode::Freeze,
    };
    p.preset(&given, &part.module_of).unwrap();
    let plan = plan_all();
    let mut opt = Optimizers::new(&p, 0.2).unwrap();
    let r0 = to_host(p.r.as_tensor()).unwrap();
    let mu0 = to_host(p.mu.as_tensor()).unwrap();
    let b0 = to_host(p.b_g.as_tensor()).unwrap();
    for _ in 0..5 {
        let (_, loss) = total(&p, &units, &um, &part, &sup, &plan, 0.0);
        let grads = loss.backward().unwrap();
        apply(&mut p, &mut opt, &grads, 0.2, 0.01).unwrap();
    }
    let h = p.h;
    let r1 = to_host(p.r.as_tensor()).unwrap();
    for g in [0usize, 3] {
        assert_eq!(
            &r1[g * h..(g + 1) * h],
            &r0[g * h..(g + 1) * h],
            "pinned row {g} moved"
        );
    }
    assert_ne!(&r1[2 * h..3 * h], &r0[2 * h..3 * h], "a free row trains");
    assert_eq!(
        to_host(p.mu.as_tensor()).unwrap(),
        mu0,
        "μ is pinned with the rows"
    );
    let b1 = to_host(p.b_g.as_tensor()).unwrap();
    assert!(
        b1[0] != b0[0] || b1[3] != b0[3],
        "a pinned gene's bias still trains"
    );
}

/// Autograd against central differences on the four LoRA factors, with the
/// shared factors moved off zero so the row factors see a gradient too.
#[test]
fn autograd_matches_finite_differences_on_the_lora_factors() {
    let (units, part, um, sup, mut p) = fixture();
    let given = PresetGenes {
        ids: vec![0, 3, 4],
        rows: vec![0.5, -0.5, 0.25, 0.75, -0.3, 0.1],
        mode: PresetMode::Lora(LoraSpec {
            rank: 1,
            lr_ratio: 1.0,
            ridge: 0.0,
        }),
    };
    p.preset(&given, &part.module_of).unwrap();
    let l = p.lora.as_ref().unwrap();
    for v in [&l.module.v, &l.gene.v] {
        v.set(&Tensor::from_vec(vec![0.2f32, -0.4], (1, 2), &Device::Cpu).unwrap())
            .unwrap();
    }
    let plan = plan_all();
    let (_, loss) = total(&p, &units, &um, &part, &sup, &plan, 0.0);
    let grads = loss.backward().unwrap();
    let loss_at = || total(&p, &units, &um, &part, &sup, &plan, 0.0).0;
    let l = p.lora.as_ref().unwrap();
    for (name, var, flat) in [
        ("a", &l.module.u, 1usize),
        ("v_m", &l.module.v, 0),
        ("u", &l.gene.u, 3),
        ("v_g", &l.gene.v, 1),
    ] {
        let analytic = f64::from(grad_of(&grads, var)[flat]);
        let numeric = finite_difference(var, flat, 1e-3, &loss_at);
        assert!(
            (analytic - numeric).abs() < 2e-3 * (1.0 + analytic.abs()),
            "{name}[{flat}]: autograd {analytic} vs finite difference {numeric}"
        );
        assert!(analytic != 0.0, "{name} receives a gradient");
    }
    // A free gene's `u` row is masked at the step, not in the gradient itself.
    assert_eq!(
        to_host(&l.gene.u_mask).unwrap(),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    );
}

/// The gene offset is `u · V`: after steps its dense form has rank ≤ rank. A
/// given base under freeze holds verbatim while the residual on the other
/// genes moves, and the pinned base rows hold on both tracks.
#[test]
fn the_gene_offset_stays_low_rank_and_a_pinned_offset_base_holds() {
    let (units, part, um, sup, mut p) = fixture_tracks();
    let plan = plan_all_tracks();
    let mut opt = Optimizers::new(&p, 0.2).unwrap();
    for _ in 0..5 {
        let (_, loss) = total(&p, &units, &um, &part, &sup, &plan, OFFSET_L2);
        let grads = loss.backward().unwrap();
        apply(&mut p, &mut opt, &grads, 0.2, 0.0).unwrap();
    }
    let delta = nalgebra::DMatrix::<f32>::from_row_slice(7, 2, &p.offsets[0].delta_host().unwrap());
    let sv = delta.singular_values();
    assert!(sv[0] > 1e-4, "the offset moved: {sv}");
    assert!(sv[1] <= 1e-5 * sv[0], "rank 1: {sv}");

    let (units, part, um, sup, mut q) = fixture_tracks();
    q.preset(
        &PresetGenes {
            ids: vec![0, 3],
            rows: vec![0.5, -0.5, 0.25, 0.75],
            mode: PresetMode::Freeze,
        },
        &part.module_of,
    )
    .unwrap();
    q.preset_offsets(
        &[PresetOffsets {
            track: 1,
            ids: vec![0],
            rows: vec![0.3, -0.2],
        }],
        PresetMode::Freeze,
    )
    .unwrap();
    // The residual was rebuilt around the pin; move its shared factor again
    // so the free genes' part is visible from the first step.
    q.offsets[0]
        .d_r
        .v
        .set(&Tensor::from_vec(vec![0.3f32, -0.2], (1, 2), &Device::Cpu).unwrap())
        .unwrap();
    let r0 = to_host(q.r.as_tensor()).unwrap();
    let mut opt = Optimizers::new(&q, 0.2).unwrap();
    for _ in 0..5 {
        let (_, loss) = total(&q, &units, &um, &part, &sup, &plan, OFFSET_L2);
        let grads = loss.backward().unwrap();
        apply(&mut q, &mut opt, &grads, 0.2, 0.0).unwrap();
    }
    let h = q.h;
    let delta = q.offsets[0].delta_host().unwrap();
    assert_eq!(
        &delta[0..h],
        &[0.3, -0.2],
        "the given offset base holds verbatim"
    );
    assert!(
        delta[3 * h..4 * h].iter().any(|&x| x != 0.0),
        "a free gene's offset trained"
    );
    let r1 = to_host(q.r.as_tensor()).unwrap();
    for g in [0usize, 3] {
        assert_eq!(&r1[g * h..(g + 1) * h], &r0[g * h..(g + 1) * h]);
    }
}
