use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

const H: usize = 8;

/// Type `t`'s centre: a distinct axis at radius `sep`.
fn centre(t: usize, sep: f32) -> Vec<f32> {
    let mut v = vec![0f32; H];
    v[t % H] = sep;
    v
}

fn jitter(base: &[f32], sd: f32, rng: &mut StdRng) -> Vec<f32> {
    let d = Normal::new(0f32, sd).unwrap();
    base.iter().map(|&x| x + d.sample(rng)).collect()
}

struct World {
    feature_emb: Vec<f32>,
    cell_flat: Vec<f32>,
    type_markers: Vec<Vec<(u32, f32)>>,
}

/// `marks[t]` markers per type, `cells[t]` cells per type. `marker_sd` = how tightly a type's
/// markers sit on its centre — i.e. how well the panel actually pins the type down.
fn build(marks: &[usize], cells: &[usize], sep: f32, marker_sd: f32, seed: u64) -> World {
    let c = marks.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut feature_emb = Vec::new();
    let mut type_markers: Vec<Vec<(u32, f32)>> = vec![Vec::new(); c];
    for (t, &m) in marks.iter().enumerate() {
        for _ in 0..m {
            let gi = (feature_emb.len() / H) as u32;
            feature_emb.extend(jitter(&centre(t, sep), marker_sd, &mut rng));
            type_markers[t].push((gi, 1.0));
        }
    }
    let mut cell_flat = Vec::new();
    for (t, &k) in cells.iter().enumerate() {
        for _ in 0..k {
            cell_flat.extend(jitter(&centre(t, sep), 1.0, &mut rng));
        }
    }
    World {
        feature_emb,
        cell_flat,
        type_markers,
    }
}

fn run(w: &World, n_perm: usize) -> PanelNull {
    run_panel_null(&w.feature_emb, &w.cell_flat, &w.type_markers, H, n_perm, 11)
}

/// A panel whose genes really do sit on their type beats any random draw of the same size — the
/// null is clearable, so it is not vacuous.
#[test]
fn a_real_panel_beats_random_genes_of_the_same_size() {
    let w = build(&[40, 40, 40], &[100, 100, 100], 6.0, 0.3, 1);
    let r = run(&w, 500);
    for t in 0..3 {
        assert!(
            r.p[t] < 0.05,
            "type {t}: real occupancy {:.3} should beat the null's {:.3} (p={:.3})",
            r.occupancy[t],
            r.null_occupancy[t],
            r.p[t]
        );
    }
}

/// **The bias catch, and the reason this exists.** A type whose listed markers are scattered all
/// over the embedding — the panel is *wrong*, not merely small — still gets a centroid, and
/// `argmin` still hands it cells. The bootstrap cannot see this: every resample of a wrong panel
/// is wrong the same way, so the call comes back stable. The null does see it: random genes place
/// the type just as well as its own genes do, so it cannot clear the bar.
#[test]
fn a_panel_that_does_not_identify_its_type_fails_the_null() {
    // Types 0 and 1 are real. Type 2's markers are strewn across the whole marker cloud — they
    // pick out nothing.
    let mut w = build(&[40, 40, 40], &[100, 100, 100], 6.0, 0.3, 2);
    let mut rng = StdRng::seed_from_u64(99);
    for &(gi, _) in &w.type_markers[2].clone() {
        let anywhere = jitter(&[0f32; H], 6.0, &mut rng);
        w.feature_emb[gi as usize * H..(gi as usize + 1) * H].copy_from_slice(&anywhere);
    }
    let r = run(&w, 500);
    assert!(
        r.p[0] < 0.05 && r.p[1] < 0.05,
        "the two honest panels must still clear the null: p = {:.3}, {:.3}",
        r.p[0],
        r.p[1]
    );
    assert!(
        r.p[2] > 0.05,
        "a panel that identifies nothing must NOT beat random genes: \
         occupancy {:.3} vs null {:.3}, p = {:.3}",
        r.occupancy[2],
        r.null_occupancy[2],
        r.p[2]
    );
}

/// **The winner's curse cancels.** A 6-marker type has a wobbly centroid and steals cells it has
/// no right to — but a *random* 3-marker panel is exactly as wobbly and steals just as many, so
/// the comparison divides the advantage out. The small honest panel is therefore not punished for
/// being small: it still clears its own null.
#[test]
fn a_small_but_honest_panel_is_not_punished_for_being_small() {
    let w = build(&[40, 6], &[100, 100], 6.0, 0.3, 3);
    let r = run(&w, 500);
    assert_eq!(r.n_live, vec![40, 6]);
    assert!(
        r.p[1] < 0.05,
        "6 markers that genuinely sit on their type must still beat 6 random ones \
         (occupancy {:.3} vs null {:.3}, p = {:.3})",
        r.occupancy[1],
        r.null_occupancy[1],
        r.p[1]
    );
}

/// Per cell: a cell that random genes explain just as well carries no marker-specific evidence,
/// and says so. Cells sitting squarely on a well-marked type do not.
#[test]
fn cell_p_flags_cells_that_any_panel_would_have_explained() {
    let clean = run(&build(&[40, 40], &[100, 100], 6.0, 0.3, 4), 500);
    let mush = run(&build(&[40, 40], &[100, 100], 0.5, 6.0, 4), 500);
    let mean = |r: &PanelNull| {
        let v: Vec<f32> = r.cell_p.iter().copied().filter(|x| !x.is_nan()).collect();
        v.iter().sum::<f32>() / v.len() as f32
    };
    assert!(
        mean(&clean) < mean(&mush),
        "cells held by a real panel should be harder for random genes to explain: {:.3} vs {:.3}",
        mean(&clean),
        mean(&mush)
    );
}

/// **The trap this module exists to avoid, pinned as a test.** On a cleanly separated panel a
/// random draw captures *as many cells as the real one* — because once type t's real centroid is
/// removed from the competition, its cells have no near rival and anything in the vicinity sweeps
/// them up by elimination. So **occupancy cannot be the statistic**, and this asserts that it
/// cannot: the counts agree, while the cost — which is the statistic — separates them cleanly.
#[test]
fn occupancy_is_blind_where_cost_is_not() {
    let w = build(&[40, 40, 40], &[100, 100, 100], 6.0, 0.3, 7);
    let r = run(&w, 500);
    for t in 0..3 {
        assert!(
            (r.occupancy[t] - r.null_occupancy[t]).abs() < 0.05,
            "type {t}: random genes capture just as many cells ({:.3} vs {:.3}) — \
             this is exactly why occupancy is not the test",
            r.null_occupancy[t],
            r.occupancy[t]
        );
        assert!(
            r.null_cost[t] > 1.5 * r.cost[t],
            "…while the cost separates them: real {:.0} vs null {:.0}",
            r.cost[t],
            r.null_cost[t]
        );
    }
}

/// The draws are keyed by `(seed, type, perm)`, not by rayon's scheduling, so a seed reproduces.
#[test]
fn seed_reproduces() {
    let w = build(&[30, 30], &[60, 60], 3.0, 2.0, 5);
    assert_eq!(run(&w, 50).p, run(&w, 50).p);
}

/// A type the embedding never trained (all-dead markers) cannot compete and cannot be tested.
#[test]
fn a_dead_type_is_not_tested() {
    let mut w = build(&[40, 40], &[100, 100], 6.0, 0.3, 6);
    for &(gi, _) in &w.type_markers[1].clone() {
        w.feature_emb[gi as usize * H..(gi as usize + 1) * H].fill(0.0);
    }
    let r = run(&w, 100);
    assert_eq!(r.n_live[1], 0);
    assert_eq!(r.occupancy[1], 0.0);
    assert_eq!(
        r.p[1], 1.0,
        "a type with no live markers has nothing to test"
    );
}
