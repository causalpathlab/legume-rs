use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

const H: usize = 8;

struct World {
    feature_emb: Vec<f32>,
    cell_flat: Vec<f32>,
    type_markers: Vec<Vec<(u32, f32)>>,
    truth: Vec<usize>,
}

/// Type `t`'s centre: a distinct axis at radius `sep`.
fn centre_at(t: usize, sep: f32) -> Vec<f32> {
    let mut v = vec![0f32; H];
    v[t % H] = sep;
    v
}

/// The default, comfortably-separated geometry.
fn centre(t: usize) -> Vec<f32> {
    centre_at(t, 10.0)
}

fn jitter(base: &[f32], sd: f32, rng: &mut StdRng) -> Vec<f32> {
    let d = Normal::new(0f32, sd).unwrap();
    base.iter().map(|&x| x + d.sample(rng)).collect()
}

/// `marks[t]` markers and `cells[t]` cells per type; `marker_sd` = how tightly a type's
/// markers cluster about its centre, i.e. how well the panel actually pins the type down.
fn build(marks: &[usize], cells: &[usize], marker_sd: f32, seed: u64) -> World {
    let c = marks.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut feature_emb = Vec::new();
    let mut type_markers: Vec<Vec<(u32, f32)>> = vec![Vec::new(); c];
    for (t, &m) in marks.iter().enumerate() {
        for _ in 0..m {
            let gi = (feature_emb.len() / H) as u32;
            feature_emb.extend(jitter(&centre(t), marker_sd, &mut rng));
            type_markers[t].push((gi, 1.0));
        }
    }
    let mut cell_flat = Vec::new();
    let mut truth = Vec::new();
    for (t, &k) in cells.iter().enumerate() {
        for _ in 0..k {
            cell_flat.extend(jitter(&centre(t), 1.0, &mut rng));
            truth.push(t);
        }
    }
    World {
        feature_emb,
        cell_flat,
        type_markers,
        truth,
    }
}

/// Drive the panel bootstrap alone (no clustering step) — the unit under test here is the
/// resampling and what it says about the marker panel, not the surrounding pipeline.
fn run(w: &World, min_support: f32, seed: u64) -> BootstrapResult {
    run_marker_bootstrap(
        &w.feature_emb,
        &w.cell_flat,
        &w.type_markers,
        H,
        &MarkerBootstrapConfig {
            n_boot: 200,
            abstain: Abstain::Support(min_support),
            set_coverage: 0.8,
            max_set_size: 3,
            recluster: false,
        },
        seed,
        None,
    )
    .expect("panel-only bootstrap cannot fail")
}

/// Clean, well-separated, richly-marked types are recovered, and the bootstrap says so: the
/// panel can be resampled freely without the calls moving. If this fails nothing else matters.
#[test]
fn recovers_clean_types() {
    let w = build(&[20, 20, 20], &[100, 100, 100], 0.3, 1);
    let p = run(&w, 0.5, 7);
    let correct = p
        .assign
        .iter()
        .zip(&w.truth)
        .filter(|&(&a, &t)| a == t)
        .count();
    assert!(
        correct >= 295,
        "expected ~all 300 cells recovered, got {correct}"
    );
    let mean = p.support.iter().sum::<f32>() / p.support.len() as f32;
    assert!(
        mean > 0.95,
        "support should be near 1 on clean data, got {mean}"
    );
}

/// **The winner's curse, which is the point of the whole thing.** A marker-poor type has a
/// centroid that flies around under resampling, so it wins its cells on only some draws and
/// its support collapses — even though `argmin` would hand it a confident call every time.
///
/// The types are placed close together on purpose: the curse only bites when the centroid's
/// jitter is comparable to the margin the assignment is decided by, which is precisely the
/// regime real marker panels sit in. Far-apart types survive a wobbly centroid and *should*
/// come out confident.
#[test]
fn marker_poor_type_has_lower_support() {
    let mut rng = StdRng::seed_from_u64(2);
    let (sep, marker_sd) = (3.0f32, 1.5f32);
    let mut feature_emb = Vec::new();
    let mut type_markers: Vec<Vec<(u32, f32)>> = vec![Vec::new(); 2];
    for (t, &m) in [20usize, 3].iter().enumerate() {
        for _ in 0..m {
            let gi = (feature_emb.len() / H) as u32;
            feature_emb.extend(jitter(&centre_at(t, sep), marker_sd, &mut rng));
            type_markers[t].push((gi, 1.0));
        }
    }
    let (mut cell_flat, mut truth) = (Vec::new(), Vec::new());
    for t in 0..2 {
        for _ in 0..100 {
            cell_flat.extend(jitter(&centre_at(t, sep), 0.5, &mut rng));
            truth.push(t);
        }
    }
    let w = World {
        feature_emb,
        cell_flat,
        type_markers,
        truth,
    };
    let p = run(&w, 0.0, 7);

    let mean_support = |t: usize| -> f32 {
        let idx: Vec<usize> = (0..w.truth.len()).filter(|&i| w.truth[i] == t).collect();
        idx.iter().map(|&i| p.post[i * p.c + t]).sum::<f32>() / idx.len() as f32
    };
    let (rich, poor) = (mean_support(0), mean_support(1));
    assert!(
        poor < rich,
        "the 3-marker type should be the less reproducible one: poor={poor}, rich={rich}"
    );
    assert!(
        p.type_qc[1].centroid_jitter > p.type_qc[0].centroid_jitter,
        "the 3-marker type's centroid should move more under resampling: {} vs {}",
        p.type_qc[1].centroid_jitter,
        p.type_qc[0].centroid_jitter
    );
}

/// A panel the embedding scatters is a panel that cannot place its type: resampling it moves
/// the centroid a long way, so the calls it produces are not reproducible.
#[test]
fn scattered_panel_is_unstable() {
    let tight = run(&build(&[20, 20], &[100, 100], 0.3, 3), 0.0, 7);
    let loose = run(&build(&[20, 20], &[100, 100], 8.0, 3), 0.0, 7);
    assert!(
        loose.type_qc[0].centroid_jitter > tight.type_qc[0].centroid_jitter,
        "a scattered panel should jitter more: {} vs {}",
        loose.type_qc[0].centroid_jitter,
        tight.type_qc[0].centroid_jitter
    );
    let mean = |p: &BootstrapResult| p.support.iter().sum::<f32>() / p.support.len() as f32;
    assert!(
        mean(&loose) < mean(&tight),
        "a scattered panel should give less reproducible calls: {} vs {}",
        mean(&loose),
        mean(&tight)
    );
}

/// Abstention is instability: with a panel that cannot tell its types apart, `min_support`
/// leaves the cells uncalled rather than handing out a confident wrong label.
#[test]
fn unstable_calls_abstain() {
    // Two types whose markers overlap in the same region — the panel cannot separate them.
    let mut w = build(&[10, 10], &[100, 100], 0.3, 4);
    for (gi, _) in w.type_markers[1].clone() {
        // Put type 1's markers on top of type 0's centre: the two panels now disagree with
        // the cells, and no resample can stably prefer one.
        let base = centre(0);
        let mut rng = StdRng::seed_from_u64(u64::from(gi));
        let v = jitter(&base, 0.3, &mut rng);
        w.feature_emb[gi as usize * H..(gi as usize + 1) * H].copy_from_slice(&v);
    }
    let p = run(&w, 0.9, 7);
    let unassigned = p.assign.iter().filter(|&&a| a == UNASSIGNED).count();
    assert!(
        unassigned > 50,
        "an unresolvable panel should abstain, but only {unassigned}/200 cells were left uncalled"
    );
}

/// A single marker cannot be bootstrapped — resampling it always returns itself, so it would
/// look *perfectly* stable and win cells with full confidence. Such a type must be excluded
/// from the assignment, not trusted.
#[test]
fn single_marker_type_cannot_compete() {
    let w = build(&[20, 1], &[100, 100], 0.3, 8);
    let p = run(&w, 0.0, 7);
    assert_eq!(p.type_qc[1].n_live, 1);
    assert_eq!(
        p.type_qc[1].occupancy, 0.0,
        "a 1-marker type has unmeasurable centroid variance and must not claim cells"
    );
    assert!(
        p.assign.iter().all(|&a| a != 1),
        "no cell may be assigned to an unbootstrappable type"
    );
}

/// A dead marker (all-zero β row) carries no evidence and must not be resampled into the
/// panel, nor reported as corroborated.
#[test]
fn dead_markers_are_excluded() {
    let mut w = build(&[20, 20], &[100, 100], 0.3, 5);
    // Kill 5 of type 0's markers.
    for &(gi, _) in w.type_markers[0][..5].iter() {
        w.feature_emb[gi as usize * H..(gi as usize + 1) * H].fill(0.0);
    }
    let p = run(&w, 0.5, 7);
    assert_eq!(
        p.type_qc[0].n_live, 15,
        "5 of the 20 markers should be dead"
    );
    assert_eq!(
        p.marker_live[0].iter().filter(|&&l| l).count(),
        15,
        "the dead markers should be reported as such"
    );
    assert!(
        p.marker_dev[0][..5].iter().all(|d| d.is_nan()),
        "a dead marker has no deviation to report"
    );
}

/// The resamples are keyed by `(seed, draw, type)`, not by rayon's scheduling, so a seed must
/// reproduce exactly — and on a panel loose enough for the draws to actually disagree, a
/// different seed must not. (On a clean, well-separated panel every draw gives the same
/// answer, so *any* seed reproduces; that is correct, and is what `recovers_clean_types`
/// asserts. The point of a seed is to pin the chain down where it has freedom to differ.)
#[test]
fn seed_reproduces() {
    let w = build(&[10, 10], &[60, 60], 6.0, 6);
    assert_eq!(run(&w, 0.5, 11).post, run(&w, 0.5, 11).post);
    assert_ne!(run(&w, 0.5, 11).post, run(&w, 0.5, 12).post);
}
