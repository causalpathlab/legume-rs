use super::*;
use crate::fit::config::{ParentModulesOwned, TrackInfo, TrackSpec};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};

#[test]
fn parent_warm_start_carries_matched_rows_and_initializes_the_rest() {
    let parent_pi = DMatrix::<f32>::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let parent_mu = DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let parent_rho = &parent_pi * &parent_mu;
    // New axis: gene 0 = parent 2, gene 1 = parent 0, gene 2 = unseen with gene 1's
    // profile, gene 3 = unseen with parent-1-like profile (parent 1 is MISSING here,
    // so it is initialized from the matched neighbours only).
    let row_to_parent = vec![Some(2), Some(0), None, None];
    let profiles = DMatrix::<f32>::from_row_slice(
        4,
        4,
        &[
            5.0, 5.0, 5.0, 5.0, //
            9.0, 1.0, 9.0, 1.0, //
            9.0, 1.0, 9.0, 1.0, //
            1.0, 9.0, 1.0, 9.0,
        ],
    );
    let logits = parent_module_logits(
        &ParentModulesOwned {
            rho: parent_rho,
            pi: parent_pi.clone(),
            mu: parent_mu,
            row_to_parent,
            knobs: crate::transfer::AlignKnobs {
                k: 2,
                similarity_floor: 0.5,
            },
        },
        &profiles,
    );
    assert_eq!(logits.nrows(), 4);
    assert_eq!(logits.ncols(), 2);
    assert_eq!(logits.row(0), parent_pi.row(2));
    assert_eq!(logits.row(1), parent_pi.row(0));
    assert_eq!(
        logits.row(2),
        parent_pi.row(0),
        "same profile as gene 1 → parent 0's row"
    );
    // Anti-correlated with every matched gene → diffuse module average of the parent.
    let avg: Vec<f32> = (0..2)
        .map(|m| parent_pi.column(m).iter().sum::<f32>() / 3.0)
        .collect();
    for m in 0..2 {
        assert!((logits[(3, m)] - avg[m]).abs() < 1e-6);
    }
}

/// The partition groups GENES, so a multi-track feature axis has to be
/// reduced to the base track's rows first, re-keyed by gene id.
#[test]
fn base_track_profile_is_the_identity_on_a_one_track_axis() {
    let p = DMatrix::<f32>::from_fn(6, 4, |i, j| (i * 4 + j) as f32);
    let got = base_track_profile(&p, &TrackSpec::base(p.nrows()));
    assert_eq!(got, p);
}

#[test]
fn base_track_profile_keeps_only_the_base_rows_re_keyed_by_gene() {
    // 5 feature rows over 3 genes: rows 0..2 are the base track (genes 1, 0 —
    // deliberately NOT in gene order), rows 2..5 are a second track.
    let profile = DMatrix::<f32>::from_row_slice(
        5,
        2,
        &[
            10.0, 11.0, //
            20.0, 21.0, //
            30.0, 31.0, //
            40.0, 41.0, //
            50.0, 51.0,
        ],
    );
    let tracks = TrackSpec {
        track_of_row: vec![0, 0, 1, 1, 1],
        gene_of_row: vec![1, 0, 0, 1, 2],
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
    let got = base_track_profile(&profile, &tracks);
    assert_eq!(got.nrows(), 3);
    assert_eq!(got.ncols(), 2);
    assert_eq!(got.row(0), profile.row(1)); // gene 0 sits on base row 1
    assert_eq!(got.row(1), profile.row(0)); // gene 1 on base row 0
    assert!(
        got.row(2).iter().all(|&x| x == 0.0),
        "gene 2 has no base row"
    );
}

#[test]
fn a_modality_is_module_only_from_its_row_count() {
    // 3 rows of modality 0, 5 of modality 1.
    let modality = [0, 0, 0, 1, 1, 1, 1, 1];
    assert_eq!(
        module_only_modalities(&modality, 4),
        Some(vec![false, true])
    );
    assert_eq!(
        module_only_modalities(&modality, 6),
        None,
        "neither is big enough"
    );
    assert_eq!(module_only_modalities(&modality, 0), None, "0 turns it off");
}

//////////////////////////////////////////////
// The module partition and feature coarsening //
//////////////////////////////////////////////

/// `n_pb` pseudobulks of 6 cells. `n_programs` programs of `per` features,
/// each program raised 8× on its own block of pseudobulks, then `n_flat`
/// features at one flat rate and `n_empty` never counted — the spurious
/// empty features a real gene axis is mostly made of.
fn planted(
    n_programs: usize,
    per: usize,
    n_flat: usize,
    n_empty: usize,
    n_pb: usize,
    seed: u64,
) -> (DMatrix<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let sizes = vec![6.0f32; n_pb];
    let d = n_programs * per + n_flat + n_empty;
    let mut counts = DMatrix::<f32>::zeros(d, n_pb);
    let block = n_pb / n_programs.max(1);
    for p in 0..n_programs {
        for i in 0..per {
            let g = p * per + i;
            let base = 0.5 + (i % 3) as f64 * 0.5;
            for s in 0..n_pb {
                let on = s / block.max(1) == p;
                let rate = base * if on { 8.0 } else { 1.0 } * f64::from(sizes[s]);
                counts[(g, s)] = Poisson::new(rate).unwrap().sample(&mut rng) as f32;
            }
        }
    }
    for f in 0..n_flat {
        let g = n_programs * per + f;
        for s in 0..n_pb {
            let rate = 0.3 * f64::from(sizes[s]);
            counts[(g, s)] = Poisson::new(rate).unwrap().sample(&mut rng) as f32;
        }
    }
    (counts, sizes)
}

/// Labels of a group of rows, as a set.
fn groups_of(labels: &[u32], rows: std::ops::Range<usize>) -> std::collections::BTreeSet<u32> {
    rows.map(|i| labels[i]).collect()
}

#[test]
fn programs_are_recovered_and_empty_features_share_the_background() {
    let (counts, sizes) = planted(3, 10, 20, 20, 24, 7);
    let labels = partition_modules(&counts, &sizes, 4, 3).unwrap();
    for p in 0..3 {
        assert_eq!(
            groups_of(&labels, p * 10..(p + 1) * 10).len(),
            1,
            "program {p} split: {labels:?}"
        );
    }
    let programs: std::collections::BTreeSet<u32> = (0..3).map(|p| labels[p * 10]).collect();
    assert_eq!(programs.len(), 3, "programs merged: {labels:?}");
    let background = groups_of(&labels, 30..70);
    assert_eq!(
        background.len(),
        1,
        "flat and empty features split: {labels:?}"
    );
    assert!(
        programs.is_disjoint(&background),
        "a program joined the background: {labels:?}"
    );
}

#[test]
fn the_partition_is_reproducible() {
    let (counts, sizes) = planted(3, 10, 20, 20, 24, 7);
    let a = partition_modules(&counts, &sizes, 4, 3).unwrap();
    let b = partition_modules(&counts, &sizes, 4, 3).unwrap();
    assert_eq!(a, b);
}

#[test]
fn partition_by_group_never_mixes_groups() {
    let (a, sizes) = planted(2, 6, 4, 4, 16, 1);
    let (b, _) = planted(2, 6, 4, 4, 16, 2);
    let mut counts = DMatrix::<f32>::zeros(a.nrows() + b.nrows(), a.ncols());
    counts.rows_mut(0, a.nrows()).copy_from(&a);
    counts.rows_mut(a.nrows(), b.nrows()).copy_from(&b);
    let group: Vec<u32> = (0..counts.nrows())
        .map(|i| u32::from(i >= a.nrows()))
        .collect();
    let part =
        partition_modules_by_group(&counts, &sizes, &group, &[3, 3], &[0, 0], None, 5).unwrap();
    let (labels, n) = (part.labels, part.n_modules);
    assert_eq!(n, 6);
    assert!(labels[..a.nrows()].iter().all(|&m| m < 3), "{labels:?}");
    assert!(
        labels[a.nrows()..].iter().all(|&m| (3..6).contains(&m)),
        "{labels:?}"
    );
}

/// The background-to-module-only switch is opt-in: off, nothing is flagged.
#[test]
fn flat_module_only_is_off_unless_asked() {
    let (counts, sizes) = planted(2, 6, 4, 4, 16, 3);
    let spec = TrackSpec::base(counts.nrows());
    assert!(flat_module_only(&counts, &sizes, &spec, false).is_none());
}

/// On, a one-track axis flags exactly the rows the coarsener would put in its
/// background group, and never a planted program's row.
#[test]
fn flat_module_only_flags_the_background_rows() {
    let (counts, sizes) = planted(2, 6, 4, 4, 16, 3);
    let spec = TrackSpec::base(counts.nrows());
    let flags = flat_module_only(&counts, &sizes, &spec, true).expect("flags");
    assert_eq!(flags.len(), counts.nrows());
    assert!(
        flags[..12].iter().all(|&f| !f),
        "a program row flagged: {flags:?}"
    );
    assert!(
        flags[12..].iter().all(|&f| f),
        "a flat/empty row kept: {flags:?}"
    );
}

/// A multi-track axis never gets the switch: module-only rows need a one-track
/// axis, and the partition there is over genes, not rows.
#[test]
fn flat_module_only_skips_a_multi_track_axis() {
    let (counts, sizes) = planted(2, 6, 4, 4, 16, 3);
    let n_g = counts.nrows();
    let mut stacked = DMatrix::<f32>::zeros(2 * n_g, counts.ncols());
    stacked.rows_mut(0, n_g).copy_from(&counts);
    stacked.rows_mut(n_g, n_g).copy_from(&counts);
    let spec = TrackSpec {
        track_of_row: (0..2 * n_g).map(|r| u32::from(r >= n_g)).collect(),
        gene_of_row: (0..2 * n_g).map(|r| (r % n_g) as u32).collect(),
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
    assert!(flat_module_only(&stacked, &sizes, &spec, true).is_none());
}

/// `planted` plus `n_scattered` isolated rows: each counted heavily in ONE
/// pseudobulk and nowhere else, so it is not flat (informative) yet shares a
/// profile with nothing — a peak that arises at random.
fn planted_with_scattered(n_scattered: usize) -> (DMatrix<f32>, Vec<f32>, usize) {
    let (base, sizes) = planted(3, 10, 5, 10, 24, 11);
    let d0 = base.nrows();
    let mut counts = DMatrix::<f32>::zeros(d0 + n_scattered, base.ncols());
    counts.rows_mut(0, d0).copy_from(&base);
    for j in 0..n_scattered {
        counts[(d0 + j, (7 * j + 3) % base.ncols())] = 40.0;
    }
    (counts, sizes, d0)
}

/// The per-group partition reports which module ids are background, only for
/// groups asked to drop scattered rows; group 0 keeps plain coarsening.
#[test]
fn partition_by_group_reports_background_modules() {
    let (counts, sizes, d0) = planted_with_scattered(4);
    let n = counts.nrows();
    let group: Vec<u32> = vec![1; n];
    let part =
        partition_modules_by_group(&counts, &sizes, &group, &[3, 8], &[0, 4], None, 5).unwrap();
    assert_eq!(part.n_modules, 11);
    assert_eq!(part.background.len(), 11);
    let bg: Vec<u32> = (0..11u32)
        .filter(|&m| part.background[m as usize])
        .collect();
    assert_eq!(bg.len(), 1, "one background module for group 1: {bg:?}");
    for i in d0..n {
        assert_eq!(part.labels[i], bg[0]);
    }
}

/// Rows whose module is a flagged background (flat or scattered features)
/// are flagged; every other row is not.
#[test]
fn rows_in_background_modules_are_flagged() {
    let labels = [0u32, 1, 2, 1, 0];
    let background = [false, true, false];
    assert_eq!(
        background_rows(&labels, &background),
        vec![false, true, false, true, false]
    );
}
