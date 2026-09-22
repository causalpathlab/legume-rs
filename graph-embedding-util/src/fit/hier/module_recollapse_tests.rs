use super::*;
use crate::data::Triplet;
use crate::fit::hier::partition::Partition;
use crate::fit::hier::units::UnitTable;

fn t(cell: u32, feature: u32, count: f32) -> Triplet {
    Triplet {
        cell,
        feature,
        count,
    }
}

#[test]
fn identical_pb_profiles_in_different_modules_merge() {
    // 2 units; genes 0 and 1 have identical counts but sit in modules 0 and 1.
    let rna = vec![t(0, 0, 5.0), t(0, 1, 5.0), t(1, 0, 3.0), t(1, 1, 3.0)];
    let units = UnitTable::from_pseudobulk_axes(&[&[&rna]], &[2], &[2]);
    let part = Partition::from_labels(&[0, 1], 2);
    let (new_part, map) = recollapse_modules(&units, 0, &part, 0.99).expect("should merge");
    assert!(map.merged());
    assert_eq!(new_part.n_modules(), 1);
    assert_eq!(new_part.module_of, vec![0, 0]);
}

#[test]
fn other_axis_partition_unchanged_by_recollapse_on_axis_zero() {
    let rna = vec![t(0, 0, 5.0), t(1, 1, 3.0)];
    let atac = vec![t(0, 0, 1.0), t(1, 1, 2.0)];
    let units = UnitTable::from_pseudobulk_axes(&[&[&rna], &[&atac]], &[2], &[2, 2]);
    let gene = Partition::from_labels(&[0, 1], 2);
    let peak = Partition::from_labels(&[1, 0], 2);
    let peak_before = peak.module_of.clone();
    let _ = recollapse_modules(&units, 0, &gene, 0.99);
    assert_eq!(peak.module_of, peak_before);
}

/// 8 units, 6 features: 0..3 share profile A, 3..6 share profile B, each with a
/// small per-unit wobble so no two are exactly equal.
fn two_profile_units() -> Vec<Triplet> {
    let mut v = Vec::new();
    for u in 0..8u32 {
        for f in 0..6u32 {
            let a = u < 4;
            let fa = f < 3;
            let c = if a == fa {
                10.0 + (u % 2) as f32 + 0.1 * f as f32
            } else {
                1.0
            };
            v.push(t(u, f, c));
        }
    }
    v
}

/// An empty module never blocks compaction: merging around it yields a map
/// whose `n_modules` is the real module count and that reports `merged()`.
#[test]
fn an_empty_module_does_not_hide_a_merge() {
    let units = UnitTable::from_pseudobulk_axes(&[&[&two_profile_units()]], &[8], &[6]);
    // module 1 is empty; modules 0 and 2 hold profile-A features; 3 holds B.
    let part = Partition::from_labels(&[0, 0, 2, 3, 3, 3], 4);
    let (new_part, map) = recollapse_modules(&units, 0, &part, 0.9).expect("A merges");
    assert!(map.merged());
    assert_eq!(
        map.n_modules, 2,
        "profile A, profile B; no empty module kept"
    );
    assert_eq!(new_part.n_modules(), 2);
    assert!(new_part.members.iter().all(|m| !m.is_empty()));
    assert_eq!(new_part.module_of[0], new_part.module_of[2]);
    assert_ne!(new_part.module_of[0], new_part.module_of[3]);
}

/// Whenever the labels change, the map says so (so μ rows get compacted with
/// them); an unchanged labelling comes back as `None`.
#[test]
fn a_relabelling_is_never_silent() {
    let units = UnitTable::from_pseudobulk_axes(&[&[&two_profile_units()]], &[8], &[6]);
    // module 1 empty, nothing merges at this threshold.
    let part = Partition::from_labels(&[0, 0, 0, 2, 2, 2], 3);
    match recollapse_modules(&units, 0, &part, 0.999) {
        None => {}
        Some((new_part, map)) => {
            assert!(
                map.merged() || new_part.module_of == part.module_of,
                "labels moved but the map claims nothing changed"
            );
            for (old_m, mem) in part.members.iter().enumerate() {
                for &f in mem {
                    assert_eq!(
                        map.old_to_new[old_m] as usize, new_part.module_of[f as usize] as usize,
                        "feature {f}: map and labels disagree"
                    );
                }
            }
        }
    }
}

/// Exact average linkage: the module-pair similarity is the mean pairwise
/// cosine of their members' unit profiles, so the greedy merge order matches a
/// brute-force reference on a random fixture.
#[test]
fn merge_order_matches_brute_force_average_linkage() {
    // A tiny LCG keeps the fixture deterministic without a rand dependency.
    let mut state = 5u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f32) / (u32::MAX >> 1) as f32
    };
    let (n_u, n_f, n_m) = (12usize, 30usize, 6usize);
    let mut trip = Vec::new();
    let mut dense = vec![vec![0f32; n_u]; n_f];
    for (f, row) in dense.iter_mut().enumerate() {
        let program = f % 3;
        for (u, cell) in row.iter_mut().enumerate() {
            if next() < 0.6 {
                let c = if u % 3 == program {
                    5.0 + 10.0 * next()
                } else {
                    0.5 + 1.5 * next()
                };
                trip.push(t(u as u32, f as u32, c));
                *cell = c;
            }
        }
    }
    let units = UnitTable::from_pseudobulk_axes(&[&[&trip]], &[n_u], &[n_f]);
    let labels: Vec<u32> = (0..n_f).map(|f| (f % n_m) as u32).collect();
    let part = Partition::from_labels(&labels, n_m);

    // Brute force: normalize, greedy merge on mean pairwise cosine.
    let unit = |v: &[f32]| {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter()
            .map(|x| if n > 0.0 { x / n } else { 0.0 })
            .collect::<Vec<_>>()
    };
    let prof: Vec<Vec<f32>> = dense.iter().map(|v| unit(v)).collect();
    let mut clusters: Vec<Vec<usize>> = part
        .members
        .iter()
        .map(|m| m.iter().map(|&f| f as usize).collect())
        .collect();
    let min_cos = 0.5f32;
    loop {
        let mut best = (f32::NEG_INFINITY, 0, 0);
        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                let mut s = 0.0;
                for &a in &clusters[i] {
                    for &b in &clusters[j] {
                        s += prof[a]
                            .iter()
                            .zip(&prof[b])
                            .map(|(x, y)| x * y)
                            .sum::<f32>();
                    }
                }
                let sim = s / (clusters[i].len() * clusters[j].len()) as f32;
                if sim > best.0 {
                    best = (sim, i, j);
                }
            }
        }
        if best.0 < min_cos {
            break;
        }
        let (_, i, j) = best;
        let mut mj = clusters.remove(j);
        clusters[i].append(&mut mj);
    }
    let mut want: Vec<Vec<usize>> = clusters
        .into_iter()
        .map(|mut c| {
            c.sort();
            c
        })
        .collect();
    want.sort();

    let (new_part, _) = recollapse_modules(&units, 0, &part, min_cos).expect("some merge");
    let mut got: Vec<Vec<usize>> = new_part
        .members
        .iter()
        .filter(|m| !m.is_empty())
        .map(|m| m.iter().map(|&f| f as usize).collect())
        .collect();
    got.sort();
    assert_eq!(got, want);
}

/// On a tracked axis 0 the unit rows are `(track, gene)` rows and the partition
/// is over GENES: a gene's profile spans every track's row, so a gene whose
/// track-1 row differs does not merge with genes it matches on track 0 alone.
#[test]
fn a_tracked_axis_profiles_genes_over_every_track() {
    use crate::fit::config::{TrackInfo, TrackSpec};
    let (n_g, n_u) = (3u32, 6u32);
    let mut trip = Vec::new();
    for u in 0..n_u {
        for g in 0..n_g {
            // track 0: every gene identical
            trip.push(t(u, g, 5.0 + (u % 2) as f32));
            // track 1: genes 0,1 identical, gene 2 anti-correlated
            let c1 = if g < 2 {
                5.0 + (u % 2) as f32
            } else {
                5.0 - (u % 2) as f32 * 4.0
            };
            trip.push(t(u, g + n_g, c1));
        }
    }
    let n_rows = 2 * n_g as usize;
    let tracks = TrackSpec {
        track_of_row: (0..n_rows).map(|r| (r >= n_g as usize) as u32).collect(),
        gene_of_row: (0..n_rows).map(|r| (r % n_g as usize) as u32).collect(),
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
        n_rows,
        tracks,
    );
    let part = Partition::from_labels(&[0, 1, 2], 3);
    let (new_part, _) = recollapse_modules(&units, 0, &part, 0.99).expect("0 and 1 merge");
    assert_eq!(new_part.module_of[0], new_part.module_of[1]);
    assert_ne!(
        new_part.module_of[0], new_part.module_of[2],
        "gene 2 differs on track 1"
    );
    assert_eq!(new_part.n_modules(), 2);
}
