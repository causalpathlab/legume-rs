//! Gene / peak feature partitions for the hierarchical embed.

mod common;

use chickpea::p2g::link_map::PeakGeneEdge;
use chickpea::p2g::module_init::{coarsen_profile_levels, init_gene_peak_partitions};
use chickpea::p2g::pb_levels::PbLevels;
use common::mat;

fn edge(peak: usize, gene: usize, weight: f32) -> PeakGeneEdge {
    PeakGeneEdge { peak, gene, weight }
}

/// Six features over five pbs in two programs: 0..3 are expressed in pbs 0..2,
/// 3..6 in pbs 2..5, with within-program magnitudes that differ so the k-means
/// sees shape, not identical rows.
fn two_program_profile(scale: f32) -> chickpea::common::Mat {
    mat(6, 5, |f, s| {
        let program = usize::from(f >= 3);
        let on = if program == 0 { s < 2 } else { s >= 2 };
        if on {
            scale * (3.0 + f as f32 + (s % 2) as f32)
        } else {
            0.0
        }
    })
}

fn levels(rna: chickpea::common::Mat, atac: chickpea::common::Mat) -> PbLevels {
    PbLevels {
        rna: Some(vec![rna]),
        atac: vec![atac],
        parent: vec![],
    }
}

/// The warm start clusters the real profile: co-expressed genes share a module
/// and the two programs are apart. A layout bug in the profile hand-off would
/// scramble this.
#[test]
fn warm_start_groups_genes_by_their_pb_profile() {
    let rna = two_program_profile(1.0);
    let lv = levels(rna.clone(), two_program_profile(0.5));
    let (gene_part, _) = init_gene_peak_partitions(&lv, &rna, &[], 2, 2, 7).unwrap();
    let m = &gene_part.module_of;
    assert_eq!(m[0], m[1]);
    assert_eq!(m[1], m[2]);
    assert_eq!(m[3], m[4]);
    assert_eq!(m[4], m[5]);
    assert_ne!(m[0], m[3], "the two programs must not share a module");
}

#[test]
fn warm_start_groups_peaks_by_their_pb_profile() {
    let rna = two_program_profile(1.0);
    let lv = levels(rna.clone(), two_program_profile(0.5));
    let (_, peak_part) = init_gene_peak_partitions(&lv, &rna, &[], 2, 2, 7).unwrap();
    let m = &peak_part.module_of;
    assert_eq!(m[0], m[1]);
    assert_eq!(m[1], m[2]);
    assert_eq!(m[3], m[4]);
    assert_eq!(m[4], m[5]);
    assert_ne!(m[0], m[3]);
}

/// A coarse level has one column per PARENT pb, holding the sum of its children.
#[test]
fn coarsen_sums_children_into_parent_columns() {
    let finest = mat(2, 4, |g, s| (g * 4 + s) as f32);
    let parent = vec![vec![0usize, 0, 1, 1]];
    let out = coarsen_profile_levels(&finest, &parent);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], finest);
    assert_eq!((out[1].nrows(), out[1].ncols()), (2, 2));
    assert_eq!(out[1][(0, 0)], 0.0 + 1.0);
    assert_eq!(out[1][(0, 1)], 2.0 + 3.0);
    assert_eq!(out[1][(1, 0)], 4.0 + 5.0);
    assert_eq!(out[1][(1, 1)], 6.0 + 7.0);
}

/// A linked peak takes the module of its strongest linked gene, whatever the
/// edge order; peaks with no link are clustered on their own, in modules no
/// linked peak uses.
#[test]
fn linked_peaks_take_their_strongest_genes_module_and_unlinked_peaks_stay_apart() {
    let rna = two_program_profile(1.0);
    let lv = levels(rna.clone(), two_program_profile(0.5));
    let n_gene_modules = 2;
    // peak 0: weak link to gene 0 listed AFTER a strong link to gene 3;
    // peak 1: the reverse order; peaks 2..6 unlinked.
    let edges = vec![
        edge(0, 3, 0.9),
        edge(0, 0, 0.2),
        edge(1, 0, 0.2),
        edge(1, 3, 0.9),
    ];
    let (gene_part, peak_part) =
        init_gene_peak_partitions(&lv, &rna, &edges, n_gene_modules, 2, 7).unwrap();
    let g = &gene_part.module_of;
    let p = &peak_part.module_of;
    assert_eq!(p[0], g[3], "peak 0 follows its strongest gene");
    assert_eq!(p[1], g[3], "peak 1 follows its strongest gene");
    for (peak, &m) in p.iter().enumerate().skip(2) {
        assert!(
            m as usize >= n_gene_modules,
            "unlinked peak {peak} landed in a linked module {m}"
        );
    }
    // Unlinked peaks are still grouped by their own profile.
    assert_eq!(p[3], p[4]);
    assert_eq!(p[4], p[5]);
    assert_ne!(p[2], p[3]);
}
