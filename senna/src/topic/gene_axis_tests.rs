//! Growing a masked model's gene-keyed state onto an axis the source run did not
//! have: known genes keep what they had, unknown genes are placed by the
//! pseudobulk profile, and the module set the decoders are keyed to never
//! changes.

use super::{grow_fine_to_coarse, grow_rho, GeneAxisRemap};
use crate::embed_common::Mat;
use data_beans_alg::feature_coarsening::FeatureCoarsening;

fn coarsening(fine_to_coarse: &[usize], num_coarse: usize) -> FeatureCoarsening {
    let mut coarse_to_fine = vec![Vec::new(); num_coarse];
    for (g, &m) in fine_to_coarse.iter().enumerate() {
        coarse_to_fine[m].push(g);
    }
    FeatureCoarsening {
        fine_to_coarse: fine_to_coarse.to_vec(),
        coarse_to_fine,
        num_coarse,
    }
}

/// Source axis [g0 g1 g2 g3], modules {g0,g1} and {g2,g3}. This run's axis is
/// [g1 gX g3 gY g0]: two of the source run's genes reordered, one dropped, two new.
fn fixture() -> (FeatureCoarsening, GeneAxisRemap, Mat) {
    let source = coarsening(&[0, 0, 1, 1], 2);
    let remap = GeneAxisRemap {
        new_to_source: vec![Some(1), None, Some(3), None, Some(0)],
        n_source: 4,
    };
    // Two pseudobulks. Module 0's genes live on the first, module 1's on the
    // second; gX leans to the first, gY to the second.
    let profiles = Mat::from_row_slice(
        5,
        2,
        &[
            1.0, 0.0, // g1
            0.9, 0.1, // gX
            0.0, 1.0, // g3
            0.1, 0.9, // gY
            1.0, 0.0, // g0
        ],
    );
    (source, remap, profiles)
}

#[test]
fn known_genes_keep_their_module_and_new_ones_join_the_nearest() {
    let (source, remap, profiles) = fixture();
    let grown = grow_fine_to_coarse(&source, &remap, &profiles).unwrap();
    assert_eq!(grown.fine_to_coarse, vec![0, 0, 1, 1, 0]);
    assert_eq!(grown.num_coarse, 2, "the decoders are keyed to the module count");
    assert_eq!(grown.coarse_to_fine[0], vec![0, 1, 4]);
    assert_eq!(grown.coarse_to_fine[1], vec![2, 3]);
}

#[test]
fn a_module_with_no_surviving_member_attracts_nothing_but_keeps_its_index() {
    // Three source run's modules; module 2's only gene is absent from this run.
    let source = coarsening(&[0, 1, 2], 3);
    let remap = GeneAxisRemap {
        new_to_source: vec![Some(0), Some(1), None],
        n_source: 3,
    };
    // The new gene's profile is nowhere near module 0 or 1 in particular, but
    // it must land in one of them: module 2 has no members to compare against.
    let profiles = Mat::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let grown = grow_fine_to_coarse(&source, &remap, &profiles).unwrap();
    assert_eq!(grown.num_coarse, 3);
    assert!(grown.coarse_to_fine[2].is_empty());
    assert!(matches!(grown.fine_to_coarse[2], 0 | 1));
}

#[test]
fn nothing_in_common_is_an_error_not_a_guess() {
    let source = coarsening(&[0, 0], 1);
    let remap = GeneAxisRemap {
        new_to_source: vec![None, None],
        n_source: 2,
    };
    let profiles = Mat::from_row_slice(2, 1, &[1.0, 1.0]);
    assert!(grow_fine_to_coarse(&source, &remap, &profiles).is_err());
}

#[test]
fn rho_rows_are_copied_for_known_genes_and_module_means_for_new_ones() {
    let (source, remap, profiles) = fixture();
    let grown = grow_fine_to_coarse(&source, &remap, &profiles).unwrap();
    // Source ρ, H = 2, one distinct row per source gene.
    let source_rho = Mat::from_row_slice(4, 2, &[1.0, 0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 7.0]);
    let rho = grow_rho(&source_rho, &remap, &grown).unwrap();
    assert_eq!(rho.nrows(), 5);
    assert_eq!(rho.row(0), source_rho.row(1), "g1 copies");
    assert_eq!(rho.row(2), source_rho.row(3), "g3 copies");
    assert_eq!(rho.row(4), source_rho.row(0), "g0 copies");
    // gX joined module 0, whose surviving members are g1 and g0: mean of rows 1 and 0.
    assert_eq!(rho.row(1), Mat::from_row_slice(1, 2, &[2.0, 0.0]).row(0));
    // gY joined module 1, whose only surviving member is g3.
    assert_eq!(rho.row(3), source_rho.row(3));
}

#[test]
fn an_identical_axis_is_recognised_so_the_exact_path_is_taken() {
    let same = GeneAxisRemap {
        new_to_source: vec![Some(0), Some(1), Some(2)],
        n_source: 3,
    };
    assert!(same.is_identity());
    let reordered = GeneAxisRemap {
        new_to_source: vec![Some(1), Some(0), Some(2)],
        n_source: 3,
    };
    assert!(!reordered.is_identity());
    assert_eq!(reordered.n_matched(), 3);
}
