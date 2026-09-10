//! Carrying a masked model's gene-keyed state onto an axis the source run did
//! not have: the alignment that decides whether anything is needed, and the
//! module-mean restart of an unseen gene's ρ row.

use super::fill_rows_by_module;
use crate::embed_common::Mat;
use crate::topic::eval::GeneRemap;
use data_beans_alg::feature_coarsening::FeatureCoarsening;

#[test]
fn an_identical_axis_is_recognised_so_the_exact_path_is_taken() {
    let same = GeneRemap {
        new_to_train: vec![Some(0), Some(1), Some(2)],
        d_train: 3,
        n_mapped: 3,
    };
    assert!(same.is_identity());
    let reordered = GeneRemap {
        new_to_train: vec![Some(1), Some(0), Some(2)],
        d_train: 3,
        n_mapped: 3,
    };
    assert!(!reordered.is_identity());
    let shorter = GeneRemap {
        new_to_train: vec![Some(0), Some(1)],
        d_train: 3,
        n_mapped: 2,
    };
    assert!(!shorter.is_identity());
}

/// Axis [g1 gX g3 gY g0], modules {g1,g0} = 0 and {g3} = 1 with gX in 0 and gY
/// in 1: each unknown row becomes the mean of its module's KNOWN rows.
#[test]
fn an_unseen_gene_restarts_at_the_mean_of_its_module_known_members() {
    let modules = FeatureCoarsening::from_fine_to_coarse(vec![0, 0, 1, 1, 0], 2).unwrap();
    let known = [true, false, true, false, true];
    let mut rho = Mat::from_row_slice(
        5,
        2,
        &[
            1.0, 2.0, // g1
            9.0, 9.0, // gX: whatever the loader left here
            5.0, 6.0, // g3
            9.0, 9.0, // gY
            3.0, 4.0, // g0
        ],
    );
    fill_rows_by_module(&mut rho, &known, &modules).unwrap();
    assert_eq!(rho.row(1).iter().copied().collect::<Vec<_>>(), vec![2.0, 3.0]);
    assert_eq!(rho.row(3).iter().copied().collect::<Vec<_>>(), vec![5.0, 6.0]);
    assert_eq!(rho.row(0).iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0], "known rows untouched");
}

#[test]
fn an_unseen_gene_in_a_module_with_no_known_member_is_refused() {
    let modules = FeatureCoarsening::from_fine_to_coarse(vec![0, 1], 2).unwrap();
    let mut rho = Mat::zeros(2, 3);
    assert!(fill_rows_by_module(&mut rho, &[true, false], &modules).is_err());
}
