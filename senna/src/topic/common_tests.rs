//! Inheriting a source run's coarsening ladder under `--init-from`.

use super::inherit_level_coarsenings;
use crate::embed_common::Mat;
use crate::topic::eval::GeneRemap;
use crate::topic::model_metadata::save_coarsening_levels;
use data_beans_alg::feature_coarsening::FeatureCoarsening;

fn coarsening(fine_to_coarse: &[usize], num_coarse: usize) -> FeatureCoarsening {
    FeatureCoarsening::from_fine_to_coarse(fine_to_coarse.to_vec(), num_coarse).unwrap()
}

/// Same gene COUNT as the source run, but the axis is reordered and one gene
/// swapped: the level has to be grown by name, never inherited positionally.
#[test]
fn a_reordered_axis_of_the_same_length_is_grown_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("source").to_string_lossy().into_owned();
    // Source axis [g0 g1 g2 g3]: modules {g0,g1} and {g2,g3}.
    save_coarsening_levels(&[Some(coarsening(&[0, 0, 1, 1], 2))], &prefix).unwrap();
    // This run: [g3 g1 gX g0] — same length, g2 gone, gX new.
    let remap = GeneRemap {
        new_to_train: vec![Some(3), Some(1), None, Some(0)],
        d_train: 4,
        n_mapped: 3,
    };
    // gX's profile sits with module 1's genes.
    let profiles = Mat::from_row_slice(4, 2, &[0.0, 1.0, 1.0, 0.0, 0.1, 0.9, 1.0, 0.0]);
    let levels = inherit_level_coarsenings(&prefix, 1, 4, Some(&remap), &profiles).unwrap();
    let fc = levels[0].as_ref().unwrap();
    assert_eq!(fc.fine_to_coarse, vec![1, 0, 1, 0]);
}

/// Without a remap the same-length case is the exact axis and stays positional.
#[test]
fn without_a_remap_a_same_length_level_is_inherited_as_is() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("source").to_string_lossy().into_owned();
    save_coarsening_levels(&[Some(coarsening(&[0, 0, 1, 1], 2))], &prefix).unwrap();
    let profiles = Mat::zeros(4, 1);
    let levels = inherit_level_coarsenings(&prefix, 1, 4, None, &profiles).unwrap();
    assert_eq!(levels[0].as_ref().unwrap().fine_to_coarse, vec![0, 0, 1, 1]);
}
