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

/// Every family asks the same question before it warm-starts: is this run's
/// gene axis the source run's, and if not, how do the two line up? The answer
/// comes from the source run's own feature-mean row order.
mod for_init_from {
    use super::super::remap_for_init_from;
    use auxiliary_data::feature_names::FeatureNameKindArg;
    use crate::topic::model_metadata::save_feature_mean;

    fn genes(names: &[&str]) -> Vec<Box<str>> {
        names.iter().map(|g| (*g).into()).collect()
    }

    fn source(dir: &std::path::Path, names: &[&str]) -> String {
        let prefix = dir.join("src").to_string_lossy().into_owned();
        save_feature_mean(&vec![1.0; names.len()], &genes(names), &prefix).unwrap();
        prefix
    }

    #[test]
    fn no_source_run_means_no_alignment_to_make() {
        let r =
            remap_for_init_from(None, &FeatureNameKindArg::default(), &genes(&["a", "b"])).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn the_same_axis_in_the_same_order_needs_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = source(dir.path(), &["a", "b", "c"]);
        let r = remap_for_init_from(
            Some(&prefix),
            &FeatureNameKindArg::default(),
            &genes(&["a", "b", "c"]),
        )
        .unwrap();
        assert!(r.is_none(), "an identical axis takes the exact warm start");
    }

    #[test]
    fn a_reordered_axis_with_a_new_gene_lines_up_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = source(dir.path(), &["a", "b", "c"]);
        let r = remap_for_init_from(
            Some(&prefix),
            &FeatureNameKindArg::default(),
            &genes(&["c", "new", "a"]),
        )
        .unwrap()
        .expect("the axes differ, so they have to be aligned");
        assert_eq!(r.new_to_train, vec![Some(2), None, Some(0)]);
        assert_eq!(r.d_train, 3);
        assert_eq!(r.n_mapped, 2);
    }

    /// The rule this run reads its own row names under is the rule the source
    /// run's names are matched under, so one flag means one thing per command.
    ///
    /// The direction matters: the fallback matcher already reduces a suffixed
    /// axis to a bare symbol, but not the reverse, so a run whose files carry
    /// the suffixed spelling only reaches a bare-symbol source run when the
    /// name rule canonicalizes it first.
    #[test]
    fn the_name_rule_decides_what_counts_as_the_same_gene() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = source(dir.path(), &["A", "B"]);
        let under = |kind: FeatureNameKindArg| {
            remap_for_init_from(Some(&prefix), &kind, &genes(&["ENSG1_A", "ENSG2_B"]))
                .map(|r| r.map_or(2, |r| r.n_mapped))
        };
        assert_eq!(under(FeatureNameKindArg::Gene).unwrap(), 2, "the suffix is canonicalized away");
        assert_eq!(under(FeatureNameKindArg::Auto).unwrap(), 2, "auto resolves to the gene rule");
        // Exact spelling was asked for, so the suffixed names are different
        // genes and the two axes share nothing at all.
        assert!(under(FeatureNameKindArg::Exact).is_err(), "exact spelling was asked for");
    }

    /// An `--init-from` that shares no gene with this run is a retrain wearing
    /// a continuation's clothes, so it is refused rather than logged.
    #[test]
    fn an_axis_with_nothing_in_common_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = source(dir.path(), &["a", "b"]);
        let err = match remap_for_init_from(
            Some(&prefix),
            &FeatureNameKindArg::default(),
            &genes(&["x", "y"]),
        ) {
            Ok(_) => panic!("an axis sharing nothing must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("No query gene maps"), "{err}");
    }
}
