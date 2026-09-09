use super::*;

fn boxed(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

/// Without `label=`, the modality tag is the file's position in its group, so
/// the same tag names the same assay across groups.
#[test]
fn declared_groups_tag_modality_by_position() {
    let plan = declared_plan(&boxed(&["rna1.zarr,atac1.zarr", "rna2.zarr,atac2.zarr"]))
        .unwrap()
        .expect("two groups of two");
    assert_eq!(plan.group_sizes, vec![2, 2]);
    assert_eq!(&*plan.modality[0], "m0");
    assert_eq!(&*plan.modality[1], "m1");
    assert_eq!(&*plan.modality[2], "m0");
    assert_eq!(&*plan.modality[3], "m1");
    assert_eq!(&*plan.files[2], "rna2.zarr");
    assert_eq!(&*plan.group[0], "g0");
    assert_eq!(&*plan.group[3], "g1");
}

#[test]
fn declared_labels_override_positions() {
    let plan = declared_plan(&boxed(&["spliced=s.zarr,unspliced=u.zarr"]))
        .unwrap()
        .expect("one group of two");
    assert_eq!(&*plan.modality[0], "spliced");
    assert_eq!(&*plan.modality[1], "unspliced");
}

/// Declared groups keep the documented contract: no barcode namespacing, so a
/// cross-group collision still surfaces as an error rather than silently
/// merging two donors.
#[test]
fn declared_groups_are_not_barcode_namespaced() {
    let plan = declared_plan(&boxed(&["a1.zarr,b1.zarr", "a2.zarr,b2.zarr"]))
        .unwrap()
        .expect("two groups of two");
    assert!(!plan.barcode_tagged);
    assert!(plan.n_bridge_cells.is_none());
}

#[test]
fn declared_empty_group_is_an_error() {
    assert!(declared_plan(&boxed(&["rna.zarr,atac.zarr", ""])).is_err());
}

/// One file has nothing to glue against, and `auto_plan` says so without
/// logging a "reason" on every single-file run.
#[test]
fn one_file_is_never_auto_detected() {
    assert!(auto_plan(&boxed(&["only.zarr"]), 0, false)
        .unwrap()
        .is_none());
}

/// A carried pseudobulk reference is positional, and Union loading does not
/// promise to keep it contiguous.
#[test]
fn carried_pseudobulks_veto_auto_detection() {
    assert_eq!(
        auto_detect_veto(0, true),
        Some(NoAutoDetect::CarriedPseudobulks)
    );
}

/// Per-file batch labels only compose when each cell lives in one file.
#[test]
fn per_file_batch_labels_veto_auto_detection() {
    assert_eq!(
        auto_detect_veto(4, false),
        Some(NoAutoDetect::PerFileBatchFiles)
    );
    // One unified batch file is fine — that is what Union mode wants.
    assert_eq!(auto_detect_veto(1, false), None);
    assert_eq!(auto_detect_veto(0, false), None);
}

/// A single-file `--multiome` has nothing to glue against. It must not produce
/// a plan: the load would fall back to Disjoint while the feature suffixes and
/// the recorded layout still claimed Union.
#[test]
fn a_single_file_declaration_is_not_a_layout() {
    assert!(declared_plan(&boxed(&["only.zarr"])).unwrap().is_none());
}
