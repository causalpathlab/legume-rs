use super::*;

fn boxed(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| (*s).into()).collect()
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

fn citeseq_record() -> RunMultiome {
    RunMultiome {
        modality: strings(&["scRNA", "scADT", "scRNA", "scADT"]),
        group: strings(&["BMMC_D1T1", "BMMC_D1T1", "PBMC_D4T1", "PBMC_D4T1"]),
        barcode_tagged: true,
    }
}

fn plain(n: usize) -> ReadSharedRowsArgs {
    ReadSharedRowsArgs {
        data_files: (0..n).map(|i| format!("f{i}.zarr").into()).collect(),
        preload: true,
        ..Default::default()
    }
}

/// A single-modality run records nothing, and the replay is then exactly the
/// loader's own defaults — applying it to a load must change nothing.
#[test]
fn no_record_replays_as_the_plain_load() {
    let out = recorded_layout(None, 4).unwrap().apply(plain(4)).unwrap();
    assert_eq!(out.column_alignment, ColumnAlignment::Disjoint);
    assert!(out.feature_kind.is_none());
    assert!(out.per_file_feature_suffix.is_none());
    assert!(out.per_file_barcode_suffix.is_none());
}

#[test]
fn recorded_layout_replays_alignment_and_both_suffixes() {
    let out = recorded_layout(Some(&citeseq_record()), 4)
        .unwrap()
        .apply(plain(4))
        .unwrap();
    assert_eq!(out.column_alignment, ColumnAlignment::Union);
    assert!(matches!(out.feature_kind, Some(FeatureNameKind::Mixed)));
    assert_eq!(
        out.per_file_feature_suffix.as_deref().unwrap(),
        boxed(&["scRNA", "scADT", "scRNA", "scADT"]).as_slice()
    );
    let bs = out.per_file_barcode_suffix.unwrap();
    assert_eq!(bs[0].as_deref(), Some("BMMC_D1T1"));
    assert_eq!(bs[3].as_deref(), Some("PBMC_D4T1"));
}

/// One group needs no barcode tag; replaying one would rename every cell.
#[test]
fn untagged_record_replays_without_barcode_suffixes() {
    let r = RunMultiome {
        modality: strings(&["rna", "atac"]),
        group: strings(&["g0", "g0"]),
        barcode_tagged: false,
    };
    let out = recorded_layout(Some(&r), 2)
        .unwrap()
        .apply(plain(2))
        .unwrap();
    assert_eq!(out.column_alignment, ColumnAlignment::Union);
    assert!(out.per_file_barcode_suffix.is_none());
}

/// The record is positional. A caller that swapped the file list would
/// silently namespace the wrong rows, so refuse rather than guess.
#[test]
fn record_of_the_wrong_length_is_an_error() {
    let r = citeseq_record();
    let err = recorded_layout(Some(&r), 3).unwrap_err().to_string();
    assert!(err.contains('4') && err.contains('3'), "{err}");
}

/// Query files are a fresh set, so their layout is detected. The detected
/// modality tags come from THEIR filenames and need not match the training
/// run's, so they are renamed by which trained row block their features
/// actually land in.
#[test]
fn query_modalities_are_renamed_to_the_trained_ones() {
    let trained = boxed(&[
        "CD3E/scRNA",
        "MS4A1/scRNA",
        "LYZ/scRNA",
        "ADT-CD14/scADT",
        "ADT-CD3/scADT",
    ]);
    let detected = ge::MultiomePlan {
        files: boxed(&["gex_q.zarr", "prot_q.zarr"]),
        modality: boxed(&["gex", "prot"]),
        group: boxed(&["q", "q"]),
        group_sizes: vec![2],
        barcode_tagged: false,
        n_bridge_cells: Some(500),
    };
    let rows = vec![
        boxed(&["CD3E", "MS4A1", "LYZ"]),
        boxed(&["ADT-CD14", "ADT-CD3"]),
    ];
    let named = reconcile_modalities(detected, &rows, &trained).unwrap();
    assert_eq!(&*named.modality[0], "scRNA");
    assert_eq!(&*named.modality[1], "scADT");
}

/// A query modality whose features match no trained block keeps its own tag —
/// renaming it would claim an overlap that is not there.
#[test]
fn an_unmatched_query_modality_keeps_its_detected_tag() {
    let trained = boxed(&["CD3E/scRNA", "MS4A1/scRNA"]);
    let detected = ge::MultiomePlan {
        files: boxed(&["gex_q.zarr", "atac_q.zarr"]),
        modality: boxed(&["gex", "atac"]),
        group: boxed(&["q", "q"]),
        group_sizes: vec![2],
        barcode_tagged: false,
        n_bridge_cells: Some(10),
    };
    let rows = vec![
        boxed(&["CD3E", "MS4A1"]),
        boxed(&["chr1:100-200", "chr2:300-400"]),
    ];
    let named = reconcile_modalities(detected, &rows, &trained).unwrap();
    assert_eq!(&*named.modality[0], "scRNA");
    assert_eq!(&*named.modality[1], "atac");
}

/// Two query modalities must not both claim one trained block: that would put
/// two different assays on the same rows.
#[test]
fn two_query_modalities_cannot_claim_one_trained_block() {
    let trained = boxed(&["CD3E/scRNA", "MS4A1/scRNA", "LYZ/scRNA"]);
    let detected = ge::MultiomePlan {
        files: boxed(&["a.zarr", "b.zarr"]),
        modality: boxed(&["a", "b"]),
        group: boxed(&["q", "q"]),
        group_sizes: vec![2],
        barcode_tagged: false,
        n_bridge_cells: Some(10),
    };
    let rows = vec![boxed(&["CD3E", "MS4A1"]), boxed(&["MS4A1", "LYZ"])];
    let err = reconcile_modalities(detected, &rows, &trained)
        .unwrap_err()
        .to_string();
    assert!(err.contains("scRNA"), "{err}");
}

/// A trained run with no `/modality` rows is single-modality; a query is then
/// loaded the plain way even if its own files look multi-modal.
#[test]
fn a_single_modality_model_takes_no_query_layout() {
    let trained = boxed(&["CD3E", "MS4A1", "LYZ"]);
    assert!(trained_modalities(&trained).is_empty());
}

#[test]
fn trained_modalities_are_read_off_the_row_names() {
    let trained = boxed(&["CD3E/scRNA", "ADT-CD3/scADT", "MS4A1/scRNA"]);
    let mut m = trained_modalities(&trained);
    m.sort();
    assert_eq!(m, boxed(&["scADT", "scRNA"]));
}

/// The plain load must pass through `apply` untouched — every
/// single-modality call site goes through it and none may change.
#[test]
fn apply_is_a_no_op_for_a_plain_load() {
    let base = ReadSharedRowsArgs {
        data_files: boxed(&["a.zarr", "b.zarr"]),
        preload: true,
        feature_kind: Some(FeatureNameKind::Gene { delim: '_' }),
        ..Default::default()
    };
    let out = ReloadLayout::default().apply(base).unwrap();
    assert_eq!(out.column_alignment, ColumnAlignment::Disjoint);
    assert!(matches!(
        out.feature_kind,
        Some(FeatureNameKind::Gene { delim: '_' })
    ));
    assert!(out.per_file_feature_suffix.is_none());
    assert!(out.per_file_barcode_suffix.is_none());
    assert!(out.preload);
}

#[test]
fn apply_stamps_a_multiome_layout_and_keeps_the_rest() {
    let l = recorded_layout(Some(&citeseq_record()), 4).unwrap();
    assert!(l.is_multiome());
    let out = l.apply(plain(4)).unwrap();
    assert_eq!(out.per_file_feature_suffix.unwrap().len(), 4);
    assert!(out.preload, "caller's own settings must survive");
}

/// Same rule on the model side: a partially-tagged trained axis is a
/// single-modality model, not a multiome one, so a query must not be routed
/// down the multiome path on the strength of one stray separator.
#[test]
fn a_partially_tagged_model_axis_is_single_modality() {
    assert!(trained_modalities(&boxed(&["A/B", "CD3E", "LYZ"])).is_empty());
    assert!(trained_modalities(&boxed(&["CD3E", "LYZ"])).is_empty());
    assert!(trained_modalities(&boxed(&[])).is_empty());
    assert_eq!(
        trained_modalities(&boxed(&["CD3E/scRNA", "ADT-CD3/scADT"])).len(),
        2
    );
}

/// faba/gem rows are `{gene}/{modality}/{channel}` — two separators, and the
/// trailing field is a splice channel, not a modality. Reading that as
/// multiome would re-scope a gem model's whole feature axis on query.
#[test]
fn a_gem_splice_axis_is_not_a_multiome_model() {
    let gem = boxed(&["G1/count/spliced", "G1/count/unspliced", "G2/count/spliced"]);
    assert!(trained_modalities(&gem).is_empty());
    assert!(query_layout(&boxed(&["a.zarr", "b.zarr"]), &gem)
        .unwrap()
        .is_none());
}

/// A caller's own feature-name rule survives a plain load — `apply` must not
/// reach into a field the layout has no business setting.
#[test]
fn a_plain_load_keeps_the_callers_feature_kind() {
    let mut base = plain(2);
    base.feature_kind = Some(FeatureNameKind::Gene { delim: '_' });
    let out = ReloadLayout::default().apply(base).unwrap();
    assert!(matches!(
        out.feature_kind,
        Some(FeatureNameKind::Gene { delim: '_' })
    ));
}

/// A multiome axis carries gene names and chrX:s-e loci together, so only
/// `Mixed` can canonicalize it. An explicit rule there is refused rather than
/// silently overridden — the override would be invisible and would break the
/// axis match against the dictionary.
#[test]
fn an_explicit_feature_kind_on_a_multiome_axis_is_refused() {
    let l = recorded_layout(Some(&citeseq_record()), 4).unwrap();
    let mut base = plain(4);
    base.feature_kind = Some(FeatureNameKind::Gene { delim: '_' });
    // `ReadSharedRowsArgs` is not Debug, so `unwrap_err` is unavailable.
    let err = match l.apply(base) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("an explicit feature kind must be refused on a multiome axis"),
    };
    assert!(err.contains("Gene"), "{err}");
    assert!(err.contains("scRNA") && err.contains("scADT"), "{err}");
}

/// Passing `Mixed` explicitly agrees with what the layout needs, so it loads.
#[test]
fn an_explicit_mixed_kind_is_accepted_on_a_multiome_axis() {
    let l = recorded_layout(Some(&citeseq_record()), 4).unwrap();
    let mut base = plain(4);
    base.feature_kind = Some(FeatureNameKind::Mixed);
    assert!(l.apply(base).is_ok());
}
