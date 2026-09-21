use super::{resolve_gem_preset, row_name_of};
use crate::gem::tracks::assign_tracks;
use graph_embedding_util::PresetMode;
use legume_numeric::matrix::traits::IoOps;
use nalgebra::DMatrix;

fn names(rows: &[&str]) -> Vec<Box<str>> {
    rows.iter().map(|&s| s.into()).collect()
}

/// A source run at `dir/src`: `{src}.feature_embedding.parquet` over `rows`,
/// H = 2, row `i` = `[i, 10 + i]`.
fn write_source(dir: &std::path::Path, rows: &[&str]) -> String {
    let prefix = dir.join("src").to_string_lossy().into_owned();
    let m = DMatrix::<f32>::from_fn(rows.len(), 2, |i, k| i as f32 + 10.0 * k as f32);
    m.to_parquet_with_names(
        &format!("{prefix}.feature_embedding.parquet"),
        (Some(&names(rows)), Some("gene")),
        None,
    )
    .unwrap();
    prefix
}

fn axis() -> Vec<Box<str>> {
    names(&[
        "ENSG1_GENE1/count/spliced",
        "ENSG1_GENE1/count/unspliced",
        "GENE2/count/spliced",
        "ENSG1_GENE1/m6a/methylated",
    ])
}

#[test]
fn row_name_of_maps_a_bare_name_onto_the_spliced_row_only() {
    assert_eq!(row_name_of("GENE1").as_ref(), "GENE1/count/spliced");
    assert_eq!(
        row_name_of("ENSG1_GENE1").as_ref(),
        "ENSG1_GENE1/count/spliced"
    );
    assert_eq!(
        row_name_of("GENE1/m6a/methylated").as_ref(),
        "GENE1/m6a/methylated"
    );
    assert_eq!(
        row_name_of("GENE1/count/unspliced").as_ref(),
        "GENE1/count/unspliced"
    );
}

/// A bare name is the gene's `count/spliced` row (matched by canonical gene
/// key); a name in the grammar is that row. Base rows become the preset on
/// the gene axis, given rows on other tracks become those tracks' offsets
/// relative to the base row, and the rows that match nothing are carried
/// under their lifted names.
#[test]
fn bare_names_are_spliced_rows_and_track_rows_are_offsets_on_their_base() {
    let axis = axis();
    let plan = assign_tracks(&axis).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let src = write_source(
        dir.path(),
        &[
            "GENE1",
            "GENE2/count/spliced",
            "GENE1/count/unspliced",
            "GENE1/m6a/methylated",
            "GENE3",
            "GENE2/count/unspliced",
        ],
    );
    let p = resolve_gem_preset(Some((&src, PresetMode::Freeze)), &axis, &plan).unwrap();
    let base = p.base.expect("base rows");
    assert_eq!(
        base.ids,
        vec![0, 1],
        "gene ids in the plan's first-seen order"
    );
    assert_eq!(base.rows, vec![0.0, 10.0, 1.0, 11.0]);
    assert!(matches!(base.mode, PresetMode::Freeze));
    assert_eq!(p.offsets.len(), 2);
    let unspliced = &p.offsets[0];
    assert_eq!((unspliced.track, &unspliced.ids[..]), (1, &[0u32][..]));
    assert_eq!(
        unspliced.rows,
        vec![2.0, 2.0],
        "unspliced row minus the base row"
    );
    let m6a = &p.offsets[1];
    assert_eq!((m6a.track, &m6a.ids[..]), (2, &[0u32][..]));
    assert_eq!(m6a.rows, vec![3.0, 3.0]);
    let carried = p.carried.expect("the two rows that match nothing");
    assert_eq!(
        carried.names,
        names(&["GENE3/count/spliced", "GENE2/count/unspliced"])
    );
    assert_eq!(
        carried.rows.row(0).iter().copied().collect::<Vec<_>>(),
        vec![4.0, 14.0]
    );
    assert_eq!(
        carried.rows.row(1).iter().copied().collect::<Vec<_>>(),
        vec![5.0, 15.0]
    );
}

/// A given track row whose gene has no given base row is ignored (an offset
/// is relative to the base row); under init nothing is carried; a table
/// with no base row at all is refused, naming the row it looked for.
#[test]
fn a_track_row_without_a_base_row_is_ignored_and_a_table_without_one_refused() {
    let axis = axis();
    let plan = assign_tracks(&axis).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let src = write_source(dir.path(), &["GENE2", "GENE1/count/unspliced"]);
    let p = resolve_gem_preset(Some((&src, PresetMode::Freeze)), &axis, &plan).unwrap();
    assert_eq!(p.base.unwrap().ids, vec![1]);
    assert!(p.offsets.is_empty());
    assert!(p.carried.is_none(), "both rows are on the axis");

    let src = write_source(dir.path(), &["GENE2", "GENE3"]);
    let p = resolve_gem_preset(Some((&src, PresetMode::Init)), &axis, &plan).unwrap();
    assert_eq!(p.base.unwrap().ids, vec![1]);
    assert!(p.carried.is_none(), "init carries nothing");

    let src = write_source(dir.path(), &["GENE1/count/unspliced"]);
    let e = resolve_gem_preset(Some((&src, PresetMode::Freeze)), &axis, &plan)
        .err()
        .expect("no base row")
        .to_string();
    assert!(e.contains("count/spliced"), "{e}");

    let p = resolve_gem_preset(None, &axis, &plan).unwrap();
    assert!(p.base.is_none() && p.offsets.is_empty() && p.carried.is_none());
}
