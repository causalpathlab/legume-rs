use super::*;
use crate::parquet::{write_named_table, Column};

/// Write `content` to a scratch file named `name` inside `dir`.
fn scratch(dir: &tempfile::TempDir, name: &str, content: &str) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, content).expect("write scratch file");
    path.to_string_lossy().into_owned()
}

fn names(v: &[Box<str>]) -> Vec<&str> {
    v.iter().map(std::convert::AsRef::as_ref).collect()
}

#[test]
fn plain_txt_one_name_per_line() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "CD8A\nMS4A1\nLYZ\n");
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1", "LYZ"]);
}

/// The headline case: a curated `gene<TAB>celltype` marker table passed as-is.
/// The celltype column must be ignored, not treated as a second gene.
#[test]
fn tsv_gene_celltype_keeps_only_the_gene_column() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(
        &dir,
        "markers.tsv",
        "gene\tcelltype\nCD8A\tT cell\nMS4A1\tB cell\nLYZ\tMonocyte\n",
    );
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1", "LYZ"]);
}

#[test]
fn csv_without_header_falls_back_to_first_column() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "markers.csv", "CD8A,T cell\nMS4A1,B cell\n");
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1"]);
}

/// A gene-like header anywhere picks that column, not column 0.
#[test]
fn header_selects_the_named_column() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(
        &dir,
        "markers.tsv",
        "celltype\tsymbol\nT cell\tCD8A\nB cell\tMS4A1\n",
    );
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1"]);
}

/// Without a recognizable header the first row is data, not a header — losing it
/// would silently drop a gene.
#[test]
fn unrecognized_header_is_treated_as_data() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "CD8A\nMS4A1\n");
    assert_eq!(read_name_list(&f).unwrap().len(), 2);
}

#[test]
fn duplicates_and_blank_lines_are_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "CD8A\n\nMS4A1\nCD8A\n   \nLYZ\n");
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1", "LYZ"]);
}

/// Ragged leading/repeated whitespace must not shift the column.
#[test]
fn whitespace_separated_with_ragged_indent() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "  CD8A   T cell\nMS4A1  B cell\n");
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1"]);
}

#[test]
fn quoted_csv_fields_are_unquoted() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(
        &dir,
        "genes.csv",
        "\"gene\",\"celltype\"\n\"CD8A\",\"T cell\"\n",
    );
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A"]);
}

#[test]
fn empty_file_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "\n\n");
    assert!(read_name_list(&f).is_err());
}

/// The parquet path: an all-string table, which the numeric `ParquetReader`
/// cannot read at all.
#[test]
fn parquet_gene_celltype_keeps_only_the_gene_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("markers.parquet");
    let path = path.to_string_lossy().into_owned();

    let genes: Vec<Box<str>> = vec!["CD8A".into(), "MS4A1".into(), "LYZ".into()];
    let celltypes: Vec<Box<str>> = vec!["T cell".into(), "B cell".into(), "Monocyte".into()];
    write_named_table(
        &path,
        "gene",
        &genes,
        &[("celltype".into(), Column::Str(&celltypes))],
    )
    .unwrap();

    let got = read_name_list(&path).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1", "LYZ"]);
}
