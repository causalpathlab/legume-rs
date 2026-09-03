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

////////////////////////////////////
// First-line header detection     //
////////////////////////////////////

fn scratch_gz(dir: &tempfile::TempDir, name: &str, content: &str) -> String {
    use std::io::Write;
    let path = dir.path().join(name);
    let f = std::fs::File::create(&path).expect("create gz");
    let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::default());
    enc.write_all(content.as_bytes()).expect("write gz");
    enc.finish().expect("finish gz");
    path.to_string_lossy().into_owned()
}

#[test]
fn a_non_numeric_first_line_is_a_header() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.tsv", "gene\ts0\ts1\ng0\t1\t2\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), Some(0));
}

#[test]
fn an_all_numeric_first_line_is_data() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.tsv", "g0\t1\t2\ng1\t3\t4\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), None);
}

/// A fully quoted numeric field still reads as numeric, or a quoted headerless
/// file loses its first data row to a phantom header.
#[test]
fn quoted_numbers_are_still_numbers() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.csv", "\"g0\",\"1.5\",\"2\"\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), None);
}

/// Delimited tables are routinely gzipped; detection has to read through that.
#[test]
fn header_detection_reads_through_gzip() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch_gz(&dir, "t.tsv.gz", "gene\ts0\ts1\ng0\t1\t2\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), Some(0));
}

#[test]
fn first_line_fields_are_unquoted_and_trimmed() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.csv", "\"gene\", s0 ,s1\r\ng0,1,2\n");
    let got = first_line_fields(&f, &['\t', ',']).unwrap();
    assert_eq!(names(&got), ["gene", "s0", "s1"]);
}

/// R writes missing values as `NA`, which is not a number to `f64::from_str`;
/// a headerless file with one in its first row must not lose that row to a
/// phantom header.
#[test]
fn r_missing_values_do_not_make_a_data_row_a_header() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.tsv", "g0\tNA\t2\ng1\t3\tN/A\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), None);
}
