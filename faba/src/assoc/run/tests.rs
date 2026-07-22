//! Round-trip tests for the tidy report schema — these read the parquet back off disk,
//! because the failure mode being guarded is a *silently* wrong file: `write_named_table`
//! pairs a column-name list with a value list positionally, so a mismatch does not fail to
//! compile, it writes a table whose headers do not describe its contents.

use super::*;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::parquet::{peek_parquet_field_names, read_parquet_string_column};
use matrix_util::traits::IoOps;

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("faba_assoc_schema_{tag}"));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn two_sites() -> Vec<Site> {
    vec![
        Site {
            gene: "GENE1".into(),
            subunit: "chr1:100".into(),
            k: vec![],
            n: vec![],
        },
        Site {
            gene: "GENE2".into(),
            subunit: "chr2:200".into(),
            k: vec![],
            n: vec![],
        },
    ]
}

#[test]
fn branch_level_writes_an_integer_branch_column() {
    let dir = tmpdir("branch");
    let path = dir.join("t.parquet");
    let p = path.to_str().unwrap();
    let sites = two_sites();
    let level = ReportLevel {
        names: None,
        drop_group: None,
        tag: "branch",
        unit: "branch",
        strict: true,
    };

    write_report_table(
        p,
        &sites,
        &level,
        &[(0, 3), (1, 7)],
        vec![("lfsr", Val::F32(vec![0.01, 0.5]))],
    )
    .unwrap();

    let fields = peek_parquet_field_names(p).unwrap();
    let fields: Vec<&str> = fields.iter().map(AsRef::as_ref).collect();
    assert_eq!(fields, vec!["site", "gene", "subunit", "branch", "lfsr"]);

    // The identity is broken out, not `/`-joined into the key.
    assert_eq!(
        read_parquet_string_column(p, 1).unwrap(),
        vec!["GENE1".into(), "GENE2".into()] as Vec<Box<str>>
    );
    assert_eq!(
        read_parquet_string_column(p, 2).unwrap(),
        vec!["chr1:100".into(), "chr2:200".into()] as Vec<Box<str>>
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn celltype_level_has_no_branch_column_and_keeps_slashes() {
    let dir = tmpdir("celltype");
    let path = dir.join("t.parquet");
    let p = path.to_str().unwrap();
    let sites = two_sites();
    // A Cell-Ontology label with a `/` in it — the old `/`-joined row key forced this to be
    // rewritten as `-`, silently corrupting the name. In its own column it survives.
    let names: Vec<Box<str>> = vec!["B cell/plasma".into(), "CD8+ T".into()];
    let level = ReportLevel {
        names: Some(&names),
        drop_group: None,
        tag: "celltype",
        unit: "cell-type",
        strict: false,
    };

    write_report_table(
        p,
        &sites,
        &level,
        &[(0, 0), (1, 1)],
        vec![("lfsr", Val::F32(vec![0.02, 0.4]))],
    )
    .unwrap();

    let fields = peek_parquet_field_names(p).unwrap();
    let fields: Vec<&str> = fields.iter().map(AsRef::as_ref).collect();
    assert_eq!(
        fields,
        vec!["site", "gene", "subunit", "cell_type", "lfsr"],
        "a cell-type aggregate pools cells ACROSS branches — it must not carry a branch column"
    );
    assert!(!fields.contains(&"branch"));

    assert_eq!(
        read_parquet_string_column(p, 3).unwrap(),
        vec!["B cell/plasma".into(), "CD8+ T".into()] as Vec<Box<str>>,
        "the cell-type name is written verbatim, `/` and all"
    );

    // The in-house f32 matrix reader must still cope with the string columns now sitting
    // between the key and the values — it is what any downstream faba/matrix-util consumer
    // will point at these files.
    let m = DMatrix::<f32>::from_parquet(p).unwrap();
    assert_eq!(m.mat.nrows(), 2);
    let cols: Vec<&str> = m.cols.iter().map(AsRef::as_ref).collect();
    assert!(
        cols.contains(&"lfsr"),
        "numeric columns still readable, got {cols:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
