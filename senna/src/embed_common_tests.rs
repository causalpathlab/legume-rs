//! Tests for the latent-sharpness summary.

use super::{latent_sharpness, Mat};
use matrix_util::traits::IoOps;

#[test]
fn a_flat_latent_has_k_effective_topics() {
    let k = 8;
    let theta = Mat::from_element(10, k, 1.0 / k as f32);
    let (eff, mx) = latent_sharpness(&theta);
    assert!((eff - k as f32).abs() < 1e-3, "effective topics {eff}");
    assert!((mx - 1.0 / k as f32).abs() < 1e-6, "max weight {mx}");
}

#[test]
fn a_one_hot_latent_has_one_effective_topic() {
    let k = 8;
    let mut theta = Mat::zeros(10, k);
    for i in 0..10 {
        theta[(i, i % k)] = 1.0;
    }
    let (eff, mx) = latent_sharpness(&theta);
    assert!((eff - 1.0).abs() < 1e-6, "effective topics {eff}");
    assert!((mx - 1.0).abs() < 1e-6, "max weight {mx}");
}

/// Rows are renormalized, so unnormalized weights and proportions agree.
#[test]
fn rows_are_renormalized_before_the_entropy() {
    let theta = Mat::from_row_slice(2, 3, &[0.2, 0.3, 0.5, 0.2, 0.3, 0.5]);
    let scaled = &theta * 7.0;
    assert_eq!(latent_sharpness(&theta), latent_sharpness(&scaled));
}

#[test]
fn an_empty_latent_is_nan_not_zero() {
    let (eff, mx) = latent_sharpness(&Mat::zeros(0, 4));
    assert!(eff.is_nan() && mx.is_nan());
}

/// A diverged latent must report NaN, not `+inf`. `f32::max` drops a NaN operand,
/// so the guarded row sum has to check finiteness explicitly.
#[test]
fn a_non_finite_latent_is_nan_not_infinity() {
    let mut theta = Mat::from_element(3, 4, 0.25);
    theta[(1, 2)] = f32::NAN;
    let (eff, mx) = latent_sharpness(&theta);
    assert!(eff.is_nan(), "effective topics was {eff}");
    assert!(mx.is_nan(), "mean max theta was {mx}");
}

#[test]
fn zero_rows_survive_l2_normalization_untouched() {
    let mut m = Mat::from_row_slice(2, 2, &[3.0, 4.0, 0.0, 0.0]);
    super::l2_normalize_rows_inplace(&mut m);
    assert!((m.row(0).norm() - 1.0).abs() < 1e-6);
    assert!(m.row(1).iter().all(|&x| x == 0.0));
}

//////////////////////////////////////////////////////
// Dense bulk tables: header detection + orientation //
//////////////////////////////////////////////////////

fn bulk_fixture() -> (Vec<Box<str>>, Vec<Box<str>>, Mat) {
    let genes: Vec<Box<str>> = ["g0", "g1", "g2"].iter().map(|s| Box::from(*s)).collect();
    let samples: Vec<Box<str>> = ["s0", "s1"].iter().map(|s| Box::from(*s)).collect();
    let mat = Mat::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    (genes, samples, mat)
}

fn write_tsv(path: &std::path::Path, header: bool) {
    let (genes, samples, mat) = bulk_fixture();
    let mut s = String::new();
    if header {
        s.push_str(&format!("gene\t{}\n", samples.join("\t")));
    }
    for (i, g) in genes.iter().enumerate() {
        let vals: Vec<String> = mat.row(i).iter().map(|v| format!("{v}")).collect();
        s.push_str(&format!("{g}\t{}\n", vals.join("\t")));
    }
    std::fs::write(path, s).expect("write tsv");
}

/// Regression: `read_mat` reads text with no header, so a header line was parsed
/// as counts and the `.expect` in the parser panicked. A headered TSV is the
/// ordinary case for a bulk table and has to carry its sample names through.
#[test]
fn a_headered_tsv_keeps_its_sample_names() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk.tsv");
    write_tsv(&path, true);
    let (genes, samples, mat) = bulk_fixture();
    let got =
        super::read_labeled_mat(path.to_str().unwrap(), super::HeaderArg::Auto).expect("read");
    assert_eq!(got.rows, genes);
    assert_eq!(got.cols, samples);
    assert_eq!(got.mat, mat);
}

#[test]
fn a_headerless_tsv_still_reads_its_genes_and_values() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk.tsv");
    write_tsv(&path, false);
    let (genes, _, mat) = bulk_fixture();
    let got =
        super::read_labeled_mat(path.to_str().unwrap(), super::HeaderArg::Auto).expect("read");
    assert_eq!(got.rows, genes);
    assert_eq!(got.mat, mat);
    assert_eq!(
        got.cols.len(),
        2,
        "one label per sample even without a header"
    );
    assert_ne!(got.cols[0], got.cols[1]);
}

#[test]
fn parquet_and_headered_tsv_read_the_same_table() {
    let dir = tempfile::tempdir().expect("tmp");
    let tsv = dir.path().join("bulk.tsv");
    let pq = dir.path().join("bulk.parquet");
    write_tsv(&tsv, true);
    let (genes, samples, mat) = bulk_fixture();
    mat.to_parquet_with_names(
        pq.to_str().unwrap(),
        (Some(&genes), Some("gene")),
        Some(&samples),
    )
    .expect("write parquet");
    let a = super::read_labeled_mat(tsv.to_str().unwrap(), super::HeaderArg::Auto).expect("tsv");
    let b = super::read_labeled_mat(pq.to_str().unwrap(), super::HeaderArg::Auto).expect("parquet");
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    assert_eq!(a.mat, b.mat);
}

fn labels(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

fn model_genes() -> Vec<Box<str>> {
    labels(&["TGFB1", "CD8A", "LYZ", "MS4A1", "GNLY"])
}

#[test]
fn genes_on_rows_is_genes_by_samples() {
    let rows = labels(&["TGFB1", "CD8A", "LYZ", "NOTINMODEL"]);
    let cols = labels(&["s0", "s1"]);
    let o = super::resolve_orientation(&rows, &cols, &model_genes(), None).expect("resolve");
    assert_eq!(o, super::Orientation::GenesBySamples);
}

#[test]
fn genes_on_columns_is_samples_by_genes() {
    let rows = labels(&["s0", "s1"]);
    let cols = labels(&["TGFB1", "CD8A", "LYZ", "NOTINMODEL"]);
    let o = super::resolve_orientation(&rows, &cols, &model_genes(), None).expect("resolve");
    assert_eq!(o, super::Orientation::SamplesByGenes);
}

/// The bridge is the shared canonicalizer, not string equality: an
/// `ENSG…_SYMBOL` axis must be recognised as genes against a bare-symbol model,
/// on either axis.
#[test]
fn orientation_sees_through_the_naming_convention() {
    let ensg = labels(&[
        "ENSG00000105329_TGFB1",
        "ENSG00000153563_CD8A",
        "ENSG00000090382_LYZ",
    ]);
    let samples = labels(&["s0", "s1"]);
    let m = model_genes();
    assert_eq!(
        super::resolve_orientation(&ensg, &samples, &m, None).unwrap(),
        super::Orientation::GenesBySamples
    );
    assert_eq!(
        super::resolve_orientation(&samples, &ensg, &m, None).unwrap(),
        super::Orientation::SamplesByGenes
    );
}

/// Nothing matching is a naming failure, and the error has to show what the
/// file holds so the user can see why, not just that it failed.
#[test]
fn no_axis_matching_the_model_shows_both_axes_and_the_model() {
    let rows = labels(&["r0", "r1"]);
    let cols = labels(&["c0", "c1"]);
    let err = super::resolve_orientation(&rows, &cols, &model_genes(), None)
        .expect_err("must fail")
        .to_string();
    assert!(err.contains("r0"), "{err}");
    assert!(err.contains("c0"), "{err}");
    assert!(err.contains("TGFB1"), "{err}");
}

/// Sample IDs that are themselves gene symbols make both axes score; that is
/// an ambiguity to report, never a coin flip to resolve.
#[test]
fn both_axes_matching_is_an_error_naming_the_override() {
    let rows = labels(&["TGFB1", "CD8A", "LYZ"]);
    let cols = labels(&["MS4A1", "GNLY"]);
    let err = super::resolve_orientation(&rows, &cols, &model_genes(), None)
        .expect_err("must fail")
        .to_string();
    assert!(err.contains("--bulk-orientation"), "{err}");
}

#[test]
fn a_forced_orientation_wins_over_the_evidence() {
    let rows = labels(&["TGFB1", "CD8A", "LYZ"]);
    let cols = labels(&["s0", "s1"]);
    let o = super::resolve_orientation(
        &rows,
        &cols,
        &model_genes(),
        Some(super::Orientation::SamplesByGenes),
    )
    .expect("forced never errors on evidence");
    assert_eq!(o, super::Orientation::SamplesByGenes);
}

/// `oriented` is the only place a table is ever turned; genes end on rows.
#[test]
fn oriented_transposes_a_samples_by_genes_table_once() {
    let (genes, samples, mat) = bulk_fixture();
    let t = matrix_util::traits::MatWithNames {
        rows: samples.clone(),
        cols: genes.clone(),
        mat: mat.transpose(),
    };
    let got = super::oriented(t, super::Orientation::SamplesByGenes);
    assert_eq!(got.rows, genes);
    assert_eq!(got.cols, samples);
    assert_eq!(got.mat, mat);
}

////////////////////////////////////////////
// read_bulk_data_aligned (deconvolve)     //
////////////////////////////////////////////

fn write_labeled_tsv(path: &std::path::Path, rows: &[Box<str>], cols: &[Box<str>], mat: &Mat) {
    let mut s = format!("id\t{}\n", cols.join("\t"));
    for (i, r) in rows.iter().enumerate() {
        let vals: Vec<String> = mat.row(i).iter().map(|v| format!("{v}")).collect();
        s.push_str(&format!("{r}\t{}\n", vals.join("\t")));
    }
    std::fs::write(path, s).expect("write tsv");
}

/// The deconvolve entry point on an ordinary headered TSV: the sample names
/// come through and the counts land on the reference rows.
#[test]
fn a_headered_bulk_tsv_aligns_to_the_reference() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk.tsv");
    let genes = labels(&["TGFB1", "CD8A"]);
    let samples = labels(&["s0", "s1"]);
    let mat = Mat::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    write_labeled_tsv(&path, &genes, &samples, &mat);
    let reference = labels(&["CD8A", "TGFB1", "LYZ"]);
    let out = super::read_bulk_data_aligned(
        &[path.to_str().unwrap().into()],
        &reference,
        &super::BulkTableOpts::default(),
    )
    .expect("aligned");
    assert_eq!(out.genes, reference);
    assert_eq!(out.samples, samples);
    // CD8A is reference row 0, TGFB1 row 1, LYZ row 2 (unobserved → 0).
    assert_eq!(
        out.data,
        Mat::from_row_slice(3, 2, &[3.0, 4.0, 1.0, 2.0, 0.0, 0.0])
    );
}

/// A samples × genes file is turned, not refused: the data was fine.
#[test]
fn a_transposed_bulk_file_aligns_to_the_reference() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk_t.tsv");
    let genes = labels(&["TGFB1", "CD8A"]);
    let samples = labels(&["s0", "s1"]);
    let mat = Mat::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]); // genes × samples
    write_labeled_tsv(&path, &samples, &genes, &mat.transpose());
    let reference = labels(&["CD8A", "TGFB1", "LYZ"]);
    let out = super::read_bulk_data_aligned(
        &[path.to_str().unwrap().into()],
        &reference,
        &super::BulkTableOpts::default(),
    )
    .expect("aligned after transposing");
    assert_eq!(out.samples, samples);
    assert_eq!(
        out.data,
        Mat::from_row_slice(3, 2, &[3.0, 4.0, 1.0, 2.0, 0.0, 0.0])
    );
}

/////////////////////////////////////////
// The header blind spot and overrides   //
/////////////////////////////////////////

/// Sample IDs that are all numbers are indistinguishable BY TYPE from a count
/// row, so `auto` reads them as data; the documented way out is `--bulk-header
/// yes`, which has to take that line as the names.
#[test]
fn a_header_of_numeric_sample_ids_is_data_under_auto_and_names_under_yes() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("ids.tsv");
    std::fs::write(&path, "gene\t2019\t2020\ng0\t1\t2\n").unwrap();
    let p = path.to_str().unwrap();
    let auto = super::read_labeled_mat(p, super::HeaderArg::Auto).expect("auto");
    assert_eq!(
        auto.rows,
        labels(&["gene", "g0"]),
        "auto cannot see this header"
    );
    let yes = super::read_labeled_mat(p, super::HeaderArg::Yes).expect("yes");
    assert_eq!(yes.rows, labels(&["g0"]));
    assert_eq!(yes.cols, labels(&["2019", "2020"]));
}

/// `--bulk-header no` on a file that plainly has one must be an error, not the
/// parser's panic on `s0`.
#[test]
fn forcing_no_header_on_a_headered_file_is_an_error_not_a_panic() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("bulk.tsv");
    write_tsv(&path, true);
    let err = match super::read_labeled_mat(path.to_str().unwrap(), super::HeaderArg::No) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("must refuse"),
    };
    assert!(err.contains("--bulk-header"), "{err}");
    assert!(err.contains("s0"), "the offending fields are shown: {err}");
}

/// The parquet name column is found by type, not by position.
#[test]
fn a_parquet_whose_name_column_is_not_first_still_reads_genes() {
    use matrix_util::parquet::{write_table, Column};
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("t.parquet");
    let p = path.to_str().unwrap();
    let genes = labels(&["g0", "g1"]);
    write_table(
        p,
        &[
            ("s0".into(), Column::F32(&[1.0, 2.0])),
            ("gene".into(), Column::Str(&genes)),
            ("s1".into(), Column::F32(&[3.0, 4.0])),
        ],
    )
    .unwrap();
    let got = super::read_labeled_mat(p, super::HeaderArg::Auto).expect("read");
    assert_eq!(got.rows, genes);
    assert_eq!(got.cols, labels(&["s0", "s1"]));
    assert_eq!(got.mat, Mat::from_row_slice(2, 2, &[1.0, 3.0, 2.0, 4.0]));
}

/// A parquet written without its index has no names to align on. Say so;
/// do not stringify the first sample's counts into "gene names".
#[test]
fn a_parquet_with_no_string_column_is_refused() {
    use matrix_util::parquet::{write_table, Column};
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("nonames.parquet");
    let p = path.to_str().unwrap();
    write_table(
        p,
        &[
            ("s0".into(), Column::F32(&[1.0, 2.0])),
            ("s1".into(), Column::F32(&[3.0, 4.0])),
        ],
    )
    .unwrap();
    let err = match super::read_labeled_mat(p, super::HeaderArg::Auto) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("must refuse"),
    };
    assert!(err.contains("name column"), "{err}");
}
