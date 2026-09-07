use super::*;
use crate::test_support::{scratch, write_gz_members};

/// A well-formed coordinate-format body: `nnz` entries, one per line.
fn mtx_text(nrow: usize, ncol: usize, triplets: &[(u64, u64, f32)]) -> String {
    let mut s = String::from("%%MatrixMarket matrix coordinate real general\n");
    s.push_str(&format!("{}\t{}\t{}\n", nrow, ncol, triplets.len()));
    for &(r, c, v) in triplets {
        s.push_str(&format!("{}\t{}\t{}\n", r + 1, c + 1, v));
    }
    s
}

fn sample() -> Vec<(u64, u64, f32)> {
    vec![
        (0, 0, 1.0),
        (3, 0, 2.0),
        (1, 1, 3.0),
        (4, 2, 4.0),
        (2, 3, 5.0),
        (0, 3, 6.0),
    ]
}

#[test]
fn triplets_come_back_in_file_order_with_the_shape() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "m.mtx", &mtx_text(5, 4, &sample()));

    let (got, shape) = read_mtx_triplets(&f).unwrap();
    assert_eq!(shape, (5, 4, 6));
    assert_eq!(got, sample());
}

/// Plain `gzip`, several concatenated members, and BGZF-framed members all go
/// through the same reader. A decoder that stops at the end of the first
/// member drops every later one with no error and no non-zero exit code.
#[test]
fn every_gzip_framing_keeps_every_triplet() {
    for (members, bgzf_extra) in [(1, false), (3, false), (3, true)] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.mtx.gz");
        write_gz_members(&path, &mtx_text(5, 4, &sample()), members, bgzf_extra);

        let (got, shape) = read_mtx_triplets(path.to_str().unwrap()).unwrap();
        assert_eq!(shape, (5, 4, 6));
        assert_eq!(got, sample(), "members={members} bgzf_extra={bgzf_extra}");
    }
}

/// The guard that turns a silent truncation into a build failure.
#[test]
fn a_body_shorter_than_the_declared_nnz_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut text = mtx_text(5, 4, &sample());
    let cut = text.find("2\t2\t3").expect("third entry");
    text.truncate(cut);
    let f = scratch(&dir, "short.mtx", &text);

    let err = read_mtx_triplets(&f).expect_err("a short body must not parse");
    let msg = err.to_string();
    assert!(
        msg.contains('6'),
        "error should name the declared count: {msg}"
    );
}

#[test]
fn a_body_longer_than_the_declared_nnz_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let text = mtx_text(5, 4, &sample()) + "5\t4\t7.0\n";
    let f = scratch(&dir, "long.mtx", &text);

    read_mtx_triplets(&f).expect_err("a long body must not parse");
}

/// A line that is not a triplet used to be dropped on the floor, which is the
/// same silent-loss failure mode as the truncated read.
#[test]
fn a_malformed_triplet_line_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let text = mtx_text(5, 4, &sample()).replace("2\t2\t3", "2\tx\t3");
    let f = scratch(&dir, "bad.mtx", &text);

    read_mtx_triplets(&f).expect_err("a malformed line must not parse");
}

#[test]
fn an_entry_outside_the_declared_shape_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    for bad in ["0\t1\t1", "6\t1\t1", "1\t5\t1"] {
        let text = mtx_text(5, 4, &sample()).replace("1\t1\t1", bad);
        let f = scratch(&dir, "oob.mtx", &text);
        read_mtx_triplets(&f).expect_err(bad);
    }
}

/// Comment and blank lines inside the body stay skippable.
#[test]
fn comments_and_blank_lines_in_the_body_are_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let text = mtx_text(5, 4, &sample()).replace("2\t2\t3", "% a note\n\n2\t2\t3");
    let f = scratch(&dir, "cmt.mtx", &text);

    let (got, _) = read_mtx_triplets(&f).unwrap();
    assert_eq!(got, sample());
}
