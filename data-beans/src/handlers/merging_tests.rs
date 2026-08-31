//! merge-backend's first tests: a clean merge round-trips, and a corrupted
//! input FAILS rather than shipping a silently short matrix.
//!
//! The failure test has two layers of protection to pin. The block-read loop
//! used to swallow a failed read (`.ok()` in a filter_map), dropping the block;
//! since the output nnz is precomputed from file headers, the drop left a gap
//! of fill values in the CSC arrays — corruption, not truncation. The streaming
//! writer's finalize audit now catches that as appended != declared, so the
//! run errors either way; propagating the read's own error on top of it turns
//! a confusing post-hoc audit failure into the actual cause.

use super::*;

fn write_fixture(path: &str, cols: usize, seed_val: f32) -> anyhow::Result<()> {
    let mut arr = ndarray::Array2::<f32>::zeros((3, cols));
    for r in 0..3 {
        for c in 0..cols {
            arr[(r, c)] = seed_val + (r * cols + c) as f32;
        }
    }
    let mut data = create_sparse_from_ndarray(&arr, Some(path), Some(&SparseIoBackend::Zarr))?;
    let rows: Vec<Box<str>> = (0..3).map(|r| format!("g{r}").into()).collect();
    let names: Vec<Box<str>> = (0..cols)
        .map(|c| format!("{seed_val}c{c}").into())
        .collect();
    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&names);
    Ok(())
}

fn merge_args(files: &[&str], output: &str) -> MergeBackendArgs {
    MergeBackendArgs {
        data_files: files.iter().map(|f| Box::from(*f)).collect(),
        backend: SparseIoBackend::Zarr,
        output: output.into(),
        zip: false,
        do_squeeze: false,
        row_nnz_cutoff: 0,
        column_nnz_cutoff: 0,
        block_size: None,
    }
}

#[test]
fn a_clean_merge_round_trips() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let a = dir.path().join("a.zarr");
    let b = dir.path().join("b.zarr");
    write_fixture(a.to_str().expect("utf8"), 4, 100.0)?;
    write_fixture(b.to_str().expect("utf8"), 3, 900.0)?;
    let out = dir.path().join("m");
    run_merge_backend(&merge_args(
        &[a.to_str().expect("utf8"), b.to_str().expect("utf8")],
        out.to_str().expect("utf8"),
    ))?;
    let merged = open_sparse_matrix(
        &format!("{}.zarr", out.to_str().expect("utf8")),
        &SparseIoBackend::Zarr,
    )?;
    assert_eq!(merged.num_rows(), Some(3));
    assert_eq!(merged.num_columns(), Some(7));
    assert_eq!(merged.num_non_zeros(), Some(21));
    Ok(())
}

#[test]
fn a_corrupted_input_fails_instead_of_shipping_a_short_matrix() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let a = dir.path().join("a.zarr");
    let b = dir.path().join("b.zarr");
    write_fixture(a.to_str().expect("utf8"), 4, 100.0)?;
    write_fixture(b.to_str().expect("utf8"), 3, 900.0)?;

    // Garbage over one data chunk of the second input: the read of that block
    // fails to decode, which used to be swallowed.
    let mut corrupted = false;
    for entry in walk(&b.join("by_column").join("data"))? {
        if entry.file_name().is_some_and(|n| n != "zarr.json") && entry.is_file() {
            std::fs::write(&entry, b"not a chunk")?;
            corrupted = true;
            break;
        }
    }
    assert!(
        corrupted,
        "fixture layout changed; no chunk found to corrupt"
    );

    let out = dir.path().join("m");
    let result = run_merge_backend(&merge_args(
        &[a.to_str().expect("utf8"), b.to_str().expect("utf8")],
        out.to_str().expect("utf8"),
    ));
    assert!(
        result.is_err(),
        "a merge that cannot read an input must fail, not write fewer entries \
         than its own header declares"
    );
    Ok(())
}

fn walk(root: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    if root.is_dir() {
        for e in std::fs::read_dir(root)? {
            let p = e?.path();
            if p.is_dir() {
                out.extend(walk(&p)?);
            } else {
                out.push(p);
            }
        }
    }
    Ok(out)
}
