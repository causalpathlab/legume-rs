//! Round-trips for the subsample write path — its first tests, written against
//! the materialising implementation to pin behaviour before the streaming
//! rewrite. What must survive: which cells and genes are kept (seeded, so the
//! selection is reproducible), the ascending renumbering, names, and values.

use super::*;

fn fixture_at(path: &str) -> anyhow::Result<()> {
    // 4 genes x 6 cells, all values distinct so aliasing cannot hide.
    let mut arr = ndarray::Array2::<f32>::zeros((4, 6));
    for r in 0..4 {
        for c in 0..6 {
            arr[(r, c)] = (r * 10 + c + 1) as f32;
        }
    }
    let mut data = create_sparse_from_ndarray(&arr, Some(path), Some(&SparseIoBackend::Zarr))?;
    let rows: Vec<Box<str>> = (0..4).map(|r| format!("g{r}").into()).collect();
    let cols: Vec<Box<str>> = (0..6).map(|c| format!("cell{c}").into()).collect();
    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);
    Ok(())
}

fn args(input: &str, output: &str) -> SubsampleArgs {
    SubsampleArgs {
        data_file: input.into(),
        cells: None,
        cell_frac: None,
        genes: None,
        gene_frac: None,
        seed: 11,
        backend: SparseIoBackend::Zarr,
        output: output.into(),
        zip: false,
    }
}

#[test]
fn a_cell_subsample_keeps_whole_columns_untouched() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("in.zarr");
    fixture_at(input.to_str().expect("utf8"))?;
    let output = dir.path().join("out");
    let mut a = args(
        input.to_str().expect("utf8"),
        output.to_str().expect("utf8"),
    );
    a.cells = Some(3);
    run_subsample(&a)?;

    let out = open_sparse_matrix(
        &format!("{}.zarr", output.to_str().expect("utf8")),
        &SparseIoBackend::Zarr,
    )?;
    assert_eq!(out.num_rows(), Some(4), "no gene filter: full row axis");
    assert_eq!(out.num_columns(), Some(3));
    // Every kept column must be the original column for its name, bit for bit.
    let src = open_sparse_matrix(input.to_str().expect("utf8"), &SparseIoBackend::Zarr)?;
    let src_names = src.column_names()?;
    let src_dense = src.read_columns_dmatrix((0..6).collect())?;
    let out_dense = out.read_columns_dmatrix((0..3).collect())?;
    for (k, name) in out.column_names()?.iter().enumerate() {
        let orig = src_names
            .iter()
            .position(|n| n == name)
            .expect("kept cell exists in the source");
        for r in 0..4 {
            assert_eq!(
                out_dense[(r, k)],
                src_dense[(r, orig)],
                "cell {name} row {r}"
            );
        }
    }
    Ok(())
}

#[test]
fn a_gene_and_cell_subsample_renumbers_ascending() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("in.zarr");
    fixture_at(input.to_str().expect("utf8"))?;
    let output = dir.path().join("out");
    let mut a = args(
        input.to_str().expect("utf8"),
        output.to_str().expect("utf8"),
    );
    a.cells = Some(4);
    a.genes = Some(2);
    run_subsample(&a)?;

    let out = open_sparse_matrix(
        &format!("{}.zarr", output.to_str().expect("utf8")),
        &SparseIoBackend::Zarr,
    )?;
    assert_eq!(out.num_rows(), Some(2));
    assert_eq!(out.num_columns(), Some(4));
    // The kept gene names identify the original rows; values must follow them.
    let src = open_sparse_matrix(input.to_str().expect("utf8"), &SparseIoBackend::Zarr)?;
    let src_dense = src.read_columns_dmatrix((0..6).collect())?;
    let src_rows = src.row_names()?;
    let src_cols = src.column_names()?;
    let out_dense = out.read_columns_dmatrix((0..4).collect())?;
    for (rk, rname) in out.row_names()?.iter().enumerate() {
        let ro = src_rows
            .iter()
            .position(|n| n == rname)
            .expect("gene exists");
        for (ck, cname) in out.column_names()?.iter().enumerate() {
            let co = src_cols
                .iter()
                .position(|n| n == cname)
                .expect("cell exists");
            assert_eq!(out_dense[(rk, ck)], src_dense[(ro, co)], "{rname}/{cname}");
        }
    }
    // Ascending selection: kept names appear in their original relative order.
    let kept: Vec<usize> = out
        .row_names()?
        .iter()
        .map(|n| src_rows.iter().position(|m| m == n).expect("gene"))
        .collect();
    assert!(
        kept.windows(2).all(|w| w[0] < w[1]),
        "gene order preserved: {kept:?}"
    );
    Ok(())
}
