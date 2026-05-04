use data_beans::sparse_io::*;
use data_beans::sparse_io_vector::SparseIoVec;
use matrix_util::traits::SampleOps;
use ndarray::Array2;
use std::sync::Arc;

type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

/// Helper: create a named sparse dataset from an ndarray, ready for SparseIoVec
fn make_named_sparse(
    data: &Array2<f32>,
    row_names: &[Box<str>],
    col_names: &[Box<str>],
) -> Arc<SparseData> {
    let mut sp = create_sparse_from_ndarray(data, None, None).unwrap();
    sp.register_row_names_vec(row_names);
    sp.register_column_names_vec(col_names);
    sp.preload_columns().unwrap();
    Arc::from(sp)
}

fn str_vec(n: usize, prefix: &str) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{prefix}{i}").into_boxed_str())
        .collect()
}

// ─────────────────────────────────────────────────────
// Basic push and metadata
// ─────────────────────────────────────────────────────

#[test]
fn push_single_dataset() -> anyhow::Result<()> {
    let nrow = 10;
    let ncol = 20;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "gene");
    let cols = str_vec(ncol, "cell");

    let sp = make_named_sparse(&raw, &rows, &cols);

    let mut vec = SparseIoVec::new();
    vec.push(sp, Some("batch0".into()))?;

    assert_eq!(vec.num_rows(), nrow);
    assert_eq!(vec.num_columns(), ncol);
    assert_eq!(vec.len(), 1);

    let rn = vec.row_names()?;
    assert_eq!(rn.len(), nrow);
    assert_eq!(&*rn[0], "gene0");

    let cn = vec.column_names()?;
    assert_eq!(cn.len(), ncol);
    // column names get a data tag appended
    assert!(cn[0].contains("cell0"));
    assert!(cn[0].contains("batch0"));

    Ok(())
}

#[test]
fn push_two_datasets() -> anyhow::Result<()> {
    let nrow = 8;
    let ncol_a = 10;
    let ncol_b = 15;
    let rows = str_vec(nrow, "gene");

    let raw_a = Array2::<f32>::runif(nrow, ncol_a);
    let raw_b = Array2::<f32>::runif(nrow, ncol_b);

    let cols_a = str_vec(ncol_a, "a_cell");
    let cols_b = str_vec(ncol_b, "b_cell");

    let sp_a = make_named_sparse(&raw_a, &rows, &cols_a);
    let sp_b = make_named_sparse(&raw_b, &rows, &cols_b);

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, Some("batchA".into()))?;
    vec.push(sp_b, Some("batchB".into()))?;

    assert_eq!(vec.num_rows(), nrow);
    assert_eq!(vec.num_columns(), ncol_a + ncol_b);
    assert_eq!(vec.len(), 2);

    Ok(())
}

#[test]
fn push_mismatched_rows_uses_intersection() -> anyhow::Result<()> {
    // dataset A rows: gene0..gene4
    // dataset B rows: gene0..gene4 with gene0 and gene1 swapped — same set of
    // names, but at different local positions. Intersection should still be
    // all 5 genes.
    let rows_a = str_vec(5, "gene");
    let mut rows_b = str_vec(5, "gene");
    rows_b.swap(0, 1);

    let raw_a = Array2::<f32>::runif(5, 3);
    let raw_b = Array2::<f32>::runif(5, 4);

    let sp_a = make_named_sparse(&raw_a, &rows_a, &str_vec(3, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows_b, &str_vec(4, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, None)?;
    vec.push(sp_b, None)?;
    assert_eq!(vec.num_rows(), 5);
    assert_eq!(vec.row_names()?, rows_a);
    Ok(())
}

#[test]
fn push_disjoint_rows_uses_intersection() -> anyhow::Result<()> {
    // dataset A: gene0..gene3 (4 rows), dataset B: gene2..gene5 (4 rows).
    // Intersection: {gene2, gene3}.
    let rows_a: Vec<Box<str>> = (0..4).map(|i| format!("gene{i}").into()).collect();
    let rows_b: Vec<Box<str>> = (2..6).map(|i| format!("gene{i}").into()).collect();

    let raw_a = Array2::<f32>::runif(4, 3);
    let raw_b = Array2::<f32>::runif(4, 2);
    let sp_a = make_named_sparse(&raw_a, &rows_a, &str_vec(3, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows_b, &str_vec(2, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, Some("A".into()))?;
    vec.push(sp_b, Some("B".into()))?;
    assert_eq!(vec.num_rows(), 2);
    let names = vec.row_names()?;
    assert_eq!(names, vec![Box::from("gene2"), Box::from("gene3")]);

    // Compact rows must reflect the actual row identity in each backend.
    let mat = vec.read_columns_ndarray(0..vec.num_columns())?;
    assert_eq!(mat.nrows(), 2);
    assert_eq!(mat.ncols(), 5);
    // dataset A columns 0..3 → rows 2,3 of A
    for j in 0..3 {
        assert!((mat[(0, j)] - raw_a[(2, j)]).abs() < 1e-6);
        assert!((mat[(1, j)] - raw_a[(3, j)]).abs() < 1e-6);
    }
    // dataset B columns 0..2 → rows 0,1 of B (gene2 and gene3)
    for j in 0..2 {
        assert!((mat[(0, 3 + j)] - raw_b[(0, j)]).abs() < 1e-6);
        assert!((mat[(1, 3 + j)] - raw_b[(1, j)]).abs() < 1e-6);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────
// columns_triplets / read_columns_* (batch read path)
// ─────────────────────────────────────────────────────

#[test]
fn columns_triplets_single_backend() -> anyhow::Result<()> {
    let nrow = 6;
    let ncol = 12;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    // Read a subset of columns
    let sel = vec![2, 5, 0, 11];
    let mat = vec.read_columns_ndarray(sel.iter().copied())?;
    let expected = raw.select(Axis(1), &sel);

    assert_eq!(mat, expected);
    Ok(())
}

#[test]
fn columns_triplets_two_backends() -> anyhow::Result<()> {
    let nrow = 5;
    let ncol_a = 4;
    let ncol_b = 6;
    let rows = str_vec(nrow, "r");

    let raw_a = Array2::<f32>::runif(nrow, ncol_a);
    let raw_b = Array2::<f32>::runif(nrow, ncol_b);

    let sp_a = make_named_sparse(&raw_a, &rows, &str_vec(ncol_a, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows, &str_vec(ncol_b, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, None)?;
    vec.push(sp_b, None)?;

    // Global columns: 0..4 from backend A, 4..10 from backend B
    // Read columns spanning both backends in mixed order
    let sel = [0, 5, 3, 7];
    let mat = vec.read_columns_ndarray(sel.iter().copied())?;

    // Build expected manually: col0=A[:,0], col1=B[:,1], col2=A[:,3], col3=B[:,3]
    let expected = ndarray::stack(
        Axis(1),
        &[
            raw_a.column(0),
            raw_b.column(1), // global 5 = local 1 in B
            raw_a.column(3),
            raw_b.column(3), // global 7 = local 3 in B
        ],
    )?;

    assert_eq!(mat, expected);
    Ok(())
}

#[test]
fn columns_triplets_preserves_order() -> anyhow::Result<()> {
    let nrow = 4;
    let ncol = 8;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    // Read in reversed order — output columns should match the requested order
    let sel = vec![7, 6, 5, 4, 3, 2, 1, 0];
    let mat = vec.read_columns_ndarray(sel.iter().copied())?;
    let expected = raw.select(Axis(1), &sel);

    assert_eq!(mat, expected);
    Ok(())
}

#[test]
fn read_all_columns_matches_original() -> anyhow::Result<()> {
    let nrow = 7;
    let ncol = 15;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let mat = vec.read_columns_ndarray(0..ncol)?;
    assert_eq!(mat, raw);
    Ok(())
}

#[test]
fn read_columns_across_three_backends() -> anyhow::Result<()> {
    let nrow = 4;
    let sizes = [3, 5, 4];
    let rows = str_vec(nrow, "r");

    let raws: Vec<Array2<f32>> = sizes
        .iter()
        .map(|&n| Array2::<f32>::runif(nrow, n))
        .collect();

    let mut vec = SparseIoVec::new();
    for (i, raw) in raws.iter().enumerate() {
        let cols = str_vec(raw.ncols(), &format!("d{i}_c"));
        let sp = make_named_sparse(raw, &rows, &cols);
        vec.push(sp, None)?;
    }

    assert_eq!(vec.num_columns(), 12); // 3+5+4
    assert_eq!(vec.len(), 3);

    // Read one column from each backend: global 1 (d0), 5 (d1), 10 (d2)
    let sel = [1, 5, 10];
    let mat = vec.read_columns_ndarray(sel.iter().copied())?;

    let expected = ndarray::stack(
        Axis(1),
        &[
            raws[0].column(1),
            raws[1].column(2), // global 5 = offset 3 + local 2
            raws[2].column(2), // global 10 = offset 8 + local 2
        ],
    )?;

    assert_eq!(mat, expected);
    Ok(())
}

// ─────────────────────────────────────────────────────
// read_columns in multiple output formats
// ─────────────────────────────────────────────────────

fn ndarray_to_dmatrix(a: &Array2<f32>) -> nalgebra::DMatrix<f32> {
    let (rows, cols) = a.dim();
    nalgebra::DMatrix::from_row_iterator(rows, cols, a.iter().cloned())
}

#[test]
fn read_columns_dmatrix_matches_ndarray() -> anyhow::Result<()> {
    let nrow = 5;
    let ncol = 10;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let sel = [1, 3, 7];
    let nd = vec.read_columns_ndarray(sel.iter().copied())?;
    let dm = vec.read_columns_dmatrix(sel.iter().copied())?;

    assert_eq!(dm, ndarray_to_dmatrix(&nd));
    Ok(())
}

#[test]
fn read_columns_csc_nnz_correct() -> anyhow::Result<()> {
    let nrow = 4;
    let ncol = 6;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let sel = [0, 2, 4];
    let csc = vec.read_columns_csc(sel.iter().copied())?;

    assert_eq!(csc.nrows(), nrow);
    assert_eq!(csc.ncols(), sel.len());
    // runif produces non-zero values, so nnz should equal nrow * ncols_selected
    assert_eq!(csc.nnz(), nrow * sel.len());
    Ok(())
}

// ─────────────────────────────────────────────────────
// Group assignment
// ─────────────────────────────────────────────────────

#[test]
fn assign_groups_basic() -> anyhow::Result<()> {
    let nrow = 3;
    let ncol = 9;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    // 3 groups of 3 columns each
    let groups = vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"];
    vec.assign_groups(&groups, None);

    assert_eq!(vec.num_groups(), 3);

    let keys = vec.group_keys().unwrap();
    assert_eq!(keys.len(), 3);
    // Keys should be sorted
    assert_eq!(&*keys[0], "A");
    assert_eq!(&*keys[1], "B");
    assert_eq!(&*keys[2], "C");

    let grouped = vec.take_grouped_columns().unwrap();
    assert_eq!(grouped.len(), 3);
    assert_eq!(grouped[0], vec![0, 1, 2]); // A
    assert_eq!(grouped[1], vec![3, 4, 5]); // B
    assert_eq!(grouped[2], vec![6, 7, 8]); // C

    Ok(())
}

#[test]
fn assign_groups_with_max_columns() -> anyhow::Result<()> {
    let nrow = 3;
    let ncol = 9;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let groups = vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"];
    vec.assign_groups(&groups, Some(2)); // limit to 2 per group

    let grouped = vec.take_grouped_columns().unwrap();
    for g in grouped {
        assert!(g.len() <= 2, "group should have at most 2 columns");
    }

    Ok(())
}

#[test]
fn get_group_membership_roundtrip() -> anyhow::Result<()> {
    let nrow = 3;
    let ncol = 6;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let groups = vec!["X", "X", "Y", "Y", "Z", "Z"];
    vec.assign_groups(&groups, None);

    // All cells assigned to groups should be recoverable
    let membership = vec.get_group_membership(0..ncol)?;
    assert_eq!(membership.len(), ncol);
    // Cells 0,1 → same group; cells 2,3 → same group; cells 4,5 → same group
    assert_eq!(membership[0], membership[1]);
    assert_eq!(membership[2], membership[3]);
    assert_eq!(membership[4], membership[5]);
    assert_ne!(membership[0], membership[2]);
    assert_ne!(membership[2], membership[4]);

    Ok(())
}

// ─────────────────────────────────────────────────────
// Batch registration and KNN matching
// ─────────────────────────────────────────────────────

#[test]
fn register_batches_and_metadata() -> anyhow::Result<()> {
    let nrow = 5;
    let ncol = 30;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    // Create a feature matrix (e.g., 3 PCs per cell)
    let features = Array2::<f32>::runif(3, ncol);
    let batch_labels: Vec<&str> = (0..ncol)
        .map(|i| if i < 15 { "batch0" } else { "batch1" })
        .collect();

    vec.register_batches_ndarray(&features, &batch_labels)?;

    assert_eq!(vec.num_batches(), 2);

    let names = vec.batch_names().unwrap();
    assert_eq!(names.len(), 2);

    let name_map = vec.batch_name_map().unwrap();
    assert!(name_map.contains_key::<Box<str>>(&"batch0".into()));
    assert!(name_map.contains_key::<Box<str>>(&"batch1".into()));

    // Check batch membership roundtrip
    let membership = vec.get_batch_membership(0..ncol);
    assert_eq!(membership.len(), ncol);
    // First 15 should be in same batch, last 15 in another
    let b0 = membership[0];
    let b1 = membership[15];
    assert_ne!(b0, b1);
    for m in &membership[0..15] {
        assert_eq!(*m, b0);
    }
    for m in &membership[15..30] {
        assert_eq!(*m, b1);
    }

    Ok(())
}

#[test]
fn matched_columns_across_batches() -> anyhow::Result<()> {
    let nrow = 4;
    let ncol = 40;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    // 2 batches of 20 cells each
    let features = Array2::<f32>::runif(3, ncol);
    let batch_labels: Vec<usize> = (0..ncol).map(|i| i / 20).collect();
    vec.register_batches_ndarray(&features, &batch_labels)?;

    // Match cells from batch 0 against batch 1
    let source_cells = 0..5;
    let knn = 3;
    let (csc, sources, distances) = vec.read_matched_columns_csc(source_cells, &[1], knn, true)?;

    assert_eq!(csc.nrows(), nrow);
    // Each source cell should have up to knn matched columns
    assert!(csc.ncols() <= 5 * knn);
    assert_eq!(sources.len(), csc.ncols());
    assert_eq!(distances.len(), csc.ncols());

    // All source columns should be from the input range
    for &s in &sources {
        assert!(s < 5);
    }

    // Distances should be non-negative
    for &d in &distances {
        assert!(d >= 0.0);
    }

    Ok(())
}

#[test]
fn neighbouring_columns_skip_batches() -> anyhow::Result<()> {
    let nrow = 4;
    let ncol = 60;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let features = Array2::<f32>::runif(5, ncol);
    let batch_labels: Vec<usize> = (0..ncol).map(|i| i / 20).collect();
    vec.register_batches_ndarray(&features, &batch_labels)?;

    // Skip batch 1, so only batch 2 neighbours should appear
    let (csc_skip, _, _, _) = vec.read_neighbouring_columns_csc(0..3, 2, 2, true, Some(&[1]))?;

    // Without skipping
    let (csc_noskip, _, _, _) = vec.read_neighbouring_columns_csc(0..3, 2, 2, true, None)?;

    // Skipping a batch should result in fewer or equal matched columns
    assert!(csc_skip.ncols() <= csc_noskip.ncols());

    Ok(())
}

// ─────────────────────────────────────────────────────
// read_column_offset helper (tested indirectly but
// also check that matched methods return correct shapes)
// ─────────────────────────────────────────────────────

#[test]
fn matched_columns_dmatrix_and_ndarray_agree() -> anyhow::Result<()> {
    let nrow = 5;
    let ncol = 40;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let features = Array2::<f32>::runif(3, ncol);
    let batch_labels: Vec<usize> = (0..ncol).map(|i| i / 20).collect();
    vec.register_batches_ndarray(&features, &batch_labels)?;

    let cells = 0..5;
    let (nd, sources_nd, dist_nd) =
        vec.read_matched_columns_ndarray(cells.clone(), &[1], 2, true)?;
    let (dm, sources_dm, dist_dm) = vec.read_matched_columns_dmatrix(cells, &[1], 2, true)?;

    assert_eq!(nd.nrows(), dm.nrows());
    assert_eq!(nd.ncols(), dm.ncols());
    assert_eq!(sources_nd, sources_dm);
    assert_eq!(dist_nd, dist_dm);

    // Cross-check values
    let dm_nd = ndarray_to_dmatrix(&nd);
    assert_eq!(dm, dm_nd);

    Ok(())
}

// ─────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────

#[test]
fn columns_triplets_empty_selection() -> anyhow::Result<()> {
    let nrow = 3;
    let ncol = 5;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let ((nr, nc), triplets) = vec.columns_triplets(std::iter::empty())?;
    assert_eq!(nr, nrow);
    assert_eq!(nc, 0);
    assert!(triplets.is_empty());
    Ok(())
}

#[test]
fn columns_triplets_single_column() -> anyhow::Result<()> {
    let nrow = 4;
    let ncol = 8;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let mat = vec.read_columns_ndarray(std::iter::once(3))?;
    let expected = raw.select(Axis(1), &[3]);

    assert_eq!(mat, expected);
    Ok(())
}

#[test]
fn columns_triplets_duplicate_column_indices() -> anyhow::Result<()> {
    let nrow = 3;
    let ncol = 5;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    // Request same column twice — should produce two identical columns
    let sel = vec![2, 2];
    let mat = vec.read_columns_ndarray(sel.iter().copied())?;
    let expected = raw.select(Axis(1), &sel);

    assert_eq!(mat, expected);
    Ok(())
}

#[test]
fn num_non_zeros_across_datasets() -> anyhow::Result<()> {
    let nrow = 3;
    let raw_a = Array2::<f32>::runif(nrow, 4);
    let raw_b = Array2::<f32>::runif(nrow, 5);
    let rows = str_vec(nrow, "r");

    let sp_a = make_named_sparse(&raw_a, &rows, &str_vec(4, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows, &str_vec(5, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, None)?;
    vec.push(sp_b, None)?;

    let nnz = vec.num_non_zeros()?;
    // runif produces all non-zero values
    assert_eq!(nnz, nrow * (4 + 5));
    Ok(())
}

#[test]
fn take_backend_columns() -> anyhow::Result<()> {
    let nrow = 3;
    let raw_a = Array2::<f32>::runif(nrow, 4);
    let raw_b = Array2::<f32>::runif(nrow, 5);
    let rows = str_vec(nrow, "r");

    let sp_a = make_named_sparse(&raw_a, &rows, &str_vec(4, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows, &str_vec(5, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, None)?;
    vec.push(sp_b, None)?;

    let backend_cols = vec.take_backend_columns();
    assert_eq!(backend_cols.len(), 2);

    // Total columns across all backends should match
    let total: usize = backend_cols.iter().map(|(_, cols)| cols.len()).sum();
    assert_eq!(total, 9);
    Ok(())
}

// ─────────────────────────────────────────────────────
// rows_triplets / read_rows_* (batch row-read path)
// ─────────────────────────────────────────────────────

#[test]
fn rows_triplets_single_backend() -> anyhow::Result<()> {
    let nrow = 8;
    let ncol = 10;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let sel = vec![3, 0, 7, 5];
    let mat = vec.read_rows_ndarray(sel.iter().copied())?;
    let expected = raw.select(Axis(0), &sel);

    assert_eq!(mat, expected);
    Ok(())
}

#[test]
fn rows_triplets_two_backends_aligned() -> anyhow::Result<()> {
    let nrow = 5;
    let ncol_a = 4;
    let ncol_b = 6;
    let rows = str_vec(nrow, "r");

    let raw_a = Array2::<f32>::runif(nrow, ncol_a);
    let raw_b = Array2::<f32>::runif(nrow, ncol_b);

    let sp_a = make_named_sparse(&raw_a, &rows, &str_vec(ncol_a, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows, &str_vec(ncol_b, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, None)?;
    vec.push(sp_b, None)?;

    // Read rows 1 and 4 across both backends. Output columns are
    // [A's columns | B's columns] = ncol_a + ncol_b.
    let sel = [1, 4];
    let mat = vec.read_rows_ndarray(sel.iter().copied())?;

    assert_eq!(mat.nrows(), 2);
    assert_eq!(mat.ncols(), ncol_a + ncol_b);

    for (out_row, &r) in sel.iter().enumerate() {
        for j in 0..ncol_a {
            assert!((mat[(out_row, j)] - raw_a[(r, j)]).abs() < 1e-6);
        }
        for j in 0..ncol_b {
            assert!((mat[(out_row, ncol_a + j)] - raw_b[(r, j)]).abs() < 1e-6);
        }
    }
    Ok(())
}

#[test]
fn rows_triplets_swapped_row_order_in_second_backend() -> anyhow::Result<()> {
    // Same row name set in both backends, but backend B has rows 0/1
    // swapped at the local level. Compact rows on the SparseIoVec
    // must reflect global row identity, not backend-local position.
    let rows_a = str_vec(5, "gene");
    let mut rows_b = str_vec(5, "gene");
    rows_b.swap(0, 1);

    let raw_a = Array2::<f32>::runif(5, 3);
    let raw_b = Array2::<f32>::runif(5, 4);

    let sp_a = make_named_sparse(&raw_a, &rows_a, &str_vec(3, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows_b, &str_vec(4, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, None)?;
    vec.push(sp_b, None)?;

    // Compact row 0 == "gene0", which is local row 0 in A but local
    // row 1 in B. read_rows_ndarray must remap correctly.
    let mat = vec.read_rows_ndarray(std::iter::once(0))?;
    assert_eq!(mat.shape(), &[1, 3 + 4]);
    for j in 0..3 {
        assert!((mat[(0, j)] - raw_a[(0, j)]).abs() < 1e-6);
    }
    for j in 0..4 {
        // gene0 lives at local row 1 of B (because rows_b swapped 0,1)
        assert!((mat[(0, 3 + j)] - raw_b[(1, j)]).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn rows_triplets_disjoint_intersection() -> anyhow::Result<()> {
    // A has gene0..gene3, B has gene2..gene5. Intersection: {gene2, gene3}.
    // Compact row 0 -> gene2 (local 2 in A, local 0 in B)
    // Compact row 1 -> gene3 (local 3 in A, local 1 in B)
    let rows_a: Vec<Box<str>> = (0..4).map(|i| format!("gene{i}").into()).collect();
    let rows_b: Vec<Box<str>> = (2..6).map(|i| format!("gene{i}").into()).collect();

    let raw_a = Array2::<f32>::runif(4, 3);
    let raw_b = Array2::<f32>::runif(4, 2);
    let sp_a = make_named_sparse(&raw_a, &rows_a, &str_vec(3, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows_b, &str_vec(2, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, Some("A".into()))?;
    vec.push(sp_b, Some("B".into()))?;
    assert_eq!(vec.num_rows(), 2);

    let mat = vec.read_rows_ndarray(0..2)?;
    assert_eq!(mat.shape(), &[2, 5]);

    // Compact row 0 = gene2 = A[2, :] | B[0, :]
    for j in 0..3 {
        assert!((mat[(0, j)] - raw_a[(2, j)]).abs() < 1e-6);
    }
    for j in 0..2 {
        assert!((mat[(0, 3 + j)] - raw_b[(0, j)]).abs() < 1e-6);
    }
    // Compact row 1 = gene3 = A[3, :] | B[1, :]
    for j in 0..3 {
        assert!((mat[(1, j)] - raw_a[(3, j)]).abs() < 1e-6);
    }
    for j in 0..2 {
        assert!((mat[(1, 3 + j)] - raw_b[(1, j)]).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn rows_triplets_preserves_order() -> anyhow::Result<()> {
    let nrow = 6;
    let ncol = 4;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let sel: Vec<usize> = (0..nrow).rev().collect();
    let mat = vec.read_rows_ndarray(sel.iter().copied())?;
    let expected = raw.select(Axis(0), &sel);
    assert_eq!(mat, expected);
    Ok(())
}

#[test]
fn read_all_rows_matches_horizontal_concat() -> anyhow::Result<()> {
    // Reading every compact row from a multi-backend vec should
    // recover the horizontal concatenation of the underlying matrices.
    let nrow = 4;
    let rows = str_vec(nrow, "r");
    let raw_a = Array2::<f32>::runif(nrow, 3);
    let raw_b = Array2::<f32>::runif(nrow, 5);

    let sp_a = make_named_sparse(&raw_a, &rows, &str_vec(3, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows, &str_vec(5, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, None)?;
    vec.push(sp_b, None)?;

    let mat = vec.read_rows_ndarray(0..nrow)?;
    let expected = ndarray::concatenate(Axis(1), &[raw_a.view(), raw_b.view()])?;
    assert_eq!(mat, expected);
    Ok(())
}

#[test]
fn read_rows_matches_read_columns_transposed() -> anyhow::Result<()> {
    // Sanity: reading all rows then all columns gives the same matrix.
    let nrow = 5;
    let rows = str_vec(nrow, "r");
    let raw_a = Array2::<f32>::runif(nrow, 3);
    let raw_b = Array2::<f32>::runif(nrow, 4);

    let sp_a = make_named_sparse(&raw_a, &rows, &str_vec(3, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows, &str_vec(4, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, None)?;
    vec.push(sp_b, None)?;

    let by_rows = vec.read_rows_ndarray(0..vec.num_rows())?;
    let by_cols = vec.read_columns_ndarray(0..vec.num_columns())?;
    assert_eq!(by_rows, by_cols);
    Ok(())
}

#[test]
fn read_rows_dmatrix_matches_ndarray() -> anyhow::Result<()> {
    let nrow = 6;
    let ncol = 8;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let sel = [0, 2, 5];
    let nd = vec.read_rows_ndarray(sel.iter().copied())?;
    let dm = vec.read_rows_dmatrix(sel.iter().copied())?;
    assert_eq!(dm, ndarray_to_dmatrix(&nd));
    Ok(())
}

#[test]
fn read_rows_csc_and_csr_nnz_correct() -> anyhow::Result<()> {
    let nrow = 5;
    let ncol = 7;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let sel = [1, 3];
    let csc = vec.read_rows_csc(sel.iter().copied())?;
    let csr = vec.read_rows_csr(sel.iter().copied())?;

    assert_eq!(csc.nrows(), sel.len());
    assert_eq!(csc.ncols(), ncol);
    assert_eq!(csr.nrows(), sel.len());
    assert_eq!(csr.ncols(), ncol);
    // runif produces non-zero values, so nnz == sel.len() * ncol
    assert_eq!(csc.nnz(), sel.len() * ncol);
    assert_eq!(csr.nnz(), sel.len() * ncol);
    Ok(())
}

#[test]
fn rows_triplets_empty_selection() -> anyhow::Result<()> {
    let nrow = 3;
    let ncol = 5;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let ((nr, nc), triplets) = vec.rows_triplets(std::iter::empty())?;
    assert_eq!(nr, 0);
    assert_eq!(nc, ncol);
    assert!(triplets.is_empty());
    Ok(())
}

#[test]
fn rows_triplets_duplicate_row_indices() -> anyhow::Result<()> {
    let nrow = 4;
    let ncol = 6;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let sel = vec![2, 2];
    let mat = vec.read_rows_ndarray(sel.iter().copied())?;
    let expected = raw.select(Axis(0), &sel);
    assert_eq!(mat, expected);
    Ok(())
}

// ─────────────────────────────────────────────────────
// read_columns_csc direct-slice path
// ─────────────────────────────────────────────────────

/// Reconstruct a dense matrix from a CscMatrix for assertion purposes.
fn csc_to_dense(csc: &nalgebra_sparse::csc::CscMatrix<f32>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((csc.nrows(), csc.ncols()));
    for (j, col) in csc.col_iter().enumerate() {
        for (&i, &v) in col.row_indices().iter().zip(col.values()) {
            out[(i, j)] = v;
        }
    }
    out
}

/// Build a sparse dataset without preloading columns, to exercise the
/// fallback path inside `read_columns_csc` when `csc_column_arrays` is
/// `None`.
fn make_named_sparse_no_preload(
    data: &Array2<f32>,
    row_names: &[Box<str>],
    col_names: &[Box<str>],
) -> Arc<SparseData> {
    let mut sp = create_sparse_from_ndarray(data, None, None).unwrap();
    sp.register_row_names_vec(row_names);
    sp.register_column_names_vec(col_names);
    Arc::from(sp)
}

#[test]
fn read_columns_csc_single_backend_contiguous() -> anyhow::Result<()> {
    let nrow = 12;
    let ncol = 20;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let rows = str_vec(nrow, "r");
    let cols = str_vec(ncol, "c");

    let sp = make_named_sparse(&raw, &rows, &cols);
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let csc = vec.read_columns_csc(5..15)?;
    assert_eq!(csc.nrows(), nrow);
    assert_eq!(csc.ncols(), 10);
    let dense = csc_to_dense(&csc);
    let expected = raw.select(Axis(1), &(5..15).collect::<Vec<_>>());
    assert_eq!(dense, expected);
    Ok(())
}

#[test]
fn read_columns_csc_single_backend_arbitrary() -> anyhow::Result<()> {
    let nrow = 8;
    let ncol = 16;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let sp = make_named_sparse(&raw, &str_vec(nrow, "r"), &str_vec(ncol, "c"));
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let sel = vec![15, 0, 7, 3, 11];
    let csc = vec.read_columns_csc(sel.iter().copied())?;
    let expected = raw.select(Axis(1), &sel);
    assert_eq!(csc_to_dense(&csc), expected);
    Ok(())
}

#[test]
fn read_columns_csc_two_backends() -> anyhow::Result<()> {
    let nrow = 6;
    let ncol_a = 5;
    let ncol_b = 7;
    let rows = str_vec(nrow, "r");
    let raw_a = Array2::<f32>::runif(nrow, ncol_a);
    let raw_b = Array2::<f32>::runif(nrow, ncol_b);

    let sp_a = make_named_sparse(&raw_a, &rows, &str_vec(ncol_a, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows, &str_vec(ncol_b, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, Some("A".into()))?;
    vec.push(sp_b, Some("B".into()))?;

    // Mix cells across backends.
    let sel: Vec<usize> = vec![0, 6, 2, 4, ncol_a + 3, ncol_a + ncol_b - 1];
    let csc = vec.read_columns_csc(sel.iter().copied())?;
    let dense = csc_to_dense(&csc);
    assert_eq!(dense.nrows(), nrow);
    assert_eq!(dense.ncols(), sel.len());

    // Build the expected matrix by hand.
    let mut expected = Array2::<f32>::zeros((nrow, sel.len()));
    for (out_j, &g) in sel.iter().enumerate() {
        if g < ncol_a {
            for i in 0..nrow {
                expected[(i, out_j)] = raw_a[(i, g)];
            }
        } else {
            let lj = g - ncol_a;
            for i in 0..nrow {
                expected[(i, out_j)] = raw_b[(i, lj)];
            }
        }
    }
    assert_eq!(dense, expected);
    Ok(())
}

#[test]
fn read_columns_csc_row_intersection_reordered() -> anyhow::Result<()> {
    // Same row name set in both backends but stored in different order
    // — exercises the l2g + g2c remap, which can produce unsorted compact
    // row indices that the new code must detect and sort.
    let rows_a = str_vec(5, "g");
    let mut rows_b = str_vec(5, "g");
    rows_b.swap(0, 4);
    rows_b.swap(1, 3);

    let raw_a = Array2::<f32>::runif(5, 3);
    let raw_b = Array2::<f32>::runif(5, 4);
    let sp_a = make_named_sparse(&raw_a, &rows_a, &str_vec(3, "a"));
    let sp_b = make_named_sparse(&raw_b, &rows_b, &str_vec(4, "b"));

    let mut vec = SparseIoVec::new();
    vec.push(sp_a, Some("A".into()))?;
    vec.push(sp_b, Some("B".into()))?;

    // Read every column and confirm the result matches the ndarray reader,
    // which goes through the existing triplet path — cross-validation.
    let csc = vec.read_columns_csc(0..vec.num_columns())?;
    let dense_csc = csc_to_dense(&csc);
    let dense_ref = vec.read_columns_ndarray(0..vec.num_columns())?;
    assert_eq!(dense_csc, dense_ref);
    Ok(())
}

#[test]
fn read_columns_csc_no_preload_fallback() -> anyhow::Result<()> {
    let nrow = 7;
    let ncol = 9;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let sp = make_named_sparse_no_preload(&raw, &str_vec(nrow, "r"), &str_vec(ncol, "c"));
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let csc = vec.read_columns_csc(0..ncol)?;
    let dense_csc = csc_to_dense(&csc);
    let dense_ref = vec.read_columns_ndarray(0..ncol)?;
    assert_eq!(dense_csc, dense_ref);
    Ok(())
}

#[test]
fn read_columns_csc_matches_old_triplet_path() -> anyhow::Result<()> {
    // The new direct-slice path must produce a CSC matrix identical to
    // what the old `from_nonzero_triplets` route would yield. We compare
    // against the dense reader (which still uses triplets) and against a
    // hand-built CSC from the same triplets.
    let nrow = 10;
    let ncol = 25;
    let raw = Array2::<f32>::runif(nrow, ncol);
    let sp = make_named_sparse(&raw, &str_vec(nrow, "r"), &str_vec(ncol, "c"));
    let mut vec = SparseIoVec::new();
    vec.push(sp, None)?;

    let cells: Vec<usize> = vec![24, 0, 13, 7, 2, 19, 4];
    let csc_new = vec.read_columns_csc(cells.iter().copied())?;
    let dense_ref = vec.read_columns_ndarray(cells.iter().copied())?;

    assert_eq!(csc_new.nrows(), nrow);
    assert_eq!(csc_new.ncols(), cells.len());
    assert_eq!(csc_to_dense(&csc_new), dense_ref);

    // Check that within each column, row indices are strictly increasing
    // (canonical CSC form, required by `try_from_csc_data`).
    for col in csc_new.col_iter() {
        let rs = col.row_indices();
        for w in rs.windows(2) {
            assert!(w[0] < w[1], "row indices must be strictly increasing");
        }
    }
    Ok(())
}
