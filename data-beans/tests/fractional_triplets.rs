//! Fractional values round-trip through the sparse backend unchanged.
//!
//! Carried pseudobulks store per-cell *rates* — non-integer by construction —
//! and the whole `pb_reference` scheme rests on the backend not quantizing
//! them (`svd`'s `.round()` on adjusted counts is that caller's own choice,
//! not a backend constraint). This pins bit-exact round-trip, not "close".

use data_beans::sparse_io::create_sparse_from_triplets;
use data_beans::sparse_io_vector::SparseIoVec;
use std::sync::Arc;

#[test]
fn fractional_rates_come_back_bit_exact() {
    let dir = std::env::temp_dir().join(format!("db_fractional_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("fractional.zarr");
    let _ = std::fs::remove_dir_all(&path);

    // Awkward values on purpose: sub-unit rates, non-dyadic fractions, and a
    // near-integer that a cast would silently floor.
    let vals: [f32; 6] = [0.083_333_336, 2.5, 1.0 / 3.0, 7.999_999_5, 1e-6, 123.456_79];
    let triplets: Vec<(u64, u64, f32)> = vals
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u64, (i % 2) as u64, v))
        .collect();

    let mut b = create_sparse_from_triplets(
        &triplets,
        (vals.len(), 2, triplets.len()),
        Some(path.to_str().expect("utf8")),
        Some(&data_beans::sparse_io::SparseIoBackend::Zarr),
    )
    .expect("create");
    b.register_row_names_vec(
        &(0..vals.len())
            .map(|g| format!("g{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(&["c0".into(), "c1".into()]);

    let mut v = SparseIoVec::new();
    v.push(Arc::from(b), None).expect("push");
    let csc = v.read_columns_csc(0..2).expect("read");

    let mut seen = 0;
    for (col, col_view) in csc.col_iter().enumerate() {
        for (&g, &val) in col_view.row_indices().iter().zip(col_view.values().iter()) {
            assert_eq!(
                val.to_bits(),
                vals[g].to_bits(),
                "gene {g} col {col}: wrote {} read {val}",
                vals[g],
            );
            seen += 1;
        }
    }
    assert_eq!(seen, vals.len(), "every triplet must come back");
}
