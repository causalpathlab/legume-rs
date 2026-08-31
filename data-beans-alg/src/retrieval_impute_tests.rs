//! The contract under test: neighbours are found in the caller's latent,
//! weights follow distance, and the imputed row is exactly the weighted
//! average of the retrieved reference columns — with zero-latent rows
//! left zero rather than parked on an arbitrary neighbour.

use super::*;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use std::sync::Arc;

fn make_ref_data(
    dir: &tempfile::TempDir,
    triplets: &[(u64, u64, f32)],
    n_genes: usize,
    n_cells: usize,
) -> anyhow::Result<SparseIoVec> {
    let path = dir.path().join("ref.zarr");
    let mut backend = create_sparse_from_triplets(
        triplets,
        (n_genes, n_cells, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )?;
    let gene_names: Vec<Box<str>> = (0..n_genes).map(|i| format!("gene_{i}").into()).collect();
    let cell_names: Vec<Box<str>> = (0..n_cells).map(|i| format!("cell_{i}").into()).collect();
    backend.register_row_names_vec(&gene_names);
    backend.register_column_names_vec(&cell_names);
    let mut data_vec = SparseIoVec::new();
    data_vec.push(Arc::from(backend), None)?;
    Ok(data_vec)
}

#[test]
fn weights_sum_to_one_and_favor_the_nearer_neighbour() {
    let w = dist_to_softmax_weights(&[0.1, 0.9], 1.0);
    assert_eq!(w.len(), 2);
    assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(w[0] > w[1]);
}

#[test]
fn lower_temperature_sharpens_the_weights() {
    let warm = dist_to_softmax_weights(&[0.1, 0.9], 1.0);
    let cold = dist_to_softmax_weights(&[0.1, 0.9], 0.1);
    assert!(cold[0] > warm[0]);
}

#[test]
fn empty_distances_yield_empty_weights() {
    assert!(dist_to_softmax_weights(&[], 1.0).is_empty());
}

#[test]
fn imputed_rows_average_the_matched_cluster() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    // Two well-separated latent clusters: cells {0,1} at e_x, cells {2,3}
    // at e_y. Counts differ within a cluster so the average is visible.
    let ref_latent = Mat::from_row_slice(
        4,
        2,
        &[
            1.0, 0.0, // cell_0
            1.0, 0.0, // cell_1
            0.0, 1.0, // cell_2
            0.0, 1.0, // cell_3
        ],
    );
    let triplets: Vec<(u64, u64, f32)> = vec![
        (0, 0, 2.0),
        (0, 1, 4.0), // gene_0 lives in cluster A
        (1, 2, 6.0),
        (1, 3, 10.0), // gene_1 lives in cluster B
    ];
    let ref_data = make_ref_data(&dir, &triplets, 2, 4)?;

    // Queries sit exactly on each cluster's coordinates: both neighbours
    // are equidistant, so their softmax weights are exactly 1/2 each.
    let query_latent = Mat::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let imputed = retrieval_impute(
        &query_latent,
        &ref_latent,
        &ref_data,
        &RetrievalImputeConfig {
            knn: 2,
            temperature: 1.0,
            chunk: 64,
        },
    )?;

    assert_eq!(imputed.nrows(), 2);
    assert_eq!(imputed.ncols(), 2);
    assert!((imputed[(0, 0)] - 3.0).abs() < 1e-4, "mean of 2 and 4");
    assert!(imputed[(0, 1)].abs() < 1e-4, "cluster B gene stays out");
    assert!(imputed[(1, 0)].abs() < 1e-4, "cluster A gene stays out");
    assert!((imputed[(1, 1)] - 8.0).abs() < 1e-4, "mean of 6 and 10");
    Ok(())
}

#[test]
fn zero_latent_query_rows_stay_zero() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let ref_latent = Mat::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let triplets: Vec<(u64, u64, f32)> = vec![(0, 0, 5.0), (1, 1, 7.0)];
    let ref_data = make_ref_data(&dir, &triplets, 2, 2)?;

    let query_latent = Mat::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 0.0]);
    let imputed = retrieval_impute(
        &query_latent,
        &ref_latent,
        &ref_data,
        &RetrievalImputeConfig {
            knn: 1,
            temperature: 1.0,
            chunk: 64,
        },
    )?;
    assert!(imputed.row(0).iter().all(|&x| x == 0.0));
    assert!(imputed[(1, 0)] > 0.0);
    Ok(())
}

#[test]
fn dimension_and_count_mismatches_are_refused() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let triplets: Vec<(u64, u64, f32)> = vec![(0, 0, 1.0), (1, 1, 1.0)];
    let ref_data = make_ref_data(&dir, &triplets, 2, 2)?;
    let cfg = RetrievalImputeConfig {
        knn: 1,
        temperature: 1.0,
        chunk: 64,
    };

    // K mismatch between query and reference latents.
    let bad_query = Mat::zeros(1, 3);
    let ref_latent = Mat::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    assert!(retrieval_impute(&bad_query, &ref_latent, &ref_data, &cfg).is_err());

    // Reference latent rows disagree with the backend's cell count.
    let short_ref = Mat::from_row_slice(1, 2, &[1.0, 0.0]);
    let query = Mat::from_row_slice(1, 2, &[1.0, 0.0]);
    assert!(retrieval_impute(&query, &short_ref, &ref_data, &cfg).is_err());
    Ok(())
}
