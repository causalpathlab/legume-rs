#[test]
fn order_test() {
    use ndarray::{arr2, Array2};
    // Create a 2x3 array
    let array = arr2(&[[1, 2, 3], [4, 5, 6]]);

    // Convert to a slice
    let slice = array.as_slice().unwrap();
    println!("Slice: {:?}", slice);

    // Reconstruct the array from the slice
    let reconstructed = Array2::from_shape_vec((2, 3), slice.to_vec()).unwrap();
    println!("Reconstructed array:\n{:?}", reconstructed);

    // Check if the original and reconstructed arrays are equal
    assert_eq!(array, reconstructed);
}

#[test]
fn dmatrix_rsvd_test() -> anyhow::Result<()> {
    use matrix_util::traits::*;

    let mut xx = nalgebra::DMatrix::<f32>::zeros(8, 8);
    xx.fill_with_identity();

    dbg!(&xx);

    let svd = xx.rsvd(3)?;

    dbg!(&svd);

    dbg!(svd.0.transpose() * &svd.0);
    dbg!(svd.2.transpose() * &svd.2);

    Ok(())
}

#[test]
fn dmatrix_csc_rsvd_test() -> anyhow::Result<()> {
    use matrix_util::traits::*;

    let mut xx = nalgebra::DMatrix::<f32>::zeros(8, 8);
    xx.fill_with_identity();
    dbg!(&xx);

    let t = xx.to_nonzero_triplets()?;

    let xx = nalgebra_sparse::CscMatrix::<f32>::from_nonzero_triplets(t.nrow, t.ncol, &t.triplets)?;

    let svd = xx.rsvd(3)?;

    dbg!(&svd);

    dbg!(svd.0.transpose() * &svd.0);
    dbg!(svd.2.transpose() * &svd.2);

    Ok(())
}

#[test]
fn dmatrix_rsvd_test2() -> anyhow::Result<()> {
    use matrix_util::traits::*;
    let x = nalgebra::DMatrix::<f32>::rnorm(100, 50);

    let svd = x.rsvd(3)?;
    let mut identity = nalgebra::DMatrix::<f32>::zeros(3, 3);
    identity.fill_with_identity();

    let eye = svd.0.transpose() * svd.0;
    approx::assert_abs_diff_eq!(eye, identity, epsilon = 1e-4);

    let eye = svd.2.transpose() * svd.2;
    approx::assert_abs_diff_eq!(eye, identity, epsilon = 1e-4);

    Ok(())
}
