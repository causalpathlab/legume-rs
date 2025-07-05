use ndarray::Array2;

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

    let (nrows, ncols, triplets) = xx.to_nonzero_triplets()?;

    let xx = nalgebra_sparse::CscMatrix::<f32>::from_nonzero_triplets(nrows, ncols, triplets)?;

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

#[test]
fn ndarray_rsvd_test() -> anyhow::Result<()> {
    use matrix_util::traits::*;

    let xx = ndarray::Array2::<f32>::eye(8);
    dbg!(&xx);

    let svd = xx.rsvd(3)?;

    dbg!(&svd);

    dbg!(svd.0.t().dot(&svd.0));
    dbg!(svd.2.t().dot(&svd.2));

    Ok(())
}

#[test]
fn ndarray_rsvd_test2() -> anyhow::Result<()> {
    use matrix_util::traits::*;

    let xx = ndarray::Array2::<f32>::rnorm(10, 8);

    let svd = xx.rsvd(5)?;

    let mut identity = Array2::zeros((5, 5));
    identity.diag_mut().fill(1.);

    let eye = svd.0.t().dot(&svd.0);
    approx::assert_abs_diff_eq!(eye, identity, epsilon = 1e-4);

    let eye = svd.2.t().dot(&svd.2);
    approx::assert_abs_diff_eq!(eye, identity, epsilon = 1e-4);

    Ok(())
}

#[test]
fn ndarray_test() -> anyhow::Result<()> {
    use ndarray::prelude::*;

    // Define two matrices
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let b = array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];

    // Perform matrix multiplication
    let c = a.dot(&b);

    // Print the result
    println!("Matrix A:\n{}", a);
    println!("Matrix B:\n{}", b);
    println!("Matrix C = (A * B):\n{}", c);

    // Expected result
    let expected = array![[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]];

    // Verify the result
    assert_eq!(c, expected);

    Ok(())
}
