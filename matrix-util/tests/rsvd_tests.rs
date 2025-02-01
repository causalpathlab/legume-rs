#[test]
fn dmatrix_rsvd_test() -> anyhow::Result<()> {
    use matrix_util::dmatrix_rsvd::RSVD;

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
fn ndarray_rsvd_test() -> anyhow::Result<()> {
    use matrix_util::ndarray_rsvd::RSVD;

    let xx = ndarray::Array2::<f32>::eye(8);
    dbg!(&xx);

    let svd = xx.rsvd(3)?;

    dbg!(&svd);

    dbg!(svd.0.t().dot(&svd.0));
    dbg!(svd.2.t().dot(&svd.2));

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
