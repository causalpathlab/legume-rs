use crate::traits::*;
use nalgebra::LU;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{csc::CscMatrix, csr::CsrMatrix};

trait IntoDense<OutMat> {
    fn matmul(&self, other: &OutMat) -> OutMat;
    fn transpose_matmul(&self, other: &OutMat) -> OutMat;
    fn num_rows(&self) -> usize;
    fn num_columns(&self) -> usize;
}

impl<T> IntoDense<DMatrix<T>> for DMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    fn matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        self * other
    }

    fn transpose_matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        self.transpose() * other
    }

    fn num_rows(&self) -> usize {
        self.nrows()
    }
    fn num_columns(&self) -> usize {
        self.ncols()
    }
}

impl<T> IntoDense<DMatrix<T>> for CscMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    fn matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        self * other
    }
    fn transpose_matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        self.transpose() * other
    }
    fn num_rows(&self) -> usize {
        self.nrows()
    }
    fn num_columns(&self) -> usize {
        self.ncols()
    }
}

impl<T> IntoDense<DMatrix<T>> for CsrMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    fn matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        self * other
    }
    fn transpose_matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        self.transpose() * other
    }
    fn num_rows(&self) -> usize {
        self.nrows()
    }
    fn num_columns(&self) -> usize {
        self.ncols()
    }
}

fn _subspace_iteration<T, D>(xx: &D, rank_and_oversample: usize) -> anyhow::Result<DMatrix<T>>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
    D: IntoDense<DMatrix<T>>,
{
    let max_iter = 5; // five should be enough

    let nc = xx.num_columns();
    let nr = xx.num_rows();
    let mut ll = DMatrix::<T>::zeros(nr, rank_and_oversample);
    let mut qq = DMatrix::<T>::runif(nc, rank_and_oversample);
    let zero = T::from(0.).expect("no zero found");

    for _i in 0..max_iter {
        let lu1 = xx.matmul(&qq);
        let lu1 = LU::new(lu1);
        let lu1 = lu1.l();

        // note: LU may shrink the matrix
        ll.fill(zero);
        ll.fill_with_identity();
        ll.view_mut((0, 0), (nr, rank_and_oversample.min(lu1.ncols())))
            .lower_triangle()
            .copy_from(&lu1);

        let lu2 = xx.transpose_matmul(&ll);
        let lu2 = LU::new(lu2);
        let lu2 = lu2.l();

        // note: LU may shrink the matrix
        qq.fill(zero);
        qq.fill_with_identity();
        qq.view_mut((0, 0), (nc, rank_and_oversample.min(lu2.ncols())))
            .lower_triangle()
            .copy_from(&lu2);
    }

    // let qq = DMatrix::<T>::runif(nc, rank_and_oversample);

    let qr_q = xx.matmul(&qq).qr().q();
    let kk = rank_and_oversample.min(qr_q.ncols());
    let ret = qr_q.columns(0, kk).into_owned();

    Ok(ret)
}

fn _randomized_svd<T, D>(
    xx: &D,
    max_rank: usize,
) -> anyhow::Result<(DMatrix<T>, DVector<T>, DMatrix<T>)>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
    D: IntoDense<DMatrix<T>>,
{
    let nr = xx.num_rows();
    let nc = xx.num_columns();

    let mut rank = nr.min(nc);
    let mut oversample = 0;

    if max_rank > 0 && rank > max_rank {
        rank = max_rank;
        oversample = 5;
    }

    debug_assert!(rank > 0, "Must be at least rank = 1");

    let qq = _subspace_iteration(xx, rank + oversample)?;
    let rank = rank.min(qq.ncols());

    let qq = qq.columns(0, rank).into_owned();

    // let bb = qq.transpose() * xx
    let bb = xx.transpose_matmul(&qq).transpose();

    let svd = bb.svd(true, true);

    if let (Some(svd_u), Some(svd_vt)) = (svd.u, svd.v_t) {
        return Ok((
            qq.clone() * svd_u.columns(0, rank).into_owned(),
            svd.singular_values.rows(0, rank).into_owned(),
            svd_vt.transpose().columns(0, rank).into_owned(),
        ));
    }
    Err(anyhow::anyhow!("randomized SVD failed"))
}

impl<T> RandomizedAlgs for DMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    type InMat = DMatrix<T>;
    type OutMat = DMatrix<T>;
    type DVec = DVector<T>;
    type Scalar = T;

    fn rsvd(&self, max_rank: usize) -> anyhow::Result<(Self::OutMat, Self::DVec, Self::OutMat)> {
        _randomized_svd(self, max_rank)
    }
}

impl<T> RandomizedAlgs for CscMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    type InMat = CscMatrix<T>;
    type OutMat = DMatrix<T>;
    type DVec = DVector<T>;
    type Scalar = T;

    fn rsvd(&self, max_rank: usize) -> anyhow::Result<(Self::OutMat, Self::DVec, Self::OutMat)> {
        _randomized_svd(self, max_rank)
    }
}

impl<T> RandomizedAlgs for CsrMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    type InMat = CsrMatrix<T>;
    type OutMat = DMatrix<T>;
    type DVec = DVector<T>;
    type Scalar = T;

    fn rsvd(&self, max_rank: usize) -> anyhow::Result<(Self::OutMat, Self::DVec, Self::OutMat)> {
        _randomized_svd(self, max_rank)
    }
}
