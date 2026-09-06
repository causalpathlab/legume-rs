use crate::traits::*;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{csc::CscMatrix, csr::CsrMatrix};

/// Fixed start-vector seed for the randomized-SVD subspace iteration. The
/// iteration converges onto the dominant subspace, so pinning the start makes
/// `rsvd` reproducible without altering the subspace it recovers.
const RSVD_SUBSPACE_SEED: u64 = 0x5253_5644_5342_5350; // "RSVDSBSP"

/// Compute the Nystrom basis: `U * diag(1 / (s + eps))`.
///
/// Given the left singular vectors `u` and singular values `s` from an SVD,
/// returns the pseudo-inverted projection matrix used for out-of-sample
/// Nystrom extension.
pub fn nystrom_basis(u: &DMatrix<f32>, s: &DVector<f32>) -> DMatrix<f32> {
    let eps = 1e-8;
    let sinv = DVector::from_iterator(s.len(), s.iter().map(|&si| 1.0 / (si + eps)));
    u * DMatrix::from_diagonal(&sinv)
}

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
    // Fixed seed: the subspace iterations below converge onto the dominant
    // subspace regardless of the start, so a pinned (rather than entropy) draw
    // makes the whole randomized SVD reproducible run-to-run — which in turn
    // pins every downstream consumer (binary-sketch collapse, layout, SVD fits)
    // — without changing what subspace it recovers.
    let mut qq = DMatrix::<T>::runif_seeded(nc, rank_and_oversample, RSVD_SUBSPACE_SEED);
    let half = T::from(0.5).expect("no half found");
    qq.iter_mut().for_each(|x| *x -= half);

    // Each half-step re-orthonormalises the iterate with a thin QR. The
    // basis must span exactly the range of the product it came from: a
    // pivoted LU factor does not (its permutation is lost), and iterating on
    // a row-permuted range does not converge onto the dominant subspace.
    for _i in 0..max_iter {
        let ll = xx.matmul(&qq).qr().q();
        qq = xx.transpose_matmul(&ll).qr().q();
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

    // Keep the oversampled basis through the projection: its columns are
    // not ordered by singular value, so truncating here would discard part
    // of the dominant subspace. The rank is applied to the small SVD below.
    let qq = _subspace_iteration(xx, rank + oversample)?;
    let rank = rank.min(qq.ncols());

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

#[cfg(test)]
#[path = "dmatrix_rsvd_tests.rs"]
mod tests;
