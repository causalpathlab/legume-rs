use crate::traits::*;
use ndarray::{s, Array1, Array2};
use ndarray_linalg::qr::QR;
use ndarray_linalg::svd::SVD;

use num_traits::{Float, FromPrimitive};

type Mat = Array2<f32>;
type Vec = Array1<f32>;

pub trait RSVD {
    fn rsvd(&self, rank: usize) -> anyhow::Result<(Mat, Vec, Mat)>;
}

impl RSVD for Mat {
    fn rsvd(&self, rank: usize) -> anyhow::Result<(Mat, Vec, Mat)> {
        let default_iter = 5;
        let mut rsvd = RandomizedSVD::new(rank, default_iter);
        rsvd.compute(&self)?;
        Ok((
            rsvd.matrix_u().clone(),
            rsvd.singular_values().clone(),
            rsvd.matrix_v().clone(),
        ))
    }
}

impl<T> RandomizedAlgs for Array2<T>
where
    T: ndarray::ScalarOperand
        + ndarray_linalg::Scalar<Real = T>
        + ndarray_linalg::Lapack
        + Float
        + FromPrimitive
        + Send,
{
    type InMat = Array2<T>;
    type OutMat = Array2<T>;
    type DVec = Array1<T>;
    type Scalar = T;
    fn rsvd(&self, max_rank: usize) -> anyhow::Result<(Self::OutMat, Self::DVec, Self::OutMat)> {
        let default_iter = 5_usize;
        let mut rsvd = RandomizedSVD::<T>::new(max_rank, default_iter);
        rsvd.compute(&self)?;
        Ok((
            rsvd.matrix_u().clone(),
            rsvd.singular_values().clone(),
            rsvd.matrix_v().clone(),
        ))
    }
}

/// Randomized SVD
///
/// Implement Alg 4.4 of Halko et al. (2009)
/// Modified from <https://github.com/kazuotani14/RandomizedSvd>
///
pub struct RandomizedSVD<T>
where
    T: ndarray::ScalarOperand
        + ndarray_linalg::Scalar<Real = T>
        + ndarray_linalg::Lapack
        + Float
        + FromPrimitive
        + Send,
{
    max_rank: usize,
    iter: usize,
    u_vectors: Array2<T>,
    singular_values: Array1<T>,
    v_vectors: Array2<T>,
    qq: Array2<T>,
    verbose: bool,
}

impl<T> RandomizedSVD<T>
where
    T: ndarray::ScalarOperand
        + ndarray_linalg::Scalar<Real = T>
        + ndarray_linalg::Lapack
        + Float
        + FromPrimitive
        + Send,
{
    pub fn new(max_rank: usize, iter: usize) -> Self {
        Self {
            max_rank,
            iter,
            u_vectors: Array2::<T>::zeros((0, 0)),
            singular_values: Array1::<T>::zeros(0),
            v_vectors: Array2::<T>::zeros((0, 0)),
            qq: Array2::<T>::zeros((0, 0)),
            verbose: false,
        }
    }

    pub fn matrix_u(&self) -> &Array2<T> {
        &self.u_vectors
    }

    pub fn matrix_v(&self) -> &Array2<T> {
        &self.v_vectors
    }

    pub fn singular_values(&self) -> &Array1<T> {
        &self.singular_values
    }

    pub fn compute(&mut self, xx: &Array2<T>) -> anyhow::Result<()> {
        let nr = xx.nrows();
        let nc = xx.ncols();

        let mut rank = nr.min(nc);
        let mut oversample = 0;

        if self.max_rank > 0 && rank > self.max_rank {
            rank = self.max_rank;
            oversample = 5;
        }

        debug_assert!(rank > 0, "Must be at least rank = 1");

        self.qq = Array2::<T>::zeros((nr, rank + oversample));
        self.rand_subspace_iteration(xx, rank + oversample)?;

        let rank = rank.min(self.qq.ncols());
        self.qq = self.qq.slice(s![.., 0..rank]).to_owned();

        let bb = self.qq.t().dot(xx);

        if self.verbose {
            eprintln!("Final svd on [{} x {}]", bb.nrows(), bb.ncols());
        }

        if let (Some(svd_u), singular_values, Some(svd_vt)) = bb.svd(true, true)? {
            if self.verbose {
                eprintln!("Construct U, D, V");
            }

            self.u_vectors = self.qq.dot(&svd_u.slice(s![.., 0..rank]).to_owned());
            self.v_vectors = svd_vt.t().slice(s![.., 0..rank]).to_owned();
            self.singular_values = singular_values.slice(s![0..rank]).to_owned();
        } else {
            anyhow::bail!("SVD failed");
        }

        if self.verbose {
            eprintln!("Done: RandomizedSVD.compute()");
        }
        Ok(())
    }

    pub fn set_verbose(&mut self) {
        self.verbose = true;
    }

    // Find an orthonormal matrix qq whose range approximates the range of xx
    fn rand_subspace_iteration(
        &mut self,
        xx: &Array2<T>,
        rank_and_oversample: usize,
    ) -> anyhow::Result<()> {
        let nr = xx.nrows();
        let nc = xx.ncols();

        let mut ll = Array2::<T>::zeros((nr, rank_and_oversample));

        let zero = T::from(0.).expect("no zero found");
        let mut qq = Array2::<T>::runif(nc, rank_and_oversample);

        for i in 0..self.iter {
            if self.verbose {
                eprintln!("[Start] LU iteration {:>10}", i + 1);
            }

            let lu1 = xx.dot(&qq);
            ll.fill(zero);
            ll.slice_mut(s![..nr, ..rank_and_oversample]).assign(&lu1);

            let lu2 = xx.t().dot(&ll);
            qq.fill(zero);
            qq.slice_mut(s![..nc, ..rank_and_oversample]).assign(&lu2);

            if self.verbose {
                eprintln!("[Done] LU iteration {:>10}", i + 1);
            }
        }

        let (qr_q, _) = xx.dot(&qq).qr().unwrap();

        let kk = rank_and_oversample.min(qr_q.ncols());
        self.qq = qr_q.slice(s![.., 0..kk]).to_owned();

        if self.verbose {
            eprintln!("Found Q [{} x {}]", self.qq.nrows(), self.qq.ncols());
        }
        Ok(())
    }
}
