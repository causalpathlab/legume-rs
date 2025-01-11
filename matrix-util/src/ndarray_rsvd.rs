extern crate ndarray;
extern crate ndarray_linalg;
use crate::traits::SampleOps;
use ndarray::{s, Array1, Array2};
use ndarray_linalg::{QR, SVD};

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

/// Randomized SVD
///
/// Implement Alg 4.4 of Halko et al. (2009)
/// Modified from https://github.com/kazuotani14/RandomizedSvd
///
pub struct RandomizedSVD {
    max_rank: usize,
    iter: usize,
    u_vectors: Mat,
    singular_values: Vec,
    v_vectors: Mat,
    qq: Mat,
    verbose: bool,
}

impl RandomizedSVD {
    pub fn new(max_rank: usize, iter: usize) -> Self {
        Self {
            max_rank,
            iter,
            u_vectors: Mat::zeros((0, 0)),
            singular_values: Vec::zeros(0),
            v_vectors: Mat::zeros((0, 0)),
            qq: Mat::zeros((0, 0)),
            verbose: false,
        }
    }

    pub fn matrix_u(&self) -> &Mat {
        &self.u_vectors
    }

    pub fn matrix_v(&self) -> &Mat {
        &self.v_vectors
    }

    pub fn singular_values(&self) -> &Vec {
        &self.singular_values
    }

    pub fn compute(&mut self, xx: &Mat) -> anyhow::Result<()> {
        let nr = xx.nrows();
        let nc = xx.ncols();

        let mut rank = nr.min(nc);
        let mut oversample = 0;

        if self.max_rank > 0 && rank > self.max_rank {
            rank = self.max_rank;
            oversample = 5;
        }

        debug_assert!(rank > 0, "Must be at least rank = 1");

        self.qq = Mat::zeros((nr, rank + oversample));
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
        xx: &Mat,
        rank_and_oversample: usize,
    ) -> anyhow::Result<()> {
        let nr = xx.nrows();
        let nc = xx.ncols();

        let mut ll = Mat::zeros((nr, rank_and_oversample));

        let mut qq = Mat::runif(nc, rank_and_oversample);

        for i in 0..self.iter {
            if self.verbose {
                eprintln!("[Start] LU iteration {:>10}", i + 1);
            }

            let lu1 = xx.dot(&qq);
            ll.fill(0.0);
            ll.slice_mut(s![..nr, ..rank_and_oversample]).assign(&lu1);

            let lu2 = xx.t().dot(&ll);
            qq.fill(0.0);
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
