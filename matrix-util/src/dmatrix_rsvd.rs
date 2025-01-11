use crate::traits::SampleOps;
use nalgebra::{DMatrix, DVector};

type Mat = DMatrix<f32>;
type Vec = DVector<f32>;

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
            u_vectors: Mat::zeros(0, 0),
            singular_values: Vec::zeros(0),
            v_vectors: Mat::zeros(0, 0),
            qq: Mat::zeros(0, 0),
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

        self.qq = Mat::zeros(nr, rank + oversample);
        self.rand_subspace_iteration(xx, rank + oversample);

        let rank = rank.min(self.qq.ncols());
        self.qq = self.qq.columns(0, rank).into_owned();

        let bb = self.qq.transpose() * xx;

        if self.verbose {
            eprintln!("Final svd on [{} x {}]", bb.nrows(), bb.ncols());
        }

        let svd = bb.svd(true, true);

        if let (Some(svd_u), Some(svd_vt)) = (svd.u, svd.v_t) {
            if self.verbose {
                eprintln!("Construct U, D, V");
            }

            self.u_vectors = self.qq.clone() * svd_u.columns(0, rank).into_owned();
            self.v_vectors = svd_vt.transpose().columns(0, rank).into_owned();
            self.singular_values = svd.singular_values.rows(0, rank).into_owned();
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
    fn rand_subspace_iteration(&mut self, xx: &Mat, rank_and_oversample: usize) {
        let nr = xx.nrows();
        let nc = xx.ncols();

        let mut ll = Mat::zeros(nr, rank_and_oversample);

        let mut qq = Mat::runif(nc, rank_and_oversample);

        for i in 0..self.iter {
            if self.verbose {
                eprintln!("[Start] LU iteration {:>10}", i + 1);
            }

            let lu1 = xx * &qq;
            ll.fill(0.);
            ll.fill_with_identity();
            ll.view_mut((0, 0), (nr, rank_and_oversample))
                .lower_triangle()
                .copy_from(&lu1);

            let lu2 = xx.transpose() * &ll;
            qq.fill(0.);
            qq.fill_with_identity();
            qq.view_mut((0, 0), (nc, rank_and_oversample))
                .lower_triangle()
                .copy_from(&lu2);

            if self.verbose {
                eprintln!("[Done] LU iteration {:>10}", i + 1);
            }
        }

        let qr = (xx * &qq).qr();
        let kk = rank_and_oversample.min(qr.q().ncols());
        self.qq = qr.q().columns(0, kk).into_owned();

        if self.verbose {
            eprintln!("Found Q [{} x {}]", self.qq.nrows(), self.qq.ncols());
        }
    }
}
