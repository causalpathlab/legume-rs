use crate::dmatrix_util::*;
use nalgebra::{DMatrix, DVector};

type Mat = DMatrix<f32>;
type Vec = DVector<f32>;

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
    pub fn new(max_rank: usize, iter: Option<usize>) -> Self {
        Self {
            max_rank,
            iter: iter.unwrap_or(5),
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

    pub fn compute(&mut self, xx: &Mat) {
        let nr = xx.nrows();
        let nc = xx.ncols();

        let mut rank = nr.min(nc);
        let mut oversample = 0;

        if self.max_rank > 0 && self.max_rank < rank {
            rank = self.max_rank;
            oversample = 5;
        }

        debug_assert!(rank > 0, "Must be at least rank = 1");

        self.qq = Mat::zeros(nr, rank + oversample);
        self.rand_subspace_iteration(xx, rank + oversample);

        self.qq = self.qq.columns(0, rank).into_owned();

        let bb = self.qq.transpose() * xx;

        if self.verbose {
            eprintln!("Final svd on [{} x {}]", bb.nrows(), bb.ncols());
        }

        let svd = bb.svd(true, true);

        if self.verbose {
            eprintln!("Construct U, D, V");
        }

        self.u_vectors = self.qq.clone() * svd.u.unwrap().columns(0, rank).into_owned();
        self.v_vectors = svd.v_t.unwrap().transpose().columns(0, rank).into_owned();
        self.singular_values = svd.singular_values.rows(0, rank).into_owned();

        if self.verbose {
            eprintln!("Done: RandomizedSVD.compute()");
        }
    }

    pub fn set_verbose(&mut self) {
        self.verbose = true;
    }

    // Find an orthonormal matrix qq whose range approximates the range of xx
    fn rand_subspace_iteration(&mut self, xx: &Mat, rank_and_oversample: usize) {
        let nr = xx.nrows();
        let nc = xx.ncols();

        let mut ll = Mat::zeros(nr, rank_and_oversample);

        let mut qq = runif(nc, rank_and_oversample);

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
        self.qq = qr.q().columns(0, rank_and_oversample).into_owned();

        if self.verbose {
            eprintln!("Found Q [{} x {}]", self.qq.nrows(), self.qq.ncols());
        }
    }
}
