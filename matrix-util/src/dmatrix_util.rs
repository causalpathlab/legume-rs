pub use nalgebra::DMatrix;
pub use rand::{thread_rng, Rng};
pub use rand_distr::{StandardNormal, Uniform};
pub use rayon::prelude::*;

type Mat = DMatrix<f32>;

#[allow(dead_code)]
/// Sample d,n matrix from U(0,1)
pub fn runif(dd: usize, nn: usize) -> Mat {
    let runif = Uniform::new(0_f32, 1_f32);

    let rvec = (0..(dd * nn))
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.sample(runif))
        .collect();

    Mat::from_vec(dd, nn, rvec)
}

#[allow(dead_code)]
/// Sample d,n matrix from N(0,1)
pub fn rnorm(dd: usize, nn: usize) -> Mat {
    let rvec = (0..(dd * nn))
        .into_par_iter()
        .map_init(thread_rng, |rng, _| rng.sample(StandardNormal))
        .collect();

    Mat::from_vec(dd, nn, rvec)
}

#[allow(dead_code)]
/// Normalize d x m matrix X by columns
/// Y[,j] = X[,j] / max(1, norm(X[,j]))
pub fn normalize_columns(xx: &Mat) -> Mat {
    let mut ret = xx.clone();
    for mut xx_j in ret.column_iter_mut() {
        let denom = xx_j.norm().max(1.0);
        xx_j /= denom;
    }
    ret
}
