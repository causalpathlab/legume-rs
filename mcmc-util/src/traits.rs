pub trait LikelihoodSampler {
    type Scalar;

    /// `f' <- f cos θ + ν sin θ`
    /// where `f` is a current value
    ///       `ν` is sampled from prior
    fn sample<LogLikelihood>(&self, llik: &LogLikelihood, current: &Self)
    where
        LogLikelihood: Fn(&Self, &Self) -> Self::Scalar;
}

// impl LikelihoodSampler for Mat {
//     type Scalar = f32;
//     fn sample<LogLikelihoodFn>(&self, llik_fn: &LogLikelihoodFn, current: &Self)
//     where
//         LogLikelihoodFn: Fn(&Self, &Self) -> Self::Scalar,
//     {
//         let llik = llik_fn(&self, current);

//         use std::f32::consts::PI;

//         let mut rng = rand::rng();

//         let theta = rng.random_range(0.0..2. * PI);

//         let prior = Mat::rnorm(current.nrows(), current.ncols());

//         let proposal = current * theta.cos() + prior * theta.sin();
//     }
// }
