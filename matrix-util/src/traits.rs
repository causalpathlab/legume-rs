use candle_core::Tensor;

/// Convert to and from triplets
pub trait MatTriplets {
    type Mat;
    type Scalar;

    fn from_nonzero_triplets(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(usize, usize, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat>;

    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)>;
}

/// Normalize or scale columns
pub trait ConvertMatOps {
    type Mat;
    type Scalar;

    fn from_tensor(_: &Tensor) -> anyhow::Result<Self::Mat>;
}

/// Normalize or scale columns
pub trait MatOps {
    type Mat;
    type Scalar;

    fn normalize_columns_inplace(&mut self);
    fn normalize_columns(&self) -> Self::Mat;
    fn scale_columns_inplace(&mut self);
    fn scale_columns(&self) -> Self::Mat;
    fn centre_columns_inplace(&mut self);
    fn centre_columns(&self) -> Self::Mat;
}

/// Operations to sample random matrices, only works for
/// `nalgebra::DMatrix` and `ndarray::Array2`
pub trait SampleOps {
    type Mat;
    type Scalar;

    /// Sample a matrix from a uniform distribution `U(0,1)`
    fn runif(dd: usize, nn: usize) -> Self::Mat;

    /// Sample a matrix from a normal distribution `N(0,1)`
    fn rnorm(dd: usize, nn: usize) -> Self::Mat;

    /// Sample a matrix from a gamma distribution with `param` is
    /// `(shape α, scale θ)`
    ///
    /// $$f(x|\alpha,\theta) = \frac{\theta^{-\alpha}}{\Gamma(\alpha)} x^{\alpha - 1} e^{-x/\theta}$$
    ///
    /// Note: `rate = 1/scale` or $\beta = 1/\theta$
    fn rgamma(dd: usize, nn: usize, param: (f32, f32)) -> Self::Mat;
}

/// Operations that involves multiple types
pub trait CompositeOps {
    type Scalar;
    type Mat;
    type Other;

    /// `self[:,col] += other[:,col]`
    /// * `other` - `CscMatrix`
    /// * `col` - column index
    fn add_assign_column(&mut self, other: &Self::Other, col: usize);

    /// `self += other`
    /// * `other` - `CscMatrix`
    fn add_assign(&mut self, other: &Self::Other);
}

/// Read and write matrices from and to files
pub trait IoOps {
    type Scalar;
    type Mat;

    fn from_tsv(tsv_file: &str, skip: Option<usize>) -> anyhow::Result<Self::Mat>;
    fn to_tsv(&self, tsv_file: &str) -> anyhow::Result<()>;
}
