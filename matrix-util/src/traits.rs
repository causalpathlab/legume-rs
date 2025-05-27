use crate::common_io::Delimiter;
use candle_core::Device;
use candle_core::Tensor;

/// some linear algebra routines
pub trait RandomizedAlgs {
    type InMat;
    type OutMat;
    type DVec;
    type Scalar;

    fn rsvd(&self, max_rank: usize) -> anyhow::Result<(Self::OutMat, Self::DVec, Self::OutMat)>;
}

/// Convert to and from the vector of triplets
pub trait MatTriplets {
    type Mat;
    type Scalar;

    fn from_nonzero_triplets<I>(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(I, I, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat>
    where
        I: TryInto<usize> + Copy,
        <I as TryInto<usize>>::Error: std::fmt::Debug;

    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)>;
}

/// Reading off from `Tensor`
pub trait ConvertMatOps {
    type Mat;
    type Scalar;

    fn from_tensor(_: &Tensor) -> anyhow::Result<Self::Mat>;
    fn to_tensor(&self, dev: &Device) -> anyhow::Result<Tensor>;
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

pub trait DistanceOps {
    type Scalar;
    type Other;

    /// A vector of Euclidean distances between sources and targets
    fn euclidean_distance_matched_columns(
        &self,
        target_mat: &Self::Other,
        target_columns: &Vec<usize>,
    ) -> anyhow::Result<Vec<Self::Scalar>>;
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

    fn read_file_delim(
        file: &str,
        delim: impl Into<Delimiter>,
        skip: Option<usize>,
    ) -> anyhow::Result<Self::Mat>;

    fn from_tsv(tsv_file: &str, skip: Option<usize>) -> anyhow::Result<Self::Mat> {
        Self::read_file_delim(tsv_file, "\t", skip)
    }

    fn write_file_delim(&self, file: &str, delim: &str) -> anyhow::Result<()>;

    fn to_tsv(&self, tsv_file: &str) -> anyhow::Result<()> {
        self.write_file_delim(tsv_file, "\t")
    }

    fn to_csv(&self, csv_file: &str) -> anyhow::Result<()> {
        self.write_file_delim(csv_file, ",")
    }
}
