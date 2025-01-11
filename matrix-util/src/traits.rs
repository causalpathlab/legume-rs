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

pub trait MatOps {
    type Mat;
    type Scalar;

    fn normalize_columns_inplace(&mut self);
    fn normalize_columns(&mut self) -> Self::Mat;
    fn scale_columns_inplace(&mut self);
}

pub trait SampleOps {
    type Mat;
    type Scalar;

    fn runif(dd: usize, nn: usize) -> Self::Mat;
    fn rnorm(dd: usize, nn: usize) -> Self::Mat;
}
