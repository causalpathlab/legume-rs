use crate::common_io::{Delimiter, ReadLinesOut};

pub use candle_util::candle_core;

use candle_core::{Device, Tensor};

/// some linear algebra routines
pub trait RandomizedAlgs {
    type InMat;
    type OutMat;
    type DVec;
    type Scalar;

    /// randomized singular value decomposition
    /// # input
    /// * `X` - `n x d` matrix
    /// # output
    /// * `U` - `n x k`
    /// * `D` - `k x 1`
    /// * `V` - `d x k`
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

    /// A vector of Euclidean distances between sources and targets `other`
    ///
    /// * `other` - other data matrix
    fn euclidean_distance(
        &self,
        other: &Self::Other,
    ) -> anyhow::Result<Vec<(usize, usize, Self::Scalar)>>;

    /// A vector of Euclidean distances between sources and targets `other`
    ///
    /// * `other` - other data matrix
    /// * `select_columns_in_other` - specific columns
    fn euclidean_distance_on_select_columns(
        &self,
        other: &Self::Other,
        select_columns_in_other: &[usize],
    ) -> anyhow::Result<Vec<(usize, usize, Self::Scalar)>>;
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
        file_path: &str,
        delim: impl Into<Delimiter>,
        skip: Option<usize>,
    ) -> anyhow::Result<Self::Mat>;

    ///
    /// * `file_path` - data file name
    /// * `skip` - header line (0-based)
    /// * `row_name_index` - column index (0-based) corresponds to row name
    /// * `column_indices` - column indices (0-based) to include
    /// * `column_names` - column names to include
    ///
    fn read_data(
        file_path: &str,
        delim: impl Into<Delimiter>,
        skip: Option<usize>,
        row_name_index: Option<usize>,
        column_indices: Option<&[usize]>,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Self::Mat)>;

    fn read_names_and_data_with_indices_names(
        file_path: &str,
        delim: impl Into<Delimiter>,
        skip: Option<usize>,
        row_name_index: Option<usize>,
        column_indices: Option<&[usize]>,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Vec<Self::Scalar>)>
    where
        Self::Scalar: std::str::FromStr,
        <Self::Scalar as std::str::FromStr>::Err: std::fmt::Debug,
    {
        let hdr_line = match skip {
            Some(skip) => skip as i64,
            None => -1, // no skipping
        };

        let ReadLinesOut { lines, header } =
            crate::common_io::read_lines_of_words_delim(file_path, delim, hdr_line)?;

        let mut relevant_indices: Vec<usize> = vec![];

        if let Some(indices) = column_indices {
            relevant_indices.extend(indices.iter().copied());
        }

        if let Some(names) = column_names {
            let name_indices: Vec<usize> = header
                .iter()
                .enumerate()
                .filter_map(|(i, name)| {
                    if names.iter().any(|n| n == name) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            relevant_indices.extend(name_indices);
        }
        relevant_indices.sort_unstable();
        relevant_indices.dedup();

        let row_names: Vec<Box<str>> = match row_name_index {
            Some(row_name_index) => lines
                .iter()
                .map(|words| words[row_name_index].clone())
                .collect(),
            _ => (0..lines.len())
                .map(|x| x.to_string().into_boxed_str())
                .collect(),
        };

        let column_names: Vec<Box<str>> = if header.is_empty() {
            relevant_indices
                .iter()
                .map(|x| x.to_string().into_boxed_str())
                .collect()
        } else {
            relevant_indices
                .iter()
                .map(|&j| header[j].clone())
                .collect()
        };

        let data: Vec<Vec<Self::Scalar>> = lines
            .iter()
            .map(|words| {
                relevant_indices
                    .iter()
                    .map(|&i| words[i].parse::<Self::Scalar>().expect("failed to parse"))
                    .collect()
            })
            .collect();

        let data = data.into_iter().flatten().collect::<Vec<_>>();

        Ok((row_names, column_names, data))
    }

    fn from_tsv(tsv_file: &str, skip: Option<usize>) -> anyhow::Result<Self::Mat> {
        Self::read_file_delim(tsv_file, "\t", skip)
    }

    fn from_csv(csv_file: &str, skip: Option<usize>) -> anyhow::Result<Self::Mat> {
        Self::read_file_delim(csv_file, ",", skip)
    }

    fn write_file_delim(&self, file: &str, delim: &str) -> anyhow::Result<()>;

    fn to_tsv(&self, tsv_file: &str) -> anyhow::Result<()> {
        self.write_file_delim(tsv_file, "\t")
    }

    fn to_csv(&self, csv_file: &str) -> anyhow::Result<()> {
        self.write_file_delim(csv_file, ",")
    }

    fn to_parquet(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> anyhow::Result<()>;

    /// Read a real-valued numeric matrix with the default row
    /// index(0) and all the other available columns
    ///
    fn from_parquet(file_path: &str) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Self::Mat)> {
        Self::from_parquet_with_indices(file_path, None, None)
    }

    /// Read a real-valued numeric matrix from the parquet file. We
    /// can specify row name index. We can specify the row name column
    /// index and desired column indices.
    /// * `row_name_index` - column index (0-based) corresponds to row name
    /// * `column_indices` - column indices (0-based) to include
    fn from_parquet_with_indices(
        file_path: &str,
        row_name_index: Option<usize>,
        column_indices: Option<&[usize]>,
    ) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Self::Mat)> {
        Self::from_parquet_with_indices_names(file_path, row_name_index, column_indices, None)
    }

    /// Read a real-valued numeric matrix from the parquet file.  We
    /// can specify row name index.  We can specify the row name
    /// column index and desired column names.
    /// * `row_name_index` - column index (0-based) corresponds to row name
    /// * `column_names` - column names to include
    fn from_parquet_with_names(
        file_path: &str,
        row_name_index: Option<usize>,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Self::Mat)> {
        Self::from_parquet_with_indices_names(file_path, row_name_index, None, column_names)
    }

    /// Read a real-valued numeric matrix from the parquet file.  We
    /// can specify row name index.  We can specify the row name
    /// column index and desired column indices and names.
    /// * `row_name_index` - column index (0-based) corresponds to row name
    /// * `column_indices` - column indices (0-based) to include
    /// * `column_names` - column names to include
    fn from_parquet_with_indices_names(
        file_path: &str,
        row_name_index: Option<usize>,
        column_indices: Option<&[usize]>,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Self::Mat)>;
}

/// melt a matrix
pub trait MeltOps {
    type Scalar;
    type Mat;
    fn melt_with_indexes(&self) -> (Vec<Self::Scalar>, Vec<Vec<usize>>);
    fn melt(&self) -> Vec<Self::Scalar>;
}
