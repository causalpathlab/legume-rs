use crate::common_io::{Delimiter, ReadLinesOut};
use candle_core::{Device, Tensor};
use num_traits::Float;

/// Trait for running statistics operations
///
/// Provides a common interface for both dense (ndarray-based) and
/// sparse running statistics implementations.
pub trait RunningStatOps<T>
where
    T: Float,
{
    type Output;

    fn clear(&mut self);
    fn count_positives(&self) -> Self::Output;
    fn sum(&self) -> Self::Output;
    fn mean(&self) -> Self::Output;
    fn variance(&self) -> Self::Output;
    fn std(&self) -> Self::Output;
}

/// some linear algebra routines
pub trait RandomizedAlgs {
    type InMat;
    type OutMat;
    type DVec;
    type Scalar;

    /// randomized singular value decomposition
    /// # input
    /// * `X`: `n x d` matrix
    /// # output
    /// * `U`: `n x k`
    /// * `D`: `k x 1`
    /// * `V`: `d x k`
    fn rsvd(&self, max_rank: usize) -> anyhow::Result<(Self::OutMat, Self::DVec, Self::OutMat)>;
}

/// Convert to and from the vector of triplets
pub trait MatTriplets {
    type Mat;
    type Scalar;

    fn from_nonzero_triplets<I>(
        nrow: usize,
        ncol: usize,
        triplets: &[(I, I, Self::Scalar)],
    ) -> anyhow::Result<Self::Mat>
    where
        I: TryInto<usize> + Copy,
        <I as TryInto<usize>>::Error: std::fmt::Debug;

    fn to_nonzero_triplets(&self) -> anyhow::Result<NRowNColTriplets<Self::Scalar>>;
}

pub struct NRowNColTriplets<Scalar> {
    pub nrow: usize,
    pub ncol: usize,
    pub triplets: Vec<(usize, usize, Scalar)>,
}

/// Reading off from `Tensor`
pub trait ConvertMatOps {
    type Mat;
    type Scalar;

    fn from_tensor(_: &Tensor) -> anyhow::Result<Self::Mat>;
    fn to_tensor(&self, dev: &Device) -> anyhow::Result<Tensor>;
}

/// normalize, sum_to_one, scale, and centre columns
pub trait MatOps {
    type Mat;
    type Scalar;

    /// make each column sum to 1
    fn sum_to_one_columns_inplace(&mut self);
    /// make each column sum to 1
    fn sum_to_one_columns(&self) -> Self::Mat;

    /// make each row sum to 1
    fn sum_to_one_rows_inplace(&mut self);
    /// make each row sum to 1
    fn sum_to_one_rows(&self) -> Self::Mat;

    /// normalize logits after taking exp `(log-sum-exp)`
    fn normalize_exp_logits_columns_inplace(&mut self);
    /// normalize logits after taking exp `(log-sum-exp)`
    fn normalize_exp_logits_columns(&self) -> Self::Mat;

    /// vector norm for each column
    fn normalize_columns_inplace(&mut self);
    /// vector norm for each column
    fn normalize_columns(&self) -> Self::Mat;

    /// standardization for each column
    fn scale_columns_inplace(&mut self);
    /// standardization for each column
    fn scale_columns(&self) -> Self::Mat;

    /// standardization for each row
    fn scale_rows_inplace(&mut self);
    /// standardization for each row
    fn scale_rows(&self) -> Self::Mat;

    /// centering for each column
    fn centre_columns_inplace(&mut self);
    /// centering for each column
    fn centre_columns(&self) -> Self::Mat;
}

pub trait AdjustByDivisionOp<Other, Scalar> {
    /// Adjust each column with the column of the matching batch index
    ///
    /// Assume: `Y[g] ~ Poisson(X[g] * λ)`
    /// (1) Estimate the λ parameter by taking overall ratio, namely,
    /// `λ = Σ Y[g] / Σ X[g]`
    ///
    /// (2) Take the residual (in the log space)
    /// `ln Y[g] - ln (λ X[g])` or `Y[g]/λX[g]` if `X[g] > 0`
    /// otherwise, do nothing
    fn adjust_by_division_of_selected_inplace(&mut self, denom_db: &Other, batches: &[usize]);

    /// adjust each column with the corresponding column of the denom
    ///
    /// Assume: `Y[g] ~ Poisson(X[g] * λ)`
    /// (1) Estimate the λ parameter by taking overall ratio, namely,
    /// `λ = Σ Y[g] / Σ X[g]`
    ///
    /// (2) Take the residual (in the log space)
    /// `ln Y[g] - ln (λ X[g])` or `Y[g]/λX[g]` if `X[g] > 0`
    /// otherwise, do nothing
    fn adjust_by_division_inplace(&mut self, denom: &Other);
}

pub trait MatElemOps {
    type Mat;
    type Scalar;
    fn log1p_inplace(&mut self);
    fn log1p(&self) -> Self::Mat;
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
    /// * `other`: other data matrix
    fn euclidean_distance(
        &self,
        other: &Self::Other,
    ) -> anyhow::Result<Vec<(usize, usize, Self::Scalar)>>;

    /// A vector of Euclidean distances between sources and targets `other`
    ///
    /// * `other`: other data matrix
    /// * `select_columns_in_other`: specific columns
    fn euclidean_distance_on_select_columns(
        &self,
        other: &Self::Other,
        select_columns_in_other: &[usize],
    ) -> anyhow::Result<Vec<(usize, usize, Self::Scalar)>>;
}

pub trait EncodingOps
where
    Self: Sized,
{
    type Mat;
    type Scalar;

    /// Sinusoidal Positional Encoding
    /// * `emb_dim` - embedding dimension, say `d`
    /// * returns each column's embedding results (row x 2d)
    ///
    /// for each element r of each column c:
    ///  ret[r, 2i] = sin(x[r,c]/10000^(2i/d))
    ///  ret[r, 2i + 1] = cos(x[r,c]/10000^(2i/d))
    /// where i in [0, d/2-1]
    fn positional_embedding_columns(&self, emb_dim: usize) -> anyhow::Result<Self::Mat>;
}

/// Operations that involves multiple types
pub trait CompositeOps {
    type Scalar;
    type Mat;
    type Other;

    /// `self[:,col] += other[:,col]`
    /// * `other`: `CscMatrix`
    /// * `col`: column index
    fn add_assign_column(&mut self, other: &Self::Other, col: usize);

    /// `self += other`
    /// * `other`: `CscMatrix`
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

    /// Read the data matrix with row and column names
    ///
    /// * `file_path`: data file name
    /// * `delim`: delimiter (`char` vector or string)
    /// * `header_row`: header line (0-based); `None` will find no header
    /// * `row_name_column_index`: column index (0-based) corresponds to row name
    /// * `select_column_indices`: column indices (0-based) to include
    /// * `select_column_names`: column names to include
    ///
    fn read_data(
        file_path: &str,
        delim: impl Into<Delimiter>,
        header_row: Option<usize>,
        row_name_column_index: Option<usize>,
        select_column_indices: Option<&[usize]>,
        select_column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<MatWithNames<Self::Mat>>;

    /// Read the data matrix with row and column names
    ///
    /// * `file_path`: data file name
    /// * `delim`: delimiter (`char` vector or string)
    /// * `header_row`: header line (0-based); `None` will find no header
    /// * `header_column`: column index (0-based) corresponds to row name
    ///
    fn read_data_with_names(
        file_path: &str,
        delim: impl Into<Delimiter>,
        header_row: Option<usize>,
        header_column: Option<usize>,
    ) -> anyhow::Result<MatWithNames<Self::Mat>> {
        Self::read_data(file_path, delim, header_row, header_column, None, None)
    }

    fn read_data_vec_with_indices_names(
        file_path: &str,
        delim: impl Into<Delimiter>,
        header_line: Option<usize>,
        row_name_index: Option<usize>,
        column_indices: Option<&[usize]>,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Vec<Self::Scalar>)>
    where
        Self::Scalar: std::str::FromStr,
        <Self::Scalar as std::str::FromStr>::Err: std::fmt::Debug,
    {
        let hdr_line = match header_line {
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

    /// Read a `tsv` file while skipping until the header row
    fn from_tsv(tsv_file: &str, skip: Option<usize>) -> anyhow::Result<Self::Mat> {
        Self::read_file_delim(tsv_file, "\t", skip)
    }

    /// Read a `csv` file while skipping until the header row
    fn from_csv(csv_file: &str, skip: Option<usize>) -> anyhow::Result<Self::Mat> {
        Self::read_file_delim(csv_file, ",", skip)
    }

    /// write the matrix down to a file with delimiter
    /// * `file_path`: output file path
    /// * `delim`: separation character or string
    fn write_file_delim(&self, file: &str, delim: &str) -> anyhow::Result<()>;

    /// write the matrix down to a tsv file
    /// * `file_path`: output file path
    fn to_tsv(&self, tsv_file: &str) -> anyhow::Result<()> {
        self.write_file_delim(tsv_file, "\t")
    }

    /// write the matrix down to a csv file
    /// * `file_path`: output file path
    fn to_csv(&self, csv_file: &str) -> anyhow::Result<()> {
        self.write_file_delim(csv_file, ",")
    }

    /// write the matrix down to parquet
    /// * `row_names`: if `None`, just add `[0, n)` numbers.
    /// * `column_names`: if `None`, just add `[0, n)` numbers.
    /// * `file_path`: output file path
    fn to_parquet(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> anyhow::Result<()>;

    /// Read a real-valued numeric matrix with the default row
    /// index(0) and all the other available columns.
    /// Assumes column 0 contains row names.
    ///
    fn from_parquet(file_path: &str) -> anyhow::Result<MatWithNames<Self::Mat>> {
        Self::from_parquet_with_indices(file_path, Some(0), None)
    }

    /// Read a real-valued numeric matrix treating all columns as data.
    /// Row names will be generated as "0", "1", "2", ...
    ///
    fn from_parquet_no_row_names(file_path: &str) -> anyhow::Result<MatWithNames<Self::Mat>> {
        Self::from_parquet_with_indices(file_path, None, None)
    }

    /// Read a real-valued numeric matrix from the parquet file. We
    /// can specify row name index. We can specify the row name column
    /// index and desired column indices.
    /// * `row_name_index`: column index (0-based) corresponds to row name
    fn from_parquet_with_row_names(
        file_path: &str,
        row_name_index: Option<usize>,
    ) -> anyhow::Result<MatWithNames<Self::Mat>> {
        Self::from_parquet_with_indices_names(file_path, row_name_index, None, None)
    }

    /// Read a real-valued numeric matrix from the parquet file. We
    /// can specify row name index. We can specify the row name column
    /// index and desired column indices.
    /// * `row_name_index`: column index (0-based) corresponds to row name
    /// * `column_indices`: column indices (0-based) to include
    fn from_parquet_with_indices(
        file_path: &str,
        row_name_index: Option<usize>,
        column_indices: Option<&[usize]>,
    ) -> anyhow::Result<MatWithNames<Self::Mat>> {
        Self::from_parquet_with_indices_names(file_path, row_name_index, column_indices, None)
    }

    /// Read a real-valued numeric matrix from the parquet file.  We
    /// can specify row name index.  We can specify the row name
    /// column index and desired column names.
    /// * `row_name_index`: column index (0-based) corresponds to row name
    /// * `column_names`: column names to include
    fn from_parquet_with_names(
        file_path: &str,
        row_name_index: Option<usize>,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<MatWithNames<Self::Mat>> {
        Self::from_parquet_with_indices_names(file_path, row_name_index, None, column_names)
    }

    /// Read a real-valued numeric matrix from the parquet file.  We
    /// can specify row name index.  We can specify the row name
    /// column index and desired column indices and names.
    /// * `row_name_index`: column index (0-based) corresponds to row name
    /// * `column_indices`: column indices (0-based) to include
    /// * `column_names`: column names to include
    fn from_parquet_with_indices_names(
        file_path: &str,
        row_name_index: Option<usize>,
        column_indices: Option<&[usize]>,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<MatWithNames<Self::Mat>>;
}

/// intput data matrix `mat` with `rows` and `cols`
pub struct MatWithNames<M> {
    pub rows: Vec<Box<str>>,
    pub cols: Vec<Box<str>>,
    pub mat: M,
}

pub trait MeltOps {
    type Scalar;
    type Mat;
    /// melt a matrix with indices
    fn melt_with_indexes(&self) -> (Vec<Self::Scalar>, Vec<Vec<usize>>);
    /// melt a matrix
    fn melt(&self) -> Vec<Self::Scalar>;
    /// Melt multiple matrices/tensors together in a single traversal for cache efficiency.
    /// All inputs must have the same dimensions.
    /// Returns (values for each input, indices for each dimension).
    fn melt_many_with_indexes(&self, others: &[&Self]) -> (Vec<Vec<Self::Scalar>>, Vec<Vec<usize>>);
}

pub trait CandleDataLoaderOps {
    type Scalar;
    type Mat;
    // /// unify transpose
    // fn transpose(&self) -> Self::Mat;
    /// take each row vector as a sample
    fn rows_to_tensor_vec(&self) -> Vec<Tensor>;
}
