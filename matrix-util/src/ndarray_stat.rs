#![allow(dead_code)]

use crate::{
    common_io::{file_ext, write_lines},
    traits::{IoOps, RunningStatOps},
};
use ndarray::{stack, ArrayBase, Axis, Data, Dimension, NdIndex, OwnedRepr, RemoveAxis};

/// A container to keep track of sufficient statistics of an arbitrary
/// shape `ndarray`
///
/// # Type parameters
/// - `S` : The shape of the array
///
#[derive(Clone)]
pub struct RunningStatistics<S>
where
    S: Dimension + RemoveAxis,
{
    npos: ArrayBase<OwnedRepr<f32>, S>,
    s0: ArrayBase<OwnedRepr<f32>, S>,
    s1: ArrayBase<OwnedRepr<f32>, S>,
    s2: ArrayBase<OwnedRepr<f32>, S>,
}

impl<S> RunningStatistics<S>
where
    S: Dimension + RemoveAxis,
{
    /// Create a new RunningStatistics object
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the array
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix_util::ndarray_stat::RunningStatistics;
    /// use ndarray::Ix1;
    /// let nrow = 10;
    /// RunningStatistics::new(Ix1(nrow));
    /// ```
    ///
    pub fn new(shape: S) -> Self {
        let npos = ArrayBase::zeros(shape.clone());
        let s0 = ArrayBase::zeros(shape.clone());
        let s1 = ArrayBase::zeros(shape.clone());
        let s2 = ArrayBase::zeros(shape);

        RunningStatistics { npos, s0, s1, s2 }
    }

    pub fn add<V>(&mut self, xx: &ArrayBase<V, S>)
    where
        V: Data<Elem = f32>,
    {
        self.npos += &xx.mapv(Self::_is_positive);
        self.s0 += &xx.mapv(Self::_is_finite);
        self.s1 += &xx.mapv(Self::_finite);
        self.s2 += &xx.mapv(Self::_finite).mapv(|v| v * v);
    }

    pub fn add_element<I>(&mut self, idx: &I, val: f32)
    where
        I: NdIndex<S> + Clone,
    {
        fn get<'a, S, I>(mat: &'a mut ArrayBase<OwnedRepr<f32>, S>, idx: &'a I) -> &'a mut f32
        where
            S: Dimension + RemoveAxis,
            I: NdIndex<S> + Clone,
        {
            mat.get_mut(idx.clone()).expect("failed to access matrix")
        }

        let idx_clone = idx.clone();

        *get(&mut self.npos, &idx_clone) += Self::_is_positive(val);
        *get(&mut self.s0, &idx_clone) += Self::_is_finite(val);
        let safe_val = Self::_finite(val);
        *get(&mut self.s1, &idx_clone) += safe_val;
        *get(&mut self.s2, &idx_clone) += safe_val * safe_val;
    }

    pub fn clear(&mut self) {
        self.npos.fill(0.0);
        self.s0.fill(0.0);
        self.s1.fill(0.0);
        self.s2.fill(0.0);
    }

    /// Frequency of positive values. For a sparse count matrix, this
    /// will reflect the number of non-zero values
    ///
    pub fn count_positives(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        self.npos.clone()
    }

    /// Sum of values
    pub fn sum(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        self.s1.clone()
    }

    /// Average statistic
    pub fn mean(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        self.s1.clone() / &self.s0.mapv(Self::_add_pseudo_count)
    }

    /// Variance
    pub fn variance(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        let mean = self.mean();
        let nn = &self.s0.mapv(Self::_add_pseudo_count);

        &self.s2 / nn - &mean * &mean
    }

    /// Standard deviation
    pub fn std(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        self.variance().mapv(f32::sqrt)
    }

    pub fn shape(&self) -> &[usize] {
        self.s0.shape()
    }

    //////////////////////
    // helper functions //
    //////////////////////

    fn _finite(x: f32) -> f32 {
        if x.is_finite() {
            x
        } else {
            0_f32
        }
    }

    fn _is_finite(x: f32) -> f32 {
        if x.is_finite() {
            1_f32
        } else {
            0_f32
        }
    }

    fn _is_positive(x: f32) -> f32 {
        if x.is_finite() && x > 0_f32 {
            1_f32
        } else {
            0_f32
        }
    }

    fn _add_pseudo_count(x: f32) -> f32 {
        x + 1e-8
    }

    /// Save the statistics to a file
    /// # Arguments
    /// * `filename` - The name of the file to save the statistics to
    /// * `names` - The names of the statistics
    /// * `sep` - Separator for text formats
    /// * `row_column_name` - Name for the row column in parquet format (defaults to "stat")
    pub fn save(
        &self,
        filename: &str,
        names: &[Box<str>],
        sep: &str,
        row_column_name: Option<&str>,
    ) -> anyhow::Result<()> {
        match file_ext(filename).unwrap_or(Box::from("")).as_ref() {
            "parquet" => {
                let nnz = &self.count_positives();
                let tot = &self.s1;
                let mu = &self.mean();
                let sig = &self.std();

                let n = nnz.len();
                let nnz_col = nnz.clone().into_shape_with_order((n,)).unwrap();
                let tot_col = tot.clone().into_shape_with_order((n,)).unwrap();
                let mu_col = mu.clone().into_shape_with_order((n,)).unwrap();
                let sig_col = sig.clone().into_shape_with_order((n,)).unwrap();

                let stacked = stack(
                    Axis(1),
                    &[
                        nnz_col.view(),
                        tot_col.view(),
                        mu_col.view(),
                        sig_col.view(),
                    ],
                )
                .unwrap();

                let column_names: Vec<Box<str>> = vec!["nnz", "tot", "mu", "sig"]
                    .into_iter()
                    .map(|s| s.into())
                    .collect();

                let row_col = row_column_name.or(Some("stat"));
                stacked.to_parquet_with_names(
                    filename,
                    (Some(names), row_col),
                    Some(&column_names),
                )?;
            }
            _ => {
                let mut out = self.to_string_vec(names, sep)?;
                let header = format!("#name{}nnz{}tot{}mu{}sig", sep, sep, sep, sep);
                out.insert(0, header.into_boxed_str());
                write_lines(&out, filename)?;
            }
        };

        Ok(())
    }

    pub fn to_string_vec(&self, names: &[Box<str>], sep: &str) -> anyhow::Result<Vec<Box<str>>> {
        if names.len() != self.shape()[0] {
            anyhow::bail!(
                "The number of names does not match the number of the first dimension of the statistics"
            );
        }

        let nnz_: Vec<Box<str>> = to_string_vec(&self.count_positives(), sep);
        let tot_ = to_string_vec(&self.s1, sep);
        let mu_: Vec<Box<str>> = to_string_vec(&self.mean(), sep);
        let sig_: Vec<Box<str>> = to_string_vec(&self.std(), sep);

        let out: Vec<Box<str>> = (0..self.shape()[0])
            .map(|i| {
                format!(
                    "{}{}{}{}{}{}{}{}{}",
                    names[i], sep, nnz_[i], sep, tot_[i], sep, mu_[i], sep, sig_[i]
                )
                .into_boxed_str()
            })
            .collect();
        Ok(out)
    }
}

impl<S> RunningStatOps<f32> for RunningStatistics<S>
where
    S: Dimension + RemoveAxis,
{
    type Output = ArrayBase<OwnedRepr<f32>, S>;

    fn clear(&mut self) {
        self.npos.fill(0.0);
        self.s0.fill(0.0);
        self.s1.fill(0.0);
        self.s2.fill(0.0);
    }

    fn count_positives(&self) -> Self::Output {
        self.npos.clone()
    }

    fn sum(&self) -> Self::Output {
        self.s1.clone()
    }

    fn mean(&self) -> Self::Output {
        self.s1.clone() / &self.s0.mapv(Self::_add_pseudo_count)
    }

    fn variance(&self) -> Self::Output {
        let mean = <Self as RunningStatOps<f32>>::mean(self);
        let nn = &self.s0.mapv(Self::_add_pseudo_count);
        &self.s2 / nn - &mean * &mean
    }

    fn std(&self) -> Self::Output {
        <Self as RunningStatOps<f32>>::variance(self).mapv(f32::sqrt)
    }
}

fn to_string_vec<S>(xx: &ArrayBase<OwnedRepr<f32>, S>, sep: &str) -> Vec<Box<str>>
where
    S: Dimension + RemoveAxis,
{
    xx.axis_iter(Axis(0))
        .map(|m| {
            m.iter()
                .map(|v| {
                    if *v > 1e-4 {
                        format!("{:.4}", v)
                            .trim_end_matches('0')
                            .trim_end_matches('.')
                            .to_string()
                    } else if *v > 1e-20 {
                        format!("{:.4e}", v)
                    } else {
                        "0".to_string()
                    }
                })
                .collect::<Vec<String>>()
                .join(sep)
                .clone()
                .into_boxed_str()
        })
        .collect()
}
