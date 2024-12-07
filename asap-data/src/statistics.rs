use crate::common_io::write_lines;
// use ndarray::prelude::*;
use ndarray::{ArrayBase, Axis, Data, Dimension, OwnedRepr, RemoveAxis};

#[allow(dead_code)]
/// A container to keep track of sufficient statistics of an arbitrary
/// shape `ndarray`
///
/// # Type parameters
/// - `S` : The shape of the array
///
pub struct RunningStatistics<S>
where
    S: Dimension + RemoveAxis,
{
    npos: ArrayBase<OwnedRepr<f32>, S>,
    s0: ArrayBase<OwnedRepr<f32>, S>,
    s1: ArrayBase<OwnedRepr<f32>, S>,
    s2: ArrayBase<OwnedRepr<f32>, S>,
}

#[allow(dead_code)]
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

    /// Save the statistics to a file
    /// # Arguments
    /// * `filename` - The name of the file to save the statistics to
    /// * `names` - The names of the statistics
    pub fn save(&self, filename: &str, names: &Vec<Box<str>>) -> anyhow::Result<()> {
        let nrow = names.len();

        if nrow != self.shape()[0] {
            anyhow::bail!("The number of names must match the number of statistics");
        }

        let nnz_: Vec<Box<str>> = to_string_vec(&self.num_positives());
        let mu_: Vec<Box<str>> = to_string_vec(&self.mean());
        let sig_: Vec<Box<str>> = to_string_vec(&self.std());
        let out: Vec<Box<str>> = (0..nrow)
            .map(|i| format!("{}\t{}\t{}\t{}", names[i], nnz_[i], mu_[i], sig_[i]).into_boxed_str())
            .collect();

        write_lines(&out, filename)?;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.npos.fill(0.0);
        self.s0.fill(0.0);
        self.s1.fill(0.0);
        self.s2.fill(0.0);
    }

    pub fn num_positives(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        self.npos.clone()
    }

    pub fn mean(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        self.s1.clone() / &self.s0.mapv(Self::_add_pseudo_count)
    }

    pub fn variance(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        let mean = self.mean();
        let nn = &self.s0.mapv(Self::_add_pseudo_count);
        let variance = &self.s2 / nn - &mean * &mean;
        variance
    }

    pub fn shape(&self) -> &[usize] {
        self.s0.shape()
    }

    pub fn std(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        self.variance().mapv(f32::sqrt)
    }

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
}

fn to_string_vec<S>(xx: &ArrayBase<OwnedRepr<f32>, S>) -> Vec<Box<str>>
where
    S: Dimension + RemoveAxis,
{
    xx.axis_iter(Axis(0))
        .map(|m| {
            m.iter()
                .map(|v| format!("{:.6}", v))
                .collect::<Vec<String>>()
                .join("\t")
                .clone()
                .into_boxed_str()
        })
        .collect()
}
