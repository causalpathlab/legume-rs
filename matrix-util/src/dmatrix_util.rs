use nalgebra::{ComplexField, DMatrix, Matrix};
use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix, csr::CsrMatrix};

use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Gamma, StandardNormal, Uniform};
use rayon::prelude::*;

use crate::traits::*;

pub fn subset_columns<T, D, S>(
    matrix: &Matrix<T, nalgebra::Dyn, D, S>,
    indices: &[usize],
) -> anyhow::Result<DMatrix<T>>
where
    T: nalgebra::RealField,
    D: nalgebra::Dim,
    S: nalgebra::RawStorage<T, nalgebra::Dyn, D>,
{
    let cols = indices
        .into_iter()
        .map(|&j| matrix.column(j))
        .collect::<Vec<_>>();

    concatenate_horizontal(&cols)
}

pub fn assign_columns<T, D, S, R>(
    source: &Matrix<T, nalgebra::Dyn, D, S>,
    indices: &[usize],
    target: &mut Matrix<T, nalgebra::Dyn, D, R>,
) where
    T: nalgebra::RealField,
    D: nalgebra::Dim,
    S: nalgebra::RawStorage<T, nalgebra::Dyn, D>,
    R: nalgebra::RawStorageMut<T, nalgebra::Dyn, D>,
{
    indices
        .iter()
        .zip(source.column_iter())
        .for_each(|(&j, x_j)| {
            target.column_mut(j).copy_from(&x_j);
        });
}

/// concatenate matrices or row vectors vertically
pub fn concatenate_vertical<T, D, S>(
    matrices: &[Matrix<T, D, nalgebra::Dyn, S>],
) -> anyhow::Result<DMatrix<T>>
where
    T: nalgebra::RealField,
    D: nalgebra::Dim,
    S: nalgebra::RawStorage<T, D, nalgebra::Dyn>,
{
    if matrices.is_empty() {
        anyhow::bail!("empty in concatenate_vertical");
    }

    let ncols = matrices[0].ncols();
    assert!(
        matrices.iter().all(|m| m.ncols() == ncols),
        "should have the same number of columns"
    );

    let rows = matrices
        .iter()
        .flat_map(|m| m.row_iter().map(|row| row.into_owned()))
        .collect::<Vec<_>>();

    Ok(DMatrix::from_rows(&rows))
}

/// concatenate matrices or column vectors horizontally
pub fn concatenate_horizontal<T, D, S>(
    matrices: &[Matrix<T, nalgebra::Dyn, D, S>],
) -> anyhow::Result<DMatrix<T>>
where
    T: nalgebra::RealField,
    D: nalgebra::Dim,
    S: nalgebra::RawStorage<T, nalgebra::Dyn, D>,
{
    if matrices.is_empty() {
        anyhow::bail!("empty in concatenate_horizontal");
    }

    let nrows = matrices[0].nrows();
    assert!(
        matrices.iter().all(|m| m.nrows() == nrows),
        "should have the same number of rows"
    );

    let cols = matrices
        .iter()
        .flat_map(|m| m.column_iter().map(|col| col.into_owned()))
        .collect::<Vec<_>>();

    Ok(DMatrix::from_columns(&cols))
}

impl<T> AdjustByDivisionOp<nalgebra_sparse::CscMatrix<T>, T> for nalgebra_sparse::CscMatrix<T>
where
    T: nalgebra::RealField + Copy + std::iter::Sum<T>,
{
    fn adjust_by_division_inplace(&mut self, denom: &nalgebra_sparse::CscMatrix<T>) {
        self.col_iter_mut()
            .zip(denom.col_iter())
            .for_each(|(mut x_j, d_j)| {
                let dsum = d_j.values().iter().copied().sum::<T>();
                let xsum = x_j.values().iter().copied().sum::<T>();
                let scale = if dsum > T::zero() {
                    xsum / dsum
                } else {
                    T::one()
                };

                let (x_rows, x_values) = x_j.rows_and_values_mut();

                let mut d_j_values = vec![T::zero(); x_rows.len()];

                x_rows.iter().enumerate().for_each(|(idx, &i)| {
                    if let Some(pos) = d_j.row_indices().iter().position(|&d_i| d_i == i) {
                        d_j_values[idx] = d_j.values()[pos];
                    }
                });

                x_values
                    .iter_mut()
                    .zip(d_j_values)
                    .for_each(|(x_ij, d_ij)| {
                        if d_ij > T::zero() {
                            *x_ij /= d_ij * scale;
                        }
                    });
            });
    }

    fn adjust_by_division_of_selected_inplace(
        &mut self,
        denom_db: &nalgebra_sparse::CscMatrix<T>,
        batches: &[usize],
    ) {
        self.col_iter_mut().zip(batches).for_each(|(mut x_j, &b)| {
            let d_j = denom_db.col(b);

            let dsum = d_j.values().iter().copied().sum::<T>();
            let xsum = x_j.values().iter().copied().sum::<T>();
            let scale = if dsum > T::zero() {
                xsum / dsum
            } else {
                T::one()
            };

            let (x_rows, x_values) = x_j.rows_and_values_mut();

            let mut d_j_values = vec![T::zero(); x_rows.len()];

            x_rows.iter().enumerate().for_each(|(idx, &i)| {
                if let Some(pos) = d_j.row_indices().iter().position(|&d_i| d_i == i) {
                    d_j_values[idx] = d_j.values()[pos];
                }
            });

            x_values
                .iter_mut()
                .zip(d_j_values)
                .for_each(|(x_ij, d_ij)| {
                    if d_ij > T::zero() {
                        *x_ij /= d_ij * scale;
                    }
                });
        });
    }
}

impl<T> AdjustByDivisionOp<nalgebra::DMatrix<T>, T> for nalgebra_sparse::CscMatrix<T>
where
    T: nalgebra::RealField + Copy + std::iter::Sum<T>,
{
    fn adjust_by_division_of_selected_inplace(
        &mut self,
        denom_db: &nalgebra::DMatrix<T>,
        batches: &[usize],
    ) {
        self.col_iter_mut().zip(batches).for_each(|(mut x_j, &b)| {
            let d_j = x_j
                .row_indices()
                .iter()
                .map(|&i| denom_db[(i, b)])
                .collect::<Vec<_>>();

            let dsum = d_j.iter().map(|&x| x).sum::<T>();
            let xsum = x_j.values().iter().map(|&x| x).sum::<T>();
            let scale = if dsum > T::zero() {
                xsum / dsum
            } else {
                T::one()
            };

            x_j.values_mut()
                .iter_mut()
                .zip(d_j)
                .for_each(|(x_ij, d_ij)| {
                    if d_ij > T::zero() {
                        *x_ij /= d_ij * scale;
                    }
                });
        });
    }

    fn adjust_by_division_inplace(&mut self, denom: &nalgebra::DMatrix<T>) {
        self.col_iter_mut()
            .zip(denom.column_iter())
            .for_each(|(mut x_j, d_j)| {
                let d_j = x_j
                    .row_indices()
                    .iter()
                    .map(|&i| d_j[i])
                    .collect::<Vec<_>>();

                let dsum = d_j.iter().map(|&x| x).sum::<T>();
                let xsum = x_j.values().iter().map(|&x| x).sum::<T>();
                let scale = if dsum > T::zero() {
                    xsum / dsum
                } else {
                    T::one()
                };

                x_j.values_mut()
                    .iter_mut()
                    .zip(d_j)
                    .for_each(|(x_ij, d_ij)| {
                        if d_ij > T::zero() {
                            *x_ij /= d_ij * scale;
                        }
                    });
            });
    }
}

/// Generate one-hot membership matrix (`row x K`) where the number of
/// rows corresponds to the length of the membership vector and the
/// `K` corresponds to the maximum membership value + 1.
///
pub fn row_membership_matrix<T>(row_membership: Vec<usize>) -> anyhow::Result<DMatrix<T>>
where
    T: nalgebra::RealField + Copy,
{
    let kk = match row_membership.iter().max() {
        Some(&m) => m + 1,
        _ => 1,
    };

    let mut ret_dm = DMatrix::zeros(row_membership.len(), kk);
    let oneval = T::from_f32(1.).ok_or(anyhow::anyhow!("cannot find 1 value"))?;
    for (i, k) in row_membership.into_iter().enumerate() {
        ret_dm[(i, k)] += oneval;
    }

    Ok(ret_dm)
}

fn csc_euclidean_distance_on_select_columns<T>(
    lhs: &CscMatrix<T>,
    rhs: &CscMatrix<T>,
    select_columns_in_rhs: Option<&[usize]>,
) -> anyhow::Result<Vec<(usize, usize, T)>>
where
    T: nalgebra::RealField + Copy + std::iter::Sum<T>,
{
    let all_rhs = (0..rhs.ncols()).collect::<Vec<_>>();
    let select_columns_in_rhs = select_columns_in_rhs.unwrap_or(all_rhs.as_slice());

    if select_columns_in_rhs.len() != rhs.ncols() {
        return Err(anyhow::anyhow!(
            "found mismatches in `select_columns` maps: lhs -> rhs"
        ));
    }

    let mut ret = Vec::with_capacity(select_columns_in_rhs.len() * lhs.ncols());

    for (src_pos, src_col) in lhs.col_iter().enumerate() {
        for &tgt_pos in select_columns_in_rhs {
            let tgt_col = rhs.col(tgt_pos);

            let nn = src_col.nrows();
            let denom = T::from_usize(nn).unwrap_or(T::one());

            let idx_src = src_col.row_indices();
            let idx_tgt = tgt_col.row_indices();
            let val_src = src_col.values();
            let val_tgt = tgt_col.values();

            // sum_g (src[g] - tgt[g])^2 / sum_g 1
            // sum_g tgt[g]^2 + sum_g src[g]^2 - 2 sum_g tgt[g] * src[g]
            let mut s: usize = 0;
            let mut t: usize = 0;

            let tgt_sq_sum = val_tgt.iter().map(|&x| x * x).sum::<T>();
            let src_sq_sum = val_src.iter().map(|&x| x * x).sum::<T>();
            let mut overlap = T::zero();
            while s < idx_src.len() && t < idx_tgt.len() {
                if idx_src[s] == idx_tgt[t] {
                    overlap += val_src[s] * val_tgt[t];
                    s += 1;
                    t += 1;
                } else if idx_src[s] < idx_tgt[t] {
                    s += 1;
                } else {
                    t += 1;
                }
            }

            let dist = ((src_sq_sum + tgt_sq_sum - overlap - overlap) / denom).sqrt();

            ret.push((src_pos, tgt_pos, dist));
        }
    }

    Ok(ret)
}

impl<T> DistanceOps for CscMatrix<T>
where
    T: nalgebra::RealField + Copy + std::iter::Sum<T>,
{
    type Scalar = T;
    type Other = CscMatrix<T>;

    fn euclidean_distance(
        &self,
        other: &Self::Other,
    ) -> anyhow::Result<Vec<(usize, usize, Self::Scalar)>> {
        csc_euclidean_distance_on_select_columns(&self, other, None)
    }

    fn euclidean_distance_on_select_columns(
        &self,
        other: &Self::Other,
        select_columns_in_other: &[usize],
    ) -> anyhow::Result<Vec<(usize, usize, Self::Scalar)>> {
        csc_euclidean_distance_on_select_columns(&self, other, Some(select_columns_in_other))
    }
}

impl<T> MeltOps for DMatrix<T>
where
    T: nalgebra::RealField + Copy,
{
    type Scalar = T;
    type Mat = Self;
    fn melt_with_indexes(&self) -> (Vec<Self::Scalar>, Vec<Vec<usize>>) {
        let nelem = self.nrows() * self.ncols();

        let mut idx: Vec<Vec<usize>> = vec![Vec::with_capacity(nelem), Vec::with_capacity(nelem)];
        let mut val: Vec<Self::Scalar> = Vec::with_capacity(nelem);

        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                idx[0].push(r);
                idx[1].push(c);
                val.push(self[(r, c)]);
            }
        }

        (val, idx)
    }

    fn melt(&self) -> Vec<Self::Scalar> {
        let nelem = self.len();
        let mut val: Vec<Self::Scalar> = Vec::with_capacity(nelem);
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                val.push(self[(r, c)]);
            }
        }
        val
    }
}

impl<T> EncodingOps for DMatrix<T>
where
    T: nalgebra::RealField + Float,
    f32: From<T>,
{
    type Scalar = T;
    type Mat = Self;

    fn positional_embedding_columns(&self, emb_dim: usize) -> anyhow::Result<Self::Mat> {
        let ncols = self.ncols();

        let ncodes_per_col = emb_dim * 2;

        let mut output = Self::zeros(self.nrows(), ncols * ncodes_per_col);

        for j in 0..ncols {
            let x_j = self.column(j);

            for i in 0..emb_dim {
                let power = T::from(2.0 * i as f32 / emb_dim as f32).unwrap();
                let denom = T::from(10000_f32.powf(f32::from(power))).unwrap();

                let column_data = if i % 2 == 0 {
                    x_j.map(|x_ij| ComplexField::sin(x_ij / denom))
                } else {
                    x_j.map(|x_ij| ComplexField::cos(x_ij / denom))
                };

                // Assign the computed column to the output matrix
                output
                    .column_mut(j * ncodes_per_col + i)
                    .copy_from(&column_data);
            }
        }
        Ok(output)
    }
}

impl<T> CompositeOps for DMatrix<T>
where
    T: nalgebra::RealField + Copy,
{
    type Scalar = T;
    type Mat = Self;
    type Other = CscMatrix<T>;

    /// `self[:,col] += other[:,col]`
    /// * `col` - column index
    fn add_assign_column(&mut self, other: &Self::Other, j: usize) {
        debug_assert_eq!(self.nrows(), other.nrows());
        debug_assert_eq!(self.ncols(), other.ncols());
        if let Some(x_j) = other.get_col(j) {
            let vals = x_j.values();
            let rows = x_j.row_indices();
            for k in 0..vals.len() {
                let i = rows[k];
                let x_ij = vals[k];
                self[(i, j)] += x_ij;
            }
        }
    }

    /// `self += other`
    /// * `other` - `CscMatrix`
    fn add_assign(&mut self, other: &Self::Other) {
        debug_assert_eq!(self.nrows(), other.nrows());
        debug_assert_eq!(self.ncols(), other.ncols());
        for j in 0..other.ncols() {
            if let Some(x_j) = other.get_col(j) {
                let vals = x_j.values();
                let rows = x_j.row_indices();
                for k in 0..vals.len() {
                    let i = rows[k];
                    let x_ij = vals[k];
                    self[(i, j)] += x_ij;
                }
            }
        }
    }
}

impl<T> ConvertMatOps for DMatrix<T>
where
    T: nalgebra::RealField + Copy + candle_core::WithDType,
{
    type Mat = Self;
    type Scalar = T;

    fn from_tensor(tensor: &candle_core::Tensor) -> anyhow::Result<Self::Mat> {
        if tensor.dims().len() != 2 {
            return Err(anyhow::anyhow!("expected 2D tensor"));
        }

        let nrows = tensor.dims()[0];
        let ncols = tensor.dims()[1];
        let data: Vec<T> = tensor.flatten_all()?.to_vec1()?;
        Ok(Self::from_row_iterator(nrows, ncols, data.iter().cloned()))
    }

    fn to_tensor(&self, dev: &candle_core::Device) -> anyhow::Result<candle_core::Tensor> {
        use candle_core::Tensor;
        // Note: x.as_slice() will take values in the column-major order
        // However, Tensor::from_slice will take them in the row-major order
        let nrow = self.nrows();
        let ncol = self.ncols();
        Ok(Tensor::from_slice(self.as_slice(), (ncol, nrow), dev)?.transpose(0, 1)?)
    }
}

impl<T> ConvertMatOps for CscMatrix<T>
where
    T: nalgebra::RealField + Copy + candle_core::WithDType,
{
    type Mat = Self;
    type Scalar = T;
    fn from_tensor(tensor: &candle_core::Tensor) -> anyhow::Result<Self::Mat> {
        if tensor.dims().len() != 2 {
            return Err(anyhow::anyhow!("expected 2D tensor"));
        }

        let nrows = tensor.dims()[0];
        let ncols = tensor.dims()[1];
        let data: Vec<T> = tensor.flatten_all()?.to_vec1()?;

        let dense_matrix = DMatrix::from_row_iterator(nrows, ncols, data.iter().cloned());
        let csc_matrix = CscMatrix::from(&dense_matrix);

        Ok(csc_matrix)
    }

    fn to_tensor(&self, dev: &candle_core::Device) -> anyhow::Result<candle_core::Tensor> {
        use candle_core::Tensor;
        let dense_matrix = DMatrix::from(self);
        let nrow = dense_matrix.nrows();
        let ncol = dense_matrix.ncols();
        // Note: x.as_slice() will take values in the column-major order
        // However, Tensor::from_slice will take them in the row-major order
        Ok(Tensor::from_slice(dense_matrix.as_slice(), (ncol, nrow), dev)?.transpose(0, 1)?)
    }
}

impl<T> SampleOps for DMatrix<T>
where
    T: nalgebra::RealField + Float,
{
    type Mat = Self;
    type Scalar = T;

    fn runif(nrow: usize, ncol: usize) -> Self::Mat {
        let u01 = Uniform::<f32>::new(0., 1.).expect("failed to create uniform distribution");

        let rvec = (0..(nrow * ncol))
            .into_par_iter()
            .map_init(rand::rng, |rng, _| {
                let x = rng.sample(u01);
                T::from(x).expect("failed to type")
            })
            .collect();

        DMatrix::<T>::from_vec(nrow, ncol, rvec)
    }

    fn rnorm(nrow: usize, ncol: usize) -> Self::Mat {
        let rvec = (0..(nrow * ncol))
            .into_par_iter()
            .map_init(rand::rng, |rng, _| {
                let x: f32 = rng.sample(StandardNormal);
                T::from(x).expect("failed to type")
            })
            .collect();

        DMatrix::<T>::from_vec(nrow, ncol, rvec)
    }

    fn rgamma(nrow: usize, ncol: usize, param: (f32, f32)) -> Self::Mat {
        let (shape, scale) = param;
        let pdf = Gamma::new(shape, scale).unwrap();

        let rvec = (0..(nrow * ncol))
            .into_par_iter()
            .map_init(rand::rng, |rng, _| {
                let x: f32 = pdf.sample(rng);
                T::from(x).expect("failed to type")
            })
            .collect();

        DMatrix::<T>::from_vec(nrow, ncol, rvec)
    }
}

impl<T> MatElemOps for CscMatrix<T>
where
    T: nalgebra::RealField + Copy,
{
    type Mat = Self;
    type Scalar = T;

    fn log1p_inplace(&mut self) {
        for x in self.values_mut() {
            *x = (*x).ln_1p();
        }
    }

    fn log1p(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.log1p_inplace();
        ret
    }
}

impl<T> MatOps for CscMatrix<T>
where
    T: nalgebra::RealField + Copy,
{
    type Mat = Self;
    type Scalar = T;

    fn normalize_exp_logits_columns_inplace(&mut self) {
        let ncol = self.ncols();

        for j in 0..ncol {
            if let Some(log_j) = self.get_col(j) {
                let mut log_max = log_j.values()[0];
                for &logx_ij in log_j.values() {
                    log_max = log_max.min(logx_ij);
                }

                let mut denom = T::zero();
                for &logx_ij in log_j.values() {
                    denom += (logx_ij - log_max).exp();
                }

                if let Some(mut log_j) = self.get_col_mut(j) {
                    for x_ij in log_j.values_mut() {
                        *x_ij = (*x_ij - log_max).exp() / denom;
                    }
                }
            }
        }
    }

    fn normalize_exp_logits_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.normalize_exp_logits_columns_inplace();
        ret
    }

    fn sum_to_one_columns_inplace(&mut self) {
        let ncol = self.ncols();

        for j in 0..ncol {
            if let Some(x_j) = self.get_col(j) {
                let mut denom = T::zero();
                for &x_ij in x_j.values() {
                    denom += x_ij;
                }
                if let Some(mut x_j) = self.get_col_mut(j) {
                    if denom > T::zero() {
                        for x_ij in x_j.values_mut() {
                            *x_ij /= denom;
                        }
                    }
                }
            }
        }
    }

    fn sum_to_one_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.sum_to_one_columns_inplace();
        ret
    }

    fn sum_to_one_rows(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.sum_to_one_rows_inplace();
        ret
    }

    fn sum_to_one_rows_inplace(&mut self) {
        let nrow = self.nrows();
        let mut denom = vec![T::zero(); nrow];

        for col in self.col_iter() {
            col.row_indices()
                .iter()
                .zip(col.values().iter())
                .for_each(|(&i, &x_ij)| {
                    denom[i] += x_ij;
                });
        }

        for mut col in self.col_iter_mut() {
            let (row_indices, values) = col.rows_and_values_mut();
            row_indices.iter().zip(values).for_each(|(&i, x_ij)| {
                *x_ij /= denom[i];
            });
        }
    }

    fn normalize_columns_inplace(&mut self) {
        let ncol = self.ncols();

        for j in 0..ncol {
            if let Some(x_j) = self.get_col(j) {
                let mut denom = T::zero();
                for &x_ij in x_j.values() {
                    denom += x_ij * x_ij;
                }
                denom = denom.sqrt().max(T::one());

                if let Some(mut x_j) = self.get_col_mut(j) {
                    for x_ij in x_j.values_mut() {
                        *x_ij /= denom;
                    }
                }
            }
        }
    }

    fn normalize_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.normalize_columns_inplace();
        ret
    }

    fn scale_columns_inplace(&mut self) {
        let ncol = self.ncols();

        for j in 0..ncol {
            if let Some(x_j) = self.get_col(j) {
                let mut s0 = T::zero();
                let mut s1 = T::zero();
                let mut s2 = T::zero();

                for &x_ij in x_j.values() {
                    s0 += T::one();
                    s1 += x_ij;
                    s2 += x_ij * x_ij;
                }

                let mu = s1 / s0.max(T::one());
                let sig = (s2 / s0.max(T::one()) - mu * mu).sqrt();

                if let Some(mut x_j) = self.get_col_mut(j) {
                    if sig > T::zero() {
                        for x_ij in x_j.values_mut() {
                            *x_ij = (*x_ij - mu) / sig;
                        }
                    } else {
                        for x_ij in x_j.values_mut() {
                            *x_ij -= mu;
                        }
                    }
                }
            }
        }
    }

    fn scale_rows_inplace(&mut self) {
        let nrow = self.nrows();
        let ncol = self.ncols();
        let mut s0 = vec![T::zero(); nrow];
        let mut s1 = vec![T::zero(); nrow];
        let mut s2 = vec![T::zero(); nrow];

        for j in 0..ncol {
            if let Some(x_j) = self.get_col(j) {
                for (&x_ij, &i) in x_j.values().iter().zip(x_j.row_indices()) {
                    s0[i] += T::one();
                    s1[i] += x_ij;
                    s2[i] += x_ij * x_ij;
                }
            }
        }

        let mu = s1
            .into_iter()
            .zip(s0.iter())
            .map(|(x, &n)| x / n.max(T::one()))
            .collect::<Vec<_>>();

        let sig = s2
            .into_iter()
            .zip(mu.iter())
            .zip(s0.iter())
            .map(|((s2, &mu), &s0)| (s2 / s0.max(T::one()) - mu * mu).sqrt())
            .collect::<Vec<_>>();

        for j in 0..ncol {
            if let Some(mut x_j) = self.get_col_mut(j) {
                let (rows, values) = x_j.rows_and_values_mut();

                for (&i, x_ij) in rows.iter().zip(values) {
                    let mu_i = mu[i];
                    let sig_i = sig[i];
                    if sig_i > T::zero() {
                        *x_ij = (*x_ij - mu_i) / sig_i;
                    } else {
                        *x_ij -= mu_i;
                    }
                }
            }
        }
    }

    fn scale_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.scale_columns_inplace();
        ret
    }

    fn scale_rows(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.scale_rows_inplace();
        ret
    }

    fn centre_columns_inplace(&mut self) {
        let ncol = self.ncols();

        for j in 0..ncol {
            if let Some(x_j) = self.get_col(j) {
                let mut s0 = T::zero();
                let mut s1 = T::zero();

                for &x_ij in x_j.values() {
                    s0 += T::one();
                    s1 += x_ij;
                }

                let mu = s1 / s0.max(T::one());

                if let Some(mut x_j) = self.get_col_mut(j) {
                    for x_ij in x_j.values_mut() {
                        *x_ij -= mu;
                    }
                }
            }
        }
    }

    fn centre_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.centre_columns_inplace();
        ret
    }
}

impl<T> MatOps for DMatrix<T>
where
    T: nalgebra::RealField,
{
    type Mat = Self;
    type Scalar = T;

    fn normalize_exp_logits_columns_inplace(&mut self) {
        for mut x_j in self.column_iter_mut() {
            let log_max = x_j.max();
            let denom = x_j.map(|l| (l - log_max.clone()).exp()).sum();
            x_j.iter_mut()
                .for_each(|l| *l = (l.clone() - log_max.clone()).exp() / denom.clone());
        }
    }

    fn normalize_exp_logits_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.normalize_exp_logits_columns_inplace();
        ret
    }

    fn sum_to_one_columns_inplace(&mut self) {
        self.column_iter_mut()
            .for_each(|mut c| c.unscale_mut(c.sum()));
    }

    fn sum_to_one_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.sum_to_one_columns_inplace();
        ret
    }

    fn sum_to_one_rows_inplace(&mut self) {
        self.row_iter_mut().for_each(|mut r| r.unscale_mut(r.sum()));
    }

    fn sum_to_one_rows(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.sum_to_one_rows_inplace();
        ret
    }

    fn normalize_columns_inplace(&mut self) {
        for mut xx_j in self.column_iter_mut() {
            let denom = xx_j.norm();
            xx_j /= denom;
        }
    }

    fn normalize_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.normalize_columns_inplace();
        ret
    }

    fn scale_columns_inplace(&mut self) {
        for mut xx_j in self.column_iter_mut() {
            let mu = xx_j.mean();
            let sig = xx_j.variance().sqrt();
            xx_j.add_scalar_mut(-mu);
            if sig > T::zero() {
                xx_j /= sig;
            }
        }
    }

    fn scale_rows_inplace(&mut self) {
        for mut xx_i in self.row_iter_mut() {
            let mu = xx_i.mean();
            let sig = xx_i.variance().sqrt();
            xx_i.add_scalar_mut(-mu);
            if sig > T::zero() {
                xx_i /= sig;
            }
        }
    }

    fn scale_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.scale_columns_inplace();
        ret
    }

    fn scale_rows(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.scale_rows_inplace();
        ret
    }

    fn centre_columns_inplace(&mut self) {
        for mut xx_j in self.column_iter_mut() {
            let mu = xx_j.mean();
            xx_j.add_scalar_mut(-mu);
        }
    }

    fn centre_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.centre_columns_inplace();
        ret
    }
}

////////////////////////////////////////
// Input and output in triplet format //
////////////////////////////////////////

impl<T> MatTriplets for DMatrix<T>
where
    T: nalgebra::RealField + Float,
{
    type Mat = Self;
    type Scalar = T;

    fn from_nonzero_triplets<I>(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(I, I, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat>
    where
        I: TryInto<usize> + Copy,
        <I as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let mut data = vec![T::zero(); ncol * nrow];
        for (ii, jj, x_ij) in triplets {
            let ii: usize = ii.try_into().expect("failed to convert index ii");
            let jj: usize = jj.try_into().expect("failed to convert index jj");
            data[ii * ncol + jj] = x_ij;
        }
        Ok(DMatrix::from_row_slice(nrow, ncol, &data))
    }

    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)> {
        if let Some(eps) = T::from(1e-6) {
            let nrow = self.nrows();
            let ncol = self.ncols();
            let mut triplets = vec![];
            for j in 0..ncol {
                for i in 0..nrow {
                    let x_ij = &self[(i, j)];
                    if x_ij.abs() > eps {
                        triplets.push((i, j, *x_ij));
                    }
                }
            }
            Ok((nrow, ncol, triplets))
        } else {
            anyhow::bail!("eps is not defined")
        }
    }
}

impl<T> MatTriplets for CsrMatrix<T>
where
    T: nalgebra::RealField + Float,
{
    type Mat = Self;
    type Scalar = T;

    fn from_nonzero_triplets<I>(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(I, I, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat>
    where
        I: TryInto<usize> + Copy,
        <I as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let mut coo = CooMatrix::<T>::new(nrow, ncol);
        for (ii, jj, x_ij) in triplets {
            let ii: usize = ii.try_into().expect("failed to convert index ii");
            let jj: usize = jj.try_into().expect("failed to convert index jj");
            coo.push(ii, jj, x_ij);
        }
        Ok(CsrMatrix::from(&coo))
    }
    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)> {
        if let Some(eps) = T::from(1e-6) {
            let nrow = self.nrows();
            let ncol = self.ncols();

            let mut triplets = Vec::new();

            for i in 0..nrow {
                if let Some(x_i) = self.get_row(i) {
                    let cols = x_i.col_indices();
                    let vals = x_i.values();
                    for k in 0..cols.len() {
                        let j = cols[k];
                        let x_ij = vals[k];
                        if x_ij > eps {
                            triplets.push((i, j, x_ij))
                        };
                    }
                }
            }

            Ok((nrow, ncol, triplets))
        } else {
            anyhow::bail!("eps is not defined")
        }
    }
}

impl<T> MatTriplets for CscMatrix<T>
where
    T: nalgebra::RealField + Float,
{
    type Mat = Self;
    type Scalar = T;

    fn from_nonzero_triplets<I>(
        nrow: usize,
        ncol: usize,
        triplets: Vec<(I, I, Self::Scalar)>,
    ) -> anyhow::Result<Self::Mat>
    where
        I: TryInto<usize> + Copy,
        <I as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let mut coo = CooMatrix::<T>::new(nrow, ncol);
        for (ii, jj, x_ij) in triplets {
            let ii: usize = ii.try_into().expect("failed to convert index ii");
            let jj: usize = jj.try_into().expect("failed to convert index jj");
            coo.push(ii, jj, x_ij);
        }
        Ok(CscMatrix::from(&coo))
    }

    fn to_nonzero_triplets(
        &self,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, Self::Scalar)>)> {
        if let Some(eps) = T::from(1e-6) {
            let nrow = self.nrows();
            let ncol = self.ncols();

            let mut triplets = Vec::new();
            for j in 0..ncol {
                if let Some(x_j) = self.get_col(j) {
                    let rows = x_j.row_indices();
                    let vals = x_j.values();
                    for k in 0..rows.len() {
                        let i = rows[k];
                        let x_ij = vals[k];
                        if x_ij > eps {
                            triplets.push((i, j, x_ij))
                        }
                    }
                }
            }

            Ok((nrow, ncol, triplets))
        } else {
            anyhow::bail!("eps is not defined")
        }
    }
}

impl<T> CandleDataLoaderOps for DMatrix<T>
where
    T: nalgebra::RealField + Copy + candle_core::WithDType,
{
    type Scalar = T;
    type Mat = Self;

    // fn transpose(&self) -> Self::Mat {
    //     self.transpose()
    // }

    fn rows_to_tensor_vec(&self) -> Vec<candle_core::Tensor> {
        let mut idx_data = self
            .row_iter()
            .enumerate()
            .par_bridge()
            .map(|(i, row)| {
                let mut v =
                    candle_core::Tensor::from_iter(row.iter().copied(), &candle_core::Device::Cpu)
                        .expect("failed to create tensor");
                v = v.reshape((1, row.len())).expect("failed to reshape");
                (i, v)
            })
            .collect::<Vec<_>>();

        idx_data.sort_by_key(|(i, _)| *i);
        idx_data.into_iter().map(|(_, t)| t).collect()
    }
}
