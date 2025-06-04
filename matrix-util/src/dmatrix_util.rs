use nalgebra::DMatrix;
use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix, csr::CsrMatrix};

use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Gamma, StandardNormal, Uniform};
use rayon::prelude::*;

use crate::traits::*;

pub fn concatenate_vertical<T>(matrices: &[DMatrix<T>]) -> anyhow::Result<DMatrix<T>>
where
    T: nalgebra::RealField,
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

pub fn concatenate_horizontal<T>(matrices: &[DMatrix<T>]) -> anyhow::Result<DMatrix<T>>
where
    T: nalgebra::RealField,
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

pub fn row_membership_matrix<T>(row_membership: Vec<usize>) -> anyhow::Result<DMatrix<T>>
where
    T: nalgebra::RealField + Copy,
{
    let mtot = match row_membership.iter().max() {
        Some(&m) => m + 1,
        _ => 1,
    };

    let mut ret_dm = DMatrix::zeros(row_membership.len(), mtot);
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

    let ret: Vec<(usize, usize, T)> = lhs
        .col_iter()
        .enumerate()
        .zip(select_columns_in_rhs.into_iter())
        .map(|((src_pos, src_col), &tgt_pos)| {
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
            (src_pos, tgt_pos, dist)
        })
        .collect();

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

impl<T> MatOps for CscMatrix<T>
where
    T: nalgebra::RealField + Copy,
{
    type Mat = Self;
    type Scalar = T;

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

    fn scale_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.scale_columns_inplace();
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

    fn normalize_columns_inplace(&mut self) {
        for mut xx_j in self.column_iter_mut() {
            let denom = xx_j.norm().max(T::one());
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

    fn scale_columns(&self) -> Self::Mat {
        let mut ret = self.clone();
        ret.scale_columns_inplace();
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
