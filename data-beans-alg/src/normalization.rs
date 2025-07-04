pub trait AdjustByDivisionOp {
    fn adjust_by_division(&mut self, denom_db: &nalgebra::DMatrix<f32>, batches: &[usize]);
}

impl AdjustByDivisionOp for nalgebra_sparse::CscMatrix<f32> {
    fn adjust_by_division(&mut self, denom_db: &nalgebra::DMatrix<f32>, batches: &[usize]) {
        let eps: f32 = 1e-4;

        self.col_iter_mut().zip(batches).for_each(|(mut x_j, &b)| {
            let d_j = denom_db.column(b);
            let dsum: f32 = d_j.sum();
            let xsum: f32 = x_j.values().iter().sum();
            let scale: f32 = if dsum > 0.0 { xsum / dsum } else { 1.0 };

            x_j.values_mut()
                .iter_mut()
                .zip(&d_j)
                .for_each(|(x_ij, &d_ij)| {
                    *x_ij /= d_ij * scale + eps;
                });
        });
    }
}

/// normalize distance
pub trait NormalizeDistance: Iterator<Item = f32> + Sized {
    fn exp_sum(self, lambda: f32) -> f32
    where
        Self: Sized,
    {
        self.map(|d| (-d * lambda).exp()).sum()
    }

    fn normalized_exp(self, target_exp_sum: f32) -> Vec<f32>
    where
        Self: Clone,
    {
        // a. discount distance values by the minimum
        let dmin = self.clone().fold(f32::INFINITY, |a, b| a.min(b));
        let dist = self.map(move |d| d - dmin);

        // b. find optimum lambda value by a line search
        let mut lambda = 10.0;
        let mut fval = target_exp_sum - dist.clone().exp_sum(lambda);
        let max_iter = 100;

        for _ in 0..max_iter {
            let lambda_old = lambda;
            if fval < 0.0 {
                lambda *= 2.0; // total weight sum > target_exp_sum
            } else {
                lambda *= 0.5; // total weight sum < target_exp_sum
            }

            let fval_new = target_exp_sum - dist.clone().exp_sum(lambda);

            if fval_new.abs() > fval.abs() {
                lambda = lambda_old;
                break;
            }

            fval = fval_new;
        }

        // c. final weight calibration
        dist.map(|v| (-v * lambda).exp()).collect()
    }
}

impl<I> NormalizeDistance for I where I: Iterator<Item = f32> {}
