use crate::asap_embed_common::*;

/// For each column of the matched matrix
/// RMSE(j,k) = sqrt( sum_g (target[g,j] - source[g,k])^2 / sum_g 1 )
pub fn compute_euclidean_distance(
    source: &CscMat,
    target: &Mat,
    matched_position_in_target: &Vec<usize>,
) -> Vec<f32> {
    let denom = source.nrows() as f32;

    source
        .col_iter()
        .zip(matched_position_in_target.iter())
        .map(|(src_j, &k)| {
            // target column/cell
            let tgt_k = target.column(k);
            // source column/cell
            let src_rows = src_j.row_indices();
            let src_vals = src_j.values();

            let tgt_tot = tgt_k.map(|x| x * x).sum();
            // to avoid double counting
            let overlap = src_rows.iter().map(|&i| tgt_k[i] * tgt_k[i]).sum::<f32>();
            let delta_overlap = src_rows
                .iter()
                .zip(src_vals.iter())
                .map(|(&i, &z)| (z - tgt_k[i]) * (z - tgt_k[i]))
                .sum::<f32>();
            ((tgt_tot - overlap + delta_overlap) / denom).sqrt()
        })
        .collect::<Vec<f32>>()
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
