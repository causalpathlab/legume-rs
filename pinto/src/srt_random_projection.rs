// #![allow(dead_code)]

use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use data_beans_alg::random_projection::*;

pub trait SrtRandProjOps {
    fn random_projection<T>(
        &self,
        target_dim: usize,
        block_size: usize,
        batch_membership: Option<&[T]>,
    ) -> anyhow::Result<SrtRandProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

    fn assign_pairs_to_samples(
        &mut self,
        proj: &SrtRandProjOut,
        num_sorting_features: Option<usize>,
        npairs_per_sample: Option<usize>,
    ) -> anyhow::Result<usize>;
}

pub struct SrtRandProjOut {
    pub left: Mat,
    pub right: Mat,
    pub left_delta: Mat,
    pub right_delta: Mat,
}

impl SrtRandProjOps for SrtCellPairs<'_> {
    fn assign_pairs_to_samples(
        &mut self,
        proj: &SrtRandProjOut,
        num_sorting_features: Option<usize>,
        npairs_per_sample: Option<usize>,
    ) -> anyhow::Result<usize> {
        let mm = self.len();

        if proj.left.ncols() != mm {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

        if proj.right.ncols() != mm {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

        if proj.left_delta.ncols() != mm {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

        if proj.right_delta.ncols() != mm {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

        let embedding_kn = self
            .coordinate_embedding_pairs()?
            .scale_columns()
            .transpose();

        let proj_kn = concatenate_vertical(&[
            proj.left_delta.clone(),
            proj.right_delta.clone(),
            embedding_kn,
        ])?;

        let target_kk = num_sorting_features.unwrap_or(proj_kn.nrows());
        let kk = proj_kn.nrows().min(target_kk).min(mm);
        let binary_codes = binary_sort_columns(&proj_kn, kk)?;

        let max_group = *binary_codes
            .iter()
            .max()
            .ok_or(anyhow::anyhow!("unable to determine max element"))?;

        self.assign_samples(binary_codes, npairs_per_sample);

        info!("Assigned them to {} samples", max_group + 1);

        Ok(max_group + 1)
    }

    fn random_projection<T>(
        &self,
        target_dim: usize,
        block_size: usize,
        batch_membership: Option<&[T]>,
    ) -> anyhow::Result<SrtRandProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let nn = self.pairs.len();

        // 1. random projection basis
        let nrows = self.data.num_rows()?;
        let basis_dk = Mat::rnorm(nrows, target_dim);

        // 2. visit the pairs with the projected basis
        let mut ret = SrtRandProjOut {
            left: Mat::zeros(target_dim, nn),
            right: Mat::zeros(target_dim, nn),
            left_delta: Mat::zeros(target_dim, nn),
            right_delta: Mat::zeros(target_dim, nn),
        };

        self.visit_pairs_by_block(
            &random_project_pairs_visitor,
            &basis_dk,
            &mut ret,
            block_size,
        )?;

        info!("successfully projected {} pairs", self.len());

        fn adjust_batch(mat: &mut Mat, indices: &[usize]) -> anyhow::Result<()> {
            let xx_left_delta = subset_columns(&mat, &indices)?
                .transpose()
                .centre_columns()
                .transpose();
            assign_columns(&xx_left_delta, &indices, mat);
            Ok(())
        }

        if let Some(col_to_batch) = batch_membership {
            let batches = partition_by_membership(col_to_batch, None);

            if col_to_batch.len() == self.data.num_columns()? && batches.len() > 1 {
                info!("adjusting batch biases ...");
                for (_, cols) in batches.iter() {
                    let left_cols = cols.iter().map(|&j| self.pairs[j].left).collect::<Vec<_>>();
                    let right_cols = cols
                        .iter()
                        .map(|&j| self.pairs[j].right)
                        .collect::<Vec<_>>();

                    adjust_batch(&mut ret.left, &left_cols)?;
                    adjust_batch(&mut ret.left_delta, &left_cols)?;
                    adjust_batch(&mut ret.right, &right_cols)?;
                    adjust_batch(&mut ret.right_delta, &right_cols)?;
                }
                info!("done with adjusting batch biases");
            }
        }

        ret.left.scale_rows_inplace();
        ret.right.scale_rows_inplace();
        ret.left_delta.scale_rows_inplace();
        ret.right_delta.scale_rows_inplace();

        Ok(ret)
    }
}

fn random_project_pairs_visitor(
    bound: (usize, usize),
    data: &SrtCellPairs,
    basis_dk: &Mat,
    arc_pair_proj: Arc<Mutex<&mut SrtRandProjOut>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let left = pairs.into_iter().map(|pp| pp.left);
    let right = pairs.into_iter().map(|pp| pp.right);
    let pairs_neighbours = &data.pairs_neighbours[lb..ub];

    let log_y_left_dm = data
        .data
        .read_columns_csc(left)?
        .log1p()
        .normalize_columns();

    let log_y_right_dm = data
        .data
        .read_columns_csc(right)?
        .log1p()
        .normalize_columns();

    let mut y_left_km = (log_y_left_dm.transpose() * basis_dk).transpose();
    let mut y_right_km = (log_y_right_dm.transpose() * basis_dk).transpose();

    y_left_km.centre_columns_inplace();
    y_right_km.centre_columns_inplace();

    // adjust the left by the neighbours of the right
    let y_delta_left = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, n)| -> anyhow::Result<Mat> {
            let left = pairs[j].left;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(left))?;
            let y_right_neigh_dm = data.data.read_columns_csc(n.right_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_right_neigh_dm)?;

            y_d1.adjust_by_division_inplace(&y_hat_d1);
            y_d1.log1p_inplace();
            y_d1.normalize_columns_inplace();

            Ok((y_d1.transpose() * basis_dk).transpose())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut y_delta_left_km = concatenate_horizontal(&y_delta_left)?;
    y_delta_left_km.centre_columns_inplace();

    // adjust the right by the neighbours of the left
    let y_delta_right = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, n)| -> anyhow::Result<Mat> {
            let right = pairs[j].right;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(right))?;
            let y_left_neigh_dm = data.data.read_columns_csc(n.left_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_left_neigh_dm)?;

            y_d1.adjust_by_division_inplace(&y_hat_d1);
            y_d1.log1p_inplace();
            y_d1.normalize_columns_inplace();

            Ok((y_d1.transpose() * basis_dk).transpose())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut y_delta_right_km = concatenate_horizontal(&y_delta_right)?;
    y_delta_right_km.centre_columns_inplace();

    let mut proj = arc_pair_proj.lock().expect("lock proj");

    for (src, tgt) in (lb..ub).enumerate() {
        proj.left.column_mut(tgt).copy_from(&y_left_km.column(src));
        proj.left_delta
            .column_mut(tgt)
            .copy_from(&y_delta_left_km.column(src));
        proj.right
            .column_mut(tgt)
            .copy_from(&y_right_km.column(src));
        proj.right_delta
            .column_mut(tgt)
            .copy_from(&y_delta_right_km.column(src));
    }

    Ok(())
}
