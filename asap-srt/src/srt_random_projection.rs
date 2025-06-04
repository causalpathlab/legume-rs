// #![allow(dead_code)]

use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use asap_alg::random_projection::*;

pub trait SrtRandProjOps {
    fn random_projection(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<SrtRandProjOut>;

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
}

impl<'a> SrtRandProjOps for SrtCellPairs<'a> {
    fn assign_pairs_to_samples(
        &mut self,
        proj: &SrtRandProjOut,
        num_sorting_features: Option<usize>,
        npairs_per_sample: Option<usize>,
    ) -> anyhow::Result<usize> {
        let nn = self.len();

        if proj.left.ncols() != nn {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

        if proj.right.ncols() != nn {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

        let distances = Mat::from_column_slice(nn, 1, &self.distances)
            .scale_columns()
            .transpose();

        let proj_kn = concatenate_vertical(&[proj.left.clone(), proj.right.clone(), distances])?;

        let target_kk = num_sorting_features.unwrap_or(proj_kn.nrows());
        let kk = proj_kn.nrows().min(target_kk).min(nn);
        let binary_codes = binary_sort_columns(&proj_kn, kk)?;

        let max_group = *binary_codes
            .iter()
            .max()
            .ok_or(anyhow::anyhow!("unable to determine max element"))?;

        self.assign_samples(binary_codes, npairs_per_sample);

        info!("Assigned them to {} samples", max_group + 1);

        Ok(max_group + 1)
    }

    fn random_projection(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<SrtRandProjOut> {
        let nn = self.pairs.len();

        // 1. just to have projection results
        let col_proj = self.data.project_columns(target_dim, block_size)?;

        // 2. simply copy them down based on pairs
        let mut ret = SrtRandProjOut {
            left: Mat::zeros(target_dim, nn),
            right: Mat::zeros(target_dim, nn),
        };

        self.data.visit_column_pairs_by_block(
            &self.pairs,
            &random_project_pairs_visitor,
            &col_proj,
            &mut ret,
            block_size,
        )?;

        info!("successfully projected {} pairs", self.len());

        Ok(ret)
    }
}

fn random_project_pairs_visitor(
    pairs_with_bounds: PairsWithBounds,
    _data: &SparseIoVec,
    column_proj: &RandColProjOut,
    arc_pair_proj: Arc<Mutex<&mut SrtRandProjOut>>,
) -> anyhow::Result<()> {
    let lb = pairs_with_bounds.lb;
    let ub = pairs_with_bounds.ub;
    let pairs = pairs_with_bounds.pairs;

    let left = pairs.into_iter().map(|&(i, _)| i);
    let right = pairs.into_iter().map(|&(_, j)| j);

    let mut proj = arc_pair_proj.lock().expect("lock proj");

    for (src, tgt) in left.zip(lb..ub) {
        proj.left
            .column_mut(tgt)
            .copy_from(&column_proj.proj.column(src));
    }

    for (src, tgt) in right.zip(lb..ub) {
        proj.right
            .column_mut(tgt)
            .copy_from(&column_proj.proj.column(src));
    }

    Ok(())
}

// ///
// /// each position vector (1 x d) `*` random_proj (d x r)
// ///
// fn positional_projection<T>(
//     coords: &Mat,
//     batch_membership: &Vec<T>,
//     d: usize,
// ) -> anyhow::Result<Mat>
// where
//     T: Sync + Send + Clone + Eq + std::hash::Hash,
// {
//     if coords.nrows() != batch_membership.len() {
//         return Err(anyhow::anyhow!("incompatible batch membership"));
//     }
//     let maxval = coords.max();
//     let minval = coords.min();
//     let coords = coords
//         .map(|x| (x - minval) / (maxval - minval + 1.))
//         .transpose();
//     let batches = partition_by_membership(batch_membership, None);
//     let mut ret = Mat::zeros(d, coords.ncols());
//     for (_b, points) in batches.into_iter() {
//         let rand_proj = Mat::rnorm(d, coords.nrows());
//         points.into_iter().for_each(|p| {
//             ret.column_mut(p)
//                 .copy_from(&(&rand_proj * coords.column(p)));
//         });
//     }
//     let (lb, ub) = (-4., 4.);
//     ret.scale_columns_inplace();
//     if ret.max() > ub || ret.min() < lb {
//         info!("Clamping values [{}, {}] after standardization", lb, ub);
//         ret.iter_mut().for_each(|x| {
//             *x = x.clamp(lb, ub);
//         });
//         ret.scale_columns_inplace();
//     }
//     Ok(ret)
// }
