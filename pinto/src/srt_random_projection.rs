// #![allow(dead_code)]

use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use data_beans_alg::random_projection::*;

pub trait SrtRandProjOps {
    fn random_projection(
        &self,
        target_dim: usize,
        block_size: usize,
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
        block_size: usize,
    ) -> anyhow::Result<SrtRandProjOut> {
        let nn = self.pairs.len();

        // 1. just to have projection results
        let col_proj = self.data.project_columns(target_dim, None)?;

        // 2. simply copy them down based on pairs
        let mut ret = SrtRandProjOut {
            left: Mat::zeros(target_dim, nn),
            right: Mat::zeros(target_dim, nn),
        };

        self.visit_pairs_by_block(
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
    bound: (usize, usize),
    data: &SrtCellPairs,
    column_proj: &RandColProjOut,
    arc_pair_proj: Arc<Mutex<&mut SrtRandProjOut>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let left = pairs.into_iter().map(|pp| pp.left);
    let right = pairs.into_iter().map(|pp| pp.right);

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
