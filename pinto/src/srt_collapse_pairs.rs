use crate::srt_cell_pairs::*;
use crate::srt_common::*;

use indicatif::ProgressIterator;
use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::Inference;
use matrix_param::traits::TwoStatParam;
use std::iter::once;

pub struct SrtCollapsedStat {
    pub left_sum_ds: Mat,
    pub left_neigh_sum_ds: Mat,
    pub right_sum_ds: Mat,
    pub right_neigh_sum_ds: Mat,
    pub shared_neigh_sum_ds: Mat,
    pub size_s: DVec,
    pub left_coordinates: Mat,
    pub right_coordinates: Mat,
    n_rows: usize,
    n_cols: usize,
}

pub struct SrtCollapsedParameters {
    pub left: GammaMatrix,
    pub right: GammaMatrix,
    pub left_neigh: GammaMatrix,
    pub right_neigh: GammaMatrix,
    pub shared_neigh: GammaMatrix,
    pub left_boundary: GammaMatrix,
    pub right_boundary: GammaMatrix,
}

pub trait SrtCollapsePairsOps {
    fn collapse_pairs(&self) -> anyhow::Result<SrtCollapsedStat>;
}

impl<'a> SrtCollapsePairsOps for SrtCellPairs<'a> {
    fn collapse_pairs(&self) -> anyhow::Result<SrtCollapsedStat> {
        let mut srt_stat = SrtCollapsedStat::new(
            self.data.num_rows()?,
            self.num_coordinates(),
            self.num_samples()?,
        );
        self.visit_pairs_by_sample(&collect_pair_stat_visitor, &mut srt_stat)?;
        Ok(srt_stat)
    }
}

fn collect_pair_stat_visitor(
    indices: &[usize],
    data: &SrtCellPairs,
    sample: usize,
    arc_stat: Arc<Mutex<&mut SrtCollapsedStat>>,
) -> anyhow::Result<()> {
    let pairs = indices
        .into_iter()
        .filter_map(|&j| data.pairs.get(j))
        .collect::<Vec<_>>();

    let left = pairs.iter().map(|&x| x.left);
    let right = pairs.iter().map(|&x| x.right);

    let y_left = data.data.read_columns_csc(left)?;
    let y_right = data.data.read_columns_csc(right)?;

    {
        let mut stat = arc_stat.lock().expect("lock stat");

        for y_j in y_left.col_iter() {
            let rows = y_j.row_indices();
            let vals = y_j.values();
            for (&gene, &y) in rows.iter().zip(vals.iter()) {
                stat.left_sum_ds[(gene, sample)] += y;
            }
        }

        for y_j in y_right.col_iter() {
            let rows = y_j.row_indices();
            let vals = y_j.values();
            for (&gene, &y) in rows.iter().zip(vals.iter()) {
                stat.right_sum_ds[(gene, sample)] += y;
            }
        }

        let (left_coord, right_coord) = data.average_position(&indices);

        stat.left_coordinates
            .column_mut(sample)
            .copy_from(&left_coord);

        stat.right_coordinates
            .column_mut(sample)
            .copy_from(&right_coord);
    }

    ////////////////////////////////////////////////////
    // imputation by neighbours and update statistics //
    ////////////////////////////////////////////////////

    let pairs_neighbours = indices
        .iter()
        .filter_map(|&j| data.pairs_neighbours.get(j))
        .collect::<Vec<_>>();

    let y_hat_left = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<CscMat> {
            let left = pairs[j].left;
            let y_left = data.data.read_columns_csc(once(left))?;
            let y_left_neigh = data.data.read_columns_csc(n.left_only.iter().cloned())?;
            impute_with_neighbours(y_left, y_left_neigh)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    {
        let mut stat = arc_stat.lock().expect("lock stat");
        for yy in y_hat_left {
            for y_j in yy.col_iter() {
                let rows = y_j.row_indices();
                let vals = y_j.values();
                for (&gene, &y) in rows.iter().zip(vals.iter()) {
                    stat.left_neigh_sum_ds[(gene, sample)] += y;
                }
            }
        }
    }

    let y_hat_right = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<CscMat> {
            let right = pairs[j].right;
            let y_right = data.data.read_columns_csc(once(right))?;
            let y_right_neigh = data.data.read_columns_csc(n.right_only.iter().cloned())?;
            impute_with_neighbours(y_right, y_right_neigh)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    {
        let mut stat = arc_stat.lock().expect("lock stat");
        for yy in y_hat_right {
            for y_j in yy.col_iter() {
                let rows = y_j.row_indices();
                let vals = y_j.values();
                for (&gene, &y) in rows.iter().zip(vals.iter()) {
                    stat.right_neigh_sum_ds[(gene, sample)] += y;
                }
            }
        }
    }

    let y_hat_shared = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<CscMat> {
            let left = pairs[j].left;
            let right = pairs[j].right;

            // impute on the average
            let y = data.data.read_columns_csc(once(left))? * 0.5
                + data.data.read_columns_csc(once(right))? * 0.5;

            let y_shared_neigh = data.data.read_columns_csc(n.shared.iter().cloned())?;
            impute_with_neighbours(y, y_shared_neigh)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    {
        let mut stat = arc_stat.lock().expect("lock stat");
        for yy in y_hat_shared {
            for y_j in yy.col_iter() {
                let rows = y_j.row_indices();
                let vals = y_j.values();
                for (&gene, &y) in rows.iter().zip(vals.iter()) {
                    stat.shared_neigh_sum_ds[(gene, sample)] += y;
                }
            }
        }
    }

    Ok(())
}

impl SrtCollapsedStat {
    pub fn new(dd: usize, nc: usize, ss: usize) -> Self {
        Self {
            left_sum_ds: Mat::zeros(dd, ss),
            left_neigh_sum_ds: Mat::zeros(dd, ss),
            right_sum_ds: Mat::zeros(dd, ss),
            right_neigh_sum_ds: Mat::zeros(dd, ss),
            shared_neigh_sum_ds: Mat::zeros(dd, ss),
            size_s: DVec::zeros(ss),
            left_coordinates: Mat::zeros(nc, ss),
            right_coordinates: Mat::zeros(nc, ss),
            n_rows: dd,
            n_cols: ss,
        }
    }
    pub fn nrows(&self) -> usize {
        self.n_rows
    }
    pub fn ncols(&self) -> usize {
        self.n_cols
    }

    /// Optimize Poisson-Gamma probabilities
    ///
    /// * `hyper_param` : prior gamma parameters
    /// * `num_iter`: number of optimization steps
    ///
    /// Left boundary (target is Δ)
    /// * foreground: `left(g,j) ~ Poisson[ ρl(g,s) Δl(g,s) ]`
    /// * background: `right_neigh(g,j) ~ Poisson[ ρl(g,s) ]`
    ///
    /// Right boundary (target is Δ)
    /// * foreground: `right(g,j) ~ Poisson[ ρr(g,s) Δr(g,s) ]`
    /// * background: `left_neigh(g,j) ~ Poisson[ ρr(g,s) ]`
    ///
    /// β will capture background signals (no need to report)
    ///
    /// Marginal statistics
    /// * left: `left(g,j) ~ Poisson[μl(g,s)]`
    /// * right: `right(g,j) ~ Poisson[μr(g,s)]`
    /// * shared: `shared(g,j) ~ Poisson[μs(g,s)]`
    /// * left_neigh: `left_neigh(g,j) ~ Poisson[δl(g,s)]`
    /// * right_neigh: `right_neigh(g,j) ~ Poisson[δr(g,s)]`
    ///
    /// We train models like this...
    ///
    /// Encoders:
    /// * `[E(left) - E(right_neigh)] -> zΔl`
    /// * `[E(right) - E(left_neigh)] -> zΔr`
    /// * `E(left) -> zl`, `E(right) -> zr`, and `E(shared) -> zs`
    ///
    /// Decoders:
    /// * `Δl = θ(zΔl) * β`
    /// * `Δr = θ(zΔr) * β`
    /// * `μl = θ(zl + z_Δl + zs) * β`
    /// * `μr = θ(zr + z_Δr + zs) * β`
    /// * `μs = θ(zs) * β`
    ///
    pub fn optimize(
        &self,
        hyper_param: Option<(f32, f32)>,
        num_iter: Option<usize>,
    ) -> anyhow::Result<SrtCollapsedParameters> {
        let (a0, b0) = hyper_param.unwrap_or((1_f32, 1_f32));

        let shape = (self.nrows(), self.ncols());

        let mut left_raw_ds = GammaMatrix::new(shape, a0, b0);
        let mut right_raw_ds = GammaMatrix::new(shape, a0, b0);
        let mut shared_neigh_ds = GammaMatrix::new(shape, a0, b0);
        let mut left_neigh_ds = GammaMatrix::new(shape, a0, b0);
        let mut right_neigh_ds = GammaMatrix::new(shape, a0, b0);

        let size_s = &self.size_s.transpose();
        let denom_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        // 1. Estimate the marginal models
        info!("Calibrating marginal statistics");
        {
            left_raw_ds.update_stat(&self.left_sum_ds, &denom_ds);
            left_raw_ds.calibrate();
            right_raw_ds.update_stat(&self.right_sum_ds, &denom_ds);
            right_raw_ds.calibrate();
            shared_neigh_ds.update_stat(&self.shared_neigh_sum_ds, &denom_ds);
            shared_neigh_ds.calibrate();
            left_neigh_ds.update_stat(&self.left_neigh_sum_ds, &denom_ds);
            left_neigh_ds.calibrate();
            right_neigh_ds.update_stat(&self.right_neigh_sum_ds, &denom_ds);
            right_neigh_ds.calibrate();
        }

        // 2. Estimate the boundary models
        let mut left_fg_ds = GammaMatrix::new(shape, a0, b0);
        let mut right_fg_ds = GammaMatrix::new(shape, a0, b0);
        let mut left_bg_ds = GammaMatrix::new(shape, a0, b0);
        let mut right_bg_ds = GammaMatrix::new(shape, a0, b0);
        info!("Calibrating boundary statistics");

        let num_iter = num_iter.unwrap_or(10);

        (0..num_iter).progress().for_each(|_iter| {
            //           sum_left + sum_right_neigh
            // ρ(left) = ---------------------------
            //           n * (Δ(left) + 1)
            left_bg_ds.update_stat(
                &(&self.left_sum_ds + &self.right_neigh_sum_ds),
                &denom_ds.component_mul(&left_fg_ds.posterior_mean().map(|x| x + 1.0)),
            );
            left_bg_ds.calibrate();

            //           sum_left
            // Δ(left) = -------------
            //           n * ρ(left)
            left_fg_ds.update_stat(
                &self.left_sum_ds,
                &denom_ds.component_mul(&left_bg_ds.posterior_mean()),
            );
            left_fg_ds.calibrate();

            //           sum_right + sum_left_neigh
            // ρ(right) = ------------------------------------
            //           n * (Δ(right) + 1)
            right_bg_ds.update_stat(
                &(&self.right_sum_ds + &self.left_neigh_sum_ds),
                &denom_ds.component_mul(&right_fg_ds.posterior_mean().map(|x| x + 1.0)),
            );
            right_bg_ds.calibrate();

            //           sum_right
            // Δ(right) = -------------
            //           n * ρ(right)
            right_fg_ds.update_stat(
                &self.right_sum_ds,
                &denom_ds.component_mul(&right_bg_ds.posterior_mean()),
            );
            right_fg_ds.calibrate();
        });

        info!("Done: optimization of collapsed statistics");

        Ok(SrtCollapsedParameters {
            left: left_raw_ds,
            right: right_raw_ds,
            left_neigh: left_neigh_ds,
            right_neigh: right_neigh_ds,
            shared_neigh: shared_neigh_ds,
            left_boundary: left_fg_ds,
            right_boundary: right_fg_ds,
        })
    }
}
