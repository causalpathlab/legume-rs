use crate::srt_cell_pairs::*;
use crate::srt_common::*;

use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::Inference;
use matrix_param::traits::TwoStatParam;

pub struct SrtCollapsedStat {
    pub left_sum_ds: Mat,
    pub left_delta_sum_ds: Mat,
    pub right_sum_ds: Mat,
    pub right_delta_sum_ds: Mat,
    pub size_s: DVec,
    pub left_coordinates: Mat,
    pub right_coordinates: Mat,
    n_rows: usize,
    n_cols: usize,
}

pub struct SrtCollapsedParameters {
    pub left: GammaMatrix,
    pub right: GammaMatrix,
    // pub left_neigh: GammaMatrix,
    // pub right_neigh: GammaMatrix,
    pub left_delta: GammaMatrix,
    pub right_delta: GammaMatrix,
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

    // adjust the left by the neighbours of the right
    let y_delta_left = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<CscMat> {
            let left = pairs[j].left;

            let mut y_dm = data.data.read_columns_csc(std::iter::once(left))?;
            let y_neigh_dm = data.data.read_columns_csc(n.right_only.iter().cloned())?;
            let y_hat_dm = impute_with_neighbours(&y_dm, &y_neigh_dm)?;
            y_dm.adjust_by_division_inplace(&y_hat_dm);

            Ok(y_dm)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    {
        let mut stat = arc_stat.lock().expect("lock stat");
        for yy in y_delta_left {
            for y_j in yy.col_iter() {
                let rows = y_j.row_indices();
                let vals = y_j.values();
                for (&gene, &y) in rows.iter().zip(vals.iter()) {
                    stat.left_delta_sum_ds[(gene, sample)] += y;
                }
            }
        }
    }

    // adjust the right by the neighbours of the left
    let y_delta_right = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<CscMat> {
            let right = pairs[j].right;

            let mut y_dm = data.data.read_columns_csc(std::iter::once(right))?;
            let y_neigh_dm = data.data.read_columns_csc(n.left_only.iter().cloned())?;
            let y_hat_dm = impute_with_neighbours(&y_dm, &y_neigh_dm)?;
            y_dm.adjust_by_division_inplace(&y_hat_dm);

            Ok(y_dm)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    {
        let mut stat = arc_stat.lock().expect("lock stat");
        for yy in y_delta_right {
            for y_j in yy.col_iter() {
                let rows = y_j.row_indices();
                let vals = y_j.values();
                for (&gene, &y) in rows.iter().zip(vals.iter()) {
                    stat.right_delta_sum_ds[(gene, sample)] += y;
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
            left_delta_sum_ds: Mat::zeros(dd, ss),
            right_sum_ds: Mat::zeros(dd, ss),
            right_delta_sum_ds: Mat::zeros(dd, ss),
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
    pub fn optimize(
        &self,
        hyper_param: Option<(f32, f32)>,
    ) -> anyhow::Result<SrtCollapsedParameters> {
        let (a0, b0) = hyper_param.unwrap_or((1_f32, 1_f32));

        let shape = (self.nrows(), self.ncols());

        let mut left_raw_ds = GammaMatrix::new(shape, a0, b0);
        let mut right_raw_ds = GammaMatrix::new(shape, a0, b0);
        let mut left_delta_ds = GammaMatrix::new(shape, a0, b0);
        let mut right_delta_ds = GammaMatrix::new(shape, a0, b0);

        // let mut shared_neigh_ds = GammaMatrix::new(shape, a0, b0);
        // let mut left_neigh_ds = GammaMatrix::new(shape, a0, b0);
        // let mut right_neigh_ds = GammaMatrix::new(shape, a0, b0);

        let size_s = &self.size_s.transpose();
        let sample_size_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        info!("Calibrating marginal statistics");

        left_raw_ds.update_stat(&self.left_sum_ds, &sample_size_ds);
        left_raw_ds.calibrate();
        right_raw_ds.update_stat(&self.right_sum_ds, &sample_size_ds);
        right_raw_ds.calibrate();
        left_delta_ds.update_stat(&self.left_delta_sum_ds, &sample_size_ds);
        left_delta_ds.calibrate();
        right_delta_ds.update_stat(&self.right_delta_sum_ds, &sample_size_ds);
        right_delta_ds.calibrate();

        info!("Done: optimization of collapsed statistics");

        Ok(SrtCollapsedParameters {
            left: left_raw_ds,
            right: right_raw_ds,
            left_delta: left_delta_ds,
            right_delta: right_delta_ds,
        })
    }
}
