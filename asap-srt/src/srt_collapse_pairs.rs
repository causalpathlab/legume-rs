// #![allow(dead_code)]

use crate::srt_cell_pairs::*;
use crate::srt_common::*;
use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::Inference;
use matrix_param::traits::TwoStatParam;

pub struct SrtCollapsedParameters {
    pub left: GammaMatrix,
    pub right: GammaMatrix,
}

pub struct SrtCollapsedStat {
    pub left_data_sum_ds: Mat,
    pub right_data_sum_ds: Mat,
    pub size_s: DVec,
    pub left_coordinates: Mat,
    pub right_coordinates: Mat,
    n_rows: usize,
    n_cols: usize,
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
    job: PairsWithIndices,
    data: &SrtCellPairs,
    sample: usize,
    arc_stat: Arc<Mutex<&mut SrtCollapsedStat>>,
) -> anyhow::Result<()> {
    let indices = job.indices;
    let left = job.pairs.into_iter().map(|&(i, _)| i);
    let right = job.pairs.into_iter().map(|&(_, j)| j);

    let y_left = data.data.read_columns_csc(left)?;
    let y_right = data.data.read_columns_csc(right)?;

    let mut stat = arc_stat.lock().expect("lock stat");

    for y_j in y_left.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.left_data_sum_ds[(gene, sample)] += y;
        }
        stat.size_s[sample] += 1_f32; // each pair is a sample
    }

    for y_j in y_right.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.right_data_sum_ds[(gene, sample)] += y;
        }
    }

    let (left_coord, right_coord) = data.average_position(&indices);

    stat.left_coordinates
        .column_mut(sample)
        .copy_from(&left_coord);

    stat.right_coordinates
        .column_mut(sample)
        .copy_from(&right_coord);

    Ok(())
}

impl SrtCollapsedStat {
    pub fn new(dd: usize, nc: usize, ss: usize) -> Self {
        Self {
            left_data_sum_ds: Mat::zeros(dd, ss),
            right_data_sum_ds: Mat::zeros(dd, ss),
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

    pub fn optimize(&self) -> anyhow::Result<SrtCollapsedParameters> {
        let mut denom_ds = Mat::zeros(self.nrows(), self.ncols());
        let size_s = &self.size_s.transpose();
        denom_ds.row_iter_mut().for_each(|mut row| {
            row.copy_from(size_s);
        });

        let (a0, b0) = (1_f32, 1_f32);

        let mut left_param_ds = GammaMatrix::new((self.nrows(), self.ncols()), a0, b0);
        let mut right_param_ds = GammaMatrix::new((self.nrows(), self.ncols()), a0, b0);
	let mut centre_param_ds = GammaMatrix::new((self.nrows(), self.ncols()), a0, b0);


	// E[left(g,j)] = μ(g,s) * α(g,s)
	// E[right(g,j)] = μ(g,s) * β(g,s)

	// L = sum_j left(g,j) * log[ μ(g,s) * α(g,s) ] - n(s) * μ(g,s) * α(g,s)
	//   = left_sum(g,s) * log[ μ(g,s) * α(g,s) ] - n(s) * μ(g,s) * α(g,s)

	// R = sum_j right(g,j) * log[ μ(g,s) * β(g,s) ]- n(s) * μ(g,s) * β(g,s)
	//   = right_sum(g,s) * log[ μ(g,s) * β(g,s) ] - n(s) * μ(g,s) * β(g,s)
	
	
	//           left_sum(g,s) + right_sum(g,s)
	// μ(g,s) = ---------------------------------
	//            n(s) * [ α(g,s) + β(g,s) ]

	//           left_sum(g,s)
	// α(g,s) = -----------------
	//            n(s) * μ(g,s)

	//           right_sum(g,s)
	// β(g,s) = -----------------
	//            n(s) * μ(g,s)



        left_param_ds.update_stat(&self.left_data_sum_ds, &denom_ds);
        left_param_ds.calibrate();


        right_param_ds.update_stat(&self.right_data_sum_ds, &denom_ds);
        right_param_ds.calibrate();

        Ok(SrtCollapsedParameters {
            left: left_param_ds,
            right: right_param_ds,
        })
    }
}
