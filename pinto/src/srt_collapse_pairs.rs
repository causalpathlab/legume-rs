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
    pub left_coord_emb: Mat,
    pub right_coord_emb: Mat,
    n_rows: usize,
    n_cols: usize,
}

pub struct SrtCollapsedParameters {
    pub left: GammaMatrix,
    pub right: GammaMatrix,
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
            self.num_coordinate_embedding(),
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

    let (left_coord, right_coord) = data.average_position(&indices);
    let (left_coord_emb, right_coord_emb) = data.average_coordinate_embedding(&indices);

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

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(left))?;
            let y_right_neigh_dm = data.data.read_columns_csc(n.right_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_right_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);

            Ok(y_d1)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // adjust the right by the neighbours of the left
    let y_delta_right = pairs_neighbours
        .iter()
        .enumerate()
        .map(|(j, &n)| -> anyhow::Result<CscMat> {
            let right = pairs[j].right;

            let mut y_d1 = data.data.read_columns_csc(std::iter::once(right))?;
            let y_left_neigh_dm = data.data.read_columns_csc(n.left_only.iter().cloned())?;
            let y_hat_d1 = impute_with_neighbours(&y_d1, &y_left_neigh_dm)?;
            y_d1.adjust_by_division_inplace(&y_hat_d1);

            Ok(y_d1)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    {
        let mut stat = arc_stat.lock().expect("lock stat");

        //////////////////////////////////////////
        // update the left and right statistics //
        //////////////////////////////////////////

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

        ////////////////////////////////////////////
        // keeping track of the average positions //
        ////////////////////////////////////////////

        stat.left_coordinates
            .column_mut(sample)
            .copy_from(&left_coord);

        stat.right_coordinates
            .column_mut(sample)
            .copy_from(&right_coord);

        stat.left_coord_emb
            .column_mut(sample)
            .copy_from(&left_coord_emb);

        stat.right_coord_emb
            .column_mut(sample)
            .copy_from(&right_coord_emb);

        ////////////////////////////////////////////////
        // update the left and right delta statistics //
        ////////////////////////////////////////////////

        for yy in y_delta_left {
            for y_j in yy.col_iter() {
                let rows = y_j.row_indices();
                let vals = y_j.values();
                for (&gene, &y) in rows.iter().zip(vals.iter()) {
                    stat.left_delta_sum_ds[(gene, sample)] += y;
                }
            }
        }

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
    pub fn new(dd: usize, nc: usize, ne: usize, ss: usize) -> Self {
        Self {
            left_sum_ds: Mat::zeros(dd, ss),
            left_delta_sum_ds: Mat::zeros(dd, ss),
            right_sum_ds: Mat::zeros(dd, ss),
            right_delta_sum_ds: Mat::zeros(dd, ss),
            size_s: DVec::zeros(ss),
            left_coordinates: Mat::zeros(nc, ss),
            right_coordinates: Mat::zeros(nc, ss),
            left_coord_emb: Mat::zeros(ne, ss),
            right_coord_emb: Mat::zeros(ne, ss),
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

    pub fn num_coordinates(&self) -> usize {
        self.left_coordinates.nrows()
    }

    pub fn num_coordinate_embedding(&self) -> usize {
        self.left_coord_emb.nrows()
    }

    /// Write all the coordinate pairs into `.parquet` file
    /// * `file_path`: destination file name (try to include a recognizable extension in the end, e.g., `.parquet`)
    /// * `coordinate_names`: column names for the left (`left_{}`) and right (`right_{}`) where each `{}` will be replaced with the corresponding column name
    pub fn to_parquet(
        &self,
        file_path: &str,
        coordinate_names: Option<Vec<Box<str>>>,
    ) -> anyhow::Result<()> {
        let left = self.left_coordinates.transpose();
        let right = self.right_coordinates.transpose();
        let coordinate_names = coordinate_names.unwrap_or(
            (0..self.num_coordinates())
                .map(|x| x.to_string().into_boxed_str())
                .collect(),
        );

        if coordinate_names.len() != self.num_coordinates() {
            return Err(anyhow::anyhow!("invalid coordinate names"));
        }

        let column_names = coordinate_names
            .iter()
            .map(|x| format!("left_{}", x).into_boxed_str())
            .chain(
                coordinate_names
                    .iter()
                    .map(|x| format!("right_{}", x).into_boxed_str()),
            )
            .collect::<Vec<_>>();

        concatenate_horizontal(&[left, right])?.to_parquet(None, Some(&column_names), file_path)?;

        Ok(())
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

        let size_s = &self.size_s.transpose();
        let sample_size_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        info!("Calibrating statistics");

        left_raw_ds.update_stat(&self.left_sum_ds, &sample_size_ds);
        left_raw_ds.calibrate();
        right_raw_ds.update_stat(&self.right_sum_ds, &sample_size_ds);
        right_raw_ds.calibrate();
        left_delta_ds.update_stat(&self.left_delta_sum_ds, &sample_size_ds);
        left_delta_ds.calibrate();
        right_delta_ds.update_stat(&self.right_delta_sum_ds, &sample_size_ds);
        right_delta_ds.calibrate();

        info!("Resolved collapsed statistics");

        Ok(SrtCollapsedParameters {
            left: left_raw_ds,
            right: right_raw_ds,
            left_delta: left_delta_ds,
            right_delta: right_delta_ds,
        })
    }
}
