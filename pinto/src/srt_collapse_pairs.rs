use crate::srt_cell_pairs::*;
use crate::srt_common::*;

use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::TwoStatParam;

pub struct SrtCollapsedStat {
    pub left_sum_ds: Mat,
    pub right_sum_ds: Mat,
    pub size_s: DVec,
    pub left_coordinates: Mat,
    pub right_coordinates: Mat,
    pub left_coord_emb: Mat,
    pub right_coord_emb: Mat,
    n_rows: usize,
    n_cols: usize,
}

#[allow(unused)]
pub struct SrtCollapsedParameters {
    pub left: GammaMatrix,
    pub right: GammaMatrix,
}

pub trait SrtCollapsePairsOps {
    fn collapse_pairs(&self, batch_effect: Option<&Mat>)
    -> anyhow::Result<SrtCollapsedStat>;
}

impl<'a> SrtCollapsePairsOps for SrtCellPairs<'a> {
    fn collapse_pairs(
        &self,
        batch_effect: Option<&Mat>,
    ) -> anyhow::Result<SrtCollapsedStat> {
        let mut srt_stat = SrtCollapsedStat::new(
            self.data.num_rows(),
            self.num_coordinates(),
            self.num_coordinate_embedding(),
            self.num_samples()?,
        );
        self.visit_pairs_by_sample(&collect_pair_stat_visitor, &batch_effect, &mut srt_stat)?;
        Ok(srt_stat)
    }
}

fn collect_pair_stat_visitor(
    indices: &[usize],
    data: &SrtCellPairs,
    sample: usize,
    batch_effect: &Option<&Mat>,
    arc_stat: Arc<Mutex<&mut SrtCollapsedStat>>,
) -> anyhow::Result<()> {
    let pairs = indices
        .iter()
        .filter_map(|&j| data.pairs.get(j))
        .collect::<Vec<_>>();

    let left = pairs.iter().map(|&x| x.left);
    let right = pairs.iter().map(|&x| x.right);

    let mut y_left = data.data.read_columns_csc(left)?;
    let mut y_right = data.data.read_columns_csc(right)?;

    // batch adjustment if needed
    if let Some(delta_db) = *batch_effect {
        let left = pairs.iter().map(|&x| x.left);
        let right = pairs.iter().map(|&x| x.right);
        let left_batches = data.data.get_batch_membership(left);
        y_left.adjust_by_division_of_selected_inplace(delta_db, &left_batches);
        let right_batches = data.data.get_batch_membership(right);
        y_right.adjust_by_division_of_selected_inplace(delta_db, &right_batches);
    }

    let (left_coord, right_coord) = data.average_position(indices);
    let (left_coord_emb, right_coord_emb) = data.average_coordinate_embedding(indices);

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
    }

    Ok(())
}

#[allow(unused)]
impl SrtCollapsedStat {
    pub fn new(dd: usize, nc: usize, ne: usize, ss: usize) -> Self {
        Self {
            left_sum_ds: Mat::zeros(dd, ss),
            right_sum_ds: Mat::zeros(dd, ss),
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

        let mut left_ds = GammaMatrix::new(shape, a0, b0);
        let mut right_ds = GammaMatrix::new(shape, a0, b0);

        let size_s = &self.size_s.transpose();
        let sample_size_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        info!("Calibrating statistics");

        left_ds.update_stat(&self.left_sum_ds, &sample_size_ds);
        left_ds.calibrate();
        right_ds.update_stat(&self.right_sum_ds, &sample_size_ds);
        right_ds.calibrate();

        info!("Resolved collapsed statistics");

        Ok(SrtCollapsedParameters {
            left: left_ds,
            right: right_ds,
        })
    }
}
