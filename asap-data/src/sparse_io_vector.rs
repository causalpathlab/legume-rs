use crate::sparse_io::*;
use log::info;
use matrix_util::knn_match::ColumnDict;
use matrix_util::traits::*;
use matrix_util::utils::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;

type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

pub struct SparseIoVec {
    data_vec: Vec<Arc<SparseData>>,
    col_to_data: Vec<usize>,
    data_to_cols: HashMap<usize, Vec<usize>>,
    col_glob_to_loc: Vec<usize>,
    offset: usize,
    row_name_position: HashMap<Box<str>, usize>,
    column_names_with_batch: Vec<Box<str>>,
    col_to_group: Option<Vec<usize>>,
    batch_knn_lookup: Option<Vec<ColumnDict<usize>>>,
    col_to_batch: Option<Vec<usize>>,
    batch_idx_to_name: Option<Vec<Box<str>>>,
}

impl Index<usize> for SparseIoVec {
    type Output = Arc<SparseData>;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data_vec[idx]
    }
}

impl SparseIoVec {
    pub fn new() -> Self {
        Self {
            data_vec: vec![],
            col_to_data: vec![],
            data_to_cols: HashMap::new(),
            col_glob_to_loc: vec![],
            offset: 0,
            row_name_position: HashMap::new(),
            column_names_with_batch: vec![],
            col_to_group: None,
            batch_knn_lookup: None,
            col_to_batch: None,
            batch_idx_to_name: None,
        }
    }

    pub fn len(&self) -> usize {
        self.data_vec.len()
    }

    pub fn assign_groups(&mut self, cell_to_group: Vec<usize>) {
        self.col_to_group = Some(cell_to_group);
    }

    pub fn take_groups(&self) -> Option<&Vec<usize>> {
        self.col_to_group.as_ref()
    }

    pub fn push(&mut self, data: Arc<SparseData>) -> anyhow::Result<()> {
        if let Some(ncol_data) = data.num_columns() {
            debug_assert!(self.col_glob_to_loc.len() == self.offset);
            debug_assert!(self.col_to_data.len() == self.offset);
            let didx = self.data_vec.len();
            self.data_to_cols.insert(didx, vec![]);
            let data_to_cells = self
                .data_to_cols
                .get_mut(&didx)
                .ok_or(anyhow::anyhow!("failed to take didx {}", didx))?;

            for loc in 0..ncol_data {
                let glob = loc + self.offset;
                self.col_glob_to_loc.push(loc);
                self.col_to_data.push(didx);
                data_to_cells.push(glob);
                debug_assert!(glob == self.col_to_data.len() - 1);
            }
            info!("Extending column names...");

            let batch_tag = COLUMN_SEP.to_string() + &didx.to_string();
            self.column_names_with_batch.extend(
                data.column_names()?
                    .into_iter()
                    .map(|x| (x.to_string() + &batch_tag).into_boxed_str())
                    .collect::<Vec<_>>(),
            );
            info!("Checking row names...");

            for (data_row_pos, row) in data.row_names()?.iter().enumerate() {
                let glob_row_pos = self
                    .row_name_position
                    .entry(row.clone())
                    .or_insert(data_row_pos);
                if *glob_row_pos != data_row_pos {
                    return Err(anyhow::anyhow!(
                        "Row names mismatched: {} vs. {}",
                        *glob_row_pos,
                        data_row_pos
                    ));
                }
            }

            self.data_vec.push(data.clone());
            self.offset += ncol_data;
            info!("{} columns", self.offset);
        } else {
            return Err(anyhow::anyhow!("data file has no columns"));
        }
        Ok(())
    }

    pub fn num_columns_by_data(&self) -> anyhow::Result<Vec<usize>> {
        Ok(self
            .data_vec
            .iter()
            .map(|d| d.num_columns().unwrap_or(0_usize))
            .collect())
    }

    pub fn remove_backend_file(&mut self) -> anyhow::Result<()> {
        for dat in self.data_vec.iter() {
            dat.remove_backend_file()?;
        }
        Ok(())
    }

    pub fn num_rows(&self) -> anyhow::Result<usize> {
        let mut ret = 0;
        for dat in self.data_vec.iter() {
            let nr = dat
                .num_rows()
                .ok_or(anyhow::anyhow!("can't figure out the number of rows"))?;
            ret = ret.max(nr);
        }
        Ok(ret)
    }

    /// total number of columns across all data files
    pub fn num_columns(&self) -> anyhow::Result<usize> {
        let mut ret = 0;
        for data in self.data_vec.iter() {
            let nc = data
                .num_columns()
                .ok_or(anyhow::anyhow!("can't figure out the number of columns"))?;
            ret += nc;
        }
        Ok(ret)
    }

    ////////////////////
    // access columns //
    ////////////////////

    pub fn columns_triplets<I>(
        &self,
        cells: I,
    ) -> anyhow::Result<((usize, usize), Vec<(usize, usize, f32)>)>
    where
        I: Iterator<Item = usize>,
    {
        let mut triplets = vec![];
        let mut nrow = 0;
        let mut ncol = 0;
        // Note: each cell is a global index
        for glob in cells {
            let didx = self.col_to_data[glob];
            let loc = self.col_glob_to_loc[glob];

            let (loc_nrow, loc_ncol, loc_triplets) =
                self.data_vec[didx].read_triplets_by_single_column(loc)?;

            nrow = nrow.max(loc_nrow);
            triplets.extend(loc_triplets.iter().map(|&(i, j, v)| (i, j + ncol, v)));
            ncol += loc_ncol;
        }

        Ok(((nrow, ncol), triplets))
    }

    pub fn read_columns_ndarray<I>(&self, cells: I) -> anyhow::Result<ndarray::Array2<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.columns_triplets(cells)?;
        ndarray::Array2::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    pub fn read_columns_dmatrix<I>(&self, cells: I) -> anyhow::Result<nalgebra::DMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.columns_triplets(cells)?;
        DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    pub fn read_columns_csc<I>(&self, cells: I) -> anyhow::Result<CscMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.columns_triplets(cells)?;
        nalgebra_sparse::CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    pub fn read_columns_csr<I>(&self, cells: I) -> anyhow::Result<CsrMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.columns_triplets(cells)?;
        nalgebra_sparse::CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    pub fn read_columns_tensor<I>(&self, cells: I) -> anyhow::Result<Tensor>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets) = self.columns_triplets(cells)?;
        Ok(Tensor::from_nonzero_triplets(nrow, ncol, triplets)?)
    }

    /////////////////////
    // matched columns //
    /////////////////////

    /// Take columns matched with the given `cells` on a specific
    /// target batch
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batch` - a batch for targeted kNN search
    /// * `knn` - k-nearest neighbours
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * shape - (nrows, ncols)
    /// * triplets
    /// * distances
    pub fn matched_columns_triplets_on_target_batch<I>(
        &self,
        cells: I,
        target_batch: usize,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(
        (usize, usize),
        Vec<(usize, usize, f32)>,
        (Vec<usize>, Vec<usize>),
    )>
    where
        I: Iterator<Item = usize>,
    {
        let mut triplets = vec![];
        let mut nrow = 0;
        let mut ncol = 0;

        let lookups = self
            .batch_knn_lookup
            .as_ref()
            .ok_or(anyhow::anyhow!("no knn lookup"))?;

        let cell_to_batch = self
            .col_to_batch
            .as_ref()
            .ok_or(anyhow::anyhow!("no cell to batch"))?;

        debug_assert!(target_batch < self.num_batches());

        let mut source_columns = vec![];
        let mut source_positions = vec![];

        for (idx, glob) in cells.enumerate() {
            let source_batch = cell_to_batch[glob]; // this cell's batch

            if skip_same_batch && source_batch == target_batch {
                continue; // skip cells in the same batch
            }

            if let (Some(source_lookup), Some(target_lookup)) =
                (lookups.get(source_batch), lookups.get(target_batch))
            {
                let matched = source_lookup.match_against_by_name(&glob, knn, &target_lookup)?;
                for glob_matched in matched {
                    if glob == glob_matched {
                        continue; // avoid identical cell pairs
                    }
                    let didx = self.col_to_data[glob_matched];
                    let loc = self.col_glob_to_loc[glob_matched];
                    let (loc_nrow, loc_ncol, loc_triplets) =
                        self.data_vec[didx].read_triplets_by_single_column(loc)?;

                    nrow = nrow.max(loc_nrow);
                    triplets.extend(loc_triplets.iter().map(|&(i, j, v)| (i, j + ncol, v)));
                    ncol += loc_ncol;
                    source_columns.push(glob);
                    source_positions.push(idx);
                }
            }
        }

        Ok(((nrow, ncol), triplets, (source_columns, source_positions)))
    }

    /// Take columns matched with the given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batches` - the batches for targeted kNN search
    /// * `knn` - k-nearest neighbours
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * shape - (nrows, ncols)
    /// * triplets - a vector of triplets
    /// * `source_columns` - a vector of the source columns
    /// * distance - a vector of distances between the matched columns
    pub fn matched_columns_triplets<I>(
        &self,
        cells: I,
        target_batches: &Vec<usize>,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(
        (usize, usize),
        Vec<(usize, usize, f32)>,
        Vec<usize>, // source positions
        Vec<f32>,   // distances
    )>
    where
        I: Iterator<Item = usize>,
    {
        let cells: Vec<usize> = cells.collect();
        let y1 = self.read_columns_csc(cells.iter().cloned())?;

        let mut matched_triplets: Vec<(usize, usize, f32)> = vec![];
        let mut euclidean_distances: Vec<f32> = vec![];
        let mut tot_ncells_matched: usize = 0;
        let mut source_columns: Vec<usize> = vec![];

        for &target_b in target_batches.iter() {
            let (shape, triplets, sources) = self.matched_columns_triplets_on_target_batch(
                cells.iter().cloned(),
                target_b,
                knn,
                skip_same_batch,
            )?;

            matched_triplets.extend(
                triplets
                    .iter()
                    .map(|(i, j, z_ij)| (*i, *j + tot_ncells_matched, *z_ij)),
            );

            // Temporarily construct y0 and compute distances between the matched columns
            let y0 = CscMatrix::<f32>::from_nonzero_triplets(shape.0, shape.1, triplets)?;
            let y0_pos = &sources.1;
            euclidean_distances.extend(y0.euclidean_distance_matched_columns(&y1, y0_pos)?);

            tot_ncells_matched += shape.1;
            source_columns.extend(sources.0);
        }

        let shape = (y1.nrows(), tot_ncells_matched);

        Ok((shape, matched_triplets, source_columns, euclidean_distances))
    }

    /// Take columns matched with the given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batches` - the batches for targeted kNN search
    /// * `knn` - k-nearest neighbours
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * the knn-matched matrix
    /// * `source_columns` - a vector of the source columns
    /// * a vector of distances between the matched columns
    pub fn read_matched_columns_csc<I>(
        &self,
        cells: I,
        target_batches: &Vec<usize>,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(CscMatrix<f32>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets, source_columns, distances) =
            self.matched_columns_triplets(cells, target_batches, knn, skip_same_batch)?;
        Ok((
            CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)?,
            source_columns,
            distances,
        ))
    }

    /// Take columns matched with the given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batches` - the batches for targeted kNN search
    /// * `knn` - k-nearest neighbours
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * the knn-matched matrix
    /// * `source_columns` - a vector of the source columns
    /// * distances - a vector of distances between the matched columns
    pub fn read_matched_columns_csr<I>(
        &self,
        cells: I,
        target_batches: &Vec<usize>,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(CsrMatrix<f32>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets, source_columns, distances) =
            self.matched_columns_triplets(cells, target_batches, knn, skip_same_batch)?;
        Ok((
            CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)?,
            source_columns,
            distances,
        ))
    }

    /// Take columns matched with the given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batches` - the batches for targeted kNN search
    /// * `knn` - k-nearest neighbours
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * the knn-matched matrix
    /// * `source_columns` - a vector of the source columns
    /// * distances - a vector of distances between the matched columns
    pub fn read_matched_columns_ndarray<I>(
        &self,
        cells: I,
        target_batches: &Vec<usize>,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(ndarray::Array2<f32>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets, source_columns, distances) =
            self.matched_columns_triplets(cells, target_batches, knn, skip_same_batch)?;
        Ok((
            ndarray::Array2::<f32>::from_nonzero_triplets(nrow, ncol, triplets)?,
            source_columns,
            distances,
        ))
    }

    /// Take columns matched with the given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batches` - the batches for targeted kNN search
    /// * `knn` - k-nearest neighbours
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * the knn-matched matrix
    /// * `source_columns` - a vector of the source columns
    /// * distances - a vector of distances between the matched columns
    pub fn read_matched_columns_dmatrix<I>(
        &self,
        cells: I,
        target_batches: &Vec<usize>,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(nalgebra::DMatrix<f32>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let ((nrow, ncol), triplets, source_columns, distances) =
            self.matched_columns_triplets(cells, target_batches, knn, skip_same_batch)?;
        Ok((
            DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)?,
            source_columns,
            distances,
        ))
    }

    /// Register batch membership information along with the feature
    /// matrix for quick look up operations.
    ///
    /// # Arguments
    /// * `feature_matrix` - A feature matrix where each column corresponds to a cell.
    /// * `batch_membership` - A vector of batch membership information for each cell.
    pub fn register_batches_ndarray<T>(
        &mut self,
        feature_matrix: &ndarray::Array2<f32>,
        batch_membership: &Vec<T>,
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        {
            debug_assert_eq!(batch_membership.len(), feature_matrix.ncols());
        }
        self._register_batches(
            feature_matrix,
            batch_membership,
            |feature_matrix, batch_cells| {
                let columns = batch_cells
                    .iter()
                    .map(|&c| feature_matrix.column(c))
                    .collect::<Vec<_>>();
                ColumnDict::<usize>::from_ndarray_views(columns, batch_cells.clone())
            },
        )
    }

    /// Register batch membership information along with the feature
    /// matrix for quick look up operations.
    ///
    /// # Arguments
    /// * `feature_matrix` - A feature matrix where each column corresponds to a cell.
    /// * `batch_membership` - A vector of batch membership information for each cell.
    pub fn register_batches_dmatrix<T>(
        &mut self,
        feature_matrix: &nalgebra::DMatrix<f32>,
        batch_membership: &Vec<T>,
    ) -> anyhow::Result<()>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        {
            debug_assert_eq!(batch_membership.len(), feature_matrix.ncols());
        }

        self._register_batches(
            feature_matrix,
            batch_membership,
            |feature_matrix, batch_cells| {
                let columns = batch_cells
                    .iter()
                    .map(|&c| feature_matrix.column(c))
                    .collect::<Vec<_>>();
                ColumnDict::<usize>::from_dvector_views(columns, batch_cells.clone())
            },
        )
    }

    fn _register_batches<M, F, T>(
        &mut self,
        feature_matrix: &M,
        batch_membership: &Vec<T>,
        create_column_dict: F,
    ) -> anyhow::Result<()>
    where
        M: Sync,
        F: Fn(&M, &Vec<usize>) -> ColumnDict<usize> + Sync,
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let batches = partition_by_membership(&batch_membership, None);

        let ntot = self.num_columns()?;
        let mut col_to_batch = vec![0; ntot];

        let mut idx_name_dict = batches
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(batch_index, (batch_name, batch_cells))| {
                (
                    batch_index,
                    batch_name.to_string().into_boxed_str(),
                    create_column_dict(&feature_matrix, batch_cells),
                )
            })
            .collect::<Vec<_>>();

        idx_name_dict.sort_by_key(|&(idx, _, _)| idx);

        for (idx, _, dict) in idx_name_dict.iter() {
            dict.names()
                .iter()
                .for_each(|&cell| col_to_batch[cell] = *idx);
        }

        let batch_names = idx_name_dict
            .iter()
            .map(|(_, name, _)| name.clone())
            .collect::<Vec<_>>();

        let dictionaries = idx_name_dict
            .into_iter()
            .map(|(_, _, dict)| dict)
            .collect::<Vec<_>>();

        self.batch_knn_lookup = Some(dictionaries);
        self.col_to_batch = Some(col_to_batch);
        self.batch_idx_to_name = Some(batch_names);

        Ok(())
    }

    pub fn batch_name_map(&self) -> Option<HashMap<Box<str>, usize>> {
        if let Some(names) = &self.batch_idx_to_name {
            Some(
                names
                    .iter()
                    .enumerate()
                    .map(|(idx, name)| (name.clone(), idx))
                    .collect::<HashMap<Box<str>, usize>>(),
            )
        } else {
            None
        }
    }

    pub fn num_batches(&self) -> usize {
        self.batch_knn_lookup.as_ref().map_or(0, |v| v.len())
    }

    pub fn get_batch_membership<I>(&self, cells: I) -> Vec<usize>
    where
        I: Iterator<Item = usize>,
    {
        let cell_to_batch = self
            .col_to_batch
            .as_ref()
            .expect("cell_to_batch not initialized");
        cells.into_iter().map(|c| cell_to_batch[c]).collect()
    }

    pub fn column_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        debug_assert_eq!(self.num_columns()?, self.column_names_with_batch.len());
        Ok(self.column_names_with_batch.clone())
    }

    pub fn row_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        let ntot = self.num_rows()?;
        debug_assert_eq!(ntot, self.row_name_position.len());
        let mut ret = vec![Box::from(""); ntot];
        for (row, &index) in &self.row_name_position {
            debug_assert!(index < ntot);
            if index < ntot {
                ret[index] = row.clone();
            }
        }
        Ok(ret)
    }
}
