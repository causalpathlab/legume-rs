use crate::sparse_io::*;
use matrix_util::knn_match::ColumnDict;
use rand::prelude::SliceRandom;
use std::sync::Arc;

type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

#[allow(dead_code)]
pub struct SparseIoVec {
    data_vec: Vec<Arc<SparseData>>,
    col_to_data: Vec<usize>,
    data_to_cols: HashMap<usize, Vec<usize>>,
    col_glob_to_loc: Vec<usize>,
    offset: usize,
    col_to_group: Option<Vec<usize>>,
    batch_knn_lookup: Option<Vec<ColumnDict<usize>>>,
    col_to_batch: Option<Vec<usize>>,
}

impl SparseIoVec {
    pub fn new() -> Self {
        Self {
            data_vec: vec![],
            col_to_data: vec![],
            data_to_cols: HashMap::new(),
            col_glob_to_loc: vec![],
            offset: 0,
            col_to_group: None,
            batch_knn_lookup: None,
            col_to_batch: None,
        }
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

            self.data_vec.push(data.clone());
            self.offset += ncol_data;
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

    pub fn collect_columns_triplets<I>(
        &self,
        cells: I,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, f32)>)>
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

        Ok((nrow, ncol, triplets))
    }

    pub fn collect_matched_columns_triplets<I>(
        &self,
        cells: I,
        target_batch: usize,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(usize, usize, Vec<(usize, usize, f32)>, Vec<usize>)>
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

        let mut source_cells = vec![];

        for glob in cells {
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
                    source_cells.push(glob);
                }
            }
        }

        Ok((nrow, ncol, triplets, source_cells))
    }

    pub fn read_columns_ndarray<I>(&self, cells: I) -> anyhow::Result<ndarray::Array2<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let (nrow, ncol, triplets) = self.collect_columns_triplets(cells)?;
        ndarray::Array2::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    pub fn read_columns_dmatrix<I>(&self, cells: I) -> anyhow::Result<nalgebra::DMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let (nrow, ncol, triplets) = self.collect_columns_triplets(cells)?;
        DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    pub fn read_columns_csc<I>(&self, cells: I) -> anyhow::Result<CscMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let (nrow, ncol, triplets) = self.collect_columns_triplets(cells)?;
        nalgebra_sparse::CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    pub fn read_columns_csr<I>(&self, cells: I) -> anyhow::Result<CsrMatrix<f32>>
    where
        I: Iterator<Item = usize>,
    {
        let (nrow, ncol, triplets) = self.collect_columns_triplets(cells)?;
        nalgebra_sparse::CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)
    }

    pub fn read_matched_columns_ndarray<I>(
        &self,
        cells: I,
        target_batch: usize,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(ndarray::Array2<f32>, Vec<usize>)>
    where
        I: Iterator<Item = usize>,
    {
        let (nrow, ncol, triplets, sources) =
            self.collect_matched_columns_triplets(cells, target_batch, knn, skip_same_batch)?;
        Ok((
            ndarray::Array2::<f32>::from_nonzero_triplets(nrow, ncol, triplets)?,
            sources,
        ))
    }

    pub fn read_matched_columns_dmatrix<I>(
        &self,
        cells: I,
        target_batch: usize,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(nalgebra::DMatrix<f32>, Vec<usize>)>
    where
        I: Iterator<Item = usize>,
    {
        let (nrow, ncol, triplets, sources) =
            self.collect_matched_columns_triplets(cells, target_batch, knn, skip_same_batch)?;
        Ok((
            DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)?,
            sources,
        ))
    }

    pub fn read_matched_columns_csc<I>(
        &self,
        cells: I,
        target_batch: usize,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(CscMatrix<f32>, Vec<usize>)>
    where
        I: Iterator<Item = usize>,
    {
        let (nrow, ncol, triplets, sources) =
            self.collect_matched_columns_triplets(cells, target_batch, knn, skip_same_batch)?;
        Ok((
            CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)?,
            sources,
        ))
    }

    pub fn read_matched_columns_csr<I>(
        &self,
        cells: I,
        target_batch: usize,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(CsrMatrix<f32>, Vec<usize>)>
    where
        I: Iterator<Item = usize>,
    {
        let (nrow, ncol, triplets, sources) =
            self.collect_matched_columns_triplets(cells, target_batch, knn, skip_same_batch)?;
        Ok((
            CsrMatrix::<f32>::from_nonzero_triplets(nrow, ncol, triplets)?,
            sources,
        ))
    }

    pub fn register_batches_ndarray(
        &mut self,
        feature_matrix: &ndarray::Array2<f32>,
        batch_membership: Vec<usize>,
    ) -> anyhow::Result<()> {
        {
            debug_assert_eq!(batch_membership.len(), feature_matrix.ncols());
        }
        let batches = partition_by_membership(&batch_membership, None);

        let batch_names = batches.keys().cloned().collect::<Vec<usize>>();

        let num_batch = batch_names
            .iter()
            .max()
            .copied()
            .ok_or(anyhow::anyhow!("unable to determine batch size"))?
            + 1;

        let dictionaries = (0..num_batch)
            .map(|b| {
                if let Some(batch_cells) = batches.get(&b) {
                    let columns = batch_cells
                        .iter()
                        .map(|&c| feature_matrix.column(c))
                        .collect::<Vec<_>>();
                    ColumnDict::<usize>::from_ndarray_views(columns, batch_cells.clone())
                } else {
                    ColumnDict::<usize>::empty_ndarray_views()
                }
            })
            .collect::<Vec<_>>();

        self.batch_knn_lookup = Some(dictionaries);
        Ok(())
    }

    pub fn register_batches_dmatrix(
        &mut self,
        feature_matrix: &nalgebra::DMatrix<f32>,
        batch_membership: Vec<usize>,
    ) -> anyhow::Result<()> {
        {
            debug_assert_eq!(batch_membership.len(), feature_matrix.ncols());
        }

        let batches = partition_by_membership(&batch_membership, None);
        let batch_names = batches.keys().cloned().collect::<Vec<usize>>();

        let num_batch = batch_names
            .iter()
            .max()
            .copied()
            .ok_or(anyhow::anyhow!("unable to determine batch size"))?
            + 1;

        let ntot = self.num_columns()?;
        let mut dictionaries = vec![];
        let mut cell_to_batch = vec![0; ntot];

        for b in 0..num_batch {
            if let Some(batch_cells) = batches.get(&b) {
                let columns = batch_cells
                    .iter()
                    .map(|&c| feature_matrix.column(c))
                    .collect::<Vec<_>>();

                dictionaries.push(ColumnDict::<usize>::from_dvector_views(
                    columns,
                    batch_cells.clone(),
                ));

                for &j in batch_cells {
                    cell_to_batch[j] = b;
                }
            } else {
                dictionaries.push(ColumnDict::<usize>::empty_dvector_views());
            }
        }

        self.batch_knn_lookup = Some(dictionaries);
        self.col_to_batch = Some(cell_to_batch);
        Ok(())
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
}

fn partition_by_membership(
    membership: &Vec<usize>,
    nelem_per_group: Option<usize>,
) -> HashMap<usize, Vec<usize>> {
    // Take care of empty pseudobulk samples
    let mut pb_position: HashMap<usize, usize> = HashMap::new();
    {
        let mut pos = 0_usize;
        for k in membership {
            if !pb_position.contains_key(k) {
                pb_position.insert(*k, pos);
                pos += 1;
            }
        }
    }
    // dbg!(&pb_position);

    let mut pb_cells: HashMap<usize, Vec<usize>> = HashMap::new();
    for (cell, &k) in membership.iter().enumerate() {
        let &s = pb_position.get(&k).expect("failed to get position");
        pb_cells.entry(s).or_default().push(cell);
    }

    // Down sample cells if needed
    pb_cells.par_iter_mut().for_each(|(_, cells)| {
        let ncells = cells.len();
        if let Some(ntarget) = nelem_per_group {
            if ncells > ntarget {
                let mut rng = rand::thread_rng();
                cells.shuffle(&mut rng);
                cells.truncate(ntarget);
            }
        }
        // dbg!(cells.len());
    });
    pb_cells
}
