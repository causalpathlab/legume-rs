#![allow(dead_code)]

use super::*;

impl SparseIoVec {
    /////////////////////////////////////
    // data dictionary related methods //
    /////////////////////////////////////

    /// Register batch membership information along with the feature
    /// matrix for quick look up operations.
    ///
    /// # Arguments
    /// * `feature_matrix` - A feature matrix where each column corresponds to a cell.
    /// * `batch_membership` - A vector of batch membership information for each cell.
    pub fn register_batches_ndarray<T>(
        &mut self,
        feature_matrix: &ndarray::Array2<f32>,
        batch_membership: &[T],
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
            |feature_matrix, batch_cells, parallel_insert| {
                let columns = batch_cells
                    .iter()
                    .map(|&c| feature_matrix.column(c))
                    .collect::<Vec<_>>();
                if parallel_insert {
                    ColumnDict::<usize>::from_ndarray_views(columns, batch_cells.clone())
                } else {
                    ColumnDict::<usize>::from_ndarray_views_serial(columns, batch_cells.clone())
                }
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
        batch_membership: &[T],
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
            |feature_matrix, batch_cells, parallel_insert| {
                let columns = batch_cells
                    .iter()
                    .map(|&c| feature_matrix.column(c))
                    .collect::<Vec<_>>();
                if parallel_insert {
                    ColumnDict::<usize>::from_dvector_views(columns, batch_cells.clone())
                } else {
                    ColumnDict::<usize>::from_dvector_views_serial(columns, batch_cells.clone())
                }
            },
        )
    }

    fn _register_batches<M, F, T>(
        &mut self,
        feature_matrix: &M,
        batch_membership: &[T],
        create_column_dict: F,
    ) -> anyhow::Result<()>
    where
        M: Sync,
        F: Fn(&M, &Vec<usize>, bool) -> ColumnDict<usize> + Sync,
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let batches = partition_by_membership(batch_membership, None);

        let ntot = self.num_columns();
        let mut col_to_batch = vec![0; ntot];

        // Pick the parallelism level with the most work: when batches >=
        // threads, parallelize the outer loop and keep HNSW insertion
        // serial; when batches are few, build batches sequentially and let
        // each HNSW build use the full thread pool. Either way avoids
        // nested rayon contention.
        let n_threads = rayon::current_num_threads();
        let outer_parallel = batches.len() >= n_threads;

        info!(
            "building per-batch HNSW indices ({} batches, {} cells, {}) ...",
            batches.len(),
            ntot,
            if outer_parallel {
                "outer parallel"
            } else {
                "inner parallel"
            }
        );

        let prog_bar = ProgressBar::new(batches.len() as u64);
        prog_bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:30.cyan/blue} {pos}/{len} batches HNSW",
            )
            .unwrap()
            .progress_chars("##-"),
        );

        // Canonical batch id = SORTED LABEL order (consistent with
        // `register_batch_membership`), so per-batch stats / δ columns carry the
        // same batch ids as the rest of the pipeline. Previously this sorted by
        // descending size and used the processing position as the id, which
        // silently permuted batch ids vs the label order — a latent bug for δ /
        // `AdjMethod::Batch` consumers.
        let mut batches_vec: Vec<_> = batches.into_iter().collect();
        batches_vec.sort_by(|a, b| a.0.to_string().cmp(&b.0.to_string()));
        // In outer-parallel mode every per-batch HNSW runs on 1 rayon thread.
        // For batches much larger than the average that causes a long serial
        // tail.  Allow large batches to use inner parallelism instead: once
        // the small batches finish, their rayon threads work-steal subtasks
        // from the large batch's parallel_insert_slice, effectively donating
        // all available cores to the outlier.  The threshold is generous
        // (4× average) to avoid inner-parallel overhead on typical batches.
        let avg_cells = if batches_vec.is_empty() {
            1
        } else {
            ntot / batches_vec.len()
        };
        let large_batch_threshold = (avg_cells * 4).max(n_threads * 512);
        // Ids assigned above; now schedule largest batches first (LPT) so the
        // heavy tail overlaps with the small batches. The canonical id rides
        // along; `sort_by_key(idx)` below restores canonical order.
        let mut enumerated: Vec<_> = batches_vec.iter().enumerate().collect();
        enumerated.sort_by_key(|(_, (_, cells))| std::cmp::Reverse(cells.len()));
        let mut idx_name_glob_dict: Vec<_> = if outer_parallel {
            enumerated
                .into_par_iter()
                .progress_with(prog_bar.clone())
                .map(|(batch_index, (batch_name, batch_glob_indices))| {
                    let use_inner_parallel = batch_glob_indices.len() > large_batch_threshold;
                    (
                        batch_index,
                        batch_name.to_string().into_boxed_str(),
                        batch_glob_indices.clone(),
                        create_column_dict(feature_matrix, batch_glob_indices, use_inner_parallel),
                    )
                })
                .collect()
        } else {
            enumerated
                .into_iter()
                .progress_with(prog_bar.clone())
                .map(|(batch_index, (batch_name, batch_glob_indices))| {
                    (
                        batch_index,
                        batch_name.to_string().into_boxed_str(),
                        batch_glob_indices.clone(),
                        create_column_dict(feature_matrix, batch_glob_indices, true),
                    )
                })
                .collect()
        };
        prog_bar.finish_and_clear();

        idx_name_glob_dict.sort_by_key(|&(idx, _, _, _)| idx);

        let mut batch_names = vec![];
        let mut batch_to_cols = vec![];
        let mut dictionaries = vec![];

        for (batch_idx, batch_name, glob_indices, dict) in idx_name_glob_dict.into_iter() {
            dict.names()
                .iter()
                .for_each(|&cell| col_to_batch[cell] = batch_idx);

            batch_names.push(batch_name);
            batch_to_cols.push(glob_indices);
            dictionaries.push(dict);
        }

        self.derived.batch_knn_lookup = Some(dictionaries);
        self.derived.col_to_batch = Some(col_to_batch);
        self.derived.batch_to_cols = Some(batch_to_cols);
        self.derived.batch_idx_to_name = Some(batch_names);

        if self.num_batches() > 2 {
            self.sort_batch_proximity()?;
        }

        Ok(())
    }

    fn sort_batch_proximity(&mut self) -> anyhow::Result<()> {
        let lookups = self
            .derived
            .batch_knn_lookup
            .as_ref()
            .ok_or(anyhow::anyhow!("no knn lookup"))?;

        use nalgebra::DMatrix;

        info!("retrieving batch-specific lookups");
        let batch_data = lookups
            .iter()
            .flat_map(|dict| {
                let data: Vec<f32> = dict.data_vec.iter().flat_map(|x| x.data.clone()).collect();
                let ncols = dict.data_vec.len();
                let nrows = data.len() / ncols;
                DMatrix::from_vec(nrows, ncols, data)
                    .column_mean()
                    .data
                    .as_vec()
                    .clone()
            })
            .collect::<Vec<_>>();

        let ncols = self.num_batches();
        let nrows = batch_data.len() / ncols;
        let batch_features = DMatrix::<f32>::from_vec(nrows, ncols, batch_data);

        info!(
            "built feature matrix across batches: {} x {}",
            batch_features.nrows(),
            batch_features.ncols()
        );

        let nbatches = self.num_batches();
        let batches = (0..nbatches).collect();

        let dict = ColumnDict::<usize>::from_dvector_views(
            batch_features.column_iter().collect(),
            batches,
        );

        let ret: Vec<Vec<usize>> = (0..nbatches)
            .into_par_iter()
            .map(|b| {
                dict.search_by_query_name(&b, nbatches, false)
                    .map(|(others, _)| others)
            })
            .collect::<anyhow::Result<Vec<Vec<usize>>>>()?;
        self.derived.between_batch_proximity = Some(ret);

        Ok(())
    }

    pub fn batch_name_map(&self) -> Option<HashMap<Box<str>, usize>> {
        self.derived.batch_idx_to_name.as_ref().map(|names| {
            names
                .iter()
                .enumerate()
                .map(|(idx, name)| (name.clone(), idx))
                .collect::<HashMap<Box<str>, usize>>()
        })
    }

    pub fn num_batches(&self) -> usize {
        if let Some(v) = &self.derived.batch_to_cols {
            v.len()
        } else if let Some(v) = &self.derived.batch_knn_lookup {
            v.len()
        } else {
            0
        }
    }

    /// Borrow the per-batch HNSW lookups populated by
    /// `build_hnsw_per_batch` / `register_batches_dmatrix`. Returns `None`
    /// before the indices have been built.
    pub fn batch_knn_lookup(&self) -> Option<&Vec<ColumnDict<usize>>> {
        self.derived.batch_knn_lookup.as_ref()
    }

    /// Register batch membership information without building HNSW
    /// indices. This is a lightweight alternative to `register_batches_dmatrix`
    /// for use with pb-sample based batch correction.
    pub fn register_batch_membership<T>(&mut self, batch_membership: &[T])
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let batches = partition_by_membership(batch_membership, None);
        let ntot = self.num_columns();
        let mut col_to_batch = vec![0; ntot];

        let mut sorted_batches: Vec<_> = batches.into_iter().collect();
        sorted_batches.sort_by(|a, b| a.0.to_string().cmp(&b.0.to_string()));

        let mut batch_names = Vec::with_capacity(sorted_batches.len());
        let mut batch_to_cols = Vec::with_capacity(sorted_batches.len());

        for (batch_idx, (batch_name, glob_indices)) in sorted_batches.into_iter().enumerate() {
            for &cell in &glob_indices {
                col_to_batch[cell] = batch_idx;
            }
            batch_names.push(batch_name.to_string().into_boxed_str());
            batch_to_cols.push(glob_indices);
        }

        self.derived.col_to_batch = Some(col_to_batch);
        self.derived.batch_to_cols = Some(batch_to_cols);
        self.derived.batch_idx_to_name = Some(batch_names);
    }

    pub fn batch_names(&self) -> Option<Vec<Box<str>>> {
        self.derived.batch_idx_to_name.clone()
    }

    pub fn batch_to_columns(&self, batch: usize) -> Option<&Vec<usize>> {
        if let Some(batch_to_cols) = &self.derived.batch_to_cols {
            Some(&batch_to_cols[batch])
        } else {
            None
        }
    }

    pub fn get_batch_membership<I>(&self, cells: I) -> Vec<usize>
    where
        I: Iterator<Item = usize>,
    {
        let cell_to_batch = self
            .derived
            .col_to_batch
            .as_ref()
            .expect("cell_to_batch not initialized");
        cells.into_iter().map(|c| cell_to_batch[c]).collect()
    }

    pub fn column_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        debug_assert_eq!(self.num_columns(), self.column_names_with_data_tag.len());
        Ok(self.column_names_with_data_tag.clone())
    }
}
