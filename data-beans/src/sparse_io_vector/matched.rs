use super::*;

impl SparseIoVec {
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
    fn matched_columns_triplets_on_one_target<I>(
        &self,
        cells: I,
        target_batch: usize,
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<TripletsMatched>
    where
        I: Iterator<Item = usize> + Clone,
    {
        let lookups = self
            .batch_knn_lookup
            .as_ref()
            .ok_or(anyhow::anyhow!("no knn lookup"))?;

        let cell_to_batch = self
            .col_to_batch
            .as_ref()
            .ok_or(anyhow::anyhow!("no cell to batch"))?;

        debug_assert!(target_batch < self.num_batches());

        let nrow = self.num_rows();
        let mut ncol = 0;
        let mut triplets = Vec::new();
        let mut distances = Vec::new();
        let mut source_columns = Vec::new();
        let mut matched_columns = Vec::new();

        for glob in cells {
            let source_batch = cell_to_batch[glob]; // this cell's batch

            if skip_same_batch && source_batch == target_batch {
                continue; // skip cells in the same batch
            }

            if let (Some(source_lookup), Some(target_lookup)) =
                (lookups.get(source_batch), lookups.get(target_batch))
            {
                let (matched, matched_distances) =
                    source_lookup.match_by_query_name_against(&glob, knn, target_lookup)?;
                for (glob_matched, dist) in matched.into_iter().zip(matched_distances.into_iter()) {
                    if glob == glob_matched {
                        continue; // avoid identical cell pairs
                    }
                    self.read_column_offset(glob_matched, &mut ncol, &mut triplets)?;
                    source_columns.push(glob);
                    matched_columns.push(glob_matched);
                    distances.push(dist);
                }
            }
        }

        Ok(TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            matched_columns,
            distances,
        })
    }

    /// Take columns matched with the given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batches` - the batches for targeted kNN search
    /// * `knn_columns` - k-nearest neighbours of columns
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * shape - (nrows, ncols)
    /// * triplets - a vector of triplets
    /// * `source_columns` - a vector of the source columns
    /// * distance - a vector of distances between the matched columns
    fn matched_columns_triplets<I>(
        &self,
        cells: I,
        target_batches: &[usize],
        knn_columns: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<TripletsMatched>
    where
        I: Iterator<Item = usize>,
    {
        let cells: Vec<usize> = cells.collect();

        let nrows = self.num_rows();
        let nbatches = self.num_batches();
        let ncols = cells.len();
        let approx_ncols = ncols * knn_columns * nbatches;

        let mut tot_triplets: Vec<(u64, u64, f32)> = Vec::with_capacity(approx_ncols);
        let mut tot_distances: Vec<f32> = Vec::with_capacity(approx_ncols);
        let mut tot_sources: Vec<usize> = Vec::with_capacity(approx_ncols);
        let mut tot_matched: Vec<usize> = Vec::with_capacity(approx_ncols);
        let mut tot_ncells_matched: usize = 0;

        for &target_b in target_batches.iter() {
            let TripletsMatched {
                shape,
                triplets,
                source_columns,
                matched_columns,
                distances,
            } = self.matched_columns_triplets_on_one_target(
                cells.iter().cloned(),
                target_b,
                knn_columns,
                skip_same_batch,
            )?;

            tot_triplets.extend(
                triplets
                    .into_iter()
                    .map(|(i, j, z_ij)| (i, j + (tot_ncells_matched as u64), z_ij)),
            );

            tot_distances.extend(distances);
            tot_ncells_matched += shape.1;
            tot_sources.extend(source_columns);
            tot_matched.extend(matched_columns);
        }

        let shape = (nrows, tot_ncells_matched);

        Ok(TripletsMatched {
            shape,
            triplets: tot_triplets,
            source_columns: tot_sources,
            matched_columns: tot_matched,
            distances: tot_distances,
        })
    }

    /// Take columns with the neighbourhood of given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `knn_batches` - k-nearest neighbour batches
    /// * `knn_columns` - k-nearest neighbour columns
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * shape - (nrows, ncols)
    /// * triplets
    /// * source positions
    /// * distances
    fn neighbouring_columns_triplets<I>(
        &self,
        cells: I,
        knn_batches: usize,
        knn_columns: usize,
        skip_same_batch: bool,
        skip_batches: Option<&[usize]>,
    ) -> anyhow::Result<TripletsMatched>
    where
        I: Iterator<Item = usize>,
    {
        let lookups = self
            .batch_knn_lookup
            .as_ref()
            .ok_or(anyhow::anyhow!("no knn lookup"))?;

        let cell_to_batch = self
            .col_to_batch
            .as_ref()
            .ok_or(anyhow::anyhow!("no cell to batch"))?;

        let approx_ncol = knn_columns * knn_batches;

        let nrow = self.num_rows();
        let mut ncol = 0_usize;
        let mut triplets = Vec::with_capacity(approx_ncol * nrow);

        let mut distances = Vec::with_capacity(approx_ncol);
        let mut source_columns = Vec::with_capacity(approx_ncol);
        let mut matched_columns = Vec::with_capacity(approx_ncol);

        let nbatches = self.num_batches();

        // Pre-compute neighbouring batches per source batch
        let neighbouring_batches_by_source: Vec<Vec<usize>> = (0..nbatches)
            .map(|source_batch| match self.between_batch_proximity.as_ref() {
                Some(prox) => prox[source_batch]
                    .iter()
                    .copied()
                    .filter(|&b| skip_batches.is_none_or(|skip| !skip.contains(&b)))
                    .filter(|&b| !skip_same_batch || b != source_batch)
                    .collect(),
                _ => (0..nbatches)
                    .filter(|&b| skip_batches.is_none_or(|skip| !skip.contains(&b)))
                    .filter(|&b| !skip_same_batch || b != source_batch)
                    .collect(),
            })
            .collect();

        for glob_index in cells {
            let source_batch = cell_to_batch[glob_index];

            for &target_batch in &neighbouring_batches_by_source[source_batch] {
                if let (Some(source_lookup), Some(target_lookup)) =
                    (lookups.get(source_batch), lookups.get(target_batch))
                {
                    let (matched, matched_distances) = source_lookup.match_by_query_name_against(
                        &glob_index,
                        knn_columns,
                        target_lookup,
                    )?;
                    for (glob_matched_index, dist) in
                        matched.into_iter().zip(matched_distances.into_iter())
                    {
                        if glob_index == glob_matched_index {
                            continue;
                        }
                        self.read_column_offset(glob_matched_index, &mut ncol, &mut triplets)?;
                        source_columns.push(glob_index);
                        matched_columns.push(glob_matched_index);
                        distances.push(dist);
                    }
                }
            }
        }

        Ok(TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            matched_columns,
            distances,
        })
    }

    fn query_columns_by_data_triplets<T>(
        &self,
        query: T,
        knn_per_batch: usize,
    ) -> anyhow::Result<TripletsMatched>
    where
        T: MakeVecPoint,
    {
        let lookups = self
            .batch_knn_lookup
            .as_ref()
            .ok_or(anyhow::anyhow!("no knn lookup"))?;

        let nrow = self.num_rows();
        let mut ncol = 0_usize;

        let approx_knn = self.num_batches() * knn_per_batch;
        let mut triplets = Vec::with_capacity(approx_knn);
        let mut source_columns = Vec::with_capacity(approx_knn);
        let mut matched_columns = Vec::with_capacity(approx_knn);
        let mut distances = Vec::with_capacity(approx_knn);

        for lookup in lookups {
            let (matched, matched_distances) =
                lookup.search_by_query_data(&query.to_vp(), knn_per_batch)?;

            for (&glob_idx, &dist) in matched.iter().zip(matched_distances.iter()) {
                self.read_column_offset(glob_idx, &mut ncol, &mut triplets)?;
                source_columns.push(glob_idx);
                matched_columns.push(glob_idx);
                distances.push(dist);
            }
        }

        Ok(TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            matched_columns,
            distances,
        })
    }

    /// Take columns within the neighbourhood of given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batches` - the batches for targeted kNN search
    /// * `knn_batches` - k-nearest neighbour batches
    /// * `knn_columns` - k-nearest neighbour columns
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * the knn-matched matrix
    /// * `source_columns` - a vector of the source columns
    /// * a vector of distances between the matched columns
    #[allow(clippy::type_complexity)]
    pub fn read_neighbouring_columns_csc<I>(
        &self,
        cells: I,
        knn_batches: usize,
        knn_columns: usize,
        skip_same_batch: bool,
        skip_batches: Option<&[usize]>,
    ) -> anyhow::Result<(CscMatrix<f32>, Vec<usize>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            matched_columns,
            distances,
        } = self.neighbouring_columns_triplets(
            cells,
            knn_batches,
            knn_columns,
            skip_same_batch,
            skip_batches,
        )?;

        Ok((
            CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)?,
            source_columns,
            matched_columns,
            distances,
        ))
    }

    /// Take columns neighbouring with the given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batches` - the batches for targeted kNN search
    /// * `knn` - k-nearest neighbours
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * the knn-neighbouring matrix
    /// * `source_columns` - a vector of the source columns
    /// * distances - a vector of distances between the neighbouring columns
    pub fn read_neighbouring_columns_ndarray<I>(
        &self,
        cells: I,
        knn_batches: usize,
        knn_columns: usize,
        skip_same_batch: bool,
        skip_batches: Option<&[usize]>,
    ) -> anyhow::Result<(ndarray::Array2<f32>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            distances,
            ..
        } = self.neighbouring_columns_triplets(
            cells,
            knn_batches,
            knn_columns,
            skip_same_batch,
            skip_batches,
        )?;
        Ok((
            ndarray::Array2::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)?,
            source_columns,
            distances,
        ))
    }

    /// Take columns neighbouring with the given `cells`
    ///
    /// # Arguments
    /// * `cells` - global column indices
    /// * `target_batches` - the batches for targeted kNN search
    /// * `knn` - k-nearest neighbours
    /// * `skip_same_batch` - skip the same batch
    ///
    /// # Returns
    /// * the knn-neighbouring matrix
    /// * `source_columns` - a vector of the source columns
    /// * distances - a vector of distances between the neighbouring columns
    pub fn read_neighbouring_columns_dmatrix<I>(
        &self,
        cells: I,
        knn_batches: usize,
        knn_columns: usize,
        skip_same_batch: bool,
        skip_batches: Option<&[usize]>,
    ) -> anyhow::Result<(nalgebra::DMatrix<f32>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            distances,
            ..
        } = self.neighbouring_columns_triplets(
            cells,
            knn_batches,
            knn_columns,
            skip_same_batch,
            skip_batches,
        )?;
        Ok((
            DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)?,
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
    /// * a vector of distances between the matched columns
    pub fn read_matched_columns_csc<I>(
        &self,
        cells: I,
        target_batches: &[usize],
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(CscMatrix<f32>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            distances,
            ..
        } = self.matched_columns_triplets(cells, target_batches, knn, skip_same_batch)?;
        Ok((
            CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)?,
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
        target_batches: &[usize],
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(ndarray::Array2<f32>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            distances,
            ..
        } = self.matched_columns_triplets(cells, target_batches, knn, skip_same_batch)?;

        Ok((
            ndarray::Array2::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)?,
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
        target_batches: &[usize],
        knn: usize,
        skip_same_batch: bool,
    ) -> anyhow::Result<(nalgebra::DMatrix<f32>, Vec<usize>, Vec<f32>)>
    where
        I: Iterator<Item = usize>,
    {
        let TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            distances,
            ..
        } = self.matched_columns_triplets(cells, target_batches, knn, skip_same_batch)?;

        Ok((
            DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)?,
            source_columns,
            distances,
        ))
    }

    /// Query columns with projection data
    ///
    /// # Arguments
    /// * `query` - a data vector
    /// * `knn_per_batch` - k-nearest neighbour columns per batch
    ///
    /// # Returns
    /// * the knn-matched matrix
    /// * `source_columns` - a vector of the source columns
    /// * a vector of distances between the matched columns
    pub fn query_columns_by_data_csc<T>(
        &self,
        query: T,
        knn_per_batch: usize,
    ) -> anyhow::Result<(CscMatrix<f32>, Vec<usize>, Vec<f32>)>
    where
        T: MakeVecPoint,
    {
        let TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            distances,
            ..
        } = self.query_columns_by_data_triplets(query, knn_per_batch)?;

        Ok((
            CscMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)?,
            source_columns,
            distances,
        ))
    }

    /// Query columns with projection data
    ///
    /// # Arguments
    /// * `query` - a data vector
    /// * `knn_per_batch` - k-nearest neighbour columns per batch
    ///
    /// # Returns
    /// * the knn-matched matrix
    /// * `source_columns` - a vector of the source columns
    /// * a vector of distances between the matched columns
    pub fn query_columns_by_data_ndarray<T>(
        &self,
        query: T,
        knn_per_batch: usize,
    ) -> anyhow::Result<(ndarray::Array2<f32>, Vec<usize>, Vec<f32>)>
    where
        T: MakeVecPoint,
    {
        let TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            distances,
            ..
        } = self.query_columns_by_data_triplets(query, knn_per_batch)?;

        Ok((
            ndarray::Array2::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)?,
            source_columns,
            distances,
        ))
    }

    /// Query columns with projection data
    ///
    /// # Arguments
    /// * `query` - a data vector
    /// * `knn_per_batch` - k-nearest neighbour columns per batch
    ///
    /// # Returns
    /// * the knn-matched matrix
    /// * `source_columns` - a vector of the source columns
    /// * a vector of distances between the matched columns
    pub fn query_columns_by_data_dmatrix<T>(
        &self,
        query: T,
        knn_per_batch: usize,
    ) -> anyhow::Result<(nalgebra::DMatrix<f32>, Vec<usize>, Vec<f32>)>
    where
        T: MakeVecPoint,
    {
        let TripletsMatched {
            shape: (nrow, ncol),
            triplets,
            source_columns,
            distances,
            ..
        } = self.query_columns_by_data_triplets(query, knn_per_batch)?;

        Ok((
            DMatrix::<f32>::from_nonzero_triplets(nrow, ncol, &triplets)?,
            source_columns,
            distances,
        ))
    }
}
