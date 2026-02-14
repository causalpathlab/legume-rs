use crate::srt_common::*;
use crate::srt_knn_graph::{KnnGraph, KnnGraphArgs};
use crate::srt_random_projection::{SrtRandProjOps, SrtRandProjOut};
use matrix_util::parquet::*;
use matrix_util::utils::generate_minibatch_intervals;

use parquet::basic::Type as ParquetType;

pub struct SrtCellPairs<'a> {
    pub data: &'a SparseIoVec,
    pub coordinates: &'a Mat,
    pub graph: KnnGraph,
    pub pairs: Vec<Pair>,
    pub coordinate_embedding: Mat,
    pub pair_to_sample: Option<Vec<usize>>,
    pub sample_to_pair: Option<Vec<Vec<usize>>>,
}

pub struct CollapsePairsArgs<'a> {
    pub proj_out: &'a SrtRandProjOut,
    pub finest_sort_dim: usize,
    pub num_levels: usize,
    pub down_sample: Option<usize>,
}

pub struct SrtCellPairsArgs {
    pub knn: usize,
    pub coordinate_emb_dim: usize,
    pub block_size: usize,
}

impl<'a> SrtCellPairs<'a> {
    /// number of pairs
    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Write all the coordinate pairs into `.parquet` file
    /// * `file_path`: destination file name (try to include a recognizable extension in the end, e.g., `.parquet`)
    /// * `coordinate_names`: column names for the left (`left_{}`) and right (`right_{}`) where each `{}` will be replaced with the corresponding column name
    pub fn to_parquet(
        &self,
        file_path: &str,
        coordinate_names: Option<Vec<Box<str>>>,
    ) -> anyhow::Result<()> {
        let coordinate_names = coordinate_names.unwrap_or(
            (0..self.num_coordinates())
                .map(|x| x.to_string().into_boxed_str())
                .collect(),
        );

        if coordinate_names.len() != self.num_coordinates() {
            return Err(anyhow::anyhow!("invalid coordinate names"));
        }

        let mut column_names = vec![];
        let mut column_types = vec![];

        // 1. left data column names
        column_names.push("left_cell".to_string().into_boxed_str());
        column_types.push(ParquetType::BYTE_ARRAY);

        // 2. right data column names
        column_names.push("right_cell".to_string().into_boxed_str());
        column_types.push(ParquetType::BYTE_ARRAY);

        // 3. left coordinate names
        for x in coordinate_names.iter() {
            column_names.push(format!("left_{}", x).into_boxed_str());
            column_types.push(ParquetType::FLOAT);
        }

        // 4. right coordinate names
        for x in coordinate_names.iter() {
            column_names.push(format!("right_{}", x).into_boxed_str());
            column_types.push(ParquetType::FLOAT);
        }

        // 5. distance
        column_names.push("distance".to_string().into_boxed_str());
        column_types.push(ParquetType::FLOAT);

        //////////////////////////////////////
        // write them down column by column //
        //////////////////////////////////////

        let shape = (self.num_pairs(), column_names.len());

        let writer = ParquetWriter::new(
            file_path,
            shape,
            (None, Some(&column_names)),
            Some(&column_types),
            Some("cell_pair"),
        )?;
        let row_names = writer.row_names_vec();

        let mut writer = writer.get_writer()?;
        let mut row_group_writer = writer.next_row_group()?;

        // 0. row names
        parquet_add_bytearray(&mut row_group_writer, row_names)?;

        let cell_names = self.data.column_names()?;
        // 1. left column names
        parquet_add_string_column(
            &mut row_group_writer,
            &self
                .pairs
                .iter()
                .map(|pp| cell_names[pp.left].clone())
                .collect::<Vec<_>>(),
        )?;

        // 2. right column names
        parquet_add_string_column(
            &mut row_group_writer,
            &self
                .pairs
                .iter()
                .map(|pp| cell_names[pp.right].clone())
                .collect::<Vec<_>>(),
        )?;

        let (left_coord, right_coord) = self.all_pairs_positions()?;

        // 3. left coordinates
        for coord in left_coord.iter() {
            parquet_add_numeric_column(&mut row_group_writer, coord)?;
        }

        // 4. right coordinates
        for coord in right_coord.iter() {
            parquet_add_numeric_column(&mut row_group_writer, coord)?;
        }

        // 5. distances
        parquet_add_numeric_column(&mut row_group_writer, &self.graph.distances)?;

        row_group_writer.close()?;
        writer.close()?;
        Ok(())
    }

    ///
    /// Take all pairs' positions
    ///
    /// returns `(left_coordinates, right_coordinates)`
    ///
    #[allow(clippy::type_complexity)]
    pub fn all_pairs_positions(&self) -> anyhow::Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let left: Vec<Vec<f32>> = self
            .coordinates
            .column_iter()
            .map(|coord_vec| self.pairs.iter().map(|pp| coord_vec[pp.left]).collect())
            .collect();
        let right: Vec<Vec<f32>> = self
            .coordinates
            .column_iter()
            .map(|coord_vec| self.pairs.iter().map(|pp| coord_vec[pp.right]).collect())
            .collect();

        Ok((left, right))
    }

    /// put together coordinate embedding results for all the pairs
    pub fn coordinate_embedding_pairs(&self) -> anyhow::Result<Mat> {
        concatenate_vertical(
            &self
                .pairs
                .iter()
                .map(|pp| {
                    self.coordinate_embedding.row(pp.left) + self.coordinate_embedding.row(pp.right)
                })
                .collect::<Vec<_>>(),
        )
    }

    /// visit cell pairs by regular-sized block
    ///
    /// A visitor function takes
    /// - `(lb,ub)` `(usize,usize)`
    /// - data itself
    /// - `shared_input`
    /// - `shared_out` (`Arc(Mutex())`)
    pub fn visit_pairs_by_block<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
        block_size: usize,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(
                (usize, usize),
                &SrtCellPairs,
                &SharedIn,
                Arc<Mutex<&mut SharedOut>>,
            ) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedIn: Sync + Send + ?Sized,
        SharedOut: Sync + Send,
    {
        let all_pairs = &self.pairs;
        let ntot = all_pairs.len();
        let jobs = generate_minibatch_intervals(ntot, block_size);
        let arc_shared_out = Arc::new(Mutex::new(shared_out));

        jobs.par_iter()
            .progress_count(jobs.len() as u64)
            .map(|&(lb, ub)| -> anyhow::Result<()> {
                visitor((lb, ub), self, shared_in, arc_shared_out.clone())
            })
            .collect::<anyhow::Result<()>>()
    }

    /// visit cell pairs by sample
    ///
    /// A visitor function takes
    /// - `pair_id` `&[usize]`
    /// - data itself
    /// - `sample_id`
    /// - `shared_out` (`Arc(Mutex())`)
    pub fn visit_pairs_by_sample<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(
                &[usize],
                &SrtCellPairs,
                usize,
                &SharedIn,
                Arc<Mutex<&mut SharedOut>>,
            ) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedIn: Sync + Send + ?Sized,
        SharedOut: Sync + Send,
    {
        if let Some(sample_to_pair) = self.sample_to_pair.as_ref() {
            let arc_shared_out = Arc::new(Mutex::new(shared_out));
            let num_samples = sample_to_pair.len();
            sample_to_pair
                .into_par_iter()
                .enumerate()
                .progress_count(num_samples as u64)
                .map(|(sample, indices)| {
                    visitor(indices, self, sample, shared_in, arc_shared_out.clone())
                })
                .collect()
        } else {
            Err(anyhow::anyhow!("no sample was assigned"))
        }
    }

    pub fn num_coordinates(&self) -> usize {
        self.coordinates.ncols()
    }

    pub fn num_samples(&self) -> anyhow::Result<usize> {
        let sample_to_pair = self
            .sample_to_pair
            .as_ref()
            .ok_or(anyhow::anyhow!("no sample was assigned"))?;

        Ok(sample_to_pair.len())
    }

    ///
    /// Create a thin wrapper for cell pairs
    ///
    /// * `data` - sparse matrix data vector
    /// * `coordinates` - n x 2 or n x 3 or more spatial coordinates
    /// * `knn` - k-nearest neighbours
    /// * `block_size` block size for parallel processing
    ///
    pub fn new(
        data: &'a SparseIoVec,
        coordinates: &'a Mat,
        args: SrtCellPairsArgs,
    ) -> anyhow::Result<SrtCellPairs<'a>> {
        let nn = coordinates.nrows();

        if data.num_columns() != nn {
            return Err(anyhow::anyhow!("incompatible data and coordinates"));
        }

        let points = coordinates.transpose();

        let graph = KnnGraph::from_columns(
            &points,
            KnnGraphArgs {
                knn: args.knn,
                block_size: args.block_size,
                reciprocal: true,
            },
        )?;

        /////////////////////////////////////////////
        // precompute positional embedding          //
        /////////////////////////////////////////////

        let coordinate_embedding =
            coordinates.positional_embedding_columns(args.coordinate_emb_dim)?;

        let pairs = graph
            .edges
            .iter()
            .map(|&(i, j)| Pair { left: i, right: j })
            .collect::<Vec<_>>();

        Ok(SrtCellPairs {
            data,
            coordinates,
            graph,
            pairs,
            coordinate_embedding,
            pair_to_sample: None,
            sample_to_pair: None,
        })
    }

    /// number of pairs
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// assign pairs to samples
    pub fn assign_samples(&mut self, pair_to_sample: Vec<usize>, npairs_per_sample: Option<usize>) {
        self.sample_to_pair = Some(
            partition_by_membership(&pair_to_sample, npairs_per_sample)
                .into_values()
                .collect(),
        );
        self.pair_to_sample = Some(pair_to_sample);
    }

    /// Multi-level collapsing for pair-based pipelines.
    ///
    /// For each level (coarse → fine): re-assign pairs at that sort_dim,
    /// create a fresh stat via `stat_factory`, visit all samples, collect.
    /// Returns `Vec<SharedOut>` from coarsest to finest.
    pub fn collapse_pairs_multilevel<Visitor, SharedIn, SharedOut>(
        &mut self,
        collapse_args: &CollapsePairsArgs<'_>,
        visitor: &Visitor,
        shared_in: &SharedIn,
        stat_factory: impl Fn(usize) -> SharedOut,
    ) -> anyhow::Result<Vec<SharedOut>>
    where
        Visitor: Fn(
                &[usize],
                &SrtCellPairs,
                usize,
                &SharedIn,
                Arc<Mutex<&mut SharedOut>>,
            ) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedIn: Sync + Send + ?Sized,
        SharedOut: Sync + Send,
    {
        let level_dims = compute_level_sort_dims(
            collapse_args.finest_sort_dim,
            collapse_args.num_levels,
        );
        let mut results = Vec::with_capacity(level_dims.len());

        for (level, &sort_dim) in level_dims.iter().enumerate() {
            info!(
                "Level {}/{}: sort_dim={}",
                level + 1,
                level_dims.len(),
                sort_dim
            );
            self.assign_pairs_to_samples(
                collapse_args.proj_out,
                Some(sort_dim),
                collapse_args.down_sample,
            )?;
            let mut stat = stat_factory(self.num_samples()?);
            self.visit_pairs_by_sample(visitor, shared_in, &mut stat)?;
            results.push(stat);
        }

        Ok(results)
    }
}
