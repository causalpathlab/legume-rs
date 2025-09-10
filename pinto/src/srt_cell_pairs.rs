use crate::srt_common::*;
use candle_util::candle_data_loader::generate_minibatch_intervals;

pub struct Pair {
    pub left: usize,
    pub right: usize,
}

pub struct SrtCellPairs<'a> {
    pub data: &'a SparseIoVec,
    pub coordinates: &'a Mat,
    pub pairs: Vec<Pair>,
    pub pairs_neighbours: Vec<PairsNeighbours>,
    pub distances: Vec<f32>,
    pub pair_to_sample: Option<Vec<usize>>,
    pub sample_to_pair: Option<Vec<Vec<usize>>>,
}

pub struct PairsNeighbours {
    pub left_only: Vec<usize>,
    pub right_only: Vec<usize>,
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
        let (left, right) = self.all_pairs_positions()?;

        let coordinate_names = coordinate_names.unwrap_or(
            (0..self.num_coordinates())
                .map(|x| x.to_string().into_boxed_str())
                .collect(),
        );

        if coordinate_names.len() != self.num_coordinates() {
            return Err(anyhow::anyhow!("invalid coordinate names"));
        }

        let paired_column_names = coordinate_names
            .iter()
            .map(|x| format!("left_{}", x).into_boxed_str())
            .chain(
                coordinate_names
                    .iter()
                    .map(|x| format!("right_{}", x).into_boxed_str()),
            )
            .collect::<Vec<_>>();

        concatenate_horizontal(&[left, right])?.to_parquet(
            None,
            Some(&paired_column_names),
            file_path,
        )?;

        Ok(())
    }

    ///
    /// Take all pairs' positions
    ///
    /// returns `(left_coordinates, right_coordinates)`
    ///
    pub fn all_pairs_positions(&self) -> anyhow::Result<(Mat, Mat)> {
        let left = self
            .pairs
            .iter()
            .map(|pp| self.coordinates.row(pp.left))
            .collect::<Vec<_>>();
        let right = self
            .pairs
            .iter()
            .map(|pp| self.coordinates.row(pp.right))
            .collect::<Vec<_>>();
        Ok((concatenate_vertical(&left)?, concatenate_vertical(&right)?))
    }

    ///
    /// Take average position of these cell pairs (indices)
    ///
    /// returns `(left_coordinate, right_coordinate)`
    ///
    pub fn average_position(&self, select_pair_indices: &[usize]) -> (DVec, DVec) {
        let mut left = DVec::zeros(self.coordinates.ncols());
        let mut right = DVec::zeros(self.coordinates.ncols());
        let mut npairs = 0_f32;

        select_pair_indices
            .into_iter()
            .filter_map(|&pp| self.pairs.get(pp))
            .for_each(|pp| {
                left += self.coordinates.row(pp.left).transpose();
                right += self.coordinates.row(pp.right).transpose();
                npairs += 1.;
            });

        left /= npairs;
        right /= npairs;
        (left, right)
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
        SharedIn: Sync + Send,
        SharedOut: Sync + Send,
    {
        let all_pairs = &self.pairs;
        let ntot = all_pairs.len();
        let jobs = generate_minibatch_intervals(ntot, block_size);
        let arc_shared_out = Arc::new(Mutex::new(shared_out));

        jobs.par_iter()
            .progress_count(jobs.len() as u64)
            .map(|&(lb, ub)| -> anyhow::Result<()> {
                visitor((lb, ub), &self, shared_in, arc_shared_out.clone())
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
    pub fn visit_pairs_by_sample<Visitor, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_out: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(&[usize], &SrtCellPairs, usize, Arc<Mutex<&mut SharedOut>>) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedOut: Sync + Send,
    {
        if let Some(sample_to_pair) = self.sample_to_pair.as_ref() {
            let arc_shared_out = Arc::new(Mutex::new(shared_out));
            let num_samples = sample_to_pair.len();
            sample_to_pair
                .into_par_iter()
                .enumerate()
                .progress_count(num_samples as u64)
                .map(|(sample, indices)| visitor(&indices, &self, sample, arc_shared_out.clone()))
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
        knn: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<SrtCellPairs<'a>> {
        let nn = coordinates.nrows();

        if data.num_columns()? != nn {
            return Err(anyhow::anyhow!("incompatible data and coordinates"));
        }

        let points = coordinates.transpose();
        let points = points.column_iter().collect::<Vec<_>>();
        let names = (0..nn).collect::<Vec<_>>();

        let dict = ColumnDict::from_dvector_views(points, names);
        let nquery = (knn + 1).min(nn).max(2);

        let jobs = create_jobs(nn, block_size);
        let njobs = jobs.len() as u64;

        let triplets = jobs
            .into_par_iter()
            .progress_count(njobs)
            .map(|(lb, ub)| -> anyhow::Result<Vec<((usize, usize), f32)>> {
                let mut ret = Vec::with_capacity((ub - lb) * nquery);

                for i in lb..ub {
                    let (_indices, _distances) = dict.search_others(&i, nquery)?;
                    ret.extend(
                        _indices
                            .into_iter()
                            .zip(_distances)
                            .map(|(j, d_ij)| ((i, j), d_ij))
                            .collect::<Vec<_>>(),
                    );
                }

                Ok(ret)
            })
            .flatten()
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<HashMap<_, _>>();

        info!("{} triplets by spatial kNN matching", triplets.len());

        if triplets.len() < 1 {
            return Err(anyhow::anyhow!("empty triplets"));
        }

        let keys = triplets
            .iter()
            .map(|entry| *entry.key())
            .collect::<Vec<_>>();

        let mut triplets = keys
            .into_par_iter()
            .filter_map(|(i, j)| {
                // if the reciprocal key (j, i) exists
                if triplets.contains_key(&(j, i)) {
                    if let Some(x) = triplets.get(&(j, i)) {
                        if i < j {
                            return Some(((i, j), *x.value()));
                        } else if j < i {
                            return Some(((j, i), *x.value()));
                        }
                    }
                }
                None
            })
            .collect::<Vec<_>>();

        triplets.par_sort_by_key(|&(ij, _)| ij);
        triplets.dedup();
        info!("{} triplets after reciprocal matching", triplets.len());

        // construct sparse network backbone
        use nalgebra_sparse::{CooMatrix, CscMatrix};

        let n = data.num_columns()?;
        let mut coo = CooMatrix::new(n, n);
        for &((i, j), v) in triplets.iter() {
            coo.push(i, j, v);
            coo.push(j, i, v);
        }

        let graph = CscMatrix::from(&coo);

        info!("compiling the list of neighbouring cells for all the pairs");
        let neighbours = |i: usize| -> HashSet<usize> {
            graph
                .get_col(i)
                .unwrap()
                .row_indices()
                .iter()
                .cloned()
                .collect()
        };

        let triplets_with_neighbours = triplets
            .into_iter()
            .par_bridge()
            .filter_map(
                |((left, right), v)| -> Option<((usize, usize), f32, PairsNeighbours)> {
                    let n_left = neighbours(left);
                    let n_right = neighbours(right);
                    // remove left-right edge
                    n_left.remove(&right);
                    n_right.remove(&left);

                    let s_ij = n_left
                        .iter()
                        .filter_map(|x| {
                            if n_right.contains(x.key()) {
                                Some(*x)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();

                    // add self-loop
                    n_left.insert(left);
                    n_right.insert(right);

                    let left_only = n_left
                        .iter()
                        .filter_map(|x| {
                            if !n_right.contains(x.key()) && !s_ij.contains(x.key()) {
                                Some(*x)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();

                    let right_only = n_right
                        .iter()
                        .filter_map(|x| {
                            if !n_left.contains(x.key()) && !s_ij.contains(x.key()) {
                                Some(*x)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();

                    if left_only.is_empty() || right_only.is_empty() {
                        return None;
                    }

                    Some((
                        (left, right),
                        v,
                        PairsNeighbours {
                            left_only,
                            right_only,
                        },
                    ))
                },
            )
            .collect::<Vec<_>>();

        if triplets_with_neighbours.len() < 1 {
            return Err(anyhow::anyhow!(
                "unable to extract two types of neighbours (increase `--knn-spatial`)"
            ));
        }

        let pairs = triplets_with_neighbours
            .iter()
            .map(|&((i, j), _, _)| Pair { left: i, right: j })
            .collect::<Vec<_>>();
        let distances = triplets_with_neighbours
            .iter()
            .map(|&(_, x, _)| x)
            .collect::<Vec<_>>();
        let pairs_neighbours = triplets_with_neighbours
            .into_iter()
            .map(|(_, _, x)| x)
            .collect::<Vec<_>>();

        Ok(SrtCellPairs {
            data,
            coordinates,
            pairs,
            distances,
            pairs_neighbours,
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
}
