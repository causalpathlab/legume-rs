// #![allow(dead_code)]

use crate::srt_common::*;
use candle_util::candle_data_loader::generate_minibatch_intervals;

pub struct PairsWithBounds<'a> {
    pub lb: usize,
    pub ub: usize,
    pub pairs: &'a [(usize, usize)],
}

pub struct PairsWithIndices<'a> {
    pub indices: &'a Vec<usize>,
    pub pairs: &'a Vec<(usize, usize)>,
}

pub struct SrtCellPairs<'a> {
    pub data: &'a SparseIoVec,
    pub coordinates: &'a Mat,
    pub pairs: Vec<(usize, usize)>,
    pub distances: Vec<f32>,
    pub pair_to_sample: Option<Vec<usize>>,
    pub sample_to_pair: Option<Vec<Vec<usize>>>,
}

impl<'a> SrtCellPairs<'a> {
    /// number of pairs
    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
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
            .map(|&(i, _)| self.coordinates.row(i))
            .collect::<Vec<_>>();
        let right = self
            .pairs
            .iter()
            .map(|&(_, j)| self.coordinates.row(j))
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
            .for_each(|&(l, r)| {
                left += self.coordinates.row(l).transpose();
                right += self.coordinates.row(r).transpose();
                npairs += 1.;
            });

        left /= npairs;
        right /= npairs;
        (left, right)
    }

    /// visit cell pairs by regular-sized block
    pub fn visit_pairs_by_block<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
        block_size: usize,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(
                PairsWithBounds,
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
                let pairs_per_job = &all_pairs[lb..ub];
                visitor(
                    PairsWithBounds {
                        lb,
                        ub,
                        pairs: pairs_per_job,
                    },
                    &self,
                    shared_in,
                    arc_shared_out.clone(),
                )
            })
            .collect::<anyhow::Result<()>>()
    }

    /// visit cell pairs by sample
    pub fn visit_pairs_by_sample<Visitor, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_out: &mut SharedOut,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(
                PairsWithIndices,
                &SrtCellPairs,
                usize,
                Arc<Mutex<&mut SharedOut>>,
            ) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedOut: Sync + Send,
    {
        if let Some(sample_to_pair) = self.sample_to_pair.as_ref() {
            let arc_shared_out = Arc::new(Mutex::new(shared_out));
            let num_samples = sample_to_pair.len();
            let all_pairs = &self.pairs;
            sample_to_pair
                .into_par_iter()
                .enumerate()
                .progress_count(num_samples as u64)
                .map(|(sample, indices)| {
                    let pairs = &indices
                        .into_iter()
                        .filter_map(|&i| all_pairs.get(i).cloned())
                        .collect::<Vec<_>>();

                    visitor(
                        PairsWithIndices { indices, pairs },
                        &self,
                        sample,
                        arc_shared_out.clone(),
                    )
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
    /// * `coordinates` - n x 2 or n x 3 spatial coordinates
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
            .map(|(lb, ub)| -> anyhow::Result<Vec<(usize, usize, f32)>> {
                let mut ret = Vec::with_capacity((ub - lb) * nquery);

                for i in lb..ub {
                    let (_indices, _distances) = dict.search_others(&i, nquery)?;
                    ret.extend(
                        _indices
                            .into_iter()
                            .zip(_distances)
                            .map(|(j, d_ij)| (i, j, d_ij))
                            .collect::<Vec<_>>(),
                    );
                }

                Ok(ret)
            })
            .flatten()
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        if triplets.len() < 1 {
            return Err(anyhow::anyhow!("empty triplets"));
        }

        info!("{} triplets for spatial kNN matching", triplets.len());

        let pairs = triplets.iter().map(|&(i, j, _)| (i, j)).collect::<Vec<_>>();
        let distances = triplets.iter().map(|&(_, _, d)| d).collect::<Vec<_>>();

        Ok(SrtCellPairs {
            data,
            coordinates,
            pairs,
            distances,
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
