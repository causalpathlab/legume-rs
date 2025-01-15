use asap_data::sparse_io::*;
use indicatif::ParallelProgressIterator;
use indicatif::ProgressIterator;
use matrix_util::dmatrix_match::ColumnDict;
use matrix_util::dmatrix_rsvd::*;
use matrix_util::dmatrix_util::*;
use rand::seq::SliceRandom;
use std::sync::Arc;
use std::sync::Mutex;

pub type Data = dyn SparseIo<IndexIter = Vec<usize>>;
pub type Dict = ColumnDict<usize>;

type Mat = DMatrix<f32>;
type DVec = DVector<f32>;

const DEFAULT_BLOCK_SIZE: usize = 10;

/// Random projection for a vector of SparseIo data sets
///
///
///
pub struct RandProjVec<'a> {
    raw_data_vec: &'a Vec<Arc<Data>>,                  // data matrix vec
    block_size: usize,                                 // block size for parallel
    num_data: usize,                                   // number of raw data matrices
    rand_basis_kd: Option<DMatrix<f32>>,               // K x d random basis matrix
    rand_proj_kn: Option<DMatrix<f32>>,                // K x ncol projection results
    data_to_cells: Option<HashMap<usize, Vec<usize>>>, // map: data id -> { global ids }
    cell_to_data: Option<Vec<usize>>,                  // map: global cell id -> data id
    cell_to_sample: Option<Vec<usize>>,                // map: global cell id -> sample id
    batch_dictionaries: Option<Vec<Dict>>,             // dictionary for each batch
    cell_to_batch: Option<Vec<usize>>,                 // map: global cell id -> batch id
}

#[allow(dead_code)]
impl<'a> RandProjVec<'a> {
    ///
    /// Create a new instance of `RandProjVec`
    ///
    /// # Arguments
    /// * `_vec`: a vector of data sets to be concatenated (Arc<Data>). Each element contains a atomic reference to the data sets, so it can be shared across threads and cloned.
    /// * `_blk`: block size for parallel computation
    ///
    pub fn new(_vec: &'a Vec<Arc<Data>>, _blk: Option<usize>) -> anyhow::Result<Self> {
        let nb = _vec.len();
        let blk = _blk.unwrap_or(DEFAULT_BLOCK_SIZE);
        if nb < 1 {
            anyhow::bail!("no data set in the vector");
        }

        Ok(Self {
            raw_data_vec: _vec,
            block_size: blk,
            num_data: nb,
            rand_basis_kd: None,
            rand_proj_kn: None,
            data_to_cells: None,
            cell_to_data: None,
            cell_to_sample: None,
            batch_dictionaries: None,
            cell_to_batch: None,
        })
    }

    ///
    /// Step 0: Sample random projection basis matrix
    /// # Arguments
    /// * `dim`: target dimensionality
    ///
    pub fn step0_sample_basis_cbind(&mut self, dim: usize) -> anyhow::Result<()> {
        let first_data = &self.raw_data_vec[0];

        let nrow = first_data.num_rows().ok_or_else(|| {
            anyhow::anyhow!(
                "can't figure out #rows in the data: {}",
                first_data.get_backend_file_name()
            )
        })?;

        let rand_basis_kd = DMatrix::<f32>::rnorm(dim.min(nrow), nrow);

        for (ii, data) in self.raw_data_vec.iter().enumerate().skip(1) {
            let data_nrow = data.num_rows().ok_or_else(|| {
                anyhow::anyhow!(
                    "can't figure out #rows in the data: {}",
                    data.get_backend_file_name()
                )
            })?;
            if data_nrow != nrow {
                anyhow::bail!(
                    "#rows in the data #{} don't match with the first one ({})",
                    ii,
                    nrow
                );
            }
        }

        self.rand_basis_kd = Some(rand_basis_kd);

        Ok(())
    }

    ///
    /// Step 1: Create K x ncol projection by concatenating data across columns
    ///
    pub fn step1_proj_cbind(&mut self) -> anyhow::Result<()> {
        if let Some(rand_basis_kd) = &self.rand_basis_kd {
            let num_batch = self.num_data;
            let kk = rand_basis_kd.nrows();
            let nrow = rand_basis_kd.ncols();
            let ncol_tot = self.raw_data_vec.iter().fold(0, |ncols, data| -> usize {
                ncols + data.num_columns().expect("failed to figure out # cols")
            });

            // data set index to global membership and vice versa
            let mut data_to_cells: HashMap<usize, Vec<usize>> = HashMap::new();
            for b in 0..num_batch {
                data_to_cells.insert(b, Vec::new());
            }
            let mut cell_to_data: Vec<usize> = vec![0; ncol_tot];
            let mut offset = 0;

            // the results of random projection
            let mut rand_proj_kn = Mat::zeros(kk, ncol_tot);
            let arc_rand_proj_kn = Arc::new(Mutex::new(&mut rand_proj_kn));

            for (didx, raw_data) in self.raw_data_vec.iter().enumerate() {
                #[cfg(debug_assertions)]
                {
                    anyhow::ensure!(raw_data.num_rows().unwrap() == nrow);
                }
                let ncol_data = raw_data.num_columns().unwrap();
                let jobs = self.create_jobs(ncol_data);

                let basis_dk = rand_basis_kd.transpose();

                /////////////////////////////////
                // populate projection results //
                /////////////////////////////////

                jobs.par_iter()
                    .progress_count(jobs.len() as u64)
                    .for_each(|&job| {
                        let (lb, ub) = job;
                        let (lb_glob, ub_glob) = (lb + offset, ub + offset);

                        let mut xx_dm = raw_data
                            .read_columns_csc((lb..ub).collect())
                            .expect("failed to read columns");

                        xx_dm.normalize_columns_inplace();
                        let mut chunk = (xx_dm.transpose() * &basis_dk).transpose();
                        chunk.scale_columns_inplace();

                        {
                            arc_rand_proj_kn
                                .lock()
                                .expect("failed to lock proj")
                                .columns_range_mut(lb_glob..ub_glob)
                                .copy_from(&chunk);
                        }
                    });

                ////////////////////////////
                // sorted out the indexes //
                ////////////////////////////
                {
                    let cells = data_to_cells.get_mut(&didx).expect("batch glob index");
                    for &(lb, ub) in jobs.iter() {
                        let (lb_glob, ub_glob) = (lb + offset, ub + offset);
                        cells.extend(lb_glob..ub_glob);
                        cell_to_data.splice(lb_glob..ub_glob, (lb..ub).map(|_| didx));
                    }
                    cells.sort();
                }

                offset += ncol_data;
            } // for each raw data matrix

            // self.dictionaries = Some(cell_dictionaries);
            self.rand_proj_kn = Some(rand_proj_kn);
            self.data_to_cells = Some(data_to_cells);
            self.cell_to_data = Some(cell_to_data);

            Ok(())
        } else {
            Err(anyhow::anyhow!("Random basis matrix is not available"))
        }
    }

    pub fn step2_random_sorting_cbind(&mut self) -> anyhow::Result<()> {
        if let Some(proj_kn) = &self.rand_proj_kn {
            let kk = proj_kn.nrows();
            let nn = proj_kn.ncols();
            let (_, _, vt_nk) = proj_kn.rsvd(kk)?;

            let q_nk = vt_nk.scale_columns();

            let mut binary_codes = DVector::<usize>::zeros(nn);

            for k in 0..kk {
                let binary_shift = |x: f32| -> usize {
                    if x > 0.0 {
                        1 << k
                    } else {
                        0
                    }
                };

                binary_codes += q_nk.column(k).map(binary_shift);
            }

            let collapse_membership = binary_codes.data.as_vec().clone();
            self.cell_to_sample = Some(collapse_membership);
        } else {
            return Err(anyhow::anyhow!("Step 1 (projection) incomplete"));
        }

        Ok(())
    }

    /// Build fast look-up dictionary for each batch
    /// using the previous random projection matrix
    ///
    pub fn build_dictionary_per_batch(
        &mut self,
        batch_membership: Option<&Vec<usize>>,
    ) -> anyhow::Result<()> {
        let batch_membership = match batch_membership {
            Some(bm) => bm.clone(),
            None => match &self.cell_to_data {
                Some(c2d) => c2d.clone(),
                None => anyhow::bail!("Should've had cell to data mapping"),
            },
        };

        if let Some(rand_proj_kn) = &self.rand_proj_kn {
            if rand_proj_kn.ncols() != batch_membership.len() {
                return Err(anyhow::anyhow!(
                    "batch membership should cover all the cells"
                ));
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
                    if let Some(cells) = batches.get(&b) {
                        // a. gather all the corresponding cells/columns
                        let columns = cells
                            .iter()
                            .map(|&c| rand_proj_kn.column(c))
                            .collect::<Vec<_>>();

                        Dict::from_column_views(&columns, cells)
                    } else {
                        Dict::empty()
                    }
                })
                .collect();

            self.batch_dictionaries = Some(dictionaries);
            self.cell_to_batch = Some(batch_membership);
        } else {
            return Err(anyhow::anyhow!("Random projection not available"));
        }

        Ok(())
    }

    /// Collapse cells to samples
    /// * `cells_per_sample` - number of cells per sample
    pub fn step4_collapse_columns_cbind(
        &mut self,
        cells_per_sample: Option<usize>,
    ) -> anyhow::Result<()> {
        if let (Some(cell_to_sample), Some(rand_basis_kd)) =
            (&self.cell_to_sample, &self.rand_basis_kd)
        {
            let sample2cells = partition_by_membership(cell_to_sample, cells_per_sample);
            let cell2sample = partition_to_membership(&sample2cells);

            let ngenes = rand_basis_kd.ncols(); // number of genes/features
            let nsample = sample2cells.len(); // pseudobulk samples

            #[cfg(debug_assertions)]
            {
                for raw_data in self.raw_data_vec.iter() {
                    anyhow::ensure!(raw_data.num_rows().unwrap() == ngenes);
                }
            }

            /////////////////////////////////
            // accumulate basic statistics //
            /////////////////////////////////

            let mut stat = Stat::new(ngenes, nsample, self.num_data);
            self.aggregate_observed_cells_within_sample(&mut stat, &cell2sample);

            ///////////////////////////////////////
            // collect counterfactual statistics //
            ///////////////////////////////////////

            let mut offset = 0;
            for (data_idx, raw_data) in self.raw_data_vec.iter().enumerate() {
                let ncol_data = raw_data.num_columns().unwrap();

                offset += ncol_data;
            } // data set

            ////////////////
            // parameters //
            ////////////////

            stat.prob_bs = stat.n_bs;
            stat.prob_bs
                .column_iter_mut()
                .for_each(|mut p_s| p_s /= p_s.sum());
        } else {
            return Err(anyhow::anyhow!("Step 2 (cell sorting) incomplete"));
        }

        Ok(())
    }

    /// A helper function to collect cell statistics within each sample
    /// * `stat` - a mutable reference to Stat
    /// * `cell2sample` - map: cell -> sample
    fn aggregate_observed_cells_within_sample(
        &self,
        stat: &mut Stat,
        cell2sample: &HashMap<usize, usize>,
    ) {
        let arc_stat = Arc::new(Mutex::new(stat));

        let mut offset = 0;
        for (data_idx, raw_data) in self.raw_data_vec.iter().enumerate() {
            let ncol_data = raw_data.num_columns().unwrap();
            let jobs = self.create_jobs(ncol_data);

            jobs.par_iter()
                .progress_count(jobs.len() as u64)
                .for_each(|&job| {
                    let (lb, ub) = job;

                    let mut xx_dm = raw_data
                        .read_columns_csc((lb..ub).collect())
                        .expect("failed to read columns");

                    xx_dm.normalize_columns_inplace();

                    let mut stat = arc_stat.lock().expect("failed to lock stat");

                    for loc in lb..ub {
                        let glob = loc + offset;
                        if let Some(&s) = cell2sample.get(&glob) {
                            if let Some(xv) = xx_dm.get_col(loc) {
                                let rows = xv.row_indices();
                                let vals = xv.values();
                                for (&i, &x) in rows.iter().zip(vals.iter()) {
                                    stat.size_s[s] += 1_f32;
                                    stat.ysum_ds[(i, s)] += x;
                                    stat.n_bs[(data_idx, s)] += 1_f32;
                                }
                            }
                        }
                    }
                });
            offset += ncol_data;
        } // data set
    }

    fn aggregate_matched_cells_within_sample(&self, stat: &mut Stat, knn: usize) {
        let arc_stat = Arc::new(Mutex::new(stat));

        if let (Some(cell_to_batch), Some(batch_dictionaries)) =
            (&self.cell_to_batch, &self.batch_dictionaries)
        {
            use instant_distance::Search;
            let num_batches = batch_dictionaries.len();
            let mut offset = 0;
            for (data_idx, raw_data) in self.raw_data_vec.iter().enumerate() {
                let ncol_data = raw_data.num_columns().unwrap();
                let jobs = self.create_jobs(ncol_data);

                jobs.par_iter()
                    .progress_count(jobs.len() as u64)
                    .for_each(|&job| {
                        let (lb, ub) = job;
                        let mut search = Search::default();
                        let mut stat = arc_stat.lock().expect("failed to lock stat");

                        for loc in lb..ub {
                            let glob = loc + offset;
                            let batch = cell_to_batch[glob];
                            let mut matched = vec![];
                            for other in 0..num_batches {
                                if other == batch {
                                    continue;
                                }
                                matched.extend(
                                    batch_dictionaries[batch]
                                        .match_against_by_name(
                                            &glob,
                                            knn,
                                            &batch_dictionaries[other],
                                        )
                                        .unwrap(),
                                );
                            }
                            // estimate
                        }
                    });

                offset += ncol_data;
            }
        }
    }

    fn create_jobs(&self, ntot: usize) -> Vec<(usize, usize)> {
        let nblock = (ntot + self.block_size - 1) / self.block_size;

        (0..nblock)
            .map(|block| {
                let lb: usize = block * self.block_size;
                let ub: usize = ((block + 1) * self.block_size).min(ntot);
                (lb, ub)
            })
            .collect::<Vec<_>>()
    }
}

struct Stat {
    pub mu_ds: Mat,
    pub mu_cf_ds: Mat,
    pub ysum_ds: Mat,
    pub size_s: DVec,
    pub delta_num_db: Mat,
    pub delta_denom_db: Mat,
    pub n_bs: Mat,
    pub prob_bs: Mat,
}

impl Stat {
    pub fn new(ngene: usize, nsample: usize, nbatch: usize) -> Self {
        Self {
            mu_ds: Mat::zeros(ngene, nsample),
            mu_cf_ds: Mat::zeros(ngene, nsample),
            ysum_ds: Mat::zeros(ngene, nsample),
            size_s: DVec::zeros(nsample),
            delta_num_db: Mat::zeros(ngene, nbatch),
            delta_denom_db: Mat::zeros(ngene, nbatch),
            n_bs: Mat::zeros(nbatch, nsample),
            prob_bs: Mat::zeros(nbatch, nsample),
        }
    }
}

fn partition_to_membership(pb_cells: &HashMap<usize, Vec<usize>>) -> HashMap<usize, usize> {
    let mut sample_membership: HashMap<usize, usize> = HashMap::new();
    let arc = Arc::new(Mutex::new(&mut sample_membership));
    pb_cells.par_iter().for_each(|(&s, cells)| {
        if let Ok(mut sample_membership) = arc.lock() {
            for j in cells {
                sample_membership.insert(*j, s);
            }
        }
    });
    sample_membership
}

fn partition_by_membership(
    sample_membership: &Vec<usize>,
    cells_per_sample: Option<usize>,
) -> HashMap<usize, Vec<usize>> {
    // Take care of empty pseudobulk samples
    let mut pb_position: HashMap<usize, usize> = HashMap::new();
    {
        let mut pos = 0_usize;
        for k in sample_membership {
            if !pb_position.contains_key(k) {
                pb_position.insert(*k, pos);
                pos += 1;
            }
        }
    }
    // dbg!(&pb_position);

    let mut pb_cells: HashMap<usize, Vec<usize>> = HashMap::new();
    for (cell, &k) in sample_membership.iter().enumerate() {
        let &s = pb_position.get(&k).expect("failed to get position");
        pb_cells.entry(s).or_default().push(cell);
    }

    // Down sample cells if needed
    pb_cells.par_iter_mut().for_each(|(_, cells)| {
        let ncells = cells.len();
        if let Some(ntarget) = cells_per_sample {
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
