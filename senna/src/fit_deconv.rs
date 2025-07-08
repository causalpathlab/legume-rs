use crate::embed_common::*;
use crate::routines_latent_representation::*;
use crate::routines_post_process::*;
use crate::routines_pre_process::*;

use data_beans_alg::normalization::NormalizeDistance;
use matrix_util::knn_match::ColumnDict;
use matrix_util::knn_match::MakeVecPoint;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Args, Debug)]
pub struct DeconvArgs {
    /// single-cell data files (`zarr` or `h5`)
    #[arg(required = true)]
    sc_data_files: Vec<Box<str>>,

    /// bulk data files (`.parquet`, `.tsv.gz`, `.csv.gz`)
    /// where the first column corresponds to gene names
    #[arg(short, long, required = true)]
    bulk_data_files: Vec<Box<str>>,

    /// random projection dimension to project the data.
    #[arg(long, short = 'p', default_value_t = 50)]
    proj_dim: usize,

    /// output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    /// use top `S` components of projection. #samples < `2^S+1`.
    #[arg(long, short = 'd', default_value_t = 10)]
    sort_dim: usize,

    /// batch membership files (comma-separated names). Each bach file
    /// should correspond to each data file.
    #[arg(long, short, value_delimiter(','))]
    batch_files: Option<Vec<Box<str>>>,

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 3)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// reference batch names
    #[arg(long, value_delimiter(','))]
    reference_batches: Option<Vec<Box<str>>>,

    /// #downsampling columns per each collapsed sample. If None, no
    /// downsampling.
    #[arg(long, short = 's')]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value_t = 15)]
    iter_opt: usize,

    /// block_size (# columns) for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// number of latent topics
    #[arg(short = 't', long, default_value_t = 10)]
    n_latent_topics: usize,

    /// to reduce row features (#gene modules ~ 2^r)
    #[arg(short = 'r', long, default_value_t = 10)]
    n_row_proj_dim: usize,

    /// encoder layers
    #[arg(long, short = 'e', value_delimiter(','), default_values_t = vec![128,1024,128])]
    encoder_layers: Vec<usize>,

    /// intensity levels for frequency embedding
    #[arg(long, default_value_t = 100)]
    vocab_size: usize,

    /// intensity embedding dimension
    #[arg(long, default_value_t = 3)]
    vocab_emb: usize,

    /// # training epochs
    #[arg(long, short = 'i', default_value_t = 1000)]
    epochs: usize,

    /// # data jitter during the training
    #[arg(long, short = 'j', default_value_t = 10)]
    num_jitter: usize,

    /// Minibatch size
    #[arg(long, default_value_t = 100)]
    minibatch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,

    /// candle device
    #[arg(long, value_enum, default_value = "cpu")]
    device: ComputeDevice,

    /// verbosity
    #[arg(long, short)]
    verbose: bool,
}

pub fn fit_deconv(args: &DeconvArgs) -> anyhow::Result<()> {
    // use sc data as reference features?

    // 1. read_data_vec_membership(args)
    // 2. random projection
    // 3. deconvolution by matching neighbours

    // the question is about how we get training data
    // a. how to address uncertainty? include variation?
    // b. how to adjust bias between the reference and target data?

    // 1. Read sc data with batch membership
    let SparseDataWithBatch {
        data: mut sc_data,
        batch: batch_membership,
    } = read_sparse_data_with_membership(ReadArgs {
        data_files: args.sc_data_files.clone(),
        batch_files: args.batch_files.clone(),
    })?;

    // 2. Random projection
    let proj_dim = args.proj_dim.max(args.n_latent_topics);
    let rand_proj = sc_data.project_columns(proj_dim, Some(args.block_size))?;
    let proj_kn = rand_proj.proj.scale_columns();
    // info!("Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

    info!("Registering batch information...");
    // sc_data.assign_batch_membership_by_nearest_bulk(bulk, rand_proj)?;

    let n_pb_samples =
        sc_data.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), args.down_sample)?;

    // sc_data.collapse_columns(knn_batches, knn_cells, reference_batch_names, num_opt_iter)

    // for each bulk sample
    // - find k-nearest neighbour cells across all the pseudobulk samples

    unimplemented!("take both bulk and single cell data");
    Ok(())
}

trait KnnWithBulk {
    /// assign column's batch membership based on the nearest
    /// neighbour found in bulk columns
    fn assign_batch_membership_by_nearest_bulk(
        &mut self,
        bulk: &Mat,
        rand_proj: &RandColProjOut,
    ) -> anyhow::Result<()>;
}

impl KnnWithBulk for SparseIoVec {
    fn assign_batch_membership_by_nearest_bulk(
        &mut self,
        bulk: &Mat,
        rand_proj: &RandColProjOut,
    ) -> anyhow::Result<()> {
        let proj_kn = rand_proj.proj.scale_columns();
        info!("Sc Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

        let basis_dk = &rand_proj.basis;
        let ln_x = bulk.map(|x| x.ln_1p()).normalize_columns();
        let mut bulk_km = basis_dk.transpose() * &ln_x;
        bulk_km.scale_columns_inplace();
        info!("Bulk Proj: {} x {} ...", bulk_km.nrows(), bulk.ncols());

        let bulk_samples = (0..bulk_km.ncols()).collect();
        let columns = (0..bulk_km.ncols()).map(|j| bulk_km.column(j)).collect();
        let dict = ColumnDict::from_dvector_views(columns, bulk_samples);

        let batch_membership = proj_kn
            .column_iter()
            .par_bridge()
            .map(|x| -> anyhow::Result<usize> {
                let query = x.to_vp();
                let search = dict.search_by_query_data(&query, 1)?;

                if search.0.is_empty() {
                    return Err(anyhow::anyhow!("failed to find the nearest bulk sample"));
                }

                Ok(search.0[0])
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        self.register_batches_dmatrix(&proj_kn, &batch_membership)?;
        Ok(())
    }
}

trait InteractWithScData {
    fn predict_by_knn(
        &self,
        sc_data_vec: &SparseIoVec,
        rand_proj: &RandColProjOut,
        knn: usize,
    ) -> anyhow::Result<Mat>;
}

impl InteractWithScData for Mat {
    fn predict_by_knn(
        &self,
        sc_data_vec: &SparseIoVec,
        rand_proj: &RandColProjOut,
        knn: usize,
    ) -> anyhow::Result<Mat> {
        let mut imputed_dm = Mat::zeros(self.nrows(), self.ncols());

        let basis_dk = &rand_proj.basis;

        let ln_x = self.map(|x| x.ln_1p()).normalize_columns();
        let mut bulk_km = basis_dk.transpose() * &ln_x;
        bulk_km.scale_columns_inplace();

        let norm_target = 2_f32.ln();

        for (i, query) in bulk_km.column_iter().enumerate() {
            let (y, _, distances) = sc_data_vec.query_columns_by_data_csc(query, knn)?;

            let weights = distances.into_iter().normalized_exp(norm_target);
            let denom = weights.iter().sum::<f32>().max(1e-8);
            let weights = DVec::from_vec(weights).unscale(denom);
            imputed_dm.column_mut(i).copy_from(&(y * weights));
        }
        Ok(imputed_dm)
    }
}
