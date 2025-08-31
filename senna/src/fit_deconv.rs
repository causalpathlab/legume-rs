use crate::embed_common::*;
use crate::routines_latent_representation::*;
use crate::routines_post_process::*;
use crate::routines_pre_process::*;

use dashmap::DashMap as HashMap;
use data_beans_alg::normalization::NormalizeDistance;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_util::common_io::extension;
use matrix_util::dmatrix_util::concatenate_horizontal;
use matrix_util::knn_match::*;
use ndarray_rand::rand::seq::IteratorRandom;
use ndarray_rand::rand::thread_rng;
use rayon::prelude::*;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
enum ComputeDevice {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Args, Debug)]
pub struct DeconvArgs {
    /// single-cell data (`.zarr` or `.h5`)
    #[arg(short = 's', long, value_delimiter = ',', required = true)]
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

    #[arg(long, default_value_t = false)]
    ignore_batch_effects: bool,

    /// #k-nearest neighbours batches
    #[arg(long, default_value_t = 10)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 10)]
    knn_cells: usize,

    /// #k-nearest neighbours within each bulk
    #[arg(long, default_value_t = 50)]
    knn_bulk: usize,

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

struct BulkDataOut {
    genes: Vec<Box<str>>,
    samples: Vec<Box<str>>,
    data: Mat,
}

/// a helper function to read bulk data with the matching row names
/// * `bulk_data_files` - bulk file names
/// * `sc_data` - opened `SparseIoVec`
///
fn read_bulk_data_consistent_with_sc(
    bulk_data_files: &[Box<str>],
    sc_data: &SparseIoVec,
) -> anyhow::Result<BulkDataOut> {
    let genes = sc_data.row_names()?;
    let gene_to_position: HashMap<Box<str>, usize> = genes
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect();

    let ngenes = gene_to_position.len();
    info!("use {} genes as common features", ngenes);

    let mut samples = vec![];
    let mut bulk_data_vec = vec![];

    for bulk_file in bulk_data_files {
        let MatWithNames {
            rows: raw_genes,
            cols: raw_samples,
            mat: raw_ds,
        } = match extension(bulk_file.as_ref())?.as_ref() {
            "parquet" => Mat::from_parquet(bulk_file.as_ref())?,
            _ => Mat::read_data(bulk_file.as_ref(), &['\t', ','], None, Some(0), None, None)?,
        };

        let ncols = raw_samples.len();

        let mut padded_ds = Mat::zeros(ngenes, ncols);
        for (i, g) in raw_genes.iter().enumerate() {
            if let Some(r) = gene_to_position.get(g) {
                padded_ds.row_mut(*r.value()).copy_from(&raw_ds.row(i));
            }
        }

        samples.extend(raw_samples);
        bulk_data_vec.push(padded_ds);
    }
    let bulk_data = concatenate_horizontal(&bulk_data_vec)?;

    info!(
        "Read bulk data {} genes x {} samples",
        ngenes,
        samples.len()
    );
    Ok(BulkDataOut {
        genes,
        samples,
        data: bulk_data,
    })
}

/// a master function to perform the deconvolution of bulk data files
///
pub fn fit_deconv(args: &DeconvArgs) -> anyhow::Result<()> {
    info!("Opening single-cell data files...");
    // 1. Read sc data with batch membership
    let SparseDataWithBatch {
        data: mut sc_data,
        batch: sc_batch,
    } = read_sparse_data_with_membership(ReadArgs {
        data_files: args.sc_data_files.clone(),
        batch_files: args.batch_files.clone(),
    })?;

    // 2. Read bulk data
    info!("Reading in bulk data files...");
    let BulkDataOut {
        genes: bulk_genes,
        samples: bulk_samples,
        data: bulk_data,
    } = read_bulk_data_consistent_with_sc(&args.bulk_data_files, &sc_data)?;

    if bulk_genes != sc_data.row_names()? {
        return Err(anyhow::anyhow!("bulk and sc data gene names should match"));
    }

    info!("Finding a shared basis matrix for random projection...");
    let proj_dim = args.proj_dim.max(args.n_latent_topics);
    let rand_proj = sc_data.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(sc_batch.as_ref()),
    )?;

    info!("Constructing PB in the sc data...");
    let proj_kn = rand_proj.proj.scale_columns();

    let nsamp =
        sc_data.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), args.down_sample)?;

    if !args.ignore_batch_effects {
        info!("Registering batch information");
        sc_data.build_hnsw_per_batch(&proj_kn, &sc_batch)?;
    }

    let collapsed = sc_data.collapse_columns(
        Some(args.knn_batches),
        Some(args.knn_cells),
        args.reference_batches.as_deref(),
        Some(args.iter_opt),
    )?;

    // data 1: use collapsed
    // collapsed.mu_observed;
    // collapsed.mu_residual;
    // collapsed.mu_adjusted;

    // data 2: bulk projected
    // let bulk_yhat =

    // bulk_data.predict_from_collapsed_data(&collapsed, args.knn_bulk)?;

    // todo: bootstrap bulk data imputed by single cell data

    let bulk_yhat = bulk_data.predict_from_collapsed_data(&collapsed, args.knn_bulk)?;

    // E[Y] ~ μ δ
    // E[Y.hat] ~ μ

    //     Σ Y + Σ Y.hat
    // μ = -------------
    //     n (δ + 1)

    //     Σ Y
    // δ = ----
    //     n μ

    // let bulk_to_cells = (0..sc_data.num_batches())
    //     .map(|b| sc_data.batch_to_columns(b).unwrap_or_default().clone())
    //     .collect::<Vec<_>>();

    // train on sc data
    // There are two types of additional factors present in bulk data
    //
    // (1) multiplicative bias, inconsistency in technology, platform, etc. gene-level difference
    //
    // (2) additive bias due to cell types missing in reference data, cell type level difference
    //

    // If we train two types of bulk data (pseudobulk and observed
    // bulk), the second type of missingness can be inferred by
    // looking at the cell types/topics only present in the actual
    // bulk data.

    // The first type of bias factors can be uncovered Y vs. Y hat

    // jointly train... we need to pick up common beta
    // [x.mixed, x.resid] -> z -> x.clean
    //
    // oversample bulk data and simulate? yes, both y.enclosed and y.residual
    // y.clean -> z, y.resid -> Δ, (z + Δ) -> y.hat?
    //

    // the goal is to find a new topic?
    // GOAL: missing celltype!!!!
    //

    // let proj_kn = rand_proj.proj.scale_columns();
    // sc_data.register_batches_dmatrix(&proj_kn, &sc_batch)?;

    // // train
    // let mixed_dn = &collapsed.mu_observed;
    // let clean_dn = collapsed.mu_adjusted.as_ref();
    // let batch_dn = collapsed.mu_residual.as_ref();

    // let delta = collapsed.delta.clone();

    // let bulk_yhat = bulk_data.predict_from_sparse_data(&sc_data, &rand_proj, args.knn_bulk)?;

    info!("Check how much we can regress the bulk on the sc data");

    // let mut mu = GammaMatrix::new((bulk_data.nrows(), bulk_data.ncols()), 1.0, 1.0);

    // mu.update_stat(update_a, update_b);

    // identify y1 ~ mu * delta
    // identify y0 ~ mu * tau

    info!("Identify sc bias");

    // should pair up
    // for each

    // identify x1 ~ gamma * tau
    // identify x0 ~ gamma * eta

    // let collapsed = sc_data.collapse_columns(
    //     Some(args.knn_batches),
    //     Some(args.knn_cells),
    //     args.reference_batches.as_deref(),
    //     Some(args.iter_opt),
    // )?;

    // let bulk_yhat = bulk_data.predict_from_collapsed_data(&collapsed, args.knn_bulk)?;
    // sc_data.collapse_columns(knn_batches, knn_cells, reference_batch_names, num_opt_iter)

    // the question is about how we get training data
    // a. how to address uncertainty? include variation?
    // b. how to adjust bias between the reference and target data?

    // let n_pb_samples =
    //     sc_data.partition_columns_to_groups(&proj_kn, Some(args.sort_dim), args.down_sample)?;

    // sc_data.collapse_columns(knn_batches, knn_cells, reference_batch_names, num_opt_iter)

    // for each bulk sample
    // - find k-nearest neighbour cells across all the pseudobulk samples

    info!("Done");
    Ok(())
}

trait InteractWithScData {
    fn predict_from_collapsed_data(
        &self,
        collapsed: &CollapsedOut,
        knn: usize,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn predict_from_sparse_data(
        &self,
        sc_data_vec: &SparseIoVec,
        rand_proj: &RandColProjOut,
        knn: usize,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn bootstrap_from_sparse_data(
        &self,
        sc_data_vec: &SparseIoVec,
        rand_proj: &RandColProjOut,
        knn: usize,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl InteractWithScData for Mat {
    fn predict_from_collapsed_data(
        &self,
        collapsed: &CollapsedOut,
        knn: usize,
    ) -> anyhow::Result<Self> {
        let pb = &collapsed.mu_observed;
        let max_rank = pb.nrows().min(pb.ncols());

        let basis_dk = pb
            .posterior_log_mean()
            .scale_columns()
            .rsvd(max_rank)?
            .0
            .scale_columns()
            .map(|x| x.clamp(-4., 4.))
            .scale_columns();

        let ln_x = self.map(|x| x.ln_1p()).scale_columns();
        let bulk_km = (basis_dk.transpose() * &ln_x).scale_columns();

        todo!("need bootstrap");

        // bootstrap pb data

        // let mut stoch_data_vec = vec![];

        // let nboot = 100;
        // let n_pb_cols = pb.ncols();
        // let pb_columns = 0..n_pb_cols;

        // let mut rng = thread_rng();

        // for _ in 0..n_pb_cols {
        //     if let Some(s) = pb_columns.choose(&mut rng) {
        //         let _x_d = match collapsed.mu_adjusted.as_ref() {
        //             Some(x) => x.posterior_sample(),
        //             _ => collapsed.mu_observed.posterior_sample(),
        //         }?
        //         .column(s);

        //         stoch_data_vec.push(_x_d);
        //     }
        // }

        // let stoch_data_dm = concatenate_horizontal(&stoch_data_vec)?;
        // let stoch_ln_dm = stoch_data_dm.map(|x| x.ln_1p()).scale_columns();
        // let stoch_proj_km = (basis_dk.transpose() * &stoch_ln_dm).scale_columns();

        // let column_names = (0..stoch_proj_km.ncols()).collect();
        // let dict = ColumnDict::from_dmatrix(stoch_proj_km, column_names);
        // let norm_target = 2_f32.ln();

        // let imputed = bulk_km
        //     .column_iter()
        //     .enumerate()
        //     .map(|(i, query)| -> anyhow::Result<DVec> {
        //         let (neighbours, distances) = dict.search_by_query_data(&query.to_vp(), knn)?;

        //         let weights = distances.into_iter().normalized_exp(norm_target);
        //         let denom = weights.iter().sum::<f32>().max(1e-8);
        //         let mut ret = DVec::zeros(self.nrows());
        //         for (j, w) in neighbours.into_iter().zip(weights) {
        //             ret += stoch_data_dm.column(j) * w / denom;
        //         }
        //         Ok(ret)
        //     })
        //     .collect::<anyhow::Result<Vec<_>>>()?;

        // concatenate_horizontal(&imputed)
    }

    fn predict_from_sparse_data(
        &self,
        sc_data_vec: &SparseIoVec,
        rand_proj: &RandColProjOut,
        knn: usize,
    ) -> anyhow::Result<Self> {
        let mut imputed_dm = Mat::zeros(self.nrows(), self.ncols());

        let basis_dk = &rand_proj.basis;

        let ln_x = self.map(|x| x.ln_1p()).normalize_columns();
        let bulk_km = (basis_dk.transpose() * &ln_x).scale_columns();

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

    fn bootstrap_from_sparse_data(
        &self,
        sc_data_vec: &SparseIoVec,
        rand_proj: &RandColProjOut,
        knn: usize,
    ) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        todo!("");
    }
}

// trait ScKnnWithBulk {
//     /// assign column's batch membership based on the nearest
//     /// neighbour found in bulk columns
//     fn assign_batch_membership_by_nearest_bulk(
//         &mut self,
//         bulk: &Mat,
//         rand_proj: &RandColProjOut,
//     ) -> anyhow::Result<()>;
// }

// impl ScKnnWithBulk for SparseIoVec {
//     fn assign_batch_membership_by_nearest_bulk(
//         &mut self,
//         bulk: &Mat,
//         rand_proj: &RandColProjOut,
//     ) -> anyhow::Result<()> {
//         let proj_kn = rand_proj.proj.scale_columns();
//         info!("Sc Proj: {} x {} ...", proj_kn.nrows(), proj_kn.ncols());

//         let basis_dk = &rand_proj.basis;
//         let ln_x = bulk.map(|x| x.ln_1p()).normalize_columns();
//         let mut bulk_km = basis_dk.transpose() * &ln_x;
//         bulk_km.scale_columns_inplace();
//         info!("Bulk Proj: {} x {} ...", bulk_km.nrows(), bulk.ncols());

//         let bulk_samples = (0..bulk_km.ncols()).collect();
//         let columns = (0..bulk_km.ncols()).map(|j| bulk_km.column(j)).collect();
//         let dict = ColumnDict::from_dvector_views(columns, bulk_samples);

//         let batch_membership = proj_kn
//             .column_iter()
//             .par_bridge()
//             .map(|x| -> anyhow::Result<usize> {
//                 let query = x.to_vp();
//                 let search = dict.search_by_query_data(&query, 1)?;

//                 if search.0.is_empty() {
//                     return Err(anyhow::anyhow!("failed to find the nearest bulk sample"));
//                 }

//                 Ok(search.0[0])
//             })
//             .collect::<anyhow::Result<Vec<_>>>()?;

//         self.register_batches_dmatrix(&proj_kn, &batch_membership)?;
//         Ok(())
//     }
// }
