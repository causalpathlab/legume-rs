use crate::embed_common::*;
use crate::routines_latent_representation::*;
use crate::routines_post_process::*;
use crate::routines_pre_process::*;

use dashmap::DashMap as HashMap;
use data_beans_alg::normalization::NormalizeDistance;
use matrix_util::common_io::extension;
use matrix_util::dmatrix_util::concatenate_horizontal;
use matrix_util::knn_match::*;
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
    #[arg(long, default_value_t = 10)]
    knn_batches: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value_t = 50)]
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

struct BulkDataOut {
    genes: Vec<Box<str>>,
    samples: Vec<Box<str>>,
    data: Mat,
}

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
        data: y1,
    } = read_bulk_data_consistent_with_sc(&args.bulk_data_files, &sc_data)?;

    if bulk_genes != sc_data.row_names()? {
        return Err(anyhow::anyhow!(
            "bulk and sc data gene names should
        match"
        ));
    }

    info!("Finding a shared basis matrix for random projection...");
    let proj_dim = args.proj_dim.max(args.n_latent_topics);
    let rand_proj = sc_data.project_columns_with_batch_correction(
        proj_dim,
        Some(args.block_size),
        Some(sc_batch.as_ref()),
    )?;

    info!("Registering batch information...");
    let proj_kn = rand_proj.proj.scale_columns();
    sc_data.register_batches_dmatrix(&proj_kn, &sc_batch)?;

    info!("Characterize batch effects within the sc data");

    // sc_data.collapse_columns(knn_batches, knn_cells, reference_batch_names, num_opt_iter);

    // use this δ to adjust in the following imputation steps




    info!("Check how much we can regress the bulk on the sc data");
    let y0 = y1.predict_by_knn(&sc_data, &rand_proj, args.knn_cells)?;

    // identify y1 ~ mu * delta
    // identify y0 ~ mu * tau

    // 


    info!("Identify sc bias");

    // identify x1 ~ gamma * tau
    // identify x0 ~ gamma * eta
    sc_data.assign_batch_membership_by_nearest_bulk(&y0, &rand_proj)?;




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
