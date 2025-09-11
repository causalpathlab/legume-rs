use crate::embed_common::*;

use data_beans::sparse_data_visitors::VisitColumnsOps;

use candle_util::candle_inference::TrainConfig;
use candle_util::candle_model_traits::*;

use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

fn adjust_triplets_visitor(
    job: (usize, usize),
    full_data_vec: &SparseIoVec,
    delta_db: &Mat,
    triplets: Arc<Mutex<&mut Vec<(u64, u64, f32)>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let batches = full_data_vec.get_batch_membership(lb..ub);

    let mut x_dn = full_data_vec.read_columns_csc(lb..ub)?;

    x_dn.adjust_by_division_of_selected_inplace(delta_db, &batches);

    let new_triplets = x_dn
        .triplet_iter()
        .filter_map(|(i, j, &x_ij)| {
            let x_ij = x_ij.round();
            if x_ij < 1_f32 {
                None
            } else {
                Some((i as u64, (j + lb) as u64, x_ij))
            }
        })
        .collect::<Vec<_>>();

    let mut triplets = triplets.lock().expect("lock triplets");
    triplets.extend(new_triplets);
    Ok(())
}

/// Adjust the original data by eliminating batch effects `delta_db`
/// (`d x b`) from each column. We will directly call
/// `get_batch_membership` in `data_vec`.
///
/// # Arguments
/// * `data_vec` - sparse data vector
/// * `delta_db` - row/feature by batch average effect matrix
///
/// # Returns
/// * `triplets` - we can feed this vector to create a new backend
pub fn triplets_adjusted_by_batch(
    data_vec: &SparseIoVec,
    delta_db: &Mat,
) -> anyhow::Result<Vec<(u64, u64, f32)>> {
    let mut triplets = vec![];
    data_vec.visit_columns_by_block(&adjust_triplets_visitor, delta_db, &mut triplets, None)?;
    Ok(triplets)
}

/// Evaluate latent representation with the trained encoder network
///
/// #Arguments
/// * `data_vec` - full data vector
/// * `encoder` - encoder network
/// * `train_config` - training configuration
/// * `delta_db` - batch effect matrix (feature x batch)
pub fn evaluate_latent_by_encoder<Enc>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    train_config: &TrainConfig,
    delta_db: Option<&Mat>,
) -> anyhow::Result<Mat>
where
    Enc: EncoderModuleT + Send + Sync + 'static,
{
    let dev = &train_config.device;
    let ntot = data_vec.num_columns()?;
    let kk = encoder.dim_latent();

    let block_size = train_config.batch_size;

    let jobs = create_jobs(ntot, Some(block_size));
    let njobs = jobs.len() as u64;
    let arc_enc = Arc::new(Mutex::new(encoder));

    let delta_bd = delta_db.map(|delta_db| {
        delta_db
            .to_tensor(dev)
            .expect("delta to tensor")
            .transpose(0, 1)
            .expect("transpose")
    });

    let mut chunks = jobs
        .par_iter()
        .progress_count(njobs)
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Mat)> {
            let x0_nd = delta_bd.as_ref().map(|delta_bm| {
                let batches = data_vec
                    .get_batch_membership(lb..ub)
                    .into_iter()
                    .map(|x| x as u32);

                let batches = Tensor::from_iter(batches.clone(), dev).unwrap();
                delta_bm.index_select(&batches, 0).expect("expand delta")
            });

            let x_nd = data_vec
                .read_columns_dmatrix(lb..ub)?
                .to_tensor(dev)?
                .transpose(0, 1)?;

            let enc = arc_enc.lock().expect("enc lock");
            let (z_nk, _) = enc.forward_t(&x_nd, x0_nd.as_ref(), false)?;
            let z_nk = z_nk.to_device(&candle_core::Device::Cpu)?;
            Ok((lb, Mat::from_tensor(&z_nk).expect("to mat")))
        })
        .filter_map(Result::ok)
        .collect::<Vec<_>>();

    chunks.sort_by_key(|&(lb, _)| lb);
    let chunks = chunks.into_iter().map(|(_, z_nk)| z_nk).collect::<Vec<_>>();

    let mut ret = Mat::zeros(ntot, kk);
    {
        let mut lb = 0;
        for z in chunks {
            let ub = lb + z.nrows();
            ret.rows_range_mut(lb..ub).copy_from(&z);
            lb = ub;
        }
    }
    Ok(ret)
}
