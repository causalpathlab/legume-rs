use crate::{srt_cell_pairs::SrtCellPairs, srt_common::*};

use candle_util::candle_inference::TrainConfig;
use matrix_util::traits::ConvertMatOps;

// fn latent_pair_visitor(
//     job: PairsWithIndices,
//     data: &SrtCellPairs,
//         sample: usize,

pub trait SrtLatentStatePairsOps {
    fn evaluate_latent_states(&self) -> anyhow::Result<Mat>;
}

impl<'a> SrtLatentStatePairsOps for SrtCellPairs<'a> {
    fn evaluate_latent_states(&self) -> anyhow::Result<Mat> {



	unimplemented!("");
    }
}

/// Evaluate latent representation with the trained encoder network
///
/// #Arguments
/// * `data_vec` - full data vector
/// * `coord_map` - full spatial coordinates
/// * `encoder` - encoder network
/// * `aggregate_rows` - `d x m` aggregate
/// * `train_config` - training configuration
/// * `delta_db` - batch effect matrix (feature x batch)
pub fn evaluate_latent_by_encoder<Enc>(
    data_vec: &SparseIoVec,
    encoder: &Enc,
    aggregate_rows: &Mat,
    train_config: &TrainConfig,
) -> anyhow::Result<Mat>
where
    Enc: MatchedEncoderModuleT + Send + Sync + 'static,
{
    let dev = &train_config.device;
    let ntot = data_vec.num_columns()?;
    let kk = encoder.dim_latent();
    let block_size = train_config.batch_size;

    let jobs = create_jobs(ntot, Some(block_size));
    let njobs = jobs.len() as u64;
    let arc_enc = Arc::new(Mutex::new(encoder));
    let aggregate = aggregate_rows.to_tensor(dev)?;

    unimplemented!("");

    let mut chunks = jobs
        .par_iter()
        .progress_count(njobs)
        .map(|&(lb, ub)| -> anyhow::Result<()> {
            let enc = arc_enc.lock().expect("enc lock");

            Ok(())
        });

    //         let x0_nm = delta_bm.as_ref().map(|delta_bm| {
    //             let batches = data_vec
    //                 .get_batch_membership(lb..ub)
    //                 .into_iter()
    //                 .map(|x| x as u32);

    //             let batches = Tensor::from_iter(batches.clone(), dev).unwrap();
    //             delta_bm.index_select(&batches, 0).expect("expand delta")
    //         });

    //         let s_nc = coord_map.map(|x| {
    //             let s_nc: Mat = x.rows_range(lb..ub).into();
    //             s_nc.to_tensor(dev).expect("spatial n x c")
    //         });

    //         let x_nd = data_vec
    //             .read_columns_dmatrix(lb..ub)
    //             .expect("read columns")
    //             .to_tensor(dev)
    //             .expect("x")
    //             .transpose(0, 1)
    //             .expect("transpose x_dn -> x_nd");

    //         let x_nm = x_nd.matmul(&aggregate).expect("x aggregate");

    //         let (z_nk, _) = enc
    //             .forward_t(&x_nm, s_nc.as_ref(), x0_nm.as_ref(), false)
    //             .expect("");

    //         let z_nk = z_nk.to_device(&candle_core::Device::Cpu).expect("to cpu");
    //         (lb, Mat::from_tensor(&z_nk).expect("to mat"))
    //     })
    //     .collect::<Vec<_>>();

    // chunks.sort_by_key(|&(lb, _)| lb);
    // let chunks = chunks.into_iter().map(|(_, z_nk)| z_nk).collect::<Vec<_>>();

    // let mut ret = Mat::zeros(ntot, kk);
    // {
    //     let mut lb = 0;
    //     for z in chunks {
    //         let ub = lb + z.nrows();
    //         ret.rows_range_mut(lb..ub).copy_from(&z);
    //         lb = ub;
    //     }
    // }
    // Ok(ret)
}
