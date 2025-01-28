// use asap_embed::candle_etm::*;
use asap_embed::candle_data_loader;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, ModuleT, Optimizer, VarBuilder, VarMap};
use matrix_util::traits::SampleOps;
use rayon::join;

#[test]
fn temp() -> anyhow::Result<()> {
    // Tensor::randn(0_f32, 1_f32, (1000,10), &Device::Cpu())?;

    Ok(())
}

#[test]
fn pmf() -> anyhow::Result<()> {
    let dev = Device::new_metal(0)?;

    let dd = 10_usize;
    let nn = 500_usize;
    let kk = 3_usize;

    let beta_dk = Tensor::rgamma(dd, kk, (1.0, 1.0));
    let theta_nk = Tensor::rgamma(nn, kk, (1.0, 1.0));
    let y = beta_dk.matmul(&theta_nk.transpose(0, 1)?)?;

    //////////////////////////////////////////////////
    // it's a torch-like system, sample by features //
    //////////////////////////////////////////////////

    let x_nd = y.transpose(0, 1)?.to_device(&dev)?;

    let mut data_loader = candle_data_loader::InMemoryData::from_tensor(&x_nd)?;
    data_loader.shuffle_minibatch(10);

    for j in 0..data_loader.num_minibatch() {
	let x = data_loader.minibatch(j, &dev)?;
	println!("{:?}", x);
    }

    ///////////////////
    // build a model //
    ///////////////////

    let vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&vm, DType::F32, &dev);

    let layers = vec![16, 16];
    // let etm = ETM::new(dd, kk, &layers, vs.clone())?;

    // /////////////////////////
    // // train ETM with adam //
    // /////////////////////////

    // let mut adam = AdamW::new_lr(vm.all_vars(), 1e-3)?;

    // for _epoch in 0..10 {
    //     let llik = etm.forward_t(&x_nd, true)?.mean_all()?;
    //     let kl = etm.encoder.kl_loss(&x_nd, true)?.mean_all()?;
    //     println!(
    //         "t = {}, llik = {}, kl = {}",
    //         _epoch,
    //         &llik.to_scalar::<f32>()?,
    //         &kl.to_scalar::<f32>()?
    //     );
    //     let loss = (kl - llik)?;
    //     adam.backward_step(&loss)?;
    // }

    Ok(())
}
