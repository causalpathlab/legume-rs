// use asap_embed::candle_etm::*;
use asap_embed::candle_data_loader::*;
use asap_embed::candle_inference::*;
use asap_embed::candle_loss_functions::*;
use asap_embed::candle_model_decoder::*;
use asap_embed::candle_model_encoder::*;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use matrix_util::traits::SampleOps;

#[test]
fn pmf() -> anyhow::Result<()> {

    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

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



    ///////////////////
    // build a model //
    ///////////////////

    let vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&vm, DType::F32, &dev);

    let layers = vec![16, 16];
    let enc = NonNegEncoder::new(dd, kk, &layers, vs.clone())?;
    let dec = ETMDecoder::new(dd, kk, vs.clone())?;

    let mut vae = Vae::build(enc, dec, &vm);

    let mut data_loader = InMemoryData::from_tensor(&x_nd)?;

    let _llik = vae.train(
        &mut data_loader,
        &topic_likelihood,
        TrainingConfig {
            learning_rate: 1e-3,
            batch_size: 10,
            num_epochs: 100,
            device: dev,
	    verbose: true,
        },
    )?;

    // let x = llik.first();
    // println!("llik: {:?}", llik);

    Ok(())
}
