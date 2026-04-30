use super::common::{compute_level_epochs, process_blocks};
use crate::embed_common::*;
use crate::fit_joint_topic::JointTopicArgs;
use crate::logging::new_progress_bar;

use candle_core::{Device, Tensor};
use candle_nn::AdamW;
use candle_nn::Optimizer;
use candle_util::candle_encoder_joint_softmax::LogSoftmaxJointEncoder;
use candle_util::candle_joint_data_loader::*;
use candle_util::candle_loss_functions::topic_likelihood;
use candle_util::candle_model_traits::*;
use matrix_util::dmatrix_util::concatenate_vertical;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct SaveContext<'a> {
    pub collapsed_levels: &'a [Vec<CollapsedOut>],
    pub encoder: &'a LogSoftmaxJointEncoder,
    pub train_config: &'a ProgressiveTrainConfig<'a>,
    pub coarsenings: &'a [Option<FeatureCoarsening>],
    pub n_features_full: &'a [usize],
    pub gene_names: &'a [Box<str>],
    pub data_stack: &'a SparseIoStack,
    pub args: &'a JointTopicArgs,
}

/// Train decoder, write dictionaries, log-likelihood, and latent states.
pub(crate) fn train_and_save<Dec: JointDecoderModuleT>(
    decoder: &Dec,
    ctx: &SaveContext,
) -> anyhow::Result<()> {
    let scores = train_encoder_decoder_progressive(
        ctx.collapsed_levels,
        ctx.encoder,
        decoder,
        ctx.train_config,
    )?;

    info!("Writing down the model parameters");
    write_joint_dictionaries(
        decoder,
        ctx.coarsenings,
        ctx.n_features_full,
        ctx.gene_names,
        &ctx.args.out,
    )?;
    scores.to_parquet(&format!("{}.log_likelihood.parquet", &ctx.args.out))?;

    info!("Moving parameters to CPU for multi-threaded inference");
    let cpu_dev = candle_core::Device::Cpu;
    super::common::move_varmap_to_cpu(ctx.train_config.parameters)?;

    info!("Writing down the latent states");
    write_latent_states(
        ctx.data_stack,
        ctx.encoder,
        ctx.collapsed_levels.last().unwrap(),
        &cpu_dev,
        ctx.args,
        ctx.coarsenings,
    )?;
    Ok(())
}

/// Write effective dictionaries for any `JointDecoderModuleT`, expanding
/// coarsening if needed and stacking vertically across modalities.
pub(crate) fn write_joint_dictionaries<Dec: JointDecoderModuleT>(
    decoder: &Dec,
    coarsenings: &[Option<FeatureCoarsening>],
    n_features_full: &[usize],
    gene_names: &[Box<str>],
    out: &str,
) -> anyhow::Result<()> {
    let dictionaries = decoder
        .get_dictionary()?
        .into_iter()
        .zip(coarsenings)
        .zip(n_features_full)
        .map(|((dict, fc), &n_full)| -> anyhow::Result<Mat> {
            let dict = dict.to_device(&candle_core::Device::Cpu)?;
            let dict_mat = Mat::from_tensor(&dict)?;
            if let Some(fc) = fc {
                info!(
                    "Expanding dictionary from {} to {} features",
                    fc.num_coarse, n_full
                );
                Ok(fc.expand_log_dict_dk(&dict_mat, n_full))
            } else {
                Ok(dict_mat)
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    concatenate_vertical(&dictionaries)?.to_parquet_with_names(
        &(out.to_string() + ".dictionary.parquet"),
        (Some(gene_names), Some("gene")),
        None,
    )?;
    Ok(())
}

/// Evaluate latent states and write to parquet.
pub(crate) fn write_latent_states<Enc: JointEncoderModuleT + Send + Sync>(
    data_stack: &SparseIoStack,
    encoder: &Enc,
    collapsed_data_vec: &[CollapsedOut],
    dev: &candle_core::Device,
    args: &JointTopicArgs,
    coarsenings: &[Option<FeatureCoarsening>],
) -> anyhow::Result<()> {
    let z_nk = evaluate_latent_by_encoder(
        data_stack,
        encoder,
        collapsed_data_vec,
        dev,
        args,
        coarsenings,
    )?;
    let cell_names = data_stack.column_names()?;
    z_nk.to_parquet_with_names(
        &(args.out.to_string() + ".latent.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&axis_id_names("T", z_nk.ncols())),
    )?;
    Ok(())
}

pub(crate) fn evaluate_latent_by_encoder<Enc>(
    data_stack: &SparseIoStack,
    encoder: &Enc,
    collapsed_vec: &[CollapsedOut],
    dev: &candle_core::Device,
    args: &JointTopicArgs,
    coarsenings: &[Option<FeatureCoarsening>],
) -> anyhow::Result<Mat>
where
    Enc: JointEncoderModuleT + Send + Sync,
{
    let ntot = data_stack.num_columns()?;
    let kk = encoder.dim_latent();

    let block_size = args.minibatch_size;

    // Delta coarsened to D_coarse — encoder operates at D_coarse
    let delta = collapsed_vec
        .iter()
        .zip(coarsenings)
        .map(|(x, fc)| {
            match args.adj_method {
                AdjMethod::Residual => x.mu_residual.as_ref(),
                AdjMethod::Batch => x.delta.as_ref(),
            }
            .map(|delta| -> anyhow::Result<Tensor> {
                let mut delta_mat = delta.posterior_mean().clone();
                if let Some(fc) = fc {
                    delta_mat = fc.aggregate_rows_ds(&delta_mat);
                }
                Ok(delta_mat.to_tensor(dev)?.transpose(0, 1)?.contiguous()?)
            })
            .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let eval_config = JointEvalConfig {
        dev,
        delta: &delta,
        coarsenings,
    };

    let adj_method = args.adj_method.clone();

    process_blocks(ntot, kk, block_size, dev, |block| {
        evaluate_block(block, data_stack, encoder, &eval_config, &adj_method)
    })
}

struct JointEvalConfig<'a> {
    dev: &'a Device,
    delta: &'a [Option<Tensor>],
    coarsenings: &'a [Option<FeatureCoarsening>],
}

fn read_block_tensors(
    data_stack: &SparseIoStack,
    lb: usize,
    ub: usize,
    config: &JointEvalConfig,
) -> anyhow::Result<Vec<Tensor>> {
    data_stack
        .stack
        .iter()
        .zip(config.coarsenings.iter())
        .map(|(dv, fc)| -> anyhow::Result<Tensor> {
            let x_dn = dv.read_columns_csc(lb..ub)?;
            let x_nd = if let Some(fc) = fc {
                fc.aggregate_sparse_csc(&x_dn)
                    .to_tensor(config.dev)?
                    .transpose(0, 1)?
            } else {
                x_dn.to_tensor(config.dev)?.transpose(0, 1)?
            };
            Ok(x_nd)
        })
        .collect()
}

fn evaluate_block<Enc>(
    block: (usize, usize),
    data_stack: &SparseIoStack,
    encoder: &Enc,
    config: &JointEvalConfig,
    adj_method: &AdjMethod,
) -> anyhow::Result<(usize, Mat)>
where
    Enc: JointEncoderModuleT,
{
    let (lb, ub) = block;
    let x_vec = read_block_tensors(data_stack, lb, ub, config)?;

    // Delta is already coarsened before conversion to Tensor
    let x0_vec = data_stack
        .stack
        .iter()
        .zip(config.delta)
        .map(|(dv, delta)| {
            delta
                .as_ref()
                .map(|delta| -> anyhow::Result<Tensor> {
                    let membership: Vec<u32> = match *adj_method {
                        AdjMethod::Batch => dv
                            .get_batch_membership(lb..ub)
                            .into_iter()
                            .map(|j| j as u32)
                            .collect(),
                        AdjMethod::Residual => dv
                            .get_group_membership(lb..ub)?
                            .into_iter()
                            .map(|j| j as u32)
                            .collect(),
                    };
                    let indices = Tensor::from_iter(membership.into_iter(), config.dev)?;
                    Ok(delta.index_select(&indices, 0)?)
                })
                .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let (log_z_nk, _) = encoder.forward_t(&x_vec, &x0_vec, false)?;
    let z_nk = log_z_nk.to_device(&candle_core::Device::Cpu)?;
    Ok((lb, Mat::from_tensor(&z_nk)?))
}

///////////////////////
// training routines //
///////////////////////

pub(crate) struct ProgressiveTrainConfig<'a> {
    pub parameters: &'a candle_nn::VarMap,
    pub dev: &'a candle_core::Device,
    pub args: &'a JointTopicArgs,
    pub coarsenings: &'a [Option<FeatureCoarsening>],
    pub stop: &'a AtomicBool,
}

pub(crate) fn train_encoder_decoder_progressive<Enc, Dec>(
    collapsed_levels: &[Vec<CollapsedOut>],
    encoder: &Enc,
    decoder: &Dec,
    config: &ProgressiveTrainConfig,
) -> anyhow::Result<TrainScores>
where
    Enc: JointEncoderModuleT,
    Dec: JointDecoderModuleT,
{
    let num_levels = collapsed_levels.len();
    let total_epochs = config.args.epochs;

    let level_epochs = compute_level_epochs(total_epochs, num_levels);

    info!(
        "Progressive training: {} levels, epoch allocation: {:?} (total {})",
        num_levels,
        level_epochs,
        level_epochs.iter().sum::<usize>()
    );

    let mut adam = AdamW::new_lr(
        config.parameters.all_vars(),
        f64::from(config.args.learning_rate),
    )?;

    let total_actual_epochs: usize = level_epochs.iter().sum();
    let pb = new_progress_bar(total_actual_epochs as u64);

    let mut llik_trace = Vec::with_capacity(total_actual_epochs);
    let mut kl_trace = Vec::with_capacity(total_actual_epochs);

    let num_modalities = collapsed_levels[0].len();
    let output_null: Vec<Option<Mat>> = vec![None; num_modalities];

    type LevelEntry = (Vec<Mat>, Vec<Option<Mat>>, Vec<Option<Mat>>);
    let level_data: Vec<LevelEntry> = collapsed_levels
        .iter()
        .map(|collapsed_data_vec| -> anyhow::Result<_> {
            let input = collapsed_data_vec
                .iter()
                .zip(config.coarsenings)
                .map(|(x, fc)| -> anyhow::Result<Mat> {
                    let mat = x
                        .mu_observed
                        .posterior_sample()?
                        .sum_to_one_columns()
                        .scale(config.args.column_sum_norm);
                    let mat = mat.transpose();
                    let mat = if let Some(fc) = fc {
                        fc.aggregate_columns_nd(&mat)
                    } else {
                        mat
                    };
                    Ok(mat)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let input_null = collapsed_data_vec
                .iter()
                .zip(config.coarsenings)
                .map(|(x, fc)| -> anyhow::Result<Option<Mat>> {
                    x.mu_residual
                        .as_ref()
                        .map(|y| {
                            let mut mat = y.posterior_sample()?.transpose();
                            if let Some(fc) = fc {
                                mat = fc.aggregate_columns_nd(&mat);
                            }
                            Ok(mat)
                        })
                        .transpose()
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let output = collapsed_data_vec
                .iter()
                .zip(config.coarsenings)
                .map(|(x, fc)| -> anyhow::Result<Option<Mat>> {
                    Ok(x.mu_adjusted
                        .as_ref()
                        .map(matrix_param::traits::Inference::posterior_sample)
                        .transpose()?
                        .map(|y| {
                            let mat = y.sum_to_one_columns().scale(config.args.column_sum_norm);
                            let mut mat = mat.transpose();
                            if let Some(fc) = fc {
                                mat = fc.aggregate_columns_nd(&mat);
                            }
                            mat
                        }))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            Ok((input, input_null, output))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut data_loaders: Vec<JointInMemoryData> = level_data
        .iter()
        .map(|(input, input_null, output)| {
            JointInMemoryData::from_device(
                JointInMemoryArgs {
                    input,
                    input_null,
                    output,
                    output_null: &output_null,
                },
                config.dev,
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    for (level, (collapsed_data_vec, &level_ep)) in
        collapsed_levels.iter().zip(level_epochs.iter()).enumerate()
    {
        let label = if level == 0 {
            "coarsest"
        } else if level + 1 == num_levels {
            "finest"
        } else {
            ""
        };
        info!(
            "Level {}/{}: {} epochs, {} samples {}",
            level + 1,
            num_levels,
            level_ep,
            collapsed_data_vec[0].mu_observed.ncols(),
            label,
        );

        let data_loader = &mut data_loaders[level];

        for epoch in 0..level_ep {
            data_loader.shuffle_minibatch_on_device(config.args.minibatch_size)?;

            let mut llik_tot = 0f32;
            let mut kl_tot = 0f32;

            for b in 0..data_loader.num_minibatch() {
                let mb = data_loader.minibatch_cached(b);

                let (z_nk, kl) = encoder.forward_t(&mb.input, &mb.input_null, true)?;

                let y_vec: Vec<Tensor> = mb
                    .output
                    .iter()
                    .zip(mb.input.iter())
                    .map(|(y, x)| y.clone().unwrap_or_else(|| x.clone()))
                    .collect();

                let (_, llik) = decoder.forward_with_llik(&z_nk, &y_vec, &topic_likelihood)?;

                let loss = (&kl - &llik)?.mean_all()?;
                adam.backward_step(&loss)?;

                let llik_val = llik.sum_all()?.to_scalar::<f32>()?;
                let kl_val = kl.sum_all()?.to_scalar::<f32>()?;
                llik_tot += llik_val;
                kl_tot += kl_val;
            }

            let n_mb = data_loader.num_minibatch() as f32;
            kl_trace.push(kl_tot / n_mb);
            llik_trace.push(llik_tot / n_mb);

            pb.inc(1);

            info!(
                "[level {}/{}][{}] {} {}",
                level + 1,
                num_levels,
                epoch,
                llik_tot / n_mb,
                kl_tot / n_mb
            );

            if config.stop.load(Ordering::SeqCst) {
                pb.finish_and_clear();
                info!(
                    "Stopping training early at level {}/{}, epoch {}",
                    level + 1,
                    num_levels,
                    epoch
                );
                return Ok(TrainScores {
                    llik: llik_trace,
                    kl: kl_trace,
                });
            }
        }
    }
    pb.finish_and_clear();

    info!("done model training");
    Ok(TrainScores {
        llik: llik_trace,
        kl: kl_trace,
    })
}
