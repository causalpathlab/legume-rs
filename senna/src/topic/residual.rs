use super::eval::{evaluate_latent_by_encoder, EvaluateLatentConfig};
use super::train::{train_mixed_multi_decoder, TrainConfig};
use crate::embed_common::*;
use crate::fit_topic::TopicArgs;

use candle_util::candle_decoder_topic::TopicDecoder;
use candle_util::candle_dyn_decoder::*;
use candle_util::candle_encoder_softmax::*;
use data_beans_alg::collapse_data::CollapsedOut;
use log::info;
use matrix_param::dmatrix_gamma::GammaMatrix;
use nalgebra::DMatrix;

type Mat = DMatrix<f32>;

/// Compute pseudobulk residuals: log1p(observed) - log1p(predicted).
///
/// `dict_dk` is [D × K] log-probability dictionary from the finest decoder.
/// `collapsed` provides the observed pseudobulk [D × S].
///
/// Pseudobulk topic proportions are estimated by projecting pseudobulk means
/// onto the dictionary: z_sk = softmax(log(obs_ds).T @ dict_dk).
///
/// Returns residual [D × S] (genes × pseudobulk samples).
pub(crate) fn compute_pseudobulk_residuals(
    dict_dk: &Mat,
    collapsed: &CollapsedOut,
) -> anyhow::Result<Mat> {
    let observed_ds = collapsed.mu_observed.posterior_mean(); // [D, S]
    let n_samples = observed_ds.ncols();
    let n_genes = dict_dk.nrows();
    let n_topics = dict_dk.ncols();

    assert_eq!(observed_ds.nrows(), n_genes);

    // Estimate pseudobulk z by projecting log(obs) onto log-prob dictionary:
    // score_sk = log(obs_ds + 1).T @ dict_dk  [S, K]
    // z_sk = softmax(score_sk, dim=1)          [S, K]
    let log_obs_ds = observed_ds.map(|v| (v + 1.0).ln()); // [D, S]
    let score_sk = log_obs_ds.transpose() * dict_dk; // [S, K]

    // Row-wise softmax
    let mut z_probs_sk = Mat::zeros(n_samples, n_topics);
    for s in 0..n_samples {
        let row = score_sk.row(s);
        let max_val = row.max();
        let mut sum_exp = 0.0_f32;
        for k in 0..n_topics {
            let e = (row[k] - max_val).exp();
            z_probs_sk[(s, k)] = e;
            sum_exp += e;
        }
        for k in 0..n_topics {
            z_probs_sk[(s, k)] /= sum_exp;
        }
    }

    // predicted proportions [S, D]
    let beta_kd: Mat = dict_dk.map(|v| v.exp()).transpose(); // [K, D]
    let predicted_sd = &z_probs_sk * &beta_kd; // [S, D]

    // Per-sample library sizes
    let lib_sizes: Vec<f32> = (0..n_samples)
        .map(|s| observed_ds.column(s).sum())
        .collect();

    // residual[g, s] = log1p(obs) - log1p(pred * lib_size)
    let mut residual_ds = Mat::zeros(n_genes, n_samples);
    for s in 0..n_samples {
        let lib = lib_sizes[s];
        for g in 0..n_genes {
            let obs = observed_ds[(g, s)];
            let pred = predicted_sd[(s, g)] * lib;
            residual_ds[(g, s)] = (obs + 1.0).ln() - (pred + 1.0).ln();
        }
    }

    Ok(residual_ds)
}

/// Wrap a residual matrix into a CollapsedOut with proper Gamma posterior
/// for stochastic sampling during training (data augmentation).
///
/// Shifts residuals to positive (Gamma requires positive support), then
/// sets Gamma sufficient statistics with moderate pseudo-count so that
/// `posterior_sample()` adds meaningful noise.
fn residual_to_collapsed(residual_ds: &Mat) -> CollapsedOut {
    let min_val = residual_ds.min();
    let shift = if min_val < 0.0 { -min_val + 1e-4 } else { 1e-4 };
    let shifted = residual_ds.map(|v| v + shift);

    let pseudo_count = 10.0_f32;
    let a_stat = shifted.map(|v| v * pseudo_count);
    let b_stat = Mat::from_element(shifted.nrows(), shifted.ncols(), pseudo_count);

    CollapsedOut {
        mu_observed: GammaMatrix::from_sufficient_stats(a_stat, b_stat),
        mu_adjusted: None,
        mu_residual: None,
        gamma: None,
        delta: None,
    }
}

pub(crate) struct ResidualModelInput<'a> {
    pub dict_dk: &'a Mat,
    pub coarsening: Option<&'a FeatureCoarsening>,
    pub args: &'a TopicArgs,
    pub decoder_weights: &'a [f64],
    pub finest_collapsed: &'a CollapsedOut,
    pub data_vec: &'a SparseIoVec,
    pub cell_names: &'a [Box<str>],
}

/// Train a residual topic model and evaluate latent states.
///
/// Uses the same model class (encoder + decoder types) as the main model.
pub(crate) fn fit_residual_model(input: &ResidualModelInput) -> anyhow::Result<()> {
    let dict_dk = input.dict_dk;
    let coarsening = input.coarsening;
    let args = input.args;
    let decoder_weights = input.decoder_weights;
    let finest_collapsed = input.finest_collapsed;
    let data_vec = input.data_vec;
    let cell_names = input.cell_names;
    // Coarsen the observed pseudobulk to match the decoder's feature resolution
    let coarsened_collapsed;
    let working_collapsed = if let Some(fc) = coarsening {
        let obs_mean = finest_collapsed.mu_observed.posterior_mean();
        let coarsened_obs = fc.aggregate_rows_ds(obs_mean);
        let pseudo_count = 10.0_f32;
        let a_stat = coarsened_obs.map(|v| v.max(1e-10) * pseudo_count);
        let b_stat = Mat::from_element(coarsened_obs.nrows(), coarsened_obs.ncols(), pseudo_count);
        coarsened_collapsed = CollapsedOut {
            mu_observed: GammaMatrix::from_sufficient_stats(a_stat, b_stat),
            mu_adjusted: None,
            mu_residual: None,
            gamma: None,
            delta: None,
        };
        &coarsened_collapsed
    } else {
        finest_collapsed
    };

    info!("Computing pseudobulk residuals for residual model");
    let residual_ds = compute_pseudobulk_residuals(dict_dk, working_collapsed)?;
    info!(
        "Residual matrix: {} genes × {} samples, range [{:.2}, {:.2}]",
        residual_ds.nrows(),
        residual_ds.ncols(),
        residual_ds.min(),
        residual_ds.max(),
    );

    let residual_collapsed = residual_to_collapsed(&residual_ds);
    let collapsed_levels = vec![residual_collapsed];

    let n_features = residual_ds.nrows();
    let n_topics = dict_dk.ncols();

    // Fresh parameters on CPU
    let resid_parameters = candle_nn::VarMap::new();
    let resid_dev = candle_core::Device::Cpu;
    let resid_builder =
        candle_nn::VarBuilder::from_varmap(&resid_parameters, candle_core::DType::F32, &resid_dev);

    let mut resid_encoder = LogSoftmaxEncoder::new(
        LogSoftmaxEncoderArgs {
            n_features,
            n_topics,
            layers: &args.encoder_layers,
        },
        resid_builder.clone(),
    )?;

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let no_coarsenings: Vec<Option<FeatureCoarsening>> = vec![None];

    // Clone args and override epochs for residual training
    let mut resid_args = args.clone();
    resid_args.epochs = args.residual_epochs;
    resid_args.vcd_epochs = 0;

    let train_config = TrainConfig {
        parameters: &resid_parameters,
        dev: &resid_dev,
        args: &resid_args,
        stop: &stop,
    };

    info!(
        "Training residual model: {} topics, {} epochs, {} decoder(s)",
        n_topics,
        args.residual_epochs,
        args.decoder.len()
    );

    let decoders_per_level: Vec<Vec<Box<dyn DynDecoderModuleT>>> = vec![args
        .decoder
        .iter()
        .map(|dec_type| {
            let name = dec_type.as_str();
            let prefix = format!("dec_0.{name}");
            create_dyn_decoder(name, n_features, n_topics, resid_builder.pp(prefix))
                .expect("decoder creation")
        })
        .collect()];

    train_mixed_multi_decoder(
        &collapsed_levels,
        &mut resid_encoder,
        &decoders_per_level,
        &no_coarsenings,
        decoder_weights,
        &train_config,
    )?;

    // Evaluate residual latent on CPU (already on CPU)
    info!("Evaluating residual latent states");

    let eval_config = EvaluateLatentConfig {
        dev: &resid_dev,
        adj_method: &args.adj_method,
        minibatch_size: args.minibatch_size,
        feature_coarsening: coarsening,
        decoder: None::<&TopicDecoder>,
        refine_config: None,
    };

    let resid_z =
        evaluate_latent_by_encoder(data_vec, &resid_encoder, finest_collapsed, &eval_config)?;

    resid_z.to_parquet_with_names(
        &(args.out.to_string() + ".residual_latent.parquet"),
        (Some(cell_names), Some("cell")),
        None,
    )?;

    info!("Residual model done");
    Ok(())
}
