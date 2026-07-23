//! Entry point for `faba gem-encoder`.
//! **Velocity** is the post-hoc cell-level delta `Δθ = θ^u − θ^s`, each θ fitted
//! to its own track against the frozen dictionaries. The model itself has ONE
//! latent, so it cannot express that difference during training — deliberately,
//! since a latent delta competes with `δ` for the same signal and the encoder
//! wins (see [`candle_util::vae::masked_gem`]).
//!
//! ## Batch handling
//!
//! One vocabulary throughout, the collapse's own:
//!
//! ```text
//! mu_observed  ≈  mu_adjusted · mu_residual
//! ```
//!
//! With `--batch-adjust` (ON by default) the encoder reads `observed` with
//! `residual` as its batch signal — **per track**, since the residual is fitted
//! per row and intronic capture varies by batch differently from exonic — while
//! the decoder is scored against `adjusted`. At inference the encoder gets the
//! same pair: a cell's observed counts plus its pseudobulk's residual, so the
//! two phases see the same input distribution.
//!
//! Batch identity resolves in three tiers — `--batch-files`, then an embedded
//! `@`-tag on the cell names, then the file name. With several inputs and no
//! `--batch-files` the batches are the SAMPLES, so on a `rep{1,2,3}_{wt,mut}`
//! design adjustment removes the wt/mut contrast along with donor effects. If
//! that contrast is the biology, pass `--batch-files` or `--no-batch-adjust`.
//!
//! ## Where the work lives
//!
//! This module is the sequence and nothing else: [`crate::gem_encoder::load`] →
//! [`crate::gem_encoder::prepare`] → training → [`crate::gem_encoder::infer`],
//! with every artifact written through [`crate::gem_encoder::write`],
//! [`crate::gem_encoder::report`] and [`crate::gem_encoder::velocity`].

use anyhow::Context;
use candle_util::candle_core::{DType, Device};
use candle_util::candle_nn::{VarBuilder, VarMap};
use candle_util::decoder::gem_etm::GemEtmDecoder;
use candle_util::encoder::gem_encoder::{GemIndexedEncoder, GemIndexedEncoderArgs};
use candle_util::vae::masked_gem::{train_masked_gem, GemTrainConfig, GemTrainOpts};
use log::{info, warn};
use matrix_util::common_io::mkdir_parent;
use rayon::ThreadPoolBuilder;

use crate::gem_encoder::args::GemEncoderArgs;
use crate::gem_encoder::infer::infer_cells;
use crate::gem_encoder::load::{load_and_collapse, PreparedData};
use crate::gem_encoder::prepare::{build_training_data, cell_qc_keep};
use crate::gem_encoder::report::{
    report_training_health, save_model_metadata, write_scores, write_splice_ratio_qc,
};
use crate::gem_encoder::velocity::write_velocity_tables;
use crate::gem_encoder::write::{
    write_cell_embedding, write_cell_table, write_coembedding, write_dictionaries,
    write_feature_embedding, write_pseudobulk_tables,
};

/// `faba gem-encoder` — the latent head, likelihood, masking schedule and batch
/// handling are all flags; see [`GemEncoderArgs`].
pub fn run_gem_encoder(args: &GemEncoderArgs) -> anyhow::Result<()> {
    args.validate()?;
    mkdir_parent(&args.out)?;
    let dev = init_runtime(args)?;

    let prepared = load_and_collapse(args)?;
    let PreparedData {
        ref collapsed_levels,
        ref map,
        ref gene_names,
        ref cell_names,
        ref row_mean,
        ..
    } = prepared;
    let finest = collapsed_levels
        .last()
        .context("collapse produced no levels")?;

    let qc_keep = cell_qc_keep(args, &prepared)?;
    let qc = qc_keep.as_deref();

    let mut training = build_training_data(args, &prepared)?;

    ///////////
    // model //
    ///////////
    let parameters = VarMap::new();
    let (encoder, decoders) =
        build_model(args, collapsed_levels.len(), map.n_genes, &parameters, &dev)?;

    let stop = graph_embedding_util::setup_stop_handler();
    let opts = GemTrainOpts {
        likelihood: args.likelihood.to_lib(),
        mask_fraction: args.mask_fraction,
        delta_l2: args.delta_l2,
        feature_embedding_l2: args.feature_embedding_l2,
        topic_smoothing: args.topic_smoothing,
        // The nascent track is scored only where its count is positive,
        // symmetrically with the mature one. Scoring the zeros too was a flag
        // until it was measured: it makes the nascent objective mostly "predict
        // zero", and because each track is normalized by its own scored-position
        // count the easy task ends up with equal weight. On a six-sample fit it
        // moved the spliced likelihood from -2.88 to -4.27 and the splice-ratio
        // check from r = 0.37 to r = 0.33. `GemTrainOpts` keeps the knob.
        nascent_observed_nonzero_only: true,
    };

    let scores = train_masked_gem(
        &mut training.levels,
        &encoder,
        &decoders,
        &GemTrainConfig {
            parameters: &parameters,
            dev: &dev,
            epochs: args.epochs,
            minibatch_size: args.minibatch_size,
            learning_rate: args.learning_rate,
            weight_decay: args.weight_decay,
            grad_clip: args.grad_clip,
            stop: &stop,
        },
        &opts,
    )?;

    if stop.load(std::sync::atomic::Ordering::Relaxed) {
        warn!(
            "Interrupted! Training stopped early, so every output below reflects \n\
	     a partially-fitted model. The latent and velocity are still written \n\
	     (that is usually what you want after a Ctrl-C), \n\
	     but do not compare them against a complete run."
        );
    }
    report_training_health(&scores, args.delta_l2);

    /////////////
    // outputs //
    /////////////
    let finest_decoder = decoders.last().context("no decoder")?;
    write_dictionaries(finest_decoder, &encoder, gene_names, &args.out)?;
    write_scores(&scores, &args.out)?;
    write_splice_ratio_qc(finest_decoder, gene_names, map, row_mean, &args.out)?;

    // Per-cell inference: three encoder passes per block, streamed straight off
    // the sparse backend.
    let inferred = infer_cells(&prepared, &encoder, finest_decoder, &training, args, &dev)?;
    write_cell_table(
        &inferred.latent,
        cell_names.len(),
        args.n_latent,
        &format!("{}.latent.parquet", args.out),
        cell_names,
        "T",
        qc,
    )?;
    // The two per-track fits. `latent.parquet` stays the ENCODER's θ — it is
    // what `faba annotate --mode enrichment` reads as cell membership and what
    // that result was validated on. These are the post-hoc fits against the
    // frozen dictionaries; their difference is the cell-level delta.
    for (buf, suffix) in [
        (&inferred.latent_mature, "latent_mature"),
        (&inferred.latent_nascent, "latent_nascent"),
    ] {
        write_cell_table(
            buf,
            cell_names.len(),
            args.n_latent,
            &format!("{}.{suffix}.parquet", args.out),
            cell_names,
            "T",
            qc,
        )?;
    }
    write_pseudobulk_tables(
        &prepared,
        finest,
        &inferred.latent,
        args.n_latent,
        &args.out,
    )?;

    // The basis of the shared cell/gene space: `α [K,H]` from the decoder,
    // `ρ+δ [G,H]` from the encoder. Everything below — the cell embedding, the
    // co-embedded features, the velocity projection — is written off this one
    // pair.
    let alpha = finest_decoder.topic_embeddings().to_device(&Device::Cpu)?;
    let (mature_feat, nascent_feat) = write_feature_embedding(&encoder, gene_names, &args.out)?;
    let z = write_cell_embedding(
        &inferred.latent,
        &alpha,
        cell_names,
        args.n_latent,
        &args.out,
        qc,
    )?;
    write_coembedding(
        &z,
        &mature_feat,
        &nascent_feat,
        gene_names,
        args.n_latent,
        &args.out,
    )?;

    let common_mode =
        write_velocity_tables(&inferred, &alpha, cell_names, args.n_latent, &args.out, qc)?;
    save_model_metadata(args, map.n_genes, &common_mode, &parameters)?;

    info!("done (gem-encoder) — prefix '{}'", args.out);
    Ok(())
}

/// Rayon pool, then compute device. The device probe is the first thing that
/// fails on a misconfigured GPU, so it happens before any data is read.
fn init_runtime(args: &GemEncoderArgs) -> anyhow::Result<Device> {
    let n_threads = if args.runtime.threads == 0 {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    } else {
        args.runtime.threads
    };
    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok();
    info!(
        "rayon thread pool: {} threads",
        rayon::current_num_threads()
    );

    let dev = args
        .runtime
        .device
        .to_device(args.runtime.device_no)
        .context("compute device init")?;
    info!("compute device = {dev:?}");
    Ok(dev)
}

/// The encoder plus one decoder per collapse level.
///
/// The decoders share the encoder's `ρ` and `δ` handles (ETM tying): one Var,
/// two gradient paths. They are per-level so each collapse scale gets its own
/// `α`.
fn build_model(
    args: &GemEncoderArgs,
    n_levels: usize,
    n_genes: usize,
    parameters: &VarMap,
    dev: &Device,
) -> anyhow::Result<(GemIndexedEncoder, Vec<GemEtmDecoder>)> {
    let vb = VarBuilder::from_varmap(parameters, DType::F32, dev);

    let encoder = GemIndexedEncoder::new(
        GemIndexedEncoderArgs {
            n_genes,
            n_latent: args.n_latent,
            embedding_dim: args.embedding_dim,
            layers: &args.encoder_layers,
            latent_noise: args.latent_noise,
        },
        parameters,
        vb.pp("enc"),
    )
    .context("encoder init")?;

    let rho = encoder.feature_embeddings().clone();
    let delta = encoder.delta_embeddings().clone();
    let decoders: Vec<GemEtmDecoder> = (0..n_levels)
        .map(|i| {
            GemEtmDecoder::new(
                args.n_latent,
                rho.clone(),
                delta.clone(),
                vb.pp(format!("dec_{i}")),
            )
        })
        .collect::<candle_util::candle_core::Result<_>>()
        .context("decoder init")?;

    info!(
        "model: {} genes → ρ,δ [{}×{}] → {} factors ({} levels, ctx {})",
        n_genes, n_genes, args.embedding_dim, args.n_latent, n_levels, args.context_size,
    );
    Ok((encoder, decoders))
}
