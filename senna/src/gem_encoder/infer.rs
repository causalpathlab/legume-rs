//! Per-cell inference: stream the sparse backend in blocks, run the encoder on
//! each, and land the result in the one latent contract every consumer reads.

use candle_util::candle_core::Device;
use candle_util::data::indexed::{gem_samples_from_csc, GemIndexedData};
use candle_util::decoder::gem_etm::{GemEtmDecoder, Track};
use candle_util::encoder::gem_encoder::GemIndexedEncoder;
use candle_util::vae::masked_gem::{fit_theta_to_track, infer_minibatch, ThetaFitConfig};
use log::info;

use crate::gem_encoder::args::GemEncoderArgs;
use crate::gem_encoder::load::PreparedData;
use crate::gem_encoder::prepare::TrainingData;

pub struct Inferred {
    /// `[N, K]` **log θ**, not the encoder's raw logits.
    ///
    /// The map is applied once, at the end of inference, so that this buffer,
    /// `{out}.latent.parquet`, and every consumer agree. Raw logits and log θ
    /// are the same shape and differ only by a per-row constant, so getting
    /// this wrong does not error — it silently yields a plausible, wrong θ,
    /// which invalidated two rounds of analysis in this project. Holding log θ
    /// also matches senna's topic-family contract (`RunKind::
    /// latent_is_log_simplex`), which is what `senna rest` requires of a run it
    /// freezes θ from.
    pub latent: Vec<f32>,
    /// `[N, K]` **log θ** fitted to the MATURE track alone, against the frozen
    /// dictionary — see [`fit_theta_to_track`].
    pub latent_mature: Vec<f32>,
    /// `[N, K]` **log θ** fitted to the NASCENT track alone.
    ///
    /// `exp(latent_nascent) − exp(latent_mature)` is the cell-level delta this
    /// model deliberately cannot express during training. It replaces the old
    /// two-pass `Δz = z_u − z_s`, which was a difference of ENCODER passes and
    /// therefore measured the encoder's response to input sparsity as much as
    /// anything biological (55-95 % of its own sum-of-squares was a between-pass
    /// offset). These two are fits to the COUNTS under frozen dictionaries, so
    /// they are depth-invariant compositions and their difference is a genuine
    /// movement between two reachable states.
    pub latent_nascent: Vec<f32>,
}

/// Stream the sparse backend in blocks, running the three inference passes on
/// each. Never materializes a dense cell × gene matrix.
///
/// The encoder is handed the same `(nascent_mean, mature_mean, residual)` it was
/// trained against — see [`TrainingData`] — so the two phases see one input
/// distribution.
pub fn infer_cells(
    prepared: &PreparedData,
    encoder: &GemIndexedEncoder,
    decoder: &GemEtmDecoder,
    training: &TrainingData,
    args: &GemEncoderArgs,
    dev: &Device,
) -> anyhow::Result<Inferred> {
    let n_cells = prepared.data_vec.num_columns();
    let k = args.n_latent;
    let mut latent = vec![0f32; n_cells * k];
    let mut latent_mature = vec![0f32; n_cells * k];
    let mut latent_nascent = vec![0f32; n_cells * k];
    let fit_cfg = ThetaFitConfig::default();

    let gene_weights = &prepared.gene_weights;
    let residual_pd = training.residuals.last().and_then(Option::as_ref);

    let block = args.minibatch_size.max(1);
    let bar = matrix_util::progress::new_progress_bar(n_cells as u64).with_message("Inference");
    if residual_pd.is_some() {
        info!("inference: expanding the per-batch residual per cell (matches training)");
    }

    let mut lb = 0usize;
    while lb < n_cells {
        let ub = (lb + block).min(n_cells);
        let x_dn = prepared.data_vec.read_columns_csc(lb..ub)?;
        let samples = gem_samples_from_csc(&x_dn, &prepared.map, gene_weights, args.context_size);
        // Expand the pseudobulk-group residual onto this block's cells, so the
        // encoder is handed the same divisor it was trained with. Without this
        // the two sides disagree and the latent drifts off-distribution.
        let residual_rows = residual_pd
            .map(|null_pd| -> anyhow::Result<Vec<Vec<f32>>> {
                let groups = prepared.data_vec.get_group_membership(lb..ub)?;
                Ok(groups
                    .into_iter()
                    .map(|g| {
                        let g = g.min(null_pd.nrows().saturating_sub(1));
                        null_pd.row(g).iter().copied().collect()
                    })
                    .collect())
            })
            .transpose()?;
        let data = GemIndexedData::from_samples(
            samples,
            &prepared.map,
            args.context_size,
            Some(&training.nascent_mean),
            Some(&training.mature_mean),
            residual_rows,
        )?;
        let mb = data.minibatch_ordered(0, ub - lb, dev)?;
        // `true` = nascent scored only where positive, matching training.
        let z = infer_minibatch(encoder, &mb, true, dev)?;
        // Post-hoc per-track fits, WARM-STARTED from the encoder's own z: that
        // is what makes them close the amortization gap rather than start over.
        let z_s = fit_theta_to_track(decoder, &mb, &z, Track::Mature, &fit_cfg)?;
        let z_u = fit_theta_to_track(decoder, &mb, &z, Track::Nascent, &fit_cfg)?;

        // Raw encoder logits → log θ, once, here. See `Inferred::latent`.
        let z_host: Vec<f32> = candle_util::vae::masked_gem::theta_log_simplex(&z)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;
        latent[lb * k..ub * k].copy_from_slice(&z_host);
        // Already log θ — `fit_theta_to_track` returns the simplex map applied.
        let host = |t: &candle_util::candle_core::Tensor| -> anyhow::Result<Vec<f32>> {
            Ok(t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?)
        };
        latent_mature[lb * k..ub * k].copy_from_slice(&host(&z_s)?);
        latent_nascent[lb * k..ub * k].copy_from_slice(&host(&z_u)?);

        bar.inc((ub - lb) as u64);
        lb = ub;
    }
    bar.finish_and_clear();

    Ok(Inferred {
        latent,
        latent_mature,
        latent_nascent,
    })
}
