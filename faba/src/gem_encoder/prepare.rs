//! Everything computed between "the counts are loaded" and "the optimizer can
//! start": the OUTPUT-only cell QC set, and the per-level training tensors with
//! their batch triple.

use anyhow::Context;
use candle_util::data::indexed::{GemIndexedArgs, GemIndexedData};
use log::{info, warn};
use matrix_param::traits::Inference;
use nalgebra::DMatrix;

use crate::gem_encoder::args::GemEncoderArgs;
use crate::gem_encoder::load::{per_gene_track_mean, PreparedData};

type Mat = DMatrix<f32>;

/// The per-level tensors the training loop consumes, plus the two normalizers
/// and the batch signal that inference has to reproduce exactly.
pub struct TrainingData {
    /// One entry per collapse level, coarse-first … finest-last.
    pub levels: Vec<GemIndexedData>,
    /// Per-level `mu_residual`, or `None` where there is no batch structure to
    /// regress out. The FINEST level's is expanded back onto cells at inference
    /// so the encoder sees the same divisor it trained with.
    pub residuals: Vec<Option<Mat>>,
    /// Per-gene typical nascent rate, from the finest pseudobulk posterior.
    pub nascent_mean: Vec<f32>,
    /// Per-gene typical mature rate, same source.
    pub mature_mean: Vec<f32>,
}

/// Which cells make it into the per-cell OUTPUT tables, or `None` for all of
/// them.
///
/// An OUTPUT filter, exactly as `senna bge` does it: every cell still informs
/// the collapse, the encoder and the dictionaries — only the per-cell tables are
/// filtered. Anything stronger would change what the model was fitted on, which
/// is a much larger claim than "these rows are not worth reporting".
pub fn cell_qc_keep(
    args: &GemEncoderArgs,
    prepared: &PreparedData,
) -> anyhow::Result<Option<Vec<usize>>> {
    let Some(cfg) = args.qc.to_config() else {
        return Ok(None);
    };
    if cfg.feature_min_cells > 0 {
        warn!(
            "--qc-feature-min-cells is ignored by gem-encoder \n\
	     (cell-only QC; the dictionary keeps all genes)"
        );
    }
    let data_vec = &prepared.data_vec;
    let report = data_beans::qc_lib::compute_qc(data_vec, &cfg, None).context("cell QC")?;
    let keep = report.emit_idx_unmasked();
    info!(
        "cell QC: {} / {} cells kept for OUTPUT \n\
	 ({} near-empty, {} MAD-outlier); \n\
         training uses all of them",
        keep.len(),
        data_vec.num_columns(),
        report.near_empty.iter().filter(|&&e| e).count(),
        report.n_cells_dropped,
    );
    if let Some(path) = args.qc.qc_report.as_deref() {
        data_beans::qc_lib::write_qc_report(path, &prepared.cell_names, &report)
            .context("writing the QC report")?;
        info!("wrote {path}");
    }
    Ok((keep.len() < data_vec.num_columns()).then_some(keep))
}

/// Build one [`GemIndexedData`] per collapse level.
///
/// # Posterior mean, not a sample
///
/// Posterior MEAN of the pseudobulk counts, matching `faba gem`
/// (`graph-embedding-util/src/fit/mod.rs`, which also reads only the mean).
///
/// An earlier version drew a posterior SAMPLE here on the theory that it
/// exposed the collapse's uncertainty to training. It did not: the draw
/// happens once, outside the epoch loop, so every epoch saw the same fixed
/// noisy matrix — all of the variance and none of the resampling benefit,
/// which is strictly worse than the mean. Sampling would only help if it were
/// redrawn per epoch, and that would mean rebuilding each level's top-K
/// selection every epoch.
///
/// # The batch triple
///
/// Batch adjustment enters as the `(observed, residual, adjusted)` triple
/// `masked_topic` uses: the encoder reads the batch-MIXED counts with the
/// residual as its batch signal, and the decoder is scored against the
/// batch-FREE `mu_adjusted`.
///
/// That asymmetry is the point. Correcting one side by division (the old
/// `--batch-adjust`) left the decoder scored on uncorrected counts, so batch
/// had nowhere to go but the factors — which is why the flag measured
/// backwards, raising sample-explained latent variance rather than lowering
/// it. Correcting the TARGET instead means nothing has to be restored
/// downstream, and the multinomial head's ignoring of `residual` stops
/// mattering.
pub fn build_training_data(
    args: &GemEncoderArgs,
    prepared: &PreparedData,
) -> anyhow::Result<TrainingData> {
    let collapsed_levels = &prepared.collapsed_levels;
    let map = &prepared.map;
    let finest = collapsed_levels
        .last()
        .context("collapse produced no levels")?;

    // Per-gene typical rate for each track, from the finest pseudobulk
    // posterior. The encoder divides by this before Anscombe so what it pools
    // is a cell's deviation from the gene's usual level, not the level itself.
    let mu_dp = finest.mu_observed.posterior_mean();
    let nascent_mean = per_gene_track_mean(mu_dp, map, true);
    let mature_mean = per_gene_track_mean(mu_dp, map, false);

    let level_mats: Vec<Mat> = collapsed_levels
        .iter()
        .map(|lvl| lvl.mu_observed.posterior_mean().transpose())
        .collect();

    let (residuals, targets) = batch_triple(args, prepared);

    let levels: Vec<GemIndexedData> = level_mats
        .iter()
        .zip(residuals.iter())
        .zip(targets.iter())
        .map(|((m, null), target)| {
            GemIndexedData::from_dense(GemIndexedArgs {
                observed: m,
                residual: null.as_ref(),
                adjusted: target.as_ref(),
                map,
                context_size: args.context_size,
                gene_weights: &prepared.gene_weights,
                nascent_mean: Some(&nascent_mean),
                mature_mean: Some(&mature_mean),
            })
        })
        .collect::<anyhow::Result<_>>()?;

    Ok(TrainingData {
        levels,
        residuals,
        nascent_mean,
        mature_mean,
    })
}

/// Per-level `(residual, adjusted)`, both `None` when there is no batch
/// structure to regress out — see [`build_training_data`].
fn batch_triple(
    args: &GemEncoderArgs,
    prepared: &PreparedData,
) -> (Vec<Option<Mat>>, Vec<Option<Mat>>) {
    let n_levels = prepared.collapsed_levels.len();
    // `(nulls, targets)` both absent: the encoder sees no batch signal and the
    // decoder scores against its own observed values. What "no batch structure"
    // means, whether that is because adjustment is off or because there is only
    // one batch to adjust.
    let no_batch = || (vec![None; n_levels], vec![None; n_levels]);

    if !args.batch_adjust {
        if prepared.data_vec.num_batches() > 1 {
            info!(
                "{} batches present, but '--batch-adjust' is off. \
                 Batch/condition structure will therefore appear in the latent,\n\
		 which is what you want when the groups are biological conditions, \n\
		 and not when they are technical.",
                prepared.data_vec.num_batches()
            );
        }
        return no_batch();
    }

    let pairs: Vec<(Option<Mat>, Option<Mat>)> = prepared
        .collapsed_levels
        .iter()
        .map(|lvl| {
            (
                lvl.mu_residual
                    .as_ref()
                    .map(|r| r.posterior_mean().transpose()),
                lvl.mu_adjusted
                    .as_ref()
                    .map(|a| a.posterior_mean().transpose()),
            )
        })
        .collect();
    let have = pairs.iter().filter(|(_, a)| a.is_some()).count();
    if have == 0 {
        // One batch ⇒ nothing to adjust, and that is not an error: the
        // adjusted target IS `mu_observed` and the residual is `None`, which
        // is exactly what `(None, None)` already means downstream. Failing
        // here made the DEFAULT arguments reject a single-file run, and the
        // message named `--batch-adjust` as the flag to drop when the flag
        // is on by default and spelled `--no-batch-adjust`.
        info!(
            "--batch-adjust: every cell is in one batch, \n\
	     so there is nothing to regress out: \n\
	     the decoder targets mu_observed and no residual is fed.\n\
             This is the correct no-op, not a fallback."
        );
        return no_batch();
    }
    info!(
        "--batch-adjust: encoder reads observed + residual, decoder targets \n\
         mu_adjusted, on {have} level(s)"
    );
    pairs.into_iter().unzip()
}
