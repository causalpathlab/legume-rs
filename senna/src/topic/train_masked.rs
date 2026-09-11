//! Senna-side glue for the masked-topic trainer.
//!
//! The training hot loop lives in [`candle_util::vae::masked_topic`].
//! This module owns senna-specific bits the candle-util trainer does not
//! see: per-level data assembly from [`CollapsedOut`], bulk-vs-SC delta
//! estimation, bulk evaluation, and the dictionary / feature-embedding
//! writers. The `pub(crate) use` re-exports keep existing call sites
//! (the `masked_topic` command, `train_cell_embedded`) on stable import paths.

use super::common::sample_collapsed_data;
use crate::embed_common::*;

use candle_core::Tensor;
use candle_util::encoder::IndexedEmbeddingEncoder;

// Re-export the generic trainer surface so legacy call sites stay put.
pub(crate) use candle_util::vae::masked_topic::IndexedTrainConfig;

/// Materialize per-level `(mixed, batch, target)` `Mat` triples once
/// per training run.
fn build_level_data(
    collapsed_levels: &[CollapsedOut],
) -> anyhow::Result<Vec<(Mat, Option<Mat>, Mat)>> {
    collapsed_levels.iter().map(sample_collapsed_data).collect()
}

/// Senna wrapper around [`candle_util::vae::masked_topic::train_masked`] —
/// the masked-imputation (no-ELBO) embedded topic model.
pub(crate) fn train_masked(
    collapsed_levels: &[CollapsedOut],
    encoder: &IndexedEmbeddingEncoder,
    decoders: &[candle_util::decoder::EmbeddedNbTopicDecoder],
    query_decoder: Option<&candle_util::decoder::query_decoder::QueryDecoder>,
    config: &IndexedTrainConfig,
    mask_fraction: f64,
    opts: &candle_util::vae::masked_topic::MaskedTrainOpts,
) -> anyhow::Result<TrainScores> {
    let level_data = build_level_data(collapsed_levels)?;
    let level_refs: Vec<candle_util::vae::masked_topic::LevelData> = level_data
        .iter()
        .map(|(a, b, c)| (a, b.as_ref(), c))
        .collect();
    let scores = candle_util::vae::masked_topic::train_masked(
        &level_refs,
        encoder,
        decoders,
        query_decoder,
        config,
        mask_fraction,
        opts,
    )?;
    Ok(TrainScores {
        llik: scores.llik,
        kl: scores.kl,
    })
}

/// Pull `tensor` to host and write it to `{out_prefix}.{suffix}.parquet`,
/// labelling its last-dim columns with `{col_prefix}0..H` and rows with
/// `row_names` under the column name `row_axis`.
fn write_tensor_parquet(
    tensor: &Tensor,
    out_prefix: &str,
    suffix: &str,
    row_names: &[Box<str>],
    row_axis: &str,
    col_prefix: &str,
) -> anyhow::Result<()> {
    let cols = axis_id_names(col_prefix, tensor.dims().last().copied().unwrap_or(0));
    write_tensor_parquet_named(tensor, out_prefix, suffix, row_names, row_axis, &cols)
}

/// [`write_tensor_parquet`] with the column names given rather than generated.
fn write_tensor_parquet_named(
    tensor: &Tensor,
    out_prefix: &str,
    suffix: &str,
    row_names: &[Box<str>],
    row_axis: &str,
    col_names: &[Box<str>],
) -> anyhow::Result<()> {
    let host = tensor.to_device(&candle_core::Device::Cpu)?;
    // Refuse to persist a diverged artifact. The masked/joint dictionary,
    // dispersion, and feature-embedding all funnel through here and are written
    // *before* the latent's own guard would fire, so without this a diverged run
    // leaves NaN `dictionary.parquet` on disk. (`save_latent`/`save_dictionary`
    // in output_helpers.rs guard the other writers; this covers the tensor path.)
    let path = format!("{out_prefix}.{suffix}");
    let bad = host
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .filter(|x| !x.is_finite())
        .count();
    anyhow::ensure!(
        bad == 0,
        "refusing to write {path}: {bad} non-finite (NaN/Inf) entries — training \
         diverged (check the log_likelihood trace and any \"skipped optimizer step\" \
         warnings; re-run with a lower --learning-rate)."
    );
    host.to_parquet_with_names(&path, (Some(row_names), Some(row_axis)), Some(col_names))?;
    Ok(())
}

/// The decoder's gene → coarse-group map for one level, with the pinned
/// within-group shares, and the per-group mass the background is pinned at.
///
/// A gene's share of its group is its mean rate over the group's total
/// (uniform within a group that has no mass); the group's background mass
/// is that total. With no coarsening the map is the identity and the masses
/// are the gene means themselves.
pub(crate) fn coarsening_map_for(
    coarsening: Option<&FeatureCoarsening>,
    feature_mean: &[f32],
    dev: &candle_core::Device,
) -> anyhow::Result<(
    candle_util::decoder::coarsening_map::CoarseningMap,
    Vec<f32>,
)> {
    use candle_util::decoder::coarsening_map::CoarseningMap;
    let d = feature_mean.len();
    let Some(fc) = coarsening else {
        return Ok((CoarseningMap::identity(d, dev)?, feature_mean.to_vec()));
    };
    anyhow::ensure!(
        fc.fine_to_coarse.len() == d,
        "feature coarsening covers {} genes but the run has {d}",
        fc.fine_to_coarse.len()
    );
    let mut coarse_mass = vec![0f32; fc.num_coarse];
    for (g, &m) in fc.fine_to_coarse.iter().enumerate() {
        coarse_mass[m] += feature_mean[g].max(0.0);
    }
    let share: Vec<f32> = fc
        .fine_to_coarse
        .iter()
        .enumerate()
        .map(|(g, &m)| {
            if coarse_mass[m] > 0.0 {
                feature_mean[g].max(0.0) / coarse_mass[m]
            } else {
                1.0 / fc.coarse_to_fine[m].len().max(1) as f32
            }
        })
        .collect();
    Ok((
        CoarseningMap::new(&fc.fine_to_coarse, &share, dev)?,
        coarse_mass,
    ))
}

/// Expand a coarse-level log-dictionary `[C, K]` to genes `[D, K]`:
/// `log β_kg = log β^coarse_{k,c(g)} + log π_{g|c(g)}`, so a gene takes its
/// pinned share of its group's mass and every column still sums to one.
pub(crate) fn expand_log_dict_with_shares(
    log_dict_mk: &Mat,
    fine_to_coarse: &[usize],
    log_share: &[f32],
) -> Mat {
    let k = log_dict_mk.ncols();
    Mat::from_fn(fine_to_coarse.len(), k, |g, kk| {
        log_dict_mk[(fine_to_coarse[g], kk)] + log_share[g]
    })
}

/// Write the `[D, K]` log-β dictionary + the per-gene dispersion `φ` for the
/// masked-imputation NB embedded topic decoder. A coarsened decoder is
/// expanded to genes through its pinned shares; `φ` is per coarse group, so
/// every gene of a group carries its group's dispersion.
pub(crate) fn write_masked_dictionary(
    decoder: &candle_util::decoder::EmbeddedNbTopicDecoder,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    let map = decoder.coarsening();
    let dict = decoder
        .get_dictionary()?
        .to_device(&candle_core::Device::Cpu)?;
    let phi_1d = decoder.phi()?.to_device(&candle_core::Device::Cpu)?; // [1, M]
    let (dict_dk, phi_d1) = if map.is_identity() {
        (dict, phi_1d.transpose(0, 1)?.contiguous()?)
    } else {
        let dict_mk = Mat::from_tensor(&dict)?;
        let expanded =
            expand_log_dict_with_shares(&dict_mk, map.host_fine_to_coarse(), map.host_log_share());
        let phi_m: Vec<f32> = phi_1d.flatten_all()?.to_vec1()?;
        let phi_d: Vec<f32> = map
            .host_fine_to_coarse()
            .iter()
            .map(|&m| phi_m[m])
            .collect();
        log::info!(
            "Expanded dictionary from {} coarse features to {} genes through the pinned shares",
            map.n_coarse(),
            map.n_fine()
        );
        (
            expanded.to_tensor(&candle_core::Device::Cpu)?,
            Tensor::from_vec(phi_d, (map.n_fine(), 1), &candle_core::Device::Cpu)?,
        )
    };
    write_tensor_parquet(
        &dict_dk,
        out_prefix,
        "dictionary.parquet",
        gene_names,
        "gene",
        "T",
    )?;
    write_tensor_parquet(
        &phi_d1,
        out_prefix,
        "dispersion.parquet",
        gene_names,
        "gene",
        "phi",
    )
}

/// Write the learned per-gene feature embedding ρ `[D, H]` as a parquet.
/// In the ETM factorization this is shared between encoder and decoder, so
/// it's the model's gene-level representation — directly usable for gene-gene
/// similarity, clustering into programs, or initializing downstream models.
pub(crate) fn write_feature_embedding(
    feature_embeddings: &Tensor,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    write_tensor_parquet(
        feature_embeddings,
        out_prefix,
        "feature_embedding.parquet",
        gene_names,
        "gene",
        "H",
    )
}

/// Write the encoder's learned gene modules in the shape the graph-embedding
/// family writes them: a `[D, M]` membership and an `[M, H]` dictionary, under
/// the same suffixes.
///
/// Both families now learn a membership per feature and compose a row from
/// shared module vectors, so the artifact means the same thing on either side.
/// They still differ in what surrounds it — the graph-embedding family keeps a
/// per-feature residual and warm-starts its membership from a clustering,
/// while this one composes without a residual from a flat start — so the
/// tables are shared, not the parameterization.
///
/// Returns the two suffixes when the encoder has modules, so the caller can
/// record them in the manifest, and `None` when it has none.
pub(crate) fn write_gene_modules(
    encoder: &IndexedEmbeddingEncoder,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<Option<(&'static str, &'static str)>> {
    let Some(membership) = encoder.feature_module_membership()? else {
        return Ok(None);
    };
    let module_names: Vec<Box<str>> = (0..encoder.n_gene_modules())
        .map(|i| format!("m{i}").into_boxed_str())
        .collect();
    let membership_suffix = graph_embedding_util::transfer::MODULE_MEMBERSHIP_SUFFIX;
    let dictionary_suffix = graph_embedding_util::transfer::MODULE_DICTIONARY_SUFFIX;
    write_tensor_parquet_named(
        &membership,
        out_prefix,
        membership_suffix,
        gene_names,
        "gene",
        &module_names,
    )?;
    // Already `[M, H]`: the dictionary is stored the way it is written.
    let dictionary = encoder
        .module_dictionary()
        .expect("a module encoder has a dictionary")
        .clone();
    write_tensor_parquet(
        &dictionary,
        out_prefix,
        dictionary_suffix,
        &module_names,
        "module",
        "H",
    )?;
    log::info!(
        "Wrote {} gene module(s) to {out_prefix}.{membership_suffix} and .{dictionary_suffix}",
        encoder.n_gene_modules(),
    );
    Ok(Some((membership_suffix, dictionary_suffix)))
}

#[cfg(test)]
#[path = "train_masked_tests.rs"]
mod train_masked_tests;
