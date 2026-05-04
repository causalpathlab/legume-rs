//! Decoder-specific post-training I/O for `senna topic` / `senna itopic`.
//!
//! Each topic-model decoder (NB, multinom, vMF, NB-mixture) needs its
//! own dictionary-write logic and may emit additional per-gene parameter
//! files (dispersion, ambient α, vMF κ, etc.). The [`DecoderExtras`]
//! trait gives every decoder a uniform two-method surface so the main
//! pipeline in `fit_topic.rs` doesn't have to fan out per-decoder
//! `match` arms when serializing trained models.

use crate::embed_common::*;
use candle_util::candle_decoder_nb_mixture::NbMixtureTopicDecoder;
use candle_util::candle_decoder_topic::{MultinomTopicDecoder, NbTopicDecoder};
use candle_util::candle_decoder_vmf_topic::VmfTopicDecoder;
use candle_util::candle_model_traits::*;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use matrix_util::traits::IoOps;

/// Decoder-specific post-training output (dictionary writing, extra parameters).
pub(crate) trait DecoderExtras {
    /// Write the dictionary to parquet.
    /// Default: log-prob space with optional coarsening expansion.
    fn write_dictionary(
        &self,
        coarsening: Option<&FeatureCoarsening>,
        n_features_full: usize,
        gene_names: &[Box<str>],
        out_prefix: &str,
    ) -> anyhow::Result<()>
    where
        Self: DecoderModuleT,
    {
        write_dictionary_expanded(self, coarsening, n_features_full, gene_names, out_prefix)
    }

    /// Write decoder-specific extra parameters (dispersion, kappa, etc.).
    /// No-op by default.
    fn write_extras(
        &self,
        _coarsening: Option<&FeatureCoarsening>,
        _n_features_full: usize,
        _gene_names: &[Box<str>],
        _out_prefix: &str,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

impl DecoderExtras for MultinomTopicDecoder {}

/// Expand a coarsened per-feature vector `[D_coarse]` back to `[n_features_full]`
/// by broadcasting each coarse value across its fine-feature group. For
/// simplex-valued vectors set `split_mass=true` so the expanded vector sums
/// to the same total as the coarse one; for scalar-per-gene values (φ, dispersions)
/// keep `split_mass=false`.
fn expand_coarsened_feature_vec(
    coarse_vec: Vec<f32>,
    coarsening: Option<&FeatureCoarsening>,
    n_features_full: usize,
    split_mass: bool,
) -> Vec<f32> {
    let Some(fc) = coarsening else {
        return coarse_vec;
    };
    let mut full = vec![0.0f32; n_features_full];
    for (c, group) in fc.coarse_to_fine.iter().enumerate() {
        let per_feat = if split_mass {
            coarse_vec[c] / group.len().max(1) as f32
        } else {
            coarse_vec[c]
        };
        for &f in group {
            full[f] = per_feat;
        }
    }
    full
}

/// Shared context when writing several per-gene parameter vectors from a
/// single decoder (reuses coarsening expansion + output prefix).
struct GeneParamWriteCtx<'a> {
    coarsening: Option<&'a FeatureCoarsening>,
    n_features_full: usize,
    gene_names: &'a [Box<str>],
    out_prefix: &'a str,
}

/// Write a `[D]` per-gene parameter vector as `{out}.{name}.parquet`.
fn write_per_gene_param(
    ctx: &GeneParamWriteCtx<'_>,
    values: Vec<f32>,
    file_suffix: &str,
    col_name: &str,
    split_mass: bool,
) -> anyhow::Result<()> {
    let expanded =
        expand_coarsened_feature_vec(values, ctx.coarsening, ctx.n_features_full, split_mass);
    let mat = Mat::from_column_slice(expanded.len(), 1, &expanded);
    let col = vec![col_name.to_string().into_boxed_str()];
    mat.to_parquet_with_names(
        &format!("{}.{}.parquet", ctx.out_prefix, file_suffix),
        (Some(ctx.gene_names), Some("gene")),
        Some(&col),
    )?;
    Ok(())
}

/// Pull `log_phi` from device, exp it, and flatten into a plain Vec<f32>.
fn phi_vec_from_tensor(log_phi: &candle_core::Tensor) -> anyhow::Result<Vec<f32>> {
    Ok(log_phi
        .to_device(&candle_core::Device::Cpu)?
        .exp()?
        .flatten_all()?
        .to_vec1()?)
}

impl DecoderExtras for NbTopicDecoder {
    fn write_extras(
        &self,
        coarsening: Option<&FeatureCoarsening>,
        n_features_full: usize,
        gene_names: &[Box<str>],
        out_prefix: &str,
    ) -> anyhow::Result<()> {
        let ctx = GeneParamWriteCtx {
            coarsening,
            n_features_full,
            gene_names,
            out_prefix,
        };
        let phi_vec = phi_vec_from_tensor(self.log_phi())?;
        write_per_gene_param(&ctx, phi_vec, "dispersion", "dispersion_phi", false)?;
        info!("Saved dispersion parameters to {out_prefix}.dispersion.parquet");
        Ok(())
    }
}

impl DecoderExtras for NbMixtureTopicDecoder {
    fn write_extras(
        &self,
        coarsening: Option<&FeatureCoarsening>,
        n_features_full: usize,
        gene_names: &[Box<str>],
        out_prefix: &str,
    ) -> anyhow::Result<()> {
        let cpu = candle_core::Device::Cpu;
        let ctx = GeneParamWriteCtx {
            coarsening,
            n_features_full,
            gene_names,
            out_prefix,
        };

        write_per_gene_param(
            &ctx,
            phi_vec_from_tensor(self.log_phi())?,
            "dispersion",
            "dispersion_phi",
            false,
        )?;

        let alpha_vec: Vec<f32> = self.alpha()?.to_device(&cpu)?.flatten_all()?.to_vec1()?;
        write_per_gene_param(&ctx, alpha_vec, "alpha", "ambient_alpha", true)?;

        let rho_a: f32 = self
            .rho_a()
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        let rho_b: f32 = self
            .rho_b()
            .to_device(&cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?[0];
        let rho_mat = Mat::from_column_slice(2, 1, &[rho_a, rho_b]);
        let row_names = vec![
            "rho_a".to_string().into_boxed_str(),
            "rho_b".to_string().into_boxed_str(),
        ];
        rho_mat.to_parquet_with_names(
            &(out_prefix.to_string() + ".rho.parquet"),
            (Some(&row_names), Some("param")),
            Some(&["value".to_string().into_boxed_str()]),
        )?;
        info!(
            "Saved ambient parameters: {out_prefix}.dispersion.parquet, {out_prefix}.alpha.parquet, {out_prefix}.rho.parquet (a={rho_a:.3}, b={rho_b:.3})"
        );
        Ok(())
    }
}

impl DecoderExtras for VmfTopicDecoder {
    /// vMF dictionary: expand coarse directions to full resolution and re-normalize.
    fn write_dictionary(
        &self,
        coarsening: Option<&FeatureCoarsening>,
        n_features_full: usize,
        gene_names: &[Box<str>],
        out_prefix: &str,
    ) -> anyhow::Result<()> {
        let dict_tensor = self
            .get_dictionary()?
            .to_device(&candle_core::Device::Cpu)?;
        let dict_dk: Mat = Mat::from_tensor(&dict_tensor)?;

        let out_dk = if let Some(fc) = coarsening {
            let k = dict_dk.ncols();
            let mut expanded = Mat::zeros(n_features_full, k);
            for (c, fine_indices) in fc.coarse_to_fine.iter().enumerate() {
                for &f in fine_indices {
                    for kk in 0..k {
                        expanded[(f, kk)] = dict_dk[(c, kk)];
                    }
                }
            }
            // Re-normalize each column to unit length
            for kk in 0..k {
                let col = expanded.column(kk);
                let norm = col.dot(&col).sqrt();
                if norm > 1e-12 {
                    expanded.column_mut(kk).scale_mut(1.0 / norm);
                }
            }
            expanded
        } else {
            dict_dk
        };

        out_dk.to_parquet_with_names(
            &(out_prefix.to_string() + ".dictionary.parquet"),
            (Some(gene_names), Some("gene")),
            Some(&axis_id_names("T", out_dk.ncols())),
        )?;
        Ok(())
    }

    fn write_extras(
        &self,
        _coarsening: Option<&FeatureCoarsening>,
        _n_features_full: usize,
        _gene_names: &[Box<str>],
        _out_prefix: &str,
    ) -> anyhow::Result<()> {
        let kappas = self.kappa_vec()?;
        let kappa_strs: Vec<String> = kappas.iter().map(|k| format!("{k:.2}")).collect();
        info!("vMF concentration κ = [{}]", kappa_strs.join(", "));
        Ok(())
    }
}

/// Write dictionary tensor with optional expansion from coarse to fine resolution.
pub(crate) fn write_dictionary_tensor(
    dict_tensor: &candle_core::Tensor,
    coarsening: Option<&FeatureCoarsening>,
    n_features_full: usize,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    let dict_tensor = dict_tensor.to_device(&candle_core::Device::Cpu)?;

    if let Some(fc) = coarsening {
        let dict_dk: Mat = Mat::from_tensor(&dict_tensor)?;
        let expanded_dk = fc.expand_log_dict_dk(&dict_dk, n_features_full);
        expanded_dk.to_parquet_with_names(
            &(out_prefix.to_string() + ".dictionary.parquet"),
            (Some(gene_names), Some("gene")),
            Some(&axis_id_names("T", expanded_dk.ncols())),
        )?;
        info!(
            "Expanded dictionary from {} to {} features",
            fc.num_coarse, n_features_full
        );
    } else {
        let k_topics = dict_tensor.dims().last().copied().unwrap_or(0);
        dict_tensor.to_parquet_with_names(
            &(out_prefix.to_string() + ".dictionary.parquet"),
            (Some(gene_names), Some("gene")),
            Some(&axis_id_names("T", k_topics)),
        )?;
    }
    Ok(())
}

/// Write dictionary from a decoder implementing `DecoderModuleT`.
pub(crate) fn write_dictionary_expanded<Dec: DecoderModuleT + ?Sized>(
    decoder: &Dec,
    coarsening: Option<&FeatureCoarsening>,
    n_features_full: usize,
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    let dict_tensor = decoder.get_dictionary()?;
    write_dictionary_tensor(
        &dict_tensor,
        coarsening,
        n_features_full,
        gene_names,
        out_prefix,
    )
}
