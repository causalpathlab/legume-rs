//! Continuing a masked model onto a gene axis the source run did not have.
//!
//! A checkpoint's gene-keyed state is small: the fine-to-module map of each
//! coarsening level and the per-gene embedding ρ. Everything else the decoders
//! hold is module- or topic-keyed, so a cohort that measures genes the source
//! run never saw can still be absorbed if those two are carried by NAME rather
//! than refused by position. The pieces live where their mechanism does:
//!
//! - the alignment of this run's names onto the source run's is the same
//!   matcher `predict` aligns a query with ([`remap_to_source`]);
//! - a coarsening level is grown by `FeatureCoarsening::grow_by_profile`, each
//!   unknown gene joining the inherited module whose known members its
//!   pseudobulk profile most resembles (see `inherit_level_coarsenings`);
//! - the checkpoint's gene-keyed tensors are gathered onto the new order by
//!   `candle_util::grow`, which starts an unseen gene at the checkpoint's mean;
//! - ρ alone knows more than that, so [`refine_rho_by_module`] moves an unseen
//!   gene's row from the global mean to the mean of its module's known members.

use crate::embed_common::Mat;
use crate::topic::eval::{GeneRemap, QueryNameOpts};
use data_beans_alg::feature_coarsening::FeatureCoarsening;

/// Name of the per-gene embedding ρ in a masked checkpoint.
pub(crate) const RHO_TENSOR: &str = "enc.feature.embeddings";

/// This run's genes aligned onto an `--init-from` source run's, or `None` when the
/// two axes are identical and the exact warm start applies.
///
/// The source run's axis is the row order of its `feature_mean.parquet`, which is
/// the order every gene-keyed artifact of that run shares. `opts` carries the
/// run's own `--feature-name-kind`, so the source run's names are read under
/// the same rule the loader aligned this run's under.
pub(crate) fn remap_to_source(
    source: &str,
    new_genes: &[Box<str>],
    opts: &QueryNameOpts,
) -> anyhow::Result<Option<GeneRemap>> {
    let (source_genes, _) = crate::topic::model_metadata::load_feature_mean(source)?;
    let remap = crate::topic::eval::build_gene_remap_with(&source_genes, new_genes, opts);
    if remap.is_identity() {
        return Ok(None);
    }
    log::info!(
        "--init-from {source}: this run's gene axis is not the source run's ({} genes here, {} \
         there, {} in common); continuing by name",
        new_genes.len(),
        remap.d_train,
        remap.n_mapped,
    );
    Ok(Some(remap))
}

/// After the checkpoint is loaded on the new axis, restart each unseen gene's
/// ρ row at the mean of its module's known members instead of the global mean
/// it was given, so it enters the fit inside its module's neighbourhood.
pub(crate) fn refine_rho_by_module(
    parameters: &candle_util::candle_nn::VarMap,
    remap: &GeneRemap,
    modules: &FeatureCoarsening,
) -> anyhow::Result<()> {
    use matrix_util::traits::ConvertMatOps;
    let var = parameters
        .data()
        .lock()
        .expect("VarMap lock")
        .get(RHO_TENSOR)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("warm-start: no tensor named `{RHO_TENSOR}`"))?;
    let mut rho = Mat::from_tensor(&var.as_tensor().to_device(&candle_util::candle_core::Device::Cpu)?)?;
    let known: Vec<bool> = remap.new_to_train.iter().map(Option::is_some).collect();
    fill_rows_by_module(&mut rho, &known, modules)?;
    candle_util::frozen_features::overwrite_var_2d(parameters, RHO_TENSOR, &rho, var.device())?;
    Ok(())
}

/// Overwrite every unknown row of `rho` with the mean of the known rows in its
/// module. Pure so it can be checked without a checkpoint.
pub(crate) fn fill_rows_by_module(
    rho: &mut Mat,
    known: &[bool],
    modules: &FeatureCoarsening,
) -> anyhow::Result<()> {
    let d = rho.nrows();
    anyhow::ensure!(
        known.len() == d && modules.fine_to_coarse.len() == d,
        "gene axis growth: ρ has {d} rows, {} known flags, modules cover {}",
        known.len(),
        modules.fine_to_coarse.len(),
    );
    // Sum the known rows per module: zero the unknown ones so one aggregation
    // over the whole matrix counts only known members.
    let mut known_rows = rho.clone();
    let mut members = vec![0usize; modules.num_coarse];
    for (g, &k) in known.iter().enumerate() {
        if k {
            members[modules.fine_to_coarse[g]] += 1;
        } else {
            known_rows.row_mut(g).fill(0.0);
        }
    }
    let mut mean = modules.aggregate_rows_ds(&known_rows);
    for (m, &n) in members.iter().enumerate() {
        if n > 0 {
            mean.row_mut(m).scale_mut(1.0 / n as f32);
        }
    }
    for (g, &k) in known.iter().enumerate() {
        if k {
            continue;
        }
        let m = modules.fine_to_coarse[g];
        anyhow::ensure!(
            members[m] > 0,
            "gene axis growth: gene {g} was placed in module {m}, which has no surviving member \
             to start its embedding from"
        );
        rho.row_mut(g).copy_from(&mean.row(m));
    }
    Ok(())
}

#[cfg(test)]
#[path = "gene_axis_tests.rs"]
mod gene_axis_tests;
