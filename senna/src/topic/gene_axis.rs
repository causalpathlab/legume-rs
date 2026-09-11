//! Continuing a model onto a gene axis the source run did not have.
//!
//! A checkpoint's gene-keyed state is small: the fine-to-coarse map of each
//! coarsening level, and whichever tensors are keyed by gene rather than by
//! coarse group, module or topic. A cohort that measures genes the source run never saw can
//! still be absorbed if those are carried by NAME rather than refused by
//! position. Every family with a checkpoint goes through here. The pieces live
//! where their mechanism does:
//!
//! - the alignment of this run's names onto the source run's is the same
//!   matcher `predict` aligns a query with ([`remap_for_init_from`]);
//! - a coarsening level is grown by `FeatureCoarsening::grow_by_profile`, each
//!   unknown gene joining the inherited coarse group whose known members its
//!   pseudobulk profile most resembles (see `inherit_level_coarsenings`);
//! - the checkpoint's gene-keyed tensors are gathered onto the new order by
//!   `candle_util::grow`, which starts an unseen gene at the checkpoint's mean;
//! - a free per-gene ρ knows more than that, so [`refine_rho_by_coarsening`]
//!   moves an unseen gene's row from the global mean to the mean of its coarse
//!   group's known members. A feature side composed from learned modules takes
//!   neither: its membership starts flat, the one start that leaves every
//!   module reachable (see `candle_util::grow`). The families with no per-gene
//!   embedding at all take only the gather.

use crate::embed_common::Mat;
use crate::topic::eval::{GeneRemap, QueryNameOpts};
use auxiliary_data::feature_names::FeatureNameKindArg;
use data_beans_alg::feature_coarsening::FeatureCoarsening;

/// Name of the per-gene embedding ρ in a masked checkpoint.
pub(crate) const RHO_TENSOR: &str = "enc.feature.embeddings";

/// This run's genes aligned onto an `--init-from` source run's: `None` when
/// there is no source run, or when the two axes are identical and the exact
/// warm start applies.
///
/// The source run's axis is the row order of its `feature_mean.parquet`, which
/// is the order every gene-keyed artifact of that run shares — and the
/// cheapest of them to read, which is why this does not go through
/// `predict::bulk::model_gene_names`: every family that can be a source here
/// writes that file, and the alternatives are the D×K dictionary and ρ. Its
/// names are read under `kind` — the run's own `--feature-name-kind` — so one
/// flag means one thing for the whole command rather than one rule for the
/// loader and another for the warm start.
pub(crate) fn remap_for_init_from(
    init_from: Option<&str>,
    kind: &FeatureNameKindArg,
    new_genes: &[Box<str>],
) -> anyhow::Result<Option<GeneRemap>> {
    let Some(source) = init_from else {
        return Ok(None);
    };
    let opts = QueryNameOpts {
        kind: kind.resolve_or_gene(),
        ..Default::default()
    };
    let (source_genes, _) = crate::topic::model_metadata::load_feature_mean(source)?;
    let remap = crate::topic::eval::build_gene_remap_with(&source_genes, new_genes, &opts);
    if remap.is_identity() {
        return Ok(None);
    }
    // Sharing nothing is not a continuation. Left through, every gene-keyed
    // tensor would be re-initialized from the checkpoint's mean and the run
    // would log "continuing by name" while being a retrain from scratch.
    crate::topic::eval::ensure_gene_coverage(&remap, 0.0, "--feature-name-kind")?;
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
/// ρ row at the mean of its coarse group's known members instead of the global
/// mean it was given, so it enters the fit inside that group's neighbourhood.
pub(crate) fn refine_rho_by_coarsening(
    parameters: &candle_util::candle_nn::VarMap,
    remap: &GeneRemap,
    coarsening: &FeatureCoarsening,
) -> anyhow::Result<()> {
    use matrix_util::traits::ConvertMatOps;
    let var = parameters
        .data()
        .lock()
        .expect("VarMap lock")
        .get(RHO_TENSOR)
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "gene axis growth: this run has a free per-gene embedding, but the model it \
                 just loaded registers no `{RHO_TENSOR}`. A feature side composed from learned \
                 modules has none either — it is refined by its own rule and never reaches \
                 here — so this is a model that did not load."
            )
        })?;
    let mut rho = Mat::from_tensor(
        &var.as_tensor()
            .to_device(&candle_util::candle_core::Device::Cpu)?,
    )?;
    let known: Vec<bool> = remap.new_to_train.iter().map(Option::is_some).collect();
    fill_rows_by_coarsening(&mut rho, &known, coarsening)?;
    candle_util::frozen_features::overwrite_var_2d(parameters, RHO_TENSOR, &rho, var.device())?;
    Ok(())
}

/// Overwrite every unknown row of `rho` with the mean of the known rows in its
/// coarse group. Pure so it can be checked without a checkpoint.
pub(crate) fn fill_rows_by_coarsening(
    rho: &mut Mat,
    known: &[bool],
    coarsening: &FeatureCoarsening,
) -> anyhow::Result<()> {
    let d = rho.nrows();
    anyhow::ensure!(
        known.len() == d && coarsening.fine_to_coarse.len() == d,
        "gene axis growth: ρ has {d} rows, {} known flags, the coarsening covers {}",
        known.len(),
        coarsening.fine_to_coarse.len(),
    );
    // Sum the known rows per coarse group: zero the unknown ones so one
    // aggregation over the whole matrix counts only known members.
    let mut known_rows = rho.clone();
    let mut members = vec![0usize; coarsening.num_coarse];
    for (g, &k) in known.iter().enumerate() {
        if k {
            members[coarsening.fine_to_coarse[g]] += 1;
        } else {
            known_rows.row_mut(g).fill(0.0);
        }
    }
    let mut mean = coarsening.aggregate_rows_ds(&known_rows);
    for (m, &n) in members.iter().enumerate() {
        if n > 0 {
            mean.row_mut(m).scale_mut(1.0 / n as f32);
        }
    }
    for (g, &k) in known.iter().enumerate() {
        if k {
            continue;
        }
        let m = coarsening.fine_to_coarse[g];
        anyhow::ensure!(
            members[m] > 0,
            "gene axis growth: gene {g} was placed in coarse group {m}, which has no surviving \
             member to start its embedding from"
        );
        rho.row_mut(g).copy_from(&mean.row(m));
    }
    Ok(())
}

#[cfg(test)]
#[path = "gene_axis_tests.rs"]
mod gene_axis_tests;
