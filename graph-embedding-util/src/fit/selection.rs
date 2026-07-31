//! The SELECTION PASS: sample *which* features load *which* dims, then install that
//! choice as a mask over the SGD that fits *how much*.
//!
//! ```text
//! --posterior   →  Gibbs samples the SELECTION (per-(feature,dim) `pip`)
//! then SGD      →  fits the LOADING under a mask drawn from that `pip`
//! ```
//!
//! # Why a learned gate cannot do this job
//!
//! A learned Bernoulli gate was built first, measured on BM1 (34,008 genes × 16 dims),
//! and failed for a reason no amount of tuning reaches: **12,620 of 34,008 genes (37%)
//! received exactly zero gradient**, all 16 dims frozen at their init, because the NCE
//! minibatches never drew them. (Two lesser problems came with it: the KL that has to
//! drive selection sat ~70× under the true ELBO, and any init where the gate is inert
//! puts every logit where `∂α/∂S = α(1−α)` is 1/14 of the available gradient.)
//!
//! The Gibbs column pass touches every anchor on every sweep, so it has no such blind
//! spot. That asymmetry is the whole argument for this design.

use crate::data::UnifiedData;
use crate::model::{GateKind, JointEmbedModel};
use crate::posterior::pb_gibbs::{PbGibbsResult, SpliceGibbsResult};
use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_nn::VarMap;
use log::info;

use super::config::{FeatFactorSpec, FitConfig};

/// Everything the selection pass reads. Bundled because the alternative is an
/// eight-parameter call, and because every field here is READ-ONLY to the pass — the
/// only things it writes are the pb Vars (by name, through `varmap`) and the returned
/// posteriors.
pub(crate) struct SelectionPassInput<'a> {
    pub config: &'a FitConfig,
    pub unified: &'a UnifiedData,
    pub varmap: &'a VarMap,
    pub collapsed_levels: &'a [data_beans_alg::collapse_data::CollapsedOut],
    pub cell_to_pb_per_level: &'a [Vec<usize>],
    /// The primary model's feature bias, the sampler's warm start for `b_a`.
    pub b_feat: &'a Tensor,
    /// The primary model's feature dictionary — the free-model warm start. Unused on
    /// the factored path, which rebuilds per-row loadings from `beta`/`delta` instead.
    pub e_feat: &'a Tensor,
}

/// Run the selection pass, or do nothing when `--posterior` is off.
///
/// Returns the free-model posterior; the factored (β-sharing) one goes out through
/// `splice_out`, because the two are different types and a caller that has one never
/// wants the other.
///
/// Only the **pseudobulk** half is written back. The feature half is deliberately not:
/// `beta_mean` is `E[z·β]`, already gated by the sampler's own selection, so installing
/// it as the SGD starting point and then applying the mask on top would gate twice.
/// Under this design the posterior contributes the selection and SGD fits the loading
/// from its own initialization.
pub(crate) fn run_selection_pass(
    input: &SelectionPassInput<'_>,
    use_cell_axis: bool,
    num_levels: usize,
    h: usize,
    splice_out: &mut Option<SpliceGibbsResult>,
) -> anyhow::Result<Option<PbGibbsResult>> {
    let SelectionPassInput {
        config,
        unified,
        varmap,
        collapsed_levels,
        cell_to_pb_per_level,
        b_feat,
        e_feat,
    } = *input;
    let Some(pcfg) = config.pb_posterior.as_ref() else {
        return Ok(None);
    };
    check_selection_is_applicable(config, use_cell_axis)?;
    info!(
        "SELECTION PASS — {} sweep(s) of Gibbs over the pseudobulk model, choosing which \
         features load which dims. It does NOT need to converge: it produces a `pip` that \
         phase-1 SGD then fits the loading under. --epochs still governs that SGD.",
        pcfg.n_sweeps,
    );
    match pcfg.stick_alpha {
        Some(a) => info!(
            "  inclusion prior: truncated IBP, stick-breaking at α={a} — per-dim rates \
             are forced to DECREASE with the dim index, so surplus dims are squeezed \
             off rather than each re-estimating the same rate. α is the expected dims \
             per feature and does NOT scale with --embedding-dim, which is therefore a \
             truncation rather than a knob."
        ),
        None => info!(
            "  inclusion prior: independent Beta per dim (--no-stick-breaking). With \
             many features per dim this is swamped by the data — it neither imposes \
             sparsity nor collapses unused dims. Drop the flag for the default IBP."
        ),
    }
    let pb = super::stacked_pb_view(varmap, collapsed_levels, cell_to_pb_per_level, h)?;
    let b_feat_map: Vec<f32> = b_feat.to_vec1()?;

    // gem's β-sharing side reads `beta`/`delta` per GENE; a free model's `e_feat` IS
    // the per-row table. Either way the sampler sees one row-indexed warm start.
    let e_feat_map = match config.feat_factor.as_ref() {
        Some(spec) => per_row_loadings(varmap, spec, unified.n_features(), h)?,
        None => e_feat.flatten_all()?.to_vec1()?,
    };
    let feat = crate::posterior::pb_index::FeatureSide {
        e_feat: &e_feat_map,
        b_feat: &b_feat_map,
        feature_to_backend_row: &unified.feature_to_backend_row,
    };

    match config.feat_factor.as_ref() {
        // One `β_g` per gene plus `δ_g` on the unspliced rows: TWO gates over a
        // row→gene grouping.
        Some(spec) => {
            let tracks = crate::posterior::pb_gibbs::SpliceTracks {
                row_to_gene: &spec.row_to_gene,
                unspliced_rows: &spec.unspliced_rows,
                n_genes: spec
                    .row_to_gene
                    .iter()
                    .copied()
                    .max()
                    .map_or(0, |m| m as usize + 1),
                nested: config.pb_posterior_nested_delta,
            };
            let res = crate::posterior::pb_gibbs::pb_gibbs_splice(&pb, &feat, &tracks, h, pcfg)?;
            write_back_pb_levels(varmap, &res.mean_pb, &res.mean_b_pb, res.h, num_levels)?;
            *splice_out = Some(res);
            Ok(None)
        }
        // Free model ⇒ a feature row IS an anchor, so no grouping.
        None => {
            let res = crate::posterior::pb_gibbs::pb_gibbs(&pb, &feat, None, h, pcfg)?;
            write_back_pb_levels(varmap, &res.mean_pb, &res.mean_b_pb, res.h, num_levels)?;
            Ok(Some(res))
        }
    }
}

/// Constraints on `--posterior`, enforced rather than discovered. None is reachable by
/// accident — `phase1_cells_per_pb` defaults to 0 and the lineage refine is opt-in —
/// but silently leaving a cell axis at its random init, silently skipping a refine the
/// caller asked for, or silently reporting a posterior for a model nobody asked for
/// would all be worse than stopping.
fn check_selection_is_applicable(config: &FitConfig, use_cell_axis: bool) -> anyhow::Result<()> {
    anyhow::ensure!(
        !use_cell_axis,
        "--posterior samples the PSEUDOBULK phase, and there is no cell block to \
         sample: --phase1-cells-per-pb {} would add a cell axis the selection pass \
         cannot see. Pass --phase1-cells-per-pb 0 (the default) to sample, or drop \
         --posterior.",
        config.phase1_cells_per_pb
    );
    anyhow::ensure!(
        !(config.lineage_dag && config.feat_factor.is_some()),
        "--lineage-dag orients its DAG from a trained velocity readout and then spends \
         half the epoch budget refining against it, which leaves the selection pass's \
         mask governing a fit it was never measured on. Run one or the other."
    );
    // The sampler's likelihood is the profiled Poisson, whose normalizer
    // `T_a · ln Σ exp(s)` IS the sampled-softmax estimand: dividing it by `T_a` gives
    // `E_{o~p̂}[s_o] − ln Σ exp(s_o)`, which is InfoNCE for the anchor with its positives
    // weighted by count. Against `NceObjective::Logistic` that identity simply does not
    // hold — SGNS is a sum of independent per-pair decisions, `Σ log σ(s) + Σ log σ(−s)`,
    // with no `logsumexp` anywhere. Sampling a logistic fit with this likelihood would
    // select features for a different model, silently, with plausible-looking tables.
    //
    // Sampling the logistic objective is a real option and the design is settled — it
    // stays `affine_in_anchor`, so the rank-1 column update still applies, and its
    // intercept has no closed-form profile but does have a monotone score equation
    // solvable per sweep by bisection, exactly the Bayes-EM step the Poisson path already
    // takes. It is simply not implemented, and an error is the honest way to say that.
    anyhow::ensure!(
        config.nce_objective != crate::loss::NceObjective::Logistic,
        "--posterior samples the profiled-Poisson likelihood, which is the same estimand \
         as --nce-objective softmax but NOT as logistic (SGNS is a sum of per-pair \
         decisions, with no logsumexp). Use --nce-objective softmax with --posterior, or \
         drop --posterior to train with logistic."
    );
    Ok(())
}

/// Install the sampled selection on every model that trains against it, and point them
/// all at ONE shared per-epoch draw. No-op when the selection pass did not run.
///
/// The sharing is not incidental. `train_composite` redraws `z` once per epoch on
/// `axes[0]` only; every other axis reads the draw through the shared cell. Minting a
/// fresh `Arc` per model instead — which is what `set_gate_pip` does on its own — leaves
/// each axis on its own mask and, for the δ gate, leaves all but one axis gating by the
/// mean while the axes train against different sub-models within a single epoch.
pub(crate) fn install_selection(
    cell_model: &mut JointEmbedModel,
    level_models: &mut [JointEmbedModel],
    pb: Option<&PbGibbsResult>,
    splice: Option<&SpliceGibbsResult>,
    n_features: usize,
    h: usize,
    dev: &Device,
) -> anyhow::Result<()> {
    let (pip, rows) = match (pb, splice) {
        // Free model: an anchor IS a feature row, so `pip` is already row-indexed.
        (Some(res), _) => (res.pip.as_slice(), n_features),
        // Factored: `beta_pip` is per GENE, the axis `s_beta` lives on —
        // `gathered_gate_weights` gathers rows from it via `row_to_gene`.
        (_, Some(res)) => (res.beta_pip.as_slice(), res.beta_pip.len() / h.max(1)),
        (None, None) => return Ok(()),
    };
    let n_off = pip.iter().filter(|&&p| p <= 0.0).count();
    info!(
        "Selection installed — pip over {rows} unit(s) × {h} dim(s), mean {:.3}, \
         {n_off} entrie(s) at exactly 0 (permanently masked: their loading never trains \
         and the dictionary carries exact zeros). z ~ Bern(pip) is redrawn ONCE PER \
         EPOCH, not per minibatch — z is a latent for the dataset, so a per-batch draw \
         would model each minibatch as having its own inclusion state.",
        pip.iter().sum::<f32>() / pip.len().max(1) as f32,
    );

    // gem's SECOND gate gets its OWN table. Masking δ with β's inclusion would tie a
    // gene's motion to its identity selection — the conflation `GateKind` exists to
    // prevent; measured means differ ~9× on rep2_wt (0.014 vs 0.125), so it is a silent
    // ~9× error rather than a rounding one.
    //
    // A gene whose δ is UNIDENTIFIED (it is missing one splice track) is masked off
    // outright. `delta_identified` is the authority for that, not a NaN scan of
    // `delta_pip`: the sampler leaves an unidentified gene's `delta_pip` at its prior
    // rate — finite, plausible, and meaningless — and masks it only at parquet-write
    // time. A predecessor scanned for NaN here and so was a no-op, gating every
    // unidentified δ at its prior rate.
    let dpip: Option<Vec<f32>> = splice.map(|res| {
        let mut v = res.delta_pip.clone();
        for (g, p) in v.chunks_mut(h.max(1)).enumerate() {
            let ok = res.delta_identified.get(g).copied().unwrap_or(false);
            for x in p {
                if !ok || !x.is_finite() {
                    *x = 0.0;
                }
            }
        }
        v
    });
    if let (Some(res), Some(d)) = (splice, dpip.as_ref()) {
        let n_unident = res.delta_identified.iter().filter(|ok| !**ok).count();
        info!(
            "Velocity selection installed — a SEPARATE δ pip over {rows} gene(s), mean \
             {:.3}; {n_unident} gene(s) unidentified and masked off entirely.",
            d.iter().sum::<f32>() / d.len().max(1) as f32,
        );
    }

    // Upload each table ONCE and hand every model a storage-sharing clone. The slice
    // form re-uploads per call, and at `[34k genes, 128 dims]` that is 17 MB per model
    // per gate, paid again for every collapse level.
    let upload = |v: &[f32]| -> anyhow::Result<Tensor> {
        Ok(Tensor::from_slice(v, (rows, h), &Device::Cpu)?.to_device(dev)?)
    };
    let pip_t = upload(pip)?;
    let dpip_t = dpip.as_deref().map(upload).transpose()?;

    cell_model.install_gate_pip(GateKind::Identity, &pip_t)?;
    if let Some(d) = dpip_t.as_ref() {
        cell_model.install_gate_pip(GateKind::Velocity, d)?;
    }
    let (identity_cell, velocity_cell) =
        (cell_model.gate_mask_cell(), cell_model.velocity_mask_cell());
    for m in level_models {
        m.install_gate_pip(GateKind::Identity, &pip_t)?;
        if let Some(d) = dpip_t.as_ref() {
            m.install_gate_pip(GateKind::Velocity, d)?;
        }
        m.share_gate_masks(&identity_cell, velocity_cell.as_ref());
    }
    Ok(())
}

/// The per-row loading table for a β-sharing model, read from the Vars: `β_g` on a
/// spliced row, `β_g + δ_g` on an unspliced one.
///
/// This exists because `cell_model.e_feat` is NOT the fit on this path — the factored
/// loss trains `beta`/`delta`, and `e_feat` stays at its random initialisation until
/// `materialize_e_feat`, which runs after phase 1. Reading it here would warm-start the
/// sampler from a snapshot of noise instead of from the actual parameterization.
///
/// The selection pass runs before phase-1 SGD, so at that point these Vars hold their
/// inits: randn `β`, zero `δ`. That is deliberate — the pass is cold by design, since
/// a `pip` conditioned on an SGD optimum would describe curvature around that optimum
/// rather than a posterior. The table still matters, because it fixes the SCALE and the
/// row→gene expansion the sampler starts from.
fn per_row_loadings(
    varmap: &VarMap,
    spec: &FeatFactorSpec,
    n_features: usize,
    h: usize,
) -> anyhow::Result<Vec<f32>> {
    let vars = varmap.data().lock().expect("varmap poisoned");
    let get = |name: &str| -> anyhow::Result<Option<Vec<f32>>> {
        match vars.get(name) {
            None => Ok(None),
            Some(v) => Ok(Some(v.as_tensor().flatten_all()?.to_vec1::<f32>()?)),
        }
    };
    let beta = get("beta")?
        .ok_or_else(|| anyhow::anyhow!("factored model has no `beta` var to warm-start from"))?;
    let delta = get("delta")?;
    let mut out = vec![0f32; n_features * h];
    for (uid, (&g, &unspliced)) in spec
        .row_to_gene
        .iter()
        .zip(&spec.unspliced_rows)
        .enumerate()
    {
        if g == u32::MAX || uid >= n_features {
            continue;
        }
        let (src, dst) = (g as usize * h, uid * h);
        for k in 0..h {
            let d = if unspliced {
                delta.as_ref().map_or(0.0, |v| v[src + k])
            } else {
                0.0
            };
            out[dst + k] = beta[src + k] + d;
        }
    }
    Ok(out)
}

/// Slice the stacked pb posterior mean back into each level's own Var.
///
/// Written through the `VarMap` by name rather than through `level_models[l].e_cell`:
/// the latter is a `Tensor` aliasing the `Var`'s storage, and whether it observes an
/// in-place `Var::set` is a candle implementation detail — the same reason
/// [`super::stacked_pb_view`] reads by name.
///
/// The stacked axis is level-ordered, so consuming it exactly is also the check that
/// the level sizes still agree.
fn write_back_pb_levels(
    varmap: &VarMap,
    mean_pb: &[f32],
    mean_b_pb: &[f32],
    h: usize,
    num_levels: usize,
) -> anyhow::Result<()> {
    let vars = varmap.data().lock().expect("varmap poisoned");
    let mut off = 0usize;
    for level in 0..num_levels {
        let name = format!("pb_l{level}_e_cell");
        let Some(var) = vars.get(&name) else { continue };
        let n_pb = var.elem_count() / h.max(1);
        let end = off + n_pb * h;
        anyhow::ensure!(
            end <= mean_pb.len(),
            "pb write-back for {name}: stacked axis holds {} values, level needs {end}",
            mean_pb.len()
        );
        let t = Tensor::from_slice(&mean_pb[off..end], (n_pb, h), &Device::Cpu)?
            .to_device(var.device())?;
        var.set(&t)?;
        // The paired bias moves with the loading: the profile likelihood maximised it
        // out, so they are only identified together.
        let bias_name = format!("pb_l{level}_b_cell");
        if let Some(bvar) = vars.get(&bias_name) {
            let (bs, be) = (off / h, end / h);
            anyhow::ensure!(
                bvar.elem_count() == be - bs,
                "pb bias write-back for {bias_name}: var holds {}, level needs {}",
                bvar.elem_count(),
                be - bs
            );
            let bt = Tensor::from_slice(&mean_b_pb[bs..be], be - bs, &Device::Cpu)?
                .to_device(bvar.device())?;
            bvar.set(&bt)?;
        }
        off = end;
    }
    anyhow::ensure!(
        off == mean_pb.len(),
        "pb write-back consumed {off} of {} stacked values — level sizes disagree",
        mean_pb.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests;
