//! Data-driven anchor-based β prior for topic models.
//!
//! Finds archetypal pseudobulks ("anchors") via a greedy Gram-Schmidt /
//! Arora-style vertex-selection pass on the finest collapsed level, then
//! converts those anchor PBs into a dense `[D_full, K]` simplex prior used
//! both for β initialization and as an optional training-time cross-entropy
//! penalty.
//!
//! **Coupling constraint**: the module assumes softmax-based decoders
//! register their pre-softmax logits under the `VarMap` path
//! `dec_{level}.dictionary.logits` — the convention used by
//! `candle_util::candle_aux_linear::log_softmax_linear`. vMF decoders use a
//! different path and are silently skipped by `init_decoder_dictionary` /
//! `anchor_penalty_at_level`.

use crate::anchor_common::{
    base_celltype_label, gram_schmidt_anchors, label_anchors, softmax_col_into, zscore_columns,
};
use crate::embed_common::*;
use crate::marker_support::MarkerInfo;

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use data_beans_alg::collapse_data::CollapsedOut;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use matrix_util::traits::ConvertMatOps;
use std::io::Write;

/// Suffix on the per-level `VarMap` path where softmax-based decoders store
/// their `[K, D]` pre-softmax logits.
const DICT_LOGITS_VAR_SUFFIX: &str = "dictionary.logits";
/// Full `VarMap` path for decoder level `i`'s logit tensor.
fn decoder_logits_var_path(level: usize) -> String {
    format!("dec_{level}.{DICT_LOGITS_VAR_SUFFIX}")
}

/// Everything the anchor pipeline produces in one pass.
pub(crate) struct AnchorPrior {
    /// `[D_full, K]`, each column on the gene simplex.
    pub anchor_weight_gk: Mat,
    /// Length-K display labels (celltype name or `novel_{i}`).
    pub topic_labels: Vec<Box<str>>,
    /// Length-K indices into the finest pseudobulk level.
    pub anchor_pb_idx: Vec<usize>,
    /// `(top1_z, top2_z)` marker-fit scores per anchor. All zero when
    /// markers are unavailable.
    pub margin_scores: Vec<(f32, f32)>,
}

impl AnchorPrior {
    /// Build the prior from the finest pseudobulk level.
    ///
    /// `n_topics` is the requested K (equal to the number of anchors). If
    /// `markers` is `Some`, anchors get labeled via a margin-based rule
    /// (`top1 - top2 >= margin_threshold`); otherwise all labels are
    /// `novel_{i}` and `margin_scores` stays zero.
    ///
    /// `finest_coarsening` — when `Some`, anchor *selection* runs in the
    /// coarsened feature space that the encoder and finest decoder
    /// actually see. The stored `anchor_weight_gk` is still at `D_full` so
    /// every level's own coarsening can aggregate it independently. This
    /// matters when `--max-coarse-features` groups many fine features: if
    /// we selected anchors at `D_full`, rare-gene variation could pick
    /// rows that collapse into identical coarse vectors once the model
    /// sees them, and the prior would no longer match the training
    /// geometry. The marker-labeling z-scores, by contrast, stay at
    /// `D_full` because the marker file names individual genes.
    pub(crate) fn from_pseudobulk(
        finest: &CollapsedOut,
        n_topics: usize,
        markers: Option<&MarkerInfo>,
        margin_threshold: f32,
        finest_coarsening: Option<&FeatureCoarsening>,
    ) -> anyhow::Result<Self> {
        // log1p on [D_full, n_pb] in the posterior's native orientation —
        // no transpose needed. `aggregate_rows_ds` operates on [D, S] so we
        // feed it directly, and softmax columns of `anchor_weight_gk` are
        // read from `x_gp` column views (PBs live in columns).
        let mu_gp: &Mat = crate::topic::common::preferred_posterior_mean(finest);
        let n_pb = mu_gp.ncols();
        let d_full = mu_gp.nrows();
        if n_pb < 2 {
            return Err(anyhow::anyhow!(
                "anchor prior needs ≥2 pseudobulks, got {n_pb}"
            ));
        }

        let mut x_gp = mu_gp.clone();
        for v in x_gp.as_mut_slice() {
            *v = v.max(0.0).ln_1p();
        }

        // Selection-space view: aggregate into the encoder's coarsened
        // feature space when a finest-level coarsening is active, otherwise
        // use the full-resolution matrix. Either way, transpose once to get
        // [n_pb, D_selection] rows for Gram-Schmidt.
        let x_pd_selection: Mat = match finest_coarsening {
            Some(fc) => fc.aggregate_rows_ds(&x_gp).transpose(),
            None => x_gp.transpose(),
        };

        // Z-score per feature so Gram-Schmidt residuals aren't dominated
        // by high-variance genes.
        let x_sel_zscored = zscore_columns(&x_pd_selection);

        // Greedy Gram-Schmidt vertex selection in the selection space.
        let k = n_topics.min(n_pb);
        let anchor_pb_idx = gram_schmidt_anchors(&x_sel_zscored, k);

        // [D_full, K] prior: softmax of each anchor PB's log1p expression
        // at full resolution. `x_gp.column(pb)` gives a PB's gene vector
        // without an extra transpose.
        let mut anchor_weight_gk = Mat::zeros(d_full, k);
        for (col, &pb) in anchor_pb_idx.iter().enumerate() {
            softmax_col_into(x_gp.column(pb), anchor_weight_gk.column_mut(col));
        }

        // Marker labeling uses D_full z-scores because the marker file
        // names individual genes; a coarse bin's z-score would average
        // many genes and lose the signal.
        let (topic_labels, margin_scores) = match markers {
            Some(m) => {
                // Transpose once so zscore_columns can standardize per gene
                // across PBs. Only done when a marker file is actually
                // provided — the no-marker path skips this cost entirely.
                let x_pg = x_gp.transpose();
                let x_full_zscored = zscore_columns(&x_pg);
                label_anchors(&x_full_zscored, &anchor_pb_idx, m, margin_threshold)
            }
            None => (
                (0..k)
                    .map(|i| format!("novel_{i}").into_boxed_str())
                    .collect(),
                vec![(0.0, 0.0); k],
            ),
        };

        Ok(Self {
            anchor_weight_gk,
            topic_labels,
            anchor_pb_idx,
            margin_scores,
        })
    }

    /// Per-level `[K, D_l]` anchor tensors pre-transposed for direct use as
    /// cross-entropy targets against the decoder's `[K, D_l]` log β. Built
    /// once at the start of training; eliminates the per-minibatch
    /// `transpose().contiguous()` that the penalty helper would otherwise
    /// need.
    pub(crate) fn per_level_device_tensors(
        &self,
        level_coarsenings: &[Option<FeatureCoarsening>],
        dev: &Device,
    ) -> anyhow::Result<Vec<Tensor>> {
        level_coarsenings
            .iter()
            .map(|fc| {
                // `coarsened_weight` returns [D_l, K]; transpose to [K, D_l]
                // so the penalty helper can multiply element-wise with
                // `log β [K, D_l]` without any per-step transpose.
                let w_dk = self.coarsened_weight(fc.as_ref());
                let w_kd = w_dk.transpose();
                w_kd.to_tensor(dev)
            })
            .collect()
    }

    /// `[D_level, K]` view of the prior, aggregating fine features into the
    /// coarse groups defined by `fc` and renormalizing each column on the
    /// simplex. `None` means the caller wants the full-resolution prior.
    pub(crate) fn coarsened_weight(&self, fc: Option<&FeatureCoarsening>) -> Mat {
        let mut w = match fc {
            Some(fc) => fc.aggregate_rows_ds(&self.anchor_weight_gk),
            None => self.anchor_weight_gk.clone(),
        };
        // Renormalize each column — `aggregate_rows_ds` sums fine-feature
        // mass into coarse bins, but rounding drift can leave columns
        // slightly off-simplex. This also guards against empty groups.
        for mut col in w.column_iter_mut() {
            let s: f32 = col.iter().sum();
            if s > 1e-12 {
                col /= s;
            }
        }
        w
    }

    /// Write per-anchor label + score TSV and, when markers are given,
    /// the candidate-marker expansion table.
    pub(crate) fn write_side_outputs(
        &self,
        out_prefix: &str,
        gene_names: &[Box<str>],
        markers: Option<&MarkerInfo>,
    ) -> anyhow::Result<()> {
        let labels_path = format!("{out_prefix}.anchor_labels.tsv");
        let mut f = std::fs::File::create(&labels_path)?;
        writeln!(f, "topic_idx\tpb_idx\tlabel\ttop1_z\ttop2_z\tmargin")?;
        for (i, ((&pb, label), (t1, t2))) in self
            .anchor_pb_idx
            .iter()
            .zip(self.topic_labels.iter())
            .zip(self.margin_scores.iter())
            .enumerate()
        {
            writeln!(
                f,
                "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}",
                i,
                pb,
                label,
                t1,
                t2,
                t1 - t2
            )?;
        }
        log::info!("wrote {labels_path}");

        if let Some(m) = markers {
            let expansion_path = format!("{out_prefix}.marker_expansion.tsv");
            let mut f = std::fs::File::create(&expansion_path)?;
            writeln!(f, "celltype\tgene\tanchor_z\tin_user_list")?;
            let top_n = 50usize;
            let d = gene_names.len();
            for (k, label) in self.topic_labels.iter().enumerate() {
                // Find which celltype this label corresponds to (might be
                // novel_*; skip). Multiple-anchor labels have a numeric
                // suffix (T_cells_2) — strip it to recover the base name.
                let base = base_celltype_label(label);
                let Some(ct_idx) = m.celltypes.iter().position(|c| c.as_ref() == base) else {
                    continue;
                };
                let mut ranked: Vec<(usize, f32)> =
                    (0..d).map(|g| (g, self.anchor_weight_gk[(g, k)])).collect();
                ranked.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                for (g, z) in ranked.into_iter().take(top_n) {
                    let in_user = m.membership_gc[(g, ct_idx)] > 0.0;
                    writeln!(
                        f,
                        "{}\t{}\t{:.4}\t{}",
                        label,
                        gene_names[g],
                        z,
                        if in_user { "yes" } else { "no" }
                    )?;
                }
            }
            log::info!("wrote {expansion_path}");
        }
        Ok(())
    }
}

/// Apply the β prior cross-entropy penalty for one decoder level to an
/// existing loss. No-op when the prior isn't attached, when λ ≤ 0, or when
/// the level's decoder doesn't register its dictionary under
/// `dec_{level}.dictionary.logits` (e.g. the vMF decoder).
///
/// **Batch-size semantics**: the penalty is a fixed scalar per minibatch
/// step — it does not depend on the minibatch's sample count. The main
/// VAE loss uses `mean_all()` over the minibatch, so both terms are
/// dimensionally "per-sample-averaged" and their ratio is batch-size
/// invariant within a step. Over an epoch, however, the penalty is
/// applied once per minibatch, so its cumulative gradient contribution
/// scales with the number of minibatches (M = `N_total` / `N_batch`). If you
/// change `--minibatch-size`, you will typically want to rescale
/// `--anchor-penalty` in inverse proportion (or rely on Adam's adaptive
/// step size plus linear-LR scaling to absorb the difference).
/// Cross-entropy penalty `−λ · mean(Σ · log softmax(logits))` for a
/// named Var at level `level`. Shared between anchor β and ambient α:
/// both are `−Σ prior · log softmax(var_logits)` against a simplex prior.
///
/// **Batch-size semantics** (same for anchor and ambient): the penalty is
/// a fixed scalar per minibatch step — it does not depend on the sample
/// count. The main VAE loss uses `mean_all()` over the minibatch, so both
/// terms are per-sample-averaged within a step and their ratio is
/// batch-size invariant. Over an epoch the penalty is applied once per
/// minibatch, so its cumulative gradient contribution scales with
/// `M = N_total / N_batch`; rescale `--anchor-penalty` /
/// `--ambient-penalty` inversely with `--minibatch-size` if you change it
/// (or rely on Adam's adaptive step to absorb the difference).
fn ce_penalty_at_level(
    loss: Tensor,
    parameters: &VarMap,
    priors_per_level: Option<&[Tensor]>,
    lambda: f32,
    level: usize,
    var_path: &str,
    reduce_dim: usize,
) -> anyhow::Result<Tensor> {
    let Some(priors) = priors_per_level else {
        return Ok(loss);
    };
    if lambda <= 0.0 {
        return Ok(loss);
    }
    // Clone the logits Tensor handle under the lock — a cheap Arc bump
    // that preserves TensorId / is_variable so gradients still flow.
    let logits = {
        let data = parameters.data().lock().expect("VarMap lock");
        match data.get(var_path) {
            Some(var) => var.as_tensor().clone(),
            None => return Ok(loss),
        }
    };
    let prior = &priors[level];
    let log_prob = candle_nn::ops::log_softmax(&logits, logits.rank() - 1)?;
    let ce = (prior * &log_prob)?.sum(reduce_dim)?.neg()?;
    let pen = (ce.mean_all()? * f64::from(lambda))?;
    Ok((loss + pen)?)
}

/// Apply the β prior cross-entropy penalty for one decoder level to an
/// existing loss. No-op when the prior isn't attached, when λ ≤ 0, or when
/// the level's decoder doesn't register its dictionary under
/// `dec_{level}.dictionary.logits` (e.g. the vMF decoder).
pub(crate) fn anchor_penalty_at_level(
    loss: Tensor,
    parameters: &VarMap,
    anchor_prior_per_level: Option<&[Tensor]>,
    lambda: f32,
    level: usize,
) -> anyhow::Result<Tensor> {
    // [K, D_l] anchor vs [K, D_l] log-β → sum over D then mean over K.
    ce_penalty_at_level(
        loss,
        parameters,
        anchor_prior_per_level,
        lambda,
        level,
        &decoder_logits_var_path(level),
        1,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarsened_weight_is_simplex() {
        let k = 4;
        let d_full = 10;
        let mut ap = AnchorPrior {
            anchor_weight_gk: Mat::from_fn(d_full, k, |_, _| 1.0 / d_full as f32),
            topic_labels: (0..k)
                .map(|i| format!("topic_{i}").into_boxed_str())
                .collect(),
            anchor_pb_idx: (0..k).collect(),
            margin_scores: vec![(0.0, 0.0); k],
        };
        // Nudge one topic so it isn't perfectly uniform.
        ap.anchor_weight_gk[(0, 0)] *= 2.0;
        // Fake coarsening: merge features {0,1} → 0, {2,3,4} → 1, rest identity.
        let fc = FeatureCoarsening {
            fine_to_coarse: vec![0, 0, 1, 1, 1, 2, 3, 4, 5, 6],
            coarse_to_fine: vec![
                vec![0, 1],
                vec![2, 3, 4],
                vec![5],
                vec![6],
                vec![7],
                vec![8],
                vec![9],
            ],
            num_coarse: 7,
        };
        let w_coarse = ap.coarsened_weight(Some(&fc));
        assert_eq!(w_coarse.nrows(), 7);
        assert_eq!(w_coarse.ncols(), k);
        for kk in 0..k {
            let s: f32 = w_coarse.column(kk).iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "column {} sum {} ≠ 1", kk, s);
        }
    }
}
