//! Anchor-gene-based β prior for topic models.
//!
//! Finds anchor genes via Gram-Schmidt on the TF-IDF ratio simplex
//! (genes as points in PB-sample space), then recovers per-topic gene
//! profiles via Arora-style convex decomposition. Used for:
//!
//! 1. **β initialization** — decoder logits overwritten with recovered profiles.
//! 2. **Decoder-side cross-entropy penalty** — pulls learned β toward anchors.
//! 3. **Encoder-side anchor loss** — pulls encoder q(z|anchor_pb) toward topic k.

use crate::embed_common::*;
use crate::marker_support::MarkerInfo;

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use data_beans_alg::collapse_data::CollapsedOut;
use data_beans_alg::feature_coarsening::FeatureCoarsening;
use matrix_util::traits::ConvertMatOps;
use std::io::Write;

const DICT_LOGITS_VAR_SUFFIX: &str = "dictionary.logits";

fn decoder_logits_var_path(level: usize) -> String {
    format!("dec_{level}.{DICT_LOGITS_VAR_SUFFIX}")
}

/// Everything the anchor pipeline produces in one pass.
pub(crate) struct AnchorPrior {
    /// `[D_full, K]` logits for decoder init.
    pub anchor_weight_gk: Mat,
    pub topic_labels: Vec<Box<str>>,
    pub anchor_pb_idx: Vec<usize>,
    pub margin_scores: Vec<(f32, f32)>,
    /// `[K, D_enc]` anchor PB expression for encoder-side anchor loss.
    pub anchor_pb_enc_kd: Mat,
    /// Per-topic vertex gene index (into the full gene space).
    pub vertex_genes: Vec<usize>,
    /// Per-topic anchor gene sets (including vertex).
    pub anchor_gene_sets: Vec<Vec<usize>>,
}

impl AnchorPrior {
    /// Build the prior from the finest pseudobulk level.
    ///
    /// 1. Depth-normalize PB columns, compute TF-IDF ratios.
    /// 2. Gram-Schmidt on the ratio simplex to find K anchor genes.
    /// 3. Arora-style recovery to get per-topic gene profiles.
    /// 4. For each anchor gene, pick the PB where it's most enriched.
    pub(crate) fn from_pseudobulk(
        finest: &CollapsedOut,
        n_topics: usize,
        markers: Option<&MarkerInfo>,
        margin_threshold: f32,
        finest_coarsening: Option<&FeatureCoarsening>,
        n_anchor_genes: usize,
        gene_filter: &[bool],
    ) -> anyhow::Result<Self> {
        // Prefer batch-corrected expression when available. Selection
        // should happen on the same signal the model sees after collapse.
        let mu_gp: &Mat = match finest.mu_adjusted.as_ref() {
            Some(adj) => {
                log::info!("anchor prior: using mu_adjusted (batch-corrected) as source");
                adj.posterior_mean()
            }
            None => {
                log::info!("anchor prior: mu_adjusted not present; using mu_observed");
                finest.mu_observed.posterior_mean()
            }
        };
        let n_pb = mu_gp.ncols();
        let d_full = mu_gp.nrows();
        if n_pb < 2 {
            return Err(anyhow::anyhow!(
                "anchor prior needs ≥2 pseudobulks, got {}",
                n_pb
            ));
        }

        // Depth-normalize each PB column to the median PB depth and
        // keep BOTH the pre-log1p (`x_raw_gp`, for the TF-IDF prior)
        // and log1p (`x_gp`, for Leiden selection geometry) views.
        // Clamping at 0 guards against negative entries that can
        // appear in a batch-residual matrix.
        let x_raw_gp = depth_normalize_columns(mu_gp);
        let x_gp = x_raw_gp.map(|v| v.max(0.0).ln_1p());

        // TF-IDF ratio: ratio_gp = x_raw[g,p] / mean_g
        // Non-negative, suitable for simplex projection.
        // Marker genes enriched in specific PBs get ratio >> 1,
        // housekeeping genes ≈ 1, absent genes ≈ 0/0 → 1 via eps.
        let eps_raw: f32 = 1e-6;
        let mean_g: DVec = DVec::from_fn(d_full, |g, _| {
            x_raw_gp.row(g).iter().sum::<f32>() / n_pb as f32 + eps_raw
        });
        let mut ratio_gp = Mat::zeros(d_full, n_pb);
        for p in 0..n_pb {
            for g in 0..d_full {
                ratio_gp[(g, p)] = (x_raw_gp[(g, p)] + eps_raw) / mean_g[g];
            }
        }

        // Filter genes for anchor selection only (not for training).
        // The gene_filter mask was computed upstream from cell-level CV
        // via k-means (separates housekeeping, informative, Ig/TCR).
        let k = n_topics.min(n_pb);
        let kept_idx: Vec<usize> = (0..d_full).filter(|&g| gene_filter[g]).collect();
        let n_kept = kept_idx.len();
        log::info!(
            "anchor gene filter: {}/{} genes for Gram-Schmidt",
            n_kept,
            d_full
        );

        let mut ratio_filtered = Mat::zeros(n_kept, n_pb);
        for (new_g, &orig_g) in kept_idx.iter().enumerate() {
            for p in 0..n_pb {
                ratio_filtered[(new_g, p)] = ratio_gp[(orig_g, p)];
            }
        }
        let (vertex_filtered, anchor_sets_filtered) =
            gram_schmidt_anchor_genes(&ratio_filtered, k, n_anchor_genes);

        // Map filtered indices back to full gene space
        let vertex_genes: Vec<usize> = vertex_filtered.iter().map(|&i| kept_idx[i]).collect();
        let anchor_sets: Vec<Vec<usize>> = anchor_sets_filtered
            .iter()
            .map(|set| set.iter().map(|&i| kept_idx[i]).collect())
            .collect();
        log::info!(
            "anchor vertex genes: {:?}, {} per set",
            vertex_genes,
            n_anchor_genes
        );

        // Anchor recovery using centroid of each anchor set.
        // For each topic k, average the ratio profiles of its anchor genes
        // to get a robust representative direction, then use Arora recovery.
        let anchor_weight_gk = anchor_recover_topics(&ratio_gp, &anchor_sets, k);

        // Anchor PB: for each topic, the PB where the vertex gene is most enriched.
        let anchor_pb_idx: Vec<usize> = vertex_genes
            .iter()
            .map(|&g| {
                (0..n_pb)
                    .max_by(|&a, &b| ratio_gp[(g, a)].partial_cmp(&ratio_gp[(g, b)]).unwrap())
                    .unwrap()
            })
            .collect();
        // Encoder-side anchor input: [K, D_enc]
        let anchor_pb_enc_kd: Mat = {
            let x_ed: Mat = match finest_coarsening {
                Some(fc) => fc.aggregate_rows_ds(&x_gp),
                None => x_gp.clone(),
            };
            let d_enc = x_ed.nrows();
            let mut out = Mat::zeros(k, d_enc);
            for (row, &pb) in anchor_pb_idx.iter().enumerate() {
                let src = x_ed.column(pb);
                for d in 0..d_enc {
                    out[(row, d)] = src[d];
                }
            }
            out
        };

        let (topic_labels, margin_scores) = match markers {
            Some(m) => {
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
            anchor_pb_enc_kd,
            vertex_genes,
            anchor_gene_sets: anchor_sets,
        })
    }

    /// `[K, D_enc]` encoder-side anchor input as a device tensor. Matches
    /// what `LogSoftmaxEncoder::forward_t` expects (raw non-negative
    /// features at the encoder's feature resolution).
    pub(crate) fn encoder_input_tensor(&self, dev: &Device) -> anyhow::Result<Tensor> {
        let t = self.anchor_pb_enc_kd.to_tensor(dev)?;
        Ok(t)
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
                // `coarsened_weight` returns [D_l, K] logits; transpose to
                // [K, D_l] and softmax to get probability targets for the
                // cross-entropy penalty.
                let w_dk = self.coarsened_weight(fc.as_ref());
                let w_kd = w_dk.transpose();
                let t = w_kd.to_tensor(dev)?;
                Ok(candle_nn::ops::softmax(&t, t.rank() - 1)?)
            })
            .collect()
    }

    /// `[D_level, K]` view of the prior, aggregating fine features into the
    /// coarse groups defined by `fc` and renormalizing each column on the
    /// simplex. `None` means the caller wants the full-resolution prior.
    /// Coarsen anchor logits by averaging fine-gene logits within each
    /// coarse bin. Returns `[D_coarse, K]`.
    pub(crate) fn coarsened_weight(&self, fc: Option<&FeatureCoarsening>) -> Mat {
        match fc {
            Some(fc) => {
                let k = self.anchor_weight_gk.ncols();
                let d_c = fc.num_coarse;
                let mut w = Mat::zeros(d_c, k);
                for (c, fine_indices) in fc.coarse_to_fine.iter().enumerate() {
                    let n = fine_indices.len() as f32;
                    if n > 0.0 {
                        for &f in fine_indices {
                            for kk in 0..k {
                                w[(c, kk)] += self.anchor_weight_gk[(f, kk)];
                            }
                        }
                        for kk in 0..k {
                            w[(c, kk)] /= n;
                        }
                    }
                }
                w
            }
            None => self.anchor_weight_gk.clone(),
        }
    }

    /// Overwrite the dictionary logits AND the per-gene `logit_bias` for
    /// each level's decoder. The convention matches both `fit_topic` and
    /// `fit_indexed_topic`: the decoder at level `i` lives under the
    /// VarBuilder path `dec_{i}`, so its logit tensor is at
    /// `dec_{i}.dictionary.logits` (`{n_topics}, {d_level}`) and the
    /// bias at `dec_{i}.dictionary.logit_bias` (`{1, d_level}`).
    ///
    /// **Background separation**: the decoder β is computed as
    /// `softmax_g(logits[k, :] + logit_bias[:])`, so by initializing
    /// `logit_bias` to the centered per-gene background (mean log1p
    /// expression across PBs) and setting `logits` to the
    /// gene-standardized anchor prior, we decouple housekeeping (which
    /// ends up in `logit_bias`) from cell-type-specific signal (which
    /// stays in `logits`). The anchor penalty during training acts on
    /// `logits` alone and so pulls *only* the cell-type-specific
    /// component toward the prior.
    ///
    pub(crate) fn init_decoder_dictionary(
        &self,
        parameters: &VarMap,
        level_coarsenings: &[Option<FeatureCoarsening>],
        dev: &Device,
    ) -> anyhow::Result<()> {
        let data = parameters.data().lock().expect("VarMap lock");
        for (level, fc) in level_coarsenings.iter().enumerate() {
            let logits_name = decoder_logits_var_path(level);
            if let Some(logits_var) = data.get(&logits_name) {
                let w_dk = self.coarsened_weight(fc.as_ref());
                let w_kd = w_dk.transpose();
                logits_var.set(&w_kd.to_tensor(dev)?)?;
            }
        }
        Ok(())
    }

    /// Write per-anchor label + score TSV and, when markers are given,
    /// the candidate-marker expansion table.
    pub(crate) fn write_side_outputs(
        &self,
        out_prefix: &str,
        gene_names: &[Box<str>],
        markers: Option<&MarkerInfo>,
    ) -> anyhow::Result<()> {
        // Anchor labels + vertex gene + anchor gene set
        let labels_path = format!("{out_prefix}.anchor_labels.tsv");
        let mut f = std::fs::File::create(&labels_path)?;
        writeln!(
            f,
            "topic_idx\tpb_idx\tlabel\tvertex_gene\tanchor_genes\ttop1_z\ttop2_z\tmargin"
        )?;
        for (i, ((&pb, label), (t1, t2))) in self
            .anchor_pb_idx
            .iter()
            .zip(self.topic_labels.iter())
            .zip(self.margin_scores.iter())
            .enumerate()
        {
            let vertex_name = if i < self.vertex_genes.len() {
                let gi = self.vertex_genes[i];
                if gi < gene_names.len() {
                    &*gene_names[gi]
                } else {
                    "?"
                }
            } else {
                "?"
            };
            let anchor_names: String = if i < self.anchor_gene_sets.len() {
                self.anchor_gene_sets[i]
                    .iter()
                    .map(|&gi| {
                        if gi < gene_names.len() {
                            &*gene_names[gi]
                        } else {
                            "?"
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            } else {
                String::new()
            };
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}",
                i,
                pb,
                label,
                vertex_name,
                anchor_names,
                t1,
                t2,
                t1 - t2
            )?;
        }
        log::info!("wrote {}", labels_path);

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
            log::info!("wrote {}", expansion_path);
        }
        Ok(())
    }
}

/// Encoder-side anchor penalty: adds `-λ · mean_k log q(k | anchor_pb_k)`
/// to `loss`. The encoder is asked to map each anchor PB's expression
/// vector to a topic distribution concentrated on its own topic.
///
/// `anchor_input_kd` is the `[K, D_enc]` tensor produced by
/// `AnchorPrior::encoder_input_tensor`. `None`, zero λ, or K < 2 turns
/// the penalty into a no-op.
pub(crate) fn encoder_anchor_penalty<E>(
    loss: Tensor,
    encoder: &E,
    anchor_input_kd: Option<&Tensor>,
    lambda: f32,
) -> anyhow::Result<Tensor>
where
    E: candle_util::candle_model_traits::EncoderModuleT,
{
    let Some(input) = anchor_input_kd else {
        return Ok(loss);
    };
    if lambda <= 0.0 {
        return Ok(loss);
    }
    let k = input.dim(0)?;
    if k < 2 {
        return Ok(loss);
    }

    // Encoder returns (log_softmax_kk, kl_k). We only need the log-prob.
    let (log_z_kk, _) = encoder.forward_t(input, None, true)?;

    // Diagonal cross-entropy: pull each row's mass to its own topic.
    // gather along dim 1 with index k selects log_z_kk[k, k].
    let dev = log_z_kk.device();
    let idx: Vec<u32> = (0..k as u32).collect();
    let idx = Tensor::from_vec(idx, (k, 1), dev)?;
    let diag = log_z_kk.gather(&idx, 1)?;
    let pen = (diag.mean_all()?.neg()? * lambda as f64)?;
    Ok((loss + pen)?)
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
/// scales with the number of minibatches (M = N_total / N_batch). If you
/// change `--minibatch-size`, you will typically want to rescale
/// `--anchor-penalty` in inverse proportion (or rely on Adam's adaptive
/// step size plus linear-LR scaling to absorb the difference).
pub(crate) fn anchor_penalty_at_level(
    loss: Tensor,
    parameters: &VarMap,
    anchor_prior_per_level: Option<&[Tensor]>,
    lambda: f32,
    level: usize,
) -> anyhow::Result<Tensor> {
    let Some(priors) = anchor_prior_per_level else {
        return Ok(loss);
    };
    if lambda <= 0.0 {
        return Ok(loss);
    }
    // Hold the guard just long enough to clone the logits Tensor handle.
    // Clone is a cheap Arc bump that preserves TensorId and is_variable,
    // so gradients still flow back to the underlying Var through Adam.
    let logits_kd = {
        let data = parameters.data().lock().expect("VarMap lock");
        let name = decoder_logits_var_path(level);
        match data.get(&name) {
            Some(var) => var.as_tensor().clone(),
            None => return Ok(loss),
        }
    };
    // Both `anchor_kd` and `log_beta_kd` are [K, D_l]. The anchor tensors
    // are pre-transposed at build time so no per-minibatch transpose is
    // needed here. Cross-entropy is summed over D and averaged over K.
    let anchor_kd = &priors[level];
    let log_beta_kd = candle_nn::ops::log_softmax(&logits_kd, logits_kd.rank() - 1)?;
    let ce = (anchor_kd * &log_beta_kd)?.sum(1)?.neg()?;
    let pen = (ce.mean_all()? * lambda as f64)?;
    Ok((loss + pen)?)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Write `softmax(col)` from a source column view into a destination column
/// view of the same length. Used per anchor when building
/// `anchor_weight_gk`.
/// Return a column-z-scored copy of `x_pg` (PB rows × gene columns).
/// Each gene column is shifted and scaled so its mean is 0 and its std is 1.
/// Constant columns get zero (their contribution to residuals is nil anyway).
fn zscore_columns(x_pg: &Mat) -> Mat {
    let mut out = x_pg.clone();
    let n = out.nrows() as f32;
    if n < 2.0 {
        return out;
    }
    for mut col in out.column_iter_mut() {
        let mean = col.iter().sum::<f32>() / n;
        let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        let sd = var.sqrt();
        if sd > 1e-8 {
            for v in col.iter_mut() {
                *v = (*v - mean) / sd;
            }
        } else {
            col.fill(0.0);
        }
    }
    out
}

/// K-means gene filter on log(CV). Fits K=2 and K=3, picks via BIC,
/// and keeps the majority cluster (informative genes). Filters out
/// both low-CV housekeeping (uniform across PBs) and extreme-CV
/// outliers (Ig/TCR expressed in 1-2 PBs).
pub(crate) fn kmeans_cv_filter(cv: &[f32]) -> Vec<bool> {
    use matrix_util::clustering::{Kmeans, KmeansArgs};

    let n = cv.len();
    if n < 10 {
        return vec![true; n];
    }

    // Work in log(CV + eps) space as a 1D matrix [N, 1].
    let eps = 1e-6f32;
    let log_cv: Vec<f32> = cv.iter().map(|&v| (v + eps).ln()).collect();
    let mat = Mat::from_column_slice(n, 1, &log_cv);

    // Try K=2 and K=3
    let labels_2 = mat.kmeans_rows(KmeansArgs::with_clusters(2));
    let labels_3 = mat.kmeans_rows(KmeansArgs::with_clusters(3));
    let bic_2 = kmeans_bic(&log_cv, &labels_2, 2);
    let bic_3 = kmeans_bic(&log_cv, &labels_3, 3);

    let (labels, k_best) = if bic_2 < bic_3 {
        (labels_2, 2)
    } else {
        (labels_3, 3)
    };

    // Find the largest cluster = informative genes
    let mut counts = vec![0usize; k_best];
    for &l in &labels {
        if l < k_best {
            counts[l] += 1;
        }
    }
    let majority = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap();

    // Log cluster stats
    let mut means = vec![0.0f32; k_best];
    for (i, &l) in labels.iter().enumerate() {
        if l < k_best {
            means[l] += log_cv[i];
        }
    }
    for k in 0..k_best {
        if counts[k] > 0 {
            means[k] /= counts[k] as f32;
        }
    }
    log::info!(
        "gene filter: K={}, BIC(2)={:.0} BIC(3)={:.0}, clusters: {:?} (log_cv means: {:?}), keeping cluster {}",
        k_best, bic_2, bic_3, counts,
        means.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>(),
        majority
    );

    labels.iter().map(|&l| l == majority).collect()
}

/// BIC for 1D k-means: n·log(RSS/n) + k·log(n).
fn kmeans_bic(data: &[f32], labels: &[usize], k: usize) -> f64 {
    let n = data.len();
    let mut centroids = vec![0.0f64; k];
    let mut counts = vec![0usize; k];
    for (i, &l) in labels.iter().enumerate() {
        if l < k {
            centroids[l] += data[i] as f64;
            counts[l] += 1;
        }
    }
    for j in 0..k {
        if counts[j] > 0 {
            centroids[j] /= counts[j] as f64;
        }
    }
    let rss: f64 = data
        .iter()
        .zip(labels)
        .map(|(&x, &l)| {
            let c = if l < k { centroids[l] } else { 0.0 };
            let d = x as f64 - c;
            d * d
        })
        .sum();
    n as f64 * (rss / n as f64 + 1e-30).ln() + k as f64 * (n as f64).ln()
}

/// Gram-Schmidt anchor gene selection in PB space.
///
/// Each gene g is represented by its ratio row in R^P. L1-normalize
/// to the PB simplex, then greedily select K vertex genes via
/// Gram-Schmidt. For each vertex, also return the top `n_per_vertex`
/// genes closest to that vertex direction (by cosine similarity on the
/// simplex before projection).
///
/// Returns `(vertex_genes, anchor_sets)` where `anchor_sets[k]` is
/// a Vec of gene indices for topic k (including the vertex gene).
fn gram_schmidt_anchor_genes(
    ratio_gp: &Mat,
    k: usize,
    n_per_vertex: usize,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let g = ratio_gp.nrows();
    let p = ratio_gp.ncols();

    // L1-normalize each gene row → simplex in PB space.
    // Input is non-negative ratios, so L1-norm is just row sum.
    let mut q = Mat::zeros(g, p);
    for i in 0..g {
        let mut sum = 0.0f32;
        for j in 0..p {
            sum += ratio_gp[(i, j)];
        }
        if sum > 1e-12 {
            for j in 0..p {
                q[(i, j)] = ratio_gp[(i, j)] / sum;
            }
        }
    }

    // Keep the original simplex vectors for finding nearest neighbors
    let q_orig = q.clone();

    // Gram-Schmidt: greedily pick K vertex genes
    let mut vertices = Vec::with_capacity(k);
    let mut norms: Vec<f32> = (0..g)
        .map(|i| {
            let mut s = 0.0f32;
            for j in 0..p {
                s += q[(i, j)] * q[(i, j)];
            }
            s
        })
        .collect();

    for _ in 0..k {
        let best = norms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        vertices.push(best);

        let anchor_norm_sq: f32 = norms[best];
        if anchor_norm_sq < 1e-20 {
            break;
        }
        for i in 0..g {
            let mut dot = 0.0f32;
            for j in 0..p {
                dot += q[(i, j)] * q[(best, j)];
            }
            let coeff = dot / anchor_norm_sq;
            let mut new_norm = 0.0f32;
            for j in 0..p {
                q[(i, j)] -= coeff * q[(best, j)];
                new_norm += q[(i, j)] * q[(i, j)];
            }
            norms[i] = new_norm;
        }
    }

    // For each vertex, find top-N genes by cosine similarity on
    // the original (pre-projection) simplex.
    let mut anchor_sets = Vec::with_capacity(vertices.len());
    let used: std::collections::HashSet<usize> = vertices.iter().copied().collect();

    for &vi in &vertices {
        // Cosine similarity of all genes to vertex vi
        let v_norm: f32 = (0..p)
            .map(|j| q_orig[(vi, j)] * q_orig[(vi, j)])
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        let mut sims: Vec<(usize, f32)> = (0..g)
            .filter(|i| !used.contains(i) || *i == vi)
            .map(|i| {
                let dot: f32 = (0..p).map(|j| q_orig[(i, j)] * q_orig[(vi, j)]).sum();
                let i_norm: f32 = (0..p)
                    .map(|j| q_orig[(i, j)] * q_orig[(i, j)])
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-12);
                (i, dot / (i_norm * v_norm))
            })
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let set: Vec<usize> = sims.iter().take(n_per_vertex).map(|&(i, _)| i).collect();
        anchor_sets.push(set);
    }

    (vertices, anchor_sets)
}

/// Recover topic-gene matrix from anchor gene sets.
///
/// Each topic k has a set of anchor genes. The representative direction
/// for topic k is the centroid (mean) of its anchor genes' ratio profiles.
/// Then for each gene g, express its profile as a convex combination of
/// the K centroid profiles → topic membership coefficients.
fn anchor_recover_topics(ratio_gp: &Mat, anchor_sets: &[Vec<usize>], k: usize) -> Mat {
    let g = ratio_gp.nrows();
    let p = ratio_gp.ncols();

    // Build anchor matrix A [K, P] — each row is the centroid of
    // an anchor set's ratio profiles.
    let mut a_kp = Mat::zeros(k, p);
    for (ki, set) in anchor_sets.iter().enumerate() {
        let n = set.len() as f32;
        for &gi in set {
            for j in 0..p {
                a_kp[(ki, j)] += ratio_gp[(gi, j)];
            }
        }
        if n > 0.0 {
            for j in 0..p {
                a_kp[(ki, j)] /= n;
            }
        }
    }

    // Q = A · A^T [K, K] — Gram matrix of anchor profiles
    let a_t = a_kp.transpose();
    let q_kk = &a_kp * &a_t;

    // For each gene g, solve: min ||A^T c - tfidf[g,:]||^2
    // i.e., c_g = (A A^T)^{-1} A · tfidf[g, :]
    // Then project c_g onto the simplex.

    // Pseudo-inverse approach: (A A^T + λI)^{-1} A
    let lambda = 1e-4f32;
    let mut q_reg = q_kk.clone();
    for i in 0..k {
        q_reg[(i, i)] += lambda;
    }

    // Cholesky or direct inverse for small K×K
    let q_inv = match q_reg.clone().try_inverse() {
        Some(inv) => inv,
        None => {
            log::warn!("anchor recovery: Gram matrix singular, using uniform β");
            return Mat::from_fn(g, k, |_, _| 1.0 / k as f32);
        }
    };

    // W = Q_inv · A [K, P]
    let w_kp = &q_inv * &a_kp;

    // For each gene: c_gk = (W · tfidf[g,:])_k, then project to simplex
    let mut beta_gk = Mat::zeros(g, k);
    for gi in 0..g {
        // c = W · tfidf[g, :]
        for ki in 0..k {
            let mut val = 0.0f32;
            for j in 0..p {
                val += w_kp[(ki, j)] * ratio_gp[(gi, j)];
            }
            beta_gk[(gi, ki)] = val;
        }
        // Project onto simplex: clamp negatives, normalize
        let mut row_sum = 0.0f32;
        for ki in 0..k {
            beta_gk[(gi, ki)] = beta_gk[(gi, ki)].max(0.0);
            row_sum += beta_gk[(gi, ki)];
        }
        if row_sum > 1e-12 {
            for ki in 0..k {
                beta_gk[(gi, ki)] /= row_sum;
            }
        } else {
            // Uniform fallback for genes with no signal
            for ki in 0..k {
                beta_gk[(gi, ki)] = 1.0 / k as f32;
            }
        }
    }

    // β_gk is [G, K] with each row on the topic simplex.
    // Convert to logits for decoder init (decoder applies softmax over genes per topic).
    // We want logits[g, k] such that softmax_g(logits[:, k]) ∝ β[g, k].
    // Since β[g, k] is the topic membership of gene g, and the decoder's
    // β_kg = softmax_g(logits[k, :]), we need logits[k, g] = log(β[g, k] + eps).
    // Store as [G, K] — the caller transposes when writing to the VarMap.
    let eps = 1e-8f32;
    beta_gk.map(|v| (v + eps).ln())
}

/// Library-size normalize each column of `[D, n_pb]` to the median PB
/// depth and clamp to ≥0, WITHOUT taking log1p. Returned values are
/// in the same "count-per-median-depth" units as the input.
fn depth_normalize_columns(mu_gp: &Mat) -> Mat {
    let n_pb = mu_gp.ncols();
    let totals: Vec<f32> = (0..n_pb)
        .map(|p| mu_gp.column(p).iter().map(|&v| v.max(0.0)).sum::<f32>())
        .collect();
    let mut sorted: Vec<f32> = totals.iter().copied().filter(|&t| t > 0.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        1.0
    } else {
        sorted[sorted.len() / 2]
    };

    let mut out = mu_gp.clone();
    for (p, &total) in totals.iter().enumerate() {
        let scale = if total > 1e-8 { median / total } else { 0.0 };
        let mut col = out.column_mut(p);
        for v in col.iter_mut() {
            *v = v.max(0.0) * scale;
        }
    }
    out
}

/// Score each anchor PB against every celltype by taking the mean z-score
/// of the celltype's marker genes within the anchor's PB row. Assign each
/// anchor the best-scoring celltype iff `top1 - top2 >= margin_threshold`.
/// Anchors that don't clear the margin — or whose best celltype has no
/// markers — become `novel_{i}`.
///
/// When multiple anchors pick the same celltype, later anchors get a
/// numeric suffix so the labels stay unique (e.g. `T_cells`, `T_cells_2`).
fn label_anchors(
    x_zscored: &Mat,
    anchor_pb_idx: &[usize],
    markers: &MarkerInfo,
    margin_threshold: f32,
) -> (Vec<Box<str>>, Vec<(f32, f32)>) {
    // Index marker gene rows by celltype for fast scoring. Iterate
    // column-by-column so each nonzero test is a cache-friendly column
    // scan — the membership matrix is usually mostly zeros.
    let n_ct = markers.celltypes.len();
    let mut marker_rows_per_ct: Vec<Vec<usize>> = vec![Vec::new(); n_ct];
    for (c, col) in markers.membership_gc.column_iter().enumerate() {
        for (g, &v) in col.iter().enumerate() {
            if v > 0.0 {
                marker_rows_per_ct[c].push(g);
            }
        }
    }

    let mut used_counts: rustc_hash::FxHashMap<Box<str>, usize> = Default::default();
    let mut labels: Vec<Box<str>> = Vec::with_capacity(anchor_pb_idx.len());
    let mut scores: Vec<(f32, f32)> = Vec::with_capacity(anchor_pb_idx.len());

    for (i, &pb) in anchor_pb_idx.iter().enumerate() {
        let pb_row = x_zscored.row(pb);

        // Score celltype c = mean z-score over its marker genes.
        let mut per_ct: Vec<(usize, f32)> = (0..n_ct)
            .filter_map(|c| {
                let rows = &marker_rows_per_ct[c];
                if rows.is_empty() {
                    return None;
                }
                let s: f32 = rows.iter().map(|&g| pb_row[g]).sum::<f32>() / rows.len() as f32;
                Some((c, s))
            })
            .collect();
        per_ct.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (top1_ct, top1_z) = per_ct.first().copied().unwrap_or((usize::MAX, 0.0));
        let top2_z = per_ct.get(1).map(|&(_, s)| s).unwrap_or(f32::NEG_INFINITY);
        scores.push((top1_z, top2_z));

        let clears_margin = top1_ct != usize::MAX && (top1_z - top2_z) >= margin_threshold;
        if !clears_margin {
            labels.push(format!("novel_{i}").into_boxed_str());
            continue;
        }
        let base = markers.celltypes[top1_ct].clone();
        let n = used_counts.entry(base.clone()).or_insert(0);
        *n += 1;
        let label = if *n == 1 {
            base
        } else {
            format!("{}_{}", base, n).into_boxed_str()
        };
        labels.push(label);
    }

    (labels, scores)
}

/// Strip `_<n>` suffix from disambiguated multi-anchor labels, so we can
/// look them up in the marker file's celltype list.
fn base_celltype_label(label: &str) -> &str {
    if let Some(pos) = label.rfind('_') {
        let tail = &label[pos + 1..];
        if tail.parse::<usize>().is_ok() {
            return &label[..pos];
        }
    }
    label
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_normalize_equalizes_columns() {
        // Two PBs with very different totals should produce identical
        // depth-normalized rows — they differ only by a global 10×
        // scale, so scaling to the median depth collapses them.
        let mut mu = Mat::zeros(4, 2);
        mu[(0, 0)] = 10.0;
        mu[(1, 0)] = 20.0;
        mu[(2, 0)] = 30.0;
        mu[(3, 0)] = 40.0;
        mu[(0, 1)] = 100.0;
        mu[(1, 1)] = 200.0;
        mu[(2, 1)] = 300.0;
        mu[(3, 1)] = 400.0;
        let normed = depth_normalize_columns(&mu);
        for g in 0..4 {
            let a = normed[(g, 0)];
            let b = normed[(g, 1)];
            assert!((a - b).abs() < 1e-5, "col0[{g}]={a} col1[{g}]={b}");
        }
    }

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
            anchor_pb_enc_kd: Mat::zeros(k, d_full),
            vertex_genes: (0..k).collect(),
            anchor_gene_sets: (0..k).map(|i| vec![i]).collect(),
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

    #[test]
    fn zscore_is_unit_std() {
        let x = Mat::from_row_slice(
            4,
            3,
            &[1.0, 2.0, 5.0, 2.0, 3.0, 5.0, 3.0, 4.0, 5.0, 4.0, 5.0, 5.0],
        );
        let z = zscore_columns(&x);
        for j in 0..x.ncols() {
            let col: Vec<f32> = z.column(j).iter().copied().collect();
            let mean = col.iter().sum::<f32>() / col.len() as f32;
            assert!(mean.abs() < 1e-5);
            if j < 2 {
                let var = col.iter().map(|v| v * v).sum::<f32>() / col.len() as f32;
                assert!((var - 1.0).abs() < 1e-5);
            } else {
                // constant column → all zero
                assert!(col.iter().all(|&v| v == 0.0));
            }
        }
    }

    #[test]
    fn base_label_strips_numeric_suffix() {
        assert_eq!(base_celltype_label("T_cells"), "T_cells");
        assert_eq!(base_celltype_label("T_cells_2"), "T_cells");
        assert_eq!(base_celltype_label("novel_5"), "novel"); // caller filters novels anyway
    }
}
