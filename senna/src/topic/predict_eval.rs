//! Observed-vs-predicted agreement on held-out cells, on both axes.
//!
//! `predictive.parquet`'s log-likelihood answers "how probable are this cell's
//! counts under the model". A correlation answers a different and often more
//! legible question — "does the model rank this cell's genes the way the data
//! does" — and it is the one a benchmark table usually reports.
//!
//! # Two axes, because they are different questions
//!
//! - **Per cell, across genes.** One number per test cell: does the model get
//!   the *shape* of this cell's profile right? Averaged, it is a per-cell fit.
//! - **Per gene, across cells.** One number per gene: does the model track this
//!   gene's variation *between* cells? This is the harder one and the one that
//!   collapses first — a model can reproduce every cell's overall profile while
//!   carrying no information about which cells differ.
//!
//! A method can look strong on the first and be dead on the second, so both are
//! reported rather than one standing in for the other.
//!
//! # Why the evaluation gene set is fixed by the caller
//!
//! Scoring happens on each model's own gene axis, so a model trained on 80% of
//! the genes is otherwise scored over fewer — and, on average, better-expressed
//! — genes than a model trained on all of them, and the two numbers are not
//! comparable. `--eval-features` pins one gene set for every method, whatever
//! each happens to carry, which is what makes an ablation series readable.

use crate::embed_common::*;
use crate::logging::new_progress_bar;
use matrix_util::agreement::{pearson_log1p, spearman, CellAgreement};
use rayon::prelude::*;

/// Streaming accumulator: fed one cell at a time, over the evaluation genes.
///
/// The per-gene side keeps `[n_eval_genes]` running sums rather than the values,
/// so Pearson is exact in one pass and memory does not grow with the number of
/// test cells. Spearman needs ranks over cells and therefore does keep the
/// values — bounded by the evaluation gene set, which is the other reason to
/// restrict it.
pub(crate) struct PredictEval {
    /// Model-axis gene indices to score, in output order.
    eval_genes: Vec<usize>,
    keep_values: bool,
    obs: Vec<Vec<f32>>,
    pred: Vec<Vec<f32>>,
}

impl PredictEval {
    pub fn new(eval_genes: Vec<usize>, keep_values: bool) -> Self {
        let g = eval_genes.len();
        Self {
            eval_genes,
            keep_values,
            obs: vec![Vec::new(); if keep_values { g } else { 0 }],
            pred: vec![Vec::new(); if keep_values { g } else { 0 }],
        }
    }

    #[must_use]
    pub fn genes(&self) -> &[usize] {
        &self.eval_genes
    }

    /// Score one cell. `obs_of` and `pred_of` read a value by MODEL gene index.
    /// Store one cell's observed and predicted values for the across-cell axis.
    ///
    /// The correlations themselves are computed by the caller, in parallel — this
    /// only folds the result in, so it stays a `&mut self` step in column order.
    pub fn keep(&mut self, o: &[f32], p: &[f32]) {
        if !self.keep_values {
            return;
        }
        for (i, (&a, &b)) in o.iter().zip(p).enumerate() {
            self.obs[i].push(a);
            self.pred[i].push(b);
        }
    }

    /// `(gene index, spearman, pearson_log1p, mean observed)` per evaluation
    /// gene. Empty when the accumulator was built without `keep_values`.
    #[must_use]
    pub fn per_gene(&self) -> Vec<(usize, f32, f32, f32)> {
        (0..self.obs.len())
            .map(|i| {
                let (o, p) = (&self.obs[i], &self.pred[i]);
                let mean = if o.is_empty() {
                    0.0
                } else {
                    o.iter().sum::<f32>() / o.len() as f32
                };
                (
                    self.eval_genes[i],
                    spearman(o, p),
                    pearson_log1p(o, p),
                    mean,
                )
            })
            .collect()
    }
}

/// Resolve `--eval-features` to model-axis gene indices.
///
/// A name absent from the model is skipped with a count, not an error: the whole
/// point of a fixed evaluation set is that it is shared by methods whose axes
/// differ, so some misses are expected.
///
/// Parsing goes through [`read_name_list`] rather than a local `lines()` split,
/// because this file is contractually the SAME file across every arm and both
/// commands. Three hand-rolled parsers had already diverged — and since
/// `--ablate-features` implies `--eval-features` on its own path, a file the two
/// read differently would hide fewer genes from the encoder than the scorer
/// grades on, which is the one direction that flatters a model.
pub(crate) fn resolve_eval_genes(
    path: Option<&str>,
    gene_names: &[Box<str>],
) -> anyhow::Result<Vec<usize>> {
    let Some(path) = path else {
        return Ok((0..gene_names.len()).collect());
    };
    let index: std::collections::HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), i))
        .collect();
    let wanted = matrix_util::common_io::read_name_list(path)
        .map_err(|e| anyhow::anyhow!("reading --eval-features {path}: {e}"))?;
    let mut out = Vec::with_capacity(wanted.len());
    let mut missing = 0usize;
    for name in &wanted {
        match index.get(name.as_ref()) {
            Some(&i) => out.push(i),
            None => missing += 1,
        }
    }
    anyhow::ensure!(
        !out.is_empty(),
        "{path}: none of the listed features are in this model's gene axis"
    );
    if missing > 0 {
        info!(
            "{path}: {missing} of {} names are not in this model",
            wanted.len()
        );
    }
    info!(
        "Scoring {} of {} model features",
        out.len(),
        gene_names.len()
    );
    Ok(out)
}

#[cfg(test)]
mod tests;

/// Score a topic-family prediction on held-out cells.
///
/// A separate pass over the test half rather than a hook inside the likelihood
/// loops: the reconstruction is the same `δ·exp(β)·θ` those already build, and
/// the test half is the small side of a split by construction, so one extra
/// stream buys one formula in one place instead of the same arithmetic threaded
/// through the dense, masked and bge paths.
///
/// `keep_per_gene` also holds `[eval_genes × n_cells]` of observed and predicted
/// values, which is what the across-cell axis needs; leave it off and only the
/// per-cell column is produced.
/// How a family turns a latent into an expected profile.
///
/// The two forms are genuinely different arithmetic — a topic model mixes
/// dictionaries, `bge` exponentiates a dot product — and that is the whole
/// reason this is an enum rather than a shared matrix. Everything downstream of
/// the rate (normalising, the null, both correlation axes, the nats/count) is
/// identical, so it is written once and both families are graded by it.
pub(crate) enum Reconstruction<'a> {
    /// `δ · exp(β) · θ`, the topic families.
    Topic {
        exp_beta_dk: Mat,
        theta_nk: &'a Mat,
        delta_db: Option<&'a Mat>,
    },
    /// `exp(b + ρ·θ)`, the log-linear embedding `bge` fits.
    Embedding {
        rho_dh: Mat,
        b_feat: &'a [f32],
        theta_nh: &'a Mat,
    },
}

impl Reconstruction<'_> {
    /// Number of model features — the axis both the rate and `obs` are indexed on.
    fn n_features(&self) -> usize {
        match self {
            Self::Topic { exp_beta_dk, .. } => exp_beta_dk.nrows(),
            Self::Embedding { rho_dh, .. } => rho_dh.nrows(),
        }
    }

    /// Expected profile for an explicit block of latents, as `[D × n]`.
    ///
    /// Split out from [`Self::block`] so the null can reuse the same arithmetic
    /// on a one-row mean: if the two ever drifted apart, the model and its floor
    /// would be normalised differently and the reported gain would be an artifact
    /// of that difference rather than of the model.
    fn rate(&self, theta_rows: &Mat) -> Mat {
        match self {
            Self::Topic { exp_beta_dk, .. } => exp_beta_dk * theta_rows.transpose(),
            Self::Embedding { rho_dh, b_feat, .. } => {
                debug_assert_eq!(
                    b_feat.len(),
                    rho_dh.nrows(),
                    "feature bias and embedding disagree on the feature axis"
                );
                let mut rate = rho_dh * theta_rows.transpose();
                for (d, b) in b_feat.iter().enumerate().take(rate.nrows()) {
                    rate.row_mut(d).add_scalar_mut(*b);
                }
                // Shift each column to its own max before exponentiating: the
                // logits are unbounded and only their differences matter once the
                // column is normalised, so this is exact, not an approximation.
                for mut col in rate.column_iter_mut() {
                    let m = col.max();
                    col.apply(|v| *v = (*v - m).exp());
                }
                rate
            }
        }
    }

    /// Expected profile for cells `lb..lb+n`, as `[D × n]`.
    fn block(&self, data_vec: &SparseIoVec, lb: usize, n: usize) -> Mat {
        let theta = match self {
            Self::Topic { theta_nk, .. } => theta_nk.rows(lb, n).into_owned(),
            Self::Embedding { theta_nh, .. } => theta_nh.rows(lb, n).into_owned(),
        };
        let mut rate = self.rate(&theta);
        if let Self::Topic {
            delta_db: Some(delta),
            ..
        } = self
        {
            let batch_ids = data_vec.get_batch_membership(lb..lb + n);
            for (j, &b) in batch_ids.iter().enumerate() {
                let mut col = rate.column_mut(j);
                col.component_mul_assign(&delta.column(b));
                let z = col.sum().max(1e-12);
                col /= z;
            }
        }
        rate
    }
}

/// The observed composition of a dataset over the scored genes.
///
/// Pointed at the test half it is the CEILING; pointed at the training half
/// (`--null-from`) it is the null. Same arithmetic either way — a count-weighted
/// marginal, which is what a predictor with no cell-specific information would
/// say — and the only thing that distinguishes the two is which data it reads.
///
/// Read from the test half it is fitted on the very data being scored, so it
/// knows that region's exact marginal — an upper reference, not a floor.
/// Reporting it beside the null brackets the answer: the gap between the two is
/// composition shift between the halves rather than anything about the model.
///
/// Costs one streaming pass, because the composition is not known until the data
/// has been read once.
pub(crate) fn empirical_composition(
    data_vec: &SparseIoVec,
    gene_remap: Option<&crate::topic::eval::GeneRemap>,
    eval_genes: &[usize],
    d_train: usize,
    minibatch_size: usize,
) -> anyhow::Result<Vec<f32>> {
    let ntot = data_vec.num_columns();
    let mut totals = vec![0f64; d_train];
    let bar = new_progress_bar(ntot.div_ceil(minibatch_size.max(1)) as u64)
        .with_message("null composition");
    for (lb, ub) in create_jobs(ntot, 0, Some(minibatch_size)) {
        let csc = data_vec.read_columns_csc(lb..ub)?;
        // Per-thread partial totals, summed at the end. A `[D_train]` f64 vector
        // is a few hundred KB, so the fold's per-thread copy is cheap next to
        // walking every non-zero of the block — and this pass runs twice per
        // `predict` (the test-half ceiling and the training-half null).
        let block: Vec<f64> = (0..csc.ncols())
            .into_par_iter()
            .fold(
                || vec![0f64; d_train],
                |mut acc, j| {
                    let col = csc.col(j);
                    for (&row_new, &val) in col.row_indices().iter().zip(col.values()) {
                        let row_train = match gene_remap {
                            Some(rm) => rm.new_to_train[row_new],
                            None => Some(row_new),
                        };
                        if let Some(r) = row_train {
                            acc[r] += f64::from(val);
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0f64; d_train],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    a
                },
            );
        totals.iter_mut().zip(&block).for_each(|(t, b)| *t += b);
        bar.inc(1);
    }
    bar.finish_and_clear();
    Ok(normalize_over(&totals, eval_genes, d_train))
}

/// Restrict per-gene totals to the scored genes and normalise to a composition.
///
/// Genes outside the evaluation set are zeroed rather than left to contribute to
/// the denominator: the likelihood renormalises over the scored genes, so a null
/// carrying mass outside them would be a different distribution than the model it
/// is compared against.
pub(crate) fn normalize_over(totals: &[f64], eval_genes: &[usize], d_train: usize) -> Vec<f32> {
    let mut out = vec![0f32; d_train];
    let mut z = 0f64;
    for &g in eval_genes {
        if let Some(&t) = totals.get(g) {
            out[g] = t as f32;
            z += t;
        }
    }
    if z > 0.0 {
        out.iter_mut().for_each(|v| *v /= z as f32);
    }
    out
}

/// One cell's scored result: agreement, likelihood, its ceiling contribution,
/// and the observed/predicted vectors the across-cell axis keeps.
type ScoredCell = (CellAgreement, CellLlik, f64, Vec<f32>, Vec<f32>);

pub(crate) struct EvalArgs<'a> {
    pub data_vec: &'a SparseIoVec,
    /// The null, on the training axis, already restricted to the scored genes.
    ///
    /// Supplied by the caller rather than derived here because the right floor is
    /// the *training* marginal: model-independent, so every arm shares it, and not
    /// fitted on the test half, so it is a baseline a model could actually have
    /// matched. `None` falls back to the test-half composition, which is an oracle
    /// and flatters nothing — the caller warns when that happens.
    pub null_comp: Option<Vec<f32>>,
    pub recon: Reconstruction<'a>,
    /// The *unablated* mapping: this densifies observed truth, not encoder input.
    pub gene_remap: Option<&'a crate::topic::eval::GeneRemap>,
    /// Training-axis gene positions to score.
    pub eval_genes: Vec<usize>,
    pub minibatch_size: usize,
    pub keep_per_gene: bool,
}

pub(crate) fn evaluate_predictions(a: EvalArgs<'_>) -> anyhow::Result<EvalOutcome> {
    let EvalArgs {
        data_vec,
        null_comp,
        recon,
        gene_remap,
        eval_genes,
        minibatch_size,
        keep_per_gene,
    } = a;
    let d_train = recon.n_features();
    let ceiling_comp =
        empirical_composition(data_vec, gene_remap, &eval_genes, d_train, minibatch_size)?;
    let null_comp = null_comp.unwrap_or_else(|| ceiling_comp.clone());
    let ntot = data_vec.num_columns();
    let mut eval = PredictEval::new(eval_genes, keep_per_gene);
    let mut per_cell = Vec::with_capacity(ntot);
    let mut per_cell_llik: Vec<CellLlik> = Vec::with_capacity(ntot);
    // Model and null likelihood accumulate here, both as a multinomial over the
    // *evaluation* genes only. Computing them side by side in one formula is the
    // point: the backends' own `llik` column is decoder-dependent (`multinom` is
    // NB-Fisher weighted, `nb` is a density), so it cannot be differenced against
    // a null or compared across families. These two can.
    let (mut model_llik, mut null_llik, mut eval_count) = (0f64, 0f64, 0f64);
    let mut ceiling_llik = 0f64;
    let bar = new_progress_bar(ntot.div_ceil(minibatch_size.max(1)) as u64)
        .with_message("scoring predictions");

    // Per-gene log-probabilities, formed once. Both compositions are fixed for
    // the run, so recomputing `ln(p/z)` inside the cell loop spent two divides
    // and two transcendentals per (cell, expressed gene) on a value that never
    // changes. Indexed by position within `eval.genes()`.
    let log_table = |comp: &[f32]| -> Vec<f64> {
        let z = eval
            .genes()
            .iter()
            .map(|&g| comp.get(g).copied().unwrap_or(0.0))
            .sum::<f32>()
            .max(1e-12);
        eval.genes()
            .iter()
            .map(|&g| f64::from((comp.get(g).copied().unwrap_or(0.0) / z).max(1e-12)).ln())
            .collect()
    };
    let log_null = log_table(&null_comp);
    let log_ceiling = log_table(&ceiling_comp);
    let genes = eval.genes().to_vec();

    for (lb, ub) in create_jobs(ntot, 0, Some(minibatch_size)) {
        let csc = data_vec.read_columns_csc(lb..ub)?;
        let n_block = csc.ncols();
        let recon_dn = recon.block(data_vec, lb, n_block);

        // Scored in parallel across the block. Each cell is independent and the
        // dominant term is two rank sorts over the evaluation axis — which by
        // default is every model feature. The shared accumulators are folded
        // afterwards in column order, so the result does not depend on how rayon
        // schedules the work.
        let scored: Vec<ScoredCell> = (0..n_block)
            .into_par_iter()
            .map(|j| {
                // Observed, densified onto the MODEL's gene axis so both sides
                // are indexed the same way. Genes the model lacks are not scored.
                let mut obs = vec![0f32; d_train];
                let col = csc.col(j);
                for (&row_new, &val) in col.row_indices().iter().zip(col.values()) {
                    let row_train = match gene_remap {
                        Some(rm) => rm.new_to_train[row_new],
                        None => Some(row_new),
                    };
                    if let Some(r) = row_train {
                        obs[r] += val;
                    }
                }
                let recon_z: f32 = genes
                    .iter()
                    .map(|&g| recon_dn[(g, j)])
                    .sum::<f32>()
                    .max(1e-12);

                let (mut count, mut model, mut null, mut ceiling) = (0f64, 0f64, 0f64, 0f64);
                for (i, &g) in genes.iter().enumerate() {
                    let x = f64::from(obs[g]);
                    if x <= 0.0 {
                        continue;
                    }
                    count += x;
                    model += x * f64::from((recon_dn[(g, j)] / recon_z).max(1e-12)).ln();
                    null += x * log_null[i];
                    ceiling += x * log_ceiling[i];
                }

                // Predicted COUNTS, not the raw rate: the composition over the
                // evaluation genes, put back on the cell's own depth.
                // `pearson_log1p` is not scale-invariant — log1p(c·p) is not an
                // affine function of log1p(p), and the zeros anchor the low end —
                // so feeding it a rate understates a good prediction and, worse,
                // understates it by a different amount per family, since each
                // one's rate carries its own arbitrary scale. On the count scale
                // a perfect prediction scores 1.
                let scale = count as f32 / recon_z;
                let o: Vec<f32> = genes.iter().map(|&g| obs[g]).collect();
                let p: Vec<f32> = genes.iter().map(|&g| recon_dn[(g, j)] * scale).collect();
                let agree = CellAgreement {
                    spearman: spearman(&o, &p),
                    pearson_log1p: pearson_log1p(&o, &p),
                };
                let llik = CellLlik {
                    model: model as f32,
                    null: null as f32,
                    count: count as f32,
                };
                (agree, llik, ceiling, o, p)
            })
            .collect();

        for (agree, llik, ceiling, o, p) in scored {
            eval_count += f64::from(llik.count);
            model_llik += f64::from(llik.model);
            null_llik += f64::from(llik.null);
            ceiling_llik += ceiling;
            per_cell.push(agree);
            per_cell_llik.push(llik);
            eval.keep(&o, &p);
        }
        bar.inc(1);
    }
    bar.finish_and_clear();
    let n_eval_genes = eval.genes().len();
    let per_gene = eval.per_gene();
    Ok(EvalOutcome {
        ceiling_llik,
        train_gene_names: Vec::new(),
        per_cell,
        per_cell_llik,
        per_gene,
        model_llik,
        null_llik,
        eval_count,
        n_eval_genes,
    })
}

/// One cell's likelihood on the scored gene set, model and floor.
///
/// Both are multinomials over the *same* genes with the same normaliser, so
/// their difference is the only quantity that means anything on its own: the
/// absolute value is set by how many genes the multinomial spreads over.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CellLlik {
    pub model: f32,
    pub null: f32,
    /// Observed counts on the scored genes — the denominator, and NOT the
    /// backend's `total`, which spans every gene.
    pub count: f32,
}

/// Everything one evaluation pass produces.
pub(crate) struct EvalOutcome {
    /// The model's feature axis, so a caller can name the scored genes. The
    /// indices in `per_gene` are training-axis positions, not query-axis ones —
    /// `obs` is densified onto the model's axis before scoring, so anything that
    /// labels those indices has to use the training names.
    pub train_gene_names: Vec<Box<str>>,
    pub per_cell: Vec<CellAgreement>,
    /// The comparable likelihood, per cell. Kept alongside the aggregate because
    /// a benchmark reads a file, not a log line: the backend's own `llik` column
    /// is decoder-dependent and must not be compared across families, so without
    /// these the only cross-family likelihood would live in stdout.
    pub per_cell_llik: Vec<CellLlik>,
    /// `(gene, spearman, pearson_log1p, mean_observed)`; empty unless `keep_per_gene`.
    pub per_gene: Vec<(usize, f32, f32, f32)>,
    pub model_llik: f64,
    pub null_llik: f64,
    /// The test half's own composition — an upper reference, not a floor.
    pub ceiling_llik: f64,
    pub eval_count: f64,
    pub n_eval_genes: usize,
}

impl EvalOutcome {
    #[must_use]
    pub fn summary(&self) -> EvalSummary {
        let per_count = |v: f64| {
            if self.eval_count > 0.0 {
                (v / self.eval_count) as f32
            } else {
                f32::NAN
            }
        };
        EvalSummary {
            mean_cell_spearman: mean_finite(self.per_cell.iter().map(|c| c.spearman)),
            mean_cell_pearson: mean_finite(self.per_cell.iter().map(|c| c.pearson_log1p)),
            mean_gene_spearman: mean_finite(self.per_gene.iter().map(|g| g.1)),
            mean_gene_pearson: mean_finite(self.per_gene.iter().map(|g| g.2)),
            llik_per_count: per_count(self.model_llik),
            null_llik_per_count: per_count(self.null_llik),
            ceiling_llik_per_count: per_count(self.ceiling_llik),
            n_eval_genes: self.n_eval_genes,
        }
    }
}

/// Summary of one method on one test half.
pub(crate) struct EvalSummary {
    pub mean_cell_spearman: f32,
    pub mean_cell_pearson: f32,
    pub mean_gene_spearman: f32,
    pub mean_gene_pearson: f32,
    pub llik_per_count: f32,
    pub null_llik_per_count: f32,
    /// The test half's own composition — what a predictor that had seen the test
    /// marginal would score. Not a floor; an upper reference.
    pub ceiling_llik_per_count: f32,
    pub n_eval_genes: usize,
}

impl EvalSummary {
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "llik/count {:.4} vs training null {:.4} (gain {:+.4}), \
             test-marginal ceiling {:.4}; \
             per-cell spearman {:.3}, pearson(log1p) {:.3}; \
             per-gene spearman {:.3}, pearson(log1p) {:.3}; {} genes scored",
            self.llik_per_count,
            self.null_llik_per_count,
            self.llik_per_count - self.null_llik_per_count,
            self.ceiling_llik_per_count,
            self.mean_cell_spearman,
            self.mean_cell_pearson,
            self.mean_gene_spearman,
            self.mean_gene_pearson,
            self.n_eval_genes
        )
    }
}

/// Mean over the finite entries. A cell with no expressed evaluation gene, or a
/// gene constant across the test half, yields `NaN` from the correlation and is
/// skipped here rather than dragging the mean to zero.
#[must_use]
pub(crate) fn mean_finite(v: impl Iterator<Item = f32>) -> f32 {
    let (s, n) = v
        .filter(|x| x.is_finite())
        .fold((0f64, 0usize), |(s, n), x| (s + f64::from(x), n + 1));
    if n > 0 {
        (s / n as f64) as f32
    } else {
        f32::NAN
    }
}
