//! Gene-axis alignment for `senna predict` on a bge model: the pure pieces
//! between the projector and the writer, so each is testable without a backend.
//!
//! The flow in `BgeEmbedding::score`:
//! 1. pass 1 — project every cell on the MATCHED genes (as before);
//! 2. cluster the pass-1 latents into pseudobulks and accumulate the new data's
//!    per-gene profiles over them ([`profiles_by_cluster`]);
//! 3. align the gene axes ([`graph_embedding_util::transfer::align_gene_axis`]):
//!    unseen genes get a membership-initialized row, then a moment-matched bias;
//! 4. optional pass 2 — re-project with the initialized genes as observations
//!    ([`union_remap`] maps the new rows onto the union axis);
//! 5. score the initialized genes on their OWN column ([`score_initialized`]):
//!    a prior's score is never mixed into the comparable per-gene score.

use crate::embed_common::Mat;
use nalgebra::DMatrix;
use std::collections::HashSet;

/// Rows of the NEW data that matched no training gene by name, excluding the
/// rows deliberately withheld by `--ablate-features` — those are model genes
/// being tested, not unseen ones. `new_to_train` must be the remap BEFORE the
/// hide pass zeroed the hidden rows.
pub(crate) fn unseen_rows(
    new_to_train: &[Option<usize>],
    hidden_rows: &HashSet<usize>,
) -> Vec<usize> {
    new_to_train
        .iter()
        .enumerate()
        .filter(|(n, t)| t.is_none() && !hidden_rows.contains(n))
        .map(|(n, _)| n)
        .collect()
}

/// Accumulate per-gene count profiles over pseudobulks defined by a cell
/// clustering into `profiles` (`[n_genes × n_clusters]`):
/// `profiles[g, s] += Σ_{c: labels[c] = s} x_cg`. `cells` yields
/// `(cell, feature rows, counts)` on the NEW data's row axis; call once per
/// column block.
pub(crate) fn profiles_by_cluster<'a>(
    profiles: &mut DMatrix<f32>,
    labels: &[usize],
    cells: impl Iterator<Item = (usize, &'a [usize], &'a [f32])>,
) {
    for (c, feats, counts) in cells {
        let s = labels[c];
        for (&f, &x) in feats.iter().zip(counts) {
            profiles[(f, s)] += x;
        }
    }
}

/// New-data row → union-axis index: a matched row keeps its training position,
/// an unseen row `unseen[i]` becomes `n_train + i`, and any other row (hidden,
/// or dropped) maps to `None`.
pub(crate) fn union_remap(
    new_to_train: &[Option<usize>],
    unseen: &[usize],
    n_train: usize,
) -> Vec<Option<usize>> {
    let mut out: Vec<Option<usize>> = new_to_train.to_vec();
    for (i, &row) in unseen.iter().enumerate() {
        out[row] = Some(n_train + i);
    }
    out
}

/// Per-cell score of the INITIALIZED genes, in the same multinomial-per-count
/// form as the comparable score, kept in its own column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InitScore {
    /// Observed counts on the initialized genes.
    pub count: f32,
    /// `Σ_g x_g log p_g / Σ_g x_g` with `p` the model's composition over the
    /// initialized genes; `NaN` when the cell has no counts on them.
    pub llik_per_count: f32,
    /// The same under `null_comp`.
    pub null_llik_per_count: f32,
}

/// `rows` / `bias` are the initialized genes' `[U × H]` / `[U]`; `theta` `[N × H]`
/// and `b_cell` `[N]` are the latents the rates are evaluated at; `obs[c]` lists
/// `(local initialized index, count)` for cell `c`; `null_comp` is a composition
/// over the `U` genes.
pub(crate) fn score_initialized(
    rows: &DMatrix<f32>,
    bias: &[f32],
    theta: &Mat,
    b_cell: &[f32],
    obs: &[Vec<(u32, f32)>],
    null_comp: &[f32],
) -> Vec<InitScore> {
    let (u, h) = (rows.nrows(), rows.ncols());
    let floor = f64::from(matrix_util::agreement::PROB_FLOOR);
    let log_null: Vec<f64> = null_comp
        .iter()
        .map(|&q| f64::from(q).max(floor).ln())
        .collect();
    (0..theta.nrows())
        .map(|c| {
            let count: f32 = obs[c].iter().map(|&(_, x)| x).sum();
            if count <= 0.0 {
                return InitScore {
                    count: 0.0,
                    llik_per_count: f32::NAN,
                    null_llik_per_count: f32::NAN,
                };
            }
            // Composition of the model's rates over the initialized genes; the
            // cell bias is a common factor and cancels.
            let logits: Vec<f64> = (0..u)
                .map(|g| {
                    let s: f32 = (0..h).map(|j| rows[(g, j)] * theta[(c, j)]).sum::<f32>()
                        + bias[g]
                        + b_cell[c];
                    f64::from(s)
                })
                .collect();
            let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lse = logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln() + m;
            let (mut ll, mut nl) = (0f64, 0f64);
            for &(g, x) in &obs[c] {
                let x = f64::from(x);
                ll += x * (logits[g as usize] - lse).max(floor.ln());
                nl += x * log_null[g as usize];
            }
            InitScore {
                count,
                llik_per_count: (ll / f64::from(count)) as f32,
                null_llik_per_count: (nl / f64::from(count)) as f32,
            }
        })
        .collect()
}

/// The alignment outputs of a bge `predict`, written only when the query carried
/// genes the model never saw:
///
/// * `{out}.gene_alignment.parquet` — one row per union gene: status, best
///   profile similarity, whether the diffuse prior was used, the neighbours' names,
///   and the bias (trained or moment-matched). This is the provenance every reader
///   of an initialized row is entitled to.
/// * `{out}.init_genes.parquet` — per cell: counts on the initialized genes and
///   their multinomial nats per count against the query's own composition, in
///   their own columns, never mixed into `predictive.parquet`.
/// * `{out}.gene_rates.parquet` (opt-in) — per cell, the predicted Poisson rate of
///   every missing and initialized gene, `exp(ρ_g·θ_c + a_g + b_c)`.
pub(crate) fn write_init_outputs(
    out: &str,
    model: &super::score::BgeEmbedding,
    fit: &super::score::BgeFit,
    emit_rates: bool,
) -> anyhow::Result<()> {
    use graph_embedding_util::transfer::GeneStatus;
    use log::info;
    use matrix_util::parquet::{write_named_table, Column};
    use matrix_util::traits::IoOps;

    let Some(init) = fit.init.as_ref() else {
        return Ok(());
    };
    let al = &init.alignment;
    let n_union = al.n_union();
    let names: Vec<Box<str>> = (0..n_union)
        .map(|g| match (al.union_to_train[g], al.union_to_new[g]) {
            (Some(t), _) => model.gene_names[t].clone(),
            (None, Some(n)) => init.new_gene_names[n].clone(),
            (None, None) => Box::from("?"),
        })
        .collect();
    let status: Vec<Box<str>> = al
        .status
        .iter()
        .map(|s| {
            Box::from(match s {
                GeneStatus::Matched => "matched",
                GeneStatus::Missing => "missing",
                GeneStatus::Initialized => "initialized",
                GeneStatus::Dropped => "dropped",
            })
        })
        .collect();
    let similarity: Vec<f32> = al
        .provenance
        .iter()
        .map(|p| p.as_ref().map_or(f32::NAN, |p| p.best_similarity))
        .collect();
    let diffuse: Vec<i32> = al
        .provenance
        .iter()
        .map(|p| i32::from(p.as_ref().is_some_and(|p| p.diffuse)))
        .collect();
    let neighbours: Vec<Box<str>> = al
        .provenance
        .iter()
        .map(|p| {
            Box::from(
                p.as_ref()
                    .map(|p| {
                        p.neighbours
                            .iter()
                            .map(|&t| model.gene_names[t].to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .unwrap_or_default()
                    .as_str(),
            )
        })
        .collect();
    let path = format!("{out}.gene_alignment.parquet");
    write_named_table(
        &path,
        "gene",
        &names,
        &[
            (Box::from("status"), Column::Str(&status)),
            (Box::from("best_similarity"), Column::F32(&similarity)),
            (Box::from("diffuse"), Column::I32(&diffuse)),
            (Box::from("neighbours"), Column::Str(&neighbours)),
            (Box::from("bias"), Column::F32(&al.bias)),
        ],
    )?;
    info!("Wrote {path}");

    let cells = fit.data_vec.column_names()?;
    let n = init.scores.len();
    let mut sc = Mat::zeros(n, 3);
    for (i, s) in init.scores.iter().enumerate() {
        sc[(i, 0)] = s.count;
        sc[(i, 1)] = s.llik_per_count;
        sc[(i, 2)] = s.null_llik_per_count;
    }
    let path = format!("{out}.init_genes.parquet");
    sc.to_parquet_with_names(
        &path,
        (Some(&cells), Some("cell")),
        Some(&[
            Box::from("init_count"),
            Box::from("init_llik_per_count"),
            Box::from("init_null_llik_per_count"),
        ]),
    )?;
    let scored: Vec<&super::transfer::InitScore> =
        init.scores.iter().filter(|s| s.count > 0.0).collect();
    let mean = |f: &dyn Fn(&InitScore) -> f32| {
        scored.iter().map(|s| f(s)).sum::<f32>() / scored.len().max(1) as f32
    };
    info!(
        "Wrote {path}: {} initialized genes over {} pseudobulks{}, {} cells scored, llik/count {:.4} vs query-composition null {:.4} (gain {:+.4})",
        init.unseen_rows.len(),
        init.n_clusters,
        if init.in_fit { " (observed in pass 2)" } else { "" },
        scored.len(),
        mean(&|s| s.llik_per_count),
        mean(&|s| s.null_llik_per_count),
        mean(&|s| s.llik_per_count - s.null_llik_per_count),
    );

    if emit_rates {
        let genes: Vec<usize> = (0..n_union)
            .filter(|&g| matches!(al.status[g], GeneStatus::Missing | GeneStatus::Initialized))
            .collect();
        let h = al.rows.ncols();
        let mut rates = Mat::zeros(n, genes.len());
        for (j, &g) in genes.iter().enumerate() {
            for c in 0..n {
                let s: f32 = (0..h)
                    .map(|k| al.rows[(g, k)] * fit.latent[(c, k)])
                    .sum::<f32>()
                    + al.bias[g]
                    + fit.b_cell[c];
                rates[(c, j)] = s.exp();
            }
        }
        let cols: Vec<Box<str>> = genes.iter().map(|&g| names[g].clone()).collect();
        let path = format!("{out}.gene_rates.parquet");
        rates.to_parquet_with_names(&path, (Some(&cells), Some("cell")), Some(&cols))?;
        info!("Wrote {path} ({} missing + initialized genes)", genes.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
