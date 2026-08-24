//! Descriptive per-batch LR edge scores (`pinto lra --edge-scores-only`).
//!
//! No test, no null: each row is a summary of what the spatial contacts of
//! one link community in one batch express, meant to be pivoted into a
//! batch × (pair, community) phenotype matrix and analyzed elsewhere. With
//! `ℓ(u) = log1p(x_L(u))`, `r(v) = log1p(x_R(v))`, over BOTH orientations
//! of the community's spatial edges inside the batch:
//!
//! ```text
//! product  = mean ℓ(u)·r(v)
//! coupling = mean ℓ(u)·r(v) − mean ℓ(u) · mean r(v)
//! ```
//!
//! The plain product is abundance-driven, dominated by marginal expression
//! and collinear across pairs sharing a gene; the centered coupling
//! isolates contact-level co-variation. Both ship, the analysis picks.
//! Both-orientation enumeration makes `score(L,R) = score(R,L)` structural.
//!
//! Deliberately plain `log1p` counts: inference-side precision weights
//! would make the phenotype opaque. Depth is a COVARIATE instead — each
//! row carries `n_edges` and the mean per-cell `log1p` depth of its
//! participating cells, and the downstream analysis should filter or
//! weight on them (no threshold is applied here, by policy).

use crate::lr_activity::fit::BATCH_LABEL_ALL;
use crate::lr_activity::orientation::CommunityStrata;
use crate::util::common::*;
use matrix_util::utils::generate_minibatch_intervals;
use rayon::prelude::*;

/// One (batch, community, pair) score row.
pub struct EdgeScoreRow {
    pub batch: Box<str>,
    pub community: u32,
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    /// Physical edges behind the score (each contributes two orientations).
    pub n_edges: u32,
    /// Mean `log1p` total count over the unique cells the edges touch.
    pub mean_log_depth: f32,
    pub product: f32,
    pub coupling: f32,
}

pub struct EdgeScoresInput<'a> {
    /// The tested (spatial) edge list `CommunityStrata` indexes into.
    pub edges: &'a [(usize, usize, u32, Option<Box<str>>)],
    pub strata: &'a CommunityStrata,
    /// `(ligand, receptor, gene_l, gene_r)` with GLOBAL gene ids.
    pub pairs: &'a [(Box<str>, Box<str>, usize, usize)],
    /// Global gene id → row of `x_lr`.
    pub gene_to_local: &'a HashMap<usize, usize>,
    /// Raw per-cell counts of the LR genes, `n_lr_genes × n_cells`.
    pub x_lr: &'a Mat,
    /// Per-cell `log1p` total count (all genes, not only the panel).
    pub log_depth: &'a [f32],
}

/// Compute every (batch, community, pair) row. Returns the rows and the
/// number of straddling edges dropped (endpoint batch labels differ; such
/// a contact belongs to no single batch and must not leak across two).
pub fn compute_edge_scores(input: &EdgeScoresInput<'_>) -> (Vec<EdgeScoreRow>, usize) {
    let EdgeScoresInput {
        edges,
        strata,
        pairs,
        gene_to_local,
        x_lr,
        log_depth,
    } = input;

    // Multi-batch iff any edge carries a label; with none on file the run
    // is single-batch and every edge belongs to the `all` pseudo-batch.
    let multi_batch = edges.iter().any(|e| e.3.is_some());

    let x_log = x_lr.map(|v| v.ln_1p());

    let mut rows: Vec<EdgeScoreRow> = Vec::new();
    let mut n_straddling = 0usize;
    for s in 0..strata.n_strata() {
        let community = strata.community(s);

        // Oriented instance endpoints, grouped by batch. BTreeMap so row
        // order is a function of the labels, not of hashing.
        let mut by_batch: std::collections::BTreeMap<Box<str>, (Vec<usize>, Vec<usize>)> =
            Default::default();
        for &(e, flipped) in strata.oriented(s) {
            let (i, j, _, ref b) = edges[e as usize];
            let label: Box<str> = if multi_batch {
                match b {
                    Some(b) => b.clone(),
                    None => {
                        // Once per edge, not per orientation.
                        if !flipped {
                            n_straddling += 1;
                        }
                        continue;
                    }
                }
            } else {
                BATCH_LABEL_ALL.into()
            };
            let (u, v) = if flipped { (j, i) } else { (i, j) };
            let slot = by_batch.entry(label).or_default();
            slot.0.push(u);
            slot.1.push(v);
        }

        for (batch, (us, vs)) in by_batch {
            let n = us.len();
            let unique: HashSet<usize> = us.iter().copied().collect();
            let mean_log_depth =
                unique.iter().map(|&c| log_depth[c]).sum::<f32>() / unique.len().max(1) as f32;

            let batch_rows: Vec<EdgeScoreRow> = pairs
                .par_iter()
                .map(|(ligand, receptor, gl, gr)| {
                    let li = gene_to_local[gl];
                    let ri = gene_to_local[gr];
                    let mut sp = 0f64;
                    let mut sl = 0f64;
                    let mut sr = 0f64;
                    for (&u, &v) in us.iter().zip(vs.iter()) {
                        let l = x_log[(li, u)] as f64;
                        let r = x_log[(ri, v)] as f64;
                        sp += l * r;
                        sl += l;
                        sr += r;
                    }
                    let nf = n as f64;
                    let product = sp / nf;
                    let coupling = product - (sl / nf) * (sr / nf);
                    EdgeScoreRow {
                        batch: batch.clone(),
                        community,
                        ligand: ligand.clone(),
                        receptor: receptor.clone(),
                        n_edges: (n / 2) as u32,
                        mean_log_depth,
                        product: product as f32,
                        coupling: coupling as f32,
                    }
                })
                .collect();
            rows.extend(batch_rows);
        }
    }
    (rows, n_straddling)
}

/// Write `{out}.lr_scores.parquet`, long format: one row per
/// (batch, community, ligand, receptor).
pub fn write_edge_scores(out_prefix: &str, rows: &[EdgeScoreRow]) -> anyhow::Result<()> {
    use matrix_util::parquet::{write_named_table, Column};

    let batch: Vec<Box<str>> = rows.iter().map(|r| r.batch.clone()).collect();
    let community: Vec<i32> = rows.iter().map(|r| r.community as i32).collect();
    let ligand: Vec<Box<str>> = rows.iter().map(|r| r.ligand.clone()).collect();
    let receptor: Vec<Box<str>> = rows.iter().map(|r| r.receptor.clone()).collect();
    let n_edges: Vec<i32> = rows.iter().map(|r| r.n_edges as i32).collect();
    let mean_log_depth: Vec<f32> = rows.iter().map(|r| r.mean_log_depth).collect();
    let product: Vec<f32> = rows.iter().map(|r| r.product).collect();
    let coupling: Vec<f32> = rows.iter().map(|r| r.coupling).collect();
    let row_names: Vec<Box<str>> = (0..rows.len())
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    write_named_table(
        &format!("{out_prefix}.lr_scores.parquet"),
        "row",
        &row_names,
        &[
            ("batch".into(), Column::Str(&batch)),
            ("community".into(), Column::I32(&community)),
            ("ligand".into(), Column::Str(&ligand)),
            ("receptor".into(), Column::Str(&receptor)),
            ("n_edges".into(), Column::I32(&n_edges)),
            ("mean_log_depth".into(), Column::F32(&mean_log_depth)),
            ("product".into(), Column::F32(&product)),
            ("coupling".into(), Column::F32(&coupling)),
        ],
    )
}

/// Per-cell `log1p` of the total count over ALL rows (the panel a pair is
/// scored on is a subset; depth is a property of the cell, not the panel).
pub fn per_cell_log1p_depth(
    data: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    let n_cells = data.num_columns();
    let jobs = generate_minibatch_intervals(n_cells, data.num_rows(), block_size);
    let mut depth = vec![0.0f32; n_cells];
    // Jobs partition the columns, so each writes a disjoint slice; a fold
    // would allocate a full-length accumulator per worker for no benefit.
    let totals: Vec<(usize, Vec<f32>)> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>)> {
            let x = data.read_columns_csc(lb..ub)?;
            let mut t = vec![0.0f32; ub - lb];
            for (col, tc) in t.iter_mut().enumerate() {
                *tc = x.col(col).values().iter().sum();
            }
            Ok((lb, t))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    for (lb, t) in totals {
        for (k, v) in t.into_iter().enumerate() {
            depth[lb + k] = v.ln_1p();
        }
    }
    Ok(depth)
}
