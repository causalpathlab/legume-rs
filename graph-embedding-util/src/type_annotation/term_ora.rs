//! Firm projection annotation by **term over-representation within cell
//! clusters**.
//!
//! The soft path ([`super::annotate_by_projection`]) reports a per-cell cosine
//! z against a single competitive label-shuffle null, with FDR off. This path
//! makes the call *firm*:
//!
//! 1. **Term centroid** `e_T = (Σ_g w_g·e_g) / Σ_g w_g` — the IDF-weighted,
//!    **un-normalized** mean of a type's marker feature embeddings (a prototype
//!    in the embedding space; L2-norm would discard the position the Euclidean
//!    metric needs).
//! 2. **Nearest-centroid assignment** `t(c) = argmin_T ‖e_cell[c] − e_T‖₂` —
//!    every cell hard-assigned to its closest term, with the distance kept.
//! 3. **QC prune** — per term, drop cells whose distance to their assigned
//!    centroid is a high-side robust outlier (`> median + k·MAD`): cells that
//!    argmaxed a term but don't actually sit near it (ambient/doublet). They
//!    become `unassigned` and are excluded from the counts.
//! 4. **Cluster cells** — Leiden on the cell kNN graph (the embedding's own
//!    geometry, independent of the term labels).
//! 5. **Over-representation** — per (cluster K, term T) the count
//!    `a = #{c∈K : t(c)=T}` is tested against the hypergeometric null with
//!    fixed margins `(N, m_T, n_K)`; the statistic `−ln P(X≥a)` is **calibrated
//!    by permuting the per-cell labels** (pooled across clusters per term,
//!    relabeling-invariant). BH-FDR → q, FDR-sparse softmax → Q. The cluster's
//!    call is its top over-represented term; cells inherit it.
//! 6. **Calibration assessment** — analytic-vs-permutation agreement + a
//!    permutation-machinery sanity check, written to `null_calibration.tsv`.
//! 7. **Ontology (optional)** — feed the cluster × term p (and Q) to the shared
//!    generic TreeBH core for multi-resolution CL calling.

use super::markers::parse_and_match_markers;
use super::output::{write_label_tsvs, write_marker_embeddings};
use super::score::{argmax_rows, row_major};
use super::InputEmbeddings;
use anyhow::{Context, Result};
use log::info;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::traits::IoOps;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::io::Write;

/// File-name suffixes (relative to `out_prefix`) the firm term-ORA path writes.
/// Kept explicit (never a glob) so a caller can erase a prior run without
/// touching sibling artifacts (the embedding, the manifest).
pub const TERM_ORA_OUTPUT_SUFFIXES: &[&str] = &[
    ".annot.parquet",
    ".membership.tsv",
    ".argmax.tsv",
    ".marker_embedding.parquet",
    ".cluster_term_p.parquet",
    ".cluster_term_q.parquet",
    ".cluster_term_softq.parquet",
    ".null_calibration.tsv",
    ".ontology_assignment.tsv",
    ".ontology_node_mass.parquet",
];

/// Tunables for [`annotate_embeddings_ora`].
pub struct TermOraConfig {
    /// k for the cosine cell kNN graph fed to Leiden.
    pub knn: usize,
    /// Leiden modularity resolution (higher → more, finer clusters).
    pub resolution: f64,
    /// Deterministic RNG seed (clustering + permutation null).
    pub seed: u64,
    /// Permutation draws calibrating the over-representation statistic.
    pub n_perm: usize,
    /// Prune outlier cell→term assignments (distance > median + `assign_mad`·MAD).
    pub assign_qc: bool,
    /// MAD multiplier for the assignment-distance outlier gate.
    pub assign_mad: f64,
    /// FDR α for the cluster call + Q sparsity (BH on the permutation p).
    pub fdr_alpha: f32,
    /// Softmax temperature when building the row-normalized Q over significant terms.
    pub q_temperature: f32,
    /// Cell Ontology OBO path — runs the TreeBH ontology layer when set with `label_cl`.
    pub obo: Option<String>,
    /// Curated `label<TAB>CL:id` map (paired with `obo`).
    pub label_cl: Option<String>,
    /// TreeBH per-level selective-FDR target.
    pub ontology_fdr_q: f64,
    /// Benjamini–Yekutieli within ontology families (any dependence).
    pub ontology_by: bool,
}

impl Default for TermOraConfig {
    fn default() -> Self {
        Self {
            knn: 30,
            resolution: 1.0,
            seed: 42,
            n_perm: 500,
            assign_qc: true,
            assign_mad: 2.5,
            fdr_alpha: 0.1,
            q_temperature: 1.0,
            obo: None,
            label_cl: None,
            ontology_fdr_q: 0.1,
            ontology_by: false,
        }
    }
}

const UNASSIGNED: usize = usize::MAX;

/// Per-community (cluster / MST-node) firm call returned by
/// [`annotate_with_communities`], so a caller (e.g. `faba lineage --markers`) can name
/// each trajectory node without re-reading the parquet.
pub struct CommunityCalls {
    /// Called cell type per community (or `"unassigned"`), length `n_comm`.
    pub labels: Vec<Box<str>>,
    /// Confidence of each community's call (the FDR-sparse softmax `Q`; `0` when
    /// unassigned), length `n_comm`.
    pub confidence: Vec<f32>,
}

/// Name of a term index, mapping the [`UNASSIGNED`] sentinel to `"unassigned"`.
fn label_of(t: usize, type_names: &[Box<str>]) -> Box<str> {
    if t == UNASSIGNED {
        Box::from("unassigned")
    } else {
        type_names[t].clone()
    }
}

/// End-to-end firm annotation from in-memory embeddings, clustering cells with **Leiden**
/// over their own cosine kNN graph, then delegating to [`annotate_with_communities`].
/// See the module docs for the pipeline. Writes the `{out_prefix}.*` artifacts.
pub fn annotate_embeddings_ora(
    input: &InputEmbeddings<'_>,
    markers_path: &str,
    out_prefix: &str,
    use_idf: bool,
    cfg: &TermOraConfig,
) -> Result<()> {
    let n = input.cell_emb.nrows();
    let h = input.cell_emb.ncols();
    anyhow::ensure!(n >= 2, "term-ORA needs ≥ 2 cells, found {n}");
    // Communities are Leiden over the cell kNN graph — the embedding's own geometry,
    // independent of the term labels (module docs step 4).
    let cell_flat = row_major(input.cell_emb);
    let community = cluster_cells(&cell_flat, n, h, cfg)?;
    let n_comm = community.iter().copied().max().map_or(0, |m| m + 1).max(1);
    info!(
        "clustered cells into {n_comm} communities (knn={}, res={})",
        cfg.knn, cfg.resolution
    );
    annotate_with_communities(
        input,
        markers_path,
        out_prefix,
        use_idf,
        &community,
        n_comm,
        cfg,
    )?;
    Ok(())
}

/// Firm annotation given an **externally-supplied** cell clustering (`community[i]` =
/// cell `i`'s group id, `n_comm` groups) rather than Leiden. Runs the shared pipeline —
/// term centroids, nearest-centroid `fine_label`, per-term QC, then cluster × term
/// over-representation + permutation calibration over the *given* grouping — and writes
/// every `{out_prefix}.*` artifact. [`annotate_embeddings_ora`] wraps this with Leiden;
/// `faba lineage --markers` passes the MST-node clustering, so each trajectory node gets
/// the same permutation-calibrated call.
pub fn annotate_with_communities(
    input: &InputEmbeddings<'_>,
    markers_path: &str,
    out_prefix: &str,
    use_idf: bool,
    community: &[usize],
    n_comm: usize,
    cfg: &TermOraConfig,
) -> Result<CommunityCalls> {
    anyhow::ensure!(
        cfg.obo.is_some() == cfg.label_cl.is_some(),
        "--obo and --label-cl must be given together to run the ontology layer (got only one)"
    );
    let &InputEmbeddings {
        feature_emb,
        gene_names,
        cell_emb,
        cell_names,
    } = input;
    let g = feature_emb.nrows();
    let h = feature_emb.ncols();
    let n = cell_emb.nrows();
    anyhow::ensure!(
        cell_emb.ncols() == h,
        "embedding dim mismatch: features H={h}, cells H={}",
        cell_emb.ncols()
    );
    anyhow::ensure!(gene_names.len() == g, "gene_names len != feature rows");
    anyhow::ensure!(cell_names.len() == n, "cell_names len != cell rows");
    anyhow::ensure!(n >= 2, "term-ORA needs ≥ 2 cells, found {n}");
    anyhow::ensure!(
        community.len() == n,
        "community len {} != cell rows {n}",
        community.len()
    );
    anyhow::ensure!(n_comm >= 1, "need ≥ 1 community, got {n_comm}");
    info!("term-ORA: β [{g} × {h}], cells [{n} × {h}], {n_comm} group(s)");

    let (type_names, type_markers) = parse_and_match_markers(markers_path, gene_names, use_idf)?;
    let c = type_names.len();
    anyhow::ensure!(
        c >= 2,
        "need ≥ 2 cell types with matched markers, found {c}"
    );
    info!(
        "markers: {c} types, {} matched (gene,type) entries",
        type_markers.iter().map(Vec::len).sum::<usize>()
    );
    write_marker_embeddings(
        out_prefix,
        feature_emb,
        gene_names,
        &type_names,
        &type_markers,
        h,
    )?;

    // ----- 1. term centroids (un-normalized, IDF-weighted mean) -----
    let beta_flat = row_major(feature_emb);
    let centroids = term_centroids(&beta_flat, g, &type_markers, h); // [c × h] row-major

    // ----- 2. nearest-centroid assignment (Euclidean) -----
    let cell_flat = row_major(cell_emb);
    let (mut assign, dist) = assign_nearest(&cell_flat, n, &centroids, c, h);

    // ----- 3. QC: prune high-distance outliers per assigned term -----
    let mut n_outliers = 0usize;
    if cfg.assign_qc {
        n_outliers = prune_outliers(&mut assign, &dist, c, cfg.assign_mad);
    }
    let n_assigned = assign.iter().filter(|&&t| t != UNASSIGNED).count();
    info!("assignment: {n_assigned}/{n} cells assigned ({n_outliers} pruned as distance outliers)");
    anyhow::ensure!(
        n_assigned >= 2,
        "after QC only {n_assigned} cells remain assigned — loosen --assign-mad or check markers"
    );

    // ----- 5. cluster × term over-representation + permutation calibration -----
    let ora = cluster_term_ora(&assign, community, n_comm, c, cfg);

    // ----- 6. cluster calls → per-cell firm labels -----
    // Each cluster's call is its top over-represented term, kept only if
    // FDR-significant; otherwise the cluster stays unassigned.
    let top = argmax_rows(&ora.stat, n_comm, c);
    let cluster_label: Vec<usize> = (0..n_comm)
        .map(|k| {
            let best = top[k];
            if ora.q[k * c + best] < cfg.fdr_alpha {
                best
            } else {
                UNASSIGNED
            }
        })
        .collect();
    let coarse_label: Vec<Box<str>> = (0..n)
        .map(|i| label_of(cluster_label[community[i]], &type_names))
        .collect();
    let coarse_conf: Vec<f32> = (0..n)
        .map(|i| {
            let k = community[i];
            match cluster_label[k] {
                UNASSIGNED => 0.0,
                t => ora.q_soft[k * c + t],
            }
        })
        .collect();

    // ----- outputs -----
    let comm_names: Vec<Box<str>> = (0..n_comm)
        .map(|k| format!("K{k}").into_boxed_str())
        .collect();
    write_annot_parquet(
        out_prefix,
        cell_names,
        community,
        &coarse_label,
        &assign,
        &dist,
        &type_names,
        &ora,
        &cluster_label,
    )?;
    // membership.tsv + argmax.tsv on the firm (cluster-driven) label, the shared
    // contract `gem-summary` / `data-beans stat -g` consume.
    write_label_tsvs(out_prefix, cell_names, &coarse_label, &coarse_conf)?;
    write_cluster_term_matrices(out_prefix, &comm_names, &type_names, &ora)?;
    write_calibration(out_prefix, &ora, n_assigned, n_outliers)?;
    log_cluster_calls(&cluster_label, &type_names, community, n_comm);

    // ----- 7. optional ontology (TreeBH over the cluster × term matrix) -----
    if let (Some(obo), Some(label_cl)) = (cfg.obo.as_deref(), cfg.label_cl.as_deref()) {
        run_ontology(
            out_prefix,
            obo,
            label_cl,
            &comm_names,
            &type_names,
            &ora,
            cfg,
        )?;
    }

    // Per-community calls, so a trajectory caller can name each node directly.
    let comm_calls = CommunityCalls {
        labels: (0..n_comm)
            .map(|k| label_of(cluster_label[k], &type_names))
            .collect(),
        confidence: (0..n_comm)
            .map(|k| match cluster_label[k] {
                UNASSIGNED => 0.0,
                t => ora.q_soft[k * c + t],
            })
            .collect(),
    };
    Ok(comm_calls)
}

//////////////////////////////
// 1–2. centroids + assignment
//////////////////////////////

/// `[c × h]` row-major IDF-weighted mean of each type's marker feature
/// embeddings — the **un-normalized** centroid (the Euclidean prototype). Empty
/// types get a zero row.
fn term_centroids(
    feature_emb: &[f32],
    n_features: usize,
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
) -> Vec<f32> {
    let c = type_markers.len();
    let mut out = vec![0f32; c * h];
    out.par_chunks_mut(h)
        .zip(type_markers.par_iter())
        .for_each(|(row, markers)| {
            let mut wsum = 0f32;
            for &(gi, w) in markers {
                let gi = gi as usize;
                if gi >= n_features {
                    continue;
                }
                wsum += w;
                let ef = &feature_emb[gi * h..(gi + 1) * h];
                for (r, &e) in row.iter_mut().zip(ef) {
                    *r += w * e;
                }
            }
            if wsum > 0.0 {
                for v in row.iter_mut() {
                    *v /= wsum;
                }
            }
        });
    out
}

/// Nearest-centroid assignment by squared Euclidean distance. Returns
/// `(assign[n], dist[n])` where `dist` is the Euclidean distance to the assigned
/// centroid. A type with an all-zero (no-marker) centroid can still win only if
/// it is genuinely closest; QC downstream prunes poor fits.
fn assign_nearest(
    cell_flat: &[f32],
    n: usize,
    centroids: &[f32],
    c: usize,
    h: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut assign = vec![0usize; n];
    let mut dist = vec![0f32; n];
    assign
        .par_iter_mut()
        .zip(dist.par_iter_mut())
        .enumerate()
        .for_each(|(i, (a, d))| {
            let cell = &cell_flat[i * h..(i + 1) * h];
            let mut best = 0usize;
            let mut best_d2 = f32::INFINITY;
            for t in 0..c {
                let ct = &centroids[t * h..(t + 1) * h];
                let mut s = 0f32;
                for (x, y) in cell.iter().zip(ct) {
                    let diff = x - y;
                    s += diff * diff;
                }
                if s < best_d2 {
                    best_d2 = s;
                    best = t;
                }
            }
            *a = best;
            *d = best_d2.max(0.0).sqrt();
        });
    (assign, dist)
}

/// Mark `assign[c] = UNASSIGNED` for cells whose distance to their assigned
/// centroid is a high-side robust outlier (`> median + k·MAD`) within that term
/// — the shared `data_beans::qc_lib` robust-band idiom. Terms with < 3 assigned
/// cells are left intact (too few to define a band). Returns the number pruned.
fn prune_outliers(assign: &mut [usize], dist: &[f32], c: usize, k: f64) -> usize {
    use data_beans::qc_lib::{robust_outlier_keep, Tail};
    // Per-term cell indices (post-assignment).
    let mut per_term: Vec<Vec<usize>> = vec![Vec::new(); c];
    for (i, &t) in assign.iter().enumerate() {
        if t != UNASSIGNED {
            per_term[t].push(i);
        }
    }
    let mut pruned = 0usize;
    for cells in &per_term {
        if cells.len() < 3 {
            continue; // too few to define outliers
        }
        let dists: Vec<f32> = cells.iter().map(|&i| dist[i]).collect();
        let keep = robust_outlier_keep(&dists, k as f32, Tail::Upper, false, None);
        for (&i, &keep_i) in cells.iter().zip(&keep) {
            if !keep_i {
                assign[i] = UNASSIGNED;
                pruned += 1;
            }
        }
    }
    pruned
}

//////////////////////////////
// 4. clustering
//////////////////////////////

/// Leiden communities over a cosine cell kNN graph (cells L2-normalized for the
/// graph; gem `e_cell` is already unit, so this matches the assignment geometry).
fn cluster_cells(cell_flat: &[f32], n: usize, h: usize, cfg: &TermOraConfig) -> Result<Vec<usize>> {
    let mut cell_u = cell_flat.to_vec();
    super::score::l2_normalize_rows(&mut cell_u, n, h);
    let cell_mat = DMatrix::<f32>::from_row_iterator(n, h, cell_u.iter().copied());
    let knn = cfg.knn.clamp(1, n - 1);
    let graph = KnnGraph::from_rows(
        &cell_mat,
        KnnGraphArgs {
            knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;
    Ok(super::layout::leiden_from_graph(
        &graph,
        n,
        cfg.resolution,
        cfg.seed,
    ))
}

//////////////////////////////
// 5. over-representation + calibration
//////////////////////////////

/// Cluster × term over-representation result. All `[n_comm × c]` matrices are
/// row-major (`[k*c + t]`).
struct OraResult {
    /// `−ln P(X≥a)` analytic hypergeometric statistic (larger = more enriched).
    stat: Vec<f32>,
    /// Permutation-calibrated p (pooled per term across clusters).
    p_perm: Vec<f32>,
    /// BH q of `p_perm`, per cluster row.
    q: Vec<f32>,
    /// FDR-sparse row-softmax Q over significant terms (confidence weights).
    q_soft: Vec<f32>,
    /// Calibration diagnostics.
    cal: Calibration,
}

struct Calibration {
    n_perm: usize,
    median_logratio: f64,
    frac_analytic_anticons: f64,
    lambda_perm: f64,
    ks_perm: f64,
    degenerate_frac: f64,
}

fn cluster_term_ora(
    assign: &[usize],
    community: &[usize],
    n_comm: usize,
    c: usize,
    cfg: &TermOraConfig,
) -> OraResult {
    let n = assign.len();
    // Assigned cells only (post-QC) feed the contingency.
    let assigned: Vec<usize> = (0..n).filter(|&i| assign[i] != UNASSIGNED).collect();
    let labels: Vec<usize> = assigned.iter().map(|&i| assign[i]).collect();
    let comms: Vec<usize> = assigned.iter().map(|&i| community[i]).collect();
    let n_tot = assigned.len();

    let count = contingency(&comms, &labels, n_comm, c);
    let n_k: Vec<usize> = (0..n_comm)
        .map(|k| (0..c).map(|t| count[k * c + t] as usize).sum())
        .collect();
    let m_t: Vec<usize> = (0..c)
        .map(|t| (0..n_comm).map(|k| count[k * c + t] as usize).sum())
        .collect();

    // Per-(K,T) hypergeometric SF table — margins fixed under permutation, so
    // each table is reused for the observed count and every permuted count. The
    // ln-factorials (pop = n_tot for every table) are precomputed once.
    let lnfact = ln_factorials(n_tot);
    let sf_tables: Vec<Vec<f64>> = (0..n_comm * c)
        .map(|kt| {
            let (k, t) = (kt / c, kt % c);
            hypergeom_sf_table(n_tot, m_t[t], n_k[k], &lnfact)
        })
        .collect();
    let sf_at = |k: usize, t: usize, a: usize| -> f64 {
        let tbl = &sf_tables[k * c + t];
        tbl.get(a)
            .copied()
            .unwrap_or(if tbl.is_empty() { 1.0 } else { 0.0 })
    };

    let mut p_analytic = vec![1f32; n_comm * c];
    let mut stat = vec![0f32; n_comm * c];
    for k in 0..n_comm {
        for t in 0..c {
            let p = sf_at(k, t, count[k * c + t] as usize).clamp(1e-12, 1.0);
            p_analytic[k * c + t] = p as f32;
            stat[k * c + t] = (-p.ln()) as f32;
        }
    }

    // ----- permutation null: pool stat across clusters per term -----
    let b = cfg.n_perm;
    let mut null_pool: Vec<Vec<f32>> = vec![Vec::with_capacity(b * n_comm); c];
    if b > 0 && n_tot >= 2 {
        let mut perm = labels.clone();
        let mut rng = SmallRng::seed_from_u64(cfg.seed ^ 0x5eed_0a4a);
        for _ in 0..b {
            perm.shuffle(&mut rng);
            let cnt = contingency(&comms, &perm, n_comm, c);
            for k in 0..n_comm {
                for t in 0..c {
                    let p = sf_at(k, t, cnt[k * c + t] as usize).clamp(1e-12, 1.0);
                    null_pool[t].push((-p.ln()) as f32);
                }
            }
        }
    }
    for pool in &mut null_pool {
        pool.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    // permutation p per (K,T): fraction of the term's pooled null ≥ observed.
    let mut p_perm = vec![1f32; n_comm * c];
    for k in 0..n_comm {
        for t in 0..c {
            let pool = &null_pool[t];
            p_perm[k * c + t] = if pool.is_empty() {
                p_analytic[k * c + t]
            } else {
                let obs = stat[k * c + t];
                let ge = pool.len() - lower_bound(pool, obs);
                (ge as f32 + 1.0) / (pool.len() as f32 + 1.0)
            };
        }
    }

    // BH q per cluster row on the permutation p.
    let mut q = vec![1f32; n_comm * c];
    for k in 0..n_comm {
        let row: Vec<f32> = (0..c).map(|t| p_perm[k * c + t]).collect();
        let row_q = enrichment::bh_fdr(&row);
        for t in 0..c {
            q[k * c + t] = row_q[t];
        }
    }

    // FDR-sparse row-softmax Q (confidence weights): softmax of stat over terms
    // with q < α; zero elsewhere; uniform fallback if a row has no significant term.
    let q_soft = sparse_row_softmax(&stat, &q, n_comm, c, cfg.fdr_alpha, cfg.q_temperature);

    let cal = calibrate(&p_analytic, &p_perm, &null_pool, n_comm, c, b);

    OraResult {
        stat,
        p_perm,
        q,
        q_soft,
        cal,
    }
}

/// `[n_comm × c]` row-major contingency counts.
fn contingency(comms: &[usize], labels: &[usize], n_comm: usize, c: usize) -> Vec<u32> {
    let mut count = vec![0u32; n_comm * c];
    for (&k, &t) in comms.iter().zip(labels) {
        if k < n_comm && t < c {
            count[k * c + t] += 1;
        }
    }
    count
}

/// `ln(i!)` for `i ∈ 0..=n` (i.e. `ln_gamma(i+1)`), precomputed once so the
/// per-(cluster,term) SF tables share the factorials rather than recomputing
/// `ln_gamma` for every binomial coefficient.
fn ln_factorials(n: usize) -> Vec<f64> {
    use statrs::function::gamma::ln_gamma;
    (0..=n).map(|i| ln_gamma(i as f64 + 1.0)).collect()
}

/// Upper-tail hypergeometric SF table: `sf[a] = P(X ≥ a)` for a draw of `draws`
/// from a population of `pop` with `succ` successes, `a ∈ 0..=min(succ,draws)`.
/// Log-space PMF for numerical stability. `lnfact[i] = ln(i!)` must cover
/// `0..=pop`. Empty when `pop==0`.
fn hypergeom_sf_table(pop: usize, succ: usize, draws: usize, lnfact: &[f64]) -> Vec<f64> {
    if pop == 0 || succ == 0 || draws == 0 {
        // No successes or no draws ⇒ a is always 0; P(X≥0)=1, P(X≥1)=0.
        return vec![1.0];
    }
    let lnc = |a: usize, b: usize| -> f64 {
        if b > a {
            return f64::NEG_INFINITY;
        }
        lnfact[a] - lnfact[b] - lnfact[a - b]
    };
    let x_hi = succ.min(draws);
    let x_lo = (draws + succ).saturating_sub(pop);
    let ln_den = lnc(pop, draws);
    let mut pmf = vec![0f64; x_hi + 1];
    for (x, p) in pmf.iter_mut().enumerate().take(x_hi + 1).skip(x_lo) {
        *p = (lnc(succ, x) + lnc(pop - succ, draws - x) - ln_den).exp();
    }
    let mut sf = vec![0f64; x_hi + 1];
    let mut acc = 0f64;
    for a in (0..=x_hi).rev() {
        acc += pmf[a];
        sf[a] = acc.min(1.0);
    }
    sf
}

/// Index of the first element ≥ `x` in a sorted slice (count of strictly-smaller).
fn lower_bound(sorted: &[f32], x: f32) -> usize {
    sorted.partition_point(|&v| v < x)
}

/// Per cluster row: softmax of `stat/τ` over terms with `q < α`, zero elsewhere.
/// Rows with no significant term get a uniform distribution (so the argmax
/// confidence is still defined, but small).
fn sparse_row_softmax(
    stat: &[f32],
    q: &[f32],
    n_comm: usize,
    c: usize,
    alpha: f32,
    temperature: f32,
) -> Vec<f32> {
    let tau = temperature.max(1e-6);
    let mut out = vec![0f32; n_comm * c];
    for k in 0..n_comm {
        let sig: Vec<usize> = (0..c).filter(|&t| q[k * c + t] < alpha).collect();
        if sig.is_empty() {
            let u = 1.0 / c as f32;
            for t in 0..c {
                out[k * c + t] = u;
            }
            continue;
        }
        let mx = sig
            .iter()
            .map(|&t| stat[k * c + t])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0f32;
        for &t in &sig {
            let e = ((stat[k * c + t] - mx) / tau).exp();
            out[k * c + t] = e;
            s += e;
        }
        let s = s.max(1e-12);
        for &t in &sig {
            out[k * c + t] /= s;
        }
    }
    out
}

/// Discreteness-robust calibration of the analytic hypergeometric vs the
/// permutation null. `median_logratio = median log10(p_perm/p_analytic)` (≈0
/// calibrated; >0 ⇒ analytic anticonservative); `frac_analytic_anticons` =
/// share with `p_analytic < ½·p_perm`. Machinery sanity: `lambda_perm` /
/// `ks_perm` on leave-one-out null p (≈1 / small when unbiased).
fn calibrate(
    p_analytic: &[f32],
    p_perm: &[f32],
    null_pool: &[Vec<f32>],
    n_comm: usize,
    c: usize,
    b: usize,
) -> Calibration {
    // analytic-vs-perm agreement over observed (K,T).
    let mut logratios: Vec<f64> = Vec::with_capacity(n_comm * c);
    let mut anticons = 0usize;
    for kt in 0..n_comm * c {
        let pa = p_analytic[kt].max(1e-12) as f64;
        let pp = p_perm[kt].max(1e-12) as f64;
        logratios.push((pp / pa).log10());
        if (pa) < 0.5 * pp {
            anticons += 1;
        }
    }
    let median_logratio = {
        let mut v = logratios.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() {
            0.0
        } else {
            v[v.len() / 2]
        }
    };
    let frac_analytic_anticons = anticons as f64 / (n_comm * c).max(1) as f64;

    // Degenerate fraction: terms whose pooled null has no spread.
    let degenerate_terms = null_pool
        .iter()
        .filter(|pool| pool.is_empty() || pool.first() == pool.last())
        .count();
    let degenerate_frac = degenerate_terms as f64 / c.max(1) as f64;

    // Machinery sanity: leave-one-out empirical p of every pooled null value vs
    // its term's pool (uniform under an unbiased permutation null).
    let mut loo: Vec<f64> = Vec::new();
    for pool in null_pool {
        let m = pool.len();
        if m < 2 {
            continue;
        }
        for &x in pool {
            // strictly-greater + ties-excluding-self, +1 smoothing.
            let ge = m - lower_bound(pool, x);
            let p = ge as f64 / m as f64;
            loo.push(p.clamp(1e-12, 1.0));
        }
    }
    let (lambda_perm, ks_perm) = if loo.len() >= 8 {
        (lambda_from_p(&loo), ks_uniform(&loo))
    } else {
        (f64::NAN, f64::NAN)
    };

    Calibration {
        n_perm: b,
        median_logratio,
        frac_analytic_anticons,
        lambda_perm,
        ks_perm,
        degenerate_frac,
    }
}

/// Genomic-inflation-style λ for one-sided p-values: `median(z²)/0.4549`,
/// `z = Φ⁻¹(1−p)`. ≈1 when p ~ Uniform.
fn lambda_from_p(ps: &[f64]) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let std = Normal::new(0.0, 1.0).unwrap();
    let mut zsq: Vec<f64> = ps
        .iter()
        .map(|&p| {
            let z = std.inverse_cdf((1.0 - p).clamp(1e-12, 1.0 - 1e-12));
            z * z
        })
        .collect();
    zsq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = zsq[zsq.len() / 2];
    med / 0.4549364
}

/// Kolmogorov–Smirnov distance of `ps` from Uniform(0,1).
fn ks_uniform(ps: &[f64]) -> f64 {
    let mut v = ps.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nf = v.len() as f64;
    let mut d = 0f64;
    for (i, &p) in v.iter().enumerate() {
        let lo = (i as f64) / nf;
        let hi = (i as f64 + 1.0) / nf;
        d = d.max((p - lo).abs()).max((hi - p).abs());
    }
    d
}

//////////////////////////////
// outputs
//////////////////////////////

#[allow(clippy::too_many_arguments)]
fn write_annot_parquet(
    out_prefix: &str,
    cell_names: &[Box<str>],
    community: &[usize],
    coarse_label: &[Box<str>],
    assign: &[usize],
    dist: &[f32],
    type_names: &[Box<str>],
    ora: &OraResult,
    cluster_label: &[usize],
) -> Result<()> {
    let n = cell_names.len();
    let c = type_names.len();
    let comm_i32: Vec<i32> = community.iter().map(|&k| k as i32).collect();
    let fine_label: Vec<Box<str>> = assign.iter().map(|&t| label_of(t, type_names)).collect();
    let is_outlier: Vec<i32> = assign.iter().map(|&t| (t == UNASSIGNED) as i32).collect();
    // Per-cell coarse stats = the cluster's call entry.
    let coarse_p: Vec<f32> = (0..n)
        .map(|i| {
            let k = community[i];
            match cluster_label[k] {
                UNASSIGNED => f32::NAN,
                t => ora.p_perm[k * c + t],
            }
        })
        .collect();
    let coarse_q: Vec<f32> = (0..n)
        .map(|i| {
            let k = community[i];
            match cluster_label[k] {
                UNASSIGNED => f32::NAN,
                t => ora.q[k * c + t],
            }
        })
        .collect();
    let annot_path = format!("{out_prefix}.annot.parquet");
    write_named_table(
        &annot_path,
        "cell",
        cell_names,
        &[
            (Box::from("community"), Column::I32(&comm_i32)),
            (Box::from("coarse_label"), Column::Str(coarse_label)),
            (Box::from("coarse_p"), Column::F32(&coarse_p)),
            (Box::from("coarse_q"), Column::F32(&coarse_q)),
            (Box::from("fine_label"), Column::Str(&fine_label)),
            (Box::from("fine_distance"), Column::F32(dist)),
            (Box::from("is_outlier"), Column::I32(&is_outlier)),
        ],
    )
    .with_context(|| format!("writing {annot_path}"))?;
    info!("wrote {annot_path}");
    Ok(())
}

fn write_cluster_term_matrices(
    out_prefix: &str,
    comm_names: &[Box<str>],
    type_names: &[Box<str>],
    ora: &OraResult,
) -> Result<()> {
    let n_comm = comm_names.len();
    let c = type_names.len();
    let to_mat = |flat: &[f32]| DMatrix::<f32>::from_row_iterator(n_comm, c, flat.iter().copied());
    for (suffix, flat) in [
        ("cluster_term_p", &ora.p_perm),
        ("cluster_term_q", &ora.q),
        ("cluster_term_softq", &ora.q_soft),
    ] {
        let path = format!("{out_prefix}.{suffix}.parquet");
        to_mat(flat)
            .to_parquet_with_names(&path, (Some(comm_names), Some("cluster")), Some(type_names))
            .with_context(|| format!("writing {path}"))?;
    }
    info!("wrote {out_prefix}.cluster_term_{{p,q,Q}}.parquet ({n_comm} clusters × {c} terms)");
    Ok(())
}

fn write_calibration(
    out_prefix: &str,
    ora: &OraResult,
    n_assigned: usize,
    n_outliers: usize,
) -> Result<()> {
    let cal = &ora.cal;
    let path = format!("{out_prefix}.null_calibration.tsv");
    let mut f = std::fs::File::create(&path).with_context(|| format!("creating {path}"))?;
    writeln!(f, "metric\tvalue")?;
    writeln!(f, "n_perm\t{}", cal.n_perm)?;
    writeln!(f, "n_assigned\t{n_assigned}")?;
    writeln!(f, "n_outliers_pruned\t{n_outliers}")?;
    writeln!(
        f,
        "median_logratio_perm_over_analytic\t{:.4}",
        cal.median_logratio
    )?;
    writeln!(
        f,
        "frac_analytic_anticonservative\t{:.4}",
        cal.frac_analytic_anticons
    )?;
    writeln!(f, "lambda_perm\t{:.4}", cal.lambda_perm)?;
    writeln!(f, "ks_perm_uniform\t{:.4}", cal.ks_perm)?;
    writeln!(f, "degenerate_frac\t{:.4}", cal.degenerate_frac)?;
    info!("wrote {path}");

    // Console summary + warnings.
    eprintln!("\nNull calibration (permutation B={})", cal.n_perm);
    eprintln!(
        "  analytic vs permutation:  median log10(p_perm/p_analytic)={:.3}  anticonservative-frac={:.3}",
        cal.median_logratio, cal.frac_analytic_anticons
    );
    eprintln!(
        "  permutation machinery:    lambda_perm={:.3}  ks_uniform={:.3}  degenerate-frac={:.3}",
        cal.lambda_perm, cal.ks_perm, cal.degenerate_frac
    );
    if cal.median_logratio > 0.3 || cal.frac_analytic_anticons > 0.2 {
        log::warn!(
            "analytic hypergeometric looks anticonservative (median log-ratio {:.2}); \
             the reported p/q use the permutation null",
            cal.median_logratio
        );
    }
    if cal.lambda_perm.is_finite() && !(0.7..=1.4).contains(&cal.lambda_perm) {
        log::warn!(
            "permutation null lambda_perm={:.2} strays from 1 — raise --num-perm",
            cal.lambda_perm
        );
    }
    eprintln!();
    Ok(())
}

fn log_cluster_calls(
    cluster_label: &[usize],
    type_names: &[Box<str>],
    community: &[usize],
    n_comm: usize,
) {
    let mut sizes = vec![0usize; n_comm];
    for &k in community {
        sizes[k] += 1;
    }
    info!("cluster calls ({n_comm} clusters):");
    let mut order: Vec<usize> = (0..n_comm).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]));
    for k in order {
        let name = label_of(cluster_label[k], type_names);
        info!("  K{k:<3} {:6} cells  {name}", sizes[k]);
    }
}

//////////////////////////////
// 7. ontology
//////////////////////////////

fn run_ontology(
    out_prefix: &str,
    obo: &str,
    label_cl: &str,
    comm_names: &[Box<str>],
    type_names: &[Box<str>],
    ora: &OraResult,
    cfg: &TermOraConfig,
) -> Result<()> {
    let n_comm = comm_names.len();
    let c = type_names.len();
    // cluster × term permutation p as the ontology leaf evidence; Q as node mass.
    let p_mat = DMatrix::<f32>::from_row_iterator(n_comm, c, ora.p_perm.iter().copied());
    let q_mat = DMatrix::<f32>::from_row_iterator(n_comm, c, ora.q_soft.iter().copied());
    super::ontology_obo::annotate_ontology_from_obo(
        out_prefix,
        label_cl,
        obo,
        cfg.ontology_fdr_q,
        cfg.ontology_by,
        enrichment::OntologyScore::Pvalue(&p_mat),
        Some(&q_mat),
        comm_names,
        type_names,
    )?;
    Ok(())
}
