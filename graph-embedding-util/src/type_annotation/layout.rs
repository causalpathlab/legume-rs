//! 2D cell layout (UMAP + PHATE) and feature placement for projection-based
//! annotation. Split out of the parent `type_annotation` module.

use super::{AnnotateProjConfig, AnnotateProjOutputs, FEAT_PROJ_ALPHA};
use anyhow::{Context, Result};
use log::info;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::knn_graph::{self, KnnGraph};
use matrix_util::knn_match::ColumnDict;
use matrix_util::layout::{phate_layout_2d, project_cells_nystrom, project_via_knn, PhateArgs};
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::umap::Umap;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

// Cell layout (UMAP + PHATE)
//////////////////////////////

/// Leiden community detection over a prebuilt kNN graph (modularity
/// objective), returning compacted labels. Mirrors the tail of
/// `matrix_util::clustering::leiden_clustering` but consumes a graph we
/// already built rather than rebuilding it.
pub(super) fn leiden_from_graph(
    graph: &KnnGraph,
    n: usize,
    resolution: f64,
    seed: u64,
) -> Vec<usize> {
    let (network, total_edge_weight) = graph.to_leiden_network();
    let resolution_scaled = knn_graph::modularity_to_cpm_resolution(resolution, total_edge_weight);
    let mut labels = knn_graph::run_leiden(&network, n, resolution_scaled, Some(seed as usize));
    knn_graph::compact_labels(&mut labels);
    labels
}

/// UMAP SGD layout off the fuzzy-weighted cell kNN graph. Returns `[N×2]`
/// row-major coords. Uses the shared `matrix_util::umap` kernel (same as
/// `faba gem-plot`).
pub(super) fn umap_from_graph(graph: &KnnGraph, n: usize, epochs: usize, seed: u64) -> Vec<f32> {
    let fuzzy = graph.fuzzy_kernel_weights();
    let edges: Vec<(usize, usize, f32)> = graph
        .edges
        .par_iter()
        .zip(fuzzy.par_iter())
        .filter_map(|(&(i, j), &w)| (w > 0.0).then_some((i, j, w)))
        .collect();
    info!(
        "UMAP layout: {} nodes, {} edges, {} epochs",
        n,
        edges.len(),
        epochs
    );
    // Init in [-10, 10]² so the UMAP gradient clamp doesn't dominate scale
    // (mirrors `faba gem-plot` / `senna layout umap`).
    const INIT_SCALE: f32 = 10.0;
    let mut rng = SmallRng::seed_from_u64(seed);
    let init: Vec<f32> = (0..n * 2)
        .map(|_| (rng.random_range(0.0_f32..1.0) * 2.0 - 1.0) * INIT_SCALE)
        .collect();
    let umap = Umap {
        n_epochs: epochs,
        seed,
        ..Default::default()
    };
    umap.fit(&edges, n, &init)
}

/// PHATE cell layout with the auto-fallback scaling strategy:
/// - `n_cells <= phate_max_direct`: PHATE directly on every cell's e_cell.
/// - otherwise: reuse the **Leiden communities** (already computed for
///   coarsening) as landmarks — landmark = mean e_cell per community — run
///   PHATE on those centroids, then Nyström-project every cell. No extra
///   clustering cost; raise `--resolution` for finer communities (= more
///   landmarks → higher-resolution PHATE).
///
/// `cell_u` is `[N×H]` row-major, unit-normalized. `community`/`n_comm` are the
/// per-cell Leiden labels. Returns `[N×2]` row-major, or `None` when PHATE is
/// not feasible (too few landmark communities).
pub(super) fn phate_cells(
    cell_u: &[f32],
    n_cells: usize,
    h: usize,
    community: &[usize],
    n_comm: usize,
    cfg: &AnnotateProjConfig,
) -> Option<Vec<f32>> {
    let pargs = PhateArgs {
        t: cfg.phate_t,
        knn: cfg.phate_knn,
        alpha: cfg.phate_alpha,
        ..Default::default()
    };

    // Small N: PHATE directly on every cell.
    if n_cells <= cfg.phate_max_direct {
        info!("PHATE: direct on {n_cells} cells");
        let cell_mat = DMatrix::<f32>::from_row_iterator(n_cells, h, cell_u.iter().copied());
        return Some(mat_rows_to_flat(&phate_layout_2d(&cell_mat, &pargs)));
    }

    // Large N: Leiden communities as landmarks. PHATE needs ≥3 landmarks.
    if n_comm < 3 {
        log::warn!(
            "PHATE skipped: only {n_comm} Leiden communities (need ≥3 landmarks); raise --resolution"
        );
        return None;
    }
    info!("PHATE: {n_cells} cells → {n_comm} Leiden-community landmarks + Nyström");

    // Community centroid of e_cell (mean of unit vectors), re-normalized.
    let mut cent = vec![0f32; n_comm * h];
    let mut cnt = vec![0f32; n_comm];
    for (c, &lab) in community.iter().enumerate() {
        cnt[lab] += 1.0;
        let row = &cell_u[c * h..(c + 1) * h];
        for (d, &v) in row.iter().enumerate() {
            cent[lab * h + d] += v;
        }
    }
    for lab in 0..n_comm {
        let d = cnt[lab].max(1.0);
        let nrm = {
            let mut s = 0f32;
            for j in 0..h {
                cent[lab * h + j] /= d;
                s += cent[lab * h + j] * cent[lab * h + j];
            }
            s.sqrt().max(1e-8)
        };
        for j in 0..h {
            cent[lab * h + j] /= nrm;
        }
    }

    // PHATE on the community centroids ([n_comm × H]).
    let cent_mat = DMatrix::<f32>::from_row_iterator(n_comm, h, cent.iter().copied());
    let land_coords = phate_layout_2d(&cent_mat, &pargs); // [n_comm × 2]

    // Nyström: lift every cell using e_cell distances to the centroids.
    // Both query and landmarks are column-major (H × ·).
    let query_kn = DMatrix::<f32>::from_fn(h, n_cells, |r, c| cell_u[c * h + r]);
    let land_kp = DMatrix::<f32>::from_fn(h, n_comm, |r, c| cent[c * h + r]);
    let coords = project_cells_nystrom(
        &query_kn,
        &land_kp,
        &land_coords,
        cfg.phate_knn.max(1),
        cfg.phate_alpha,
    );
    Some(mat_rows_to_flat(&coords))
}

/// Flatten an `[n×2]` nalgebra matrix into row-major `Vec<f32>`.
fn mat_rows_to_flat(m: &DMatrix<f32>) -> Vec<f32> {
    let n = m.nrows();
    let mut v = vec![0f32; n * 2];
    for i in 0..n {
        v[i * 2] = m[(i, 0)];
        v[i * 2 + 1] = m[(i, 1)];
    }
    v
}

//////////////////////////////
// Layout + feature-placement outputs (part ii)
//////////////////////////////

/// Subtract `mu` from each row of a `[m×h]` matrix, then L2-normalize, into a
/// row-major `Vec<f32>`. Centering removes the embedding's dominant common-mode
/// direction (a single shared component that can dominate cosine in low-H gem
/// runs); without it the feature→cell cosine kNN collapses every feature onto
/// the central, highest-common-mode cells instead of its lineage's cells.
fn centered_normalized(m: &DMatrix<f32>, mu: &[f32]) -> Vec<f32> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut v = vec![0f32; r * c];
    v.par_chunks_mut(c.max(1)).enumerate().for_each(|(i, row)| {
        let mut s = 0f32;
        for (j, slot) in row.iter_mut().enumerate() {
            let x = m[(i, j)] - mu[j];
            *slot = x;
            s += x * x;
        }
        let nrm = s.sqrt();
        if nrm > 1e-8 {
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
    });
    v
}

/// As [`centered_normalized`] but for an already-flat row-major `[rows×h]`
/// source (e.g. the unit signature anchors `type_emb_ch` / `coarse_emb_kh`).
fn centered_normalized_flat(src: &[f32], rows: usize, h: usize, mu: &[f32]) -> Vec<f32> {
    let mut v = vec![0f32; rows * h];
    v.par_chunks_mut(h.max(1)).enumerate().for_each(|(i, row)| {
        let mut s = 0f32;
        for (j, slot) in row.iter_mut().enumerate() {
            let x = src[i * h + j] - mu[j];
            *slot = x;
            s += x * x;
        }
        let nrm = s.sqrt();
        if nrm > 1e-8 {
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
    });
    v
}

/// `[n×2]` row-major flat → nalgebra `DMatrix` (for use as Nyström/kNN
/// landmark coords).
fn flat_to_coords(flat: &[f32], n: usize) -> DMatrix<f32> {
    DMatrix::<f32>::from_fn(n, 2, |i, j| flat[i * 2 + j])
}

/// Write `{prefix}.cell_coords.parquet` (per-cell UMAP/PHATE + community) and
/// `{prefix}.feature_coords.parquet` (genes, full features, type anchors
/// placed on both layouts via feature→cell kNN in the H-dim embedding).
#[allow(clippy::too_many_arguments)]
pub(super) fn write_layout_outputs(
    out_prefix: &str,
    cell_names: &[Box<str>],
    cell_flat: &[f32],
    n: usize,
    h: usize,
    feature_emb: &DMatrix<f32>,
    gene_names: &[Box<str>],
    gene_emb: Option<(&DMatrix<f32>, &[Box<str>])>,
    type_names: &[Box<str>],
    res: &AnnotateProjOutputs,
    cfg: &AnnotateProjConfig,
) -> Result<()> {
    let umap = res.cell_umap.as_ref().expect("layout guard ensures Some");
    let nan = f32::NAN;

    // ---- per-cell coordinates ----
    let umap_1: Vec<f32> = (0..n).map(|i| umap[i * 2]).collect();
    let umap_2: Vec<f32> = (0..n).map(|i| umap[i * 2 + 1]).collect();
    let (phate_1, phate_2): (Vec<f32>, Vec<f32>) = match res.cell_phate.as_ref() {
        Some(p) => (
            (0..n).map(|i| p[i * 2]).collect(),
            (0..n).map(|i| p[i * 2 + 1]).collect(),
        ),
        None => (vec![nan; n], vec![nan; n]),
    };
    let community: Vec<i32> = res.community.iter().map(|&k| k as i32).collect();
    let cc_path = format!("{out_prefix}.cell_coords.parquet");
    write_named_table(
        &cc_path,
        "cell",
        cell_names,
        &[
            (Box::from("community"), Column::I32(&community)),
            (Box::from("umap_1"), Column::F32(&umap_1)),
            (Box::from("umap_2"), Column::F32(&umap_2)),
            (Box::from("phate_1"), Column::F32(&phate_1)),
            (Box::from("phate_2"), Column::F32(&phate_2)),
        ],
    )
    .with_context(|| format!("writing {cc_path}"))?;
    info!("wrote {cc_path}");

    // ---- feature placement via feature→cell kNN ----
    // Global mean cell vector — subtracted from cells AND features before the
    // cosine kNN to strip the common-mode cone (otherwise every feature's
    // nearest cells collapse onto the central cluster; see `centered_normalized`).
    let mut mu = vec![0f64; h];
    for c in 0..n {
        for (j, m) in mu.iter_mut().enumerate() {
            *m += cell_flat[c * h + j] as f64;
        }
    }
    let mu: Vec<f32> = mu.iter().map(|&x| (x / n.max(1) as f64) as f32).collect();

    // Index centered+normalized cells once; both layouts share it (cosine via
    // L2 + DistL2).
    let cell_cols = DMatrix::<f32>::from_fn(h, n, |r, c| cell_flat[c * h + r] - mu[r]);
    let cell_cols = {
        let mut m = cell_cols;
        for mut col in m.column_iter_mut() {
            let nrm = col.norm().max(1e-8);
            col /= nrm;
        }
        m
    };
    let dict = ColumnDict::<usize>::from_dmatrix(cell_cols, (0..n).collect());

    let umap_coords = flat_to_coords(umap, n);
    let phate_coords = res.cell_phate.as_ref().map(|p| flat_to_coords(p, n));

    // Accumulate (name, kind, u1, u2, p1, p2) across all feature sets.
    let mut names: Vec<Box<str>> = Vec::new();
    let mut kinds: Vec<Box<str>> = Vec::new();
    let mut u1: Vec<f32> = Vec::new();
    let mut u2: Vec<f32> = Vec::new();
    let mut p1: Vec<f32> = Vec::new();
    let mut p2: Vec<f32> = Vec::new();

    // Project a feature set given its row-major centered+normalized `[m×h]`.
    let place = |feat_flat: &[f32],
                 m: usize,
                 feat_names: &[Box<str>],
                 kind: &str,
                 names: &mut Vec<Box<str>>,
                 kinds: &mut Vec<Box<str>>,
                 u1: &mut Vec<f32>,
                 u2: &mut Vec<f32>,
                 p1: &mut Vec<f32>,
                 p2: &mut Vec<f32>| {
        if m == 0 {
            return;
        }
        let q = DMatrix::<f32>::from_fn(h, m, |r, c| feat_flat[c * h + r]);
        let uc = project_via_knn(&q, &dict, &umap_coords, cfg.feat_knn, FEAT_PROJ_ALPHA);
        let pc = phate_coords
            .as_ref()
            .map(|pco| project_via_knn(&q, &dict, pco, cfg.feat_knn, FEAT_PROJ_ALPHA));
        for i in 0..m {
            names.push(feat_names[i].clone());
            kinds.push(Box::from(kind));
            u1.push(uc[(i, 0)]);
            u2.push(uc[(i, 1)]);
            match &pc {
                Some(pco) => {
                    p1.push(pco[(i, 0)]);
                    p2.push(pco[(i, 1)]);
                }
                None => {
                    p1.push(nan);
                    p2.push(nan);
                }
            }
        }
    };

    // All feature sets are centered by `mu` and L2-normalized before kNN, the
    // same transform applied to the indexed cells.
    // genes (β_g)
    if let Some((ge, gn)) = gene_emb {
        let flat = centered_normalized(ge, &mu);
        place(
            &flat,
            ge.nrows(),
            gn,
            "gene",
            &mut names,
            &mut kinds,
            &mut u1,
            &mut u2,
            &mut p1,
            &mut p2,
        );
    }
    // full features (gene/modality/region)
    {
        let flat = centered_normalized(feature_emb, &mu);
        place(
            &flat,
            feature_emb.nrows(),
            gene_names,
            "feature",
            &mut names,
            &mut kinds,
            &mut u1,
            &mut u2,
            &mut p1,
            &mut p2,
        );
    }
    // fine type anchors (unit signatures → re-centered + re-normalized)
    {
        let flat = centered_normalized_flat(&res.type_emb_ch, type_names.len(), h, &mu);
        place(
            &flat,
            type_names.len(),
            type_names,
            "type_anchor",
            &mut names,
            &mut kinds,
            &mut u1,
            &mut u2,
            &mut p1,
            &mut p2,
        );
    }
    // coarse anchors
    {
        let flat = centered_normalized_flat(&res.coarse_emb_kh, res.coarse_names.len(), h, &mu);
        place(
            &flat,
            res.coarse_names.len(),
            &res.coarse_names,
            "coarse_anchor",
            &mut names,
            &mut kinds,
            &mut u1,
            &mut u2,
            &mut p1,
            &mut p2,
        );
    }

    let fc_path = format!("{out_prefix}.feature_coords.parquet");
    write_named_table(
        &fc_path,
        "feature",
        &names,
        &[
            (Box::from("kind"), Column::Str(&kinds)),
            (Box::from("umap_1"), Column::F32(&u1)),
            (Box::from("umap_2"), Column::F32(&u2)),
            (Box::from("phate_1"), Column::F32(&p1)),
            (Box::from("phate_2"), Column::F32(&p2)),
        ],
    )
    .with_context(|| format!("writing {fc_path}"))?;
    info!("wrote {fc_path} ({} placed features)", names.len());

    Ok(())
}
