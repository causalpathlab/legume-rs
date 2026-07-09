//! 2D cell layout (UMAP + PHATE) and feature placement for projection-based
//! annotation. Split out of the parent `type_annotation` module.

use super::{AnnotateProjConfig, AnnotateProjOutputs, FEAT_PROJ_ALPHA};
use anyhow::{Context, Result};
use log::info;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::knn_graph::{self, KnnGraph};
use matrix_util::layout::{phate_layout_2d, project_cells_nystrom, PhateArgs};
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::umap::Umap;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

////////////////////////////////
// Cell layout (UMAP + PHATE) //
////////////////////////////////

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

//////////////////////////////////////////////////
// Layout + feature-placement outputs (part ii) //
//////////////////////////////////////////////////

/// `[n×2]` row-major flat → nalgebra `DMatrix` (for use as Nyström/kNN
/// landmark coords).
fn flat_to_coords(flat: &[f32], n: usize) -> DMatrix<f32> {
    DMatrix::<f32>::from_fn(n, 2, |i, j| flat[i * 2 + j])
}

/// Cap on landmark cells for the Nyström feature placement. Exact when the QC'd
/// cell count is ≤ this; above it, landmarks are subsampled (deterministic
/// stride) so the dense `O(m · n_land · h)` projection stays bounded on very
/// large runs. 16384 keeps every current rep/BM-scale run exact.
const FEAT_LANDMARK_CAP: usize = 16_384;

/// L2-normalize each column in place (→ unit vectors; near-zero columns left as
/// ~0). Makes the Nyström Euclidean kernel rank landmarks/queries by direction
/// (cosine) rather than by magnitude.
fn l2_normalize_columns(mat: &mut DMatrix<f32>) {
    for mut col in mat.column_iter_mut() {
        let nrm = col.norm().max(1e-12);
        col /= nrm;
    }
}

/// Inputs for [`write_layout_outputs`]: the per-cell layout source data plus
/// the feature embedding and co-embedded type/coarse anchors to place on it.
pub(super) struct LayoutInputs<'a> {
    pub out_prefix: &'a str,
    pub cell_names: &'a [Box<str>],
    /// Row-major `[n × h]` cell embedding.
    pub cell_flat: &'a [f32],
    pub n: usize,
    pub h: usize,
    pub feature_emb: &'a DMatrix<f32>,
    pub gene_names: &'a [Box<str>],
    /// Co-embedded `[C × H]` fine-type and `[K × H]` coarse anchors.
    pub type_co: &'a DMatrix<f32>,
    pub coarse_co: &'a DMatrix<f32>,
    pub type_names: &'a [Box<str>],
    pub res: &'a AnnotateProjOutputs,
    pub cfg: &'a AnnotateProjConfig,
}

/// Write `{prefix}.cell_coords.parquet` (per-cell UMAP/PHATE + community) and
/// `{prefix}.feature_coords.parquet` (genes, full features, type anchors
/// placed on both layouts via feature→cell kNN in the H-dim embedding).
pub(super) fn write_layout_outputs(inp: &LayoutInputs<'_>) -> Result<()> {
    let &LayoutInputs {
        out_prefix,
        cell_names,
        cell_flat,
        n,
        h,
        feature_emb,
        gene_names,
        type_co,
        coarse_co,
        type_names,
        res,
        cfg,
    } = inp;
    let umap = res.cell_umap.as_ref().expect("layout guard ensures Some");
    let nan = f32::NAN;

    //////////////////////////
    // per-cell coordinates //
    //////////////////////////
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

    /////////////////////////////////////////////////////////////////
    // feature + anchor placement: FIRM (follows the co-embedding) //
    /////////////////////////////////////////////////////////////////
    // Each feature/anchor set is already co-embedded onto the cell manifold
    // (genes via `feature_embedding`, type/coarse anchors via the returned
    // co-embed locations). Nyström-project each through the cell layout, using
    // the cells as landmarks (their 2D coords are known) — so a feature lands at
    // the weighted 2D position of its nearest cells.
    //
    // Geometry: landmark cells AND query rows are L2-normalized, so the Nyström
    // Euclidean kernel ranks by DIRECTION (cosine), not by the co-embed rows'
    // sub-unit norm (which would otherwise pull diffuse features toward dense
    // cells). NO common-mode centering — the co-embed already put each row on the
    // manifold, and centering is exactly what collapsed the raw anchors to one
    // spot. For very large N the landmarks are subsampled to FEAT_LANDMARK_CAP
    // (deterministic stride) so the dense `O(m · n_land · h)` projection stays
    // bounded; exact when N ≤ cap (every current rep/BM-scale run).
    let land_idx: Vec<usize> = if n > FEAT_LANDMARK_CAP {
        let stride = n.div_ceil(FEAT_LANDMARK_CAP);
        (0..n).step_by(stride).collect()
    } else {
        (0..n).collect()
    };
    let n_land = land_idx.len();
    let mut cell_cols = DMatrix::<f32>::from_fn(h, n_land, |r, c| cell_flat[land_idx[c] * h + r]);
    l2_normalize_columns(&mut cell_cols);
    let sub_coords = |flat: &[f32]| -> DMatrix<f32> {
        let sub: Vec<f32> = land_idx
            .iter()
            .flat_map(|&c| [flat[c * 2], flat[c * 2 + 1]])
            .collect();
        flat_to_coords(&sub, n_land)
    };
    let umap_coords = sub_coords(umap);
    let phate_coords = res.cell_phate.as_ref().map(|p| sub_coords(p.as_slice()));

    let mut names: Vec<Box<str>> = Vec::new();
    let mut kinds: Vec<Box<str>> = Vec::new();
    let mut u1: Vec<f32> = Vec::new();
    let mut u2: Vec<f32> = Vec::new();
    let mut p1: Vec<f32> = Vec::new();
    let mut p2: Vec<f32> = Vec::new();

    // Project a co-embed feature set `[m × h]` (rows already on the cell manifold).
    let mut place = |feat: &DMatrix<f32>, feat_names: &[Box<str>], kind: &str| {
        let m = feat.nrows();
        if m == 0 {
            return;
        }
        let mut q = DMatrix::<f32>::from_fn(h, m, |r, c| feat[(c, r)]); // [h × m]
        l2_normalize_columns(&mut q); // cosine geometry vs the unit-norm landmarks
        let uc = project_cells_nystrom(&q, &cell_cols, &umap_coords, cfg.feat_knn, FEAT_PROJ_ALPHA);
        let pc = phate_coords
            .as_ref()
            .map(|pco| project_cells_nystrom(&q, &cell_cols, pco, cfg.feat_knn, FEAT_PROJ_ALPHA));
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

    place(feature_emb, gene_names, "feature"); // co-embed genes
    place(type_co, type_names, "type_anchor"); // co-embed fine-type locations
    place(coarse_co, &res.coarse_names, "coarse_anchor"); // co-embed coarse locations

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
