//! The 2D layouts written for plotting: UMAP, PHATE, and the velocity warp.

use anyhow::Result;
use log::info;

use matrix_util::branching::Branching;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::layout::{phate_layout_2d, project_cells_nystrom, PhateArgs};
use matrix_util::principal_curve::PrincipalCurves;
use matrix_util::principal_graph::kmeans_centroids_seeded;
use std::collections::HashMap;

use super::args::*;
use super::input::apply_geometry;
use super::velocity_grid::*;
use super::write::*;
use crate::lineage::orient::{undirected, EdgeCall, EdgeDirection};

/// t-UMAP 2D layout on a **cosine** kNN graph (an alternative to PHATE). Cells, MST
/// nodes, and principal-curve points are stacked into ONE matrix, L2-normalized
/// (cosine geometry), fed through a single fuzzy-kNN graph, and embedded jointly by
/// `matrix_util::umap` — so all three share the 2D space (t-UMAP has no PHATE-style
/// Nyström out-of-sample). The cell rows use the `space` representation (θ, θ+δ, or
/// [θ|δ]); the backbone (nodes/curves) stays on θ (δ is a small increment, so the
/// joint embedding stays coherent). Emits `{out}.{cells,nodes,curves}_2d.parquet`,
/// plus `{out}.velocity_grid_2d.parquet` (scVelo-style gridded arrows) when δ exists.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_umap_layout(
    theta: &DMatrix<f32>,
    native: NativeField<'_>,
    space: LayoutSpace,
    geometry: LatentGeometry,
    centroids: &DMatrix<f32>,
    curves: &PrincipalCurves,
    cell_names: &[Box<str>],
    knn: usize,
    seed: u64,
    out: &str,
) -> Result<()> {
    use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};
    use matrix_util::umap::Umap;

    let (n, h) = (theta.nrows(), theta.ncols());
    let vel = native.velocity.filter(|_| space != LayoutSpace::Identity);

    // CELLS ONLY — embedding just the cells (per the `space` representation) keeps the
    // trajectory backbone from distorting the manifold (the R exercises are cells-only);
    // the backbone is projected onto the fitted layout afterwards.
    //
    // θ and δ are combined in the NATIVE space δ is expressed in, and the metric is
    // applied to the RESULT — not the other way round. Under Hellinger the nascent
    // state is then √(θ+δ), an exact point on the simplex, whereas transforming first
    // would add a raw simplex increment to a √θ coordinate: a sum of two quantities
    // that live in different spaces. For `identity` the two orders coincide.
    let feats = match (space, vel) {
        (LayoutSpace::Identity, _) | (_, None) => theta.clone(),
        (LayoutSpace::Nascent, Some(v)) => apply_geometry(&(native.theta + v), geometry),
        (LayoutSpace::Concat, Some(v)) => {
            // Two channels, each normalized on its own so identity and velocity carry
            // equal weight; δ is a signed increment, so it takes cosine rather than the
            // θ channel's metric (√ of a negative is not a coordinate).
            let (tn, vn) = (theta.clone(), l2_normalize_rows(v));
            let mut f = DMatrix::<f32>::zeros(n, 2 * h);
            f.view_mut((0, 0), (n, h)).copy_from(&tn);
            f.view_mut((0, h), (n, h)).copy_from(&vn);
            f
        }
    };

    // Re-normalize after combining channels so `nascent`/`concat` do not let ‖δ‖ set the
    // row scale. No column z-scoring: standardizing columns hands every latent dimension
    // equal variance, which promotes the near-null dimensions of a decaying spectrum to
    // the same footing as the ones carrying the structure — a reliable way to turn a
    // manifold into a blob.
    let feats_n = if geometry == LatentGeometry::Euclidean {
        feats
    } else {
        l2_normalize_rows(&feats)
    };
    info!("t-UMAP layout ({geometry:?}, space={space:?}): {n} cells, knn={knn}");
    let graph = KnnGraph::from_rows(
        &feats_n,
        KnnGraphArgs {
            knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;
    let w = graph.fuzzy_kernel_weights();
    let edges: Vec<(usize, usize, f32)> = graph
        .edges
        .iter()
        .zip(w.iter())
        .map(|(&(i, j), &wt)| (i, j, wt))
        .collect();

    // Deterministic 2D init in [−10, 10] (seeded LCG — no rng dependency).
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut init = vec![0f32; n * 2];
    for v in init.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = (((s >> 33) as f32) / ((1u32 << 31) as f32) - 1.0) * 10.0;
    }
    // t-UMAP (a=b=1) — the uwot::tumap kernel; more spread than standard UMAP.
    let coords = Umap {
        seed,
        ..Umap::tumap()
    }
    .fit(&edges, n, &init);
    let mut cells_2d = DMatrix::<f32>::zeros(n, 2);
    for i in 0..n {
        cells_2d[(i, 0)] = coords[i * 2];
        cells_2d[(i, 1)] = coords[i * 2 + 1];
    }
    write_xy(
        &cells_2d,
        cell_names,
        "cell",
        &format!("{out}.cells_2d.parquet"),
    )?;

    // Project the backbone (nodes + curve points, θ space) onto the cells-only layout:
    // each lands at the mean 2D of its θ-nearest cells (t-UMAP has no Nyström).
    let n_nodes = centroids.nrows();
    let node_names = numbered("node_", n_nodes);
    write_xy(
        &project_onto_cells(centroids, theta, &cells_2d, knn),
        &node_names,
        "node",
        &format!("{out}.nodes_2d.parquet"),
    )?;
    let total_curve: usize = curves.curves.iter().map(|c| c.points.nrows()).sum();
    let mut curve_pts = DMatrix::<f32>::zeros(total_curve, h);
    let mut meta: Vec<(usize, usize)> = Vec::with_capacity(total_curve);
    let mut r = 0usize;
    for (l, c) in curves.curves.iter().enumerate() {
        for g in 0..c.points.nrows() {
            for j in 0..h {
                curve_pts[(r, j)] = c.points[(g, j)];
            }
            meta.push((l, g));
            r += 1;
        }
    }
    write_curves_2d(
        &project_onto_cells(&curve_pts, theta, &cells_2d, knn),
        &meta,
        &format!("{out}.curves_2d.parquet"),
    )?;

    emit_velocity_field(&cells_2d, native, knn, out)
}

/// The `(θ, δ)` pair as it came off disk — the space δ is actually expressed in.
///
/// Held separately from the metric-transformed θ that the fit and layout use. The
/// velocity field is a statement about the data, not about the coordinates chosen
/// to draw it, so it is computed here and only then projected onto whatever 2D the
/// layout produced — the separation scVelo makes between transition probabilities
/// in expression space and the embedding they are rendered on.
#[derive(Copy, Clone)]
pub(super) struct NativeField<'a> {
    pub theta: &'a DMatrix<f32>,
    pub velocity: Option<&'a DMatrix<f32>>,
}

/// Write `{out}.velocity_grid_2d.parquet` — scVelo-style gridded arrows (a few
/// hundred), each cell's δ projected into 2D by the native θ-neighbour transition
/// and averaged onto a coarse lattice. A no-op when the run has no δ.
///
/// Emitted for BOTH layouts. The field is what makes an identity-space embedding
/// readable as a trajectory: `--layout-space identity` deliberately keeps δ out of
/// the coordinates, so the arrows are where the direction information goes.
pub(super) fn emit_velocity_field(
    cells_2d: &DMatrix<f32>,
    native: NativeField<'_>,
    knn: usize,
    out: &str,
) -> Result<()> {
    let Some(v) = native.velocity else {
        return Ok(());
    };
    let grid = velocity_grid_arrows(cells_2d, native.theta, v, knn);
    info!("velocity field: {} gridded arrow(s)", grid.len());
    write_velocity_grid(&grid, &format!("{out}.velocity_grid_2d.parquet"))
}

/// Project points (θ space, `[m × H]`) onto a cells-only 2D layout: each lands at the
/// mean 2D of its `knn` θ-nearest cells (a simple k-NN out-of-sample, since t-UMAP has
/// no PHATE-style Nyström). Parallel over the `m` points. `cell_theta` `[n × H]`,
/// `cells_2d` `[n × 2]`.
pub(super) fn project_onto_cells(
    pts: &DMatrix<f32>,
    cell_theta: &DMatrix<f32>,
    cells_2d: &DMatrix<f32>,
    knn: usize,
) -> DMatrix<f32> {
    use rayon::prelude::*;
    let (m, h) = (pts.nrows(), pts.ncols());
    let n = cell_theta.nrows();
    let k = knn.clamp(1, n.max(1));
    let rows: Vec<(f32, f32)> = (0..m)
        .into_par_iter()
        .map(|p| {
            let mut dist: Vec<(f32, usize)> = (0..n)
                .map(|c| {
                    let dd = (0..h)
                        .map(|j| (pts[(p, j)] - cell_theta[(c, j)]).powi(2))
                        .sum::<f32>();
                    (dd, c)
                })
                .collect();
            if k < n {
                dist.select_nth_unstable_by(k - 1, |a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let (mut sx, mut sy) = (0f32, 0f32);
            for &(_, c) in dist.iter().take(k) {
                sx += cells_2d[(c, 0)];
                sy += cells_2d[(c, 1)];
            }
            (sx / k as f32, sy / k as f32)
        })
        .collect();
    let mut out = DMatrix::<f32>::zeros(m, 2);
    for (p, &(x, y)) in rows.iter().enumerate() {
        out[(p, 0)] = x;
        out[(p, 1)] = y;
    }
    out
}

/// Row-wise L2 normalization (unit vectors): Euclidean distance on the result
/// equals cosine distance on the input. Used for both the cosine θ fit and the
/// PHATE layout.
pub(super) fn l2_normalize_rows(m: &DMatrix<f32>) -> DMatrix<f32> {
    let mut out = m.clone();
    for i in 0..out.nrows() {
        let norm = (0..out.ncols())
            .map(|j| out[(i, j)] * out[(i, j)])
            .sum::<f32>()
            .sqrt();
        // Leave a ~zero row unchanged: normalizing it would blow it up to an
        // arbitrary unit direction (e.g. a centroid of near-antipodal points).
        if norm > 1e-9 {
            for j in 0..out.ncols() {
                out[(i, j)] /= norm;
            }
        }
    }
    out
}

/// Choose the PHATE landmark set and its 2D layout. When `N ≤ n_landmarks` every
/// cell is a landmark (exact PHATE). Above that, PHATE runs on `n_landmarks`
/// **k-means centroids** and the rest are lifted with the Nyström projector — capping
/// the O(n³) PHATE work at the landmark budget and making the remainder linear in N.
/// Returns `(landmark_features L×D, coords L×2)`.
pub(super) fn phate_landmark_layout(
    theta_n: &DMatrix<f32>,
    phate: &PhateArgs,
    n_landmarks: usize,
    seed: u64,
) -> (DMatrix<f32>, DMatrix<f32>) {
    let n = theta_n.nrows();
    if n <= n_landmarks || n_landmarks < 3 {
        return (theta_n.clone(), phate_layout_2d(theta_n, phate));
    }
    // Landmarks = k-means centroids (density-representative), like reference PHATE's
    // spectral landmarks. A plain stride subsample under-represents structure and the
    // Nyström lift then smears it — collapsing the layout onto its principal curve.
    // A loose 15-iteration cap is plenty: landmarks only seed the Nyström base, so tight
    // k-means convergence isn't needed (and k ≈ 2000 rarely stabilizes exactly anyway).
    let (land, _labels) = kmeans_centroids_seeded(theta_n, n_landmarks, 15, seed);
    let coords = phate_layout_2d(&land, phate);
    (land, coords)
}

/// Lay the cells out with PHATE (trajectory-preserving), then place the node
/// centroids and principal-curve points into the *same* space via the alpha-decay
/// Nyström projection — so the trajectory overlays faithfully. `project_cells_nystrom`
/// takes points as columns (D × n), hence the transposes.
///
/// The layout runs on raw θ (Euclidean) by default; `--normalize-latent` switches it (and the
/// fit) to L2-normalized θ (cosine geometry). For large N, PHATE runs on **k-means landmarks**
/// and every cell/node/curve point is lifted onto that layout via the same Nyström projector.
/// Edge → its tested direction, keyed by the canonical `(min, max)` node pair.
pub(super) type DirsMap<'a> = HashMap<(usize, usize), &'a EdgeDirection>;

/// Warp step as a fraction of the mean selected-edge length.
pub(super) const WARP_STEP_FRAC: f32 = 0.15;

/// Nudge each node along the net 2D flow of its confident selected edges (child downstream,
/// parent upstream), magnitude ∝ confidence and `WARP_STEP_FRAC` of the mean edge length;
/// cells follow their node. Abstained/geometry-only regions stay put.
pub(super) fn warp_layout_along_flow(
    nodes_2d: &mut DMatrix<f32>,
    cells_2d: &mut DMatrix<f32>,
    dirs_map: &DirsMap,
    br: &Branching,
    labels: &[usize],
) {
    let k = nodes_2d.nrows();
    let mut disp = DMatrix::<f32>::zeros(k, 2);
    let (mut len_sum, mut len_cnt) = (0f32, 0f32);
    for v in 0..k {
        let Some(p) = br.parent[v] else { continue };
        let Some(d) = dirs_map.get(&undirected(p, v)) else {
            continue;
        };
        if d.call == EdgeCall::Abstain {
            continue;
        }
        let dx = nodes_2d[(v, 0)] - nodes_2d[(p, 0)];
        let dy = nodes_2d[(v, 1)] - nodes_2d[(p, 1)];
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        len_sum += len;
        len_cnt += 1.0;
        let (ux, uy) = (d.confidence * dx / len, d.confidence * dy / len);
        disp[(v, 0)] += ux;
        disp[(v, 1)] += uy;
        disp[(p, 0)] -= ux;
        disp[(p, 1)] -= uy;
    }
    let step = if len_cnt > 0.0 {
        WARP_STEP_FRAC * len_sum / len_cnt
    } else {
        0.0
    };
    for v in 0..k {
        nodes_2d[(v, 0)] += step * disp[(v, 0)];
        nodes_2d[(v, 1)] += step * disp[(v, 1)];
    }
    for i in 0..cells_2d.nrows() {
        let l = labels[i];
        if l < k {
            cells_2d[(i, 0)] += step * disp[(l, 0)];
            cells_2d[(i, 1)] += step * disp[(l, 1)];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_phate_layout(
    theta: &DMatrix<f32>,
    native: NativeField<'_>,
    centroids: &DMatrix<f32>,
    curves: &PrincipalCurves,
    cell_names: &[Box<str>],
    phate: &PhateArgs,
    n_landmarks: usize,
    seed: u64,
    out: &str,
    warp: Option<(&DirsMap, &Branching, &[usize])>,
) -> Result<()> {
    let n = theta.nrows();
    // θ, the centroids and the curve points all arrive already in the requested metric
    // (`--latent-geometry`), so nothing is re-normalized here: doing so would silently
    // override that choice and put the backbone in a different space from the cells.
    let (land_feat, land_2d) = phate_landmark_layout(theta, phate, n_landmarks, seed);
    let exact = land_feat.nrows() == n;
    info!(
        "PHATE layout: {n} cells ({})",
        if exact {
            "exact".to_string()
        } else {
            format!("{} landmarks + Nyström", land_feat.nrows())
        }
    );
    let land_t = land_feat.transpose(); // D × L
    let (knn, alpha) = (phate.knn, phate.alpha);

    // Cells: exact PHATE already placed them; else lift onto the landmark layout.
    let mut cells_2d = if exact {
        land_2d.clone()
    } else {
        project_cells_nystrom(&theta.transpose(), &land_t, &land_2d, knn, alpha)
    };

    // Nodes + curve points always lift onto the landmark layout via Nyström.
    let mut nodes_2d =
        project_cells_nystrom(&centroids.transpose(), &land_t, &land_2d, knn, alpha);

    if let Some((dirs_map, br, labels)) = warp {
        warp_layout_along_flow(&mut nodes_2d, &mut cells_2d, dirs_map, br, labels);
    }

    // Stack all lineage curve points (in θ space) + remember (lineage, grid).
    let d = theta.ncols();
    let total: usize = curves.curves.iter().map(|c| c.points.nrows()).sum();
    let mut cpts = DMatrix::<f32>::zeros(total, d);
    let mut meta: Vec<(usize, usize)> = Vec::with_capacity(total);
    let mut r = 0usize;
    for (l, c) in curves.curves.iter().enumerate() {
        for g in 0..c.points.nrows() {
            for j in 0..d {
                cpts[(r, j)] = c.points[(g, j)];
            }
            meta.push((l, g));
            r += 1;
        }
    }
    let curves_2d = project_cells_nystrom(&cpts.transpose(), &land_t, &land_2d, knn, alpha);

    write_xy(
        &cells_2d,
        cell_names,
        "cell",
        &format!("{out}.cells_2d.parquet"),
    )?;
    let node_names = numbered("node_", nodes_2d.nrows());
    write_xy(
        &nodes_2d,
        &node_names,
        "node",
        &format!("{out}.nodes_2d.parquet"),
    )?;
    write_curves_2d(&curves_2d, &meta, &format!("{out}.curves_2d.parquet"))?;

    // Arrows come LAST, off the final coordinates: the warp above moves cells, and a field
    // drawn from pre-warp positions would point away from where the plot puts them.
    emit_velocity_field(&cells_2d, native, knn, out)
}
