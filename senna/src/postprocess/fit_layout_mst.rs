//! `senna layout mst` — PB landmarks laid out via MST + force-directed
//! embedding on the PB-PB similarity graph.
//!
//! 1. Convert `pb_similarity` → distance `d_ij = 1 − s_ij`.
//! 2. Build a minimum spanning tree over PBs (Kruskal + union-find).
//! 3. Fruchterman–Reingold force-directed layout on the MST, seeded from
//!    uniform-random 2D coordinates (`--seed`).
//! 4. Nyström extension places cells via the shared finalizer.

use super::fit_visualize_common::{
    finalize_viz, preprocess_layout_data, random_init_2d, resolve_inputs, VisualizeCommonArgs,
};
use crate::embed_common::*;

#[derive(Args, Debug)]
pub struct LayoutMstArgs {
    #[clap(flatten)]
    common: VisualizeCommonArgs,

    #[arg(
        long,
        default_value_t = 500,
        help = "Fruchterman–Reingold iterations"
    )]
    mst_iters: usize,
}

pub fn fit_layout_mst(args: &LayoutMstArgs) -> anyhow::Result<()> {
    let mut resolved = resolve_inputs(&args.common)?;
    let prep = preprocess_layout_data(&args.common, &resolved)?;

    let n = prep.pb_similarity.nrows();
    anyhow::ensure!(n >= 2, "MST layout needs at least 2 PBs, got {n}");

    let edges = build_mst(&prep.pb_similarity);
    debug_assert_eq!(edges.len(), n - 1, "MST must have n−1 edges");
    info!("Built MST on {} PBs with {} edges", n, edges.len());

    let init = random_init_2d(n, args.common.seed);
    let pb_coords = fruchterman_reingold(&init, &edges, args.mst_iters);
    info!("Ran Fruchterman–Reingold for {} iterations", args.mst_iters);

    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords)
}

/// Kruskal MST over a dense PB-PB similarity matrix. Similarity ∈ [0, 1]
/// is converted to distance as `d = 1 − s` (keeps MST edges as
/// high-similarity ones). Only the upper triangle is enumerated.
fn build_mst(sim: &Mat) -> Vec<(usize, usize)> {
    let n = sim.nrows();
    let mut cands: Vec<(f32, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = 1.0 - sim[(i, j)];
            cands.push((d, i, j));
        }
    }
    cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }

    let mut edges = Vec::with_capacity(n - 1);
    for (_, i, j) in cands {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
            parent[ri] = rj;
            edges.push((i, j));
            if edges.len() == n - 1 {
                break;
            }
        }
    }
    edges
}

/// Fruchterman–Reingold force-directed layout on a sparse edge list.
/// Attractive forces on MST edges, repulsive forces between all node
/// pairs. Linear cooling schedule. Returns `(n × 2)` coords.
fn fruchterman_reingold(init: &Mat, edges: &[(usize, usize)], iters: usize) -> Mat {
    let n = init.nrows();
    let mut x: Vec<f32> = (0..n).map(|i| init[(i, 0)]).collect();
    let mut y: Vec<f32> = (0..n).map(|i| init[(i, 1)]).collect();

    // Ideal edge length — area is roughly [-1, 1]² so A = 4 ⇒ k = sqrt(A/n).
    let k = (4.0_f32 / n as f32).sqrt();
    let k2 = k * k;
    let t0 = 0.1_f32;

    let mut dx = vec![0.0f32; n];
    let mut dy = vec![0.0f32; n];

    for it in 0..iters {
        dx.fill(0.0);
        dy.fill(0.0);

        // Repulsive forces: F_r = k² / d
        for i in 0..n {
            for j in (i + 1)..n {
                let rx = x[i] - x[j];
                let ry = y[i] - y[j];
                let d2 = (rx * rx + ry * ry).max(1e-6);
                let d = d2.sqrt();
                let f = k2 / d;
                let ux = rx / d;
                let uy = ry / d;
                dx[i] += ux * f;
                dy[i] += uy * f;
                dx[j] -= ux * f;
                dy[j] -= uy * f;
            }
        }

        // Attractive forces on tree edges: F_a = d² / k
        for &(i, j) in edges {
            let rx = x[i] - x[j];
            let ry = y[i] - y[j];
            let d2 = (rx * rx + ry * ry).max(1e-6);
            let d = d2.sqrt();
            let f = d2 / k;
            let ux = rx / d;
            let uy = ry / d;
            dx[i] -= ux * f;
            dy[i] -= uy * f;
            dx[j] += ux * f;
            dy[j] += uy * f;
        }

        // Cooling + clamped step.
        let t = t0 * (1.0 - it as f32 / iters.max(1) as f32);
        for i in 0..n {
            let disp = (dx[i] * dx[i] + dy[i] * dy[i]).sqrt().max(1e-6);
            let step = disp.min(t);
            x[i] += dx[i] / disp * step;
            y[i] += dy[i] / disp * step;
        }
    }

    let mut coords = Mat::zeros(n, 2);
    for i in 0..n {
        coords[(i, 0)] = x[i];
        coords[(i, 1)] = y[i];
    }
    coords
}
