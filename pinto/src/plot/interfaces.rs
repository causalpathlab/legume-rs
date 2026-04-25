//! High-entropy "interface" cells + neighborhood overlay.
//!
//! Renders a single overview figure per (level, core) plus a TSV legend.
//! The figure encodes per-cell entropy as a *grayscale + size* signal:
//! every core cell is drawn as a dark-gray hex whose radius is scaled
//! by the cell's entropy quantile within the core (lowest → 0, highest
//! → the full hex tile). The TSV lists, for each focal cell (top
//! quantile), its dominant community, its 1- and 2-hop neighbor counts,
//! and the top-N marker genes per neighbor community.
//!
//! Driven by `--show-interfaces`. Silently skipped for any (level, core)
//! whose propensity parquet has no `entropy` column (older runs).

use super::args::SrtPlotArgs;
use super::load::{CellTable, EdgePair};
use super::markers;
use super::partition::CoreSpec;
use super::render::{emit_figure, Frame};
use crate::util::common::*;
use plot_utils::hull::Pt;
use plot_utils::rasterize::{rasterize_group_png, RadiusSpec};
use plot_utils::svg_emit::TopicLayer;
use std::path::Path;
use std::path::PathBuf;

/// Render the interface overview + write the TSV legend.
///
/// Returns the list of files written (added to the plot manifest).
#[allow(clippy::too_many_arguments)]
pub fn render_interfaces(
    args: &SrtPlotArgs,
    frame: &Frame,
    cells: &CellTable,
    core: &CoreSpec,
    entropy: &[f32],
    dominant: &[i64],
    edges: Option<&[EdgePair]>,
    gene_topic: Option<&(Mat, Vec<Box<str>>)>,
    out_stub: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    if entropy.len() != cells.n() {
        anyhow::bail!(
            "entropy length {} != cells.n() {}",
            entropy.len(),
            cells.n()
        );
    }

    // 1. Pick focal cells: top-(1 - quantile) by entropy *within this core*.
    let focal = pick_focal_cells(
        core,
        entropy,
        args.entropy_quantile,
        args.max_interface_cells,
    );
    if focal.is_empty() {
        log::info!(
            "interfaces[{}]: no focal cells passed --entropy-quantile={}; skipping",
            core.name,
            args.entropy_quantile
        );
        return Ok(Vec::new());
    }

    // 2. Build per-cell adjacency restricted to this core.
    let core_set: HashSet<usize> = core.cell_ixs.iter().copied().collect();
    let adjacency = build_core_adjacency(cells, &core_set, edges);

    // 3. Per focal cell: BFS for 1-hop and (optionally) 2-hop neighbors.
    let hops = args.neighborhood_hops.max(1);
    let neighborhoods: Vec<Neighborhood> = focal
        .iter()
        .map(|&fi| neighborhood_of(fi, &adjacency, hops))
        .collect();

    let mut emitted = Vec::new();
    let layers = build_interface_layers(args, frame, cells, core, entropy)?;
    emit_figure(&layers, frame, args, out_stub, &mut emitted)?;

    // 5. Write the TSV legend.
    let tsv_path = PathBuf::from(format!("{}.tsv", out_stub.display()));
    write_interface_tsv(
        &tsv_path,
        cells,
        &focal,
        entropy,
        dominant,
        &neighborhoods,
        gene_topic,
        args.interface_top_genes,
    )?;
    emitted.push(tsv_path);

    Ok(emitted)
}

#[derive(Default)]
struct Neighborhood {
    /// Cells at distance 1 from the focal (excluding the focal itself).
    one_hop: Vec<usize>,
    /// Cells at distance 2 from the focal (excluding focal + 1-hop).
    two_hop: Vec<usize>,
}

/// "Uncommitted" cells: drop any cell whose top propensity column
/// exceeds `commit_threshold` (i.e. cell is firmly in one community).
/// Returns the indices of the remaining boundary cells. Simpler /
/// less fuzzy than an entropy quantile.
pub fn pick_uncommitted_cells(
    core: &CoreSpec,
    propensity: &Mat,
    commit_threshold: f32,
) -> Vec<usize> {
    let thr = commit_threshold.clamp(0.0, 1.0);
    let k = propensity.ncols();
    if k == 0 {
        return Vec::new();
    }
    core.cell_ixs
        .iter()
        .filter_map(|&i| {
            if i >= propensity.nrows() {
                return None;
            }
            let mut max_p = 0.0f32;
            for j in 0..k {
                let v = propensity[(i, j)];
                if v.is_finite() && v > max_p {
                    max_p = v;
                }
            }
            if max_p < thr {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

pub fn pick_focal_cells(core: &CoreSpec, entropy: &[f32], quantile: f32, cap: usize) -> Vec<usize> {
    // Collect (cell_index, entropy) pairs for finite entries in this core.
    let mut scored: Vec<(usize, f32)> = core
        .cell_ixs
        .iter()
        .filter_map(|&i| {
            let h = *entropy.get(i)?;
            if h.is_finite() {
                Some((i, h))
            } else {
                None
            }
        })
        .collect();
    if scored.is_empty() {
        return Vec::new();
    }
    // Sort descending by entropy.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Quantile-based threshold: keep cells above the (quantile)-th value.
    let q = quantile.clamp(0.0, 1.0);
    let cutoff_idx = ((1.0 - q) * scored.len() as f32).round().max(1.0) as usize;
    let mut keep_n = cutoff_idx.min(scored.len());
    keep_n = keep_n.min(cap.max(1));
    scored.truncate(keep_n);
    scored.into_iter().map(|(i, _)| i).collect()
}

fn build_core_adjacency(
    cells: &CellTable,
    core_set: &HashSet<usize>,
    edges: Option<&[EdgePair]>,
) -> Vec<Vec<usize>> {
    let n = cells.n();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let Some(edges) = edges else {
        return adj;
    };
    for (l, r) in edges {
        let li = match cells.index.get(l) {
            Some(&i) => i,
            None => continue,
        };
        let ri = match cells.index.get(r) {
            Some(&i) => i,
            None => continue,
        };
        if !core_set.contains(&li) || !core_set.contains(&ri) {
            continue;
        }
        adj[li].push(ri);
        adj[ri].push(li);
    }
    // Dedup neighbor lists (a fine edge may appear once per direction).
    for nbrs in adj.iter_mut() {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    adj
}

fn neighborhood_of(focal: usize, adjacency: &[Vec<usize>], hops: u8) -> Neighborhood {
    let mut nbrs = Neighborhood::default();
    let mut seen: HashSet<usize> = Default::default();
    seen.insert(focal);

    // 1-hop
    for &n1 in &adjacency[focal] {
        if seen.insert(n1) {
            nbrs.one_hop.push(n1);
        }
    }
    if hops < 2 {
        return nbrs;
    }
    // 2-hop
    let frontier: Vec<usize> = nbrs.one_hop.clone();
    for u in frontier {
        for &n2 in &adjacency[u] {
            if seen.insert(n2) {
                nbrs.two_hop.push(n2);
            }
        }
    }
    nbrs
}

fn build_interface_layers(
    args: &SrtPlotArgs,
    frame: &Frame,
    cells: &CellTable,
    core: &CoreSpec,
    entropy: &[f32],
) -> anyhow::Result<Vec<TopicLayer>> {
    let shape = args.point_shape;
    let pts: Vec<Pt> = core
        .cell_ixs
        .iter()
        .map(|&i| frame.bounds.to_pixel(cells.coords[i], frame.extent))
        .collect();

    // Quantile-rank entropy within the core: each cell maps to its
    // rank fraction ∈ [0, 1]. Cells with NaN entropy get rank 0
    // (rendered with zero radius). Ties get the average of their
    // shared range so equal entropies render at equal size.
    let n = core.cell_ixs.len();
    let mut indexed: Vec<(usize /*core_pos*/, f32 /*entropy*/)> = core
        .cell_ixs
        .iter()
        .enumerate()
        .map(|(pos, &i)| (pos, *entropy.get(i).unwrap_or(&f32::NAN)))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut rank_frac = vec![0.0f32; n];
    let denom = (n.saturating_sub(1)).max(1) as f32;
    let mut k = 0usize;
    while k < indexed.len() {
        let v = indexed[k].1;
        if !v.is_finite() {
            // NaNs sort to either end depending on partial_cmp; force them to 0.
            rank_frac[indexed[k].0] = 0.0;
            k += 1;
            continue;
        }
        let mut j = k + 1;
        while j < indexed.len() && indexed[j].1 == v {
            j += 1;
        }
        let avg = ((k + j - 1) as f32) / 2.0 / denom;
        for tied in &indexed[k..j] {
            rank_frac[tied.0] = avg;
        }
        k = j;
    }

    let _ = args; // shape already extracted; size capped at base_radius for tile alignment.
    let radii: Vec<f32> = rank_frac
        .iter()
        .map(|&q| frame.radius_px_base * q)
        .collect();
    let png = rasterize_group_png(
        &pts,
        frame.extent,
        RadiusSpec::Per(&radii),
        (40, 40, 40),
        0.85,
        shape,
    )?;
    Ok(vec![TopicLayer {
        label: String::new(),
        png,
        hull_px: Vec::new(),
        label_xy_px: (f32::NAN, f32::NAN),
        color: (40, 40, 40),
    }])
}

#[allow(clippy::too_many_arguments)]
fn write_interface_tsv(
    path: &Path,
    cells: &CellTable,
    focal: &[usize],
    entropy: &[f32],
    dominant: &[i64],
    neighborhoods: &[Neighborhood],
    gene_topic: Option<&(Mat, Vec<Box<str>>)>,
    top_genes: usize,
) -> anyhow::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "focal_cell\tentropy\tdominant_community\tn_1hop\tn_2hop\tneighbor_communities\ttop_genes_per_neighbor_community"
    )?;

    for (focal_idx, nbrs) in focal.iter().zip(neighborhoods.iter()) {
        let i = *focal_idx;
        let name = cells.names.get(i).map(|s| s.as_ref()).unwrap_or("?");
        let h = entropy[i];
        let dom = dominant.get(i).copied().unwrap_or(-1);
        // Counts of neighbor communities across 1-hop ∪ 2-hop.
        let mut counts: HashMap<i64, usize> = Default::default();
        for &n in nbrs.one_hop.iter().chain(nbrs.two_hop.iter()) {
            let c = dominant.get(n).copied().unwrap_or(-1);
            *counts.entry(c).or_insert(0) += 1;
        }
        let mut comm_pairs: Vec<(i64, usize)> = counts.into_iter().collect();
        comm_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        let comm_str: String = comm_pairs
            .iter()
            .map(|(c, n)| format!("C{c}={n}"))
            .collect::<Vec<_>>()
            .join(",");

        let genes_str: String = match gene_topic {
            Some((gt, names)) => comm_pairs
                .iter()
                .filter(|(c, _)| *c >= 0 && (*c as usize) < gt.ncols())
                .map(|(c, _)| {
                    let top = markers::top_n_markers(gt, names, *c as usize, top_genes);
                    let g = top
                        .iter()
                        .map(|(_, n)| n.as_ref())
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("C{c}:[{g}]")
                })
                .collect::<Vec<_>>()
                .join(";"),
            None => String::new(),
        };

        writeln!(
            f,
            "{name}\t{h:.6}\t{dom}\t{}\t{}\t{comm_str}\t{genes_str}",
            nbrs.one_hop.len(),
            nbrs.two_hop.len(),
        )?;
    }
    Ok(())
}
