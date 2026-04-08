use crate::common::*;
use candle_util::candle_core::Tensor;
use genomic_data::coordinates::PeakCoord;

/// Compute log-spaced target sizes for multi-level coarsening.
fn log_spaced_targets(num_levels: usize, max_features: usize) -> Vec<usize> {
    let min_target = (max_features / num_levels).max(50);
    let log_min = (min_target as f64).ln();
    let log_max = (max_features as f64).ln();
    (0..num_levels)
        .map(|i| {
            let frac = if num_levels > 1 {
                i as f64 / (num_levels - 1) as f64
            } else {
                1.0
            };
            (log_min + frac * (log_max - log_min))
                .exp()
                .round()
                .clamp(min_target as f64, max_features as f64) as usize
        })
        .collect()
}

/// Data-driven feature coarsenings with log-spaced targets.
pub fn log_spaced_coarsenings(
    sketch: &nalgebra::DMatrix<f32>,
    num_levels: usize,
    max_features: usize,
) -> anyhow::Result<Vec<Option<FeatureCoarsening>>> {
    if max_features == 0 || sketch.nrows() <= max_features {
        return Ok(vec![None; num_levels]);
    }
    log_spaced_targets(num_levels, max_features)
        .into_iter()
        .map(|target| Ok(Some(compute_feature_coarsening(sketch, target)?)))
        .collect()
}

/// Genomic-aware ATAC coarsenings: only merges neighboring peaks on same chromosome.
pub fn log_spaced_genomic_coarsenings(
    peak_coords: &[Option<PeakCoord>],
    num_levels: usize,
    max_features: usize,
) -> Vec<Option<FeatureCoarsening>> {
    if max_features == 0 || peak_coords.len() <= max_features {
        return vec![None; num_levels];
    }
    log_spaced_targets(num_levels, max_features)
        .into_iter()
        .map(|target| Some(genomic_feature_coarsening(peak_coords, target)))
        .collect()
}

/// Merge neighboring peaks on the same chromosome to reach `target` groups.
///
/// Peaks are sorted by (chr, start). Consecutive peaks on the same chromosome
/// are grouped together. Each chromosome's peaks are split into roughly equal
/// segments proportional to the chromosome's share of total peaks.
fn genomic_feature_coarsening(
    peak_coords: &[Option<PeakCoord>],
    target: usize,
) -> FeatureCoarsening {
    let n = peak_coords.len();

    // Sort peak indices by (chr, start), putting uncoordinated peaks last
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        let ca = peak_coords[a].as_ref();
        let cb = peak_coords[b].as_ref();
        match (ca, cb) {
            (Some(pa), Some(pb)) => (&*pa.chr, pa.start).cmp(&(&*pb.chr, pb.start)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.cmp(&b),
        }
    });

    // Identify chromosome runs in sorted order
    let mut chr_runs: Vec<(Box<str>, Vec<usize>)> = Vec::new();
    for &idx in &sorted_idx {
        let chr = peak_coords[idx]
            .as_ref()
            .map(|c| c.chr.clone())
            .unwrap_or_else(|| "unknown".into());
        if chr_runs.last().map(|(c, _)| c != &chr).unwrap_or(true) {
            chr_runs.push((chr, Vec::new()));
        }
        chr_runs.last_mut().unwrap().1.push(idx);
    }

    // Allocate groups proportionally per chromosome
    let target = target.min(n);
    let mut fine_to_coarse = vec![0usize; n];
    let mut group_id = 0usize;
    let mut coarse_to_fine: Vec<Vec<usize>> = Vec::new();

    for (_chr, peaks) in &chr_runs {
        let chr_n = peaks.len();
        // This chromosome gets proportional share of target groups (at least 1)
        let chr_groups = ((chr_n as f64 / n as f64) * target as f64).round() as usize;
        let chr_groups = chr_groups.max(1);
        let group_size = chr_n.div_ceil(chr_groups);

        for (j, &peak_idx) in peaks.iter().enumerate() {
            let local_group = j / group_size;
            let gid = group_id + local_group;
            fine_to_coarse[peak_idx] = gid;

            // Extend coarse_to_fine as needed
            while coarse_to_fine.len() <= gid {
                coarse_to_fine.push(Vec::new());
            }
            coarse_to_fine[gid].push(peak_idx);
        }
        let actual_groups = chr_n.div_ceil(group_size);
        group_id += actual_groups;
    }

    FeatureCoarsening {
        fine_to_coarse,
        coarse_to_fine,
        num_coarse: group_id,
    }
}

/// Coarsen a tensor [N, D] → [N, d] using feature coarsening.
/// If coarsening is None, returns the tensor unchanged.
pub fn coarsen_tensor(
    x: &Tensor,
    fc: Option<&FeatureCoarsening>,
) -> candle_util::candle_core::Result<Tensor> {
    use rayon::prelude::*;

    match fc {
        Some(fc) => {
            let n = x.dim(0)?;
            let d = x.dim(1)?;
            let data: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let nc = fc.num_coarse;

            let out: Vec<f32> = data
                .par_chunks(d)
                .flat_map_iter(|row| {
                    let mut coarse_row = vec![0.0f32; nc];
                    for (col, &val) in row.iter().enumerate() {
                        coarse_row[fc.fine_to_coarse[col]] += val;
                    }
                    coarse_row
                })
                .collect();

            Tensor::from_vec(out, (n, nc), x.device())
        }
        None => Ok(x.clone()),
    }
}
