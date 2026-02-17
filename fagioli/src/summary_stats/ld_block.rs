use anyhow::Result;
use log::info;
use matrix_util::traits::{MatOps, RandomizedAlgs};
use nalgebra::DMatrix;
use rand::seq::index::sample;
use rand::SeedableRng;

/// An LD block: contiguous range of SNPs
#[derive(Debug, Clone)]
pub struct LdBlock {
    pub block_idx: usize,
    pub snp_start: usize, // inclusive index into global SNP array
    pub snp_end: usize,   // exclusive
    pub chr: Box<str>,
    pub bp_start: u64,
    pub bp_end: u64,
}

impl LdBlock {
    pub fn num_snps(&self) -> usize {
        self.snp_end - self.snp_start
    }
}

/// Estimate LD blocks from genotype data using Nystrom + rSVD.
///
/// 1. Sample S landmark SNPs -> N x S matrix -> rSVD -> U (N x k)
/// 2. Project all SNPs: v_j = U' x x_j -> M x k SNP embedding
/// 3. Segment: rolling distance d_j = ||v_j - v_{j+1}||^2, cut at peaks
pub fn estimate_ld_blocks(
    genotypes: &DMatrix<f32>,
    positions: &[u64],
    chromosomes: &[Box<str>],
    num_landmarks: usize,
    num_components: usize,
    min_block_snps: usize,
    max_block_snps: usize,
    seed: u64,
) -> Result<Vec<LdBlock>> {
    let n = genotypes.nrows();
    let m = genotypes.ncols();

    if m == 0 {
        anyhow::bail!("No SNPs to estimate LD blocks from");
    }

    let num_landmarks = num_landmarks.min(m);
    let num_components = num_components.min(num_landmarks);

    info!(
        "Estimating LD blocks: {} SNPs, {} landmarks, {} components",
        m, num_landmarks, num_components
    );

    // Step 1: Sample landmark SNPs and compute rSVD basis
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let landmark_indices: Vec<usize> = {
        let mut idx: Vec<usize> = sample(&mut rng, m, num_landmarks).into_vec();
        idx.sort_unstable();
        idx
    };

    let mut x_landmarks = DMatrix::<f32>::zeros(n, num_landmarks);
    for (j, &snp_idx) in landmark_indices.iter().enumerate() {
        x_landmarks.set_column(j, &genotypes.column(snp_idx));
    }
    x_landmarks.scale_columns_inplace();

    let (u, _sigma, _v) = x_landmarks.rsvd(num_components)?;
    // U is N x k basis for individual PC space

    info!("Computed rSVD basis: {} x {}", u.nrows(), u.ncols());

    // Step 2: Project all SNPs into embedding space: v_j = U' x_j
    // Compute V = U' X -> k x M, then transpose to M x k
    let embeddings = u.transpose() * genotypes; // k x M
    let embeddings = embeddings.transpose(); // M x k

    // Step 3: Compute consecutive distances
    let mut distances = vec![0.0f32; m.saturating_sub(1)];
    for j in 0..m.saturating_sub(1) {
        // Also cut at chromosome boundaries (infinite distance)
        if chromosomes[j] != chromosomes[j + 1] {
            distances[j] = f32::INFINITY;
        } else {
            let diff = embeddings.row(j + 1) - embeddings.row(j);
            distances[j] = diff.dot(&diff);
        }
    }

    if distances.is_empty() {
        // Single SNP
        return Ok(vec![LdBlock {
            block_idx: 0,
            snp_start: 0,
            snp_end: 1,
            chr: chromosomes[0].clone(),
            bp_start: positions[0],
            bp_end: positions[0],
        }]);
    }

    // Step 4: Smooth distances with moving average
    let window = 5.min(distances.len());
    let smoothed = smooth_moving_average(&distances, window);

    // Step 5: Find peaks above threshold
    let finite_vals: Vec<f32> = smoothed.iter().copied().filter(|x| x.is_finite()).collect();
    let mean_d = finite_vals.iter().sum::<f32>() / finite_vals.len().max(1) as f32;
    let var_d = finite_vals
        .iter()
        .map(|x| (x - mean_d).powi(2))
        .sum::<f32>()
        / finite_vals.len().max(1) as f32;
    let std_d = var_d.sqrt();
    let threshold = mean_d + 2.0 * std_d;

    // Find candidate cut points (peaks above threshold or chromosome boundaries)
    let mut cut_points: Vec<usize> = Vec::new();
    for j in 0..smoothed.len() {
        if smoothed[j] > threshold || !smoothed[j].is_finite() {
            // Check if local peak (or chromosome boundary)
            if !smoothed[j].is_finite() {
                cut_points.push(j + 1); // cut after position j
            } else {
                let is_peak = (j == 0 || smoothed[j] >= smoothed[j - 1])
                    && (j == smoothed.len() - 1 || smoothed[j] >= smoothed[j + 1]);
                if is_peak {
                    cut_points.push(j + 1);
                }
            }
        }
    }

    // Step 6: Enforce min/max block size constraints
    let blocks = build_blocks_from_cuts(
        &cut_points,
        m,
        positions,
        chromosomes,
        min_block_snps,
        max_block_snps,
    );

    info!("Estimated {} LD blocks", blocks.len());
    Ok(blocks)
}

/// Load LD blocks from a BED-like file (chr, start, end).
pub fn load_ld_blocks_from_file(
    path: &str,
    snp_positions: &[u64],
    snp_chromosomes: &[Box<str>],
) -> Result<Vec<LdBlock>> {
    use matrix_util::common_io::open_buf_reader;
    use std::io::BufRead;

    let reader = open_buf_reader(path)?;
    let mut regions: Vec<(Box<str>, u64, u64)> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            anyhow::bail!("LD block file must have at least 3 columns (chr, start, end)");
        }
        let chr = Box::from(fields[0]);
        let start: u64 = fields[1].parse()?;
        let end: u64 = fields[2].parse()?;
        regions.push((chr, start, end));
    }

    info!("Loaded {} LD block regions from {}", regions.len(), path);

    // Assign SNPs to blocks
    let mut blocks = Vec::new();
    let m = snp_positions.len();

    for (region_idx, (chr, bp_start, bp_end)) in regions.iter().enumerate() {
        let mut snp_start = None;
        let mut snp_end = None;

        for j in 0..m {
            if snp_chromosomes[j].as_ref() == chr.as_ref()
                && snp_positions[j] >= *bp_start
                && snp_positions[j] < *bp_end
            {
                if snp_start.is_none() {
                    snp_start = Some(j);
                }
                snp_end = Some(j + 1);
            }
        }

        if let (Some(start), Some(end)) = (snp_start, snp_end) {
            blocks.push(LdBlock {
                block_idx: region_idx,
                snp_start: start,
                snp_end: end,
                chr: chr.clone(),
                bp_start: *bp_start,
                bp_end: *bp_end,
            });
        }
    }

    // Re-index block_idx sequentially
    for (i, block) in blocks.iter_mut().enumerate() {
        block.block_idx = i;
    }

    info!("Assigned SNPs to {} non-empty LD blocks", blocks.len());
    Ok(blocks)
}

/// Create uniform-size blocks (fallback when no estimation needed).
pub fn create_uniform_blocks(
    num_snps: usize,
    block_size: usize,
    positions: &[u64],
    chromosomes: &[Box<str>],
) -> Vec<LdBlock> {
    let mut blocks = Vec::new();
    let mut start = 0;

    while start < num_snps {
        let mut end = (start + block_size).min(num_snps);

        // Don't split across chromosomes
        if end < num_snps && end > start + 1 && chromosomes[end - 1] != chromosomes[end] {
            // end is already at a chromosome boundary, keep it
        }
        // Also break at chromosome boundaries within the block
        for j in (start + 1)..end {
            if chromosomes[j] != chromosomes[start] {
                end = j;
                break;
            }
        }

        blocks.push(LdBlock {
            block_idx: blocks.len(),
            snp_start: start,
            snp_end: end,
            chr: chromosomes[start].clone(),
            bp_start: positions[start],
            bp_end: positions[end - 1],
        });

        start = end;
    }

    blocks
}

fn smooth_moving_average(values: &[f32], window: usize) -> Vec<f32> {
    let n = values.len();
    let half = window / 2;
    let mut smoothed = vec![0.0f32; n];

    for i in 0..n {
        if !values[i].is_finite() {
            smoothed[i] = f32::INFINITY;
            continue;
        }
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let mut sum = 0.0f32;
        let mut count = 0;
        for val in &values[lo..hi] {
            if val.is_finite() {
                sum += val;
                count += 1;
            }
        }
        smoothed[i] = if count > 0 { sum / count as f32 } else { 0.0 };
    }

    smoothed
}

fn build_blocks_from_cuts(
    cut_points: &[usize],
    num_snps: usize,
    positions: &[u64],
    chromosomes: &[Box<str>],
    min_block_snps: usize,
    max_block_snps: usize,
) -> Vec<LdBlock> {
    // Merge all cut points, add boundaries
    let mut boundaries: Vec<usize> = Vec::new();
    boundaries.push(0);
    for &cp in cut_points {
        if cp > 0 && cp < num_snps {
            boundaries.push(cp);
        }
    }
    boundaries.push(num_snps);
    boundaries.sort_unstable();
    boundaries.dedup();

    // Split blocks that are too large
    let mut refined: Vec<usize> = vec![0];
    for w in boundaries.windows(2) {
        let (start, end) = (w[0], w[1]);
        let size = end - start;
        if size > max_block_snps {
            let num_sub = size.div_ceil(max_block_snps);
            let sub_size = size / num_sub;
            for i in 1..num_sub {
                refined.push(start + i * sub_size);
            }
        }
        refined.push(end);
    }
    refined.sort_unstable();
    refined.dedup();

    // Merge blocks that are too small
    let mut merged: Vec<usize> = vec![refined[0]];
    for i in 1..refined.len() {
        let prev = *merged.last().unwrap();
        let cur = refined[i];
        if cur - prev < min_block_snps && i < refined.len() - 1 {
            // Skip this boundary (merge with next)
            continue;
        }
        merged.push(cur);
    }

    // Build LdBlock structs
    let mut blocks = Vec::new();
    for w in merged.windows(2) {
        let (start, end) = (w[0], w[1]);
        if start >= end {
            continue;
        }
        blocks.push(LdBlock {
            block_idx: blocks.len(),
            snp_start: start,
            snp_end: end,
            chr: chromosomes[start].clone(),
            bp_start: positions[start],
            bp_end: positions[end - 1],
        });
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix_util::traits::SampleOps;

    #[test]
    fn test_create_uniform_blocks() {
        let positions: Vec<u64> = (0..100).map(|i| i * 1000).collect();
        let chromosomes: Vec<Box<str>> = vec![Box::from("chr1"); 100];
        let blocks = create_uniform_blocks(100, 25, &positions, &chromosomes);

        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].snp_start, 0);
        assert_eq!(blocks[0].snp_end, 25);
        assert_eq!(blocks[3].snp_start, 75);
        assert_eq!(blocks[3].snp_end, 100);
    }

    #[test]
    fn test_uniform_blocks_chromosome_boundary() {
        let positions: Vec<u64> = (0..100).map(|i| i * 1000).collect();
        let mut chromosomes: Vec<Box<str>> = vec![Box::from("chr1"); 60];
        chromosomes.extend(vec![Box::from("chr2"); 40]);

        let blocks = create_uniform_blocks(100, 50, &positions, &chromosomes);

        // Should not span chromosomes
        for block in &blocks {
            let chr_start = &chromosomes[block.snp_start];
            let chr_end = &chromosomes[block.snp_end - 1];
            assert_eq!(
                chr_start, chr_end,
                "Block {} spans chromosomes",
                block.block_idx
            );
        }
    }

    #[test]
    fn test_smooth_moving_average() {
        let values = vec![1.0, 2.0, 10.0, 2.0, 1.0];
        let smoothed = smooth_moving_average(&values, 3);
        // Middle value (10.0) should be dampened
        assert!(smoothed[2] < 10.0);
        assert!(smoothed[2] > 2.0);
    }

    #[test]
    fn test_estimate_ld_blocks_synthetic() {
        let n = 100;
        let m = 200;
        let genotypes = DMatrix::<f32>::rnorm(n, m);
        let positions: Vec<u64> = (0..m as u64).map(|i| i * 1000).collect();
        let chromosomes: Vec<Box<str>> = vec![Box::from("chr1"); m];

        let blocks = estimate_ld_blocks(
            &genotypes,
            &positions,
            &chromosomes,
            50,  // landmarks
            10,  // components
            20,  // min block
            100, // max block
            42,
        )
        .unwrap();

        // Should produce at least one block covering all SNPs
        assert!(!blocks.is_empty());
        assert_eq!(blocks[0].snp_start, 0);
        assert_eq!(blocks.last().unwrap().snp_end, m);

        // Blocks should be contiguous
        for w in blocks.windows(2) {
            assert_eq!(w[0].snp_end, w[1].snp_start);
        }
    }
}
