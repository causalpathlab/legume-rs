use anyhow::Result;
use candle_util::sgvb::block_partition::BlockPartition;
use log::info;
use nalgebra::DMatrix;

use super::config::{FitConfig, SnpCoordinates};
use crate::summary_stats::{estimate_ld_blocks, LdBlockParams};

/// Estimate LD-aware partitions for multilevel SuSiE, if configured.
///
/// Returns `None` when multilevel is disabled or p is below the threshold.
/// When coordinates are available, uses Nystrom+rSVD LD estimation.
/// Otherwise falls back to regular fixed-size blocks.
pub(crate) fn estimate_multilevel_partitions(
    x_block: &DMatrix<f32>,
    config: &FitConfig,
    coords: Option<&SnpCoordinates>,
) -> Result<Option<Vec<BlockPartition>>> {
    let ml_config = match &config.multilevel {
        Some(c) => c,
        None => return Ok(None),
    };

    let p = x_block.ncols();
    if p < ml_config.min_p {
        return Ok(None);
    }

    let level0 = if let Some(c) = coords {
        let (pos, chr) = (c.positions, c.chromosomes);
        let num_landmarks = (p / 2).clamp(20, 500);
        let num_components = (num_landmarks / 5).clamp(5, 20);
        let ld_params = LdBlockParams {
            num_landmarks,
            num_components,
            min_block_snps: ml_config.min_block_snps,
            max_block_snps: ml_config.max_block_snps,
            seed: config.seed,
        };
        let ld_blocks = estimate_ld_blocks(x_block, pos, chr, &ld_params)?;
        if ld_blocks.is_empty() {
            return Ok(None);
        }
        let boundaries: Vec<usize> = ld_blocks.iter().map(|b| b.snp_start).collect();
        info!(
            "  Multilevel: {} LD sub-blocks from {} SNPs",
            boundaries.len(),
            p
        );
        BlockPartition::from_boundaries(&boundaries, p)
    } else {
        BlockPartition::regular(p, ml_config.max_block_snps)
    };

    let num_groups = level0.num_blocks();
    let upper_block_size = 50;
    let upper = BlockPartition::build_hierarchy(num_groups, upper_block_size);

    let mut partitions = vec![level0];
    partitions.extend(upper);

    Ok(Some(partitions))
}
