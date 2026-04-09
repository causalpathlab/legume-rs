//! Post-hoc refinement of fine-mapping results.
//!
//! Selects high-PIP variants via an elbow heuristic and refits a joint model.

use anyhow::Result;
use candle_util::candle_core::Device;
use log::info;
use matrix_util::dmatrix_util::{subset_columns, subset_rows};
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rustc_hash::FxHashMap;

use crate::sgvb::{fit_block_rss, FitConfig, RssParams};
use crate::summary_stats::common::BlockFitResult;
use crate::summary_stats::LdBlock;

/// Find the elbow/knee point in sorted (descending) candidates.
pub fn find_pip_elbow(candidates: &[(usize, f32)]) -> usize {
    let n = candidates.len();
    if n < 3 {
        return n;
    }
    debug_assert!(
        candidates.windows(2).all(|w| w[0].1 >= w[1].1),
        "find_pip_elbow requires descending-sorted input"
    );

    let mid_pip = candidates[n / 2].1;
    let null_thresh = (2.0 * mid_pip).max(0.01);
    let above_null = candidates.partition_point(|&(_, p)| p >= null_thresh);
    if above_null < 3 {
        return above_null;
    }

    let y1 = candidates[0].1 as f64;
    let x2 = (above_null - 1) as f64;
    let y2 = candidates[above_null - 1].1 as f64;

    let dy = y2 - y1;
    let line_len = (x2 * x2 + dy * dy).sqrt();
    if line_len < 1e-12 {
        return above_null;
    }

    let mut max_dist = 0.0f64;
    let mut elbow_idx = 0;
    for (i, &(_, pip)) in candidates.iter().enumerate().take(above_null - 1).skip(1) {
        let px = i as f64;
        let py = pip as f64 - y1;
        let dist = (px * dy - py * x2).abs() / line_len;
        if dist > max_dist {
            max_dist = dist;
            elbow_idx = i;
        }
    }

    elbow_idx + 1
}

/// Parameters for joint refinement of high-PIP variants.
pub struct RefinementParams {
    pub max_variants: usize,
    pub user_lambda: Option<f64>,
    pub ldsc_intercept: bool,
}

/// Input data for joint refinement.
pub struct RefinementInput<'a> {
    pub blocks: &'a [LdBlock],
    pub genotypes: &'a DMatrix<f32>,
    pub zscores: &'a DMatrix<f32>,
    pub snp_ids: &'a [Box<str>],
    pub num_traits: usize,
}

/// Joint refinement: refit a single model on high-PIP variants selected by elbow.
pub fn refine_high_pip_variants(
    mut globally_averaged: Vec<(usize, BlockFitResult)>,
    input: &RefinementInput<'_>,
    fit_config: &FitConfig,
    params: &RefinementParams,
    device: &Device,
) -> Result<Vec<(usize, BlockFitResult)>> {
    let max_variants = params.max_variants;
    let blocks = input.blocks;
    let genotypes = input.genotypes;
    let zscores = input.zscores;
    let snp_ids = input.snp_ids;
    let t = input.num_traits;
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for (block_idx, result) in &globally_averaged {
        let block = &blocks[*block_idx];
        for snp_j in 0..block.num_snps() {
            let max_pip = (0..t).map(|k| result.pip[(snp_j, k)]).fold(0f32, f32::max);
            candidates.push((block.snp_start + snp_j, max_pip));
        }
    }

    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

    let elbow_n = find_pip_elbow(&candidates);
    let elbow_pip = candidates
        .get(elbow_n.saturating_sub(1))
        .map_or(0.0, |c| c.1);
    candidates.truncate(elbow_n);

    if candidates.len() < 2 {
        info!(
            "Refinement: elbow at {} variants (PIP >= {:.4}), too few — skipping",
            elbow_n, elbow_pip,
        );
        return Ok(globally_averaged);
    }

    if candidates.len() > max_variants {
        log::warn!(
            "Refinement: elbow selected {} variants, capping at {}",
            candidates.len(),
            max_variants,
        );
        candidates.truncate(max_variants);
    }

    let p_sel = candidates.len();
    info!(
        "Refinement: elbow at {} variants (PIP >= {:.4}), fitting joint model",
        p_sel, elbow_pip,
    );

    let sel_indices: Vec<usize> = candidates.iter().map(|&(j, _)| j).collect();
    let mut x_joint = subset_columns(genotypes, sel_indices.iter().copied())?;
    let z_joint = subset_rows(zscores, sel_indices.iter().copied())?;
    x_joint.scale_columns_inplace();

    let n = genotypes.nrows();
    let joint_max_rank = n.min(p_sel);
    let joint_lambda = params.user_lambda.unwrap_or(0.1 / joint_max_rank as f64);

    let mut joint_config = fit_config.clone();
    joint_config.num_components = joint_config.num_components.min(p_sel / 2).max(1);
    joint_config.seed = fit_config.seed.wrapping_add(999);

    let rss_params = RssParams {
        max_rank: joint_max_rank,
        lambda: joint_lambda,
        ldsc_intercept: params.ldsc_intercept,
    };

    let joint_result = fit_block_rss(&x_joint, &z_joint, &joint_config, &rss_params, device)?;
    let refined = joint_result.best_result();

    info!(
        "Refinement: joint ELBO={:.2}, L={}",
        refined.avg_elbo, joint_config.num_components,
    );

    let mut hits: Vec<(f32, usize, usize)> = Vec::new();
    for (j_new, &(j_global, _)) in candidates.iter().enumerate() {
        for trait_k in 0..t {
            let pip = refined.pip[(j_new, trait_k)];
            if pip >= 0.5 {
                hits.push((pip, j_global, trait_k));
            }
        }
    }
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));
    for &(pip, global_snp, trait_k) in hits.iter().take(10) {
        let z = zscores[(global_snp, trait_k)];
        info!(
            "  ** refined {}: trait={}, pip={:.4}, z={:.2}",
            snp_ids[global_snp], trait_k, pip, z,
        );
    }

    let sel_lookup: FxHashMap<usize, usize> = candidates
        .iter()
        .enumerate()
        .map(|(j_new, &(j_global, _))| (j_global, j_new))
        .collect();

    for (block_idx, result) in &mut globally_averaged {
        let block = &blocks[*block_idx];
        for snp_j in 0..block.num_snps() {
            let global_snp = block.snp_start + snp_j;
            if let Some(&j_new) = sel_lookup.get(&global_snp) {
                for trait_k in 0..t {
                    result.pip[(snp_j, trait_k)] = refined.pip[(j_new, trait_k)];
                    result.effect_mean[(snp_j, trait_k)] = refined.effect_mean[(j_new, trait_k)];
                    result.effect_std[(snp_j, trait_k)] = refined.effect_std[(j_new, trait_k)];
                }
            }
        }
    }

    Ok(globally_averaged)
}
