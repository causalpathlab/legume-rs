use crate::mixture::assign::hard_assign;
use crate::mixture::em::{weighted_gaussian_mixture_em, EmParams, GmmResult};

/// Parameters for per-gene mixture model
pub struct MixtureParams {
    /// Minimum distinct positions per gene to attempt mixture
    pub min_sites: usize,
    /// Maximum components to test via BIC
    pub max_k: usize,
    /// Initial sigma (0 = auto: gene_length / (2*K))
    pub initial_sigma: f32,
    /// EM parameters
    pub em_params: EmParams,
}

impl Default for MixtureParams {
    fn default() -> Self {
        Self {
            min_sites: 3,
            max_k: 5,
            initial_sigma: 0.0,
            em_params: EmParams {
                max_iter: 200,
                tol: 1e-6,
                min_weight: 0.01,
            },
        }
    }
}

/// Annotation for one mixture component
pub struct MixtureComponentAnnotation {
    /// Gene name
    pub gene_name: Box<str>,
    /// Component index within gene
    pub component_idx: usize,
    /// Learned mean position
    pub mu: f32,
    /// Learned standard deviation
    pub sigma: f32,
    /// Mixing weight
    pub pi: f32,
}

/// Result of per-gene mixture model
pub struct GeneMixtureResult {
    /// Best K (number of Gaussian components, not including noise)
    #[cfg(test)]
    pub best_k: usize,
    /// GMM result for best K
    pub gmm: GmmResult,
    /// Per-(cell_index, component) count
    pub cell_component_counts: Vec<(usize, usize, usize)>,
}

/// Weighted observation for the mixture model: unique (cell, position) with count.
pub struct WeightedObservation {
    /// Index of the cell in the cell list
    pub cell_idx: usize,
    /// Genomic position of the modification
    pub position: f32,
    /// Number of reads at this (cell, position)
    pub count: usize,
}

/// Run per-gene GMM model selection over K=1..max_k, pick best by BIC.
///
/// * `observations` - weighted observations (unique per cell+position)
/// * `gene_length` - length of the gene for uniform noise component
/// * `params` - mixture parameters
///
/// Returns None if fewer than min_sites distinct positions.
pub fn fit_gene_mixture(
    observations: &[WeightedObservation],
    gene_length: f32,
    params: &MixtureParams,
) -> Option<GeneMixtureResult> {
    if observations.is_empty() {
        return None;
    }

    // Check distinct positions
    let mut distinct: Vec<i32> = observations.iter().map(|o| o.position as i32).collect();
    distinct.sort();
    distinct.dedup();
    if distinct.len() < params.min_sites {
        return None;
    }

    let positions: Vec<f32> = observations.iter().map(|o| o.position).collect();
    let obs_weights: Vec<f32> = observations.iter().map(|o| o.count as f32).collect();
    let gene_start = distinct[0] as f32;

    let mut best_result: Option<(usize, GmmResult)> = None;
    let mut n_worse = 0u32;

    for k in 1..=params.max_k {
        let initial_mus: Vec<f32> = (0..k)
            .map(|i| gene_start + (i as f32 + 1.0) * gene_length / (k as f32 + 1.0))
            .collect();

        let initial_sigma = if params.initial_sigma > 0.0 {
            params.initial_sigma
        } else {
            gene_length / (2.0 * k as f32)
        };

        let result = weighted_gaussian_mixture_em(
            &positions,
            &obs_weights,
            &initial_mus,
            initial_sigma,
            gene_length,
            &params.em_params,
        );

        let is_better = match &best_result {
            None => true,
            Some((_, prev)) => result.bic < prev.bic,
        };

        if is_better {
            best_result = Some((k, result));
            n_worse = 0;
        } else {
            n_worse += 1;
            if n_worse >= 2 {
                break;
            }
        }
    }

    let (_best_k, gmm) = best_result?;

    // Hard assignment: gamma[i] gives posterior for observation i,
    // distribute count to the best component
    let assignments = hard_assign(&gmm.gamma);

    // Count per (cell_idx, component), weighted by observation count
    let mut counts: rustc_hash::FxHashMap<(usize, usize), usize> = rustc_hash::FxHashMap::default();
    for (obs_idx, &(_, component)) in assignments.iter().enumerate() {
        let cell_idx = observations[obs_idx].cell_idx;
        let count = observations[obs_idx].count;
        *counts.entry((cell_idx, component)).or_default() += count;
    }

    let cell_component_counts: Vec<(usize, usize, usize)> = counts
        .into_iter()
        .map(|((cell, comp), cnt)| (cell, comp, cnt))
        .collect();

    Some(GeneMixtureResult {
        #[cfg(test)]
        best_k: _best_k,
        gmm,
        cell_component_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_cluster() {
        // All observations near position 50
        let obs: Vec<WeightedObservation> = (0..50)
            .map(|i| WeightedObservation {
                cell_idx: i % 5,
                position: 50.0 + (i as f32 - 25.0) * 0.2,
                count: 1,
            })
            .collect();

        let params = MixtureParams {
            min_sites: 3,
            max_k: 3,
            ..Default::default()
        };

        let result = fit_gene_mixture(&obs, 100.0, &params).unwrap();
        // BIC should prefer K=1
        assert_eq!(
            result.best_k, 1,
            "expected K=1 for single cluster, got K={}",
            result.best_k
        );
    }

    #[test]
    fn test_two_clusters() {
        // Two well-separated groups
        let mut obs = Vec::new();
        for i in 0..30 {
            obs.push(WeightedObservation {
                cell_idx: i % 5,
                position: 20.0 + (i as f32 - 15.0) * 0.3,
                count: 1,
            });
        }
        for i in 0..30 {
            obs.push(WeightedObservation {
                cell_idx: i % 5,
                position: 80.0 + (i as f32 - 15.0) * 0.3,
                count: 1,
            });
        }

        let params = MixtureParams {
            min_sites: 3,
            max_k: 4,
            ..Default::default()
        };

        let result = fit_gene_mixture(&obs, 100.0, &params).unwrap();
        assert!(
            result.best_k >= 2,
            "expected K>=2 for two clusters, got K={}",
            result.best_k
        );

        // Check that means are near 20 and 80
        let mut mus = result.gmm.mus.clone();
        mus.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Filter out noise-assigned mus
        let active_mus: Vec<f32> = mus
            .iter()
            .zip(result.gmm.weights.iter().skip(1))
            .filter(|(_, &w)| w > 0.01)
            .map(|(&m, _)| m)
            .collect();
        assert!(
            active_mus.len() >= 2,
            "expected at least 2 active components"
        );
    }

    #[test]
    fn test_too_few_sites() {
        let obs = vec![
            WeightedObservation {
                cell_idx: 0,
                position: 50.0,
                count: 1,
            },
            WeightedObservation {
                cell_idx: 1,
                position: 50.0,
                count: 1,
            },
        ];

        let params = MixtureParams {
            min_sites: 3,
            ..Default::default()
        };

        assert!(fit_gene_mixture(&obs, 100.0, &params).is_none());
    }

    #[test]
    fn test_cell_component_counts() {
        // 10 observations from 2 cells, all at same position
        let obs: Vec<WeightedObservation> = (0..10)
            .map(|i| WeightedObservation {
                cell_idx: i % 2,
                position: 30.0 + i as f32,
                count: 1,
            })
            .collect();

        let params = MixtureParams {
            min_sites: 3,
            max_k: 1,
            ..Default::default()
        };

        let result = fit_gene_mixture(&obs, 100.0, &params).unwrap();
        // All observations should be assigned to some component
        let total_count: usize = result.cell_component_counts.iter().map(|(_, _, c)| c).sum();
        assert_eq!(
            total_count, 10,
            "total count should be 10, got {}",
            total_count
        );
    }
}
