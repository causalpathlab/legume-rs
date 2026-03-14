/// Hard-assign each observation to the component with highest posterior probability.
/// Returns (observation_index, component_index) pairs.
pub fn hard_assign(gamma: &[Vec<f32>]) -> Vec<(usize, usize)> {
    gamma
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let best_k = g
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(k, _)| k)
                .unwrap_or(0);
            (i, best_k)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_assign() {
        let gamma = vec![vec![0.1, 0.9], vec![0.8, 0.2], vec![0.3, 0.7]];
        let assignments = hard_assign(&gamma);
        assert_eq!(assignments, vec![(0, 1), (1, 0), (2, 1)]);
    }

    #[test]
    fn test_hard_assign_empty() {
        let gamma: Vec<Vec<f32>> = vec![];
        let assignments = hard_assign(&gamma);
        assert!(assignments.is_empty());
    }
}
