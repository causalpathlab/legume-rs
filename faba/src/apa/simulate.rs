use crate::apa::fragment::FragmentRecord;
use genomic_data::sam::{CellBarcode, UmiBarcode};
use rand::distr::weighted::WeightedIndex;
use rand::distr::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

/// Parameters for generating synthetic fragments following the SCAPE model (Section 2.1).
pub struct ScapeSimParams {
    /// UTR length L
    pub utr_length: f32,
    /// Mixture weights [pi_0 (noise), pi_1, ..., pi_K]
    pub weights: Vec<f32>,
    /// APA site positions [alpha_1, ..., alpha_K]
    pub alphas: Vec<f32>,
    /// APA site dispersions [beta_1, ..., beta_K]
    pub betas: Vec<f32>,
    /// Fragment length mean
    pub mu_f: f32,
    /// Fragment length std dev
    pub sigma_f: f32,
    /// Minimum polyA tail length
    pub min_polya: f32,
    /// Maximum polyA tail length
    pub max_polya: f32,
    /// Number of fragments to generate
    pub n_fragments: usize,
    /// Number of cells (for cell barcode assignment)
    pub n_cells: usize,
    /// Probability a fragment from an APA component is a junction read
    pub junction_prob: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for ScapeSimParams {
    fn default() -> Self {
        Self {
            utr_length: 2000.0,
            weights: vec![0.1, 0.45, 0.45],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            mu_f: 300.0,
            sigma_f: 50.0,
            min_polya: 20.0,
            max_polya: 150.0,
            n_fragments: 3000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 42,
        }
    }
}

/// Simulate fragments from the SCAPE generative model.
/// Returns (fragments, true_component_labels) where label 0 = noise, 1..K = APA sites.
pub fn simulate_fragments(params: &ScapeSimParams) -> (Vec<FragmentRecord>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(params.seed);
    let comp_dist = WeightedIndex::new(&params.weights).expect("invalid weights");

    let cell_dist = Uniform::new(0usize, params.n_cells).expect("invalid cell range");

    let mut fragments = Vec::with_capacity(params.n_fragments);
    let mut labels = Vec::with_capacity(params.n_fragments);

    let mut attempts = 0;
    let max_attempts = params.n_fragments * 20;

    while fragments.len() < params.n_fragments && attempts < max_attempts {
        attempts += 1;

        let k = comp_dist.sample(&mut rng);
        let cell_idx = cell_dist.sample(&mut rng);
        let cell_barcode = CellBarcode::Barcode(format!("CELL{:04}", cell_idx).into());
        let umi = UmiBarcode::Barcode(format!("UMI{:08}", fragments.len()).into());

        if k == 0 {
            // Noise component: uniform over UTR
            let x_dist = Uniform::new(1.0f32, params.utr_length + 1.0).unwrap();
            let x = x_dist.sample(&mut rng);
            let max_l = params.utr_length - x + 1.0;
            if max_l < 1.0 {
                continue;
            }
            let l_dist = Uniform::new(1.0f32, max_l + 1.0).unwrap();
            let l = l_dist.sample(&mut rng).floor();

            fragments.push(FragmentRecord {
                x,
                l,
                r: 0.0,
                is_junction: false,
                pa_site: None,
                cell_barcode,
                umi,
            });
            labels.push(0);
        } else {
            // APA component k (1-indexed in weights, but alphas/betas are 0-indexed)
            let alpha = params.alphas[k - 1];
            let beta = params.betas[k - 1];

            // Sample theta ~ N(alpha, beta^2)
            let z: f32 =
                <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng) as f32;
            let theta = alpha + beta * z;

            // Sample s ~ U(min_polya, max_polya)
            let s_dist = Uniform::new(params.min_polya, params.max_polya).unwrap();
            let s = s_dist.sample(&mut rng);

            // Fragment length f ~ N(mu_f, sigma_f^2)
            let z_f: f32 =
                <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng) as f32;
            let f_len = params.mu_f + params.sigma_f * z_f;

            // Break position: x = theta - (f - s) + 1
            let x = theta - (f_len - s) + 1.0;

            // Aligned length: l ~ U(1, theta - x + 1)
            let max_l = theta - x + 1.0;
            if max_l < 1.0 || x < 1.0 || x + max_l > params.utr_length + 1.0 {
                continue;
            }
            let l_dist = Uniform::new(1.0f32, max_l + 1.0).unwrap();
            let l: f32 = l_dist.sample(&mut rng).floor().min(max_l);

            // Junction read with probability junction_prob
            let is_junction = rng_bool(&mut rng, params.junction_prob);

            let (r, pa_site) = if is_junction {
                (s, Some(theta))
            } else {
                (0.0, None)
            };

            // Bounds check
            if x < 1.0 || l < 1.0 || x + l > params.utr_length + 1.0 {
                continue;
            }

            fragments.push(FragmentRecord {
                x,
                l,
                r,
                is_junction,
                pa_site,
                cell_barcode,
                umi,
            });
            labels.push(k);
        }
    }

    (fragments, labels)
}

fn rng_bool(rng: &mut StdRng, p: f32) -> bool {
    use rand::RngExt;
    rng.random_bool(p as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_basic() {
        let params = ScapeSimParams {
            n_fragments: 500,
            ..Default::default()
        };
        let (frags, labels) = simulate_fragments(&params);
        assert_eq!(frags.len(), labels.len());
        assert!(frags.len() > 100, "should generate substantial fragments");

        // Check labels are within range
        for &lbl in &labels {
            assert!(lbl < params.weights.len());
        }

        // Check some junction reads exist
        let n_junction = frags.iter().filter(|f| f.is_junction).count();
        assert!(n_junction > 0, "should have junction reads");

        // Check noise reads exist
        let n_noise = labels.iter().filter(|&&l| l == 0).count();
        assert!(n_noise > 0, "should have noise reads");
    }

    #[test]
    fn test_simulate_deterministic() {
        let params = ScapeSimParams {
            n_fragments: 200,
            seed: 123,
            ..Default::default()
        };
        let (frags1, labels1) = simulate_fragments(&params);
        let (frags2, labels2) = simulate_fragments(&params);
        assert_eq!(labels1, labels2);
        assert_eq!(frags1.len(), frags2.len());
        for (a, b) in frags1.iter().zip(frags2.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.l, b.l);
        }
    }
}
