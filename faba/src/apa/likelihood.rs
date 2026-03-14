use crate::apa::fragment::FragmentRecord;

/// Parameters for the SCAPE likelihood model.
pub struct LikelihoodParams {
    /// Expected mean fragment length (mu_f), default 300
    pub mu_f: f32,
    /// Standard deviation of fragment length (sigma_f), default 50
    pub sigma_f: f32,
    /// Step size for theta enumeration, default 10
    pub theta_step: usize,
    /// Maximum polyA length (LA), default 150
    pub max_polya: f32,
    /// Minimum polyA length, default 20
    pub min_polya: f32,
}

impl Default for LikelihoodParams {
    fn default() -> Self {
        Self {
            mu_f: 300.0,
            sigma_f: 50.0,
            theta_step: 10,
            max_polya: 150.0,
            min_polya: 20.0,
        }
    }
}

/// Compute log p(x_n, l_n, r_n | theta_nk) for a single fragment and theta value.
///
/// Following SCAPE eq 5-10:
/// p(x,l,r|theta) = sum_s f(x,l,r,s,theta)
/// where f = p(l|x,theta) * p(x|s,theta) * p(r|s) * p(s)
///
/// For SE reads: s is marginalized out by enumeration.
/// For junction reads (r > 0, pa_site known): s is known, theta = pa_site.
pub fn log_lik_fragment_given_theta(
    frag: &FragmentRecord,
    theta: f32,
    utr_length: f32,
    params: &LikelihoodParams,
) -> f32 {
    let x = frag.x;
    let l = frag.l;
    let r = frag.r;

    // Junction read: s is known (s = r), theta = pa_site
    if frag.is_junction && r > 0.0 {
        // For junction reads, set p(x|s,theta) = 1 and p(r|s) = 1
        // Only contribution is from p(l|x,theta)
        let max_l = theta - x + 1.0;
        if l >= 1.0 && l <= max_l && max_l > 0.0 {
            return -(max_l).ln();
        } else {
            return f32::NEG_INFINITY;
        }
    }

    // Non-junction read: marginalize over s
    let s_min = params.min_polya;
    let s_max = params.max_polya;
    let s_step = params.theta_step as f32;

    let mut log_sum = f32::NEG_INFINITY;

    let mut s = s_min;
    while s <= s_max {
        let log_f = compute_log_f(x, l, r, s, theta, utr_length, params);
        log_sum = log_sum_exp(log_sum, log_f);
        s += s_step;
    }

    log_sum
}

/// Compute log f(x, l, r, s, theta) = log[p(l|x,theta) * p(x|s,theta) * p(r|s) * p(s)]
fn compute_log_f(
    x: f32,
    l: f32,
    r: f32,
    s: f32,
    theta: f32,
    _utr_length: f32,
    params: &LikelihoodParams,
) -> f32 {
    // p(s) = 1/(LA - min_polya) for s in [min_polya, LA]
    let s_range = params.max_polya - params.min_polya;
    if s_range <= 0.0 || s < params.min_polya || s > params.max_polya {
        return f32::NEG_INFINITY;
    }
    let log_ps = -s_range.ln();

    // p(r|s) = 1/s for 1 <= r <= s, else 0
    // For non-junction reads with r=0, we treat r as unobserved -> p(r|s) = 1
    let log_pr_s = if r > 0.0 {
        if r >= 1.0 && r <= s {
            -s.ln()
        } else {
            return f32::NEG_INFINITY;
        }
    } else {
        0.0 // r unobserved
    };

    // p(x|s,theta) = N(theta + s + 1 - mu_f, sigma_f^2) evaluated at x
    // i.e., x ~ N(theta + s + 1 - mu_f, sigma_f^2)
    let mean_x = theta + s + 1.0 - params.mu_f;
    let log_px_s_theta = log_normal_pdf(x, mean_x, params.sigma_f);

    // p(l|x,theta) = 1/(theta - x + 1) for 1 <= l <= theta - x + 1
    let max_l = theta - x + 1.0;
    let log_pl_x_theta = if max_l > 0.0 && l >= 1.0 && l <= max_l {
        -max_l.ln()
    } else {
        return f32::NEG_INFINITY;
    };

    log_ps + log_pr_s + log_px_s_theta + log_pl_x_theta
}

/// Log of the noise model likelihood (eq 13): 1/(L^2 * LA)
pub fn log_lik_noise(utr_length: f32, max_polya: f32) -> f32 {
    -2.0 * utr_length.ln() - max_polya.ln()
}

/// Precompute the theta likelihood matrix for all fragments.
/// Returns a matrix of shape [n_fragments, n_theta_values] containing
/// log p(x_n, l_n, r_n | theta) for each fragment and theta grid point.
///
/// theta_grid: the grid of theta values to evaluate (1, 1+step, 1+2*step, ..., L)
pub fn precompute_theta_lik_matrix(
    fragments: &[FragmentRecord],
    utr_length: f32,
    params: &LikelihoodParams,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let step = params.theta_step;
    let theta_grid: Vec<f32> = (1..=utr_length as usize)
        .step_by(step)
        .map(|t| t as f32)
        .collect();

    let n_frag = fragments.len();
    let n_theta = theta_grid.len();

    let mut lik_matrix = vec![vec![f32::NEG_INFINITY; n_theta]; n_frag];

    for (n, frag) in fragments.iter().enumerate() {
        for (t, &theta) in theta_grid.iter().enumerate() {
            lik_matrix[n][t] = log_lik_fragment_given_theta(frag, theta, utr_length, params);
        }
    }

    (lik_matrix, theta_grid)
}

/// Compute log p(x_n, l_n, r_n | alpha_k, beta_k) by marginalizing theta
/// over the Gaussian N(alpha_k, beta_k^2) weighted by precomputed theta likelihoods.
///
/// log p(x,l,r|alpha,beta) = log sum_theta p(x,l,r|theta) * N(theta|alpha, beta^2)
pub fn log_lik_fragment_given_site(
    frag_theta_liks: &[f32],
    theta_grid: &[f32],
    alpha: f32,
    beta: f32,
) -> f32 {
    let mut log_sum = f32::NEG_INFINITY;

    for (t, &theta) in theta_grid.iter().enumerate() {
        let log_lik = frag_theta_liks[t];
        if log_lik.is_finite() {
            let log_prior = log_normal_pdf(theta, alpha, beta);
            log_sum = log_sum_exp(log_sum, log_lik + log_prior);
        }
    }

    log_sum
}

/// Compute log-normal PDF: log N(x | mu, sigma)
fn log_normal_pdf(x: f32, mu: f32, sigma: f32) -> f32 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * std::f32::consts::TAU.ln()
}

/// Numerically stable log(exp(a) + exp(b))
pub fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_log_lik() {
        let ll = log_lik_noise(1000.0, 150.0);
        let expected = -2.0 * 1000.0_f32.ln() - 150.0_f32.ln();
        assert!(
            (ll - expected).abs() < 1e-5,
            "noise log-lik: got {}, expected {}",
            ll,
            expected
        );
    }

    #[test]
    fn test_junction_lik_peaks_at_correct_theta() {
        let params = LikelihoodParams::default();
        // Junction fragment at theta=500: x=250, l=200, r=50
        let frag = FragmentRecord {
            x: 250.0,
            l: 200.0,
            r: 50.0,
            is_junction: true,
            pa_site: Some(500.0),
            cell_barcode: genomic_data::sam::CellBarcode::Missing,
            umi: genomic_data::sam::UmiBarcode::Missing,
        };

        let ll_500 = log_lik_fragment_given_theta(&frag, 500.0, 2000.0, &params);
        let ll_200 = log_lik_fragment_given_theta(&frag, 200.0, 2000.0, &params);

        assert!(
            ll_500 > ll_200,
            "junction lik at true theta (500) should exceed distant theta (200): {} vs {}",
            ll_500,
            ll_200
        );
        // theta=200 should be -inf since l=200 > theta-x+1 = 200-250+1 = -49
        assert!(
            ll_200.is_infinite() && ll_200 < 0.0,
            "ll at theta=200 should be -inf for this fragment"
        );
    }

    #[test]
    fn test_lik_fragment_given_site_peaks_near_alpha() {
        let params = LikelihoodParams {
            theta_step: 5,
            ..Default::default()
        };
        // Non-junction fragment generated near alpha=500
        let frag = FragmentRecord {
            x: 350.0,
            l: 100.0,
            r: 0.0,
            is_junction: false,
            pa_site: None,
            cell_barcode: genomic_data::sam::CellBarcode::Missing,
            umi: genomic_data::sam::UmiBarcode::Missing,
        };

        let utr_length = 2000.0;
        let step = params.theta_step;
        let theta_grid: Vec<f32> = (1..=utr_length as usize)
            .step_by(step)
            .map(|t| t as f32)
            .collect();

        let frag_theta_liks: Vec<f32> = theta_grid
            .iter()
            .map(|&theta| log_lik_fragment_given_theta(&frag, theta, utr_length, &params))
            .collect();

        let ll_correct = log_lik_fragment_given_site(&frag_theta_liks, &theta_grid, 500.0, 30.0);
        let ll_distant = log_lik_fragment_given_site(&frag_theta_liks, &theta_grid, 1500.0, 30.0);

        assert!(
            ll_correct > ll_distant,
            "fragment near alpha=500 should have higher lik at alpha=500 than alpha=1500: {} vs {}",
            ll_correct,
            ll_distant
        );
    }

    #[test]
    fn test_log_sum_exp_basic() {
        // log(exp(0) + exp(0)) = log(2)
        let result = log_sum_exp(0.0, 0.0);
        assert!((result - 2.0_f32.ln()).abs() < 1e-5);

        // log(exp(-inf) + exp(5)) = 5
        assert!((log_sum_exp(f32::NEG_INFINITY, 5.0) - 5.0).abs() < 1e-5);
    }
}
