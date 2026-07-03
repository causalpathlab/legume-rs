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
