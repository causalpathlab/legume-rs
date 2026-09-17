use super::topic_dictionary;
use nalgebra::DMatrix;

/// Every column is a log-simplex over features, and a shared shift of all
/// topic positions (the abundance direction) leaves the readout unchanged.
#[test]
fn columns_are_log_simplices_and_a_shared_topic_shift_cancels() {
    let rho = DMatrix::<f32>::from_row_slice(4, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5]);
    let alpha = DMatrix::<f32>::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let beta = topic_dictionary(&rho, &alpha);
    assert_eq!(beta.shape(), (4, 3));
    for k in 0..3 {
        let mass: f32 = beta.column(k).iter().map(|x| x.exp()).sum();
        assert!((mass - 1.0).abs() < 1e-5, "column {k} sums to {mass}");
    }
    let shift = DMatrix::<f32>::from_row_slice(3, 2, &[2.0, -3.0, 2.0, -3.0, 2.0, -3.0]);
    let shifted = topic_dictionary(&rho, &(&alpha + &shift));
    for (a, b) in beta.iter().zip(shifted.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "shift changed the readout: {a} vs {b}"
        );
    }
    // A topic aligned with feature 0 ranks it first.
    let col0: Vec<f32> = beta.column(0).iter().copied().collect();
    assert!(col0[0] > col0[1] && col0[0] > col0[3]);
}
