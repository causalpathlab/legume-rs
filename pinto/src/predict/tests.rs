//! Which arm a predicted sample takes is decided by the model prefix and the
//! flag alone, so it is checked without a model.

use super::{resolve_pair_solver, PredictPairSolver};

#[test]
fn the_encoder_is_taken_when_present_demanded_when_asked_and_ignored_when_exact() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("run").to_string_lossy().to_string();
    let file = format!("{model}.pair_encoder.safetensors");

    // Without the file: auto and exact fall back, encoder refuses.
    assert_eq!(
        resolve_pair_solver(&model, PredictPairSolver::Auto).unwrap(),
        None
    );
    assert_eq!(
        resolve_pair_solver(&model, PredictPairSolver::Exact).unwrap(),
        None
    );
    let err = resolve_pair_solver(&model, PredictPairSolver::Encoder).unwrap_err();
    assert!(
        err.to_string().contains("pair_encoder.safetensors"),
        "{err}"
    );

    // With it: auto and encoder take it, exact still ignores it.
    std::fs::write(&file, b"").unwrap();
    assert_eq!(
        resolve_pair_solver(&model, PredictPairSolver::Auto)
            .unwrap()
            .as_deref(),
        Some(file.as_str())
    );
    assert_eq!(
        resolve_pair_solver(&model, PredictPairSolver::Encoder)
            .unwrap()
            .as_deref(),
        Some(file.as_str())
    );
    assert_eq!(
        resolve_pair_solver(&model, PredictPairSolver::Exact).unwrap(),
        None
    );
}
