use super::*;
use std::sync::Arc;

use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use matrix_param::traits::Inference;

fn make_test_data(
    dir: &tempfile::TempDir,
) -> Result<(SparseIoVec, CellAnnotations, CellTypeMembership)> {
    let path = dir.path().join("test.zarr");
    let path_str = path.to_str().unwrap();

    // 3 genes × 6 cells
    // IND_A: cell_0 (T_cell), cell_1 (T_cell), cell_2 (B_cell)
    // IND_B: cell_3 (T_cell), cell_4 (B_cell), cell_5 (B_cell)
    let triplets: Vec<(u64, u64, f32)> = vec![
        (0, 0, 5.0),
        (0, 1, 3.0),
        (0, 2, 2.0),
        (0, 3, 4.0),
        (0, 4, 1.0),
        (0, 5, 6.0),
        (1, 0, 10.0),
        (1, 1, 20.0),
        (1, 3, 15.0),
        (1, 5, 25.0),
        (2, 0, 1.0),
        (2, 2, 7.0),
        (2, 4, 3.0),
    ];

    let gene_names: Vec<Box<str>> = vec!["gene_0".into(), "gene_1".into(), "gene_2".into()];
    let cell_names: Vec<Box<str>> = (0..6).map(|i| format!("cell_{}", i).into()).collect();

    let nnz = triplets.len();
    let mut backend = create_sparse_from_triplets(
        &triplets,
        (3, 6, nnz),
        Some(path_str),
        Some(&SparseIoBackend::Zarr),
    )?;
    backend.register_row_names_vec(&gene_names);
    backend.register_column_names_vec(&cell_names);

    let mut cell_to_individual: HashMap<Box<str>, usize> = Default::default();
    cell_to_individual.insert(Box::from("cell_0"), 0); // IND_A
    cell_to_individual.insert(Box::from("cell_1"), 0); // IND_A
    cell_to_individual.insert(Box::from("cell_2"), 0); // IND_A
    cell_to_individual.insert(Box::from("cell_3"), 1); // IND_B
    cell_to_individual.insert(Box::from("cell_4"), 1); // IND_B
    cell_to_individual.insert(Box::from("cell_5"), 1); // IND_B

    let annotations = CellAnnotations {
        cell_to_individual,
        individual_ids: vec![Box::from("IND_A"), Box::from("IND_B")],
    };

    // One-hot: T_cell=col0, B_cell=col1
    // cell_0 T, cell_1 T, cell_2 B, cell_3 T, cell_4 B, cell_5 B
    let membership = CellTypeMembership {
        matrix: DMatrix::from_row_slice(
            6,
            2,
            &[
                1.0, 0.0, // cell_0: T
                1.0, 0.0, // cell_1: T
                0.0, 1.0, // cell_2: B
                1.0, 0.0, // cell_3: T
                0.0, 1.0, // cell_4: B
                0.0, 1.0, // cell_5: B
            ],
        ),
        cell_type_names: vec![Box::from("T_cell"), Box::from("B_cell")],
    };

    let mut data_vec = SparseIoVec::new();
    data_vec.push(Arc::from(backend), None)?;

    Ok((data_vec, annotations, membership))
}

#[test]
fn test_collapse_hard_membership() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (data_vec, annotations, membership) = make_test_data(&dir)?;

    let result = collapse_pseudobulk(data_vec, &annotations, &membership, 1.0, 1.0)?;

    assert_eq!(result.individual_ids.len(), 2);
    assert_eq!(result.cell_type_names.len(), 2);
    assert_eq!(result.gene_names.len(), 3);
    assert_eq!(result.gamma_params.len(), 2);

    let t_idx = result
        .cell_type_names
        .iter()
        .position(|x| x.as_ref() == "T_cell")
        .unwrap();
    let b_idx = result
        .cell_type_names
        .iter()
        .position(|x| x.as_ref() == "B_cell")
        .unwrap();
    let a_idx = result
        .individual_ids
        .iter()
        .position(|x| x.as_ref() == "IND_A")
        .unwrap();
    let b_ind_idx = result
        .individual_ids
        .iter()
        .position(|x| x.as_ref() == "IND_B")
        .unwrap();

    // T_cell weights: IND_A=2.0, IND_B=1.0
    assert!((result.cell_weights[t_idx][a_idx] - 2.0).abs() < 1e-6);
    assert!((result.cell_weights[t_idx][b_ind_idx] - 1.0).abs() < 1e-6);

    // B_cell weights: IND_A=1.0, IND_B=2.0
    assert!((result.cell_weights[b_idx][a_idx] - 1.0).abs() < 1e-6);
    assert!((result.cell_weights[b_idx][b_ind_idx] - 2.0).abs() < 1e-6);

    // Posterior mean = (a0 + sum) / (b0 + weight)
    // T_cell, IND_A, gene 0: sum=5+3=8, weight=2 → (1+8)/(1+2) = 3.0
    let t_mean = result.gamma_params[t_idx].posterior_mean();
    let expected = (1.0 + 8.0) / (1.0 + 2.0);
    assert!(
        (t_mean[(0, a_idx)] - expected).abs() < 1e-4,
        "T_cell IND_A gene0: got {}, expected {}",
        t_mean[(0, a_idx)],
        expected
    );

    // T_cell, IND_A, gene 1: sum=10+20=30, weight=2 → (1+30)/(1+2) ≈ 10.33
    let expected = (1.0 + 30.0) / (1.0 + 2.0);
    assert!(
        (t_mean[(1, a_idx)] - expected).abs() < 1e-4,
        "T_cell IND_A gene1: got {}, expected {}",
        t_mean[(1, a_idx)],
        expected
    );

    // B_cell, IND_B, gene 0: sum=1+6=7, weight=2 → (1+7)/(1+2) ≈ 2.67
    let b_mean = result.gamma_params[b_idx].posterior_mean();
    let expected = (1.0 + 7.0) / (1.0 + 2.0);
    assert!(
        (b_mean[(0, b_ind_idx)] - expected).abs() < 1e-4,
        "B_cell IND_B gene0: got {}, expected {}",
        b_mean[(0, b_ind_idx)],
        expected
    );

    Ok(())
}

#[test]
fn test_collapse_soft_membership() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (data_vec, annotations, _) = make_test_data(&dir)?;

    let soft = CellTypeMembership {
        matrix: DMatrix::from_row_slice(
            6,
            2,
            &[
                0.7, 0.3, // cell_0: mostly T
                0.8, 0.2, // cell_1: mostly T
                0.1, 0.9, // cell_2: mostly B
                0.9, 0.1, // cell_3: mostly T
                0.2, 0.8, // cell_4: mostly B
                0.3, 0.7, // cell_5: mostly B
            ],
        ),
        cell_type_names: vec![Box::from("T_cell"), Box::from("B_cell")],
    };

    let result = collapse_pseudobulk(data_vec, &annotations, &soft, 1.0, 1.0)?;

    let t_idx = result
        .cell_type_names
        .iter()
        .position(|x| x.as_ref() == "T_cell")
        .unwrap();
    let a_idx = result
        .individual_ids
        .iter()
        .position(|x| x.as_ref() == "IND_A")
        .unwrap();

    // T_cell weight for IND_A = 0.7 + 0.8 + 0.1 = 1.6
    let expected_weight = 0.7 + 0.8 + 0.1;
    assert!(
        (result.cell_weights[t_idx][a_idx] - expected_weight).abs() < 1e-4,
        "T_cell IND_A weight: got {}, expected {}",
        result.cell_weights[t_idx][a_idx],
        expected_weight
    );

    let expected_sum = 5.0 * 0.7 + 3.0 * 0.8 + 2.0 * 0.1;
    let expected_mean = (1.0 + expected_sum) / (1.0 + expected_weight);
    let t_mean = result.gamma_params[t_idx].posterior_mean();
    assert!(
        (t_mean[(0, a_idx)] - expected_mean).abs() < 1e-3,
        "Soft T_cell IND_A gene0: got {}, expected {}",
        t_mean[(0, a_idx)],
        expected_mean
    );

    Ok(())
}

#[test]
fn test_collapse_posterior_sd_decreases_with_more_cells() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (data_vec, annotations, membership) = make_test_data(&dir)?;

    let result = collapse_pseudobulk(data_vec, &annotations, &membership, 1.0, 1.0)?;

    let t_idx = result
        .cell_type_names
        .iter()
        .position(|x| x.as_ref() == "T_cell")
        .unwrap();
    let a_idx = result
        .individual_ids
        .iter()
        .position(|x| x.as_ref() == "IND_A")
        .unwrap();
    let b_idx = result
        .individual_ids
        .iter()
        .position(|x| x.as_ref() == "IND_B")
        .unwrap();

    let t_sd = result.gamma_params[t_idx].posterior_sd();

    // IND_A has 2 T_cells, IND_B has 1 T_cell → IND_A should have lower SD
    assert!(
        t_sd[(0, a_idx)] < t_sd[(0, b_idx)],
        "More cells should give lower SD: IND_A={} vs IND_B={}",
        t_sd[(0, a_idx)],
        t_sd[(0, b_idx)]
    );

    Ok(())
}
