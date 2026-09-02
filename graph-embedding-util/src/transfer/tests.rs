use super::*;

const H: usize = 3;
const M: usize = 2;
const S: usize = 4;

/// Two modules, four training genes: 0,1 in module 0; 2,3 in module 1 (gene 3
/// mixed 0.25/0.75). Rows are `π μ + r` with a nonzero residual so the trained row
/// differs from the pure module composition.
fn trained() -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>, Vec<f32>) {
    let mu = DMatrix::<f32>::from_row_slice(M, H, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let pi = DMatrix::<f32>::from_row_slice(4, M, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.25, 0.75]);
    let r = DMatrix::<f32>::from_row_slice(
        4,
        H,
        &[
            0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.05, 0.05, 0.05,
        ],
    );
    let rho = &pi * &mu + r;
    let b = vec![-1.0, -2.0, -3.0, -4.0];
    (rho, pi, mu, b)
}

/// New-data profiles: genes 0..3 of the new axis match training rows 1, 3, 0 (in
/// that order, exercising a permutation); training gene 2 is MISSING. New genes 3
/// and 4 are unseen: 3 has exactly gene 0's profile, 4 resembles nothing.
fn new_data() -> (Vec<Option<usize>>, DMatrix<f32>) {
    let new_to_train = vec![Some(1), Some(3), Some(0), None, None];
    let p = DMatrix::<f32>::from_row_slice(
        5,
        S,
        &[
            10.0, 0.0, 10.0, 0.0, // new 0 = train 1
            0.0, 10.0, 0.0, 10.0, // new 1 = train 3
            10.0, 0.0, 10.0, 0.0, // new 2 = train 0
            10.0, 0.0, 10.0, 0.0, // new 3: identical to train 0's profile
            5.0, 5.0, 5.0, 5.0, // new 4: flat, resembles nothing after centring
        ],
    );
    (new_to_train, p)
}

fn align(k: usize, floor: f32) -> GeneAlignment {
    let (rho, pi, mu, b) = trained();
    let (n2t, p) = new_data();
    align_gene_axis(&AlignInputs {
        rho: &rho,
        b_feat: &b,
        modules: Some(ModuleTables { pi: &pi, mu: &mu }),
        new_to_train: &n2t,
        profiles_new: Some(&p),
        k,
        similarity_floor: floor,
    })
}

#[test]
fn union_axis_is_training_genes_then_new_only_genes() {
    let a = align(1, 0.5);
    assert_eq!(a.n_union(), 4 + 2);
    assert_eq!(a.union_to_train[..4], [Some(0), Some(1), Some(2), Some(3)]);
    assert_eq!(a.union_to_train[4..], [None, None]);
    assert_eq!(
        a.union_to_new,
        vec![Some(2), Some(0), None, Some(1), Some(3), Some(4)]
    );
}

#[test]
fn matched_genes_keep_row_bias_and_membership_verbatim() {
    let (rho, pi, _, b) = trained();
    let a = align(1, 0.5);
    for g in [0usize, 1, 3] {
        assert_eq!(a.status[g], GeneStatus::Matched);
        assert_eq!(a.rows.row(g), rho.row(g));
        assert_eq!(a.bias[g], b[g]);
        assert_eq!(a.membership.as_ref().unwrap().row(g), pi.row(g));
        assert!(a.provenance[g].is_none());
    }
}

#[test]
fn missing_gene_keeps_its_row_and_is_flagged() {
    let (rho, _, _, b) = trained();
    let a = align(1, 0.5);
    assert_eq!(a.status[2], GeneStatus::Missing);
    assert_eq!(a.rows.row(2), rho.row(2));
    assert_eq!(a.bias[2], b[2]);
}

#[test]
fn identical_profile_inherits_the_neighbours_membership_without_its_residual() {
    let (_, pi, mu, _) = trained();
    let a = align(1, 0.5);
    let g = 4; // union index of new gene 3
    assert_eq!(a.status[g], GeneStatus::Initialized);
    let pm = a.membership.as_ref().unwrap();
    // Nearest matched gene by profile is train 0 (or train 1, identical profile and
    // identical membership), so π̂ = (1, 0) and ρ̂ = μ_0 exactly — no residual.
    assert_eq!(pm.row(g), pi.row(0));
    let want = pi.row(0) * &mu;
    for j in 0..H {
        assert!((a.rows[(g, j)] - want[j]).abs() < 1e-6, "ρ̂ must be π̂ μ");
    }
    let prov = a.provenance[g].as_ref().unwrap();
    assert!(!prov.diffuse);
    assert!(prov.best_similarity > 0.99);
    assert!(prov.neighbours[0] == 0 || prov.neighbours[0] == 1);
}

#[test]
fn k_neighbours_average_memberships_by_similarity() {
    let (_, pi, _, _) = trained();
    let a = align(3, 0.0);
    let g = 4;
    let pm = a.membership.as_ref().unwrap();
    // Neighbours of the flat-free profile (10,0,10,0): train 0 and 1 (sim 1), train 3
    // (sim −1, weight clipped to 0). So still (1, 0).
    assert_eq!(pm.row(g), pi.row(0));
    let row: Vec<f32> = pm.row(g).iter().copied().collect();
    assert!(
        (row.iter().sum::<f32>() - 1.0).abs() < 1e-6,
        "π̂ stays on the simplex"
    );
}

#[test]
fn a_gene_resembling_nothing_gets_the_module_average_and_is_marked_diffuse() {
    let (_, pi, mu, _) = trained();
    let a = align(2, 0.5);
    let g = 5; // union index of new gene 4 (flat profile → zero after centring)
    assert_eq!(a.status[g], GeneStatus::Initialized);
    let prov = a.provenance[g].as_ref().unwrap();
    assert!(prov.diffuse);
    assert!(prov.neighbours.is_empty());
    let pm = a.membership.as_ref().unwrap();
    let avg: Vec<f32> = (0..M)
        .map(|m| pi.column(m).iter().sum::<f32>() / 4.0)
        .collect();
    for m in 0..M {
        assert!((pm[(g, m)] - avg[m]).abs() < 1e-6);
    }
    let want = DMatrix::<f32>::from_row_slice(1, M, &avg) * &mu;
    for j in 0..H {
        assert!((a.rows[(g, j)] - want[(0, j)]).abs() < 1e-6);
    }
}

#[test]
fn without_profiles_unseen_genes_are_dropped() {
    let (rho, pi, mu, b) = trained();
    let (n2t, _) = new_data();
    let a = align_gene_axis(&AlignInputs {
        rho: &rho,
        b_feat: &b,
        modules: Some(ModuleTables { pi: &pi, mu: &mu }),
        new_to_train: &n2t,
        profiles_new: None,
        k: 3,
        similarity_floor: 0.5,
    });
    assert_eq!(a.status[4], GeneStatus::Dropped);
    assert_eq!(a.status[5], GeneStatus::Dropped);
    assert!(a.rows.row(4).iter().all(|&v| v == 0.0));
    assert_eq!(a.with_status(GeneStatus::Matched).len(), 3);
}

#[test]
fn without_modules_an_unseen_gene_takes_its_neighbours_row() {
    let (rho, _, _, b) = trained();
    let (n2t, p) = new_data();
    let a = align_gene_axis(&AlignInputs {
        rho: &rho,
        b_feat: &b,
        modules: None,
        new_to_train: &n2t,
        profiles_new: Some(&p),
        k: 1,
        similarity_floor: 0.5,
    });
    assert!(a.membership.is_none());
    let g = 4;
    assert_eq!(a.status[g], GeneStatus::Initialized);
    let n = a.provenance[g].as_ref().unwrap().neighbours[0];
    assert_eq!(a.rows.row(g), rho.row(n));
}

/// Planted bias: counts are the exact expected rates, so the moment-matched bias
/// must recover the planted `a_g` to float precision.
#[test]
fn moment_matching_recovers_a_planted_bias() {
    let rows = DMatrix::<f32>::from_row_slice(2, H, &[0.5, -0.2, 0.1, -0.3, 0.4, 0.2]);
    let theta =
        DMatrix::<f32>::from_row_slice(3, H, &[1.0, 0.0, 0.5, -0.5, 1.0, 0.0, 0.2, 0.2, 0.2]);
    let b_cell = [0.1f32, -0.4, 0.3];
    let planted = [-1.5f32, 0.7];
    let mut total = vec![0f32; 2];
    for g in 0..2 {
        for c in 0..3 {
            let s: f32 =
                (0..H).map(|j| rows[(g, j)] * theta[(c, j)]).sum::<f32>() + planted[g] + b_cell[c];
            total[g] += s.exp();
        }
    }
    let got = moment_matched_bias(&rows, &theta, &b_cell, &total, 0.0);
    for g in 0..2 {
        assert!(
            (got[g] - planted[g]).abs() < 1e-4,
            "gene {g}: {} vs {}",
            got[g],
            planted[g]
        );
    }
    let none = moment_matched_bias(&rows, &theta, &b_cell, &[0.0, 0.0], -9.0);
    assert_eq!(none, vec![-9.0, -9.0]);
}
