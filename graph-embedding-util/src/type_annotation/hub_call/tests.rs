use super::*;
use matrix_util::dmatrix_io::DMatrix;

const H: usize = 8;

/// A cell cloud with a definite centre: two lobes, so the hub is a real place and not a cell.
fn cells(n: usize) -> DMatrix<f32> {
    let mut m = DMatrix::zeros(n, H);
    for i in 0..n {
        let lobe = if i % 2 == 0 { 1.0 } else { -1.0 };
        for j in 0..H {
            // A lobe direction plus a small deterministic wobble.
            m[(i, j)] = lobe * (if j == 0 { 1.0 } else { 0.1 }) + 0.01 * ((i * H + j) as f32).sin();
        }
    }
    m
}

/// The hub of a cell cloud: the coordinate an unlearned gene's flat softmax lands on.
fn hub_of(c: &DMatrix<f32>) -> Vec<f32> {
    (0..H)
        .map(|j| c.column(j).iter().sum::<f32>() / c.nrows() as f32)
        .collect()
}

#[test]
fn a_gene_parked_at_the_hub_is_zeroed_and_a_real_one_is_not() {
    let c = cells(200);
    let hub = hub_of(&c);

    // 30 genes sitting AT the hub — the co-embedding's signature for a gene it never learned —
    // and 70 genes with real, distinct positions.
    let g = 100;
    let mut beta = vec![0f32; g * H];
    for i in 0..g {
        for j in 0..H {
            beta[i * H + j] = if i < 30 {
                // essentially the hub, with the tiny wobble a real projected point mass has
                hub[j] + 0.0005 * ((i * H + j) as f32).cos()
            } else {
                // a real gene: somewhere out on the manifold
                (if j == (i % H) { 1.0 } else { 0.0 }) + 0.05 * ((i + j) as f32).sin()
            };
        }
    }

    let n_zeroed = zero_hub_parked(&mut beta, &c, g, H);
    assert_eq!(n_zeroed, 30, "should zero exactly the hub-parked rows");

    // The hub-parked rows are now EXACTLY zero, which is what `live_row` tests for — so every
    // consumer downstream reads them as missing data without knowing this module exists.
    for i in 0..30 {
        assert!(
            beta[i * H..(i + 1) * H].iter().all(|&x| x == 0.0),
            "row {i} sat at the hub and should have been zeroed"
        );
    }
    // …and the real genes are untouched.
    for i in 30..g {
        assert!(
            beta[i * H..(i + 1) * H].iter().any(|&x| x != 0.0),
            "row {i} is a real gene and must not be zeroed"
        );
    }
}

#[test]
fn an_embedding_with_nothing_at_the_hub_is_left_alone() {
    // The no-op case: a healthy embedding where every gene was learned. Zeroing anything here would
    // silently delete real markers, so this is the regression guard on the false-positive side.
    let c = cells(200);
    let g = 60;
    let mut beta = vec![0f32; g * H];
    for i in 0..g {
        for j in 0..H {
            beta[i * H + j] = if j == (i % H) { 1.0 } else { 0.02 * (i as f32) };
        }
    }
    let before = beta.clone();
    let n_zeroed = zero_hub_parked(&mut beta, &c, g, H);
    assert_eq!(
        n_zeroed, 0,
        "no gene sits at the hub; nothing should be zeroed"
    );
    assert_eq!(beta, before);
}

#[test]
fn a_cloud_with_no_radius_is_declined_rather_than_dividing_by_zero() {
    // Every cell at the same point: there is no cloud, so "close to its centre" means nothing and
    // there is no scale to judge it against. Decline, rather than zero the entire gene space.
    let same = DMatrix::<f32>::from_element(50, H, 0.3);
    let mut beta = vec![1f32; 3 * H];
    assert_eq!(zero_hub_parked(&mut beta, &same, 3, H), 0);
    assert!(beta.iter().all(|&x| x == 1.0), "must not touch the rows");

    // Fewer than two cells: same story.
    let one = DMatrix::<f32>::zeros(1, H);
    let mut beta = vec![1f32; 3 * H];
    assert_eq!(zero_hub_parked(&mut beta, &one, 3, H), 0);
    assert!(beta.iter().all(|&x| x == 1.0), "must not touch the rows");
}
