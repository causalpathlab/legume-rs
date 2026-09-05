//! Tests for the gene→module map: the module embedding is the within-module
//! mean of ρ, ρ receives that gradient, the identity map is a no-op, and the
//! shares and lookups behave.

use super::ModuleMap;
use candle_core::{Device, Tensor, Var};
use nalgebra::DMatrix;

const D: usize = 6;
const H: usize = 3;

fn dev() -> Device {
    Device::Cpu
}

/// Modules {0,1}, {2,3,4}, {5}; shares from a mean vector [2,1 | 1,1,2 | 4].
fn map() -> ModuleMap {
    let f2c = [0usize, 0, 1, 1, 1, 2];
    let share = [2.0f32 / 3.0, 1.0 / 3.0, 0.25, 0.25, 0.5, 1.0];
    ModuleMap::new(&f2c, &share, &dev()).unwrap()
}

fn rho() -> Tensor {
    Tensor::from_vec(
        (0..D * H).map(|i| i as f32 * 0.5 - 4.0).collect(),
        (D, H),
        &dev(),
    )
    .unwrap()
}

#[test]
fn module_embedding_is_the_within_module_mean_of_rho() {
    let m = map();
    assert_eq!((m.n_fine(), m.n_coarse()), (6, 3));
    let bar = m.coarsen_mean_dh(&rho()).unwrap().to_vec2::<f32>().unwrap();
    let r = rho().to_vec2::<f32>().unwrap();
    for h in 0..H {
        assert!((bar[0][h] - (r[0][h] + r[1][h]) / 2.0).abs() < 1e-6);
        assert!((bar[1][h] - (r[2][h] + r[3][h] + r[4][h]) / 3.0).abs() < 1e-6);
        assert!((bar[2][h] - r[5][h]).abs() < 1e-6);
    }
}

#[test]
fn rho_receives_the_gradient_through_the_module_mean() {
    let m = map();
    let r = Var::from_tensor(&rho()).unwrap();
    let c = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        (3, H),
        &dev(),
    )
    .unwrap();
    let grads = (m.coarsen_mean_dh(r.as_tensor()).unwrap() * &c)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    let g = grads.get(&r).unwrap().to_vec2::<f32>().unwrap();
    let want = [
        [0.5, 1.0, 1.5],
        [0.5, 1.0, 1.5],
        [10.0 / 3.0, 20.0 / 3.0, 10.0],
        [10.0 / 3.0, 20.0 / 3.0, 10.0],
        [10.0 / 3.0, 20.0 / 3.0, 10.0],
        [100.0, 200.0, 300.0],
    ];
    for (gr, wr) in g.iter().zip(&want) {
        for (a, b) in gr.iter().zip(wr) {
            assert!((a - b).abs() < 1e-5, "gradient {g:?}");
        }
    }
}

#[test]
fn identity_map_is_a_no_op() {
    let m = ModuleMap::identity(D, &dev()).unwrap();
    assert!(m.is_identity());
    assert_eq!(m.n_coarse(), D);
    let r = rho();
    assert_eq!(
        m.coarsen_mean_dh(&r).unwrap().to_vec2::<f32>().unwrap(),
        r.to_vec2::<f32>().unwrap()
    );
    let ids = Tensor::from_vec(vec![5u32, 0, 3, 3], (2, 2), &dev()).unwrap();
    assert_eq!(
        m.modules_of(&ids).unwrap().to_vec2::<u32>().unwrap(),
        ids.to_vec2::<u32>().unwrap()
    );
    assert!(m
        .log_share_at(&ids)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap()
        .iter()
        .flatten()
        .all(|&x| x == 0.0));
}

#[test]
fn lookups_follow_the_map_and_shares_expand_by_module() {
    let m = map();
    assert!(!m.is_identity());
    let ids = Tensor::from_vec(vec![5u32, 0, 3, 4], (2, 2), &dev()).unwrap();
    assert_eq!(
        m.modules_of(&ids).unwrap().to_vec2::<u32>().unwrap(),
        vec![vec![2, 0], vec![1, 1]]
    );
    let ls = m.log_share_at(&ids).unwrap().to_vec2::<f32>().unwrap();
    let want = [
        [1.0f32.ln(), (2.0f32 / 3.0).ln()],
        [0.25f32.ln(), 0.5f32.ln()],
    ];
    for (a, b) in ls.iter().flatten().zip(want.iter().flatten()) {
        assert!((a - b).abs() < 1e-6);
    }
    let rows = DMatrix::from_row_slice(
        2,
        D,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    );
    let agg = m.aggregate_columns_host(&rows);
    assert_eq!(
        agg.row(0).iter().copied().collect::<Vec<_>>(),
        vec![3.0, 12.0, 6.0]
    );
    assert_eq!(
        agg.row(1).iter().copied().collect::<Vec<_>>(),
        vec![1.0, 1.0, 1.0]
    );
    // The device aggregation agrees with the host one, and the identity map
    // returns its input.
    let x = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ],
        (2, D),
        &dev(),
    )
    .unwrap();
    assert_eq!(
        m.aggregate_columns(&x).unwrap().to_vec2::<f32>().unwrap(),
        vec![vec![3.0, 12.0, 6.0], vec![1.0, 1.0, 1.0]]
    );
    let id = ModuleMap::identity(D, &dev()).unwrap();
    assert_eq!(
        id.aggregate_columns(&x).unwrap().to_vec2::<f32>().unwrap(),
        x.to_vec2::<f32>().unwrap()
    );
    let ls = m.log_share_1d().to_vec2::<f32>().unwrap();
    assert_eq!(ls.len(), 1);
    assert!((ls[0][0] - (2.0f32 / 3.0).ln()).abs() < 1e-6);
}
