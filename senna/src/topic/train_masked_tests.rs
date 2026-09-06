//! Tests for the module map a level's decoder is built from, and for the
//! expansion of a module-level dictionary back to genes.

use super::{expand_log_dict_with_shares, module_map_for};
use crate::embed_common::Mat;
use candle_util::candle_core::Device;
use data_beans_alg::feature_coarsening::FeatureCoarsening;

fn coarsening() -> FeatureCoarsening {
    FeatureCoarsening {
        fine_to_coarse: vec![0, 0, 1, 1, 1, 2],
        coarse_to_fine: vec![vec![0, 1], vec![2, 3, 4], vec![5]],
        num_coarse: 3,
    }
}

#[test]
fn shares_are_mean_rates_over_the_module_and_masses_are_module_totals() {
    let mean = [2.0f32, 1.0, 1.0, 1.0, 2.0, 4.0];
    let (map, mass) = module_map_for(Some(&coarsening()), &mean, &Device::Cpu).unwrap();
    assert_eq!(mass, vec![3.0, 4.0, 4.0]);
    let want = [2.0f32 / 3.0, 1.0 / 3.0, 0.25, 0.25, 0.5, 1.0];
    for (ls, w) in map.host_log_share().iter().zip(want) {
        assert!((ls.exp() - w).abs() < 1e-6, "share {} vs {w}", ls.exp());
    }
    assert_eq!(map.host_fine_to_coarse(), &[0, 0, 1, 1, 1, 2]);
}

#[test]
fn a_module_without_mass_splits_evenly_and_no_coarsening_is_the_identity() {
    let mean = [2.0f32, 1.0, 0.0, 0.0, 0.0, 4.0];
    let (map, mass) = module_map_for(Some(&coarsening()), &mean, &Device::Cpu).unwrap();
    assert_eq!(mass[1], 0.0);
    for g in 2..5 {
        assert!((map.host_log_share()[g].exp() - 1.0 / 3.0).abs() < 1e-6);
    }
    let (id, mass) = module_map_for(None, &mean, &Device::Cpu).unwrap();
    assert!(id.is_identity());
    assert_eq!(mass, mean.to_vec());
}

#[test]
fn expanded_dictionary_keeps_column_mass_and_splits_by_share() {
    let mean = [2.0f32, 1.0, 1.0, 1.0, 2.0, 4.0];
    let (map, _) = module_map_for(Some(&coarsening()), &mean, &Device::Cpu).unwrap();
    // Two topics over three modules, each column a log-simplex.
    let p = [[0.5f32, 0.3, 0.2], [0.1, 0.1, 0.8]];
    let log_mk = Mat::from_fn(3, 2, |m, k| p[k][m].ln());
    let out = expand_log_dict_with_shares(&log_mk, map.host_fine_to_coarse(), map.host_log_share());
    assert_eq!((out.nrows(), out.ncols()), (6, 2));
    for k in 0..2 {
        let col: f32 = (0..6).map(|g| out[(g, k)].exp()).sum();
        assert!((col - 1.0).abs() < 1e-5, "column {k} sums to {col}");
        // Gene 0 takes 2/3 of module 0, gene 5 all of module 2.
        assert!((out[(0, k)].exp() - p[k][0] * 2.0 / 3.0).abs() < 1e-6);
        assert!((out[(5, k)].exp() - p[k][2]).abs() < 1e-6);
    }
}
