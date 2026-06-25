//! Unit tests for [`super`] (cell calling): Good-Turing, OrdMag, EmptyDrops.

use super::*;
use std::sync::Arc;

fn bc(s: &str) -> CellBarcode {
    CellBarcode::Barcode(Arc::from(s))
}

/// Build a CellCounts directly from dense per-cell vectors.
fn counts_from(dense: Vec<(CellBarcode, Vec<(u32, f32)>)>, n_genes: usize) -> CellCounts {
    let cells = dense
        .into_iter()
        .map(|(barcode, entries)| {
            let total = entries.iter().map(|&(_, v)| v as f64).sum();
            CellColumn {
                barcode,
                total,
                entries,
            }
        })
        .collect();
    CellCounts { n_genes, cells }
}

#[test]
fn good_turing_is_a_probability_vector() {
    let ambient = vec![0u64, 1, 2, 5, 10, 0, 3, 100, 0, 7];
    let p = simple_good_turing(&ambient);
    assert_eq!(p.len(), ambient.len());
    assert!(p.iter().all(|&x| x > 0.0 && x.is_finite()));
    let s: f64 = p.iter().sum();
    assert!((s - 1.0).abs() < 1e-9, "sum was {s}");
}

#[test]
fn ordmag_keeps_high_count_cells() {
    // 50 high-count cells (~10000) and 200 ambient (~100).
    let n_genes = 20;
    let mut dense = vec![];
    for i in 0..50 {
        dense.push((bc(&format!("cell{i}")), vec![(0u32, 10000.0f32)]));
    }
    for i in 0..200 {
        dense.push((bc(&format!("amb{i}")), vec![(1u32, 100.0f32)]));
    }
    let counts = counts_from(dense, n_genes);
    let p = CellCallParams {
        filter: CellFilter::OrdMag,
        expected_cells: 50,
        ..Default::default()
    };
    let cells = call_cells(&counts, &p);
    assert_eq!(cells.len(), 50);
    assert!(cells.contains(&bc("cell0")));
    assert!(!cells.contains(&bc("amb0")));
}

/// Sample a sparse column: `total` draws from `profile` weights.
fn sample_column(profile: &[f64], total: usize, rng: &mut StdRng) -> Vec<(u32, f32)> {
    let dist = WeightedIndex::new(profile.iter().copied()).unwrap();
    let mut acc: FxHashMap<u32, f32> = FxHashMap::default();
    for _ in 0..total {
        let g = dist.sample(rng) as u32;
        *acc.entry(g).or_default() += 1.0;
    }
    acc.into_iter().collect()
}

#[test]
fn empty_drops_rescues_distinct_low_count_cells() {
    // Ambient droplets are genuine multinomial draws from a non-uniform
    // background over genes 0..9. Real cells concentrate on marker genes
    // (10, 11) absent from ambient, at totals below the OrdMag knee, so only
    // EmptyDrops can recover them. Big cells (on gene 10) set the knee high.
    let n_genes = 12;
    // background weights on genes 0..9; markers 10,11 absent from ambient.
    let profile: Vec<f64> = vec![10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0.];
    let mut rng = StdRng::seed_from_u64(7);
    let mut dense = vec![];

    // 50 large cells far above the knee (marker gene 10).
    for i in 0..50 {
        dense.push((bc(&format!("big{i}")), vec![(10u32, 20000.0f32)]));
    }
    // 60 planted real cells: ~1000 counts on marker gene 11 (below the knee).
    for i in 0..60 {
        dense.push((bc(&format!("real{i}")), vec![(11u32, 1000.0f32)]));
    }
    // 2000 ambient droplets: multinomial draws, totals ~400..800.
    for i in 0..2000 {
        let total = 400 + (i % 400);
        dense.push((
            bc(&format!("amb{i}")),
            sample_column(&profile, total, &mut rng),
        ));
    }
    let counts = counts_from(dense, n_genes);

    let p = CellCallParams {
        filter: CellFilter::EmptyDrops,
        expected_cells: 50,
        ed_min_umis: 500,
        ed_ambient_lo: 200,
        ed_ambient_hi: 2000,
        ed_n_sims: 10000,
        ed_fdr: 0.01,
        ..Default::default()
    };
    let cells = call_cells(&counts, &p);

    // All big cells kept by OrdMag.
    assert!(cells.contains(&bc("big0")));
    // The marker-concentrated low-count cells are rescued by EmptyDrops.
    let rescued = (0..60)
        .filter(|i| cells.contains(&bc(&format!("real{i}"))))
        .count();
    assert!(rescued >= 55, "rescued only {rescued}/60 planted cells");
    // Genuine ambient droplets are (almost) never called (BH @ 1%).
    let amb_called = (0..2000)
        .filter(|i| cells.contains(&bc(&format!("amb{i}"))))
        .count();
    assert!(
        amb_called <= 40,
        "too many ambient droplets called: {amb_called}"
    );
}
