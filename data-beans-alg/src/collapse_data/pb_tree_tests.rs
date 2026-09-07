use super::*;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson};

//////////////
// Fixtures //
//////////////

/// Columns `cells` of `csc`, in that order.
fn select_columns(csc: &CscMatrix<f32>, cells: &[usize]) -> CscMatrix<f32> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for (j, &c) in cells.iter().enumerate() {
        let col = csc.col(c);
        for (&g, &v) in col.row_indices().iter().zip(col.values()) {
            rows.push(g);
            cols.push(j);
            vals.push(v);
        }
    }
    let coo =
        nalgebra_sparse::CooMatrix::try_from_triplets(csc.nrows(), cells.len(), rows, cols, vals)
            .expect("valid triplets");
    CscMatrix::from(&coo)
}

fn csc_from_dense(x_gc: &[Vec<f32>]) -> CscMatrix<f32> {
    let ngenes = x_gc.len();
    let ncells = x_gc[0].len();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for (g, row) in x_gc.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if v != 0.0 {
                rows.push(g);
                cols.push(c);
                vals.push(v);
            }
        }
    }
    let coo = CooMatrix::try_from_triplets(ngenes, ncells, rows, cols, vals).unwrap();
    CscMatrix::from(&coo)
}

/// Two-state Poisson block: `n_a` cells of state A and `n_b` of state B over
/// `ngenes` genes. State B multiplies the first `n_up` program genes by
/// `fold` and the next `n_down` by `1/fold`; everything else is shared.
struct Planted {
    csc: CscMatrix<f32>,
    state: Vec<usize>,
    n_up: usize,
    n_down: usize,
}

#[allow(clippy::too_many_arguments)]
fn planted_block(
    n_a: usize,
    n_b: usize,
    ngenes: usize,
    n_up: usize,
    n_down: usize,
    fold: f64,
    depth: f64,
    seed: u64,
) -> Planted {
    let mut rng = SmallRng::seed_from_u64(seed);
    let base: Vec<f64> = (0..ngenes).map(|g| 1.0 + (g % 7) as f64 / 3.0).collect();
    let base_sum: f64 = base.iter().sum();
    let ncells = n_a + n_b;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut state = Vec::with_capacity(ncells);
    for c in 0..ncells {
        let is_b = c >= n_a;
        state.push(usize::from(is_b));
        for (g, &b) in base.iter().enumerate() {
            let mut rate = b / base_sum * depth;
            if is_b && g < n_up {
                rate *= fold;
            } else if is_b && g < n_up + n_down {
                rate /= fold;
            }
            let x: f64 = Poisson::new(rate).unwrap().sample(&mut rng);
            if x > 0.0 {
                rows.push(g);
                cols.push(c);
                vals.push(x as f32);
            }
        }
    }
    let coo = CooMatrix::try_from_triplets(ngenes, ncells, rows, cols, vals).unwrap();
    Planted {
        csc: CscMatrix::from(&coo),
        state,
        n_up,
        n_down,
    }
}

/// Fraction of cells whose side agrees with `state`, modulo a global flip.
fn agreement(side: &[usize], state: &[usize]) -> f32 {
    let n = side.len() as f32;
    let same = side.iter().zip(state).filter(|(s, t)| s == t).count() as f32 / n;
    same.max(1.0 - same)
}

fn single_batch(n: usize) -> Vec<usize> {
    vec![0; n]
}

//////////////////////////
// Residual closed form //
//////////////////////////

#[test]
fn pearson_residual_matches_closed_form() {
    // genes x cells; gene 3 is silent everywhere.
    let x = vec![
        vec![4.0, 2.0, 0.0],
        vec![0.0, 2.0, 4.0],
        vec![2.0, 0.0, 2.0],
        vec![0.0, 0.0, 0.0],
    ];
    let block = NodeBlock::new(csc_from_dense(&x), single_batch(3), 1);
    let local: Vec<usize> = (0..3).collect();
    let prof = NodeProfiles::new(&block, &local, 1);
    // depths 6, 4, 6; pooled rates 6/16, 6/16, 4/16, 0.
    assert!((prof.pooled[0] - 6.0 / 16.0).abs() < 1e-6);
    assert!((prof.pooled[3]).abs() < 1e-12);
    let genes: Vec<usize> = (0..4).collect();
    let r = residual_dense(&block, &local, &genes, &prof, f32::INFINITY);
    // cell 0: mu = 6 * rate = [2.25, 2.25, 1.5, 0]
    assert!((r[(0, 0)] - 1.75 / 1.5).abs() < 1e-5);
    assert!((r[(0, 1)] + 1.5).abs() < 1e-5);
    assert!((r[(0, 2)] - 0.5 / 1.5f32.sqrt()).abs() < 1e-5);
    assert_eq!(r[(0, 3)], 0.0, "a silent gene has zero residual");
    // clipping bounds every entry
    let r_clip = residual_dense(&block, &local, &genes, &prof, 1.0);
    assert!((r_clip[(0, 1)] + 1.0).abs() < 1e-6);
    assert!(r_clip.iter().all(|v| v.abs() <= 1.0 + 1e-6));
}

#[test]
fn residual_variance_matches_dense() {
    let p = planted_block(20, 20, 50, 5, 5, 3.0, 300.0, 1);
    let n = p.csc.ncols();
    let block = NodeBlock::new(p.csc, single_batch(n), 1);
    let local: Vec<usize> = (0..n).collect();
    let prof = NodeProfiles::new(&block, &local, 1);
    let genes: Vec<usize> = (0..50).collect();
    let var = residual_variance(&block, &local, &prof, f32::INFINITY);
    let r = residual_dense(&block, &local, &genes, &prof, f32::INFINITY);
    for (g, &v) in var.iter().enumerate() {
        let dense: f32 = r.column(g).iter().map(|x| x * x).sum::<f32>() / n as f32;
        assert!((v as f32 - dense).abs() < 1e-3 * dense.max(1.0), "gene {g}");
    }
}

//////////////////
// Split scores //
//////////////////

#[test]
fn two_group_llr_matches_closed_form() {
    // left: (4, 0), right: (0, 4) -> LLR = 8 ln 2 for a perfect split.
    let llr = two_group_llr(&[4.0, 0.0], &[0.0, 4.0]);
    assert!((llr - 8.0 * 2f64.ln()).abs() < 1e-9);
    // identical composition -> zero.
    assert!(two_group_llr(&[3.0, 3.0], &[6.0, 6.0]).abs() < 1e-9);
    // empty side -> zero, no NaN.
    assert_eq!(two_group_llr(&[3.0, 1.0], &[0.0, 0.0]), 0.0);
}

#[test]
fn ve_ratio_is_one_under_random_partition() {
    let n = 400;
    let p = 60;
    let r = DMatrix::<f32>::rnorm_seeded(n, p, 7);
    let mut rng = SmallRng::seed_from_u64(3);
    let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..4usize)).collect();
    let ratio = ve_ratio(&r, &labels, 30);
    assert!(
        (0.6..1.4).contains(&ratio),
        "random partition ratio {ratio}"
    );
    let single = vec![0usize; n];
    assert_eq!(ve_ratio(&r, &single, 30), 1.0);
}

#[test]
fn mp_edge_scales_with_noise_and_shape() {
    let e = mp_edge(100, 400, 1.0);
    assert!((e - (10.0 + 20.0)).abs() < 1e-6);
    assert!((mp_edge(100, 400, 4.0) - 2.0 * e).abs() < 1e-5);
}

////////////////////////
// Planted recovery   //
////////////////////////

//////////////////////////
// Code rewrite         //
//////////////////////////

/// Two lineages with strong private blocks, each holding two states that
/// differ only on a shared moderate program.
fn lineage_fixture(seed: u64) -> (CscMatrix<f32>, Vec<usize>, Vec<usize>) {
    let ngenes = 200;
    let per = 60; // cells per (lineage, state)
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut lineage = Vec::new();
    let mut state = Vec::new();
    let mut c = 0usize;
    for l in 0..2usize {
        for s in 0..2usize {
            for _ in 0..per {
                for g in 0..ngenes {
                    let mut rate = 1.0 + (g % 5) as f64 / 2.0;
                    let block_l = if g < 60 {
                        0
                    } else if g < 120 {
                        1
                    } else {
                        2
                    };
                    if block_l == l {
                        rate *= 8.0;
                    } else if block_l < 2 {
                        rate *= 0.2;
                    }
                    if s == 1 && (120..135).contains(&g) {
                        rate *= 3.0;
                    } else if s == 1 && (135..150).contains(&g) {
                        rate /= 3.0;
                    }
                    let x: f64 = Poisson::new(rate).unwrap().sample(&mut rng);
                    if x > 0.0 {
                        rows.push(g);
                        cols.push(c);
                        vals.push(x as f32);
                    }
                }
                lineage.push(l);
                state.push(s);
                c += 1;
            }
        }
    }
    let n = c;
    let coo = CooMatrix::try_from_triplets(ngenes, n, rows, cols, vals).unwrap();
    let csc = CscMatrix::from(&coo);
    (csc, lineage, state)
}

//////////////////////////////
// Frontier (target-count)  //
//////////////////////////////

/// Roots and blocks for the lineage fixture: one root per lineage.
fn lineage_roots(seed: u64) -> (Vec<RootBlock>, Vec<usize>, Vec<usize>, usize) {
    let (csc, lineage, state) = lineage_fixture(seed);
    let n = csc.ncols();
    let roots: Vec<RootBlock> = (0..2usize)
        .map(|l| {
            let cells: Vec<usize> = (0..n).filter(|&i| lineage[i] == l).collect();
            let block = NodeBlock::new(select_columns(&csc, &cells), single_batch(cells.len()), 1);
            RootBlock {
                root: l,
                cells,
                block,
            }
        })
        .collect();
    (roots, lineage, state, n)
}

fn nested(coarse: &[usize], fine: &[usize]) -> bool {
    let mut parent: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    fine.iter()
        .zip(coarse)
        .all(|(&f, &c)| *parent.entry(f).or_insert(c) == c)
}

#[test]
fn frontier_reaches_the_targets_with_nested_levels() {
    let (roots, lineage, state, n) = lineage_roots(77);
    let params = PbTreeParams {
        min_cells_to_split: 4,
        ..PbTreeParams::default()
    };
    let out = bisect_to_target(&roots, n, &[4, 16, 32], &params);
    assert_eq!(out.labels_per_level.len(), 3);
    let counts: Vec<usize> = out
        .labels_per_level
        .iter()
        .map(|l| {
            l.iter()
                .filter(|&&x| x != usize::MAX)
                .collect::<std::collections::HashSet<_>>()
                .len()
        })
        .collect();
    assert_eq!(counts, vec![4, 16, 32], "leaf counts per level");
    assert!(nested(&out.labels_per_level[0], &out.labels_per_level[1]));
    assert!(nested(&out.labels_per_level[1], &out.labels_per_level[2]));
    // the first split inside each lineage is the state axis
    for l in 0..2usize {
        let idx: Vec<usize> = (0..n).filter(|&i| lineage[i] == l).collect();
        let side: Vec<usize> = idx.iter().map(|&i| out.labels_per_level[0][i]).collect();
        let truth: Vec<usize> = idx.iter().map(|&i| state[i]).collect();
        let (c, _) = crate::dc_poisson::compact_labels(&side);
        assert!(
            agreement(&c, &truth) >= 0.9,
            "lineage {l}: coarse level should split the state"
        );
    }
    assert_eq!(out.tree.roots.len(), 2);
    let total: usize = out
        .tree
        .roots
        .iter()
        .map(|nd| nd.splits.iter().filter(|s| s.applied).count())
        .sum();
    assert_eq!(total, 30, "32 leaves from 2 roots take 30 applied splits");
}

#[test]
fn frontier_spends_the_budget_on_evidence() {
    // root 0 carries a planted program, root 1 is pure noise; one split allowed.
    let p = planted_block(40, 40, 300, 12, 12, 3.0, 2000.0, 91);
    let q = planted_block(40, 40, 300, 0, 0, 1.0, 2000.0, 92);
    let n0 = p.csc.ncols();
    let n1 = q.csc.ncols();
    let roots = vec![
        RootBlock {
            root: 0,
            cells: (0..n0).collect(),
            block: NodeBlock::new(p.csc, single_batch(n0), 1),
        },
        RootBlock {
            root: 1,
            cells: (n0..n0 + n1).collect(),
            block: NodeBlock::new(q.csc, single_batch(n1), 1),
        },
    ];
    let out = bisect_to_target(&roots, n0 + n1, &[3], &PbTreeParams::default());
    let applied: Vec<&SplitRecord> = out
        .tree
        .roots
        .iter()
        .flat_map(|nd| nd.splits.iter())
        .filter(|s| s.applied)
        .collect();
    assert_eq!(applied.len(), 1);
    assert_eq!(applied[0].root, 0, "the split with evidence is taken first");
}

#[test]
fn packed_codes_reproduce_the_levels() {
    // three nested levels over 12 cells
    let coarse = vec![0usize, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    let mid = vec![0usize, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5];
    let fine = vec![0usize, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 8];
    let levels = vec![coarse.clone(), mid.clone(), fine.clone()];
    let (codes, dims) = pack_levels(&levels);
    assert_eq!(dims.len(), 3);
    assert!(
        dims[0] > dims[1] && dims[1] > dims[2],
        "finest-first widths {dims:?}"
    );
    for (lvl, &d) in [&fine, &mid, &coarse].iter().zip(&dims) {
        let mask = (1usize << d) - 1;
        let masked: Vec<usize> = codes.iter().map(|&c| c & mask).collect();
        let (a, ka) = crate::dc_poisson::compact_labels(&masked);
        let (b, kb) = crate::dc_poisson::compact_labels(lvl);
        assert_eq!(ka, kb);
        assert!(
            nested(&a, &b) && nested(&b, &a),
            "level with {d} bits must reproduce the partition"
        );
    }
}

#[test]
fn frontier_result_is_independent_of_thread_count() {
    let (roots, _, _, n) = lineage_roots(77);
    let params = PbTreeParams {
        min_cells_to_split: 4,
        ..PbTreeParams::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| bisect_to_target(&roots, n, &[4, 16, 32], &params).labels_per_level)
    };
    assert_eq!(run(1), run(4));
}

//////////////////////////////
// Single-root behaviours   //
//////////////////////////////

fn one_root(p: Planted) -> (Vec<RootBlock>, usize) {
    let n = p.csc.ncols();
    let roots = vec![RootBlock {
        root: 0,
        cells: (0..n).collect(),
        block: NodeBlock::new(p.csc, single_batch(n), 1),
    }];
    (roots, n)
}

#[test]
fn split_recovers_planted_program() {
    let p = planted_block(30, 30, 400, 12, 12, 3.0, 2000.0, 11);
    let state = p.state.clone();
    let (n_up, n_down) = (p.n_up, p.n_down);
    let (roots, n) = one_root(p);
    let out = bisect_to_target(&roots, n, &[2], &PbTreeParams::default());
    let split = out.tree.roots[0]
        .splits
        .iter()
        .find(|s| s.applied)
        .expect("one applied split");
    assert!(
        split.passes_edge,
        "s1 {} vs edge {}",
        split.s1, split.mp_edge
    );
    let (side, _) = crate::dc_poisson::compact_labels(&out.labels_per_level[0]);
    assert!(
        agreement(&side, &state) >= 0.9,
        "agreement {}",
        agreement(&side, &state)
    );
    assert!(split.llr_split > 0.0 && split.ve_ratio > 3.0);
    let up: std::collections::HashSet<usize> = (0..n_up).collect();
    let down: std::collections::HashSet<usize> = (n_up..n_up + n_down).collect();
    let a: Vec<usize> = split.pos.iter().map(|g| g.gene).collect();
    let b: Vec<usize> = split.neg.iter().map(|g| g.gene).collect();
    let hits = |v: &[usize], s: &std::collections::HashSet<usize>| {
        v.iter().filter(|g| s.contains(g)).count()
    };
    assert!(
        (hits(&a, &up) + hits(&b, &down)).max(hits(&a, &down) + hits(&b, &up)) >= 20,
        "contrast genes should be program genes"
    );
    assert!(split.pos.iter().all(|g| g.weight > 0.0 && g.weight <= 1.0));
}

#[test]
fn null_root_stays_below_edge() {
    let p = planted_block(30, 30, 400, 0, 0, 1.0, 2000.0, 5);
    let (roots, n) = one_root(p);
    let stop = PbTreeParams {
        below_edge: BelowEdge::Stop,
        ..PbTreeParams::default()
    };
    let out = bisect_to_target(&roots, n, &[2], &stop);
    let split = &out.tree.roots[0].splits[0];
    assert!(!split.passes_edge, "s1 {} edge {}", split.s1, split.mp_edge);
    assert!(
        !split.applied,
        "a below-edge split is not applied under Stop"
    );
    assert!(
        out.labels_per_level[0].iter().all(|&l| l == 0),
        "the root stays one leaf"
    );
    let out = bisect_to_target(&roots, n, &[2], &PbTreeParams::default());
    let split = &out.tree.roots[0].splits[0];
    assert!(
        !split.passes_edge && split.applied,
        "Keep applies and flags it"
    );
}

#[test]
fn landmark_path_agrees_with_exact() {
    let p = planted_block(150, 150, 300, 12, 12, 3.0, 2000.0, 21);
    let state = p.state.clone();
    let (roots, n) = one_root(p);
    let exact = PbTreeParams {
        max_cells_exact: 1000,
        ..PbTreeParams::default()
    };
    let landmarks = PbTreeParams {
        max_cells_exact: 50,
        num_landmarks: 64,
        ..PbTreeParams::default()
    };
    let a = bisect_to_target(&roots, n, &[2], &exact)
        .labels_per_level
        .remove(0);
    let b = bisect_to_target(&roots, n, &[2], &landmarks)
        .labels_per_level
        .remove(0);
    assert!(agreement(&a, &state) >= 0.9);
    assert!(agreement(&b, &state) >= 0.9);
    assert!(agreement(&a, &b) >= 0.9);
}
