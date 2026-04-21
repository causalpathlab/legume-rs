//! Integration tests for `data_beans_alg::refine_multilevel::refine_assignments`.
//!
//! Kernel- and helper-level unit tests live inline in `src/refine_multilevel.rs`;
//! these exercise the public end-to-end entry point against a synthetic
//! layout + per-batch HNSW fixture.

use data_beans_alg::collapse_data::SuperCellLayout;
use data_beans_alg::refine_multilevel::{
    compact_labels, refine_assignments, RefineInputs, RefineParams,
};
use matrix_util::knn_match::ColumnDict;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rustc_hash::FxHashMap as HashMap;

/// Handle returned by `make_toy_layout_and_profiles` bundling the layout
/// and the auxiliary per-super-cell data needed to call `refine_assignments`.
struct ToyFixture {
    layout: SuperCellLayout,
    gene_sums: Vec<Vec<(usize, f32)>>,
    super_cell_to_cells: Vec<Vec<usize>>,
    true_cluster: Vec<usize>,
    batch_knn: Vec<ColumnDict<usize>>,
}

/// Build a synthetic refinement fixture.
///
/// - `num_batches` batches × `num_clusters` true clusters → one super-cell
///   per (batch, cluster).
/// - `cells_per_cluster_per_batch` cells per super-cell, each carrying a
///   jittered copy of the super-cell's indicator centroid. Per-batch HNSWs
///   are built over these cell features, so a centroid-query in a foreign
///   batch returns the matching-cluster super-cell first.
/// - Initial `super_cell_to_group` labels are a noisy hash partition (70%
///   aligned with truth) so refinement has something to correct.
fn make_toy_layout_and_profiles(
    num_batches: usize,
    cells_per_cluster_per_batch: usize,
    num_clusters: usize,
    num_features: usize,
    noise_seed: u64,
) -> ToyFixture {
    let num_sc = num_batches * num_clusters;
    let proj_dim = num_clusters.max(2);
    let mut centroids = DMatrix::<f32>::zeros(proj_dim, num_sc);
    let mut sc_to_batch = Vec::with_capacity(num_sc);
    let mut sc_to_group = Vec::with_capacity(num_sc);
    let mut bg_to_sc: HashMap<(usize, usize), usize> = HashMap::default();
    let mut true_cluster = Vec::with_capacity(num_sc);
    let mut cell_counts = Vec::with_capacity(num_sc);

    let mut gene_sums: Vec<Vec<(usize, f32)>> = Vec::with_capacity(num_sc);
    let mut super_cell_to_cells: Vec<Vec<usize>> = Vec::with_capacity(num_sc);
    let per_block = num_features / num_clusters;

    let mut rng = SmallRng::seed_from_u64(noise_seed);
    let mut next_cell = 0usize;
    let mut per_batch_cells: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); num_batches];
    let mut cell_to_sc_vec: Vec<usize> = Vec::new();

    for (batch, batch_cells) in per_batch_cells.iter_mut().enumerate() {
        for cluster in 0..num_clusters {
            let sc = batch * num_clusters + cluster;
            centroids[(cluster.min(proj_dim - 1), sc)] = 1.0;
            sc_to_batch.push(batch);
            let hash_group = if rng.random_range(0.0..1.0_f32) < 0.7 {
                cluster
            } else {
                (cluster + 1) % num_clusters
            };
            sc_to_group.push(hash_group);
            bg_to_sc.insert((batch, hash_group), sc);
            true_cluster.push(cluster);
            cell_counts.push(cells_per_cluster_per_batch as f32);

            let mut row: Vec<(usize, f32)> = Vec::new();
            let start = cluster * per_block;
            let end = ((cluster + 1) * per_block).min(num_features);
            for g in start..end {
                row.push((g, 10.0 + rng.random_range(0.0..1.0_f32)));
            }
            for _ in 0..3 {
                let g: usize = rng.random_range(0..num_features);
                row.push((g, rng.random_range(0.0..0.5_f32)));
            }
            row.sort_unstable_by_key(|&(g, _)| g);
            row.dedup_by_key(|&mut (g, _)| g);
            gene_sums.push(row);

            let mut cells_here = Vec::with_capacity(cells_per_cluster_per_batch);
            for _ in 0..cells_per_cluster_per_batch {
                let c = next_cell;
                next_cell += 1;
                cells_here.push(c);
                cell_to_sc_vec.push(sc);
                let mut feat = vec![0.0f32; proj_dim];
                feat[cluster.min(proj_dim - 1)] = 1.0 + rng.random_range(-0.02..0.02_f32);
                batch_cells.push((c, feat));
            }
            super_cell_to_cells.push(cells_here);
        }
    }

    let batch_knn_lookup: Vec<ColumnDict<usize>> = per_batch_cells
        .into_iter()
        .map(|cells_in_b| {
            let n = cells_in_b.len();
            let d = proj_dim;
            let mut mat = DMatrix::<f32>::zeros(d, n);
            let mut names = Vec::with_capacity(n);
            for (col, (cell_idx, feat)) in cells_in_b.into_iter().enumerate() {
                for (row, v) in feat.into_iter().enumerate() {
                    mat[(row, col)] = v;
                }
                names.push(cell_idx);
            }
            ColumnDict::<usize>::from_dmatrix(mat, names)
        })
        .collect();

    let layout = SuperCellLayout {
        centroids,
        cell_counts,
        super_cell_to_batch: sc_to_batch,
        super_cell_to_group: sc_to_group,
        bg_to_sc,
        cell_to_sc: cell_to_sc_vec,
    };
    ToyFixture {
        layout,
        gene_sums,
        super_cell_to_cells,
        true_cluster,
        batch_knn: batch_knn_lookup,
    }
}

fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    assert_eq!(n, b.len());
    let mut contingency: HashMap<(usize, usize), u64> = HashMap::default();
    let mut a_counts: HashMap<usize, u64> = HashMap::default();
    let mut b_counts: HashMap<usize, u64> = HashMap::default();
    for i in 0..n {
        *contingency.entry((a[i], b[i])).or_insert(0) += 1;
        *a_counts.entry(a[i]).or_insert(0) += 1;
        *b_counts.entry(b[i]).or_insert(0) += 1;
    }
    let choose2 = |x: u64| -> f64 { (x as f64) * ((x as f64) - 1.0) / 2.0 };
    let sum_cont_c2: f64 = contingency.values().map(|&c| choose2(c)).sum();
    let sum_a_c2: f64 = a_counts.values().map(|&c| choose2(c)).sum();
    let sum_b_c2: f64 = b_counts.values().map(|&c| choose2(c)).sum();
    let total_c2 = choose2(n as u64);
    let expected = sum_a_c2 * sum_b_c2 / total_c2;
    let max = 0.5 * (sum_a_c2 + sum_b_c2);
    if (max - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_cont_c2 - expected) / (max - expected)
}

#[test]
fn refine_assignments_is_deterministic() {
    let ToyFixture {
        layout,
        gene_sums,
        super_cell_to_cells: sc_to_cells,
        batch_knn,
        ..
    } = make_toy_layout_and_profiles(2, 5, 4, 16, 42);
    let num_genes = 16;

    let initial_finest = layout.super_cell_to_group.clone();
    let initial_coarse: Vec<usize> = initial_finest.iter().map(|&g| g / 2).collect();
    let (initial_coarse_compact, _k) = compact_labels(&initial_coarse);
    let initial_per_level = vec![initial_finest, initial_coarse_compact];

    let params = RefineParams {
        num_gibbs: 5,
        num_greedy: 3,
        seed: 42,
        ..Default::default()
    };
    let inputs = RefineInputs {
        layout: &layout,
        gene_sums: &gene_sums,
        num_genes,
        super_cell_to_cells: &sc_to_cells,
        batch_knn_lookup: &batch_knn,
        k_per_batch: 4,
        initial_sc_to_group_per_level: &initial_per_level,
    };
    let a = refine_assignments(&inputs, &params).unwrap();
    let b = refine_assignments(&inputs, &params).unwrap();
    assert_eq!(a.sc_to_group, b.sc_to_group);
    assert_eq!(a.num_groups_per_level, b.num_groups_per_level);
}

#[test]
fn refine_assignments_preserves_hierarchy() {
    let ToyFixture {
        layout,
        gene_sums,
        super_cell_to_cells: sc_to_cells,
        batch_knn,
        ..
    } = make_toy_layout_and_profiles(2, 5, 4, 16, 7);
    let num_genes = 16;
    let initial_finest = layout.super_cell_to_group.clone();
    let initial_coarse: Vec<usize> = initial_finest.iter().map(|&g| g / 2).collect();
    let (initial_coarse_compact, _k) = compact_labels(&initial_coarse);
    let initial_per_level = vec![initial_finest, initial_coarse_compact];

    let inputs = RefineInputs {
        layout: &layout,
        gene_sums: &gene_sums,
        num_genes,
        super_cell_to_cells: &sc_to_cells,
        batch_knn_lookup: &batch_knn,
        k_per_batch: 4,
        initial_sc_to_group_per_level: &initial_per_level,
    };
    let refined = refine_assignments(&inputs, &RefineParams::default()).unwrap();

    // Every fine group should have a single parent group at the coarser level.
    let mut parent_of_fine: HashMap<usize, usize> = HashMap::default();
    for sc in 0..refined.sc_to_group[0].len() {
        let fine = refined.sc_to_group[0][sc];
        let coarse = refined.sc_to_group[1][sc];
        if let Some(&p) = parent_of_fine.get(&fine) {
            assert_eq!(
                p, coarse,
                "fine group {} has inconsistent parents: {} and {}",
                fine, p, coarse
            );
        } else {
            parent_of_fine.insert(fine, coarse);
        }
    }
}

#[test]
fn refine_assignments_does_not_hurt_ari_on_planted() {
    let ToyFixture {
        layout,
        gene_sums,
        super_cell_to_cells: sc_to_cells,
        true_cluster: truth,
        batch_knn,
    } = make_toy_layout_and_profiles(2, 6, 4, 24, 11);
    let num_genes = 24;
    let initial_finest = layout.super_cell_to_group.clone();
    let initial_per_level = vec![initial_finest.clone()];

    let params = RefineParams {
        num_gibbs: 15,
        num_greedy: 10,
        seed: 11,
        ..Default::default()
    };
    let inputs = RefineInputs {
        layout: &layout,
        gene_sums: &gene_sums,
        num_genes,
        super_cell_to_cells: &sc_to_cells,
        batch_knn_lookup: &batch_knn,
        k_per_batch: 4,
        initial_sc_to_group_per_level: &initial_per_level,
    };
    let refined = refine_assignments(&inputs, &params).unwrap();

    let ari_before = adjusted_rand_index(&initial_finest, &truth);
    let ari_after = adjusted_rand_index(&refined.sc_to_group[0], &truth);
    assert!(
        ari_after >= ari_before,
        "refinement should not hurt ARI: before={:.3}, after={:.3}",
        ari_before,
        ari_after
    );
}
