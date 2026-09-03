use super::*;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::sparse_io_vector::SparseIoVec;
use rand::{rngs::StdRng, SeedableRng};

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn relation_weights_are_linspace_one_to_five_rounded_to_two_decimals_and_shrink_to_levels_present()
{
    let full = RelationTable::from_levels(&[1, 2, 3, 4, 5]);
    assert_eq!(full.len(), 5);
    assert_eq!(full.weights, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(full.rel(1), 0);
    assert_eq!(full.rel(5), 4);
    assert!(approx(f64::from(full.weight(3)), 3.0, 1e-6));

    let four = RelationTable::from_levels(&[1, 3, 4, 5]);
    assert_eq!(four.weights, vec![1.0, 2.33, 3.67, 5.0]);
    assert_eq!(four.rel(3), 1);
    assert!(approx(f64::from(four.weight(4)), 3.67, 1e-6));

    let two = RelationTable::from_levels(&[2, 5]);
    assert_eq!(two.weights, vec![1.0, 5.0]);
    assert_eq!(two.rel(5), 1);

    let one = RelationTable::from_levels(&[3]);
    assert_eq!(one.weights, vec![1.0]);
    assert_eq!(one.rel(3), 0);
}

#[test]
fn auto_wd_uses_the_small_graph_formula_below_fifty_million_edges_and_the_large_one_above() {
    assert!(approx(auto_wd(2_725_781), 0.013, 1e-12));
    assert!(approx(auto_wd(20_000_000), 0.001_772, 1e-12));
    assert!(approx(auto_wd(59_103_481), 0.000_4, 1e-12));
    assert!(approx(auto_wd(100_000_000), 0.000_236, 1e-12));
}

/// 3 cells × 4 genes; genes 1 and 3 are the HVGs. Gene 0 is a non-HVG gene
/// that dominates cell 0's library and carries the largest log-normalized
/// value in the matrix, so it must shape both the library sizes and the
/// histogram range without ever becoming an edge.
fn tiny_backend() -> SparseIoVec {
    // (row = gene, col = cell, value)
    let triplets: Vec<(u64, u64, f32)> = vec![
        (0, 0, 90.0),
        (1, 0, 5.0),
        (3, 0, 5.0), // cell 0: lib 100
        (0, 1, 10.0),
        (2, 1, 10.0),
        (3, 1, 80.0), // cell 1: lib 100
        (1, 2, 1.0),
        (2, 2, 1.0), // cell 2: lib 2
    ];
    let shape = (4usize, 3usize, triplets.len());
    let mut b = create_sparse_from_triplets(&triplets, shape, None, Some(&SparseIoBackend::Zarr))
        .expect("backend");
    b.register_row_names_vec(
        &(0..4)
            .map(|g| format!("g{g}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    b.register_column_names_vec(
        &(0..3)
            .map(|c| format!("c{c}").into_boxed_str())
            .collect::<Vec<_>>(),
    );
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(b), None).expect("push");
    v
}

#[test]
fn edge_list_from_a_tiny_zarr_backend_has_one_edge_per_nonzero_hvg_entry_with_log_normalised_levels(
) {
    let data = tiny_backend();
    let (edges, disc) = build_edge_list(&data, &[1, 3], 5).expect("edges");
    assert_eq!(edges.n_cells, 3);
    assert_eq!(edges.n_genes, 2);
    assert_eq!(edges.len(), 4);
    let mut got: Vec<(u32, u32)> = edges
        .cell
        .iter()
        .copied()
        .zip(edges.gene.iter().copied())
        .collect();
    got.sort_unstable();
    // gene index = position in the HVG list: g1 → 0, g3 → 1
    assert_eq!(got, vec![(0, 0), (0, 1), (1, 1), (2, 0)]);
    // Library sizes count the non-HVG genes: cell 0's lowest value is
    // ln1p(1e4·5/100), not ln1p(1e4·5/10); the range's top is cell 0's non-HVG
    // gene 0 (ln1p(1e4·90/100)), which is not an edge.
    assert!(approx(disc.hist_range.0, 501f64.ln(), 1e-6));
    assert!(approx(disc.hist_range.1, 9001f64.ln(), 1e-6));
    assert_eq!(disc.hist_counts.iter().sum::<u64>(), 8);
    let n = disc.n_levels();
    assert!(edges.level.iter().all(|&l| (1..=n as u8).contains(&l)));
    // Levels follow the value: cell 2's HVG entry (ln1p(5000)) sits above
    // cell 0's (ln1p(500)).
    let level_of = |c: u32, g: u32| {
        let i = (0..edges.len())
            .find(|&i| edges.cell[i] == c && edges.gene[i] == g)
            .unwrap();
        edges.level[i]
    };
    assert!(level_of(2, 0) > level_of(0, 0));
    let mut distinct = edges.level.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(edges.levels_present(), distinct);
}

#[test]
fn shuffle_range_permutes_only_the_requested_range_and_keeps_each_edge_intact() {
    let n = 12;
    let mut edges = EdgeList {
        n_cells: n,
        n_genes: 2 * n,
        cell: (0..n as u32).collect(),
        gene: (0..n as u32).map(|i| 2 * i).collect(),
        level: (0..n as u8).map(|i| i % 3 + 1).collect(),
    };
    let before: Vec<(u32, u32, u8)> = (0..n)
        .map(|i| (edges.cell[i], edges.gene[i], edges.level[i]))
        .collect();
    let mut rng = StdRng::seed_from_u64(7);
    edges.shuffle_range(0..8, &mut rng);
    let after: Vec<(u32, u32, u8)> = (0..n)
        .map(|i| (edges.cell[i], edges.gene[i], edges.level[i]))
        .collect();
    assert_eq!(&after[8..], &before[8..], "the tail is untouched");
    let mut a = after[..8].to_vec();
    let mut b = before[..8].to_vec();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(
        a, b,
        "the same edges, each still (cell, gene, level)-paired"
    );
    assert_ne!(&after[..8], &before[..8], "the range was actually permuted");
    assert!(after.iter().all(|&(c, g, _)| g == 2 * c));
}
