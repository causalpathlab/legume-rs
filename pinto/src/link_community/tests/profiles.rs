use crate::link_community::model::LinkProfileStore;
use crate::link_community::profiles::*;
use crate::util::common::*;

#[test]
fn test_coarsen_edge_profiles() {
    // 4 edges, proj_dim=3, 2 clusters of cells
    let profiles_data = vec![
        1.0, 2.0, 3.0, // edge 0: cells (0,1) → cluster pair (0,0)
        4.0, 5.0, 6.0, // edge 1: cells (0,2) → cluster pair (0,1)
        7.0, 8.0, 9.0, // edge 2: cells (1,2) → cluster pair (0,1)
        2.0, 3.0, 4.0, // edge 3: cells (2,3) → cluster pair (1,1)
    ];
    let store = LinkProfileStore::new(profiles_data, 4, 3);
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3)];
    let cell_labels = vec![0, 0, 1, 1]; // cells 0,1 → cluster 0; cells 2,3 → cluster 1

    let (super_store, f2s) = coarsen_edge_profiles(&store, &edges, &cell_labels);

    // Edge 0: (0,1) → labels (0,0) → key (0,0) → super 0
    // Edge 1: (0,2) → labels (0,1) → key (0,1) → super 1
    // Edge 2: (1,2) → labels (0,1) → key (0,1) → super 1
    // Edge 3: (2,3) → labels (1,1) → key (1,1) → super 2
    assert_eq!(super_store.n_edges, 3);
    assert_eq!(f2s[0], f2s[0]); // trivially
    assert_eq!(f2s[1], f2s[2]); // same super-edge
    assert_ne!(f2s[0], f2s[1]); // different super-edges

    // Super-edge 1 should be sum of edges 1 and 2
    let se1 = f2s[1];
    let expected: Vec<f32> = vec![4.0 + 7.0, 5.0 + 8.0, 6.0 + 9.0];
    assert_eq!(super_store.profile(se1), &expected[..]);
}

#[test]
fn test_transfer_labels() {
    let f2s = vec![0, 1, 1, 2, 0];
    let super_mem = vec![2, 0, 1];
    let fine_mem = transfer_labels(&f2s, &super_mem);
    assert_eq!(fine_mem, vec![2, 0, 0, 1, 2]);
}

#[test]
fn test_compute_node_membership() {
    let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3)];
    let membership = vec![0, 0, 1, 1];
    let nm = compute_node_membership(&edges, &membership, 4, 2);

    assert_eq!(nm.nrows(), 4);
    assert_eq!(nm.ncols(), 2);

    // Cell 0: edges 0,1 → communities 0,0 → [1.0, 0.0]
    assert!((nm[(0, 0)] - 1.0).abs() < 1e-6);
    assert!((nm[(0, 1)] - 0.0).abs() < 1e-6);

    // Cell 2: edges 1,2,3 → communities 0,1,1 → [1/3, 2/3]
    assert!((nm[(2, 0)] - 1.0 / 3.0).abs() < 1e-6);
    assert!((nm[(2, 1)] - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn test_refine_projection_basis() {
    // Create a simple centroid matrix [10 genes × 3 communities]
    let mut centroids = Mat::zeros(10, 3);
    // Community 0: genes 0-3 active
    for g in 0..4 {
        centroids[(g, 0)] = 5.0;
    }
    // Community 1: genes 4-6 active
    for g in 4..7 {
        centroids[(g, 1)] = 5.0;
    }
    // Community 2: genes 7-9 active
    for g in 7..10 {
        centroids[(g, 2)] = 5.0;
    }

    let basis = refine_projection_basis(&centroids, 3).unwrap();
    assert_eq!(basis.nrows(), 10);
    assert_eq!(basis.ncols(), 3);

    // All values should be finite
    for i in 0..basis.nrows() {
        for j in 0..basis.ncols() {
            assert!(basis[(i, j)].is_finite());
        }
    }
}

#[test]
fn test_refine_more_dims_than_communities() {
    let centroids = Mat::from_fn(10, 2, |g, c| if g % 2 == c { 3.0 } else { 0.5 });
    let basis = refine_projection_basis(&centroids, 5).unwrap();
    assert_eq!(basis.nrows(), 10);
    assert_eq!(basis.ncols(), 5);
}
