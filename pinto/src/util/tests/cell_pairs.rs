use crate::test_support::make_test_graph;
use crate::util::cell_pairs::*;
use crate::util::common::*;

#[test]
fn test_connected_components_single() {
    let graph = make_test_graph(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    let (labels, n_components) = connected_components(&graph);
    assert_eq!(n_components, 1);
    assert!(labels.iter().all(|&l| l == labels[0]));
}

#[test]
fn test_connected_components_two_cliques() {
    let graph = make_test_graph(6, vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]);
    let (labels, n_components) = connected_components(&graph);
    assert_eq!(n_components, 2);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[3], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn test_connected_components_isolates() {
    let graph = make_test_graph(4, vec![]);
    let (labels, n_components) = connected_components(&graph);
    assert_eq!(n_components, 4);
    let unique: HashSet<usize> = labels.iter().cloned().collect();
    assert_eq!(unique.len(), 4);
}
