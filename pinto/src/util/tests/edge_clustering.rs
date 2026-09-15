//! The shared edge-cut flags resolve to the algorithm-side enum with the
//! run's seed, so a k-means cut is reproducible like a Leiden one.

use crate::link_community::profiles::EdgeClustering;
use crate::util::edge_clustering::EdgeClusterArgs;
use clap::Parser;

#[derive(Parser)]
struct Wrap {
    #[command(flatten)]
    edge: EdgeClusterArgs,
}

fn parse(extra: &[&str]) -> EdgeClusterArgs {
    let mut argv = vec!["pinto"];
    argv.extend_from_slice(extra);
    Wrap::try_parse_from(argv).expect("args should parse").edge
}

#[test]
fn leiden_is_the_default_and_carries_the_seed() {
    match parse(&[]).resolve(42) {
        EdgeClustering::Leiden {
            knn,
            resolution,
            target,
            seed,
        } => {
            assert_eq!(knn, 30);
            assert_eq!(resolution, 1.0);
            assert_eq!(target, None);
            assert_eq!(seed, 42);
        }
        other => panic!("expected leiden, got {other:?}"),
    }
}

#[test]
fn kmeans_resolves_with_the_count_iterations_and_seed() {
    let resolved = parse(&[
        "--edge-cluster-method",
        "kmeans",
        "--n-edge-clusters",
        "25",
        "--kmeans-max-iter",
        "7",
    ])
    .resolve(9);
    match resolved {
        EdgeClustering::Kmeans {
            n_clusters,
            max_iter,
            seed,
        } => {
            assert_eq!(n_clusters, Some(25));
            assert_eq!(max_iter, 7);
            assert_eq!(seed, 9);
        }
        other => panic!("expected kmeans, got {other:?}"),
    }
}
