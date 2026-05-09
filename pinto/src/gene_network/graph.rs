//! `pinto`'s gene-pair graph is the canonical
//! `matrix_util::pair_graph::FeaturePairGraph` under its historical
//! name. The shared implementation lives in matrix-util so that
//! `senna gbe` (and any future feature-pair consumers) can use it too.

pub use matrix_util::pair_graph::FeaturePairGraph as GenePairGraph;

#[cfg(test)]
pub use matrix_util::pair_graph::test_graph_from_edges;
