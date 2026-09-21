//! `pinto`'s feature-pair graph is the canonical
//! `legume_numeric::matrix::pair_graph::FeaturePairGraph` under its historical
//! name. The shared implementation lives in legume_numeric::matrix so that
//! `senna gbe` (and any future feature-pair consumers) can use it too.

pub use legume_numeric::matrix::pair_graph::FeaturePairGraph;

#[cfg(test)]
pub use legume_numeric::matrix::pair_graph::test_graph_from_edges;
