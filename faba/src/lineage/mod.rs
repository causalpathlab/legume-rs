//! Velocity-oriented lineage inference over a `faba gem` embedding, driven by
//! [`crate::run_lineage::run_lineage`].
//!
//! The generic numeric primitives live in `matrix-util`: k-means centroids
//! ([`matrix_util::principal_graph::kmeans_centroids`]), the K×K distance matrix
//! ([`matrix_util::principal_graph::pairwise_sqdist_rows_to_rows`]) + MST
//! ([`matrix_util::principal_graph::mst_from_sqdist`]), and the Slingshot curves
//! ([`matrix_util::principal_curve`]). This module only holds the faba-specific
//! velocity [`orient`]ation of the tree (δ is a gem output).

pub mod orient;
