//! Velocity-oriented lineage inference over a `faba gem` embedding, driven by
//! [`crate::lineage::run::run_lineage`].
//!
//! The generic numeric primitives live in `matrix-util`: seeded k-means centroids
//! ([`matrix_util::principal_graph::kmeans_centroids_seeded`]), the K×K distance matrix
//! ([`matrix_util::principal_graph::pairwise_sqdist_rows_to_rows`]) + MST
//! ([`matrix_util::principal_graph::mst_from_sqdist`]), and the Slingshot curves
//! ([`matrix_util::principal_curve`]). This module holds two faba-specific pieces on top of
//! that generic tree: the velocity [`orient`]ation of edges (δ from gem), and the local,
//! root-free [`branch`] structure (junctions + sibling branches).

/// The `faba lineage` command-line surface.
pub mod args;
mod cluster;
mod layout;
mod root;
/// The `faba lineage` run. Binary entry: [`run::run_lineage`].
pub mod run;
mod traj_annotation;
mod velocity_grid;
mod write;
// Parked for the root-free sibling-branch association test (`faba dyn-assoc`); its former
// consumer (the junction-support bootstrap) was removed with the velocity-forest rework.
#[allow(dead_code)]
pub mod branch;
pub mod forest;
pub mod orient;
