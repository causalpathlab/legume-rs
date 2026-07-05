//! Re-export of the shared `SimplePPT` principal-graph fitter, which now lives
//! in [`matrix_util::principal_graph`]. Kept as a thin module so the existing
//! `crate::principal_graph::*` paths throughout senna resolve unchanged.

pub use matrix_util::principal_graph::*;
