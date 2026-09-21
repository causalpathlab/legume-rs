//! Re-export of the shared `SimplePPT` principal-graph fitter, which now lives
//! in [`legume_numeric::matrix::principal_graph`]. Kept as a thin module so the existing
//! `crate::principal_graph::*` paths throughout senna resolve unchanged.

pub use legume_numeric::matrix::principal_graph::*;
