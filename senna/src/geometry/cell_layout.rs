//! Cell-level layout: Nyström projection of individual cells onto the PB
//! layout. The Nyström projector now lives in `matrix_util::layout` (shared
//! with `faba gem-annotate`); re-exported here to keep the
//! `crate::geometry::cell_layout::project_cells_nystrom` path stable.
//!
//! The former per-PB t-SNE fine-tune helpers (`refine_cells_local`,
//! `LocalRefineArgs`, `local_tsne_step`) were unused dead code and were
//! removed when the projector was lifted.

pub(crate) use matrix_util::layout::project_cells_nystrom;
