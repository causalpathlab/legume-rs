//! Phase 1 by an **exact two-level softmax** over gene modules — see
//! [`step`] for the objective. Units are pseudobulks at every collapse level
//! plus the phase-1 cell subsample; genes live in one module each.

pub mod module_recollapse;
pub mod params;
pub mod partition;
pub mod step;
pub mod train;
pub mod units;

pub use module_recollapse::{merge_module_rows_host, ModuleRecollapseMap};
pub use params::PresetGenes;
pub use partition::Partition;
pub use train::{train, train_partitions, HierConfig, HierOutput};
pub use units::UnitTable;
