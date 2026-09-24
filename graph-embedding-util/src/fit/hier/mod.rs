//! Phase 1 by an **exact two-level softmax** over gene modules — see
//! [`step`] for the objective. Units are pseudobulks at every collapse level
//! plus the phase-1 cell subsample; genes live in one module each.

pub mod cis_gates;
pub mod params;
pub mod partition;
pub mod step;
pub mod train;
pub mod units;

pub use cis_gates::{CisCoupling, CisGateReadout, CisGates};
pub use params::PresetGenes;
pub use train::{train, HierConfig, HierOutput};
pub use units::UnitTable;
