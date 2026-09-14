//! Phase 1 by an **exact two-level softmax** over gene modules — see
//! [`step`] for the objective. Units are pseudobulks at every collapse level
//! plus the phase-1 cell subsample; genes live in one module each.

pub mod params;
pub mod partition;
pub mod step;
pub mod units;

pub use units::UnitTable;
