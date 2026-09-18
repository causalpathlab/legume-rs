//! `faba qc` and `faba qc-report`: the one place faba thresholds anything.
//!
//! The producers (`count`, `dartseq`, `atoi`, `apa`, `all`) are inclusive:
//! they write every called cell, every gene with a count, and every putative
//! editing site with its statistics, and they apply no p-value, effect-size or
//! reproducibility cutoff. `qc` reads such a directory and writes a new,
//! filtered fileset; `qc-report` shows what each `qc` knob keeps so the cut
//! is chosen on the data. Neither touches a BAM.

pub mod args;
pub mod layout;
pub mod matrix;
pub mod repool;
pub mod report;
pub mod run;
pub mod sites;

pub use args::{QcArgs, QcReportArgs};
pub use report::run_qc_report;
pub use run::run_qc;
