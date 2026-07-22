//! `faba annotate` — marker-set cell-type annotation of a fitted run.
//!
//! Two paths, chosen by `--mode`: nearest-centroid in the shared embedding
//! space ([`run`], the default) and factor-program over-representation
//! ([`by_enrichment`], for the topic models). The module doc on [`run`] says
//! which geometry each one assumes and why they are not interchangeable.

/// `--mode enrichment`: the factor-program path for the topic models.
pub mod by_enrichment;
/// The `faba annotate` run. Binary entry: [`run::run_annotate`].
pub mod run;
