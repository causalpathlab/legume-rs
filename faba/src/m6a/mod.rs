//! `faba m6a` — DART-seq m6A site calling.
//!
//! The subcommand entry only; the read-level machinery it drives lives in
//! [`crate::editing`], shared with [`crate::atoi`].

/// The `faba m6a` run. Binary entry: [`run::run_m6a`].
pub mod run;
