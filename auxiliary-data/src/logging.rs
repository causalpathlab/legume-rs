//! Shared progress-bar style and logger setup for the workspace.
//!
//! Every binary (senna, faba, …) and library (graph-embedding-util, …) draws
//! rayon progress bars through [`new_progress_bar`] so they share one visual
//! style and register with one [`MULTI_PROGRESS`]; [`init_logger`] wraps
//! `env_logger` in `indicatif_log_bridge` so `log` output renders above the
//! bars instead of corrupting them. Centralised here (rather than duplicated in
//! `senna::logging` and `graph_embedding_util::progress`) so a single
//! definition keeps every crate's bars visually consistent.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::LazyLock;

/// Global `MultiProgress` that every workspace progress bar registers with, so
/// `log` output routed through `indicatif_log_bridge` interleaves cleanly above
/// the bars instead of corrupting them.
pub static MULTI_PROGRESS: LazyLock<MultiProgress> = LazyLock::new(MultiProgress::new);

/// Install `env_logger` wrapped in `indicatif_log_bridge::LogWrapper` so log
/// messages render above any active progress bar. `verbose` selects between
/// `matrix_util::common_io::{VERBOSE,QUIET}_LOG_FILTER`; an external `RUST_LOG`
/// still overrides the default filter.
pub fn init_logger(verbose: bool) {
    let default_filter = if verbose {
        matrix_util::common_io::VERBOSE_LOG_FILTER
    } else {
        matrix_util::common_io::QUIET_LOG_FILTER
    };
    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
            .build();
    let max_level = logger.filter();
    let _ = indicatif_log_bridge::LogWrapper::new(MULTI_PROGRESS.clone(), logger)
        .try_init()
        .map(|()| log::set_max_level(max_level));
}

/// Create a progress bar registered with the shared [`MULTI_PROGRESS`] and
/// styled with the standard template (`[elapsed] bar pos/len (eta) msg`).
#[must_use]
pub fn new_progress_bar(len: u64) -> ProgressBar {
    let prog_bar = MULTI_PROGRESS.add(ProgressBar::new(len));
    prog_bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    prog_bar
}
