//! Shared logger setup for the workspace.
//!
//! The progress-bar style and the single [`MULTI_PROGRESS`] now live in
//! [`matrix_util::progress`] — the lowest common dependency, so `data-beans`
//! and friends can share them without a dependency cycle (this crate depends
//! *on* `data-beans`, so the primitive can't live here). This module
//! re-exports [`new_progress_bar`], [`new_spinner`], and [`MULTI_PROGRESS`] so
//! every existing caller (`auxiliary_data::logging::new_progress_bar`, senna,
//! graph-embedding-util) keeps working unchanged, and adds [`init_logger`],
//! which wraps `env_logger` in `indicatif_log_bridge` so `log` output renders
//! above the bars instead of corrupting them.

pub use matrix_util::progress::{new_progress_bar, new_spinner, MULTI_PROGRESS};

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
