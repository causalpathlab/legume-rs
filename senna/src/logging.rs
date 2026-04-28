use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::LazyLock;

/// Global `MultiProgress` that every senna progress bar registers with,
/// so that `log::info!` output routed through `indicatif_log_bridge` can
/// interleave cleanly above the bars instead of corrupting them.
pub static MULTI_PROGRESS: LazyLock<MultiProgress> = LazyLock::new(MultiProgress::new);

/// Install `env_logger` wrapped in `indicatif_log_bridge::LogWrapper` so
/// log messages render above any active progress bar.
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

/// Create a progress bar registered with the shared `MULTI_PROGRESS` and
/// styled with senna's standard template (`[elapsed] bar pos/len (eta)`).
pub fn new_progress_bar(len: u64) -> ProgressBar {
    let pb = MULTI_PROGRESS.add(ProgressBar::new(len));
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    pb
}
