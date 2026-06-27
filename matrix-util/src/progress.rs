//! Workspace-wide progress-bar style and the single [`MULTI_PROGRESS`] that
//! every bar registers with.
//!
//! This lives in `matrix-util` — the lowest common dependency of `data-beans`,
//! `data-beans-alg`, `auxiliary-data`, and the binaries — so the whole
//! workspace shares ONE style definition and ONE `MultiProgress`. That single
//! `MultiProgress` is what lets `indicatif_log_bridge` (installed by
//! `auxiliary_data::logging::init_logger`) interleave `log` output cleanly
//! above the bars. Duplicating the bar/`MultiProgress` in another crate (as
//! `graph-embedding-util` once did) silently spawns a second, *unbridged*
//! `MultiProgress` whose bars corrupt the log output — so every crate must
//! draw through this module. `auxiliary_data::logging` re-exports these names
//! for backward compatibility.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::LazyLock;
use std::time::Duration;

/// The single global `MultiProgress` every workspace progress bar registers
/// with, so `log` output routed through `indicatif_log_bridge` interleaves
/// cleanly above the bars instead of corrupting them.
pub static MULTI_PROGRESS: LazyLock<MultiProgress> = LazyLock::new(MultiProgress::new);

/// Standard bounded-bar template: `[elapsed] bar pos/len (eta) msg`. The
/// trailing `{msg}` is empty unless a caller sets one (e.g.
/// `new_progress_bar(n).with_message("blocks")`).
const BAR_TEMPLATE: &str = "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}";

/// Tick frames shared by every [`new_spinner`].
const SPINNER_TICKS: &str = "⠁⠂⠄⡀⢀⠠⠐⠈ ";

/// Create a progress bar registered with the shared [`MULTI_PROGRESS`] and
/// styled with the standard template. Attach a trailing label with
/// [`ProgressBar::with_message`], e.g. `new_progress_bar(n).with_message("blocks")`.
#[must_use]
pub fn new_progress_bar(len: u64) -> ProgressBar {
    let prog_bar = MULTI_PROGRESS.add(ProgressBar::new(len));
    prog_bar.set_style(
        ProgressStyle::with_template(BAR_TEMPLATE)
            .unwrap()
            .progress_chars("##-"),
    );
    prog_bar
}

/// Create a spinner registered with the shared [`MULTI_PROGRESS`] for
/// unbounded / streaming work (no known total). `template` is an indicatif
/// spinner template (e.g. `"{spinner} streamed {pos} fragments ({per_sec})"`);
/// the shared tick frames and a 200 ms steady tick are applied so the spinner
/// animates and stays visually consistent across crates.
#[must_use]
pub fn new_spinner(template: &str) -> ProgressBar {
    let prog_bar = MULTI_PROGRESS.add(ProgressBar::new_spinner());
    prog_bar.set_style(
        ProgressStyle::with_template(template)
            .unwrap()
            .tick_chars(SPINNER_TICKS),
    );
    prog_bar.enable_steady_tick(Duration::from_millis(200));
    prog_bar
}
