//! Graceful stop on Ctrl+C, shared process-wide.
//!
//! First SIGINT/SIGTERM sets the flag; long-running loops poll it at a safe boundary (a training
//! minibatch, a bootstrap replicate) and finalize whatever they have. A second SIGINT aborts
//! immediately, for when the first one is taking too long to land.
//!
//! The flag is installed **once per process** and handed out on every call. `ctrlc::set_handler`
//! panics on a second registration, and a workspace where one binary trains a model, then
//! bootstraps an annotation, is exactly the shape that trips it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

static STOP: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// The process-wide stop flag, installing the SIGINT/SIGTERM handler on first call.
///
/// Safe to call any number of times, from anywhere — every caller gets the same flag and the
/// handler is registered exactly once.
#[must_use]
pub fn stop_flag() -> Arc<AtomicBool> {
    STOP.get_or_init(|| {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let installed = ctrlc::set_handler(move || {
            if flag.swap(true, Ordering::SeqCst) {
                eprintln!("\nSecond interrupt — aborting immediately.");
                std::process::exit(130); // 128 + SIGINT, the shell's convention
            }
            // `eprintln!`, not `warn!`: the log macros take internal locks, and this runs on a
            // signal thread that may have interrupted the main thread mid-log.
            eprintln!(
                "\nInterrupt received — finishing the work in flight, then stopping and writing \
                 what completed (Ctrl+C again to abort now)."
            );
        });
        if installed.is_err() {
            // Someone else owns SIGINT. Not fatal — the flag simply never fires and the work
            // runs to completion, exactly as it did before this module existed.
            eprintln!(
                "warning: could not install the Ctrl+C handler; interrupts will not be graceful"
            );
        }
        stop
    })
    .clone()
}

/// Alias for [`stop_flag`], kept because eleven call sites across four crates use this name.
#[must_use]
pub fn setup_stop_handler() -> Arc<AtomicBool> {
    stop_flag()
}

/// Run `n` independent replicates in parallel, and if the user interrupts, keep the ones that
/// finished.
///
/// **A replicate is a whole answer, not a fragment of one.** Every statistic these loops produce
/// is an average over the replicates that ran, so dropping the ones that never started leaves a
/// *smaller* bootstrap, not a broken one: 47 of 200 is a 47-replicate bootstrap — noisier, but
/// unbiased, and worth incomparably more than nothing. That is the whole argument, and it is why
/// this is a shared combinator rather than a paragraph of comment in one function: the marker
/// bootstrap, the trajectory's junction bootstrap and the marker-panel null are the same shape,
/// take the same minutes, and deserve the same escape hatch.
///
/// `what` names the loop in the warning ("marker bootstrap"). Errors from `f` propagate and abort
/// as usual — an interrupt is not an error.
pub fn par_replicates<T, F>(n: usize, what: &str, f: F) -> anyhow::Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> anyhow::Result<T> + Sync + Send,
{
    use rayon::prelude::*;
    let stop = stop_flag();
    let done: Vec<T> = (0..n)
        .into_par_iter()
        .filter(|_| !stop.load(Ordering::Relaxed))
        .map(f)
        .collect::<anyhow::Result<Vec<_>>>()?;

    anyhow::ensure!(
        !done.is_empty(),
        "{what} was interrupted before a single replicate completed"
    );
    if done.len() < n {
        log::warn!(
            "{what} interrupted: {done} of {n} replicates completed; using those (estimates now \
             resolve to ~1/{done}).",
            done = done.len(),
        );
    }
    Ok(done)
}
