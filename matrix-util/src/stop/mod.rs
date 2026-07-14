//! Graceful stop on Ctrl+C, shared process-wide.
//!
//! First SIGINT/SIGTERM sets the flag; long-running loops poll it at a safe boundary (a training
//! minibatch, a bootstrap replicate) and finalize whatever they have. A second SIGINT aborts
//! immediately, for when the first one is taking too long to land.
//!
//! The flag is installed **once per process** and handed out on every call. `ctrlc::set_handler`
//! panics on a second registration, and a workspace where one binary trains a model, then
//! bootstraps an annotation, is exactly the shape that trips it.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

#[cfg(test)]
mod tests;

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
        interrupted(what, done.len(), n);
    }
    Ok(done)
}

/// Has the user already asked us to stop?
///
/// The flag **latches** — nothing ever clears it — so a stage that starts after an interrupt has
/// landed will complete *zero* replicates. Such a stage must check this and decline to run, rather
/// than discover it the hard way and report "interrupted before a single replicate completed" as
/// an **error**: that error propagates, and the interrupt then destroys the output of every stage
/// that had already finished — the precise outcome this module exists to prevent.
#[must_use]
pub fn stopped() -> bool {
    stop_flag().load(Ordering::Relaxed)
}

/// [`par_replicates`], but **folding** each replicate into a running accumulator instead of
/// keeping it. Returns `(replicates completed, the accumulator)`, or `None` if not one completed.
///
/// Same interrupt semantics, same argument. Reach for this one when a replicate's output is large
/// and only its *aggregate* is ever read: `par_replicates` holds all `n` outputs at once, which is
/// fine for a 200-draw bootstrap and ruinous for a 10 000-draw permutation null. The support null
/// emits two `n`-cell vectors per shuffle — **1.8 GB live at `P = 10 000`** — and then sums them
/// and throws them away. Folded, the same run holds one accumulator per thread.
///
/// `combine` must be **associative**: rayon splits the range into subtrees and combines them in
/// whatever order they finish, never in index order. It need not be commutative, and there is no
/// identity to supply.
///
/// Returning `None` rather than erroring on an empty run is deliberate — see [`stopped`]. The
/// caller knows what a zero-replicate result means for *its* statistic; this combinator does not.
pub fn par_reduce_replicates<A, F, C>(
    n: usize,
    what: &str,
    f: F,
    combine: C,
) -> anyhow::Result<Option<(usize, A)>>
where
    A: Send,
    F: Fn(usize) -> anyhow::Result<A> + Sync + Send,
    C: Fn(A, A) -> A + Sync + Send,
{
    reduce_in(&stop_flag(), n, what, f, combine)
}

/// [`par_reduce_replicates`] against a caller-supplied flag.
///
/// The public entry point reads the process-wide `OnceLock` flag, which no test can set without
/// poisoning every other test in the binary — so the interrupt path, and the `done` count that
/// becomes the denominator of every support-null p-value, would otherwise be untestable.
fn reduce_in<A, F, C>(
    stop: &AtomicBool,
    n: usize,
    what: &str,
    f: F,
    combine: C,
) -> anyhow::Result<Option<(usize, A)>>
where
    A: Send,
    F: Fn(usize) -> anyhow::Result<A> + Sync + Send,
    C: Fn(A, A) -> A + Sync + Send,
{
    use rayon::prelude::*;
    let done = AtomicUsize::new(0);
    let acc = (0..n)
        .into_par_iter()
        .filter(|_| !stop.load(Ordering::Relaxed))
        .map(|i| {
            let a = f(i)?;
            done.fetch_add(1, Ordering::Relaxed);
            anyhow::Ok(a)
        })
        // `try_reduce_with`, not `try_reduce`: no identity to invent, and none to get subtly
        // wrong. `None` is exactly the case where every replicate was filtered out.
        .try_reduce_with(|a, b| Ok(combine(a, b)))
        .transpose()?;

    let done = done.load(Ordering::Relaxed);
    if done < n {
        interrupted(what, done, n);
    }
    Ok(acc.map(|a| (done, a)))
}

fn interrupted(what: &str, done: usize, n: usize) {
    log::warn!(
        "{what} interrupted: {done} of {n} replicates completed; using those (estimates now \
         resolve to ~1/{done})."
    );
}
