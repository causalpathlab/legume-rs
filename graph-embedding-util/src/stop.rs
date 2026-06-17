//! Graceful stop handler shared by `senna gbe` and the senna topic
//! models. First Ctrl+C sets the flag for graceful exit after the
//! current minibatch; second Ctrl+C forces an immediate abort.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Install a SIGINT/SIGTERM handler and return a shared stop flag.
/// Train loops should poll the flag at minibatch boundaries and finalize
/// outputs when it's set.
#[must_use]
pub fn setup_stop_handler() -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            if stop.load(Ordering::SeqCst) {
                eprintln!("\nSecond interrupt — aborting immediately");
                std::process::exit(1);
            }
            // Use eprintln, not info! — log macros hold internal locks
            // and will deadlock if the main thread is mid-log.
            eprintln!("\nInterrupt received — stopping after current minibatch (Ctrl+C again to force)...");
            stop.store(true, Ordering::SeqCst);
        })
        .expect("failed to set signal handler");
    }
    stop
}
