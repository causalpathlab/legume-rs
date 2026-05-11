//! Senna-style progress bar for rayon-parallel builds. Inlined here so
//! this crate doesn't depend on senna; template matches `senna::logging`
//! byte-for-byte so multi-bar output stays visually consistent when this
//! crate runs under senna.

use indicatif::{ProgressBar, ProgressStyle};

pub fn new_progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    pb
}
