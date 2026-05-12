//! HVG selection has moved to `data_beans_alg::hvg`. This module
//! re-exports the public surface so existing `crate::hvg::*` call sites
//! in senna keep compiling unchanged.

pub use data_beans_alg::hvg::{select_hvg_streaming, HvgCliArgs, HvgSelection};

/// Resolve the effective `--multiome` flag and HVG settings against the
/// caller's CLI args. Mirrors the warn-and-zero rules from senna's
/// `gbe` and `itopic` entry points so both surface identical behavior:
///
/// - `--multiome` with a single input file is a no-op (no other backend
///   to glue against) → cleared.
/// - `--multiome` + HVG mixes scales across modalities (RNA counts vs
///   ATAC peak counts) → HVG is disabled.
///
/// Returns `(effective_multiome, effective_hvg_n, effective_hvg_list)`.
pub fn resolve_multiome_with_hvg(
    multiome: bool,
    n_files: usize,
    hvg: &HvgCliArgs,
) -> (bool, usize, Option<&str>) {
    let mut effective_multiome = multiome;
    let mut effective_hvg_n = hvg.n_hvg;
    let mut effective_hvg_list: Option<&str> = hvg.feature_list_file.as_deref();
    if effective_multiome && n_files < 2 {
        log::warn!("--multiome with a single input file is a no-op; ignoring.");
        effective_multiome = false;
    }
    if effective_multiome && (effective_hvg_n > 0 || effective_hvg_list.is_some()) {
        log::warn!(
            "--multiome + HVG mixes scales across modalities (e.g. RNA counts vs \
             ATAC peak counts) — disabling HVG for this run."
        );
        effective_hvg_n = 0;
        effective_hvg_list = None;
    }
    (effective_multiome, effective_hvg_n, effective_hvg_list)
}
