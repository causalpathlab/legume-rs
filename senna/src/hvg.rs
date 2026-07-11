//! HVG selection has moved to `data_beans_alg::hvg`. This module
//! re-exports the public surface so existing `crate::hvg::*` call sites
//! in senna keep compiling unchanged.

pub use data_beans_alg::hvg::{
    load_must_train, select_hvg_streaming, union_indices, HvgCliArgs, HvgSelection,
};

/// The `--multiome` flag and HVG settings a run will actually use, after the
/// warn-and-zero rules in [`resolve_multiome_with_hvg`] have been applied.
pub struct EffectiveHvg<'a> {
    pub multiome: bool,
    pub n_hvg: usize,
    pub feature_list_file: Option<&'a str>,
    pub must_train_file: Option<&'a str>,
}

impl EffectiveHvg<'_> {
    /// Is any feature selection happening? When not, every feature is kept and a
    /// force-include list has nothing to rescue.
    pub fn selection_on(&self) -> bool {
        self.n_hvg > 0 || self.feature_list_file.is_some()
    }
}

/// Resolve the effective `--multiome` flag and HVG settings against the
/// caller's CLI args. Mirrors the warn-and-zero rules from senna's
/// `gbe` and `masked-topic` entry points so both surface identical behavior:
///
/// - `--multiome` with a single input file is a no-op (no other backend
///   to glue against) → cleared.
/// - `--multiome` + HVG mixes scales across modalities (RNA counts vs
///   ATAC peak counts) → HVG is disabled.
pub fn resolve_multiome_with_hvg<'a>(
    multiome: bool,
    n_files: usize,
    hvg: &'a HvgCliArgs,
) -> EffectiveHvg<'a> {
    let mut out = EffectiveHvg {
        multiome,
        n_hvg: hvg.n_hvg,
        feature_list_file: hvg.feature_list_file.as_deref(),
        must_train_file: hvg.must_train_features.as_deref(),
    };
    if out.multiome && n_files < 2 {
        log::warn!("--multiome with a single input file is a no-op; ignoring.");
        out.multiome = false;
    }
    if out.multiome && out.selection_on() {
        log::warn!(
            "--multiome + HVG mixes scales across modalities (e.g. RNA counts vs \
             ATAC peak counts) — disabling HVG for this run."
        );
        out.n_hvg = 0;
        out.feature_list_file = None;
    }
    out
}
