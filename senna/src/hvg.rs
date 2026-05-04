//! HVG selection has moved to `data_beans_alg::hvg`. This module
//! re-exports the public surface so existing `crate::hvg::*` call sites
//! in senna keep compiling unchanged.

pub use data_beans_alg::hvg::{select_hvg_streaming, HvgCliArgs, HvgSelection};
