//! `faba gem-encoder` — the masked generative sibling of [`crate::gem`].
//!
//! Both fit the same geometry over the same spliced+unspliced inputs, but from
//! opposite directions. `gem` is discriminative: NCE over cell↔feature edges,
//! with a per-gene splice offset and a velocity read off the dictionaries.
//! `gem-encoder` is generative and amortized: an encoder reads a cell's top-K
//! genes with BOTH tracks attached, pools each track over that context by a
//! masked value-weighted sum, and a splice-aware ETM decoder imputes the
//! held-out track.
//!
//! The model is parameterized around the mechanism, `u + δ → s`: nascent
//! pre-mRNA is transcribed first and matures into spliced mRNA, so the
//! **unspliced embedding is the base** `ρ` and the spliced one is `ρ + δ`. That
//! makes `⟨α_t, δ_g⟩` the steady-state `log(β_g/γ_g)` of the RNA-velocity ODE —
//! see [`candle_util::decoder::gem_etm`] for the derivation.
//!
//! Note this is the OPPOSITE base from `faba gem`, whose `δ_g` shifts spliced →
//! unspliced. The two write same-named `delta_feature_embedding.parquet` files
//! that are not directly comparable; `{out}.gem.json` records `delta_base`.

pub mod args;
/// Per-cell inference: streams the sparse backend, emitting **log θ**.
pub mod infer;
pub mod load;
/// Cell QC and the per-level training tensors, between loading and fitting.
pub mod prepare;
/// Diagnostics: the score trace, the splice-ratio check, `{out}.gem.json`.
pub mod report;
pub mod run;
/// RNA velocity — the dictionary operator `v = P·θ` and its tables.
pub mod velocity;
/// Every table this run puts on disk, and the QC / non-finite gates on them.
pub mod write;
