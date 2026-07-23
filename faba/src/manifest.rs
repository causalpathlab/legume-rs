//! `{out}.gem.json` — WHICH faba program produced a prefix, and what its tables
//! mean.
//!
//! # Why this exists
//!
//! faba's gem-family steps write parquet tables under a shared prefix, and the
//! downstream steps are handed nothing but that prefix. Several of those tables
//! have the SAME name and shape across producers while meaning different things
//! — `{out}.cell_embedding.parquet` is a Euclidean embedding from `faba gem` and
//! a topic membership from `faba gem-encoder` — so a consumer that guesses gets
//! a plausible wrong answer rather than an error. This file is the one place a
//! consumer can ask.
//!
//! Both producers write it. On the consuming side only `faba annotate` reads it
//! so far: `faba lineage` still takes a gem prefix blind, including for the
//! co-embedded term-ORA in `lineage::traj_annotation`, which is the same call
//! `annotate --mode` exists to arbitrate. Wiring it is the obvious next step and
//! is why [`detect`] is public rather than private to `annotate`.
//!
//! # Why not `model.json`
//!
//! That was the old name and it was a non-name: every step here fits a model, so
//! "model" discriminated nothing, and the only way to tell a `gem` prefix from a
//! `gem-encoder` prefix was that `gem` wrote no such file at all. Absence is a
//! weak signal — a typo'd prefix, an interrupted run, and a genuine embedding
//! run are indistinguishable under it. Both producers now write this file, so
//! detection is POSITIVE on both sides.
//!
//! The old name is still read, so prefixes produced before this change keep
//! working; [`detect`] reports that it fell back.

use anyhow::Context;
use log::info;
use serde_json::{Map, Value};

/// What kind of run produced a prefix.
///
/// This is the distinction consumers actually need: not which subcommand ran,
/// but whether the outputs live in a shared metric space or on a simplex.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RunKind {
    /// `faba gem` — an embedding run. Cells and genes are placed in one
    /// Euclidean space, so nearest-centroid is meaningful.
    Embedding,
    /// `faba gem-encoder` / `gem-topic` — a topic model. The estimands are `β`
    /// and `θ`; there is no identified cell↔gene direction, so nearest-centroid
    /// is not meaningful and marker calls must go through enrichment.
    Topic,
}

impl RunKind {
    /// The `model_type` string written into the manifest.
    #[must_use]
    pub fn model_type(self) -> &'static str {
        match self {
            Self::Embedding => "gem-embedding",
            // The trainer owns the topic string, including its simplex-map
            // suffix — re-spelling it here would let the two drift.
            Self::Topic => candle_util::vae::masked_gem::MODEL_TYPE,
        }
    }

    /// Classify a `model_type` by PREFIX, not exact match, so a producer can
    /// change its simplex map (`gem-encoder-softmax` today, `gem-encoder-sbp` on
    /// older files) without every consumer needing a new arm.
    #[must_use]
    pub fn from_model_type(s: &str) -> Option<Self> {
        if s.starts_with("gem-encoder") {
            Some(Self::Topic)
        } else if s.starts_with("gem-embedding") {
            Some(Self::Embedding)
        } else {
            None
        }
    }
}

/// A manifest that was found and understood.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Detected {
    pub kind: RunKind,
    /// Read from the pre-rename name. Callers warn on this so a user knows the
    /// prefix predates the current contract.
    pub legacy: bool,
}

#[must_use]
pub fn path(prefix: &str) -> String {
    format!("{prefix}.gem.json")
}

/// The pre-rename name, still accepted on read. `faba gem-encoder` wrote this
/// and `faba gem` wrote nothing, so a legacy hit is always a topic model.
#[must_use]
pub fn legacy_path(prefix: &str) -> String {
    format!("{prefix}.model.json")
}

/// What produced this prefix, or `None` if nothing here says.
///
/// `None` covers all the cases a consumer must not distinguish by guessing: no
/// manifest, an unreadable one, a truncated one, and a `model_type` from some
/// other tool. Callers decide what to do about it — none of them should silently
/// assume a kind.
#[must_use]
pub fn detect(prefix: &str) -> Option<Detected> {
    let read = |p: &str| -> Option<RunKind> {
        let text = std::fs::read_to_string(p).ok()?;
        let v: Value = serde_json::from_str(&text).ok()?;
        RunKind::from_model_type(v["model_type"].as_str()?)
    };
    if let Some(kind) = read(&path(prefix)) {
        return Some(Detected {
            kind,
            legacy: false,
        });
    }
    read(&legacy_path(prefix)).map(|kind| Detected { kind, legacy: true })
}

/// Write `{prefix}.gem.json`, stamping `model_type` from `kind`.
///
/// `extra` carries whatever the producer wants a reader to know beyond the kind
/// (`n_genes`, the latent contract, `velocity_common_mode`, …). Built as a JSON
/// value rather than a format string so a field containing a quote or a NaN
/// cannot produce a file that parses as something else.
pub fn write(prefix: &str, kind: RunKind, extra: Map<String, Value>) -> anyhow::Result<()> {
    let mut obj = Map::new();
    obj.insert("model_type".into(), Value::String(kind.model_type().into()));
    obj.extend(extra);
    let path = path(prefix);
    let mut text = serde_json::to_string_pretty(&Value::Object(obj))
        .with_context(|| format!("serializing {path}"))?;
    text.push('\n');
    std::fs::write(&path, text).with_context(|| format!("writing {path}"))?;
    info!("wrote {path}");
    Ok(())
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
