//! `--embedding-dim`, for every command whose width can come from a table
//! given from outside: a number, or `auto`, which takes the width of the
//! given feature embedding (and, where a command has a rule of its own, that
//! rule when none is given).

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingDim {
    /// The width of the given feature embedding, or the command's own rule.
    Auto,
    /// A width given on the command line.
    Fixed(usize),
}

impl EmbeddingDim {
    /// The width against a given table's: `Auto` takes the table's, a fixed
    /// width must agree with it, and `Auto` with no table is left to the
    /// caller (`Ok(None)`).
    pub fn resolve(self, table_h: Option<usize>) -> anyhow::Result<Option<usize>> {
        Ok(match (self, table_h) {
            (Self::Auto, Some(h)) => {
                log::info!("--embedding-dim auto: the given feature embedding's width H = {h}");
                Some(h)
            }
            (Self::Fixed(d), Some(h)) => {
                anyhow::ensure!(
                    d == h,
                    "--embedding-dim {d} disagrees with the given feature embedding's width {h}; \
                     pass `auto` (or {h}) to take the table's width"
                );
                Some(d)
            }
            (Self::Auto, None) => None,
            (Self::Fixed(d), None) => Some(d),
        })
    }
}

impl FromStr for EmbeddingDim {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        match s.parse::<usize>() {
            Ok(0) => Err("0 is not a width; pass `auto` to take a given table's".into()),
            Ok(d) => Ok(Self::Fixed(d)),
            Err(_) => Err(format!("expected a width or `auto`, got `{s}`")),
        }
    }
}

impl fmt::Display for EmbeddingDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Fixed(d) => write!(f, "{d}"),
        }
    }
}

/// A fixed width is recorded as its number and `auto` as the word, and a
/// recorded `0` (the sentinel of earlier versions) reads back as `auto`.
impl serde::Serialize for EmbeddingDim {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => s.serialize_str("auto"),
            Self::Fixed(d) => s.serialize_u64(*d as u64),
        }
    }
}

impl<'de> serde::Deserialize<'de> for EmbeddingDim {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Number(u64),
            Word(String),
        }
        match Raw::deserialize(d)? {
            Raw::Number(0) => Ok(Self::Auto),
            Raw::Number(n) => Ok(Self::Fixed(n as usize)),
            Raw::Word(w) => w.parse().map_err(serde::de::Error::custom),
        }
    }
}

#[cfg(test)]
#[path = "embedding_dim_tests.rs"]
mod embedding_dim_tests;
