//! Auto-discover pinto output files under a given prefix.
//!
//! Pinto has no run manifest (unlike senna's `.senna.json`), so we glob
//! the filesystem:
//! - `{prefix}.propensity.parquet`            → level "final"
//! - `{prefix}.L{n}.propensity.parquet`       → level "L{n}"
//! - `{prefix}.draft.propensity.parquet`      → level "draft"
//!
//! Each level can have matching `.link_community.parquet` (lc / dsvd
//! runs) and `.gene_topic.parquet` siblings — `Level::*_path()` returns
//! them when present so downstream code can conditionally emit the
//! mesh plot or marker plots without re-checking the filesystem.

use regex::Regex;
use std::path::{Path, PathBuf};

/// Parsed level tag for one set of sibling pinto outputs.
#[derive(Clone, Debug)]
pub struct Level {
    /// User-facing tag used in output file names: "final", "L0", "draft", ...
    pub tag: String,
    /// Sort key (lower = plotted earlier). `final` and `draft` bracket
    /// the numeric L-levels for natural ordering in the output dir.
    pub sort_key: i32,
    /// Infix used in parquet names: "" for final, ".L0", ".draft", etc.
    pub infix: String,
}

impl Level {
    pub fn propensity_path(&self, prefix: &str) -> PathBuf {
        PathBuf::from(format!("{}{}.propensity.parquet", prefix, self.infix))
    }
    pub fn link_community_path(&self, prefix: &str) -> PathBuf {
        PathBuf::from(format!("{}{}.link_community.parquet", prefix, self.infix))
    }
    pub fn gene_topic_path(&self, prefix: &str) -> PathBuf {
        PathBuf::from(format!("{}{}.gene_topic.parquet", prefix, self.infix))
    }
}

/// User selector: `all` | `final` | `draft` | comma-list (`final,L0,draft`).
#[derive(Clone, Debug)]
pub enum LevelSelector {
    All,
    Explicit(Vec<String>),
}

impl LevelSelector {
    pub fn parse(s: &str) -> Self {
        let s = s.trim();
        if s.eq_ignore_ascii_case("all") {
            LevelSelector::All
        } else {
            LevelSelector::Explicit(
                s.split(',')
                    .map(|p| p.trim().to_string())
                    .filter(|p| !p.is_empty())
                    .collect(),
            )
        }
    }

    fn accepts(&self, tag: &str) -> bool {
        match self {
            LevelSelector::All => true,
            LevelSelector::Explicit(v) => v.iter().any(|p| p.eq_ignore_ascii_case(tag)),
        }
    }
}

/// Discover all sibling propensity parquets sharing this prefix.
///
/// Returns levels in natural plotting order: `final` → `L0 … Ln` → `draft`.
/// Filters by `selector`. Empty return is an error at the caller level.
pub fn discover_levels(prefix: &str, selector: &LevelSelector) -> anyhow::Result<Vec<Level>> {
    let (dir, stem) = split_prefix(prefix)?;
    let re = Regex::new(&format!(
        r"^{}(\.L(?P<lvl>\d+)|\.draft)?\.propensity\.parquet$",
        regex::escape(&stem)
    ))?;

    let mut out: Vec<Level> = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let caps = match re.captures(&name) {
            Some(c) => c,
            None => continue,
        };

        let (tag, sort_key, infix) = if let Some(lvl) = caps.name("lvl") {
            let n: i32 = lvl.as_str().parse().unwrap_or(0);
            (format!("L{n}"), n, format!(".L{n}"))
        } else if name.contains(".draft.") {
            ("draft".to_string(), i32::MAX, ".draft".to_string())
        } else {
            ("final".to_string(), i32::MIN, String::new())
        };

        if !selector.accepts(&tag) {
            continue;
        }
        out.push(Level {
            tag,
            sort_key,
            infix,
        });
    }

    out.sort_by_key(|l| l.sort_key);
    if out.is_empty() {
        anyhow::bail!("no propensity parquets matched prefix {prefix} with selector {selector:?}");
    }
    Ok(out)
}

fn split_prefix(prefix: &str) -> anyhow::Result<(PathBuf, String)> {
    let p = Path::new(prefix);
    let dir = p
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| d.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = p
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid prefix: {prefix}"))?
        .to_string();
    Ok((dir, stem))
}
