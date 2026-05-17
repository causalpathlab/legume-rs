//! Auto-discover pinto output files under a given prefix.
//!
//! Two discovery paths, tried in order:
//!
//! 1. **Filesystem glob** — `{prefix}{.L*,.draft,}.propensity.parquet`
//!    siblings. This is the lc / dsvd / prop convention; each Level
//!    gets paths derived from the `infix` field.
//! 2. **`.pinto.json` fallback** — when the glob finds nothing, read
//!    `{prefix}.pinto.json` and build a single `final` Level whose
//!    explicit paths point at `outputs.{propensity, link_community,
//!    gene_community}` (or the equivalent fields in `levels[]`). This
//!    is how cage / cage-mcmc plug in: their propensity-equivalent
//!    file is `cluster_propensity.parquet`, named via the JSON rather
//!    than discoverable by glob.
//!
//! `Level` carries explicit `PathBuf`s for the propensity / link-
//! community / gene-community parquets so both paths populate the same
//! shape — downstream plot code doesn't branch on which discovery
//! kind produced it.

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
    /// Explicit path to the propensity (or propensity-equivalent)
    /// parquet for this level. Always populated.
    pub propensity: PathBuf,
    /// Optional path to the link-community parquet (per-edge community
    /// labels — lc / dsvd). `None` for subcommands without a per-edge
    /// community concept (cage / cage-mcmc); plot skips mesh / edge
    /// overlays for those levels.
    pub link_community: Option<PathBuf>,
    /// Optional path to the gene-community (or feature-dictionary)
    /// parquet. `None` when the run didn't produce one.
    pub gene_community: Option<PathBuf>,
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
/// Order of operations:
/// 1. Filesystem glob for `{prefix}{.L*,.draft,}.propensity.parquet`.
///    This is the lc / dsvd / prop path; returns levels in natural
///    plotting order (`final` → `L0…Ln` → `draft`).
/// 2. If the glob found nothing, fall back to reading
///    `{prefix}.pinto.json` and constructing a single `final` Level
///    from `levels[0]`. This covers cage / cage-mcmc whose propensity
///    artifact is named `cluster_propensity.parquet` (no glob match).
///
/// Filters by `selector` in both paths. Empty result is an error.
pub fn discover_levels(prefix: &str, selector: &LevelSelector) -> anyhow::Result<Vec<Level>> {
    let glob_hits = discover_via_glob(prefix, selector)?;
    if !glob_hits.is_empty() {
        return Ok(glob_hits);
    }
    if let Some(meta_hits) = discover_via_pinto_json(prefix, selector)? {
        if !meta_hits.is_empty() {
            return Ok(meta_hits);
        }
    }
    anyhow::bail!(
        "no propensity parquets and no usable {prefix}.pinto.json matched prefix {prefix} \
         with selector {selector:?}"
    )
}

fn discover_via_glob(prefix: &str, selector: &LevelSelector) -> anyhow::Result<Vec<Level>> {
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
            propensity: PathBuf::from(format!("{prefix}{infix}.propensity.parquet")),
            link_community: Some(PathBuf::from(format!(
                "{prefix}{infix}.link_community.parquet"
            ))),
            gene_community: Some(PathBuf::from(format!(
                "{prefix}{infix}.gene_community.parquet"
            ))),
        });
    }
    out.sort_by_key(|l| l.sort_key);
    Ok(out)
}

fn discover_via_pinto_json(
    prefix: &str,
    selector: &LevelSelector,
) -> anyhow::Result<Option<Vec<Level>>> {
    let json_path = PathBuf::from(format!("{prefix}.pinto.json"));
    if !json_path.exists() {
        return Ok(None);
    }
    let meta = crate::util::metadata::PintoMetadata::read(&json_path)?;
    let levels = match meta.levels.as_ref() {
        Some(v) if !v.is_empty() => v,
        _ => return Ok(Some(Vec::new())),
    };
    let mut out = Vec::with_capacity(levels.len());
    for li in levels {
        if !selector.accepts(&li.tag) {
            continue;
        }
        out.push(Level {
            tag: li.tag.clone(),
            sort_key: li.level_index as i32,
            propensity: PathBuf::from(&li.propensity),
            link_community: li.link_community.as_ref().map(PathBuf::from),
            gene_community: li.gene_community.as_ref().map(PathBuf::from),
        });
    }
    out.sort_by_key(|l| l.sort_key);
    Ok(Some(out))
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
