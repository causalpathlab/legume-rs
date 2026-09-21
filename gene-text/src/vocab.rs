//! The word vocabulary: tokenise every document, drop stopwords and
//! filler, optionally stem, then a document-frequency QC that cuts both
//! tails by quantile — the rarest words (typos, accessions, one-offs) and
//! the most common ones that slipped past the stopword lists. The graph
//! engine has no frequency model of its own: a near-universal word would
//! become a hub node at the centroid and a false in-batch negative for
//! every gene it truly links to, and a singleton word gets one Adagrad
//! step per epoch and stays at its init. TF-IDF grades what is left.

use anyhow::{Context, Result};
use log::info;
use rust_stemmers::{Algorithm, Stemmer};
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::{BufRead, Write};
use unicode_segmentation::UnicodeSegmentation;

/// Biomedical filler the general stopword lists miss, one word per line
/// (`#` comments): words that appear in most descriptions and say nothing
/// about which gene this is. A data file, like `--extra-stopwords`, so
/// tuning it is a word-list edit rather than a code change.
pub const BUILTIN_FILLER: &str = include_str!("../data/filler_stopwords.txt");

/// The words of a stopword list: one per line, lowercased, `#` comments
/// and blank lines skipped.
fn stopword_lines(text: &str) -> impl Iterator<Item = Box<str>> + '_ {
    text.lines()
        .map(|l| l.trim().to_lowercase())
        .filter(|w| !w.is_empty() && !w.starts_with('#'))
        .map(Box::from)
}

/// Tokeniser settings.
#[derive(Clone, Debug)]
pub struct TokenizeOpts {
    pub min_chars: usize,
    pub stem: bool,
    pub stopwords: FxHashSet<Box<str>>,
}

impl TokenizeOpts {
    /// The NLTK English list plus the built-in filler, plus `extra` (one
    /// word per line, `#` comments).
    pub fn new(min_chars: usize, stem: bool, extra: Option<&str>) -> Result<Self> {
        let mut stopwords: FxHashSet<Box<str>> = stop_words::get(stop_words::LANGUAGE::English)
            .iter()
            .map(|w| Box::from(w.to_lowercase()))
            .collect();
        stopwords.extend(stopword_lines(BUILTIN_FILLER));
        if let Some(path) = extra {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("opening stopwords {path}"))?;
            let before = stopwords.len();
            stopwords.extend(stopword_lines(&text));
            info!("{path}: {} extra stopwords", stopwords.len() - before);
        }
        Ok(Self {
            min_chars,
            stem,
            stopwords,
        })
    }
}

/// One word occurrence: the vocabulary word and its byte span in the text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Occurrence {
    pub word: Box<str>,
    pub start: usize,
    pub end: usize,
}

/// Tokenise one text into vocabulary candidates with their byte spans
/// (spans are of the original surface form, so model tokens can be aligned
/// to them). Lowercased; no digits-only or short tokens; no stopwords;
/// stemmed when asked (the stopword test runs before stemming).
pub fn tokenize(text: &str, opts: &TokenizeOpts) -> Vec<Occurrence> {
    let stemmer = opts.stem.then(|| Stemmer::create(Algorithm::English));
    let mut out = Vec::new();
    for (start, w) in text.unicode_word_indices() {
        let lower = w.to_lowercase();
        // Length counts letters and digits (`tp53` is four), and a dotted
        // abbreviation — every alphabetic run a single letter, as in `e.g`
        // and `i.e` — is dropped whatever its length.
        let n_alnum = lower.chars().filter(|c| c.is_alphanumeric()).count();
        let abbreviation = lower.contains('.')
            && lower
                .split(|c: char| !c.is_alphabetic())
                .filter(|r| !r.is_empty())
                .all(|r| r.chars().count() == 1);
        if n_alnum < opts.min_chars || abbreviation || !lower.chars().any(char::is_alphabetic) {
            continue;
        }
        if opts.stopwords.contains(lower.as_str()) {
            continue;
        }
        let word: Box<str> = match &stemmer {
            Some(s) => s.stem(&lower).into_owned().into(),
            None => lower.into(),
        };
        if word.chars().count() < opts.min_chars {
            continue;
        }
        out.push(Occurrence {
            word,
            start,
            end: start + w.len(),
        });
    }
    out
}

/// The document-frequency QC settings.
#[derive(Clone, Debug)]
pub struct DfQcOpts {
    /// Drop the rarest words: those at or below the df found at this
    /// quantile of the vocabulary (0 = off).
    pub lower_quantile: f64,
    /// Drop the most common words: those at or above the df found at this
    /// quantile of the vocabulary (1 = off).
    pub upper_quantile: f64,
    /// Absolute floor on df, applied on top.
    pub min_df: usize,
    /// Absolute cap on df as a fraction of the documents (1 = off), on top.
    pub max_df_frac: f64,
}

/// Why a word was dropped, or kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Kept,
    Rare,
    Common,
}

impl Verdict {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Kept => "kept",
            Verdict::Rare => "rare",
            Verdict::Common => "common",
        }
    }
}

/// One vocabulary word with its document frequency and verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct VocabEntry {
    pub word: Box<str>,
    pub df: usize,
    pub verdict: Verdict,
}

/// The kept words of a vocabulary, their df, and the word → index map.
struct KeptWords {
    words: Vec<Box<str>>,
    df: Vec<usize>,
    index: FxHashMap<Box<str>, usize>,
}

/// The vocabulary after QC: kept words indexed, every candidate reported.
pub struct Vocabulary {
    pub entries: Vec<VocabEntry>,
    pub n_docs: usize,
    /// kept word → index into `kept`.
    pub index: FxHashMap<Box<str>, usize>,
    /// The kept words, in `entries` order.
    pub kept: Vec<Box<str>>,
    /// `df` of every kept word, parallel to `kept`.
    pub kept_df: Vec<usize>,
    pub lower_cut: Option<usize>,
    pub upper_cut: Option<usize>,
}

impl Vocabulary {
    #[must_use]
    pub fn idf(&self, word_idx: usize) -> f64 {
        let df = self.kept_df[word_idx] as f64;
        ((1.0 + self.n_docs as f64) / (1.0 + df)).ln()
    }

    /// The kept words and their df, in `entries` order, plus the word index.
    fn kept_of(entries: &[VocabEntry]) -> KeptWords {
        let kept: Vec<&VocabEntry> = entries
            .iter()
            .filter(|e| e.verdict == Verdict::Kept)
            .collect();
        let words: Vec<Box<str>> = kept.iter().map(|e| e.word.clone()).collect();
        let df: Vec<usize> = kept.iter().map(|e| e.df).collect();
        let index = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();
        KeptWords { words, df, index }
    }

    /// Build from per-document token lists: df counts each word once per
    /// document. The histogram, the cuts and the tail listings are logged.
    pub fn build(docs: &[Vec<Occurrence>], qc: &DfQcOpts) -> Self {
        let n_docs = docs.len();
        let mut df: FxHashMap<&str, usize> = FxHashMap::default();
        for d in docs {
            let mut seen: FxHashSet<&str> = FxHashSet::default();
            for o in d {
                if seen.insert(&o.word) {
                    *df.entry(&o.word).or_default() += 1;
                }
            }
        }
        let mut entries: Vec<VocabEntry> = df
            .into_iter()
            .map(|(w, df)| VocabEntry {
                word: w.into(),
                df,
                verdict: Verdict::Kept,
            })
            .collect();
        entries.sort_by(|a, b| a.df.cmp(&b.df).then_with(|| a.word.cmp(&b.word)));
        let v = entries.len();
        let (lower_cut, upper_cut) = quantile_cuts(&entries, qc);
        let max_df = if qc.max_df_frac < 1.0 {
            Some((qc.max_df_frac * n_docs as f64).floor() as usize)
        } else {
            None
        };
        for e in &mut entries {
            let rare = lower_cut.is_some_and(|c| e.df <= c) || e.df < qc.min_df;
            let common = upper_cut.is_some_and(|c| e.df >= c) || max_df.is_some_and(|m| e.df > m);
            e.verdict = if rare {
                Verdict::Rare
            } else if common {
                Verdict::Common
            } else {
                Verdict::Kept
            };
        }
        log_histogram(&entries, n_docs);
        let n_rare = entries
            .iter()
            .filter(|e| e.verdict == Verdict::Rare)
            .count();
        let n_common = entries
            .iter()
            .filter(|e| e.verdict == Verdict::Common)
            .count();
        info!(
            "vocabulary: {v} candidate words over {n_docs} documents; rare cut at df ≤ {} (quantile {}, min-df {}): {n_rare} dropped; common cut at df ≥ {} (quantile {}, max-df-frac {}): {n_common} dropped; {} kept",
            lower_cut.map_or("off".to_string(), |c| c.to_string()),
            qc.lower_quantile,
            qc.min_df,
            upper_cut.map_or("off".to_string(), |c| c.to_string()),
            qc.upper_quantile,
            qc.max_df_frac,
            v - n_rare - n_common
        );
        log_tails(&entries);
        let KeptWords {
            words: kept,
            df: kept_df,
            index,
        } = Self::kept_of(&entries);
        Self {
            entries,
            n_docs,
            index,
            kept,
            kept_df,
            lower_cut,
            upper_cut,
        }
    }

    /// `word <TAB> df <TAB> df_frac <TAB> kept <TAB> reason`, most common first.
    pub fn write_tsv(&self, path: &str) -> Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(w, "word\tdf\tdf_frac\tkept\treason")?;
        for e in self.entries.iter().rev() {
            writeln!(
                w,
                "{}\t{}\t{:.6}\t{}\t{}",
                e.word,
                e.df,
                e.df as f64 / self.n_docs.max(1) as f64,
                u8::from(e.verdict == Verdict::Kept),
                e.verdict.as_str()
            )?;
        }
        w.flush()?;
        Ok(())
    }

    /// Reload a tuned vocabulary: only the rows marked kept.
    pub fn read_tsv(path: &str, n_docs: usize) -> Result<Self> {
        let reader = legume_numeric::matrix::common_io::open_buf_reader(path)
            .with_context(|| format!("opening vocabulary {path}"))?;
        let mut entries = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if i == 0 && line.starts_with("word\t") {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 4 {
                continue;
            }
            let df: usize = f[1]
                .parse()
                .with_context(|| format!("{path}: df `{}`", f[1]))?;
            let kept = f[3].trim() == "1";
            entries.push(VocabEntry {
                word: f[0].into(),
                df,
                verdict: if kept {
                    Verdict::Kept
                } else if f.get(4).is_some_and(|r| r.trim() == "common") {
                    Verdict::Common
                } else {
                    Verdict::Rare
                },
            });
        }
        entries.sort_by(|a, b| a.df.cmp(&b.df).then_with(|| a.word.cmp(&b.word)));
        let KeptWords {
            words: kept,
            df: kept_df,
            index,
        } = Self::kept_of(&entries);
        info!(
            "{path}: {} kept words of {} listed",
            kept.len(),
            entries.len()
        );
        Ok(Self {
            entries,
            n_docs,
            index,
            kept,
            kept_df,
            lower_cut: None,
            upper_cut: None,
        })
    }
}

/// The df values at the two quantiles of the (ascending) vocabulary.
fn quantile_cuts(sorted: &[VocabEntry], qc: &DfQcOpts) -> (Option<usize>, Option<usize>) {
    let v = sorted.len();
    if v == 0 {
        return (None, None);
    }
    let at = |q: f64| -> usize {
        let i = ((q * v as f64).floor() as usize).min(v - 1);
        sorted[i].df
    };
    let lower = (qc.lower_quantile > 0.0).then(|| at(qc.lower_quantile));
    let upper = (qc.upper_quantile < 1.0).then(|| at(qc.upper_quantile));
    (lower, upper)
}

/// A text histogram of `log10 df` over the vocabulary, plus the deciles.
fn log_histogram(entries: &[VocabEntry], n_docs: usize) {
    if entries.is_empty() {
        return;
    }
    let max = (n_docs.max(1) as f64).log10().max(1e-9);
    let n_bins = 24usize;
    let mut bins = vec![0usize; n_bins];
    for e in entries {
        let x = (e.df as f64).log10() / max;
        let b = ((x * n_bins as f64).floor() as usize).min(n_bins - 1);
        bins[b] += 1;
    }
    let peak = *bins.iter().max().unwrap_or(&1) as f64;
    info!(
        "document-frequency histogram (log10 df, {} words):",
        entries.len()
    );
    for (b, &n) in bins.iter().enumerate() {
        let lo = 10f64.powf(b as f64 / n_bins as f64 * max);
        let bar = "#".repeat(((n as f64 / peak.max(1.0)) * 40.0).round() as usize);
        info!("  df ≥ {lo:>9.0} | {bar:<40} {n}");
    }
    let deciles: Vec<String> = (1..10)
        .map(|d| {
            let i = ((d as f64 / 10.0) * entries.len() as f64).floor() as usize;
            format!("{}", entries[i.min(entries.len() - 1)].df)
        })
        .collect();
    info!("  df deciles (10%..90%): {}", deciles.join(" "));
}

fn log_tails(entries: &[VocabEntry]) {
    let show = |label: &str, it: &mut dyn Iterator<Item = &VocabEntry>| {
        let words: Vec<String> = it
            .take(30)
            .map(|e| format!("{}({})", e.word, e.df))
            .collect();
        if !words.is_empty() {
            info!("  {label}: {}", words.join(" "));
        }
    };
    show(
        "rarest dropped",
        &mut entries.iter().rev().filter(|e| e.verdict == Verdict::Rare),
    );
    show(
        "rarest kept",
        &mut entries.iter().filter(|e| e.verdict == Verdict::Kept),
    );
    show(
        "commonest kept",
        &mut entries.iter().rev().filter(|e| e.verdict == Verdict::Kept),
    );
    show(
        "commonest dropped",
        &mut entries
            .iter()
            .rev()
            .filter(|e| e.verdict == Verdict::Common),
    );
}

#[cfg(test)]
#[path = "vocab_tests.rs"]
mod vocab_tests;
