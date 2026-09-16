//! From encodings to edges. A feature's words are the vocabulary words in
//! its own text, each weighted by how much the encoder ties that word, in
//! this context, to the whole description (cosine between the document's
//! pooled vector and the mean vector of the word's occurrences) times its
//! TF-IDF. The encoder sorts out the gene↔word relationship; nothing is
//! predicted. Two optional expansions use the same vectors: the nearest
//! vocabulary words a feature's text does not contain (CSLS, to counter
//! hubness) and feature–feature text similarity.

use crate::encoder::TokenVec;
use crate::vocab::{Occurrence, Vocabulary};
use anyhow::Result;
use candle_core::{Device, Tensor};
use rustc_hash::FxHashMap;
use std::io::Write;

/// One word of one document after scoring.
#[derive(Clone, Debug, PartialEq)]
pub struct WordScore {
    pub word: usize,
    /// Mean cosine between the pooled document and the word's occurrences.
    pub cos: f32,
    pub count: u32,
    /// The mean occurrence vector (for the word's global vector).
    pub vec: Vec<f32>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt().max(1e-12)
}

/// Score every kept vocabulary word occurring in a document: align each
/// occurrence's byte span to the model tokens overlapping it, average
/// those token vectors, and take the cosine with the pooled vector.
/// Occurrences with no overlapping token (truncated away) are skipped.
pub fn score_doc(
    pooled: &[f32],
    tokens: &[TokenVec],
    occurrences: &[Occurrence],
    vocab: &Vocabulary,
) -> Vec<WordScore> {
    let h = pooled.len();
    let pn = norm(pooled);
    let mut per_word: FxHashMap<usize, (Vec<f32>, u32)> = FxHashMap::default();
    for o in occurrences {
        let Some(&w) = vocab.index.get(&o.word) else {
            continue;
        };
        let mut acc = vec![0f32; h];
        let mut n = 0usize;
        for t in tokens {
            if t.start < o.end && t.end > o.start {
                for (a, v) in acc.iter_mut().zip(&t.vec) {
                    *a += v;
                }
                n += 1;
            }
        }
        if n == 0 {
            continue;
        }
        for a in &mut acc {
            *a /= n as f32;
        }
        let e = per_word.entry(w).or_insert_with(|| (vec![0f32; h], 0));
        for (s, a) in e.0.iter_mut().zip(&acc) {
            *s += a;
        }
        e.1 += 1;
    }
    let mut out: Vec<WordScore> = per_word
        .into_iter()
        .map(|(word, (sum, count))| {
            let vec: Vec<f32> = sum.iter().map(|s| s / count as f32).collect();
            let cos = dot(pooled, &vec) / (pn * norm(&vec));
            WordScore {
                word,
                cos,
                count,
                vec,
            }
        })
        .collect();
    out.sort_by_key(|s| s.word);
    out
}

/// `weight = max(cos, 0) · (1 + ln count) · idf`, the top `k` per document.
pub fn feature_word_weights(
    scores: &[WordScore],
    vocab: &Vocabulary,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut w: Vec<(usize, f32)> = scores
        .iter()
        .map(|s| {
            let tf = 1.0 + (s.count as f64).ln();
            let weight = f64::from(s.cos.max(0.0)) * tf * vocab.idf(s.word);
            (s.word, weight as f32)
        })
        .filter(|(_, w)| *w > 0.0)
        .collect();
    w.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    w.truncate(k);
    w
}

/// Write typed edges `lhs_type lhs rhs_type rhs weight`.
pub fn write_typed_edges<'a, W: Write>(
    w: &mut W,
    rows: impl Iterator<Item = (&'a str, &'a str, &'a str, &'a str, f32)>,
) -> Result<usize> {
    let mut n = 0usize;
    for (lt, l, rt, r, weight) in rows {
        writeln!(w, "{lt}\t{l}\t{rt}\t{r}\t{weight:.4}")?;
        n += 1;
    }
    Ok(n)
}

/// Row-wise L2 normalisation of a host matrix `[n, h]` after subtracting
/// `center` (the anisotropy correction), returned as a device tensor.
pub fn centred_unit(rows: &[Vec<f32>], center: &[f32], dev: &Device) -> Result<Tensor> {
    let n = rows.len();
    let h = center.len();
    let mut flat = Vec::with_capacity(n * h);
    for r in rows {
        let c: Vec<f32> = r.iter().zip(center).map(|(x, m)| x - m).collect();
        let nn = norm(&c);
        flat.extend(c.iter().map(|x| x / nn));
    }
    Ok(Tensor::from_vec(flat, (n, h), dev)?)
}

/// Mean of host rows.
pub fn mean_of(rows: &[Vec<f32>], h: usize) -> Vec<f32> {
    let mut m = vec![0f32; h];
    for r in rows {
        for (a, v) in m.iter_mut().zip(r) {
            *a += v;
        }
    }
    let n = rows.len().max(1) as f32;
    m.iter_mut().for_each(|a| *a /= n);
    m
}

/// Top-`k` cosines of every query row against the keys (`[n, h]` unit
/// rows each), in row blocks; `exclude_self` masks `query[i] == key[i]`.
/// Returns per query `(key index, cosine)` descending.
pub fn top_k_cosine(
    query: &Tensor,
    keys: &Tensor,
    k: usize,
    exclude_self: bool,
) -> Result<Vec<Vec<(usize, f32)>>> {
    let n = query.dim(0)?;
    let m = keys.dim(0)?;
    let k = k.min(if exclude_self { m.saturating_sub(1) } else { m });
    let block = (1usize << 22) / m.max(1); // ~4M scores per block
    let block = block.clamp(1, n.max(1));
    let kt = keys.t()?.contiguous()?;
    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let end = (start + block).min(n);
        let sims = query
            .narrow(0, start, end - start)?
            .matmul(&kt)?
            .to_vec2::<f32>()?;
        for (r, row) in sims.into_iter().enumerate() {
            let i = start + r;
            let mut idx: Vec<(usize, f32)> = row
                .into_iter()
                .enumerate()
                .filter(|(j, _)| !(exclude_self && *j == i))
                .collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(k);
            out.push(idx);
        }
        start = end;
    }
    Ok(out)
}

/// Cross-domain nearest keys by CSLS (Conneau et al.): `2 cos(x, y) −
/// r(x) − r(y)`, where `r` is the mean cosine to the `hub_k` nearest rows
/// of the other side, which pulls hubs back. Returns per query the top
/// `k` keys `(index, csls)`.
pub fn csls_top_k(
    query: &Tensor,
    keys: &Tensor,
    k: usize,
    hub_k: usize,
) -> Result<Vec<Vec<(usize, f32)>>> {
    let n = query.dim(0)?;
    let m = keys.dim(0)?;
    let hub_k = hub_k.max(1).min(m).min(n);
    let r_x: Vec<f32> = top_k_cosine(query, keys, hub_k, false)?
        .iter()
        .map(|v| v.iter().map(|(_, c)| c).sum::<f32>() / v.len().max(1) as f32)
        .collect();
    let r_y: Vec<f32> = top_k_cosine(keys, query, hub_k, false)?
        .iter()
        .map(|v| v.iter().map(|(_, c)| c).sum::<f32>() / v.len().max(1) as f32)
        .collect();
    let k = k.min(m);
    let block = ((1usize << 22) / m.max(1)).clamp(1, n.max(1));
    let kt = keys.t()?.contiguous()?;
    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let end = (start + block).min(n);
        let sims = query
            .narrow(0, start, end - start)?
            .matmul(&kt)?
            .to_vec2::<f32>()?;
        for (r, row) in sims.into_iter().enumerate() {
            let i = start + r;
            let mut idx: Vec<(usize, f32)> = row
                .into_iter()
                .enumerate()
                .map(|(j, c)| (j, 2.0 * c - r_x[i] - r_y[j]))
                .collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(k);
            out.push(idx);
        }
        start = end;
    }
    Ok(out)
}

#[cfg(test)]
#[path = "edges_tests.rs"]
mod edges_tests;
