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
use legume_numeric::matrix::utils::cosine;
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

/// Score every kept vocabulary word occurring in a document: align each
/// occurrence's byte span to the model tokens overlapping it, average
/// those token vectors, and take the cosine with the pooled vector.
/// Occurrences with no overlapping token (truncated away) are skipped.
/// Both lists are in text order, so one cursor walks the tokens once.
pub fn score_doc(
    pooled: &[f32],
    tokens: &[TokenVec],
    occurrences: &[Occurrence],
    vocab: &Vocabulary,
) -> Vec<WordScore> {
    let h = pooled.len();
    let mut per_word: FxHashMap<usize, (Vec<f32>, u32)> = FxHashMap::default();
    let mut first = 0usize; // first token that may still overlap
    for o in occurrences {
        while first < tokens.len() && tokens[first].end <= o.start {
            first += 1;
        }
        let Some(&w) = vocab.index.get(&o.word) else {
            continue;
        };
        let mut acc = vec![0f32; h];
        let mut n = 0usize;
        for t in &tokens[first..] {
            if t.start >= o.end {
                break;
            }
            if t.end > o.start {
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
            let cos = cosine(pooled, &vec);
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

/// Rows minus `center`, each L2-normalised (the anisotropy correction),
/// as a device tensor: one upload, then broadcast ops.
pub fn centred_unit(rows: &[Vec<f32>], center: &[f32], dev: &Device) -> Result<Tensor> {
    let n = rows.len();
    let h = center.len();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let x = Tensor::from_vec(flat, (n, h), dev)?;
    let c = Tensor::from_slice(center, (1, h), dev)?;
    let x = x.broadcast_sub(&c)?;
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12, f64::MAX)?;
    Ok(x.broadcast_div(&norm)?)
}

/// Mean of host rows.
pub fn mean_of(rows: &[Vec<f32>], h: usize) -> Vec<f32> {
    let n = rows.len();
    if n == 0 {
        return vec![0.0; h];
    }
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (n, h), &Device::Cpu)
        .and_then(|t| t.mean(0))
        .and_then(|m| m.to_vec1::<f32>())
        .unwrap_or_else(|_| vec![0.0; h])
}

/// Visit the cosine rows of `query × keysᵀ` (unit rows each) in row
/// blocks of roughly four million scores, calling `visit(i, row)` for every
/// query row `i` — the one blocked matmul the rankings below share.
fn visit_cosine_rows(
    query: &Tensor,
    keys_t: &Tensor,
    mut visit: impl FnMut(usize, Vec<f32>) -> Result<()>,
) -> Result<()> {
    let n = query.dim(0)?;
    let m = keys_t.dim(1)?;
    let block = ((1usize << 22) / m.max(1)).clamp(1, n.max(1));
    let mut start = 0usize;
    while start < n {
        let end = (start + block).min(n);
        let sims = query
            .narrow(0, start, end - start)?
            .matmul(keys_t)?
            .to_vec2::<f32>()?;
        for (r, row) in sims.into_iter().enumerate() {
            visit(start + r, row)?;
        }
        start = end;
    }
    Ok(())
}

/// The `k` largest `(index, score)` of a row, descending: a partial
/// selection, so a row of `m` scores costs `O(m + k log k)`, not a sort.
fn top_k(scores: impl Iterator<Item = (usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<(usize, f32)> = scores.collect();
    let desc = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    if k < idx.len() {
        idx.select_nth_unstable_by(k, desc);
        idx.truncate(k);
    }
    idx.sort_by(desc);
    idx
}

/// Top-`k` cosines of every query row against the keys (`[n, h]` unit
/// rows each); `exclude_self` masks `query[i] == key[i]`. Returns per
/// query `(key index, cosine)` descending.
pub fn top_k_cosine(
    query: &Tensor,
    keys: &Tensor,
    k: usize,
    exclude_self: bool,
) -> Result<Vec<Vec<(usize, f32)>>> {
    let n = query.dim(0)?;
    let m = keys.dim(0)?;
    let k = k.min(if exclude_self { m.saturating_sub(1) } else { m });
    let keys_t = keys.t()?.contiguous()?;
    let mut out = Vec::with_capacity(n);
    visit_cosine_rows(query, &keys_t, |i, row| {
        out.push(top_k(
            row.into_iter()
                .enumerate()
                .filter(|(j, _)| !(exclude_self && *j == i)),
            k,
        ));
        Ok(())
    })?;
    Ok(out)
}

/// Cross-domain nearest keys by CSLS (Conneau et al.): `2 cos(x, y) −
/// r(x) − r(y)`, where `r` is the mean cosine to the `hub_k` nearest rows
/// of the other side, which pulls hubs back. Returns per query the top
/// `k` keys `(index, csls)`. Two passes over the similarity blocks: the
/// first collects both hub means (row tops for `r(x)`, a running per-key
/// top list for `r(y)`), the second ranks.
pub fn csls_top_k(
    query: &Tensor,
    keys: &Tensor,
    k: usize,
    hub_k: usize,
) -> Result<Vec<Vec<(usize, f32)>>> {
    let n = query.dim(0)?;
    let m = keys.dim(0)?;
    let hub_k = hub_k.max(1).min(m).min(n);
    let k = k.min(m);
    let keys_t = keys.t()?.contiguous()?;
    let mut r_x: Vec<f32> = Vec::with_capacity(n);
    // Per key, its `hub_k` best cosines over the queries so far, ascending
    // so the weakest sits at index 0.
    let mut key_best: Vec<Vec<f32>> = vec![Vec::with_capacity(hub_k + 1); m];
    visit_cosine_rows(query, &keys_t, |_, row| {
        let best = top_k(row.iter().copied().enumerate(), hub_k);
        r_x.push(best.iter().map(|(_, c)| c).sum::<f32>() / best.len().max(1) as f32);
        for (j, c) in row.into_iter().enumerate() {
            let b = &mut key_best[j];
            if b.len() < hub_k {
                b.push(c);
            } else if c > b[0] {
                b[0] = c;
            } else {
                continue;
            }
            b.sort_by(|a, x| a.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        }
        Ok(())
    })?;
    let r_y: Vec<f32> = key_best
        .iter()
        .map(|b| b.iter().sum::<f32>() / b.len().max(1) as f32)
        .collect();
    let mut out = Vec::with_capacity(n);
    visit_cosine_rows(query, &keys_t, |i, row| {
        out.push(top_k(
            row.into_iter()
                .enumerate()
                .map(|(j, c)| (j, 2.0 * c - r_x[i] - r_y[j])),
            k,
        ));
        Ok(())
    })?;
    Ok(out)
}

#[cfg(test)]
#[path = "edges_tests.rs"]
mod edges_tests;
