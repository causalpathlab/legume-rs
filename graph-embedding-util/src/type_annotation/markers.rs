use super::*;
use log::warn;

/// Per-type `(feature_index, weight)` marker lists — one inner `Vec` per type.
pub(super) type MarkerSets = Vec<Vec<(u32, f32)>>;

/// Parse a marker TSV and match its genes to `gene_names`, returning the
/// sorted type vocabulary and per-type `(gene_index, weight)` lists.
///
/// Matching is exact-first (lowercased full name or its last `_`-segment
/// symbol), falling back to the shared `flexible_name_match`. Weights are IDF
/// — `ln(C / df_gene)` — unless `use_idf` is false (then unit), so markers
/// shared across many types are down-weighted (a ubiquitous gene → IDF 0,
/// dropped so it can't anchor a type).
///
/// **A type that ends up with no markers is dropped, not kept.** Its signature would
/// be the zero vector, which is not merely uninformative but an outright **magnet**
/// in the nearest-centroid assignment — see `term_ora::assign_nearest` for the
/// geometry and what it cost on a real run. Dropping the type here also keeps
/// the reported counts honest: a type we cannot score should not appear at all.
///
/// **A type that keeps only a sliver of its panel is the quieter, more dangerous case**,
/// and [`report_panel_coverage`] is what surfaces it. The embedding is trained on a
/// feature axis narrowed by the HVG cut (`--n-hvg`) and the feature-null FDR, and a marker
/// off that axis is not *projected*, it is simply **absent** — so it silently leaves the
/// panel. A type that entered with 20 markers and scores on 1 still produces a confident-
/// looking call, indistinguishable in the output from one that kept all 20. `min_coverage`
/// is the floor below which that stops being a warning and becomes an error.
pub(super) fn parse_and_match_markers(
    markers_path: &str,
    gene_names: &[Box<str>],
    use_idf: bool,
    min_coverage: f32,
) -> Result<(Vec<Box<str>>, MarkerSets)> {
    let pairs = read_marker_tsv(markers_path)?;

    // Type vocabulary (sorted, stable).
    let mut type_names: Vec<Box<str>> = pairs.iter().map(|(_, t)| t.clone()).collect();
    type_names.sort();
    type_names.dedup();
    let type_idx: FxHashMap<&str, usize> = type_names
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_ref(), i))
        .collect();

    // Shared three-tier gene matcher (exact → symbol → flexible fallback).
    let index = GeneIndex::build(gene_names);

    let c = type_names.len();
    let mut membership: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); c];
    // Distinct genes the panel *asked* for, per type — the denominator of the coverage
    // report below. Kept separate from `membership` (the ones we could find) because the
    // gap between the two is the whole point.
    let mut requested: Vec<FxHashSet<&str>> = vec![FxHashSet::default(); c];
    // Memoize matches by gene string: the same gene recurs across types, and
    // the flexible fallback is an O(genes) scan — resolve each distinct gene
    // at most once.
    let mut match_cache: FxHashMap<&str, Option<usize>> = FxHashMap::default();
    let mut unmatched = 0usize;
    for (gene, ct) in &pairs {
        let Some(&ti) = type_idx.get(ct.as_ref()) else {
            continue;
        };
        requested[ti].insert(gene.as_ref());
        match *match_cache
            .entry(gene.as_ref())
            .or_insert_with(|| index.match_gene(gene))
        {
            Some(gi) => {
                membership[ti].insert(gi as u32);
            }
            None => unmatched += 1,
        }
    }
    if unmatched > 0 {
        info!("{unmatched} marker pairs had no matching gene (dropped)");
    }
    report_panel_coverage(&type_names, &requested, &membership, min_coverage)?;

    // IDF: df_gene = #types containing the gene.
    let mut df: FxHashMap<u32, usize> = FxHashMap::default();
    for genes in &membership {
        for &gi in genes {
            *df.entry(gi).or_insert(0) += 1;
        }
    }
    let use_idf = use_idf && c >= 2;
    let type_markers: Vec<Vec<(u32, f32)>> = membership
        .iter()
        .map(|genes| {
            genes
                .iter()
                .filter_map(|&gi| {
                    if use_idf {
                        let w = idf_weight(c, df.get(&gi).copied().unwrap_or(1));
                        (w > 0.0).then_some((gi, w)) // drop ubiquitous (IDF 0)
                    } else {
                        Some((gi, 1.0))
                    }
                })
                .collect()
        })
        .collect();

    // Drop types left with no markers — either none of their genes are in the
    // embedding, or IDF zeroed every one. See the doc comment: an empty type's
    // zero signature out-competes every real one for any cell whose best cosine is
    // below 0.5, so keeping it silently mislabels the majority of the data.
    let (kept_names, kept_markers): (Vec<Box<str>>, MarkerSets) = type_names
        .iter()
        .zip(type_markers)
        .filter(|(_, m)| !m.is_empty())
        .map(|(n, m)| (n.clone(), m))
        .unzip();
    if kept_names.len() < c {
        let dropped: Vec<&str> = type_names
            .iter()
            .filter(|n| !kept_names.contains(n))
            .map(std::convert::AsRef::as_ref)
            .collect();
        warn!(
            "{} of {c} marker type(s) matched no gene in the embedding and were DROPPED: {}. \
             Their cells will be assigned among the remaining types — widen the feature axis \
             (e.g. lower `--n-hvg`, or rely on the held-out feature projection) if you need them.",
            dropped.len(),
            dropped.join(", ")
        );
    }
    anyhow::ensure!(
        !kept_names.is_empty(),
        "no marker type matched any gene in the embedding — check the marker file's gene \
         naming against the dictionary's row names"
    );

    Ok((kept_names, kept_markers))
}

/// Coverage below which a panel is reported as degraded rather than merely noted. Shared with
/// `auxiliary-data`'s gene-set reconciliation, which asks the same question of GAF/GMT term
/// sets — a marker panel is just another term→genes map, and a thin overlap means the same
/// thing in both.
use auxiliary_data::gene_sets::COVERAGE_WARN_FRAC as WARN_COVERAGE;

/// How many under-covered types to name before truncating the warning.
const MAX_LISTED: usize = 10;

/// Report what fraction of the marker panel actually reached the embedding, overall and per
/// type — and fail when it falls under `min_coverage`.
///
/// This is the guard on a failure mode that otherwise leaves no trace. Marker calls are only
/// as good as the genes they are computed on, but the embedding's feature axis is narrowed
/// long before the panel is read (`--n-hvg`, then the feature-null FDR), and a marker that
/// falls off that axis just quietly stops contributing. The result is a call that *looks*
/// identical to a well-supported one: same shape of output, same confidence, computed on a
/// handful of surviving genes. Without this, the only signal is a count of dropped pairs,
/// which says nothing about *which* types were gutted.
///
/// `min_coverage == 0` (the default) never errors — it reports and warns, so existing
/// pipelines keep running while the degradation stops being invisible. Set it to make a
/// thin panel fatal.
fn report_panel_coverage(
    type_names: &[Box<str>],
    requested: &[FxHashSet<&str>],
    membership: &[FxHashSet<u32>],
    min_coverage: f32,
) -> Result<()> {
    let (n_req, n_hit): (usize, usize) = (
        requested.iter().map(FxHashSet::len).sum(),
        membership.iter().map(FxHashSet::len).sum(),
    );
    if n_req == 0 {
        return Ok(());
    }
    let overall = n_hit as f32 / n_req as f32;

    // Per-type coverage, worst first — the overall number can look healthy while one type
    // has been reduced to a single gene.
    let mut thin: Vec<(f32, &str, usize, usize)> = (0..type_names.len())
        .filter(|&i| !requested[i].is_empty())
        .map(|i| {
            let (hit, req) = (membership[i].len(), requested[i].len());
            (hit as f32 / req as f32, type_names[i].as_ref(), hit, req)
        })
        .filter(|&(cov, ..)| cov < WARN_COVERAGE)
        .collect();
    thin.sort_by(|a, b| a.0.total_cmp(&b.0));

    info!(
        "marker panel: {n_hit}/{n_req} genes ({:.0}%) are on the embedding's feature axis, \
         across {} type(s)",
        overall * 100.0,
        type_names.len()
    );

    if !thin.is_empty() {
        let listed: Vec<String> = thin
            .iter()
            .take(MAX_LISTED)
            .map(|(cov, name, hit, req)| format!("{name} {hit}/{req} ({:.0}%)", cov * 100.0))
            .collect();
        // Name `--must-train-features`, not `faba gem --markers`: this guard fires in senna and
        // pinto too, and they have the former but not the latter. A warning whose remedy is a
        // flag your binary does not have is a warning you learn to ignore.
        warn!(
            "{} marker type(s) kept under {:.0}% of their panel — their calls rest on the few \
             genes that survived the feature axis, and will still look confident: {}{}. \
             Widen the axis (raise `--n-hvg`) or force the panel into training with \
             `--must-train-features <the marker file>` (in faba, `gem --markers` does this).",
            thin.len(),
            WARN_COVERAGE * 100.0,
            listed.join("; "),
            if thin.len() > MAX_LISTED {
                format!(", … and {} more", thin.len() - MAX_LISTED)
            } else {
                String::new()
            }
        );
    }

    anyhow::ensure!(
        overall >= min_coverage,
        "marker panel coverage {:.0}% is below the required {:.0}% ({n_hit}/{n_req} genes on \
         the embedding's feature axis). The panel the calls would be built on is mostly \
         missing. Widen the feature axis (raise `--n-hvg`) or force the panel into training \
         with `--must-train-features <the marker file>` (in faba, `gem --markers` does this).",
        overall * 100.0,
        min_coverage * 100.0
    );
    Ok(())
}

/// Parse a marker TSV/CSV into `(gene, celltype)` pairs via the shared,
/// gz-aware `read_lines_of_words_delim` (tab/comma — matching senna's marker
/// reader). Takes the first two tokens per line, skips a `gene`/`symbol`
/// header and `#` comments, and maps spaces in cell-type names → `_`.
pub(super) fn read_marker_tsv(path: &str) -> Result<Vec<(Box<str>, Box<str>)>> {
    let ReadLinesOut { lines, .. } = read_lines_of_words_delim(path, &['\t', ','][..], -1)
        .with_context(|| format!("reading markers {path}"))?;
    let out: Vec<(Box<str>, Box<str>)> = lines
        .into_iter()
        .filter_map(|words| {
            let gene = words.first()?.trim();
            let ct = words.get(1)?.trim();
            let gl = gene.to_lowercase();
            if gene.is_empty()
                || gene.starts_with('#')
                || ct.is_empty()
                || gl == "gene"
                || gl == "symbol"
            {
                return None;
            }
            Some((Box::from(gene), Box::from(ct.replace(' ', "_"))))
        })
        .collect();
    anyhow::ensure!(!out.is_empty(), "no marker pairs parsed from {path}");
    Ok(out)
}

/// Union of every type's marker gene indices (sorted, unique, in range) — the
/// universe the label-shuffle null samples from.
pub(super) fn marker_gene_pool(type_markers: &[Vec<(u32, f32)>], n_features: usize) -> Vec<u32> {
    let mut set: FxHashSet<u32> = FxHashSet::default();
    for markers in type_markers {
        for &(g, _) in markers {
            if (g as usize) < n_features {
                set.insert(g);
            }
        }
    }
    let mut pool: Vec<u32> = set.into_iter().collect();
    pool.sort_unstable();
    pool
}
