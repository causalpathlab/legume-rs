use super::*;

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
pub(super) fn parse_and_match_markers(
    markers_path: &str,
    gene_names: &[Box<str>],
    use_idf: bool,
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
    // Memoize matches by gene string: the same gene recurs across types, and
    // the flexible fallback is an O(genes) scan — resolve each distinct gene
    // at most once.
    let mut match_cache: FxHashMap<&str, Option<usize>> = FxHashMap::default();
    let mut unmatched = 0usize;
    for (gene, ct) in &pairs {
        let Some(&ti) = type_idx.get(ct.as_ref()) else {
            continue;
        };
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

    Ok((type_names, type_markers))
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
