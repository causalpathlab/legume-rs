use rustc_hash::FxHashMap as HashMap;

/// Make duplicate names unique by appending `-1`, `-2`, etc. to repeated entries.
/// Similar to scanpy's `var_names_make_unique()`.
pub fn make_names_unique(names: &mut [Box<str>]) -> usize {
    let mut counts: HashMap<Box<str>, usize> = HashMap::default();
    let mut num_duped = 0usize;
    for name in names.iter_mut() {
        if let Some(count) = counts.get_mut(name.as_ref()) {
            if *count == 1 {
                num_duped += 1;
            }
            *name = format!("{}-{}", name, count).into_boxed_str();
            *count += 1;
        } else {
            counts.insert(name.clone(), 1);
        }
    }
    if num_duped > 0 {
        log::warn!(
            "{} names had duplicates and were made unique with -N suffixes",
            num_duped
        );
    }
    num_duped
}

/// Combine feature IDs and names into composite `id_name` strings.
/// If a name is empty, the ID is used as-is.
pub fn compose_id_name(ids: Vec<Box<str>>, names: Vec<Box<str>>) -> Vec<Box<str>> {
    ids.into_iter()
        .zip(names)
        .map(|(id, name)| {
            if !name.is_empty() {
                format!("{}_{}", id, name).into_boxed_str()
            } else {
                id
            }
        })
        .collect()
}

/// Return indices of rows whose type passes select/remove filtering.
/// - `select`: if empty, all rows pass; otherwise row type must contain the pattern (case-insensitive)
/// - `remove`: if empty, nothing excluded; otherwise row type must NOT contain the pattern (case-insensitive)
pub fn filter_row_indices_by_type(
    row_types: &[Box<str>],
    select: &str,
    remove: &str,
) -> Vec<usize> {
    let sel = select.to_ascii_lowercase();
    let rem = remove.to_ascii_lowercase();
    if sel.is_empty() && rem.is_empty() {
        return (0..row_types.len()).collect();
    }
    row_types
        .iter()
        .enumerate()
        .filter_map(|(i, x)| {
            let low = x.to_ascii_lowercase();
            let selected = sel.is_empty() || low.contains(&sel);
            let removed = !rem.is_empty() && low.contains(&rem);
            if selected && !removed {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Flexible gene name matching (case-insensitive, underscore-delimited)
/// Returns true if `query` matches `target` with these rules:
/// - Exact match (case-insensitive)
/// - Suffix match: target ends with `_query`
/// - Prefix match: target starts with `query_`
/// - Segment match: target contains `_query_`
///
/// Example: "CD8A" matches "ENSG00000153563_CD8A", "CD8A_variant1", "chr1_CD8A_isoform2"
#[allow(dead_code)]
pub fn flexible_name_match(query: &str, target: &str) -> bool {
    let q = query.to_lowercase();
    let t = target.to_lowercase();
    t == q
        || t.ends_with(&format!("_{}", q))
        || t.starts_with(&format!("{}_", q))
        || t.contains(&format!("_{}_", q))
}

/// Match names by substring queries and return matched indices and names
///
/// # Arguments
/// * `all_names` - All available names to search through
/// * `queries` - Substring queries to match against
/// * `entity_type` - Description of what's being matched (e.g., "column", "row") for error messages
///
/// # Returns
/// A tuple of (matched_indices, matched_names)
pub fn match_by_substring(
    all_names: &[Box<str>],
    queries: &[Box<str>],
    entity_type: &str,
) -> anyhow::Result<(Vec<usize>, Vec<Box<str>>)> {
    let mut matched_indices = Vec::new();

    for query in queries.iter() {
        for (idx, name) in all_names.iter().enumerate() {
            if name.contains(query.as_ref()) {
                matched_indices.push(idx);
            }
        }
    }

    if matched_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "No {} names matched the provided queries",
            entity_type
        ));
    }

    let matched_names: Vec<Box<str>> = matched_indices
        .iter()
        .map(|&i| all_names[i].clone())
        .collect();

    Ok((matched_indices, matched_names))
}
