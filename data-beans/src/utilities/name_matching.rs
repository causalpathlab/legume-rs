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
