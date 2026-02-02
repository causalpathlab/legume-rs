//! Interactive marker gene augmentation for topic annotation
//!
//! This module provides an iterative human-in-the-loop workflow for
//! discovering and adding marker genes based on topic-celltype associations.

use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;
use matrix_util::common_io::open_buf_reader;
use std::io::{self, BufRead, Write};

use crate::embed_common::Mat;

/// Flexible gene name matching (case-insensitive, underscore-delimited)
/// Returns true if marker_gene matches dict_gene with these rules:
/// - Exact match (case-insensitive)
/// - Suffix match: dict_gene ends with "_marker_gene"
/// - Prefix match: dict_gene starts with "marker_gene_"
/// - Segment match: dict_gene contains "_marker_gene_"
///
/// Example: "CD8A" matches "ENSG00000153563_CD8A", "CD8A_variant1", "chr1_CD8A_isoform2"
pub fn flexible_gene_match(marker_gene: &str, dict_gene: &str) -> bool {
    let marker_lower = marker_gene.to_lowercase();
    let dict_lower = dict_gene.to_lowercase();

    // 1. Exact match
    if dict_lower == marker_lower {
        return true;
    }

    // 2. Suffix match: dict ends with "_marker"
    if dict_lower.ends_with(&format!("_{}", marker_lower)) {
        return true;
    }

    // 3. Prefix match: dict starts with "marker_"
    if dict_lower.starts_with(&format!("{}_", marker_lower)) {
        return true;
    }

    // 4. Segment match: dict contains "_marker_"
    if dict_lower.contains(&format!("_{}_", marker_lower)) {
        return true;
    }

    false
}

/// Fuzzy match for cell type names
/// Returns true if query matches target with these rules:
/// - Short strings (< 5 chars) require exact match to avoid false positives like "B" matching "Erythroblast"
/// - Otherwise: exact, substring, or common substring of 5+ chars (e.g., "erythro" in both)
fn fuzzy_match_ct(query: &str, target: &str) -> bool {
    let qchars: Vec<char> = query.chars().collect();
    let tchars: Vec<char> = target.chars().collect();
    if qchars.len() < 5 || tchars.len() < 5 {
        return query == target;
    }
    if query == target || target.contains(query) || query.contains(target) {
        return true;
    }
    // Check for common substring of 5+ chars (character-based for UTF-8 safety)
    let (shorter, longer) = if qchars.len() <= tchars.len() { (&qchars, target) } else { (&tchars, query) };
    (0..=shorter.len().saturating_sub(5)).any(|i| {
        let sub: String = shorter[i..i + 5].iter().collect();
        longer.contains(&sub)
    })
}

/// Result of interactive marker augmentation session
#[derive(Debug)]
pub struct AugmentationResult {
    /// New gene → celltype associations discovered
    pub new_markers: Vec<(Box<str>, Box<str>)>,
    /// Whether user chose to proceed (true) or cancel (false)
    pub proceed: bool,
}

/// Candidate gene for marker augmentation
#[derive(Debug, Clone)]
pub struct CandidateGene {
    pub gene_name: Box<str>,
    pub weight: f32,
}

/// A topic's match to a cell type with candidate genes
#[derive(Debug, Clone)]
pub struct TopicMatch {
    pub topic_name: Box<str>,
    pub pip: f32,
    pub candidates: Vec<CandidateGene>,
}

/// Cell type with its matching topics and candidate genes
#[derive(Debug)]
pub struct CelltypeCandidates {
    pub celltype_name: Box<str>,
    /// Topics that match this cell type, sorted by PIP descending
    pub topic_matches: Vec<TopicMatch>,
}

/// Find candidate marker genes grouped by cell type
///
/// For each cell type, finds topics that match it and candidate genes that:
/// 1. Have high weight in those topics' dictionaries
/// 2. Are NOT already markers for that celltype
pub fn find_candidate_markers(
    pip_at: &Mat,                          // annotation × topic
    dict_gt: &Mat,                         // gene × topic (log-prob or weights)
    membership_ga: &Mat,                   // gene × annotation
    gene_names: &[Box<str>],
    topic_names: &[Box<str>],
    annot_names: &[Box<str>],
    min_pip: f32,
    top_k_genes: usize,
    top_k_topics: usize,
) -> Vec<CelltypeCandidates> {
    let n_topics = pip_at.ncols();
    let n_annots = pip_at.nrows();
    let n_genes = dict_gt.nrows();

    let mut results = Vec::new();

    for a in 0..n_annots {
        // Find topics that match this cell type, sorted by PIP
        let mut topic_pips: Vec<(usize, f32)> = (0..n_topics)
            .map(|t| (t, pip_at[(a, t)]))
            .filter(|&(_, pip)| pip >= min_pip)
            .collect();
        topic_pips.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        topic_pips.truncate(top_k_topics);

        if topic_pips.is_empty() {
            continue;
        }

        // Get genes already assigned to this celltype
        let existing_markers: HashSet<usize> = (0..n_genes)
            .filter(|&g| membership_ga[(g, a)] > 0.0)
            .collect();

        // Build topic matches with candidate genes
        let mut topic_matches = Vec::new();
        for (t, pip) in topic_pips {
            // Get gene weights for this topic, filter out existing markers
            let mut gene_weights: Vec<(usize, f32)> = (0..n_genes)
                .filter(|g| !existing_markers.contains(g))
                .map(|g| {
                    let log_w = dict_gt[(g, t)];
                    let w = if log_w < 0.0 { log_w.exp() } else { log_w };
                    (g, w)
                })
                .collect();
            gene_weights.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

            let candidates: Vec<CandidateGene> = gene_weights
                .into_iter()
                .take(top_k_genes)
                .map(|(g, w)| CandidateGene {
                    gene_name: gene_names[g].clone(),
                    weight: w,
                })
                .collect();

            if !candidates.is_empty() {
                topic_matches.push(TopicMatch {
                    topic_name: topic_names[t].clone(),
                    pip,
                    candidates,
                });
            }
        }

        if !topic_matches.is_empty() {
            results.push(CelltypeCandidates {
                celltype_name: annot_names[a].clone(),
                topic_matches,
            });
        }
    }

    // Sort by max PIP descending
    results.sort_by(|a, b| {
        let max_a = a.topic_matches.iter().map(|m| m.pip).fold(0.0f32, f32::max);
        let max_b = b.topic_matches.iter().map(|m| m.pip).fold(0.0f32, f32::max);
        max_b.partial_cmp(&max_a).unwrap()
    });

    results
}

/// Interactive session action
#[derive(Debug, Clone)]
pub enum InteractiveAction {
    /// Add selected genes (by index, 0-based)
    AddGenes(Vec<usize>),
    /// Skip this cell type
    Skip,
    /// Done with all cell types, proceed to next iteration or finish
    Done,
    /// Quit/cancel the entire operation
    Quit,
}

/// Display candidates for a cell type and prompt user for action
pub fn prompt_celltype_candidates(ct: &CelltypeCandidates) -> anyhow::Result<InteractiveAction> {
    eprintln!();
    eprintln!("Cell type: {}", ct.celltype_name);

    // Collect all candidates with global numbering
    let mut all_candidates: Vec<(&CandidateGene, &str)> = Vec::new();
    for tm in &ct.topic_matches {
        eprintln!("  {} (PIP: {:.3})", tm.topic_name, tm.pip);
        for cand in &tm.candidates {
            all_candidates.push((cand, tm.topic_name.as_ref()));
            eprintln!(
                "    {:>2}. {:20} {:.4}",
                all_candidates.len(),
                cand.gene_name,
                cand.weight
            );
        }
    }

    eprint!("  Add as {} markers? [1,2,.../a=all/s=skip/d=done/q=quit]: ", ct.celltype_name);
    io::stderr().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    let n_total = all_candidates.len();

    match input.as_str() {
        "" | "s" | "skip" => Ok(InteractiveAction::Skip),
        "a" | "all" => {
            let all_indices: Vec<usize> = (0..n_total).collect();
            Ok(InteractiveAction::AddGenes(all_indices))
        }
        "d" | "done" | "p" | "proceed" => Ok(InteractiveAction::Done),
        "q" | "quit" | "cancel" => Ok(InteractiveAction::Quit),
        _ => {
            // Parse comma-separated numbers
            let indices: Vec<usize> = input
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .filter(|&n| n > 0 && n <= n_total)
                .map(|n| n - 1)
                .collect();

            if indices.is_empty() {
                eprintln!("  (no valid selection, skipping)");
                Ok(InteractiveAction::Skip)
            } else {
                Ok(InteractiveAction::AddGenes(indices))
            }
        }
    }
}

/// Run one round of interactive marker augmentation
///
/// Returns the genes to add and whether to continue/proceed/quit
pub fn run_interactive_round(
    candidates: &[CelltypeCandidates],
    iteration: usize,
) -> anyhow::Result<AugmentationResult> {
    eprintln!();
    eprintln!("--- Marker augmentation (round {}) ---", iteration);

    if candidates.is_empty() {
        eprintln!("No candidate genes found above threshold.");
        return Ok(AugmentationResult {
            new_markers: Vec::new(),
            proceed: true,
        });
    }

    eprintln!("{} cell types with candidate markers", candidates.len());

    let mut new_markers: Vec<(Box<str>, Box<str>)> = Vec::new();

    for ct in candidates {
        // Collect all candidates for this cell type
        let all_candidates: Vec<&CandidateGene> = ct
            .topic_matches
            .iter()
            .flat_map(|tm| &tm.candidates)
            .collect();

        match prompt_celltype_candidates(ct)? {
            InteractiveAction::AddGenes(indices) => {
                for idx in indices {
                    let gene = all_candidates[idx];
                    eprintln!("    + {} -> {}", gene.gene_name, ct.celltype_name);
                    new_markers.push((gene.gene_name.clone(), ct.celltype_name.clone()));
                }
            }
            InteractiveAction::Skip => {}
            InteractiveAction::Done => {
                eprintln!("Added {} markers this round.", new_markers.len());
                return Ok(AugmentationResult {
                    new_markers,
                    proceed: true,
                });
            }
            InteractiveAction::Quit => {
                eprintln!("Cancelled.");
                return Ok(AugmentationResult {
                    new_markers: Vec::new(),
                    proceed: false,
                });
            }
        }
    }

    // Finished all topics
    eprintln!();
    eprintln!("Added {} markers this round.", new_markers.len());

    if new_markers.is_empty() {
        eprint!("No new markers. Proceed? [y/n]: ");
    } else {
        eprint!("Re-fit with new markers? [y/n]: ");
    }
    io::stderr().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let proceed = matches!(input.trim().to_lowercase().as_str(), "y" | "yes" | "");

    Ok(AugmentationResult {
        new_markers,
        proceed,
    })
}

/// Save augmented markers to file
pub fn save_augmented_markers(
    original_markers: &HashMap<Box<str>, Box<str>>,
    new_markers: &[(Box<str>, Box<str>)],
    output_path: &str,
) -> anyhow::Result<()> {
    use std::fs::File;
    use std::io::BufWriter;

    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Write original markers
    for (gene, celltype) in original_markers {
        writeln!(writer, "{}\t{}", gene, celltype)?;
    }

    // Write new markers
    for (gene, celltype) in new_markers {
        writeln!(writer, "{}\t{}", gene, celltype)?;
    }

    writer.flush()?;
    Ok(())
}

/// Update membership matrix with new markers
/// Uses flexible gene name matching (same as initial marker file)
pub fn augment_membership_matrix(
    membership_ga: &mut Mat,
    gene_names: &[Box<str>],
    annot_names: &[Box<str>],
    new_markers: &[(Box<str>, Box<str>)],
    weight: f32,
) {
    // Build celltype lookup (exact match for cell types)
    let annot_to_idx: HashMap<&str, usize> = annot_names
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_ref(), i))
        .collect();

    for (marker_gene, celltype) in new_markers {
        // Find matching cell type (exact match)
        let Some(&a_idx) = annot_to_idx.get(celltype.as_ref()) else {
            continue;
        };

        // Find all matching genes using flexible matching
        for (g_idx, dict_gene) in gene_names.iter().enumerate() {
            if flexible_gene_match(marker_gene, dict_gene) {
                membership_ga[(g_idx, a_idx)] = weight;
            }
        }
    }
}

/// Print summary at end of session
pub fn print_augmentation_summary(
    all_new_markers: &[(Box<str>, Box<str>)],
    iterations: usize,
) {
    eprintln!();
    eprintln!(
        "Augmentation complete: {} iterations, {} new markers",
        iterations,
        all_new_markers.len()
    );

    if !all_new_markers.is_empty() {
        // Group by celltype
        let mut by_celltype: HashMap<&str, Vec<&str>> = HashMap::default();
        for (gene, ct) in all_new_markers {
            by_celltype
                .entry(ct.as_ref())
                .or_default()
                .push(gene.as_ref());
        }

        for (ct, genes) in &by_celltype {
            eprintln!("  {}: {}", ct, genes.join(", "));
        }
    }
}

/// Write candidates to JSON file for external analysis
pub fn write_candidates_json(
    candidates: &[CelltypeCandidates],
    output_path: &str,
) -> anyhow::Result<()> {
    use std::fs::File;
    use std::io::BufWriter;

    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "{{")?;
    writeln!(w, "  \"candidates\": [")?;

    for (ci, ct) in candidates.iter().enumerate() {
        writeln!(w, "    {{")?;
        writeln!(w, "      \"celltype\": \"{}\",", ct.celltype_name)?;
        writeln!(w, "      \"topics\": [")?;

        for (ti, tm) in ct.topic_matches.iter().enumerate() {
            writeln!(w, "        {{")?;
            writeln!(w, "          \"topic\": \"{}\",", tm.topic_name)?;
            writeln!(w, "          \"pip\": {:.4},", tm.pip)?;
            writeln!(w, "          \"genes\": [")?;

            for (gi, g) in tm.candidates.iter().enumerate() {
                let comma = if gi + 1 < tm.candidates.len() { "," } else { "" };
                writeln!(
                    w,
                    "            {{\"gene\": \"{}\", \"weight\": {:.6}}}{}",
                    g.gene_name, g.weight, comma
                )?;
            }

            writeln!(w, "          ]")?;
            let comma = if ti + 1 < ct.topic_matches.len() { "," } else { "" };
            writeln!(w, "        }}{}", comma)?;
        }

        writeln!(w, "      ]")?;
        let comma = if ci + 1 < candidates.len() { "," } else { "" };
        writeln!(w, "    }}{}", comma)?;
    }

    writeln!(w, "  ]")?;
    writeln!(w, "}}")?;

    w.flush()?;
    Ok(())
}

/// Read marker suggestions from JSON file
/// Expected format: {"suggestions": [{"gene": "X", "celltype": "Y"}, ...]}
/// Or simple array: [{"gene": "X", "celltype": "Y"}, ...]
pub fn read_suggestions_json(path: &str) -> anyhow::Result<Vec<(Box<str>, Box<str>)>> {
    use std::fs;

    let content = fs::read_to_string(path)?;
    let mut suggestions = Vec::new();

    // Simple JSON parsing - look for "gene" and "celltype" pairs
    // This is basic but avoids adding serde_json dependency
    let mut in_obj = false;
    let mut current_gene: Option<String> = None;
    let mut current_celltype: Option<String> = None;

    for line in content.lines() {
        let line = line.trim();

        if line.contains("{") {
            in_obj = true;
            current_gene = None;
            current_celltype = None;
        }

        if in_obj {
            if let Some(start) = line.find("\"gene\"") {
                if let Some(val) = extract_json_string(line, start + 6) {
                    current_gene = Some(val);
                }
            }
            if let Some(start) = line.find("\"celltype\"") {
                if let Some(val) = extract_json_string(line, start + 10) {
                    current_celltype = Some(val);
                }
            }
        }

        if line.contains("}") && in_obj {
            if let (Some(g), Some(ct)) = (current_gene.take(), current_celltype.take()) {
                suggestions.push((g.into_boxed_str(), ct.into_boxed_str()));
            }
            in_obj = false;
        }
    }

    Ok(suggestions)
}

fn extract_json_string(line: &str, start: usize) -> Option<String> {
    let rest = &line[start..];
    let colon_pos = rest.find(':')?;
    let after_colon = &rest[colon_pos + 1..];

    let first_quote = after_colon.find('"')?;
    let after_first = &after_colon[first_quote + 1..];
    let second_quote = after_first.find('"')?;

    Some(after_first[..second_quote].to_string())
}

/// Reference marker database for auto-suggestions
#[derive(Default)]
pub struct MarkerDatabase {
    /// gene (lowercase) -> set of cell types (lowercase)
    gene_to_celltypes: HashMap<String, HashSet<String>>,
}

impl MarkerDatabase {
    /// Load using flexible gene matching and fuzzy cell type matching
    /// - Genes: uses flexible_gene_match (underscore-delimited segments)
    /// - Cell types: uses fuzzy substring matching for variation tolerance
    pub fn load_with_vocab(
        path: &str,
        genes: &[Box<str>],
        celltypes: &[Box<str>],
    ) -> anyhow::Result<Self> {
        let norm = |s: &str| s.to_lowercase().replace([' ', '-', '_'], "");

        let mut db = Self::default();
        for line in open_buf_reader(path)?.lines().filter_map(|l| l.ok()) {
            let tokens: Vec<_> = line.split(['\t', ',', ';', '|']).map(|s| s.trim()).collect();

            // Find genes using flexible matching
            let found_genes: Vec<(&str, &str)> = tokens
                .iter()
                .flat_map(|token| {
                    genes
                        .iter()
                        .filter(|dict_gene| flexible_gene_match(token, dict_gene))
                        .map(move |dict_gene| (*token, dict_gene.as_ref()))
                })
                .collect();

            // Find all matching celltypes (fuzzy matching with length constraints)
            let found_cts: Vec<&str> = tokens
                .iter()
                .flat_map(|token| {
                    let tn = norm(token);
                    celltypes
                        .iter()
                        .filter(move |ct| fuzzy_match_ct(&tn, &norm(ct)))
                        .map(|ct| ct.as_ref())
                })
                .collect();

            for (token_gene, dict_gene) in &found_genes {
                for ct in &found_cts {
                    db.gene_to_celltypes
                        .entry(norm(dict_gene))
                        .or_default()
                        .insert(norm(ct));
                    // Also store the token form for lookup
                    db.gene_to_celltypes
                        .entry(norm(token_gene))
                        .or_default()
                        .insert(norm(ct));
                }
            }
        }
        Ok(db)
    }

    /// Check if gene is a known marker for celltype
    pub fn is_known_marker(&self, gene: &str, celltype: &str) -> bool {
        let norm = |s: &str| s.to_lowercase().replace([' ', '-', '_'], "");
        let gn = norm(gene);
        let cn = norm(celltype);
        // Also try gene symbol after underscore
        let gs = gene.find('_').map(|i| norm(&gene[i + 1..]));

        let check = |key: &str| {
            self.gene_to_celltypes.get(key).map_or(false, |cts| {
                cts.iter().any(|ct| fuzzy_match_ct(ct, &cn))
            })
        };
        check(&gn) || gs.map_or(false, |s| check(&s))
    }
}

/// Auto-suggest markers based on reference database
/// Returns genes that are known markers for the suggested cell type
pub fn auto_suggest_markers(
    candidates: &[CelltypeCandidates],
    db: &MarkerDatabase,
) -> Vec<(Box<str>, Box<str>)> {
    let mut suggestions = Vec::new();

    for ct in candidates {
        for tm in &ct.topic_matches {
            for gene in &tm.candidates {
                if db.is_known_marker(&gene.gene_name, &ct.celltype_name) {
                    suggestions.push((gene.gene_name.clone(), ct.celltype_name.clone()));
                }
            }
        }
    }

    // Deduplicate
    suggestions.sort();
    suggestions.dedup();
    suggestions
}
