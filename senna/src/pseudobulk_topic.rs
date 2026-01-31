//! Pseudobulk aggregation by topic proportions.
//!
//! Computes topic-weighted pseudobulk expression:
//! `Pseudobulk[gene, topic] = Σ_cell Expression[gene, cell] × TopicProportion[cell, topic]`
//!
//! Supports different weighting strategies:
//! - Soft: use topic proportions as-is (spread across topics)
//! - Hard: argmax assignment (each cell assigned to top topic only)

use crate::embed_common::*;
use crate::senna_input::{read_data_on_shared_rows, ReadSharedRowsArgs};
use data_beans::sparse_data_visitors::VisitColumnsOps;
use fnv::FnvHashSet as HashSet;
use matrix_util::membership::Membership;

/// Strategy for weighting cells when computing pseudobulk
#[derive(Clone, Copy, Debug, Default)]
pub enum TopicWeightStrategy {
    /// Use topic proportions as-is (can spread expression across many topics)
    #[default]
    Soft,
    /// Argmax: assign each cell to its top topic only (hard assignment)
    Hard,
}

impl TopicWeightStrategy {
    /// Apply the weighting strategy to topic proportions for a single cell
    pub fn apply(&self, proportions: &[f32]) -> Vec<f32> {
        match self {
            TopicWeightStrategy::Soft => proportions.to_vec(),
            TopicWeightStrategy::Hard => {
                let mut result = vec![0.0f32; proportions.len()];
                if let Some((max_idx, _)) = proportions
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                {
                    result[max_idx] = 1.0;
                }
                result
            }
        }
    }
}

/// Input data for pseudobulk visitor
struct VisitorInput {
    /// Cell index mapping: col_idx -> Option<cell_idx in topic matrix>
    col_to_cell: Vec<Option<usize>>,
    /// Topic proportions matrix (cells × topics)
    topic_proportions: Mat,
    /// Weighting strategy for topic proportions
    weight_strategy: TopicWeightStrategy,
}

/// Output accumulator for pseudobulk visitor
struct VisitorOutput {
    /// Pseudobulk matrix (genes × topics)
    pseudobulk: Mat,
    /// Sum of topic weights for normalization
    topic_sums: Vec<f32>,
}

/// Visitor function for pseudobulk aggregation
fn aggregate_visitor(
    (lb, ub): (usize, usize),
    data_vec: &SparseIoVec,
    input: &VisitorInput,
    arc_out: std::sync::Arc<std::sync::Mutex<&mut VisitorOutput>>,
) -> anyhow::Result<()> {
    let n_topics = input.topic_proportions.ncols();
    let n_genes = data_vec.num_rows();

    // Local accumulators for this block
    let mut local_pseudobulk = Mat::zeros(n_genes, n_topics);
    let mut local_topic_sums = vec![0.0f32; n_topics];

    // Process columns in this block
    for col_idx in lb..ub {
        let cell_idx = match input.col_to_cell[col_idx] {
            Some(idx) => idx,
            None => continue,
        };

        // Read expression for this cell
        let ((nrow, _ncol), triplets) = data_vec.columns_triplets(std::iter::once(col_idx))?;
        debug_assert_eq!(nrow, n_genes);

        // Get topic proportions for this cell and apply weighting strategy
        let theta_row: Vec<f32> = input.topic_proportions.row(cell_idx).iter().cloned().collect();
        let weights = input.weight_strategy.apply(&theta_row);

        // Accumulate topic weights
        for k in 0..n_topics {
            local_topic_sums[k] += weights[k];
        }

        // Accumulate weighted expression
        for (gene_idx, _col, val) in triplets {
            let gene_idx = gene_idx as usize;
            for k in 0..n_topics {
                local_pseudobulk[(gene_idx, k)] += val * weights[k];
            }
        }
    }

    // Merge local results into shared output
    let mut out = arc_out.lock().expect("lock output");
    out.pseudobulk += &local_pseudobulk;
    for k in 0..n_topics {
        out.topic_sums[k] += local_topic_sums[k];
    }

    Ok(())
}

/// Compute pseudobulk gene expression by aggregating cells weighted by topic proportions.
///
/// Uses `Membership` for flexible cell name matching (exact, base-key `@`, prefix).
/// Uses visitor pattern for efficient parallel processing.
///
/// # Arguments
/// * `data_files` - Expression data files (.h5 or .zarr)
/// * `cell_names` - Cell names from topic proportion matrix
/// * `topic_proportions` - Topic proportion matrix (cells × topics)
/// * `marker_genes` - Optional set of marker genes to filter (keeps only these genes)
/// * `weight_strategy` - Strategy for weighting cells (Soft or Hard)
///
/// # Returns
/// * `(row_names, pseudobulk_matrix)` - Gene names and pseudobulk (genes × topics)
pub fn compute_pseudobulk_by_topic(
    data_files: &[Box<str>],
    cell_names: &[Box<str>],
    topic_proportions: &Mat,
    marker_genes: Option<&HashSet<Box<str>>>,
    weight_strategy: TopicWeightStrategy,
) -> anyhow::Result<(Vec<Box<str>>, Mat)> {
    // Build membership: cell_name -> index (as string for Membership API)
    let pairs = cell_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i.to_string().into_boxed_str()));

    let membership = Membership::from_pairs(pairs, true).with_delimiter('@');

    // Read expression data
    let data = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: data_files.to_vec(),
        batch_files: None,
        preload: false,
    })?;

    let data_vec = &data.data;
    let row_names = data_vec.row_names()?;
    let col_names = data_vec.column_names()?;
    let n_genes = data_vec.num_rows();
    let n_topics = topic_proportions.ncols();
    let n_data_cells = data_vec.num_columns();

    info!(
        "Expression data: {} genes × {} cells",
        n_genes, n_data_cells
    );

    // Match expression columns to topic cells
    let (matched_map, stats) = membership.match_keys(&col_names);

    info!(
        "Cell matching: {} exact, {} base-key, {} prefix, {} unmatched",
        stats.exact, stats.base_key, stats.prefix, stats.unmatched
    );

    if stats.total_matched() == 0 {
        return Err(anyhow::anyhow!(
            "No matching cells found.\n\
             Sample expression cells: {:?}\n\
             Sample topic cells: {:?}",
            col_names.iter().take(3).collect::<Vec<_>>(),
            cell_names.iter().take(3).collect::<Vec<_>>()
        ));
    }

    // Build column -> cell index mapping
    let mut col_to_cell: Vec<Option<usize>> = vec![None; n_data_cells];
    for (col_idx, col_name) in col_names.iter().enumerate() {
        if let Some(cell_idx_str) = matched_map.get(col_name) {
            if let Ok(cell_idx) = cell_idx_str.parse::<usize>() {
                col_to_cell[col_idx] = Some(cell_idx);
            }
        }
    }

    // Prepare visitor state
    let visitor_input = VisitorInput {
        col_to_cell,
        topic_proportions: topic_proportions.clone(),
        weight_strategy,
    };

    let mut visitor_output = VisitorOutput {
        pseudobulk: Mat::zeros(n_genes, n_topics),
        topic_sums: vec![0.0f32; n_topics],
    };

    // Run parallel aggregation
    info!("Aggregating expression by topic proportions...");
    data_vec.visit_columns_by_block(
        &aggregate_visitor,
        &visitor_input,
        &mut visitor_output,
        None,
    )?;

    // Normalize by topic weights
    for k in 0..n_topics {
        if visitor_output.topic_sums[k] > 0.0 {
            for g in 0..n_genes {
                visitor_output.pseudobulk[(g, k)] /= visitor_output.topic_sums[k];
            }
        }
    }

    info!(
        "Computed pseudobulk: {} genes × {} topics ({} matched cells)",
        n_genes,
        n_topics,
        stats.total_matched()
    );

    // Filter to marker genes if specified (using same matching logic as build_annotation_matrix)
    if let Some(markers) = marker_genes {
        let mut filtered_rows = Vec::new();
        let mut filtered_indices = Vec::new();

        for (idx, row_name) in row_names.iter().enumerate() {
            let row_lower = row_name.to_lowercase();

            // Check if this gene matches any marker gene
            let is_marker = markers.iter().any(|marker| {
                let marker_lower = marker.to_lowercase();

                // 1. Exact match
                if row_lower == marker_lower {
                    return true;
                }

                // 2. Match at word boundary: row ends with "_MARKER" or starts with "MARKER_"
                let suffix_match = row_lower.ends_with(&format!("_{}", marker_lower));
                let prefix_match = row_lower.starts_with(&format!("{}_", marker_lower));

                // 3. Match marker as a complete segment between underscores
                let segment_match = row_lower.contains(&format!("_{}_", marker_lower));

                suffix_match || prefix_match || segment_match
            });

            if is_marker {
                filtered_rows.push(row_name.clone());
                filtered_indices.push(idx);
            }
        }

        if filtered_indices.is_empty() {
            return Err(anyhow::anyhow!(
                "No marker genes found in expression data.\n\
                 Sample expression genes: {:?}\n\
                 Sample marker genes: {:?}",
                row_names.iter().take(5).collect::<Vec<_>>(),
                markers.iter().take(5).collect::<Vec<_>>()
            ));
        }

        // Build filtered matrix
        let n_filtered = filtered_indices.len();
        let mut filtered_pseudobulk = Mat::zeros(n_filtered, n_topics);
        for (new_idx, &old_idx) in filtered_indices.iter().enumerate() {
            for k in 0..n_topics {
                filtered_pseudobulk[(new_idx, k)] = visitor_output.pseudobulk[(old_idx, k)];
            }
        }

        info!(
            "Filtered to {} marker genes (from {} total)",
            n_filtered, n_genes
        );

        Ok((filtered_rows, filtered_pseudobulk))
    } else {
        Ok((row_names, visitor_output.pseudobulk))
    }
}
