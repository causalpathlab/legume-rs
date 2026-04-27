//! Parquet writers, histogram formatter, and per-level output dispatch for the
//! link community model.

use crate::link_community::dict_merge::BhcMerge;
use crate::link_community::profiles::{
    compute_node_membership, fit_gene_community_param, shannon_entropy_rows,
    write_gene_community_param,
};
use crate::util::common::*;
use matrix_param::dmatrix_gamma::GammaMatrix;

/// Write link community assignments to parquet.
pub fn write_link_communities(
    file_path: &str,
    edges: &[(usize, usize)],
    membership: &[usize],
    cell_names: &[Box<str>],
) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    use parquet::basic::Type as ParquetType;

    let n_edges = edges.len();
    let left_cells: Vec<Box<str>> = edges.iter().map(|&(i, _)| cell_names[i].clone()).collect();
    let right_cells: Vec<Box<str>> = edges.iter().map(|&(_, j)| cell_names[j].clone()).collect();
    let cluster_f32: Vec<f32> = membership.iter().map(|&k| k as f32).collect();

    let col_names: Vec<Box<str>> =
        vec!["left_cell".into(), "right_cell".into(), "community".into()];
    let col_types = vec![
        ParquetType::BYTE_ARRAY,
        ParquetType::BYTE_ARRAY,
        ParquetType::FLOAT,
    ];

    let writer = ParquetWriter::new(
        file_path,
        (n_edges, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("edge"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_string_column(&mut row_group, &left_cells)?;
    parquet_add_string_column(&mut row_group, &right_cells)?;
    parquet_add_numeric_column(&mut row_group, &cluster_f32)?;

    row_group.close()?;
    writer.close()?;

    Ok(())
}

/// Write a dictionary-merge tree (cosine UPGMA) to parquet.
///
/// Columns: `merge_id`, `left`, `right`, `score`, `n_leaves`. The `score`
/// column carries the cosine similarity at which the two children were
/// merged (higher = more redundant gene programs). Reuses the
/// `BhcMerge` carrier type from `data_beans_alg::bhc` for the merge tree;
/// only the score interpretation differs from the original BHC log-BF.
pub fn write_dict_merges(file_path: &str, merges: &[BhcMerge]) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    use parquet::basic::Type as ParquetType;

    let n_rows = merges.len();
    let merge_ids: Vec<i32> = merges.iter().map(|m| m.id).collect();
    let lefts: Vec<i32> = merges.iter().map(|m| m.left).collect();
    let rights: Vec<i32> = merges.iter().map(|m| m.right).collect();
    let scores: Vec<f64> = merges.iter().map(|m| m.log_bf).collect();
    let n_leaves: Vec<i32> = merges.iter().map(|m| m.n_samples).collect();

    let col_names: Vec<Box<str>> = vec![
        "merge_id".into(),
        "left".into(),
        "right".into(),
        "score".into(),
        "n_leaves".into(),
    ];
    let col_types = vec![
        ParquetType::INT32,
        ParquetType::INT32,
        ParquetType::INT32,
        ParquetType::DOUBLE,
        ParquetType::INT32,
    ];

    let writer = ParquetWriter::new(
        file_path,
        (n_rows, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("step"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_numeric_column(&mut row_group, &merge_ids)?;
    parquet_add_numeric_column(&mut row_group, &lefts)?;
    parquet_add_numeric_column(&mut row_group, &rights)?;
    parquet_add_numeric_column(&mut row_group, &scores)?;
    parquet_add_numeric_column(&mut row_group, &n_leaves)?;

    row_group.close()?;
    writer.close()?;

    Ok(())
}

/// Write the consensus cut from `bhc_cut` to parquet (fine_id → super_id).
pub fn write_dict_cut(file_path: &str, labels: &[i32]) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    use parquet::basic::Type as ParquetType;

    let n_rows = labels.len();
    let communities: Vec<i32> = (0..n_rows as i32).collect();

    let col_names: Vec<Box<str>> = vec!["community".into(), "consensus".into()];
    let col_types = vec![ParquetType::INT32, ParquetType::INT32];

    let writer = ParquetWriter::new(
        file_path,
        (n_rows, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("row"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_numeric_column(&mut row_group, &communities)?;
    parquet_add_numeric_column(&mut row_group, labels)?;

    row_group.close()?;
    writer.close()?;

    Ok(())
}

/// ASCII histogram of link community sizes, showing communities with > 1% of edges.
pub fn link_community_histogram(membership: &[usize], k: usize, max_width: usize) -> String {
    let n = membership.len();
    let mut sizes = vec![0usize; k];
    for &c in membership {
        sizes[c] += 1;
    }

    let mut ranked: Vec<(usize, usize)> = sizes
        .iter()
        .enumerate()
        .filter(|(_, &s)| s > 0)
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(id, &s)| (id, s))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));

    let max_size = ranked.first().map(|&(_, s)| s).unwrap_or(1);
    let min_edges = n / 100; // 1% threshold

    let mut lines = Vec::new();
    lines.push(format!(
        "Link communities ({} edges, {} non-empty of {}):",
        n,
        ranked.len(),
        k
    ));
    lines.push(String::new());

    let mut shown = 0;
    for &(community_id, size) in &ranked {
        if size <= min_edges {
            break;
        }
        let pct = 100.0 * size as f64 / n as f64;
        let bar_len = ((size as f64 / max_size as f64) * max_width as f64) as usize;
        let bar = "\u{2588}".repeat(bar_len.max(1));
        lines.push(format!(
            "  Community {:3}  {:>7} edges ({:>5.1}%)  {}",
            community_id, size, pct, bar
        ));
        shown += 1;
    }

    let hidden = ranked.len() - shown;
    if hidden > 0 {
        let hidden_edges: usize = ranked[shown..].iter().map(|&(_, s)| s).sum();
        let hidden_pct = 100.0 * hidden_edges as f64 / n as f64;
        lines.push(format!(
            "  ... and {} more ({} edges, {:.1}%)",
            hidden, hidden_edges, hidden_pct
        ));
    }

    lines.join("\n")
}

/// One row of the score trace.
///
/// `level = -1` marks post-EM final rows. `sweep` is the 0-based sweep index
/// within a cascade level (Gibbs sweeps followed by greedy sweeps — the last
/// row per level is therefore the level-end summary). Post-EM emits a single
/// row with `sweep = 0`.
///
/// - `score / total_mass = -mean_H(p_k)` — mass-weighted mean per-community
///   entropy. Within-level "sharpness" indicator; not fair across levels.
/// - `mutual_information = H(p_global) − mean_H(p_k|community)` — nats of
///   information the community label carries about the gene profile.
///   Granularity-aware and comparable across cascade levels.
#[derive(Clone, Copy, Debug)]
pub struct ScoreEntry {
    pub level: i32,
    pub sweep: i32,
    pub score: f64,
    pub n_edges: usize,
    pub total_mass: f64,
    pub mutual_information: f64,
}

/// Write score trace to parquet with six columns:
/// `(level, sweep, score, n_edges, total_mass, mutual_information)`.
pub fn write_score_trace(file_path: &str, entries: &[ScoreEntry]) -> anyhow::Result<()> {
    use matrix_util::parquet::*;
    use parquet::basic::Type as ParquetType;

    let n_rows = entries.len();

    let levels: Vec<i32> = entries.iter().map(|e| e.level).collect();
    let sweeps: Vec<i32> = entries.iter().map(|e| e.sweep).collect();
    let scores: Vec<f64> = entries.iter().map(|e| e.score).collect();
    let n_edges: Vec<i32> = entries.iter().map(|e| e.n_edges as i32).collect();
    let total_mass: Vec<f64> = entries.iter().map(|e| e.total_mass).collect();
    let mi: Vec<f64> = entries.iter().map(|e| e.mutual_information).collect();

    let col_names: Vec<Box<str>> = vec![
        "level".into(),
        "sweep".into(),
        "score".into(),
        "n_edges".into(),
        "total_mass".into(),
        "mutual_information".into(),
    ];
    let col_types = vec![
        ParquetType::INT32,
        ParquetType::INT32,
        ParquetType::DOUBLE,
        ParquetType::INT32,
        ParquetType::DOUBLE,
        ParquetType::DOUBLE,
    ];

    let writer = ParquetWriter::new(
        file_path,
        (n_rows, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        Some("step"),
    )?;

    let row_names = writer.row_names_vec();
    let mut writer = writer.get_writer()?;
    let mut row_group = writer.next_row_group()?;

    parquet_add_bytearray(&mut row_group, row_names)?;
    parquet_add_numeric_column(&mut row_group, &levels)?;
    parquet_add_numeric_column(&mut row_group, &sweeps)?;
    parquet_add_numeric_column(&mut row_group, &scores)?;
    parquet_add_numeric_column(&mut row_group, &n_edges)?;
    parquet_add_numeric_column(&mut row_group, &total_mass)?;
    parquet_add_numeric_column(&mut row_group, &mi)?;

    row_group.close()?;
    writer.close()?;

    Ok(())
}

/// Write one cascade level's outputs: `.L{l}.link_community.parquet`,
/// Write `<prefix>.propensity.parquet` from cell-edge labels and return
/// the propensity matrix (reused to compute gene-community stats).
pub fn write_propensity_parquet(
    prefix: &str,
    edges: &[(usize, usize)],
    fine_labels: &[usize],
    n_cells: usize,
    k: usize,
    cell_names: &[Box<str>],
) -> anyhow::Result<Mat> {
    let propensity = compute_node_membership(edges, fine_labels, n_cells, k);

    let entropy_vec = shannon_entropy_rows(&propensity);
    let entropy_mat = Mat::from_column_slice(n_cells, 1, entropy_vec.as_slice());

    let mut col_names: Vec<Box<str>> = (0..k).map(|i| i.to_string().into_boxed_str()).collect();
    col_names.push("entropy".into());

    let combined = concatenate_horizontal(&[propensity.clone(), entropy_mat])?;
    combined.to_parquet_with_names(
        &format!("{}.propensity.parquet", prefix),
        (Some(cell_names), Some("cell")),
        Some(&col_names),
    )?;
    Ok(propensity)
}

/// Write the full per-partition output triple (link community edges,
/// cell propensity, gene×community stats) under a shared prefix. Returns the
/// propensity matrix and the fitted gene-community posterior so callers can
/// reuse them (e.g. the dictionary-merge step needs the posterior to
/// compute pairwise community cosine without re-reading the parquet).
#[allow(clippy::too_many_arguments)]
pub fn write_partition_outputs(
    prefix: &str,
    edges: &[(usize, usize)],
    fine_labels: &[usize],
    n_cells: usize,
    k: usize,
    cell_names: &[Box<str>],
    data_vec: &SparseIoVec,
    gene_weights: Option<&[f32]>,
    block_size: Option<usize>,
) -> anyhow::Result<(Mat, GammaMatrix)> {
    write_link_communities(
        &format!("{}.link_community.parquet", prefix),
        edges,
        fine_labels,
        cell_names,
    )?;
    let propensity = write_propensity_parquet(prefix, edges, fine_labels, n_cells, k, cell_names)?;
    let gene_community = fit_gene_community_param(&propensity, data_vec, gene_weights, block_size)?;
    let gene_names = data_vec.row_names()?;
    write_gene_community_param(&gene_community, &gene_names, prefix)?;
    Ok((propensity, gene_community))
}

/// Write one cascade level's outputs: `.L{l}.link_community.parquet`,
/// `.L{l}.propensity.parquet`, `.L{l}.gene_community.parquet`. The fine-edge
/// labels here are the super-edge assignment broadcast through
/// `transfer_labels`, so every per-level file is keyed on the same edge
/// list as the final output.
#[allow(clippy::too_many_arguments)]
pub fn write_level_outputs(
    out_prefix: &str,
    level_idx: usize,
    edges: &[(usize, usize)],
    fine_labels: &[usize],
    n_cells: usize,
    k: usize,
    cell_names: &[Box<str>],
    data_vec: &SparseIoVec,
    gene_weights: Option<&[f32]>,
    block_size: Option<usize>,
) -> anyhow::Result<()> {
    write_partition_outputs(
        &format!("{}.L{}", out_prefix, level_idx),
        edges,
        fine_labels,
        n_cells,
        k,
        cell_names,
        data_vec,
        gene_weights,
        block_size,
    )?;
    Ok(())
}
