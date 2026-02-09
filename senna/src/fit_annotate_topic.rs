use crate::deconv::{vmf_assign, vmf_assign_averaged};
use crate::embed_common::*;
use crate::interactive_markers::{
    augment_membership_matrix, auto_suggest_markers, find_candidate_markers, flexible_gene_match,
    print_augmentation_summary, read_suggestions_json, run_interactive_round,
    save_augmented_markers, write_candidates_json, MarkerDatabase,
};
use matrix_util::common_io::*;

use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;

#[derive(Args, Debug)]
pub struct AnnotateTopicArgs {
    // === Required inputs ===
    #[arg(
        short = 'g',
        long = "gene-dictionary",
        required = true,
        help = "Gene dictionary matrix (from `topic`)"
    )]
    dict_file: Box<str>,

    #[arg(
        short = 'z',
        long = "latent-topic",
        required = true,
        help = "Latent topic proportions (from `topic`)"
    )]
    latent_file: Box<str>,

    #[arg(
        short = 'm',
        long = "marker-genes",
        required = true,
        help = "Marker file: gene<tab>celltype per line (flexible gene matching)"
    )]
    marker_file: Box<str>,

    #[arg(
        short = 'o',
        long,
        required = true,
        help = "Output prefix (.assignment.parquet, .annotation.parquet, .argmax.tsv)"
    )]
    out: Box<str>,

    // === vMF parameters ===
    #[arg(
        short = 'k',
        long,
        value_delimiter = ',',
        help = "vMF κ (default: auto from sqrt(n_genes/2) to 1024, Bayesian averaged)"
    )]
    kappa: Vec<f32>,

    #[arg(long, default_value_t = 0.01, help = "Min probability to include topic")]
    min_pip: f32,

    #[arg(short = 'b', long, default_value_t = 0.0, help = "Background for non-markers")]
    background: f32,

    // === Interactive mode ===
    #[arg(short = 'I', long, help = "Interactive marker augmentation")]
    interactive: bool,

    #[arg(long, default_value_t = 3, help = "Candidates per topic")]
    top_candidates: usize,

    #[arg(long, default_value_t = 2, help = "Topics per celltype")]
    topics_per_celltype: usize,

    #[arg(long, default_value_t = 0.05, help = "Min probability in interactive")]
    interactive_min_pip: f32,

    #[arg(long, default_value_t = 1.0, help = "Weight for augmented markers")]
    augment_weight: f32,

    // === LLM-assisted workflow ===
    #[arg(long, help = "Output candidates JSON for LLM review")]
    suggest_only: Option<Box<str>>,

    #[arg(long, help = "Apply suggestions from JSON")]
    apply_suggestions: Option<Box<str>>,

    #[arg(long, help = "Reference marker DB for auto-suggestions")]
    marker_db: Option<Box<str>>,

    // === Misc ===
    #[arg(short = 'v', long, help = "Verbose output")]
    verbose: bool,
}

/// Training data bundle
struct TrainingData {
    /// Gene × topic data matrix (log dictionary or normalized pseudobulk)
    data: Mat,
    /// Gene × annotation membership matrix
    membership: Mat,
    /// Gene names
    gene_names: Vec<Box<str>>,
}

struct AnnotInfo {
    membership_ga: Mat,
    annot_names: Vec<Box<str>>,
}

pub fn annotate_topics(args: &AnnotateTopicArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // 1. Read dictionary (log-probabilities) and latent states (probabilities)
    let MatWithNames {
        rows: row_names,
        cols: _topics,
        mat: log_dict_dk,
    } = read_mat(&args.dict_file)?;

    info!(
        "Read dictionary {} x {}",
        log_dict_dk.nrows(),
        log_dict_dk.ncols()
    );

    let MatWithNames {
        rows: cell_names,
        cols: topic_names,
        mat: topic_nt_raw,
    } = read_mat(&args.latent_file)?;

    // Convert log-probabilities to probabilities if needed
    let topic_nt = if topic_nt_raw.max() <= 0.0 {
        info!("Detected log-probabilities in latent file, converting to probabilities");
        topic_nt_raw.map(|x| x.exp())
    } else {
        topic_nt_raw
    };

    // 2. Read marker gene annotation
    let AnnotInfo {
        membership_ga,
        annot_names: annot_names_raw,
    } = build_annotation_matrix(&args.marker_file, &row_names)?;

    // Drop empty columns (cell types with no markers in the gene set)
    let col_sums: Vec<f32> = (0..membership_ga.ncols())
        .map(|c| membership_ga.column(c).iter().sum())
        .collect();
    let non_empty_cols: Vec<usize> = col_sums
        .iter()
        .enumerate()
        .filter(|(_, &s)| s > 0.0)
        .map(|(i, _)| i)
        .collect();

    let (membership_ga, annot_names): (Mat, Vec<Box<str>>) =
        if non_empty_cols.len() < membership_ga.ncols() {
            let dropped: Vec<_> = (0..membership_ga.ncols())
                .filter(|c| col_sums[*c] == 0.0)
                .map(|c| annot_names_raw[c].as_ref())
                .collect();
            info!(
                "Dropping {} empty cell types with no markers: {:?}",
                dropped.len(),
                dropped
            );
            (
                membership_ga.select_columns(&non_empty_cols),
                non_empty_cols
                    .iter()
                    .map(|&c| annot_names_raw[c].clone())
                    .collect(),
            )
        } else {
            (membership_ga, annot_names_raw)
        };

    let nnz_features = (0..membership_ga.nrows())
        .filter(|&i| membership_ga.row(i).iter().any(|&x| x > 0.0))
        .count();

    info!(
        "Found {} cell types matched over {} marker genes",
        annot_names.len(),
        nnz_features
    );

    // 3. Prepare data
    let training_data = TrainingData {
        data: log_dict_dk.clone(),
        membership: membership_ga.clone(),
        gene_names: row_names.clone(),
    };

    // Compute kappa range: start from sqrt(n_genes/2), double up to 1024
    let kappa_range: Vec<f32> = if args.kappa.is_empty() {
        let n_genes = training_data.data.nrows();
        let kappa_start = ((n_genes as f32) / 2.0).sqrt();
        let mut kappas = Vec::new();
        let mut k = kappa_start;
        while k <= 1024.0 {
            kappas.push(k);
            k *= 2.0;
        }
        if kappas.last().map_or(true, |&last| last < 1024.0) {
            kappas.push(1024.0);
        }
        info!(
            "Auto kappa range from sqrt({}/2)={:.1}: {:?}",
            n_genes, kappa_start, kappas
        );
        kappas
    } else {
        args.kappa.clone()
    };

    // Closure: compute topic-celltype assignment with given membership matrix
    let compute_assignment = |membership: &Mat| -> anyhow::Result<Mat> {
        // Convert log-probs to probs for topic profiles
        let topic_profiles = training_data.data.map(|x| x.exp());

        // Add background frequency if requested
        let membership_with_bg = if args.background > 0.0 {
            info!(
                "Adding background frequency {} to membership matrix",
                args.background
            );
            membership.map(|x| if x == 0.0 { args.background } else { x })
        } else {
            membership.clone()
        };

        // Compute assignment probabilities
        let probs = if kappa_range.len() == 1 {
            info!("Computing vMF assignment with κ={}", kappa_range[0]);
            vmf_assign(&topic_profiles, &membership_with_bg, kappa_range[0])
        } else {
            info!(
                "Computing vMF assignment with Bayesian averaging over κ={:?}",
                kappa_range
            );
            vmf_assign_averaged(&topic_profiles, &membership_with_bg, &kappa_range)
        };

        // Result is Topic × CellType, transpose to CellType × Topic for consistency
        Ok(probs.transpose())
    };

    // 4. Handle different modes: suggest-only, apply-suggestions, interactive, or direct
    let original_markers = read_marker_gene_info(&args.marker_file)?;
    let mut membership = training_data.membership.clone();
    let mut all_new_markers: Vec<(Box<str>, Box<str>)> = Vec::new();

    // Load reference marker database if provided
    let marker_db = if let Some(db_path) = &args.marker_db {
        info!("Loading marker database from {}...", db_path);
        Some(MarkerDatabase::load_with_vocab(
            db_path,
            &training_data.gene_names,
            &annot_names,
        )?)
    } else {
        None
    };

    // Suggest-only mode: output candidates for LLM review
    if let Some(output_json) = &args.suggest_only {
        let pip_mat = compute_assignment(&membership)?;
        // pip_mat is CellType × Topic (annotation × topic), which matches expected format
        let candidates = find_candidate_markers(
            &pip_mat,            // annotation × topic
            &training_data.data, // gene × topic
            &membership,         // gene × annotation
            &training_data.gene_names,
            &topic_names,
            &annot_names,
            args.interactive_min_pip, // min_pip: f32
            args.top_candidates,      // top_k_genes: usize
            args.topics_per_celltype, // top_k_topics: usize
        );

        // Auto-accept known markers if database provided
        if let Some(db) = &marker_db {
            let auto_accepted = auto_suggest_markers(&candidates, db);
            if !auto_accepted.is_empty() {
                info!(
                    "Auto-accepted {} markers from database",
                    auto_accepted.len()
                );
                let auto_file = format!("{}.auto_suggestions.tsv", args.out);
                use std::io::Write;
                let mut f = std::fs::File::create(&auto_file)?;
                writeln!(f, "gene\tcelltype")?;
                for (gene, celltype) in &auto_accepted {
                    writeln!(f, "{}\t{}", gene, celltype)?;
                }
                info!("Wrote auto-accepted markers to {}", auto_file);
            }
        }
        write_candidates_json(&candidates, output_json)?;
        info!("Wrote candidate markers to {}", output_json);
        return Ok(());
    }

    // Apply-suggestions mode: load and apply suggestions from JSON
    if let Some(suggestions_json) = &args.apply_suggestions {
        let suggestions = read_suggestions_json(suggestions_json)?;
        info!(
            "Applying {} suggestions from {}",
            suggestions.len(),
            suggestions_json
        );

        for (gene, celltype) in &suggestions {
            // Find matching gene using flexible matching
            if let Some((gene_idx, matched_gene)) = training_data
                .gene_names
                .iter()
                .enumerate()
                .find(|(_, g)| flexible_gene_match(gene, g))
                .map(|(i, g)| (i, g.clone()))
            {
                if let Some(ct_idx) = annot_names
                    .iter()
                    .position(|n| n.as_ref() == celltype.as_ref())
                {
                    membership[(gene_idx, ct_idx)] = args.augment_weight;
                    all_new_markers.push((matched_gene, celltype.clone()));
                }
            }
        }
    }

    // Interactive mode or direct computation
    let pip_mat = if args.interactive {
        let mut iteration = 1;
        loop {
            info!("=== Interactive round {} ===", iteration);
            let pip = compute_assignment(&membership)?;

            let candidates = find_candidate_markers(
                &pip,                // annotation × topic
                &training_data.data, // gene × topic
                &membership,         // gene × annotation
                &training_data.gene_names,
                &topic_names,
                &annot_names,
                args.interactive_min_pip, // min_pip: f32
                args.top_candidates,      // top_k_genes: usize
                args.topics_per_celltype, // top_k_topics: usize
            );

            if candidates.is_empty() {
                info!("No more candidates above threshold. Done.");
                break pip;
            }

            let result = run_interactive_round(&candidates, iteration)?;

            if !result.proceed && result.new_markers.is_empty() {
                return Ok(());
            }

            if result.new_markers.is_empty() {
                info!("No new markers. Finalizing...");
                break pip;
            }

            augment_membership_matrix(
                &mut membership,
                &training_data.gene_names,
                &annot_names,
                &result.new_markers,
                args.augment_weight,
            );
            all_new_markers.extend(result.new_markers);

            if !result.proceed {
                info!("Final computation with augmented markers...");
                break compute_assignment(&membership)?;
            }

            iteration += 1;
        }
    } else {
        compute_assignment(&membership)?
    };

    // Save augmented markers if any
    if !all_new_markers.is_empty() {
        let marker_file = format!("{}.augmented_markers.tsv", args.out);
        save_augmented_markers(&original_markers, &all_new_markers, &marker_file)?;
        print_augmentation_summary(&all_new_markers, all_new_markers.len());
        info!("Saved augmented markers to {}", marker_file);
    }

    // 5. Output results
    // pip_mat is CellType × Topic (transposed), we want Topic × CellType for output
    let assignment_mat = pip_mat.transpose();
    let assignment_file = format!("{}.assignment.parquet", args.out);
    assignment_mat.to_parquet_with_names(&assignment_file, (Some(&topic_names), Some("topic")), Some(&annot_names))?;
    info!("Wrote topic-celltype assignment to {}", assignment_file);

    // Filter topics by max probability threshold
    let n_topics = assignment_mat.nrows();
    let mut topic_mask = vec![true; n_topics];
    let mut excluded_topics = Vec::new();

    for t in 0..n_topics {
        let max_prob = (0..assignment_mat.ncols())
            .map(|a| assignment_mat[(t, a)])
            .fold(0.0f32, f32::max);
        if max_prob < args.min_pip {
            topic_mask[t] = false;
            excluded_topics.push((t, max_prob));
        }
    }

    let n_included = topic_mask.iter().filter(|&&x| x).count();

    if !excluded_topics.is_empty() {
        info!(
            "Excluding {} topics with max prob < {}: {:?}",
            excluded_topics.len(),
            args.min_pip,
            excluded_topics
                .iter()
                .map(|(t, p)| format!("{}({:.3})", topic_names[*t], p))
                .collect::<Vec<_>>()
        );
    }

    // Check if any topics remain
    if n_included == 0 {
        log::warn!(
            "All {} topics were excluded (max prob < {}). Try lowering --min-pip.",
            n_topics,
            args.min_pip
        );
        let n_annots = annot_names.len();
        let uniform_prob = 1.0 / n_annots as f32;
        let topic_annot = Mat::from_fn(cell_names.len(), n_annots, |_, _| uniform_prob);
        let cell_annot_file = format!("{}.annotation.parquet", args.out);
        topic_annot.to_parquet_with_names(&cell_annot_file, (Some(&cell_names), Some("cell")), Some(&annot_names))?;

        let argmax_file = format!("{}.argmax.tsv", args.out);
        write_argmax_assignments(&topic_annot, &cell_names, &annot_names, &argmax_file)?;
        display_annotation_histogram(&topic_annot, &annot_names);
        return Ok(());
    }

    // Apply mask to assignment matrix
    let mut assignment_filtered = assignment_mat.clone();
    for t in 0..n_topics {
        if !topic_mask[t] {
            for a in 0..assignment_filtered.ncols() {
                assignment_filtered[(t, a)] = 0.0;
            }
        }
    }

    // Cell annotation = topic_proportions × assignment, then normalize rows
    // topic_nt: Cell × Topic (from latent file)
    // assignment_filtered: Topic × CellType
    // Result: Cell × CellType (each row = one cell's distribution over cell types)
    let topic_annot = (&topic_nt * &assignment_filtered).sum_to_one_rows();
    let cell_annot_file = format!("{}.annotation.parquet", args.out);
    topic_annot.to_parquet_with_names(&cell_annot_file, (Some(&cell_names), Some("cell")), Some(&annot_names))?;

    let argmax_file = format!("{}.argmax.tsv", args.out);
    write_argmax_assignments(&topic_annot, &cell_names, &annot_names, &argmax_file)?;

    display_annotation_histogram(&topic_annot, &annot_names);

    Ok(())
}

fn write_argmax_assignments(
    annot: &Mat,
    cell_names: &[Box<str>],
    annot_names: &[Box<str>],
    output_file: &str,
) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(output_file)?;
    writeln!(file, "cell\tcell_type\tprobability")?;

    for i in 0..annot.nrows() {
        let row = annot.row(i);
        let (max_idx, max_val) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        writeln!(
            file,
            "{}\t{}\t{:.4}",
            cell_names[i], annot_names[max_idx], max_val
        )?;
    }
    info!("Wrote argmax assignments to {}", output_file);
    Ok(())
}

fn display_annotation_histogram(annot: &Mat, annot_names: &[Box<str>]) {
    let n_cells = annot.nrows();
    let n_types = annot.ncols();

    let mut max_probs = Vec::with_capacity(n_cells);
    let mut assignments = Vec::with_capacity(n_cells);

    for i in 0..n_cells {
        let row = annot.row(i);
        let (max_idx, max_val) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        max_probs.push(*max_val);
        assignments.push(max_idx);
    }

    let mut type_counts = vec![0usize; n_types];
    let mut type_prob_sum = vec![0.0f32; n_types];
    for (i, &ct) in assignments.iter().enumerate() {
        type_counts[ct] += 1;
        type_prob_sum[ct] += max_probs[i];
    }

    let mut sorted_types: Vec<usize> = (0..n_types).collect();
    sorted_types.sort_by(|&a, &b| type_counts[b].cmp(&type_counts[a]));

    let max_count = *type_counts.iter().max().unwrap_or(&1);
    const MAX_BAR: usize = 20;

    let mean_prob: f32 = max_probs.iter().sum::<f32>() / n_cells as f32;
    let above_50 = max_probs.iter().filter(|&&x| x > 0.5).count();
    let above_70 = max_probs.iter().filter(|&&x| x > 0.7).count();

    eprintln!();
    eprintln!("Annotation Summary ({} cells)", n_cells);
    eprintln!(
        "  Mean max-prob: {:.3}  >0.5: {} ({:.1}%)  >0.7: {} ({:.1}%)",
        mean_prob,
        above_50,
        100.0 * above_50 as f32 / n_cells as f32,
        above_70,
        100.0 * above_70 as f32 / n_cells as f32
    );
    eprintln!();

    for &ct in &sorted_types {
        if type_counts[ct] == 0 {
            continue;
        }
        let bar_len = (type_counts[ct] * MAX_BAR) / max_count.max(1);
        let bar: String = "█".repeat(bar_len);
        let avg_prob = if type_counts[ct] > 0 {
            type_prob_sum[ct] / type_counts[ct] as f32
        } else {
            0.0
        };
        eprintln!(
            "  {:20} {:5} ({:5.1}%) {:.2} {}",
            annot_names[ct],
            type_counts[ct],
            100.0 * type_counts[ct] as f32 / n_cells as f32,
            avg_prob,
            bar
        );
    }
    eprintln!();
}

fn read_mat(file_path: &str) -> anyhow::Result<MatWithNames<Mat>> {
    Ok(match file_ext(file_path)?.as_ref() {
        "parquet" => Mat::from_parquet(file_path)?,
        _ => Mat::read_data(file_path, &['\t', ','], None, Some(0), None, None)?,
    })
}

fn read_marker_gene_info(file_path: &str) -> anyhow::Result<HashMap<Box<str>, Box<str>>> {
    let ReadLinesOut { lines, header: _ } =
        read_lines_of_words_delim(&file_path, &['\t', ','], -1)?;

    Ok(lines
        .into_iter()
        .filter_map(|words| {
            if words.len() < 2 {
                None
            } else {
                Some((words[0].clone(), words[1].clone()))
            }
        })
        .collect())
}

fn build_annotation_matrix(
    marker_gene_path: &str,
    row_names: &[Box<str>],
) -> anyhow::Result<AnnotInfo> {
    let gene_to_type = read_marker_gene_info(marker_gene_path)?;

    if gene_to_type.is_empty() {
        return Err(anyhow::anyhow!("empty/invalid marker gene information"));
    }

    // Collect unique cell types
    let mut annot_set: HashSet<Box<str>> = HashSet::default();
    for cell_type in gene_to_type.values() {
        let normalized = cell_type.replace(" ", "_");
        annot_set.insert(normalized.into_boxed_str());
    }
    let mut annot_names: Vec<Box<str>> = annot_set.into_iter().collect();
    annot_names.sort();

    // Build membership matrix
    let n_genes = row_names.len();
    let n_annots = annot_names.len();
    let mut membership = Mat::zeros(n_genes, n_annots);

    let mut matched = 0;
    let mut unmatched = Vec::new();

    for (gene, cell_type) in &gene_to_type {
        let normalized_type = cell_type.replace(" ", "_");
        let annot_idx = annot_names
            .iter()
            .position(|n| n.as_ref() == normalized_type)
            .unwrap();

        // Find matching gene using flexible matching
        if let Some(gene_idx) = row_names
            .iter()
            .position(|dict_gene| flexible_gene_match(gene, dict_gene))
        {
            membership[(gene_idx, annot_idx)] = 1.0;
            matched += 1;
        } else {
            unmatched.push(gene.clone());
        }
    }

    if !unmatched.is_empty() && unmatched.len() <= 10 {
        info!("Unmatched marker genes: {:?}", unmatched);
    } else if !unmatched.is_empty() {
        info!("{} marker genes not found in dictionary", unmatched.len());
    }

    info!(
        "Matched {}/{} marker genes to {} cell types",
        matched,
        gene_to_type.len(),
        n_annots
    );

    Ok(AnnotInfo {
        membership_ga: membership,
        annot_names,
    })
}
