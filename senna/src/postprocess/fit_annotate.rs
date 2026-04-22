//! Gene-vertex anchor annotation.
//!
//! Given a trained topic model (`β`, cell latent) and per-PB mean latent
//! (from `senna viz`), pick `K_anchor` genes that span the most orthogonal
//! directions in the reconstructed gene × PB expression space. Each anchor
//! gene is labeled against a user marker file by direct dictionary lookup
//! (gene → celltype), disambiguated toward the most *specific* celltype
//! (the one with the fewest markers, i.e. the tightest claim). Each anchor
//! also carries a set of top-N correlated genes to show its neighborhood
//! in the model.
//!
//! Cells are soft-assigned to anchors by cosine-softmax on (`cell_latent`,
//! β-row-of-anchor-gene); the cell annotation is the product of the
//! cell→anchor and anchor→celltype matrices, row-normalized.

use super::interactive_markers::{
    augment_membership_matrix, auto_suggest_markers, find_candidate_markers,
    print_augmentation_summary, read_suggestions_json, run_interactive_round,
    save_augmented_markers, write_candidates_json, MarkerDatabase,
};
use super::viz_prep::load_dictionary;
use crate::anchor_common::{base_celltype_label, gram_schmidt_anchors, zscore_columns};
use crate::embed_common::*;
use crate::marker_support::{build_annotation_matrix, flexible_gene_match, AnnotInfo};

/// Prefix used for unlabeled (novel) anchor genes; also used by
/// `make_anchor_to_celltype` as the skip-this-anchor marker.
const NOVEL_ANCHOR_PREFIX: &str = "novel_";

#[derive(Args, Debug)]
pub struct AnnotateTopicArgs {
    // === Required inputs ===
    #[arg(
        short = 'g',
        long = "gene-dictionary",
        required = true,
        help = "Gene dictionary β (genes × K) from `senna topic`"
    )]
    dict_file: Box<str>,

    #[arg(
        short = 'z',
        long = "latent-topic",
        required = true,
        help = "Latent topic proportions (cells × K) from `senna topic`"
    )]
    latent_file: Box<str>,

    #[arg(
        short = 'p',
        long = "pb-features",
        alias = "pb-mean-latent",
        required = true,
        help = "Per-PB feature matrix from `senna viz`: either pb_gene_mean.parquet \
                (n_pb × n_genes, log1p-CPM — preferred) or the legacy pb_mean_latent.parquet \
                (n_pb × K, back-projected via β)."
    )]
    pb_mean_latent_file: Box<str>,

    #[arg(
        short = 'm',
        long = "marker-genes",
        required = true,
        help = "Marker file: gene<tab>celltype per line (flexible gene matching)"
    )]
    marker_file: Box<str>,

    #[arg(short = 'o', long, required = true, help = "Output prefix")]
    out: Box<str>,

    // === Anchor selection ===
    #[arg(
        short = 'K',
        long = "num-anchors",
        help = "Number of anchor genes (default K topics, clamped to n_genes)",
        long_help = "Number of anchor genes picked via Gram-Schmidt on the\n\
                     column-z-scored gene × PB reconstruction. Defaults to K\n\
                     topics so we get one anchor per independent direction\n\
                     the dictionary can express; larger values are allowed but\n\
                     past K the residuals become degenerate."
    )]
    num_anchors: Option<usize>,

    #[arg(
        long = "top-correlated",
        default_value_t = 15,
        help = "Top-N correlated genes reported per anchor gene",
        long_help = "For each anchor gene, compute cosine similarity against\n\
                     every other gene in the z-scored gene × PB matrix and\n\
                     keep the top-N. These are emitted in\n\
                     `{out}.correlated_genes.tsv` and marked if they also\n\
                     appear in the user marker file."
    )]
    top_correlated: usize,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "Softmax temperature for cell → anchor assignment",
        long_help = "Lower values make cell → anchor assignments sharper; higher\n\
                     values spread mass across more anchors. Cell latent and\n\
                     each anchor gene's β-row are L2-normalized before the\n\
                     cosine step so this temperature is dataset-invariant."
    )]
    temperature: f32,

    // === Filtering ===
    #[arg(
        long,
        default_value_t = 0.01,
        help = "Min anchor-celltype confidence to keep the anchor"
    )]
    min_pip: f32,

    // === Interactive mode ===
    #[arg(short = 'I', long, help = "Interactive marker augmentation")]
    interactive: bool,

    #[arg(long, default_value_t = 3, help = "Candidates per anchor")]
    top_candidates: usize,

    #[arg(long, default_value_t = 2, help = "Anchors per celltype to surface")]
    topics_per_celltype: usize,

    #[arg(long, default_value_t = 0.05, help = "Min confidence in interactive")]
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
}

/// Per-anchor output packet. Kept together so the TSV writer doesn't need a
/// 7-parameter signature.
struct AnchorRecord {
    gene_idx: usize,
    gene_name: Box<str>,
    label: Box<str>,
    primary_celltype: Option<usize>,
    n_claiming: usize,
    specificity: f32,
    correlated: Vec<(usize, f32)>,
}

pub fn annotate_topics(args: &AnnotateTopicArgs) -> anyhow::Result<()> {
    // 1. Inputs
    let MatWithNames {
        rows: gene_names,
        cols: _,
        mat: _log_dict,
    } = read_mat(&args.dict_file)?;
    let n_genes = gene_names.len();

    let MatWithNames {
        rows: cell_names,
        cols: _,
        mat: topic_nt_raw,
    } = read_mat(&args.latent_file)?;
    let k_topics = topic_nt_raw.ncols();
    let topic_nt = if topic_nt_raw.max() <= 0.0 {
        info!("Detected log-probabilities in latent file, exponentiating");
        topic_nt_raw.map(f32::exp)
    } else {
        topic_nt_raw
    };

    let beta_prob = load_dictionary(&args.dict_file, k_topics)?;
    if beta_prob.nrows() != n_genes {
        anyhow::bail!(
            "Dictionary row count {} ≠ gene_names length {}",
            beta_prob.nrows(),
            n_genes
        );
    }

    let MatWithNames {
        rows: _pb_names,
        cols: _,
        mat: pb_features,
    } = read_mat(&args.pb_mean_latent_file)?;
    let n_pb = pb_features.nrows();

    // Accept either legacy per-PB mean latent (n_pb × K, reconstruct via β)
    // or the new pb_gene_mean (n_pb × n_genes, already log1p-CPM gene-space).
    enum PbMode {
        MeanLatent,
        GeneMean,
    }
    // When K ≠ n_genes the column count disambiguates cleanly. On the
    // rare collision (K == n_genes) prefer the gene-mean path: the new
    // viz output is the forward direction, and mean-latent is legacy.
    let pb_mode = if pb_features.ncols() == n_genes {
        if k_topics == n_genes {
            log::warn!(
                "pb-features has {} cols and K == n_genes; assuming gene-mean (log1p-CPM)",
                pb_features.ncols()
            );
        }
        PbMode::GeneMean
    } else if pb_features.ncols() == k_topics {
        PbMode::MeanLatent
    } else {
        anyhow::bail!(
            "pb-features has {} cols; expected K={} (mean-latent) or n_genes={} (gene-mean)",
            pb_features.ncols(),
            k_topics,
            n_genes
        );
    };
    info!(
        "Loaded β: {}×{}, latent: {}×{}, pb-features: {}×{} ({})",
        n_genes,
        k_topics,
        topic_nt.nrows(),
        k_topics,
        n_pb,
        pb_features.ncols(),
        match pb_mode {
            PbMode::MeanLatent => "mean-latent mode",
            PbMode::GeneMean => "gene-mean mode",
        }
    );

    // 2. Marker matrix (gene × celltype 0/1). Drop celltypes with zero
    //    matched markers (they can never claim an anchor).
    let AnnotInfo {
        membership_ga,
        annot_names: annot_names_raw,
    } = build_annotation_matrix(&args.marker_file, &gene_names)?;

    let non_empty: Vec<usize> = (0..membership_ga.ncols())
        .filter(|&c| membership_ga.column(c).iter().any(|&v| v > 0.0))
        .collect();
    let (mut membership_ga, annot_names): (Mat, Vec<Box<str>>) =
        if non_empty.len() < membership_ga.ncols() {
            info!(
                "Dropping {} celltypes with no matched markers",
                membership_ga.ncols() - non_empty.len()
            );
            (
                membership_ga.select_columns(&non_empty),
                non_empty
                    .iter()
                    .map(|&c| annot_names_raw[c].clone())
                    .collect(),
            )
        } else {
            (membership_ga, annot_names_raw)
        };
    info!("Markers: {} celltypes kept", annot_names.len());

    // 3. Gene × PB reconstruction, log1p-compressed, z-scored for GS.
    //    MeanLatent mode: β is stored as log-probs; load_dictionary exp's +
    //    column-normalizes so β·pb_mean_latentᵀ lives on the linear
    //    probability simplex — then log1p compresses.
    //    GeneMean mode: pb-features is already log1p-CPM in gene space, so
    //    we transpose and use directly; no β reconstruction.
    //    Per-column z-score puts each PB on equal footing and kills
    //    housekeeping rows.
    let x_gp: Mat = match pb_mode {
        PbMode::MeanLatent => {
            let gene_pb = &beta_prob * pb_features.transpose();
            let mut x = gene_pb;
            for v in x.as_mut_slice() {
                *v = v.max(0.0).ln_1p();
            }
            x
        }
        PbMode::GeneMean => pb_features.transpose(),
    };
    let x_zscored = zscore_columns(&x_gp);

    // 4. Gram-Schmidt on GENE rows. Default K_anchor = K (one anchor per
    //    independent dictionary direction).
    let requested = args.num_anchors.unwrap_or(k_topics);
    if requested > n_genes {
        log::warn!("Requested K_anchor={requested} > n_genes={n_genes}; clamping");
    }
    if requested > k_topics {
        log::info!(
            "K_anchor={requested} > K_topics={k_topics}; past K topics the Gram-Schmidt residuals \
             are ~0 (row space is rank-limited by β). Expect degenerate picks."
        );
    }
    let k_anchor = requested.min(n_genes);
    let anchor_gene_idx = gram_schmidt_anchors(&x_zscored, k_anchor);
    info!(
        "Selected {} anchor genes via Gram-Schmidt on z-scored gene × PB",
        anchor_gene_idx.len()
    );

    // 5. Label each anchor gene via marker dictionary lookup. Specificity =
    //    1 / n_claiming_celltypes (1.0 if exactly one celltype claims, 0.5
    //    if two, etc.). Among multi-claim genes we pick the celltype with
    //    the fewest total markers — the one for which this gene is a rarer
    //    claim, i.e. more diagnostic.
    let ct_sizes: Vec<usize> = (0..annot_names.len())
        .map(|c| membership_ga.column(c).iter().filter(|&&v| v > 0.0).count())
        .collect();
    let records = label_anchor_genes(
        &anchor_gene_idx,
        &gene_names,
        &membership_ga,
        &annot_names,
        &ct_sizes,
        &x_zscored,
        args.top_correlated,
    );

    // 6. Interactive / suggest / apply modes.
    let original_markers_vec = crate::marker_support::read_marker_gene_info(&args.marker_file)?;
    let marker_db = if let Some(db_path) = &args.marker_db {
        info!("Loading marker database from {db_path}...");
        Some(MarkerDatabase::load_with_vocab(
            db_path,
            &gene_names,
            &annot_names,
        )?)
    } else {
        None
    };

    if let Some(output_json) = &args.suggest_only {
        let labels = records.iter().map(|r| r.label.clone()).collect::<Vec<_>>();
        let pip_ka = make_anchor_to_celltype(&labels, &annot_names);
        let weights_gk = anchor_weight_matrix(&x_zscored, &anchor_gene_idx);
        let candidates = find_candidate_markers(
            &pip_ka.transpose(),
            &weights_gk,
            &membership_ga,
            &gene_names,
            &labels,
            &annot_names,
            args.interactive_min_pip,
            args.top_candidates,
            args.topics_per_celltype,
        );
        if let Some(db) = &marker_db {
            let auto = auto_suggest_markers(&candidates, db);
            if !auto.is_empty() {
                info!("Auto-accepted {} markers from database", auto.len());
                write_simple_tsv(
                    &format!("{}.auto_suggestions.tsv", args.out),
                    &["gene", "celltype"],
                    auto.iter().map(|(g, c)| vec![g.to_string(), c.to_string()]),
                )?;
            }
        }
        write_candidates_json(&candidates, output_json)?;
        info!("Wrote candidate markers to {output_json}");
        write_anchor_outputs(
            &args.out,
            &records,
            &annot_names,
            &gene_names,
            &membership_ga,
        )?;
        return Ok(());
    }

    // Apply-suggestions (optional) — augments membership_ga before labeling.
    let mut all_new_markers: Vec<(Box<str>, Box<str>)> = Vec::new();
    if let Some(suggestions_json) = &args.apply_suggestions {
        let suggestions = read_suggestions_json(suggestions_json)?;
        info!(
            "Applying {} suggestions from {}",
            suggestions.len(),
            suggestions_json
        );
        for (gene, celltype) in &suggestions {
            if let Some((g_idx, matched)) = gene_names
                .iter()
                .enumerate()
                .find(|(_, g)| flexible_gene_match(gene, g))
                .map(|(i, g)| (i, g.clone()))
            {
                if let Some(c_idx) = annot_names
                    .iter()
                    .position(|n| n.as_ref() == celltype.as_ref())
                {
                    membership_ga[(g_idx, c_idx)] = args.augment_weight;
                    all_new_markers.push((matched, celltype.clone()));
                }
            }
        }
    }

    let mut records = records;
    let mut n_iters = usize::from(!all_new_markers.is_empty());

    // Relabel whenever membership changes.
    let relabel = |mem: &Mat| {
        let sizes: Vec<usize> = (0..annot_names.len())
            .map(|c| mem.column(c).iter().filter(|&&v| v > 0.0).count())
            .collect();
        label_anchor_genes(
            &anchor_gene_idx,
            &gene_names,
            mem,
            &annot_names,
            &sizes,
            &x_zscored,
            args.top_correlated,
        )
    };

    if args.interactive {
        loop {
            n_iters += 1;
            info!("=== Interactive round {n_iters} ===");
            records = relabel(&membership_ga);

            let labels: Vec<Box<str>> = records.iter().map(|r| r.label.clone()).collect();
            let pip_ka = make_anchor_to_celltype(&labels, &annot_names);
            let weights_gk = anchor_weight_matrix(&x_zscored, &anchor_gene_idx);
            let candidates = find_candidate_markers(
                &pip_ka.transpose(),
                &weights_gk,
                &membership_ga,
                &gene_names,
                &labels,
                &annot_names,
                args.interactive_min_pip,
                args.top_candidates,
                args.topics_per_celltype,
            );
            if candidates.is_empty() {
                info!("No more candidates above threshold. Done.");
                break;
            }
            let result = run_interactive_round(&candidates, n_iters)?;
            if !result.proceed && result.new_markers.is_empty() {
                return Ok(());
            }
            if result.new_markers.is_empty() {
                info!("No new markers. Finalizing...");
                break;
            }
            augment_membership_matrix(
                &mut membership_ga,
                &gene_names,
                &annot_names,
                &result.new_markers,
                args.augment_weight,
            );
            all_new_markers.extend(result.new_markers);
            if !result.proceed {
                records = relabel(&membership_ga);
                break;
            }
        }
    } else if !all_new_markers.is_empty() {
        records = relabel(&membership_ga);
    }

    if !all_new_markers.is_empty() {
        let path = format!("{}.augmented_markers.tsv", args.out);
        save_augmented_markers(&original_markers_vec, &all_new_markers, &path)?;
        print_augmentation_summary(&all_new_markers, n_iters);
        info!("Saved augmented markers to {path}");
    }

    // 7. Anchor → celltype assignment with --min-pip filter.
    let labels: Vec<Box<str>> = records.iter().map(|r| r.label.clone()).collect();
    let mut assignment_ka = make_anchor_to_celltype(&labels, &annot_names);
    let mut excluded = Vec::new();
    for k in 0..assignment_ka.nrows() {
        let max_p = (0..assignment_ka.ncols())
            .map(|c| assignment_ka[(k, c)])
            .fold(0.0f32, f32::max);
        if max_p < args.min_pip {
            for c in 0..assignment_ka.ncols() {
                assignment_ka[(k, c)] = 0.0;
            }
            excluded.push((k, max_p));
        }
    }
    if !excluded.is_empty() {
        info!(
            "Excluded {} anchors below --min-pip={}",
            excluded.len(),
            args.min_pip
        );
    }

    let assignment_file = format!("{}.assignment.parquet", args.out);
    assignment_ka.to_parquet_with_names(
        &assignment_file,
        (Some(&labels), Some("anchor")),
        Some(&annot_names),
    )?;
    info!("Wrote anchor-celltype assignment to {assignment_file}");

    // 8. Cell → anchor via cosine(cell_latent, β[anchor_gene, :]).
    //    β-rows of anchor genes define each anchor's topic direction;
    //    inner product with cell latent ≈ reconstructed expression of
    //    that anchor gene in the cell.
    let anchor_beta_rows = beta_prob.select_rows(anchor_gene_idx.iter());
    let cell_to_anchor = soft_assign_cosine(&topic_nt, &anchor_beta_rows, args.temperature);

    // 9. Cell → celltype, row-normalized; rows of all-novel anchors stay 0.
    let mut cell_annot = &cell_to_anchor * &assignment_ka;
    for i in 0..cell_annot.nrows() {
        let s: f32 = cell_annot.row(i).iter().sum();
        if s > 1e-12 {
            for j in 0..cell_annot.ncols() {
                cell_annot[(i, j)] /= s;
            }
        }
    }
    let annot_file = format!("{}.annotation.parquet", args.out);
    cell_annot.to_parquet_with_names(
        &annot_file,
        (Some(&cell_names), Some("cell")),
        Some(&annot_names),
    )?;

    let argmax_file = format!("{}.argmax.tsv", args.out);
    write_argmax_assignments(&cell_annot, &cell_names, &annot_names, &argmax_file)?;
    display_annotation_histogram(&cell_annot, &annot_names);

    // 10. Side outputs: anchor genes + correlated genes.
    write_anchor_outputs(
        &args.out,
        &records,
        &annot_names,
        &gene_names,
        &membership_ga,
    )?;

    Ok(())
}

/// Build label + metadata for every selected anchor gene.
///
/// Label rule: the anchor gene is looked up in the marker matrix; if no
/// celltype claims it, the label is `novel_i`. If one or more celltypes
/// claim it, the one with the **fewest** total markers wins (more specific
/// / rarer claim), and multi-anchor collisions get `label`, `label_2`, …
/// via `base_celltype_label`-compatible suffixes.
fn label_anchor_genes(
    anchor_gene_idx: &[usize],
    gene_names: &[Box<str>],
    membership_ga: &Mat,
    annot_names: &[Box<str>],
    ct_sizes: &[usize],
    x_zscored: &Mat,
    top_correlated: usize,
) -> Vec<AnchorRecord> {
    let n_genes = gene_names.len();
    let n_ct = annot_names.len();

    // Row norms (L2) in x_zscored, used to compute cosine with the anchor.
    let row_norms: Vec<f32> = (0..n_genes)
        .map(|g| x_zscored.row(g).iter().map(|&v| v * v).sum::<f32>().sqrt())
        .collect();

    let mut used_counts = rustc_hash::FxHashMap::<Box<str>, usize>::default();
    let mut out: Vec<AnchorRecord> = Vec::with_capacity(anchor_gene_idx.len());

    for (i, &g) in anchor_gene_idx.iter().enumerate() {
        // Celltypes claiming this gene.
        let claims: Vec<usize> = (0..n_ct).filter(|&c| membership_ga[(g, c)] > 0.0).collect();

        let (label, primary) = if claims.is_empty() {
            (format!("{NOVEL_ANCHOR_PREFIX}{i}").into_boxed_str(), None)
        } else {
            // Most-specific = smallest celltype by marker count.
            let primary = *claims
                .iter()
                .min_by_key(|&&c| ct_sizes[c])
                .expect("non-empty");
            let base = annot_names[primary].clone();
            let n = used_counts.entry(base.clone()).or_insert(0);
            *n += 1;
            let lab = if *n == 1 {
                base
            } else {
                format!("{base}_{n}").into_boxed_str()
            };
            (lab, Some(primary))
        };

        // Correlated genes: cosine sim in x_zscored.
        let anchor_norm = row_norms[g];
        let anchor_row = x_zscored.row(g);
        let mut ranked: Vec<(usize, f32)> = (0..n_genes)
            .filter(|&j| j != g && row_norms[j] > 1e-12 && anchor_norm > 1e-12)
            .map(|j| {
                let dot: f32 = anchor_row
                    .iter()
                    .zip(x_zscored.row(j).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                (j, dot / (anchor_norm * row_norms[j]))
            })
            .collect();
        let top_k = top_correlated.min(ranked.len());
        if top_k > 0 && top_k < ranked.len() {
            let pivot = top_k - 1;
            ranked.select_nth_unstable_by(pivot, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(top_k);
        }
        ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        out.push(AnchorRecord {
            gene_idx: g,
            gene_name: gene_names[g].clone(),
            label,
            primary_celltype: primary,
            n_claiming: claims.len(),
            specificity: if claims.is_empty() {
                0.0
            } else {
                1.0 / claims.len() as f32
            },
            correlated: ranked,
        });
    }
    out
}

/// Build anchor × celltype 0/1 assignment matrix from labels. Novel anchors
/// contribute a zero row so cells routed entirely to them surface as
/// `unassigned` in the argmax.
fn make_anchor_to_celltype(labels: &[Box<str>], annot_names: &[Box<str>]) -> Mat {
    let mut out = Mat::zeros(labels.len(), annot_names.len());
    for (k, lab) in labels.iter().enumerate() {
        if lab.starts_with(NOVEL_ANCHOR_PREFIX) {
            continue;
        }
        let base = base_celltype_label(lab);
        if let Some(c) = annot_names.iter().position(|n| n.as_ref() == base) {
            out[(k, c)] = 1.0;
        }
    }
    out
}

/// (gene × anchor) weight matrix for `find_candidate_markers`. Each
/// column is the anchor gene's z-scored profile across PBs projected back
/// into gene space via cosine with every other gene. Higher = more likely
/// to be in the anchor's neighborhood.
fn anchor_weight_matrix(x_zscored: &Mat, anchor_gene_idx: &[usize]) -> Mat {
    let n_genes = x_zscored.nrows();
    let k = anchor_gene_idx.len();
    let mut out = Mat::zeros(n_genes, k);
    let row_norms: Vec<f32> = (0..n_genes)
        .map(|g| x_zscored.row(g).iter().map(|&v| v * v).sum::<f32>().sqrt())
        .collect();
    for (col, &g) in anchor_gene_idx.iter().enumerate() {
        let anchor_norm = row_norms[g].max(1e-12);
        let anchor_row = x_zscored.row(g);
        for j in 0..n_genes {
            let jn = row_norms[j].max(1e-12);
            let dot: f32 = anchor_row
                .iter()
                .zip(x_zscored.row(j).iter())
                .map(|(a, b)| a * b)
                .sum();
            out[(j, col)] = dot / (anchor_norm * jn);
        }
    }
    out
}

/// Unit-row cosine × softmax for cell → anchor soft assignment.
fn soft_assign_cosine(cell_z: &Mat, anchor_z: &Mat, temperature: f32) -> Mat {
    let mut cell_u = cell_z.clone();
    for mut row in cell_u.row_iter_mut() {
        let s = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if s > 1e-12 {
            row /= s;
        }
    }
    let mut anchor_u = anchor_z.clone();
    for mut row in anchor_u.row_iter_mut() {
        let s = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if s > 1e-12 {
            row /= s;
        }
    }
    let sim = &cell_u * anchor_u.transpose();
    let beta = 1.0 / temperature.max(1e-6);
    let mut out = Mat::zeros(sim.nrows(), sim.ncols());
    for i in 0..sim.nrows() {
        let row = sim.row(i);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..sim.ncols() {
            let e = ((row[j] - max) * beta).exp();
            out[(i, j)] = e;
            sum += e;
        }
        if sum > 1e-12 {
            for j in 0..sim.ncols() {
                out[(i, j)] /= sum;
            }
        }
    }
    out
}

/// Write `{out}.anchor_genes.tsv` (one row per anchor) and
/// `{out}.correlated_genes.tsv` (long format: anchor × correlated-gene).
fn write_anchor_outputs(
    out_prefix: &str,
    records: &[AnchorRecord],
    annot_names: &[Box<str>],
    gene_names: &[Box<str>],
    membership_ga: &Mat,
) -> anyhow::Result<()> {
    use std::io::Write;

    // Main anchor table.
    let anchors_path = format!("{out_prefix}.anchor_genes.tsv");
    let mut f = std::fs::File::create(&anchors_path)?;
    writeln!(
        f,
        "anchor_idx\tgene_idx\tgene\tlabel\tprimary_celltype\tn_celltypes_claiming\tspecificity"
    )?;
    for (i, r) in records.iter().enumerate() {
        let primary = r.primary_celltype.map_or("-", |c| annot_names[c].as_ref());
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}",
            i, r.gene_idx, r.gene_name, r.label, primary, r.n_claiming, r.specificity
        )?;
    }
    info!("wrote {anchors_path}");

    // Correlated genes (long format).
    let corr_path = format!("{out_prefix}.correlated_genes.tsv");
    let mut f = std::fs::File::create(&corr_path)?;
    writeln!(
        f,
        "anchor_idx\tanchor_gene\tcorrelated_gene\tcosine\tin_user_markers"
    )?;
    for (i, r) in records.iter().enumerate() {
        for &(g, sim) in &r.correlated {
            let in_user = (0..membership_ga.ncols()).any(|c| membership_ga[(g, c)] > 0.0);
            writeln!(
                f,
                "{}\t{}\t{}\t{:.4}\t{}",
                i,
                r.gene_name,
                gene_names[g],
                sim,
                if in_user { "yes" } else { "no" }
            )?;
        }
    }
    info!("wrote {corr_path}");
    Ok(())
}

fn write_simple_tsv<I, R>(path: &str, header: &[&str], rows: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = R>,
    R: IntoIterator<Item = String>,
{
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{}", header.join("\t"))?;
    for row in rows {
        let cells: Vec<String> = row.into_iter().collect();
        writeln!(f, "{}", cells.join("\t"))?;
    }
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
    for (i, cell_name) in cell_names.iter().enumerate().take(annot.nrows()) {
        let row = annot.row(i);
        let sum: f32 = row.iter().sum();
        if sum < 1e-12 {
            writeln!(file, "{cell_name}\tunassigned\t0.0000")?;
            continue;
        }
        let (max_idx, max_val) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        writeln!(
            file,
            "{}\t{}\t{:.4}",
            cell_name, annot_names[max_idx], max_val
        )?;
    }
    info!("Wrote argmax assignments to {output_file}");
    Ok(())
}

fn display_annotation_histogram(annot: &Mat, annot_names: &[Box<str>]) {
    let n_cells = annot.nrows();
    let n_types = annot.ncols();
    let mut max_probs = Vec::with_capacity(n_cells);
    let mut assignments: Vec<Option<usize>> = Vec::with_capacity(n_cells);
    let mut unassigned = 0usize;

    for i in 0..n_cells {
        let row = annot.row(i);
        let sum: f32 = row.iter().sum();
        if sum < 1e-12 {
            max_probs.push(0.0);
            assignments.push(None);
            unassigned += 1;
            continue;
        }
        let (max_idx, max_val) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        max_probs.push(*max_val);
        assignments.push(Some(max_idx));
    }

    let mut type_counts = vec![0usize; n_types];
    let mut type_prob_sum = vec![0.0f32; n_types];
    for (i, ct) in assignments.iter().enumerate() {
        if let Some(c) = ct {
            type_counts[*c] += 1;
            type_prob_sum[*c] += max_probs[i];
        }
    }

    let mut sorted_types: Vec<usize> = (0..n_types).collect();
    sorted_types.sort_by(|&a, &b| type_counts[b].cmp(&type_counts[a]));

    let max_count = *type_counts.iter().max().unwrap_or(&1).max(&unassigned);
    const MAX_BAR: usize = 20;

    let assigned_cells = n_cells - unassigned;
    let mean_prob = if assigned_cells > 0 {
        max_probs
            .iter()
            .zip(assignments.iter())
            .filter_map(|(p, a)| a.map(|_| *p))
            .sum::<f32>()
            / assigned_cells as f32
    } else {
        0.0
    };
    let above_50 = max_probs.iter().filter(|&&x| x > 0.5).count();
    let above_70 = max_probs.iter().filter(|&&x| x > 0.7).count();

    eprintln!();
    eprintln!("Annotation Summary ({n_cells} cells)");
    eprintln!(
        "  Mean max-prob (assigned): {:.3}  >0.5: {} ({:.1}%)  >0.7: {} ({:.1}%)",
        mean_prob,
        above_50,
        100.0 * above_50 as f32 / n_cells as f32,
        above_70,
        100.0 * above_70 as f32 / n_cells as f32
    );
    if unassigned > 0 {
        let bar_len = (unassigned * MAX_BAR) / max_count.max(1);
        eprintln!(
            "  {:20} {:5} ({:5.1}%)      {}",
            "unassigned",
            unassigned,
            100.0 * unassigned as f32 / n_cells as f32,
            "▒".repeat(bar_len)
        );
    }
    eprintln!();

    for &ct in &sorted_types {
        if type_counts[ct] == 0 {
            continue;
        }
        let bar_len = (type_counts[ct] * MAX_BAR) / max_count.max(1);
        let bar: String = "█".repeat(bar_len);
        let avg_prob = type_prob_sum[ct] / type_counts[ct] as f32;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 orthogonal gene blocks, 3 celltypes each marked by a distinct
    /// gene. GS on (G × n_pb) must pick one gene per block; each must
    /// get its celltype's label; cell annotation row sums ∈ {0, 1}.
    #[test]
    fn gene_vertex_recovery() {
        let k = 3;
        let n_pb = 6;
        let n_cells = 9;
        let g_per_block = 4;
        let n_genes = k * g_per_block;

        let mut beta = Mat::zeros(n_genes, k);
        for t in 0..k {
            for g in 0..g_per_block {
                beta[(t * g_per_block + g, t)] = 1.0 / g_per_block as f32;
            }
        }
        let mut pb_latent = Mat::zeros(n_pb, k);
        for t in 0..k {
            pb_latent[(t, t)] = 1.0;
        }
        for row in 3..n_pb {
            for t in 0..k {
                pb_latent[(row, t)] = 1.0 / k as f32;
            }
        }
        let mut cell_latent = Mat::zeros(n_cells, k);
        for c in 0..n_cells {
            cell_latent[(c, c / 3)] = 1.0;
        }

        let celltypes: Vec<Box<str>> = vec!["Alpha".into(), "Beta".into(), "Gamma".into()];
        // Markers: every gene in block `t` claimed by celltype `t`, so the
        // test doesn't depend on which block-member GS picks.
        let mut mem_gc = Mat::zeros(n_genes, k);
        for t in 0..k {
            for g in 0..g_per_block {
                mem_gc[(t * g_per_block + g, t)] = 1.0;
            }
        }

        // Replicate the gene-vertex pipeline.
        let gene_pb = &beta * pb_latent.transpose();
        let mut x_gp = gene_pb;
        for v in x_gp.as_mut_slice() {
            *v = v.max(0.0).ln_1p();
        }
        let x_zscored = zscore_columns(&x_gp);
        let idx = gram_schmidt_anchors(&x_zscored, k);
        assert_eq!(idx.len(), k);

        // Each picked gene must come from a different block (0..4, 4..8, 8..12).
        let blocks: std::collections::HashSet<usize> =
            idx.iter().map(|&g| g / g_per_block).collect();
        assert_eq!(
            blocks.len(),
            k,
            "expected one gene per block, got {:?} → blocks {:?}",
            idx,
            blocks
        );

        let ct_sizes = vec![1usize; k];
        let records = label_anchor_genes(
            &idx,
            &(0..n_genes)
                .map(|g| format!("gene_{}", g).into_boxed_str())
                .collect::<Vec<_>>(),
            &mem_gc,
            &celltypes,
            &ct_sizes,
            &x_zscored,
            3,
        );
        // Every anchor gene is the marker gene for its block, so every
        // anchor must be labeled (not novel).
        for r in &records {
            assert!(
                !r.label.starts_with(NOVEL_ANCHOR_PREFIX),
                "anchor gene {} got novel label",
                r.gene_name
            );
            assert_eq!(
                r.n_claiming, 1,
                "each marker claimed by exactly one celltype"
            );
        }

        // Cell → anchor → celltype.
        let anchor_beta = beta.select_rows(idx.iter());
        let cta = soft_assign_cosine(&cell_latent, &anchor_beta, 0.1);
        let labels: Vec<Box<str>> = records.iter().map(|r| r.label.clone()).collect();
        let assign_ka = make_anchor_to_celltype(&labels, &celltypes);
        let mut cell_annot = &cta * &assign_ka;
        for i in 0..cell_annot.nrows() {
            let s: f32 = cell_annot.row(i).iter().sum();
            if s > 1e-12 {
                for j in 0..cell_annot.ncols() {
                    cell_annot[(i, j)] /= s;
                }
            }
        }
        for i in 0..cell_annot.nrows() {
            let s: f32 = cell_annot.row(i).iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-4 || s.abs() < 1e-4,
                "row {} sum {} ∉ {{0, 1}}",
                i,
                s
            );
            assert!(
                cell_annot.row(i).iter().all(|&v| !v.is_nan()),
                "row {} has NaN",
                i
            );
        }
    }
}
