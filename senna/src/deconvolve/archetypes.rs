//! Empirical archetype reference: many fine-grained profiles collapsed from
//! annotated cells, plus a soft map from archetype to reported cell type.
//!
//! The low-rank reference reconstructs one profile per cell type from the gene
//! embedding. That reconstruction is rank-limited and cannot match a real bulk
//! profile even when the anchor sits exactly on the true cell-type centroid, so
//! it caps how well any composition can fit. Here the profiles are measured
//! rather than reconstructed, which removes that ceiling at the cost of needing
//! the single-cell counts at deconvolution time.
//!
//! Granularity matters. Collapsing straight to cell types would reproduce the
//! same problem in a different form — one mean profile per type is a poor
//! description of a real tissue. Leiden communities on the cell embedding give
//! several hundred sub-type-resolution profiles, each still dense enough to
//! estimate, and the annotation posterior carries each community's label
//! uncertainty into the readout instead of forcing a hard label.

use super::args::ArchetypeConfig;
use super::reference::Reference;
use super::source::EmbeddingSource;
use crate::cluster::leiden_clustering;
use crate::cluster_aggregation::{accumulate_gene_sum, weighted_mean_profile};
use crate::embed_common::Mat;
use crate::senna_input::{read_data_on_shared_columns, ReadSharedColumnsArgs};
use anyhow::{Context, Result};
use log::{info, warn};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::{IoOps, MatWithNames};
use rustc_hash::{FxHashMap, FxHashSet};

/// Neighbours used by the Leiden kNN graph over the cell embedding.
const LEIDEN_KNN: usize = 30;
/// Starting resolution; the target archetype count drives a binary search from here.
const LEIDEN_RESOLUTION: f64 = 1.0;
/// Column-block size for the streaming gene-sum accumulation.
const BLOCK_SIZE: usize = 1000;

/// Build the archetype reference, aligned to `genes` (the deconvolution's gene panel).
pub fn build(
    cfg: &ArchetypeConfig<'_>,
    src: &EmbeddingSource,
    genes: &[Box<str>],
) -> Result<Reference> {
    let embedding = load_cell_embedding(src)?;
    let annotation = load_annotation(cfg, src)?;

    let data_files: Vec<Box<str>> = if cfg.sc_data.is_empty() {
        anyhow::ensure!(
            !src.data_files.is_empty(),
            "deconvolve --reference archetype needs the single-cell counts: pass `--sc-data`, \
             or use a `--from` manifest that records its input data"
        );
        src.data_files.clone()
    } else {
        cfg.sc_data.to_vec()
    };
    let stack = read_data_on_shared_columns(ReadSharedColumnsArgs {
        data_files,
        batch_files: None,
        num_types: 1,
        preload: false,
        qc: None,
        qc_block_size: None,
        qc_report_out: None,
    })?;
    anyhow::ensure!(
        stack.data_stack.stack.len() == 1,
        "deconvolve: expected a single data stack, got {}",
        stack.data_stack.stack.len()
    );
    let data_vec = &stack.data_stack.stack[0];
    let cell_names = data_vec.column_names()?;
    let sc_genes = data_vec.row_names()?;
    info!(
        "deconvolve archetypes: counts {} genes × {} cells",
        sc_genes.len(),
        cell_names.len()
    );

    // Cells must line up across the counts, the embedding and the annotation.
    // Anything missing from either table is dropped rather than defaulted: a
    // cell with no embedding has no position, and one with no annotation has no
    // label to contribute.
    let emb_index: FxHashMap<&str, usize> = embedding
        .rows
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), i))
        .collect();
    let ann_index: FxHashMap<&str, usize> = annotation
        .rows
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_ref(), i))
        .collect();
    let keep: Option<FxHashSet<Box<str>>> = match cfg.cells {
        Some(path) => Some(read_cell_list(path)?),
        None => None,
    };

    let mut rows = Vec::with_capacity(cell_names.len());
    for (ci, name) in cell_names.iter().enumerate() {
        if keep.as_ref().is_some_and(|k| !k.contains(name)) {
            continue;
        }
        if let (Some(&ei), Some(&ai)) = (emb_index.get(name.as_ref()), ann_index.get(name.as_ref()))
        {
            rows.push((ci, ei, ai));
        }
    }
    anyhow::ensure!(
        !rows.is_empty(),
        "deconvolve archetypes: no cell name is shared by the counts, the embedding and the \
         annotation — check that they come from the same run"
    );
    if rows.len() < cell_names.len() {
        info!(
            "deconvolve archetypes: using {}/{} cells present in counts, embedding and annotation",
            rows.len(),
            cell_names.len()
        );
    }

    // Cluster the retained cells in the embedding.
    let h = embedding.mat.ncols();
    let sub = Mat::from_fn(rows.len(), h, |i, j| embedding.mat[(rows[i].1, j)]);
    let target = cfg.target.clamp(1, rows.len());
    let clustered = leiden_clustering(
        &sub,
        LEIDEN_KNN,
        LEIDEN_RESOLUTION,
        Some(target),
        Some(cfg.seed),
    )?;
    let labels_sub = merge_small(
        &clustered.labels,
        clustered.n_clusters,
        &sub,
        cfg.min_cells.max(1),
    );
    let n_arch = labels_sub.iter().copied().max().map_or(0, |m| m + 1);
    anyhow::ensure!(
        n_arch > 0,
        "deconvolve archetypes: clustering produced none"
    );
    info!(
        "deconvolve archetypes: {n_arch} archetypes over {} cells (target {target})",
        rows.len()
    );

    // Per-archetype gene sums over the full cell set, with cells outside the
    // retained subset parked in a sentinel group that is dropped afterwards.
    let mut labels_all = vec![n_arch; cell_names.len()];
    for (i, &(ci, _, _)) in rows.iter().enumerate() {
        labels_all[ci] = labels_sub[i];
    }
    let gene_sum = accumulate_gene_sum(
        data_vec,
        &labels_all,
        n_arch + 1,
        sc_genes.len(),
        BLOCK_SIZE,
    )?;
    let profiles = weighted_mean_profile(&gene_sum, n_arch + 1, sc_genes.len(), &[]);

    // Align to the deconvolution gene panel, shrink, renormalise.
    let xbar = align_and_shrink(&profiles, &sc_genes, genes, n_arch, cfg.shrink)?;

    // Archetype coordinates and the soft readout are cell means within a community.
    let (coords, readout) = summarize(&rows, &labels_sub, n_arch, &embedding, &annotation, h);

    // A cell type with no archetype behind it can never be reported, whatever
    // the bulk contains. That happens quietly when `--archetype-cells` selects a
    // skewed subset, so name the types rather than returning zeros for them.
    let missing: Vec<&str> = annotation
        .cols
        .iter()
        .enumerate()
        .filter(|&(ct, _)| (0..n_arch).all(|m| readout[(m, ct)] <= 0.0))
        .map(|(_, name)| name.as_ref())
        .collect();
    if !missing.is_empty() {
        warn!(
            "deconvolve archetypes: no reference cells for {} of {} cell types ({}); those \
             types cannot be called and their share will be redistributed over the rest",
            missing.len(),
            annotation.cols.len(),
            missing.join(", ")
        );
    }

    let mu_gm = gene_major(&xbar, genes.len(), n_arch);
    let comp_names: Vec<Box<str>> = (0..n_arch)
        .map(|m| format!("archetype{m:04}").into_boxed_str())
        .collect();

    Ok(Reference {
        mu_gm,
        n_genes: genes.len(),
        n_comp: n_arch,
        readout,
        coords,
        comp_names,
        celltype_names: annotation.cols.clone(),
    })
}

/// Pick the manifest's cell embedding whose width matches the gene embedding.
fn load_cell_embedding(src: &EmbeddingSource) -> Result<MatWithNames<Mat>> {
    anyhow::ensure!(
        !src.cell_embedding_paths.is_empty(),
        "deconvolve --reference archetype needs a cell embedding, but the `--from` manifest \
         records neither `cell_embedding` nor `latent`"
    );
    let mut last_err = None;
    for path in &src.cell_embedding_paths {
        match DMatrix::<f32>::from_parquet_with_row_names(path, Some(0)) {
            Ok(m) if m.mat.ncols() == src.h => {
                info!(
                    "deconvolve archetypes: cell embedding {path} [{} cells × {}]",
                    m.mat.nrows(),
                    m.mat.ncols()
                );
                return Ok(m);
            }
            Ok(m) => {
                // A topic `latent` sits beside the embedding and has K columns,
                // not H. Silently clustering it would put archetypes in a space
                // the gene profiles know nothing about.
                warn!(
                    "deconvolve archetypes: skipping {path} (width {} != embedding H={})",
                    m.mat.ncols(),
                    src.h
                );
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        anyhow::anyhow!(
            "deconvolve archetypes: no cell embedding of width {} among {:?}",
            src.h,
            src.cell_embedding_paths
        )
    }))
}

/// Load the `N×C` soft annotation, accepting either annotate layout.
///
/// The manifest's annotation slot is overloaded: the enrichment path records an
/// `N×C` posterior there, while the marker-bootstrap path records a tidy
/// per-cell table with string columns under the same key. Reading whichever the
/// manifest names would therefore silently work on one kind of run and fail on
/// the other, so candidates are tried in order and each is validated as a
/// numeric posterior before being accepted.
fn load_annotation(cfg: &ArchetypeConfig<'_>, src: &EmbeddingSource) -> Result<MatWithNames<Mat>> {
    let mut candidates: Vec<String> = Vec::new();
    if let Some(explicit) = cfg.annotation {
        candidates.push(explicit.to_string());
    } else if let Some(recorded) = src.annotation_path.as_deref() {
        // Sibling tables of the same annotate run, most informative first.
        let stem = recorded
            .strip_suffix(".annotation.parquet")
            .or_else(|| recorded.strip_suffix(".annot.parquet"))
            .unwrap_or(recorded);
        candidates.push(format!("{stem}.label_stability.parquet"));
        candidates.push(format!("{stem}.annotation.parquet"));
        candidates.push(recorded.to_string());
    }
    anyhow::ensure!(
        !candidates.is_empty(),
        "deconvolve --reference archetype needs a cell annotation: pass `--annotation`, \
         or use a `--from` manifest that records one"
    );

    let mut rejected = Vec::new();
    for path in &candidates {
        if !std::path::Path::new(path).exists() {
            continue;
        }
        match DMatrix::<f32>::from_parquet_with_row_names(path, Some(0)) {
            Ok(ann) if is_posterior(&ann) => {
                info!(
                    "deconvolve archetypes: annotation {path} [{} cells × {} types]",
                    ann.mat.nrows(),
                    ann.mat.ncols()
                );
                return Ok(ann);
            }
            Ok(_) => rejected.push(format!("{path} (not a numeric cell × celltype table)")),
            Err(e) => rejected.push(format!("{path} ({e})")),
        }
    }
    anyhow::bail!(
        "deconvolve: no usable cell × celltype annotation. Tried: {}. Pass `--annotation` \
         pointing at a soft posterior or label-stability table.",
        if rejected.is_empty() {
            candidates.join(", ")
        } else {
            rejected.join("; ")
        }
    )
}

/// A usable annotation is non-negative, finite, and assigns most cells somewhere.
fn is_posterior(ann: &MatWithNames<Mat>) -> bool {
    if ann.mat.nrows() == 0 || ann.mat.ncols() < 2 {
        return false;
    }
    if ann.mat.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return false;
    }
    let assigned = (0..ann.mat.nrows())
        .filter(|&i| ann.mat.row(i).sum() > 0.0)
        .count();
    assigned * 2 >= ann.mat.nrows()
}

fn read_cell_list(path: &str) -> Result<FxHashSet<Box<str>>> {
    let lines = matrix_util::common_io::read_lines(path)
        .with_context(|| format!("reading cell list {path}"))?;
    let set: FxHashSet<Box<str>> = lines
        .into_iter()
        .map(|l| l.trim().to_string().into_boxed_str())
        .filter(|l| !l.is_empty())
        .collect();
    anyhow::ensure!(!set.is_empty(), "deconvolve: cell list {path} is empty");
    info!(
        "deconvolve archetypes: restricted to {} named cells",
        set.len()
    );
    Ok(set)
}

/// Fold communities below `min_cells` into the nearest surviving centroid, then
/// renumber densely. A tiny community would otherwise carry a profile estimated
/// from a handful of cells and act as a sink for whatever the others cannot fit.
fn merge_small(labels: &[usize], n_clusters: usize, emb: &Mat, min_cells: usize) -> Vec<usize> {
    let h = emb.ncols();
    let mut sizes = vec![0usize; n_clusters];
    for &l in labels {
        if l < n_clusters {
            sizes[l] += 1;
        }
    }
    let mut centroid = vec![0f32; n_clusters * h];
    for (i, &l) in labels.iter().enumerate() {
        if l >= n_clusters {
            continue;
        }
        for j in 0..h {
            centroid[l * h + j] += emb[(i, j)];
        }
    }
    for l in 0..n_clusters {
        if sizes[l] > 0 {
            for j in 0..h {
                centroid[l * h + j] /= sizes[l] as f32;
            }
        }
    }
    let mut survivors: Vec<usize> = (0..n_clusters).filter(|&l| sizes[l] >= min_cells).collect();
    if survivors.is_empty() {
        // Asking for more archetypes than the cells can support: every community
        // is under the floor. Keeping them all would hand the sampler profiles
        // estimated from a handful of cells each, which is worse than a coarser
        // partition, so fall back to the largest communities the floor allows.
        let budget = (labels.len() / min_cells).max(1);
        let mut by_size: Vec<usize> = (0..n_clusters).collect();
        by_size.sort_unstable_by_key(|&l| std::cmp::Reverse(sizes[l]));
        by_size.truncate(budget);
        by_size.sort_unstable();
        warn!(
            "deconvolve archetypes: no community reaches {min_cells} cells; {} cells cannot \
             support {n_clusters} archetypes. Falling back to {} — lower `--archetypes`.",
            labels.len(),
            by_size.len()
        );
        survivors = by_size;
    }
    let mut remap = vec![0usize; n_clusters];
    let dense: FxHashMap<usize, usize> = survivors
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new))
        .collect();
    for l in 0..n_clusters {
        if let Some(&new) = dense.get(&l) {
            remap[l] = new;
            continue;
        }
        let mut best = (f32::INFINITY, 0usize);
        for &sv in &survivors {
            let mut d2 = 0f32;
            for j in 0..h {
                let diff = centroid[l * h + j] - centroid[sv * h + j];
                d2 += diff * diff;
            }
            if d2 < best.0 {
                best = (d2, dense[&sv]);
            }
        }
        remap[l] = best.1;
    }
    if survivors.len() < n_clusters {
        info!(
            "deconvolve archetypes: merged {} communities below {min_cells} cells",
            n_clusters - survivors.len()
        );
    }
    labels
        .iter()
        .map(|&l| remap[l.min(n_clusters - 1)])
        .collect()
}

/// Reindex profiles onto the deconvolution gene panel, shrink each toward the
/// pooled profile, and renormalise columns to sum to 1.
///
/// Shrinkage is what keeps every rate strictly positive: a gene absent from an
/// archetype would otherwise have rate zero, and a bulk count on that gene would
/// have nowhere to be allocated.
fn align_and_shrink(
    profiles: &Mat,
    sc_genes: &[Box<str>],
    genes: &[Box<str>],
    n_arch: usize,
    shrink: f32,
) -> Result<Mat> {
    use auxiliary_data::feature_names::FeatureNameKind;

    // Reconcile through the shared canonicalizer rather than string equality,
    // for the same reason the bulk loader does: the naming signature usually
    // lives on one axis only, so detect per axis and adopt whichever is
    // informative. Canonicalizing under `Gene` is a no-op for names without the
    // delimiter, so adopting the informative side is safe for both.
    let ref_kind = FeatureNameKind::auto_detect(genes);
    let sc_kind = FeatureNameKind::auto_detect(sc_genes);
    let name_kind = if ref_kind.is_exact() {
        sc_kind
    } else {
        ref_kind
    };

    let mut index: FxHashMap<Box<str>, usize> = FxHashMap::default();
    for (i, n) in sc_genes.iter().enumerate() {
        index.entry(name_kind.canonicalize(n)).or_insert(i);
    }
    let d = genes.len();
    let mut out = Mat::zeros(d, n_arch);
    let mut matched = 0usize;
    let mut pooled = vec![0f32; d];
    for (g, name) in genes.iter().enumerate() {
        let Some(&sg) = index.get(&name_kind.canonicalize(name)) else {
            continue;
        };
        matched += 1;
        for m in 0..n_arch {
            let v = profiles[(sg, m)];
            out[(g, m)] = v;
            pooled[g] += v;
        }
    }
    // Nothing matching means the counts are not the ones the embedding was
    // trained on, or the two axes name genes differently. Either way the run
    // would silently produce a flat reference, so fail with both sides shown.
    anyhow::ensure!(
        matched > 0,
        "deconvolve archetypes: none of the {d} reference genes appear in the single-cell \
         counts ({} rows, rule {name_kind:?}). Reference starts `{}`, counts start `{}` — \
         check that `--sc-data` is the matrix the embedding was trained on.",
        sc_genes.len(),
        genes.first().map_or("", |g| g.as_ref()),
        sc_genes.first().map_or("", |g| g.as_ref())
    );
    if matched * 2 < d {
        warn!(
            "deconvolve archetypes: only {matched}/{d} reference genes ({:.1}%) found in the \
             single-cell counts; the rest carry only the shrinkage floor",
            100.0 * matched as f64 / d as f64
        );
    } else if matched < d {
        info!("deconvolve archetypes: matched {matched}/{d} reference genes");
    }
    let pooled_total: f32 = pooled.iter().sum();
    if pooled_total > 0.0 {
        for p in &mut pooled {
            *p /= pooled_total;
        }
    }
    for m in 0..n_arch {
        let mut total = 0f32;
        for g in 0..d {
            let v = out[(g, m)] + shrink * pooled[g];
            out[(g, m)] = v;
            total += v;
        }
        if total > 0.0 {
            for g in 0..d {
                out[(g, m)] /= total;
            }
        }
    }
    Ok(out)
}

/// Community means of the embedding (coordinates) and of the annotation (readout).
fn summarize(
    rows: &[(usize, usize, usize)],
    labels: &[usize],
    n_arch: usize,
    embedding: &MatWithNames<Mat>,
    annotation: &MatWithNames<Mat>,
    h: usize,
) -> (Mat, Mat) {
    let c = annotation.mat.ncols();
    let mut coords = Mat::zeros(n_arch, h);
    let mut readout = Mat::zeros(n_arch, c);
    let mut counts = vec![0f32; n_arch];
    for (i, &(_, ei, ai)) in rows.iter().enumerate() {
        let m = labels[i];
        counts[m] += 1.0;
        for j in 0..h {
            coords[(m, j)] += embedding.mat[(ei, j)];
        }
        for ct in 0..c {
            readout[(m, ct)] += annotation.mat[(ai, ct)];
        }
    }
    for m in 0..n_arch {
        if counts[m] > 0.0 {
            for j in 0..h {
                coords[(m, j)] /= counts[m];
            }
        }
        // Rows must sum to 1 so that fractions sum to 1; a row of zeros (an
        // archetype whose cells are all unannotated) is left as is and simply
        // contributes nothing to any type.
        let total: f32 = (0..c).map(|ct| readout[(m, ct)]).sum();
        if total > 0.0 {
            for ct in 0..c {
                readout[(m, ct)] /= total;
            }
        }
    }
    (coords, readout)
}

/// `D×R` column-major profiles into the gene-major layout the sampler reads.
fn gene_major(xbar: &Mat, d: usize, r: usize) -> Vec<f32> {
    let mut mu = vec![0f32; d * r];
    for g in 0..d {
        for m in 0..r {
            mu[g * r + m] = xbar[(g, m)];
        }
    }
    mu
}
