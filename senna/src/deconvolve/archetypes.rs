//! Empirical archetype reference: many fine-grained profiles collapsed from
//! annotated cells, plus a soft map from archetype to reported cell type.
//!
//! Profiles are measured rather than reconstructed. Reconstructing one profile
//! per cell type from the gene embedding is rank-limited: it cannot match a real
//! bulk profile even with the anchor exactly on the true cell-type centroid, and
//! that caps how well any composition can fit. Measuring instead removes the
//! ceiling, at the cost of needing the single-cell counts at deconvolution time.
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
use crate::cluster_aggregation::accumulate_gene_sum_multi;
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

/// One reference cell, indexed into each of the three tables it must appear in.
struct CellRow {
    counts: usize,
    annotation: usize,
}

/// Everything the archetype reference needs that does not depend on how finely
/// the cells are partitioned. Loaded once and shared by every granularity, so
/// pooling over partitions costs one pass over the counts rather than one per
/// partition — the column reads are the raw matrix off disk and dominate.
struct ArchetypeInputs {
    stack: crate::senna_input::SparseStackWithBatch,
    annotation: MatWithNames<Mat>,
    /// Cells shared by counts, embedding and annotation, after `--archetype-cells`.
    rows: Vec<CellRow>,
    /// `rows.len() × H` embedding of exactly those cells, for clustering.
    sub: Mat,
    /// Panel gene → row in the single-cell counts, reconciled by name once.
    /// Panel gene → the single-cell rows that canonicalize to it, as CSR.
    ///
    /// A `Vec<Vec<_>>` here is one heap allocation per gene — tens of thousands
    /// of tiny blocks for a handful of real many-to-one collapses.
    panel_to_sc: GeneMap,
    n_sc_genes: usize,
    /// Column names of the count matrix, carried for the membership output.
    cell_names: Vec<Box<str>>,
}

/// Which cell landed in which archetype, per pooled partition.
///
/// Emitted so the readout can be audited against a known composition: without
/// the membership there is no way to evaluate what the model would report at a
/// *perfect* abundance vector, and therefore no way to tell reference error
/// apart from sampler error.
pub struct Membership {
    pub cell_names: Vec<Box<str>>,
    /// Per chain, the archetype of each cell (`usize::MAX` where the cell was
    /// not used — outside `--archetype-cells`, or missing from a table).
    pub per_chain: Vec<Vec<usize>>,
}

/// Build one archetype reference per target granularity.
///
/// The granularity is a nuisance parameter, so callers pool several; this
/// returns them together precisely so the expensive chain-invariant work — the
/// parquet loads, the cell alignment and the streaming gene sums — happens once.
pub fn build_all(
    cfg: &ArchetypeConfig<'_>,
    src: &EmbeddingSource,
    genes: &[Box<str>],
    targets: &[usize],
) -> Result<(Vec<Reference>, Membership)> {
    anyhow::ensure!(!targets.is_empty(), "deconvolve archetypes: no targets");
    let inputs = ArchetypeInputs::load(cfg, src, genes)?;

    // Partition first, then aggregate every partition in a single sweep.
    let partitions: Vec<Vec<usize>> = targets
        .iter()
        .enumerate()
        .map(|(chain, &target)| {
            inputs.partition(
                target,
                cfg.seed
                    .wrapping_add((chain as u64).wrapping_mul(super::SEED_STRIDE)),
                cfg.min_cells,
            )
        })
        .collect::<Result<_>>()?;

    let groupings: Vec<(&[usize], usize)> = partitions
        .iter()
        .map(|labels| {
            // Excluded cells carry `usize::MAX`, which the aggregator skips as
            // out of range — there is no sentinel group to allocate or drop.
            let n_arch = labels
                .iter()
                .filter(|&&l| l != usize::MAX)
                .max()
                .map_or(0, |m| m + 1);
            (labels.as_slice(), n_arch)
        })
        .collect();
    let sums = accumulate_gene_sum_multi(
        &inputs.stack.data_stack.stack[0],
        &groupings,
        inputs.n_sc_genes,
        BLOCK_SIZE,
    )?;

    let references: Vec<Reference> = partitions
        .iter()
        .zip(&groupings)
        .zip(sums)
        .map(|((labels, &(_, k)), gene_sum)| {
            inputs.assemble(labels, k, &gene_sum, genes, cfg.shrink)
        })
        .collect::<Result<_>>()?;

    let membership = Membership {
        cell_names: inputs.cell_names,
        per_chain: partitions,
    };
    Ok((references, membership))
}

impl ArchetypeInputs {
    fn load(cfg: &ArchetypeConfig<'_>, src: &EmbeddingSource, genes: &[Box<str>]) -> Result<Self> {
        let embedding = load_cell_embedding(src)?;
        let annotation = load_annotation(cfg, src)?;

        let data_files: Vec<Box<str>> = if cfg.sc_data.is_empty() {
            anyhow::ensure!(
                !src.data_files.is_empty(),
                "deconvolve needs the single-cell counts the profiles are measured from: \
                 pass `--sc-data`, or use a `--from` manifest that records its input data"
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
        let (cell_names, sc_genes) = {
            let dv = &stack.data_stack.stack[0];
            (dv.column_names()?, dv.row_names()?)
        };
        info!(
            "deconvolve archetypes: counts {} genes × {} cells",
            sc_genes.len(),
            cell_names.len()
        );

        // Cells must line up across the counts, the embedding and the annotation.
        // Anything missing from either table is dropped rather than defaulted: a
        // cell with no embedding has no position, and one with no annotation has
        // no label to contribute.
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
        let mut emb_rows: Vec<usize> = Vec::with_capacity(cell_names.len());
        for (ci, name) in cell_names.iter().enumerate() {
            if keep.as_ref().is_some_and(|k| !k.contains(name)) {
                continue;
            }
            if let (Some(&ei), Some(&ai)) =
                (emb_index.get(name.as_ref()), ann_index.get(name.as_ref()))
            {
                rows.push(CellRow {
                    counts: ci,
                    annotation: ai,
                });
                emb_rows.push(ei);
            }
        }
        anyhow::ensure!(
            !rows.is_empty(),
            "deconvolve archetypes: no cell name is shared by the counts, the embedding and the \
             annotation — check that they come from the same run"
        );
        if rows.len() < cell_names.len() {
            info!(
                "deconvolve archetypes: using {}/{} cells present in counts, embedding and \
                 annotation",
                rows.len(),
                cell_names.len()
            );
        }

        // `sub` is the only view of the embedding anything downstream needs, so
        // the full N×H matrix is dropped here rather than carried for the run.
        let h = embedding.mat.ncols();
        let sub = Mat::from_fn(rows.len(), h, |i, j| embedding.mat[(emb_rows[i], j)]);
        drop(embedding);
        let panel_to_sc = map_panel_genes(&sc_genes, genes)?;

        Ok(Self {
            n_sc_genes: sc_genes.len(),
            cell_names,
            stack,
            annotation,
            rows,
            sub,
            panel_to_sc,
        })
    }

    /// Cluster the retained cells at one granularity, returning a label per cell
    /// of the full count matrix (`usize::MAX` for cells not in the subset).
    fn partition(&self, target: usize, seed: u64, min_cells: usize) -> Result<Vec<usize>> {
        let target = target.clamp(1, self.rows.len());
        let clustered = leiden_clustering(
            &self.sub,
            LEIDEN_KNN,
            LEIDEN_RESOLUTION,
            Some(target),
            Some(seed),
        )?;
        let labels_sub = merge_small(
            &clustered.labels,
            clustered.n_clusters,
            &self.sub,
            min_cells.max(1),
        );
        let n_arch = labels_sub.iter().copied().max().map_or(0, |m| m + 1);
        anyhow::ensure!(
            n_arch > 0,
            "deconvolve archetypes: clustering produced none"
        );
        info!(
            "deconvolve archetypes: {n_arch} archetypes over {} cells (target {target})",
            self.rows.len()
        );

        let mut labels_all = vec![usize::MAX; self.cell_names.len()];
        for (i, row) in self.rows.iter().enumerate() {
            labels_all[row.counts] = labels_sub[i];
        }
        Ok(labels_all)
    }

    /// Turn one partition's gene sums into a reference.
    fn assemble(
        &self,
        labels_all: &[usize],
        n_arch: usize,
        gene_sum: &[f64],
        genes: &[Box<str>],
        shrink: f32,
    ) -> Result<Reference> {
        let mu_gm = self.align_and_shrink(gene_sum, genes.len(), n_arch, shrink);

        let labels_sub: Vec<usize> = self.rows.iter().map(|r| labels_all[r.counts]).collect();
        let (coords, readout, n_cells) = self.summarize(&labels_sub, n_arch);
        self.warn_missing_types(&readout, n_arch);

        Ok(Reference {
            mu_gm,
            readout,
            coords,
            comp_names: (0..n_arch)
                .map(|m| format!("archetype{m:04}").into_boxed_str())
                .collect(),
            n_cells,
            celltype_names: self.annotation.cols.clone(),
        })
    }

    /// Reindex profiles onto the deconvolution gene panel, shrink each toward the
    /// pooled profile, and renormalise. Writes the gene-major layout the sampler
    /// reads directly, rather than building a `D×R` matrix and transposing it.
    ///
    /// Shrinkage is what keeps every rate strictly positive: a gene absent from
    /// an archetype would otherwise have rate zero, and a bulk count on that gene
    /// would have nowhere to be allocated.
    fn align_and_shrink(&self, gene_sum: &[f64], d: usize, n_arch: usize, shrink: f32) -> Vec<f32> {
        // `gene_sum` is row-major over groups: `gene_sum[m * n_sc_genes + g]`.
        let mut mu = vec![0f32; d * n_arch];
        let mut pooled = vec![0f64; d];
        let mut totals = vec![0f64; n_arch];
        for g in 0..d {
            let slots = self.panel_to_sc.get(g);
            let row = &mut mu[g * n_arch..(g + 1) * n_arch];
            for (m, cell) in row.iter_mut().enumerate() {
                let v: f64 = slots
                    .iter()
                    .map(|&sg| gene_sum[m * self.n_sc_genes + sg as usize])
                    .sum();
                *cell = v as f32;
                pooled[g] += v;
                totals[m] += v;
            }
        }
        // Pooled profile as a probability over the panel.
        let pooled_total: f64 = pooled.iter().sum();
        if pooled_total > 0.0 {
            for p in &mut pooled {
                *p /= pooled_total;
            }
        }
        // `x̄_{m,g} = (n_{m,g} + κ·pooled_g) / (n_m + κ)`: κ is a pseudo-count
        // against real counts, so it only matters where an archetype has none.
        // Shrinking the already-normalised profile instead would put κ units of
        // prior against a profile summing to 1, making κ = 5 an 83% blend.
        for g in 0..d {
            let row = &mut mu[g * n_arch..(g + 1) * n_arch];
            for (m, cell) in row.iter_mut().enumerate() {
                let num = f64::from(*cell) + f64::from(shrink) * pooled[g];
                let den = totals[m] + f64::from(shrink);
                *cell = if den > 0.0 { (num / den) as f32 } else { 0.0 };
            }
        }
        mu
    }

    /// Community means of the embedding (coordinates) and of the annotation (readout).
    fn summarize(&self, labels: &[usize], n_arch: usize) -> (Mat, Mat, Vec<f32>) {
        let h = self.sub.ncols();
        let c = self.annotation.mat.ncols();
        let mut coords = Mat::zeros(n_arch, h);
        let mut readout = Mat::zeros(n_arch, c);
        let mut counts = vec![0f32; n_arch];
        for (i, row) in self.rows.iter().enumerate() {
            let m = labels[i];
            if m >= n_arch {
                continue;
            }
            counts[m] += 1.0;
            for j in 0..h {
                coords[(m, j)] += self.sub[(i, j)];
            }
            for ct in 0..c {
                readout[(m, ct)] += self.annotation.mat[(row.annotation, ct)];
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
        (coords, readout, counts)
    }

    /// A cell type with no archetype behind it can never be reported, whatever
    /// the bulk contains. That happens quietly when `--archetype-cells` selects a
    /// skewed subset, so name the types rather than returning zeros for them.
    fn warn_missing_types(&self, readout: &Mat, n_arch: usize) {
        let missing: Vec<&str> = self
            .annotation
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
                self.annotation.cols.len(),
                missing.join(", ")
            );
        }
    }
}

/// Panel gene → single-cell rows, in compressed row form.
struct GeneMap {
    offsets: Vec<u32>,
    rows: Vec<u32>,
}

impl GeneMap {
    fn get(&self, gene: usize) -> &[u32] {
        let (lo, hi) = (self.offsets[gene] as usize, self.offsets[gene + 1] as usize);
        &self.rows[lo..hi]
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn matched(&self) -> usize {
        (0..self.len()).filter(|&g| !self.get(g).is_empty()).count()
    }
}

/// Panel gene → single-cell row, reconciled through the shared canonicalizer.
///
/// Built once: canonicalizing tens of thousands of names per partition would
/// repeat the same work for every granularity.
fn map_panel_genes(sc_genes: &[Box<str>], genes: &[Box<str>]) -> Result<GeneMap> {
    let name_kind = crate::embed_common::reconcile_name_kind(genes, &[sc_genes]);
    // Many-to-one collapses keep every source row: these are counts, so the
    // contributions sum. The bulk loader does the same, and taking only the
    // first row here would scale those genes differently on the two sides.
    let mut index: FxHashMap<Box<str>, Vec<usize>> = FxHashMap::default();
    for (i, n) in sc_genes.iter().enumerate() {
        index.entry(name_kind.canonicalize(n)).or_default().push(i);
    }
    let mut offsets = Vec::with_capacity(genes.len() + 1);
    let mut flat: Vec<u32> = Vec::with_capacity(genes.len());
    offsets.push(0u32);
    for name in genes {
        if let Some(rows) = index.get(&name_kind.canonicalize(name)) {
            flat.extend(rows.iter().map(|&i| i as u32));
        }
        offsets.push(flat.len() as u32);
    }
    let out = GeneMap {
        offsets,
        rows: flat,
    };
    let matched = out.matched();
    let d = genes.len();
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
        // Unmatched genes have no reference rate at all, so any bulk count on
        // them is dropped from the likelihood rather than shrunk toward a
        // pooled profile — the composition is then fit to a truncated bulk.
        warn!(
            "deconvolve archetypes: only {matched}/{d} reference genes ({:.1}%) found in the \
             single-cell counts; bulk counts on the remainder are excluded from the fit",
            100.0 * matched as f64 / d as f64
        );
    } else if matched < d {
        info!("deconvolve archetypes: matched {matched}/{d} reference genes");
    }
    Ok(out)
}

/// Pick the manifest's cell embedding whose width matches the gene embedding.
fn load_cell_embedding(src: &EmbeddingSource) -> Result<MatWithNames<Mat>> {
    anyhow::ensure!(
        !src.cell_embedding_paths.is_empty(),
        "deconvolve needs a cell embedding to cluster the reference cells in, but the \
         `--from` manifest records neither `cell_embedding` nor `latent`"
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
    // The shared reader handles gzip, comments, a header row and delimited
    // tables; a hand-rolled line split silently ingests a header as a barcode.
    let set: FxHashSet<Box<str>> = matrix_util::common_io::read_name_list(path)
        .with_context(|| format!("reading cell list {path}"))?
        .into_iter()
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
