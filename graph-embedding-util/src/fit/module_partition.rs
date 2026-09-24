//! The module partition: features grouped by their counts over the finest
//! pseudobulks, built once after the pseudobulk tree and fixed for the whole
//! run. It is a single-level [`coarsen_features`] — the same grouping `senna
//! topic` coarsens its feature axis with, shared through `data-beans`: features
//! with no grouping evidence form one background module, the rest are grouped
//! by k-means on their residual profiles.
//!
//! A membership learned from a cold start is rich-get-richer — the exact module
//! term rewards putting every feature in the module that already scores well —
//! so the partition comes from the data and does not move during training.

use super::config::{ParentModulesOwned, TrackSpec};
use data_beans::alg::feature_coarsening::coarsen_features;
use legume_numeric::matrix::rand_util::mix_seed;
use log::info;
use nalgebra::DMatrix;

/// The base track's rows of a `[n_features × S]` profile, re-keyed by gene:
/// `[n_genes × S]`. The module partition is over GENES, so a multi-track
/// feature axis is reduced to its base track before clustering — the base track
/// is the one the model itself is; every other track is an offset from it. A
/// gene with no base row (never observed on the base track) keeps a zero
/// profile row, which the warm start already routes to its background module.
/// Identity on a one-track axis, where row IS gene.
#[must_use]
pub fn base_track_profile(profile: &DMatrix<f32>, tracks: &TrackSpec) -> DMatrix<f32> {
    debug_assert_eq!(
        profile.nrows(),
        tracks.track_of_row.len(),
        "the profile and the track spec describe the same feature axis"
    );
    let mut out = DMatrix::<f32>::zeros(tracks.n_genes(), profile.ncols());
    for (row, (&t, &g)) in tracks
        .track_of_row
        .iter()
        .zip(&tracks.gene_of_row)
        .enumerate()
    {
        if t == 0 {
            out.set_row(g as usize, &profile.row(row));
        }
    }
    out
}

/// The module partition: a single-level [`coarsen_features`] at `n_modules`,
/// one module id (`< n_modules`) per feature.
pub fn partition_modules(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    n_modules: usize,
    seed: u64,
) -> anyhow::Result<Vec<u32>> {
    let d = counts.nrows();
    if n_modules < 2 || d < 2 {
        return Ok(vec![0; d]);
    }
    let level = coarsen_features(counts, sizes, &[n_modules], seed)?
        .pop()
        .expect("one level requested");
    Ok(level.fine_to_coarse.iter().map(|&g| g as u32).collect())
}

/// Membership logits `[D × M]` for a fit warm-started from a parent: a matched
/// feature takes the parent's membership row verbatim (a simplex point, which
/// sparsemax reproduces exactly), an unmatched one is initialized through the
/// parent's modules from its nearest matched neighbours by profile, or the
/// parent's module-average membership below the similarity floor.
#[must_use]
pub fn parent_module_logits(parent: &ParentModulesOwned, profiles: &DMatrix<f32>) -> DMatrix<f32> {
    use crate::transfer::{align_gene_axis, AlignInputs, ModuleTables};
    let d = parent.row_to_parent.len();
    let m = parent.pi.ncols();
    let al = align_gene_axis(&AlignInputs {
        rho: &parent.rho,
        b_feat: None,
        modules: Some(ModuleTables {
            pi: &parent.pi,
            mu: &parent.mu,
        }),
        new_to_train: &parent.row_to_parent,
        profiles_new: Some(profiles),
        knobs: parent.knobs,
    });
    let membership = al
        .membership
        .as_ref()
        .expect("module tables given ⇒ membership present");
    let mut logits = DMatrix::<f32>::zeros(d, m);
    let (mut n_init, mut n_diffuse) = (0usize, 0usize);
    for g in 0..d {
        let union = al.new_to_union[g].expect("every feature is matched or initialized");
        logits.set_row(g, &membership.row(union));
        if !al.is_scored(union) {
            n_init += 1;
            if al.provenance[union].as_ref().is_some_and(|p| p.diffuse) {
                n_diffuse += 1;
            }
        }
    }
    info!(
        "module warm start from a parent: {} features carry the parent's membership, \
         {n_init} initialized through its modules ({n_diffuse} on the diffuse prior)",
        d - n_init
    );
    logits
}

/// Per modality, whether its features are **module-only**: at least
/// `min_rows` of them. `None` when no modality qualifies or `min_rows` is `0`.
/// `modality` holds one modality id per feature row.
#[must_use]
pub fn module_only_modalities(modality: &[u32], min_rows: usize) -> Option<Vec<bool>> {
    if min_rows == 0 || modality.is_empty() {
        return None;
    }
    let n = modality.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut rows = vec![0usize; n];
    for &m in modality {
        rows[m as usize] += 1;
    }
    let flags: Vec<bool> = rows.iter().map(|&r| r >= min_rows).collect();
    flags.iter().any(|&f| f).then_some(flags)
}

/// Per feature row, `true` for a FLAT row: one rate explains its counts over
/// the finest pseudobulks, by the Poisson homogeneity test of
/// [`data_beans::alg::feature_coarsening::informative_features`]
/// (`z = (D − df)/√(2·df) ≤ 5` on the deviance `D`, `df = S − 1`; an
/// all-zero row is flat). Flat rows form the coarsener's background group and
/// can go module-only, so the gene-level softmax never scores them.
///
/// The test runs on the finest level only: on pooled coarser levels the
/// Poisson test ignores overdispersion and calls almost every feature
/// non-flat.
///
/// Opt-in (`enabled`), and `None` on a multi-track axis: module-only rows need
/// one track, and there the partition is over genes rather than rows.
#[must_use]
pub fn flat_module_only(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    tracks: &TrackSpec,
    enabled: bool,
) -> Option<Vec<bool>> {
    if !enabled || !tracks.is_base() {
        return None;
    }
    let informative = data_beans::alg::feature_coarsening::informative_features(counts, sizes);
    Some(informative.into_iter().map(|inf| !inf).collect())
}

/// [`partition_modules`] with every group at least `min_size` rows, except
/// the background. Returns the labels and the background's id (`None` when
/// every row is informative and nothing was set aside).
///
/// k-means places a row whose profile matches no other alone, so on a noisy
/// axis (ATAC peaks) many groups are one or a few rows that arose at random,
/// while the rest crowd into a few large groups. Such **scattered** rows join
/// the background with the near-empty ones. Every other group is kept as it
/// is, and the slots the scattered groups held go to splitting the largest
/// groups in two (k-means on the group's own rows), a split kept only when
/// both halves reach `min_size`. Re-clustering everything with the freed
/// budget instead would tear real groups into pieces under the minimum.
/// `min_size <= 1` is the plain partition.
pub fn partition_modules_min_size(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    n_modules: usize,
    seed: u64,
    min_size: usize,
) -> anyhow::Result<(Vec<u32>, Option<u32>)> {
    let informative = data_beans::alg::feature_coarsening::informative_features(counts, sizes);
    min_size_partition(counts, sizes, n_modules, seed, min_size, &informative)
}

/// [`partition_modules_min_size`] given each row's `informative` flag (not
/// flat), so a caller that already ran the test does not run it again.
fn min_size_partition(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    n_modules: usize,
    seed: u64,
    min_size: usize,
    informative: &[bool],
) -> anyhow::Result<(Vec<u32>, Option<u32>)> {
    let d = counts.nrows();
    let background_of = |labels: &[u32], rows: &[usize]| {
        rows.iter()
            .position(|&r| !informative[r])
            .map(|i| labels[i])
    };
    let all: Vec<usize> = (0..d).collect();
    if min_size <= 1 {
        let labels = partition_modules(counts, sizes, n_modules, seed)?;
        let bg = background_of(&labels, &all);
        return Ok((labels, bg));
    }
    let plain = partition_modules(counts, sizes, n_modules, seed)?;
    let bg_plain = background_of(&plain, &all);
    // The plain partition's groups, the background left out.
    let mut by_label = std::collections::BTreeMap::<u32, Vec<usize>>::new();
    for (r, &m) in plain.iter().enumerate() {
        if Some(m) != bg_plain {
            by_label.entry(m).or_default().push(r);
        }
    }
    let (kept, scattered): (Vec<Vec<usize>>, Vec<Vec<usize>>) =
        by_label.into_values().partition(|g| g.len() >= min_size);
    let n_scattered: usize = scattered.iter().map(Vec::len).sum();
    let has_bg = bg_plain.is_some() || n_scattered > 0;
    // Freed slots split the largest groups in two, largest first.
    let mut slots = n_modules.saturating_sub(kept.len() + usize::from(has_bg));
    let mut heap: std::collections::BinaryHeap<(usize, usize)> =
        kept.iter().enumerate().map(|(i, g)| (g.len(), i)).collect();
    let mut groups: Vec<Option<Vec<usize>>> = kept.into_iter().map(Some).collect();
    let mut n_split = 0usize;
    while slots > 0 {
        let Some((len, i)) = heap.pop() else { break };
        if len < 2 * min_size {
            break;
        }
        let g = groups[i].take().expect("a live group");
        let halves = partition_modules(
            &counts.select_rows(&g),
            sizes,
            2,
            mix_seed(seed, 0x5350_4c54 + i as u64),
        )?;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for (&r, &h) in g.iter().zip(&halves) {
            if h == 0 {
                a.push(r)
            } else {
                b.push(r)
            }
        }
        if a.len() >= min_size && b.len() >= min_size {
            heap.push((a.len(), i));
            groups[i] = Some(a);
            heap.push((b.len(), groups.len()));
            groups.push(Some(b));
            slots -= 1;
            n_split += 1;
        } else {
            // Not splittable: keep it whole, out of the heap.
            groups[i] = Some(g);
        }
    }
    // Labels: the groups in order, then the background.
    let mut labels = vec![u32::MAX; d];
    let mut next = 0u32;
    for g in groups.into_iter().flatten() {
        for r in g {
            labels[r] = next;
        }
        next += 1;
    }
    for l in &mut labels {
        if *l == u32::MAX {
            *l = next;
        }
    }
    // Per call: the blocked partition calls this once per block and logs the total.
    log::debug!(
        "module partition: {n_scattered} scattered feature(s) in groups under {min_size} joined \
         the background; {n_split} large group(s) split; {next} group(s) + background"
    );
    Ok((labels, has_bg.then_some(next)))
}

/// [`partition_modules_min_size`] with no module crossing a BLOCK (`block[r]`
/// per row, e.g. a genomic window): each block is partitioned on its own with
/// a share of the `n_modules − 1` informative slots proportional to its
/// informative rows (at least one), and the near-empty and scattered rows of
/// every block share ONE background. With a single block this is
/// [`partition_modules_min_size`].
pub fn partition_modules_blocked(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    n_modules: usize,
    seed: u64,
    min_size: usize,
    block: &[u32],
) -> anyhow::Result<(Vec<u32>, Option<u32>)> {
    let d = counts.nrows();
    anyhow::ensure!(block.len() == d, "{} block ids for {d} rows", block.len());
    let mut rows_of = std::collections::BTreeMap::<u32, Vec<usize>>::new();
    for (r, &b) in block.iter().enumerate() {
        rows_of.entry(b).or_default().push(r);
    }
    let informative = data_beans::alg::feature_coarsening::informative_features(counts, sizes);
    // One summary line per call: informative rows the background took are the
    // scattered ones.
    let summary = |labels: &[u32], bg: Option<u32>, n_blocks: usize| {
        let n_groups = labels
            .iter()
            .filter(|&&m| Some(m) != bg)
            .collect::<std::collections::HashSet<_>>()
            .len();
        let n_scattered = labels
            .iter()
            .zip(&informative)
            .filter(|&(&m, &inf)| inf && Some(m) == bg)
            .count();
        info!(
            "module partition: {n_groups} group(s) + background over {n_blocks} block(s); \
             {n_scattered} scattered feature(s) (groups under {min_size}) joined the background"
        );
    };
    if rows_of.len() <= 1 {
        let (labels, bg) =
            min_size_partition(counts, sizes, n_modules, seed, min_size, &informative)?;
        summary(&labels, bg, 1);
        return Ok((labels, bg));
    }
    let n_inf = |rows: &[usize]| rows.iter().filter(|&&r| informative[r]).count();
    let live: Vec<(u32, usize)> = rows_of
        .iter()
        .map(|(&b, rows)| (b, n_inf(rows)))
        .filter(|&(_, c)| c > 0)
        .collect();
    let budget = n_modules.saturating_sub(1);
    anyhow::ensure!(
        live.len() <= budget,
        "{} blocks hold informative features but only {budget} module slots: widen the blocks",
        live.len()
    );
    // Proportional shares, at least one each, then the remainder to the blocks
    // with the most informative rows per slot.
    let total: usize = live.iter().map(|&(_, c)| c).sum();
    let mut alloc: Vec<usize> = live
        .iter()
        .map(|&(_, c)| (budget * c / total.max(1)).max(1))
        .collect();
    while alloc.iter().sum::<usize>() > budget {
        let i = (0..alloc.len())
            .max_by_key(|&i| alloc[i])
            .expect("non-empty");
        alloc[i] -= 1;
    }
    while alloc.iter().sum::<usize>() < budget {
        let i = (0..alloc.len())
            .max_by(|&a, &b| (live[a].1 * alloc[b]).cmp(&(live[b].1 * alloc[a])))
            .expect("non-empty");
        alloc[i] += 1;
    }
    let mut labels = vec![u32::MAX; d];
    let mut next = 0u32;
    for (&(b, _), &k) in live.iter().zip(&alloc) {
        let rows = &rows_of[&b];
        // One more slot when the block has near-empty rows: the partition
        // spends it on their background, which joins the shared one below.
        let has_empty = rows.len() > n_inf(rows);
        let block_informative: Vec<bool> = rows.iter().map(|&r| informative[r]).collect();
        let (local, bg_local) = min_size_partition(
            &counts.select_rows(rows),
            sizes,
            k + usize::from(has_empty),
            mix_seed(seed, u64::from(b)),
            min_size,
            &block_informative,
        )?;
        let mut compact = std::collections::HashMap::<u32, u32>::new();
        for (&r, &m) in rows.iter().zip(&local) {
            if Some(m) != bg_local {
                let id = *compact.entry(m).or_insert_with(|| {
                    next += 1;
                    next - 1
                });
                labels[r] = id;
            }
        }
    }
    let bg = next;
    let mut any_bg = false;
    for l in &mut labels {
        if *l == u32::MAX {
            *l = bg;
            any_bg = true;
        }
    }
    let bg = any_bg.then_some(bg);
    summary(&labels, bg, live.len());
    Ok((labels, bg))
}

/// Per row, `true` when its module is flagged in `background` (flat or
/// scattered features, see [`GroupedPartition::background`]).
#[must_use]
pub fn background_rows(labels: &[u32], background: &[bool]) -> Vec<bool> {
    labels
        .iter()
        .map(|&m| background.get(m as usize).copied().unwrap_or(false))
        .collect()
}

/// A partition over groups of rows (see [`partition_modules_by_group`]).
pub struct GroupedPartition {
    /// Module id of every row.
    pub labels: Vec<u32>,
    /// Total module count, `Σ n_per`.
    pub n_modules: usize,
    /// Per module, `true` for the background module of a group asked to set
    /// scattered rows aside (`min_size[k] > 1`): near-empty or scattered rows.
    pub background: Vec<bool>,
}

/// [`partition_modules_blocked`] confined to GROUPS of rows: group `k`'s rows
/// are partitioned on their own into at most `n_per[k]` modules of at least
/// `min_size[k]` rows, none crossing a block of `block_of_row` (`None`: no
/// blocks), numbered after the previous groups'. Groups are modalities, so one
/// modality can drop its residual while the others keep theirs.
pub fn partition_modules_by_group(
    counts: &DMatrix<f32>,
    sizes: &[f32],
    group_of_row: &[u32],
    n_per: &[usize],
    min_size: &[usize],
    block_of_row: Option<&[u32]>,
    seed: u64,
) -> anyhow::Result<GroupedPartition> {
    anyhow::ensure!(
        min_size.len() == n_per.len(),
        "{} minimum sizes for {} groups",
        min_size.len(),
        n_per.len()
    );
    let mut labels = vec![0u32; group_of_row.len()];
    let n_modules: usize = n_per.iter().sum();
    let mut background = vec![false; n_modules];
    let mut offset = 0usize;
    for (k, (&n_k, &min_k)) in n_per.iter().zip(min_size).enumerate() {
        let rows: Vec<usize> = (0..group_of_row.len())
            .filter(|&i| group_of_row[i] as usize == k)
            .collect();
        if !rows.is_empty() {
            // A row with no block (`None`, or `u32::MAX`) shares one block.
            let blocks: Vec<u32> = rows
                .iter()
                .map(|&r| block_of_row.map_or(u32::MAX, |b| b[r]))
                .collect();
            let (local, bg) = partition_modules_blocked(
                &counts.select_rows(&rows),
                sizes,
                n_k,
                mix_seed(seed, k as u64),
                min_k,
                &blocks,
            )?;
            for (&r, &m) in rows.iter().zip(&local) {
                labels[r] = offset as u32 + m;
            }
            if let (Some(bg), true) = (bg, min_k > 1) {
                background[offset + bg as usize] = true;
            }
        }
        offset += n_k;
    }
    Ok(GroupedPartition {
        labels,
        n_modules,
        background,
    })
}

#[cfg(test)]
mod tests;
