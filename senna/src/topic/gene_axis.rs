//! Continuing a masked model onto a gene axis the source run did not have.
//!
//! A checkpoint's gene-keyed state is small: the fine-to-module map of each
//! coarsening level and the per-gene embedding ρ. Everything else the decoders
//! hold is module- or topic-keyed, so a cohort that measures genes the parent
//! never saw can still be absorbed if those two are grown by NAME rather than
//! refused by position. Genes the source run knew keep their module and their ρ
//! row. Genes it did not are placed by the pseudobulk aggregation of THIS run:
//! each goes to the inherited module whose known members its pseudobulk
//! profile most resembles, and starts from that module's mean ρ. The decoders
//! keep their widths, so nothing they learned is disturbed.

use crate::embed_common::Mat;
use data_beans_alg::feature_coarsening::FeatureCoarsening;

/// This run's gene axis aligned onto the source run's, by name.
pub(crate) struct GeneAxisRemap {
    /// For each gene of this run, its position on the source run's axis.
    pub new_to_source: Vec<Option<usize>>,
    /// The source run's axis length.
    pub n_source: usize,
}

impl GeneAxisRemap {
    #[must_use]
    pub(crate) fn n_matched(&self) -> usize {
        self.new_to_source.iter().filter(|p| p.is_some()).count()
    }

    /// True when the two axes are the same genes in the same order, so the
    /// exact warm start applies and nothing here is needed.
    #[must_use]
    pub(crate) fn is_identity(&self) -> bool {
        self.new_to_source.len() == self.n_source
            && self
                .new_to_source
                .iter()
                .enumerate()
                .all(|(i, p)| *p == Some(i))
    }
}

/// Grow one coarsening level onto this run's axis.
///
/// `profiles_dn` is this run's finest pseudobulk posterior, one row per gene
/// of this run. A gene the source run knew keeps its module. A gene it did not is
/// assigned to the module whose known members (in this run's profile space)
/// it is closest to by cosine. A module none of whose members survived the
/// remap cannot attract anything, so it keeps its index and stays empty of
/// new genes; the decoders are keyed to it, so it is not dropped.
pub(crate) fn grow_fine_to_coarse(
    source: &FeatureCoarsening,
    remap: &GeneAxisRemap,
    profiles_dn: &Mat,
) -> anyhow::Result<FeatureCoarsening> {
    let d_new = remap.new_to_source.len();
    anyhow::ensure!(
        source.fine_to_coarse.len() == remap.n_source,
        "gene axis growth: the source run's coarsening covers {} genes but its axis has {}",
        source.fine_to_coarse.len(),
        remap.n_source,
    );
    anyhow::ensure!(
        profiles_dn.nrows() == d_new,
        "gene axis growth: {} pseudobulk profiles for {d_new} genes",
        profiles_dn.nrows(),
    );
    let (k, n_pb) = (source.num_coarse, profiles_dn.ncols());

    // Known genes keep their module, and their unit profiles accumulate into
    // the module's centroid — in THIS run's profile space, so a new gene is
    // compared against what its module looks like on these pseudobulks.
    let mut fine_to_coarse = vec![usize::MAX; d_new];
    let mut centroid = vec![vec![0f32; n_pb]; k];
    let mut members = vec![0usize; k];
    let unit = |g: usize| -> Option<Vec<f32>> {
        let row = profiles_dn.row(g);
        let nrm = row.norm();
        (nrm > 0.0).then(|| row.iter().map(|v| v / nrm).collect())
    };
    for (g, source_pos) in remap.new_to_source.iter().enumerate() {
        let Some(p) = source_pos else { continue };
        let m = source.fine_to_coarse[*p];
        fine_to_coarse[g] = m;
        if let Some(u) = unit(g) {
            for (c, v) in centroid[m].iter_mut().zip(&u) {
                *c += v;
            }
            members[m] += 1;
        }
    }
    anyhow::ensure!(
        members.iter().any(|&c| c > 0),
        "gene axis growth: no gene of this run matches the source run, so its modules cannot be \
         placed on this cohort. The two axes share nothing; see --feature-name-kind."
    );
    for (c, &n) in centroid.iter_mut().zip(&members) {
        if n > 0 {
            let nrm = c.iter().map(|v| v * v).sum::<f32>().sqrt().max(f32::EPSILON);
            c.iter_mut().for_each(|v| *v /= nrm);
        }
    }
    // The module with the most surviving members is where a gene with no
    // profile at all goes: it carries no signal to place it by, and the
    // largest module perturbs the fit least.
    let fallback = (0..k).max_by_key(|&m| members[m]).expect("k > 0");

    // Unknown genes join the module whose centroid their unit profile is
    // closest to, among modules that have any surviving member to define one.
    let mut n_placed = 0usize;
    for (g, slot) in fine_to_coarse.iter_mut().enumerate() {
        if *slot != usize::MAX {
            continue;
        }
        *slot = match unit(g) {
            None => fallback,
            Some(u) => (0..k)
                .filter(|&m| members[m] > 0)
                .map(|m| (m, centroid[m].iter().zip(&u).map(|(a, b)| a * b).sum::<f32>()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(fallback, |(m, _)| m),
        };
        n_placed += 1;
    }
    let mut coarse_to_fine = vec![Vec::new(); k];
    for (g, &m) in fine_to_coarse.iter().enumerate() {
        coarse_to_fine[m].push(g);
    }
    log::info!(
        "gene axis growth: {} of {d_new} genes matched the source run, {n_placed} new genes placed \
         by pseudobulk profile into its {k} inherited modules ({} modules had no surviving member)",
        d_new - n_placed,
        members.iter().filter(|&&c| c == 0).count(),
    );
    Ok(FeatureCoarsening {
        fine_to_coarse,
        coarse_to_fine,
        num_coarse: k,
    })
}

/// Grow the per-gene embedding onto this run's axis.
///
/// A gene the source run knew copies its row. A gene it did not starts at the mean
/// ρ of the known members of the module `grown` placed it in, so it enters the
/// fit inside its module's neighbourhood rather than at a random point.
pub(crate) fn grow_rho(
    source_rho: &Mat,
    remap: &GeneAxisRemap,
    grown: &FeatureCoarsening,
) -> anyhow::Result<Mat> {
    anyhow::ensure!(
        source_rho.nrows() == remap.n_source,
        "gene axis growth: the source run's ρ has {} rows but its axis has {}",
        source_rho.nrows(),
        remap.n_source,
    );
    let (d_new, h, k) = (remap.new_to_source.len(), source_rho.ncols(), grown.num_coarse);
    let mut rho = Mat::zeros(d_new, h);
    let mut module_sum = Mat::zeros(k, h);
    let mut members = vec![0usize; k];
    for (g, source_pos) in remap.new_to_source.iter().enumerate() {
        let Some(p) = source_pos else { continue };
        rho.row_mut(g).copy_from(&source_rho.row(*p));
        let m = grown.fine_to_coarse[g];
        let mut acc = module_sum.row_mut(m);
        acc += source_rho.row(*p);
        members[m] += 1;
    }
    for (g, source_pos) in remap.new_to_source.iter().enumerate() {
        if source_pos.is_some() {
            continue;
        }
        let m = grown.fine_to_coarse[g];
        anyhow::ensure!(
            members[m] > 0,
            "gene axis growth: gene {g} was placed in module {m}, which has no surviving \
             member to start its embedding from"
        );
        let mean = module_sum.row(m) / members[m] as f32;
        rho.row_mut(g).copy_from(&mean);
    }
    Ok(rho)
}

/// This run's genes aligned onto an `--init-from` source run's, or `None` when the
/// two axes are identical and the exact warm start applies.
///
/// The source run's axis is the row order of its `feature_mean.parquet`, which is
/// the order every gene-keyed artifact of that run shares. Matching goes
/// through the same matcher `predict` aligns a query with, so a source run that
/// spells a gene differently by case or suffix still matches.
pub(crate) fn remap_to_source(
    source: &str,
    new_genes: &[Box<str>],
) -> anyhow::Result<Option<GeneAxisRemap>> {
    let (source_genes, _) = crate::topic::model_metadata::load_feature_mean(source)?;
    let remap = crate::topic::eval::build_gene_remap_with(
        &source_genes,
        new_genes,
        &crate::topic::eval::QueryNameOpts::default(),
    );
    let remap = GeneAxisRemap {
        new_to_source: remap.new_to_train,
        n_source: source_genes.len(),
    };
    if remap.is_identity() {
        return Ok(None);
    }
    log::info!(
        "--init-from {source}: this run's gene axis differs from the source run's ({} genes here, \
         {} there, {} in common); continuing by name",
        new_genes.len(),
        remap.n_source,
        remap.n_matched(),
    );
    Ok(Some(remap))
}

/// [`grow_rho`] for a source run trained at full resolution, where there are no
/// modules to take a mean over: a new gene starts at the mean of every known
/// row instead.
pub(crate) fn grow_rho_without_modules(
    source_rho: &Mat,
    remap: &GeneAxisRemap,
) -> anyhow::Result<Mat> {
    let n_source = source_rho.nrows();
    anyhow::ensure!(
        n_source == remap.n_source,
        "gene axis growth: the source run's ρ has {n_source} rows but its axis has {}",
        remap.n_source,
    );
    let matched: Vec<usize> = remap.new_to_source.iter().flatten().copied().collect();
    anyhow::ensure!(!matched.is_empty(), "gene axis growth: no gene in common with the source run");
    let mut mean = Mat::zeros(1, source_rho.ncols());
    for &p in &matched {
        let mut acc = mean.row_mut(0);
        acc += source_rho.row(p);
    }
    mean /= matched.len() as f32;
    let mut rho = Mat::zeros(remap.new_to_source.len(), source_rho.ncols());
    for (g, source_pos) in remap.new_to_source.iter().enumerate() {
        match source_pos {
            Some(p) => rho.row_mut(g).copy_from(&source_rho.row(*p)),
            None => rho.row_mut(g).copy_from(&mean.row(0)),
        }
    }
    Ok(rho)
}

/// A parent's ρ, read back on its own gene order.
pub(crate) fn load_source_rho(source: &str) -> anyhow::Result<Mat> {
    use matrix_util::traits::IoOps;
    let path = format!("{source}.feature_embedding.parquet");
    Ok(<Mat as IoOps>::from_parquet_with_row_names(&path, Some(0))?.mat)
}

#[cfg(test)]
#[path = "gene_axis_tests.rs"]
mod gene_axis_tests;
