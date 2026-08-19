//! What a matrix ROW means, resolved once from the row names.
//!
//! A plain matrix has one gene per row. A splice-channelized one (faba's
//! `{gene}/count/{spliced,unspliced}`) has two rows per gene, and reading one
//! as the other is silent rather than loud: `n_genes < n_rows` means a per-gene
//! index into a per-row vector never goes out of bounds, it just reads the
//! wrong feature.
//!
//! Consumers split into two kinds, and the split is the whole point:
//!
//! - **Mean GENE.** `cage`'s training loop and pseudobulk selection, `lc`'s
//!   projection basis, count filter, dictionary and merge cutoff, and `lra`'s
//!   ligand and receptor lookup. All of these fold through this type.
//! - **Mean ROW.** Anything that reads the matrix directly and reports what it
//!   read: the degree-corrected Poisson refinement's blocking dimension, the
//!   saved per-row NB precisions, `dsvd`'s two stacked channels.
//!
//! The NB-Fisher weights need BOTH, and they are the reason
//! [`Self::broadcast_to_rows`] exists: the weight is a function of abundance
//! and mean and is not additive, so it is evaluated once per gene on folded
//! statistics and then spread back over that gene's rows. Folding the weights
//! afterwards would hand a gene a precision no measurement supports.
//!
//! Resolution is strict by default: a feature axis where only some rows carry
//! splice channels is an error, because pooling a row whose track is unknown
//! has no correct answer. [`GeneAxis::resolve_or_identity`] is the fallback for
//! a consumer that never pools.

use crate::util::common::*;
use auxiliary_data::feature_rows::{intern_count_rows, UnparsedRowPolicy};

/// The feature axis `cage` fits on: one entry per GENE, with a map back to the
/// matrix rows that carry it.
#[derive(Debug)]
pub struct GeneAxis {
    row_to_gene: Vec<u32>,
    row_is_nascent: Vec<bool>,
    gene_names: Vec<Box<str>>,
    channelized: bool,
}

impl GeneAxis {
    /// Resolve the axis from the matrix's row names.
    ///
    /// - no row is `{gene}/count/{spliced|unspliced}` ⇒ the identity axis.
    /// - every row is ⇒ the pooled gene axis.
    /// - some rows are ⇒ error, naming up to three offenders.
    pub fn resolve(row_names: &[Box<str>]) -> anyhow::Result<Self> {
        let map = intern_count_rows(row_names, UnparsedRowPolicy::Reject);
        let n_rows = row_names.len();

        if map.unparsed.len() == n_rows {
            return Ok(Self {
                row_to_gene: (0..n_rows as u32).collect(),
                row_is_nascent: vec![false; n_rows],
                gene_names: row_names.to_vec(),
                channelized: false,
            });
        }

        if !map.unparsed.is_empty() {
            let shown: Vec<&str> = map
                .unparsed
                .iter()
                .take(3)
                .map(|&r| row_names[r].as_ref())
                .collect();
            anyhow::bail!(
                "the feature axis mixes gene-count rows with {} row(s) that are not \
                 `{{gene}}/count/{{spliced|unspliced}}`, and pooling a row whose splice \
                 track is unknown has no correct answer. Offenders: {}. A `{{gene}}/count/total` \
                 row is the usual cause — it is already spliced + unspliced, so it lives in \
                 its own matrix and must not be concatenated with the two tracks.",
                map.unparsed.len(),
                shown.join(", ")
            );
        }

        info!(
            "Feature axis: {} rows carry splice channels over {} genes ({} nascent rows)",
            n_rows,
            map.n_genes(),
            map.n_nascent_rows()
        );
        Ok(Self {
            row_to_gene: map.row_to_gene,
            row_is_nascent: map.row_is_nascent,
            gene_names: map.gene_names,
            channelized: true,
        })
    }

    /// As [`Self::resolve`], but a mixed feature axis falls back to the
    /// identity rather than aborting.
    ///
    /// The strict form is right for a consumer that POOLS a gene's tracks: a
    /// row whose track is unknown has no correct pooled answer, so failing is
    /// better than guessing. A consumer that only needs a unit axis for
    /// filtering and reporting has a defined fallback, one unit per row, which
    /// is exactly what it did before a gene axis existed. Aborting there would
    /// reject multimodal matrices that used to work.
    pub fn resolve_or_identity(row_names: &[Box<str>]) -> anyhow::Result<Self> {
        match Self::resolve(row_names) {
            Ok(axis) => Ok(axis),
            Err(e) => {
                log::warn!(
                    "{e}\nFalling back to one unit per row, so nothing is pooled and \
                     every per-gene filter and report is per row instead. Split the \
                     modalities into their own matrices to get a gene axis."
                );
                Ok(Self {
                    row_to_gene: (0..row_names.len() as u32).collect(),
                    row_is_nascent: vec![false; row_names.len()],
                    gene_names: row_names.to_vec(),
                    channelized: false,
                })
            }
        }
    }

    #[must_use]
    pub fn n_genes(&self) -> usize {
        self.gene_names.len()
    }

    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.row_to_gene.len()
    }

    /// True when the input carries splice channels — i.e. when any fold below is
    /// more than a pass-through.
    #[must_use]
    pub fn is_channelized(&self) -> bool {
        self.channelized
    }

    /// Gene keys in id order. These, not the row names, label every gene-side
    /// output `cage` writes.
    #[must_use]
    pub fn gene_names(&self) -> &[Box<str>] {
        &self.gene_names
    }

    /// Spread a per-GENE vector back over the matrix rows, so every row of a
    /// gene carries its gene's value.
    ///
    /// This is how a quantity that is only correct per gene reaches a consumer
    /// that indexes rows. The projection basis is the case in point: its rows
    /// are matrix rows, but a Fisher precision computed per row would hand a
    /// gene's two splice tracks two different precisions, because the nascent
    /// track is sparser and lands elsewhere on the dispersion trend. Computing
    /// once per gene and spreading is exact; folding the weights afterwards
    /// would not be, since the weight is not additive.
    ///
    /// Returns the input unchanged on the identity axis.
    #[must_use]
    pub fn broadcast_to_rows<T: Copy>(&self, per_gene: &[T]) -> Vec<T> {
        if !self.channelized {
            return per_gene.to_vec();
        }
        debug_assert_eq!(per_gene.len(), self.n_genes());
        self.row_to_gene
            .iter()
            .map(|&g| per_gene[g as usize])
            .collect()
    }

    /// The row -> gene map itself, for callers that need to hand it to a
    /// folding primitive rather than call `gene_of_row` per row.
    #[must_use]
    pub fn row_to_gene(&self) -> &[u32] {
        &self.row_to_gene
    }

    #[must_use]
    pub fn gene_of_row(&self, row: usize) -> usize {
        self.row_to_gene[row] as usize
    }

    #[must_use]
    pub fn row_is_nascent(&self, row: usize) -> bool {
        self.row_is_nascent[row]
    }

    /// The gene-axis fold of a `[n_rows × k]` matrix, or `None` when the axis is
    /// the identity and a fold would only be a copy — so `None` means "nothing to
    /// do", never failure. Borrows, for the caller that keeps the row axis too.
    #[must_use]
    pub fn pool_rows_opt(&self, m: &Mat) -> Option<Mat> {
        self.channelized.then(|| self.fold_rows(m))
    }

    /// Sum rows onto the gene axis, COLUMN-major.
    ///
    /// `Mat` is nalgebra, so a column is contiguous and the column loop has to be
    /// the outer one: with the row loop outside, `m[(r, c)]` strides by `nrows`
    /// and costs a cache miss per element on a matrix the pyramid sizes in
    /// hundreds of MB. Both the read and the accumulator stay inside one column
    /// this way.
    ///
    /// Borrowing rather than consuming is what lets [`Self::pool_rows_opt`] skip
    /// cloning its input purely to satisfy a by-value signature.
    ///
    fn fold_rows(&self, m: &Mat) -> Mat {
        debug_assert_eq!(m.nrows(), self.n_rows());
        // Every output column depends only on the input column of the same index,
        // so the columns are independent — which is exactly what the workspace's
        // `build_columns_par` wants. Within a column both the read and the
        // accumulator stay resident, which is what the column-major order buys;
        // the row-outer form this replaced strided by `nrows` and took a cache
        // miss per element on matrices the pyramid sizes in hundreds of MB.
        build_columns_par(self.n_genes(), m.ncols(), |c, dst| {
            let src = m.column(c);
            for (r, &v) in src.iter().enumerate() {
                dst[self.gene_of_row(r)] += v;
            }
        })
    }

    /// Sum per-row totals onto the gene axis.
    #[must_use]
    pub fn pool_totals(&self, per_row: Vec<f64>) -> Vec<f64> {
        if !self.channelized {
            return per_row;
        }
        debug_assert_eq!(per_row.len(), self.n_rows());
        let mut out = vec![0.0f64; self.n_genes()];
        for (r, v) in per_row.into_iter().enumerate() {
            out[self.gene_of_row(r)] += v;
        }
        out
    }

    /// Sum a sparse `(row, value)` profile onto the gene axis, ascending by gene
    /// id. A gene's two channel rows merge into one entry, which is the whole
    /// point: downstream this is one gene's evidence, not two genes' halves.
    #[must_use]
    pub fn pool_profile(&self, mut obs: Vec<(u32, f32)>) -> Vec<(u32, f32)> {
        if !self.channelized {
            return obs;
        }
        for entry in obs.iter_mut() {
            entry.0 = self.row_to_gene[entry.0 as usize];
        }
        obs.sort_unstable_by_key(|&(g, _)| g);
        let mut out: Vec<(u32, f32)> = Vec::with_capacity(obs.len());
        for (g, v) in obs {
            match out.last_mut() {
                Some(last) if last.0 == g => last.1 += v,
                _ => out.push((g, v)),
            }
        }
        out
    }

    /// Widen a per-row weight vector so a gene is never half-weighted: if any of
    /// a gene's rows carries weight, all of them take that gene's maximum.
    ///
    /// This is what `--n-hvg` needs. HVG selection ranks ROWS, so on a
    /// channelized matrix it can pick a gene's spliced row and drop its
    /// unspliced one — which would weight half a gene into the projection the
    /// coarsening hierarchy is cut from. Returns the number of genes carrying
    /// weight, which is what the count in the log should say.
    pub fn promote_row_weights(&self, w: &mut [f32]) -> usize {
        debug_assert_eq!(w.len(), self.n_rows());
        let mut per_gene = vec![0.0f32; self.n_genes()];
        for (r, &x) in w.iter().enumerate() {
            let g = self.gene_of_row(r);
            per_gene[g] = per_gene[g].max(x);
        }
        for (r, x) in w.iter_mut().enumerate() {
            *x = per_gene[self.gene_of_row(r)];
        }
        per_gene.iter().filter(|&&x| x > 0.0).count()
    }

    /// Which genes could ever pin a nascent-minus-mature contrast, from per-row
    /// count totals.
    ///
    /// Mirrors the rule the pseudobulk splice Gibbs already applies
    /// (`graph-embedding-util/src/posterior/pb_gibbs/splice.rs`): `δ` needs
    /// counts on BOTH tracks. With no spliced counts only `β + δ` is pinned;
    /// with no unspliced counts `δ` enters no likelihood term at all and would
    /// be drawn straight from the prior. Reporting the count here — before any
    /// of the modelling exists — is what makes the go/no-go decidable.
    ///
    /// All-`false` on a non-channelized matrix: there is no second track to
    /// contrast against, which is a fact about the input, not a failure.
    #[must_use]
    pub fn delta_identified(&self, row_totals: &[f64]) -> Vec<bool> {
        debug_assert_eq!(row_totals.len(), self.n_rows());
        if !self.channelized {
            return vec![false; self.n_genes()];
        }
        let mut has_mature = vec![false; self.n_genes()];
        let mut has_nascent = vec![false; self.n_genes()];
        for (r, &v) in row_totals.iter().enumerate() {
            if v <= 0.0 {
                continue;
            }
            let g = self.gene_of_row(r);
            if self.row_is_nascent(r) {
                has_nascent[g] = true;
            } else {
                has_mature[g] = true;
            }
        }
        has_mature
            .into_iter()
            .zip(has_nascent)
            .map(|(m, n)| m && n)
            .collect()
    }

    /// The identifiability report, logged and returned. `None` when the input
    /// carries no channels — there is nothing to report, not a zero to report.
    pub fn report_delta_identifiability(&self, row_totals: &[f64]) -> Option<DeltaIdentifiability> {
        if !self.channelized {
            return None;
        }
        let identified = self.delta_identified(row_totals);
        let n_identified = identified.iter().filter(|&&x| x).count();
        let nascent_total: f64 = row_totals
            .iter()
            .enumerate()
            .filter(|&(r, _)| self.row_is_nascent(r))
            .map(|(_, &v)| v)
            .sum();
        let all_total: f64 = row_totals.iter().sum();
        let nascent_fraction = if all_total > 0.0 {
            nascent_total / all_total
        } else {
            0.0
        };
        info!(
            "Splice tracks: {}/{} genes carry counts on both tracks ({:.1}%); \
             the nascent track is {:.1}% of the library",
            n_identified,
            self.n_genes(),
            100.0 * n_identified as f64 / self.n_genes().max(1) as f64,
            100.0 * nascent_fraction
        );
        if n_identified == 0 {
            warn!(
                "No gene carries counts on both splice tracks, so nothing in this input \
                 pins a nascent-minus-mature contrast. The pooled fit below is still \
                 correct; a velocity read off it would not be."
            );
        }
        Some(DeltaIdentifiability {
            n_identified,
            nascent_fraction,
        })
    }
}

/// What the two tracks can and cannot pin, measured on the input rather than
/// assumed. This is the go/no-go for everything a velocity contrast would build
/// on it, so it is recorded in the run manifest and not only logged.
#[derive(Debug, Clone, Copy)]
pub struct DeltaIdentifiability {
    /// Genes with counts on BOTH tracks.
    pub n_identified: usize,
    /// Nascent share of the total library.
    pub nascent_fraction: f64,
}

#[cfg(test)]
#[path = "gene_axis/tests.rs"]
mod tests;
