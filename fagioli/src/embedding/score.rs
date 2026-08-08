//! Scoring a new cohort from a fitted embedding.
//!
//! Because `B = U V'` is already direct, LD-free, per-SD effects, the PRS is
//! `ŷ_t = X_new U v_t`. The association matters:
//!
//! ```text
//! S = X_new U   (n x H)      then    Ŷ = S V̌'   (n x T)
//! ```
//!
//! Each individual is scored **once** into `H` program scores; every trait is
//! then an `H`-vector readout of `S`. Storage drops from `M x T` weights to
//! `M x H + T x H`, and a trait added later costs `H` numbers rather than a
//! genome-wide refit.
//!
//! `U v_t` is invariant to the reparameterisation `U→UA, V̌→V̌A⁻ᵀ`, so the PRS is
//! identified even though `U` and `V̌` individually are not. Per-trait scale is
//! also irrelevant — a PRS is scale-free within a trait — which is why `V̌` can
//! be used directly without undoing `Λ_N`.
//!
//! # The two traps
//!
//! **Standardise with the panel's constants, not the new cohort's.** `U` is
//! defined against the panel's `(μ_j, σ_j)`; re-standardising `X_new` in-sample
//! silently rescales every variant. [`PanelStandardization`] carries them.
//!
//! **Impute after standardising, not before.** `BedReader` mean-imputes missing
//! genotypes using *in-sample* column means. The correct imputation here is the
//! panel mean, which after standardising is exactly zero.

use anyhow::{ensure, Result};
use log::{info, warn};
use nalgebra::DMatrix;
use rustc_hash::FxHashMap as HashMap;

use crate::genotype::GenotypeMatrix;
use crate::util::{alleles_match, chr_stripped, AlleleMatch};

/// Everything about the reference panel a fitted `U` is defined against.
///
/// These are the *empirical* constants used at fit time, not values recomputed
/// from a formula, so that scoring exactly inverts the standardisation the fit
/// applied.
#[derive(Debug, Clone)]
pub struct PanelStandardization {
    pub chromosomes: Vec<Box<str>>,
    pub positions: Vec<u64>,
    pub snp_ids: Vec<Box<str>>,
    pub allele1: Vec<Box<str>>,
    pub allele2: Vec<Box<str>>,
    /// Column mean of the panel genotypes.
    pub mean: Vec<f32>,
    /// Column standard deviation of the panel genotypes.
    pub sd: Vec<f32>,
}

impl PanelStandardization {
    /// Capture the standardisation from the panel used to fit.
    pub fn from_panel(geno: &GenotypeMatrix) -> Self {
        let n = geno.genotypes.nrows() as f32;
        let m = geno.num_snps();
        let mut mean = Vec::with_capacity(m);
        let mut sd = Vec::with_capacity(m);
        for j in 0..m {
            let col = geno.genotypes.column(j);
            let mu = col.iter().sum::<f32>() / n;
            let var = col.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / n;
            mean.push(mu);
            sd.push(var.sqrt());
        }
        Self {
            chromosomes: geno.chromosomes.clone(),
            positions: geno.positions.clone(),
            snp_ids: geno.snp_ids.clone(),
            allele1: geno.allele1.clone(),
            allele2: geno.allele2.clone(),
            mean,
            sd,
        }
    }

    pub fn num_snps(&self) -> usize {
        self.mean.len()
    }
}

/// What scoring produced.
#[derive(Debug, Clone)]
pub struct ProgramScores {
    /// `S = X_new U`, shape (n, H) — the per-individual program profile.
    pub scores: DMatrix<f32>,
    /// `Ŷ = S V̌'`, shape (n, T).
    pub prs: DMatrix<f32>,
    pub individual_ids: Vec<Box<str>>,
    /// Variants matched to the panel.
    pub matched: usize,
    /// Fraction of `Σ_g ‖u_g‖²` that the matched variants carry, per program.
    ///
    /// Reported instead of a variant count because `U` is sparse: matching half
    /// the variants can capture 95% of the signal or 5% of it, and the count
    /// cannot tell you which.
    pub coverage: Vec<f32>,
    pub flipped: usize,
    pub dropped: usize,
}

/// Score a new cohort against a fitted embedding.
///
/// `u` is the genome-wide variant loading (M x H) in the panel's SNP ordering;
/// `v_check` is the fitted trait embedding (T x H).
pub fn score_cohort(
    u: &DMatrix<f32>,
    v_check: &DMatrix<f32>,
    panel: &PanelStandardization,
    cohort: &GenotypeMatrix,
) -> Result<ProgramScores> {
    let m = panel.num_snps();
    let h = u.ncols();
    ensure!(
        u.nrows() == m,
        "U has {} rows but the panel has {} SNPs",
        u.nrows(),
        m
    );
    ensure!(
        v_check.ncols() == h,
        "V̌ has {} columns but U has {}",
        v_check.ncols(),
        h
    );

    // Panel lookup by (chr, pos), falling back to SNP id.
    let mut by_pos: HashMap<(&str, u64), usize> = HashMap::default();
    for j in 0..m {
        by_pos
            .entry((chr_stripped(&panel.chromosomes[j]), panel.positions[j]))
            .or_insert(j);
    }
    let by_id: HashMap<&str, usize> = panel
        .snp_ids
        .iter()
        .enumerate()
        .map(|(j, s)| (s.as_ref(), j))
        .collect();

    let n = cohort.num_individuals();
    let mut scores = DMatrix::<f32>::zeros(n, h);
    let mut matched_mass = vec![0.0f32; h];
    let (mut matched, mut flipped, mut dropped) = (0usize, 0usize, 0usize);

    for c in 0..cohort.num_snps() {
        let key = (chr_stripped(&cohort.chromosomes[c]), cohort.positions[c]);
        let Some(&j) = by_pos
            .get(&key)
            .or_else(|| by_id.get(cohort.snp_ids[c].as_ref()))
        else {
            continue;
        };

        // Allele alignment. Unlike the sumstat path a flip acts on the
        // *dosage*, and it must be applied to the RAW dosage before
        // standardising: negating the standardised value instead gives
        // −((2−x)−μ)/σ = (x−2+μ)/σ, which is off by a constant per variant.
        let flip = match alleles_match(
            &cohort.allele1[c],
            &cohort.allele2[c],
            &panel.allele1[j],
            &panel.allele2[j],
        ) {
            AlleleMatch::Same => false,
            AlleleMatch::Flipped => {
                flipped += 1;
                true
            }
            AlleleMatch::Ambiguous | AlleleMatch::Mismatch => {
                dropped += 1;
                continue;
            }
        };

        let sd = panel.sd[j];
        if sd <= 0.0 || !sd.is_finite() {
            continue;
        }
        matched += 1;
        for hh in 0..h {
            matched_mass[hh] += u[(j, hh)] * u[(j, hh)];
        }

        // Align the dosage, then x̃ = (x − μ_panel) / σ_panel. Missing values
        // become 0 *after* standardising, i.e. imputed to the panel mean --
        // BedReader's own in-sample mean imputation would be the wrong centre.
        for i in 0..n {
            let raw = cohort.genotypes[(i, c)];
            let x = if raw.is_finite() {
                let aligned = if flip { 2.0 - raw } else { raw };
                (aligned - panel.mean[j]) / sd
            } else {
                0.0
            };
            if x != 0.0 {
                for hh in 0..h {
                    scores[(i, hh)] += x * u[(j, hh)];
                }
            }
        }
    }

    let total_mass: Vec<f32> = (0..h)
        .map(|hh| (0..m).map(|j| u[(j, hh)] * u[(j, hh)]).sum())
        .collect();
    let coverage: Vec<f32> = matched_mass
        .iter()
        .zip(&total_mass)
        .map(|(a, b)| if *b > 0.0 { a / b } else { 0.0 })
        .collect();

    let prs = &scores * v_check.transpose();

    info!(
        "Scored {} individuals: {}/{} variants matched ({} flipped, {} dropped); \
         per-program ‖u‖² coverage {:?}",
        n,
        matched,
        cohort.num_snps(),
        flipped,
        dropped,
        coverage.iter().map(|c| format!("{c:.3}")).collect::<Vec<_>>(),
    );
    if coverage.iter().any(|&c| c < 0.5) {
        warn!(
            "A program has under half its effect mass in the scored cohort; the PRS \
             for traits loading on it is not comparable to one scored on the panel."
        );
    }

    Ok(ProgramScores {
        scores,
        prs,
        individual_ids: cohort.individual_ids.clone(),
        matched,
        coverage,
        flipped,
        dropped,
    })
}

/// Concatenate per-block `E[U_b]` into the genome-wide `U`, in SNP order.
pub fn assemble_u(u_blocks: &[DMatrix<f32>], snp_starts: &[usize], num_snps: usize) -> DMatrix<f32> {
    let h = u_blocks.first().map_or(0, DMatrix::ncols);
    let mut u = DMatrix::<f32>::zeros(num_snps, h);
    for (blk, &start) in u_blocks.iter().zip(snp_starts) {
        for g in 0..blk.nrows() {
            if start + g >= num_snps {
                break;
            }
            for hh in 0..h {
                u[(start + g, hh)] = blk[(g, hh)];
            }
        }
    }
    u
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;
