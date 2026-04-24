//! Decoy gene-pair sampling for the gene-swap null.
//!
//! For each real (L, R) pair, draws `n_null` decoy pairs whose genes match the
//! targets on (mean expression, global Moran's I). Three swap schemes are
//! supported:
//!
//! - **Both**: both ligand and receptor are swapped to matched decoys.
//! - **Ligand-only**: ligand is swapped; receptor stays as the real R.
//! - **Receptor-only**: receptor is swapped; ligand stays as the real L.
//!
//! The default `Mixed` scheme draws `n_null` decoys split evenly across all
//! three sub-schemes, producing a composite null that probes deviation from
//! the specific (L, R) identity from three directions. The candidate pool
//! excludes any gene that appears in a real LR pair. If the matched pool is
//! smaller than needed for a given sub-scheme, tolerances are doubled (up to
//! 4 rounds) until enough candidates are found.

use crate::util::common::*;
use clap::ValueEnum;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// Which gene(s) to swap when building the null.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum NullScheme {
    /// Swap both ligand and receptor to matched decoys.
    Both,
    /// Swap ligand only; receptor stays as the real R.
    Ligand,
    /// Swap receptor only; ligand stays as the real L.
    Receptor,
    /// Draw n_null/3 from each of Both/Ligand/Receptor, then concatenate.
    Mixed,
}

pub struct MatcherContext {
    /// Pool of candidate gene indices (row indices into SparseIoVec rows).
    pub candidate_genes: Vec<usize>,
    /// Mean expression per candidate gene, same order as `candidate_genes`.
    pub means: Vec<f32>,
    /// Global Moran's I per candidate gene, same order as `candidate_genes`.
    pub moran: Vec<f32>,
    /// σ of `means` across the candidate pool — used to scale `expr_tol`.
    pub mean_sigma: f32,
}

impl MatcherContext {
    pub fn new(candidate_genes: Vec<usize>, means: Vec<f32>, moran: Vec<f32>) -> Self {
        let n = means.len() as f32;
        let mu = if n > 0.0 {
            means.iter().sum::<f32>() / n
        } else {
            0.0
        };
        let var = if n > 1.0 {
            means.iter().map(|m| (m - mu).powi(2)).sum::<f32>() / (n - 1.0)
        } else {
            1.0
        };
        Self {
            candidate_genes,
            means,
            moran,
            mean_sigma: var.sqrt().max(1e-6),
        }
    }

    /// Indices (into `candidate_genes`) of genes whose (mean, moran) lie within
    /// the given tolerances of the target.
    fn matched_indices(
        &self,
        target_mean: f32,
        target_moran: f32,
        expr_tol: f32,
        moran_tol: f32,
        exclude: &HashSet<usize>,
    ) -> Vec<usize> {
        let expr_thresh = expr_tol * self.mean_sigma;
        let mut out = Vec::new();
        for (idx, (&m, &i)) in self.means.iter().zip(self.moran.iter()).enumerate() {
            let gene_row = self.candidate_genes[idx];
            if exclude.contains(&gene_row) {
                continue;
            }
            if (m - target_mean).abs() > expr_thresh {
                continue;
            }
            if !i.is_finite() || (i - target_moran).abs() > moran_tol {
                continue;
            }
            out.push(idx);
        }
        out
    }

    /// Build the matched gene pool for a (mean, moran) target with tolerance
    /// doubling. Returns row indices into `candidate_genes` that are genuinely
    /// near the target. At most 4 rounds of doubling are attempted.
    fn matched_pool(
        &self,
        target_mean: f32,
        target_moran: f32,
        expr_tol: f32,
        moran_tol: f32,
        exclude: &HashSet<usize>,
        min_needed: usize,
    ) -> Vec<usize> {
        let mut etol = expr_tol;
        let mut mtol = moran_tol;
        let mut pool = self.matched_indices(target_mean, target_moran, etol, mtol, exclude);
        for _ in 0..3 {
            if pool.len() >= min_needed {
                break;
            }
            etol *= 2.0;
            mtol *= 2.0;
            pool = self.matched_indices(target_mean, target_moran, etol, mtol, exclude);
        }
        pool
    }

    /// Sample `n_null` decoy pairs for a real (L, R) under `target.scheme`.
    /// Candidate genes match the relevant target on (mean expression, global
    /// Moran's I); pairs that coincide with a real LR pair are rejected.
    pub fn sample_decoys(
        &self,
        target: &DecoyTarget<'_>,
        rng: &mut SmallRng,
    ) -> Vec<(usize, usize)> {
        match target.scheme {
            NullScheme::Mixed => {
                let each = target.n_null.div_ceil(3);
                let mut out: Vec<(usize, usize)> = Vec::with_capacity(target.n_null + 3);
                for scheme in [NullScheme::Both, NullScheme::Ligand, NullScheme::Receptor] {
                    let sub = DecoyTarget {
                        scheme,
                        n_null: each,
                        ..*target
                    };
                    out.extend(self.sample_one_scheme(&sub, rng));
                }
                out.shuffle(rng);
                out.truncate(target.n_null);
                out
            }
            _ => self.sample_one_scheme(target, rng),
        }
    }

    /// Single-scheme draw (not `Mixed`).
    fn sample_one_scheme(&self, t: &DecoyTarget<'_>, rng: &mut SmallRng) -> Vec<(usize, usize)> {
        // For Both we need the Cartesian product, so each pool only needs √n.
        // For Ligand/Receptor only one axis varies, so we need a full-sized pool.
        let need = match t.scheme {
            NullScheme::Both => ((t.n_null as f32).sqrt().ceil() as usize).max(4),
            _ => t.n_null.max(4),
        };

        let (swap_l, swap_r) = match t.scheme {
            NullScheme::Both => (true, true),
            NullScheme::Ligand => (true, false),
            NullScheme::Receptor => (false, true),
            NullScheme::Mixed => unreachable!("handled in caller"),
        };

        let lig_pool = if swap_l {
            self.matched_pool(
                t.l_mean,
                t.l_moran,
                t.expr_tol,
                t.moran_tol,
                t.exclude_genes,
                need,
            )
            .into_iter()
            .map(|i| self.candidate_genes[i])
            .filter(|&g| g != t.real_l && g != t.real_r)
            .collect()
        } else {
            vec![t.real_l]
        };
        let rec_pool = if swap_r {
            self.matched_pool(
                t.r_mean,
                t.r_moran,
                t.expr_tol,
                t.moran_tol,
                t.exclude_genes,
                need,
            )
            .into_iter()
            .map(|i| self.candidate_genes[i])
            .filter(|&g| g != t.real_r && g != t.real_l)
            .collect()
        } else {
            vec![t.real_r]
        };

        let mut decoys: Vec<(usize, usize)> = Vec::new();
        for &lg in &lig_pool {
            for &rg in &rec_pool {
                if lg == rg
                    || (lg == t.real_l && rg == t.real_r)
                    || t.known_pairs.contains(&(lg, rg))
                {
                    continue;
                }
                decoys.push((lg, rg));
            }
        }
        decoys.shuffle(rng);
        decoys.truncate(t.n_null);
        decoys
    }
}

/// Inputs for one decoy draw. Collapses the former 13-arg `sample_decoys`.
#[derive(Copy, Clone)]
pub struct DecoyTarget<'a> {
    pub real_l: usize,
    pub real_r: usize,
    pub l_mean: f32,
    pub l_moran: f32,
    pub r_mean: f32,
    pub r_moran: f32,
    pub expr_tol: f32,
    pub moran_tol: f32,
    pub n_null: usize,
    pub scheme: NullScheme,
    pub known_pairs: &'a HashSet<(usize, usize)>,
    pub exclude_genes: &'a HashSet<usize>,
}
