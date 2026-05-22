//! LOCO + TMLE deconfounded peak→gene statistics.
//!
//! The shared ATAC embedding (see [`crate::p2g::embed`]) reads a peak→gene
//! association off inner products in a low-rank topic space, so it only ever
//! sees *topic* co-variation: a cis peak and its co-active bystanders share the
//! same cell-type program, and the embedding cannot tell the causal enhancer
//! from the confounded neighbour. The deconfounded path instead asks what is
//! left of a peak after the genome-wide topic is removed — a peak's PRIVATE,
//! topic-orthogonal fluctuation is the only part that can reach its gene, so
//! confounded bystanders collapse while identifiable cis links survive.
//!
//! Estimator (Robinson partially-linear / Chernozhukov DML; named TMLE after
//! van der Laan & Rubin's targeted-learning framing): regress both the gene and
//! each peak on a topic confounder `W`, then read the partial association off
//! the residuals (residual-on-residual). The orthogonalization is
//! leave-one-chromosome-out (LOCO): the topic is genome-wide but a gene's cis
//! signal is local, so estimating `W₋c` from features OFF the gene's chromosome
//! captures the trans confounder without absorbing the cis effect we are
//! testing — the Neyman-orthogonality split that keeps the residual z honest.
//! `W₋c` is a joint RNA+ATAC pb-sample co-embedding by default (a two-modality
//! cell-state estimate; LOCO then drops off-chromosome genes too), or ATAC-only.

use crate::common::*;
use genomic_data::coordinates::chr_stripped;
use matrix_util::dmatrix_util::concatenate_vertical;
use nalgebra::DVector;
use std::collections::HashMap;

/// A pseudobulk modality and the chromosome of each of its features, for LOCO
/// row selection. `pb` is `[features × samples]`; `chr[f]` is feature `f`'s
/// chromosome (`None` ⇒ unplaced, kept in every off-chromosome set).
pub struct ModalityBlock<'a> {
    pub pb: &'a Mat,
    pub chr: &'a [Option<Box<str>>],
}

/// Leave-one-chromosome-out topic confounder bank.
///
/// For each chromosome `c` we embed the pseudobulk restricted to features NOT on
/// `c` (standardized log1p, rSVD) and keep the `[S×m]` orthonormal sample-factor
/// matrix `W₋c`. Genes on `c` residualize against `W₋c`, which holds the
/// trans/topic structure but none of `c`'s own cis fluctuation.
///
/// With one block (ATAC) this is the ATAC-only confounder; with two blocks
/// (ATAC + RNA) it is the **joint** confounder — a two-modality estimate of each
/// pb sample's cell state. The joint variant only stays Neyman-orthogonal
/// because LOCO drops `c`'s features in BOTH modalities (a chr-`c` gene's
/// expression carries the very cis signal under test).
pub struct LocoConfounders {
    by_chr: HashMap<Box<str>, Mat>,
    /// Fallback for unplaced peaks / unseen chromosomes: the all-features factors.
    global: Mat,
    pub m: usize,
}

impl LocoConfounders {
    /// Build the per-chromosome confounder bank. Pass one block (ATAC) for the
    /// ATAC-only confounder or two (ATAC + RNA) for the joint one; all blocks
    /// must share the same number of samples `S`.
    pub fn build(blocks: &[ModalityBlock], m: usize) -> anyhow::Result<Self> {
        let all_rows: Vec<Vec<usize>> =
            blocks.iter().map(|b| (0..b.pb.nrows()).collect()).collect();
        let global = loco_factors(blocks, &all_rows, m)?;

        let mut chrs: Vec<Box<str>> = blocks
            .iter()
            .flat_map(|b| b.chr.iter())
            .filter_map(|c| c.as_ref().map(|c| chr_stripped(c).into()))
            .collect();
        chrs.sort();
        chrs.dedup();

        // One off-chromosome embedding per chromosome. Sequential so the
        // row-subset copies stay one-at-a-time; the rSVD parallelizes internally.
        let mut by_chr = HashMap::new();
        for c in &chrs {
            let rows: Vec<Vec<usize>> = blocks
                .iter()
                .map(|b| off_chromosome_rows(b.chr, c))
                .collect();
            let total: usize = rows.iter().map(|r| r.len()).sum();
            // Degenerate (nearly all features on this chr): fall back to global.
            let w = if total < m.max(2) {
                global.clone()
            } else {
                loco_factors(blocks, &rows, m)?
            };
            by_chr.insert(c.clone(), w);
        }

        Ok(Self { by_chr, global, m })
    }

    /// Confounder `W₋c` for a gene on chromosome `chr` (global fallback if the
    /// chromosome carried no features).
    pub fn get(&self, chr: &str) -> &Mat {
        self.by_chr.get(chr_stripped(chr)).unwrap_or(&self.global)
    }
}

/// Rows of `chr` whose feature is NOT on chromosome `c` (unplaced features kept).
fn off_chromosome_rows(chr: &[Option<Box<str>>], c: &str) -> Vec<usize> {
    chr.iter()
        .enumerate()
        .filter(|(_, pc)| pc.as_ref().map(|x| chr_stripped(x) != c).unwrap_or(true))
        .map(|(i, _)| i)
        .collect()
}

/// Sample-factor matrix `V [S×m]` of the standardized, block-balanced stack of
/// each block's selected rows. Each feature is log1p z-scored over samples and
/// each block is scaled by `1/√(#rows)` so neither modality dominates the joint
/// factorization; the rSVD then yields the shared cell-state axes.
fn loco_factors(blocks: &[ModalityBlock], rows: &[Vec<usize>], m: usize) -> anyhow::Result<Mat> {
    let standardized: Vec<Mat> = blocks
        .iter()
        .zip(rows)
        .map(|(b, r)| standardize_block(b.pb, r))
        .collect();
    let stacked = concatenate_vertical(&standardized)?;
    let (_, _, v) = stacked.rsvd(m)?;
    Ok(v)
}

/// log1p, z-score each selected row over samples (mean 0, unit var; zero-var →
/// 0), then scale the block by `1/√(#rows)`. Row-centering keeps the factors
/// `⊥ 1` so residualizing against them preserves mean-zero residuals.
fn standardize_block(pb: &Mat, rows: &[usize]) -> Mat {
    let s = pb.ncols();
    let nr = rows.len();
    let mut out = Mat::zeros(nr, s);
    for (i, &r) in rows.iter().enumerate() {
        let mut row = vec![0f32; s];
        let mut mean = 0.0f32;
        for (j, x) in row.iter_mut().enumerate() {
            *x = (pb[(r, j)] + 1.0).ln();
            mean += *x;
        }
        mean /= s as f32;
        let var =
            row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / (s as f32 - 1.0).max(1.0);
        let sd = var.sqrt();
        if sd < 1e-8 {
            continue; // constant feature → all-zero standardized row
        }
        let inv = 1.0 / sd;
        for (j, &x) in row.iter().enumerate() {
            out[(i, j)] = (x - mean) * inv;
        }
    }
    out.scale_mut(1.0 / (nr.max(1) as f32).sqrt());
    out
}

/// Centered log1p of a pseudobulk row over its `S` samples (mean 0).
pub fn centered_log1p(v: &[f32]) -> DVec {
    let s = v.len();
    let mut x = DVector::from_iterator(s, v.iter().map(|&a| (a + 1.0).ln()));
    let mean = x.mean();
    x.add_scalar_mut(-mean);
    x
}

/// Residualize a centered vector on the orthonormal confounder `W [S×m]`:
/// `x − W (Wᵀ x)`. With `W ⊥ 1` (true of rSVD factors of a row-centered matrix)
/// the residual stays mean-zero.
fn residualize(w: &Mat, x: &DVec) -> DVec {
    x - &(w * (w.transpose() * x))
}

/// LOCO+TMLE deconfounded marginal z + peak–peak LD `R` for a gene's cis peaks.
///
/// Project the gene and each cis peak onto the orthogonal complement of the
/// off-chromosome topic confounder `W`, then read the partial association `z`
/// (residual-on-residual t-statistic, `df = S − m − 2`) and the
/// partial-correlation LD off the residuals.
///
/// `gene_clog` and `peak_clog[..]` are pre-centered log1p `[S]`; `cis` indexes
/// into `peak_clog`. Returns `(z [C], R [C×C], n_eff = S − m)`.
pub fn cis_link_stats_tmle(
    gene_clog: &DVec,
    peak_clog: &[DVec],
    cis: &[usize],
    w: &Mat,
) -> (Vec<f32>, Mat, f64) {
    let s = gene_clog.len();
    let m = w.ncols();
    let c = cis.len();

    let gx_r = residualize(w, gene_clog);
    let cl_r: Vec<DVec> = cis.iter().map(|&p| residualize(w, &peak_clog[p])).collect();

    let df = (s as f64 - m as f64 - 2.0).max(1.0);
    let gnorm = gx_r.norm() as f64;
    let pnorm: Vec<f64> = cl_r.iter().map(|v| v.norm() as f64).collect();

    // residual-on-residual partial-correlation t-statistic (z-scale).
    let z: Vec<f32> = (0..c)
        .map(|i| {
            if gnorm < 1e-8 || pnorm[i] < 1e-8 {
                return 0.0;
            }
            let r = (gx_r.dot(&cl_r[i]) as f64 / (gnorm * pnorm[i])).clamp(-0.999, 0.999);
            (r * (df / (1.0 - r * r)).sqrt()) as f32
        })
        .collect();

    // partial-correlation LD = cosine of the residualized peak profiles.
    let mut r_ld = Mat::identity(c, c);
    for i in 0..c {
        if pnorm[i] < 1e-8 {
            r_ld[(i, i)] = 0.0;
        }
        for j in (i + 1)..c {
            let v = if pnorm[i] < 1e-8 || pnorm[j] < 1e-8 {
                0.0
            } else {
                (cl_r[i].dot(&cl_r[j]) as f64 / (pnorm[i] * pnorm[j])) as f32
            };
            r_ld[(i, j)] = v;
            r_ld[(j, i)] = v;
        }
    }

    (z, r_ld, (s - m) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    /// On topic-confounded data — gene driven by its causal peaks' PRIVATE
    /// fluctuation, confounded bystanders sharing only the topic — the LOCO
    /// residual z must keep causal peaks and collapse confounded ones, where the
    /// marginal correlation cannot. This is the in-crate counterpart of the
    /// `loco_dml_controls_confounded_fdp` end-to-end prototype.
    #[test]
    fn residual_z_separates_causal_from_confounded() {
        let s = 400usize;
        let k = 5usize; // genome-wide topics
        let n_chr = 8usize;
        let per_chr = 60usize;
        let p = n_chr * per_chr;
        let m = 10usize;

        let mut rng = SmallRng::seed_from_u64(2026);
        let randn = |rng: &mut SmallRng| -> f32 {
            let v: f64 = StandardNormal.sample(rng);
            v as f32
        };

        let topics: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..s).map(|_| randn(&mut rng)).collect())
            .collect();
        let chr: Vec<usize> = (0..p).map(|pp| pp / per_chr).collect();
        let ptopic: Vec<usize> = (0..p).map(|pp| pp % k).collect();
        let spec: Vec<Vec<f32>> = (0..p)
            .map(|_| (0..s).map(|_| randn(&mut rng)).collect())
            .collect();

        let mut atac = Mat::zeros(p, s);
        for pp in 0..p {
            let t = &topics[ptopic[pp]];
            for ss in 0..s {
                atac[(pp, ss)] = (3.0 + 1.2 * t[ss] + 0.8 * spec[pp][ss]).max(0.01);
            }
        }
        let peak_chr: Vec<Option<Box<str>>> = (0..p)
            .map(|pp| Some(format!("chr{}", chr[pp]).into()))
            .collect();
        let loco = LocoConfounders::build(
            &[ModalityBlock {
                pb: &atac,
                chr: &peak_chr,
            }],
            m,
        )
        .unwrap();
        let peak_clog: Vec<DVec> = (0..p)
            .map(|pp| centered_log1p(&atac.row(pp).iter().copied().collect::<Vec<_>>()))
            .collect();

        // gene on chr 0, topic 0: driven by 3 causal peaks' private signal.
        let cg = 0usize;
        let tg = 0usize;
        let on_chr_topic: Vec<usize> = (0..p)
            .filter(|&pp| chr[pp] == cg && ptopic[pp] == tg)
            .collect();
        let causal: Vec<usize> = on_chr_topic[..3].to_vec();
        let confound: Vec<usize> = on_chr_topic[3..6].to_vec();

        let mut gene = vec![0f32; s];
        for ss in 0..s {
            let mut v = 3.0;
            for &cp in &causal {
                v += 1.2 * topics[tg][ss] + 0.8 * spec[cp][ss];
            }
            v += 0.4 * randn(&mut rng);
            gene[ss] = v.max(0.01);
        }
        let gene_clog = centered_log1p(&gene);

        let cis: Vec<usize> = causal.iter().chain(&confound).copied().collect();
        let w = loco.get("chr0");
        let (z, _r, n_eff) = cis_link_stats_tmle(&gene_clog, &peak_clog, &cis, w);

        let mean_causal: f32 =
            z[..causal.len()].iter().map(|z| z.abs()).sum::<f32>() / causal.len() as f32;
        let mean_confound: f32 =
            z[causal.len()..].iter().map(|z| z.abs()).sum::<f32>() / confound.len() as f32;
        assert!(n_eff <= (s - m) as f64 + 1.0);
        assert!(
            mean_causal > 3.0,
            "causal residual z too small: {mean_causal:.2}"
        );
        assert!(
            mean_causal > 2.0 * mean_confound,
            "residual z did not separate causal ({mean_causal:.2}) from confounded ({mean_confound:.2})"
        );
    }
}
