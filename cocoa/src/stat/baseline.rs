//! Stage-1 baseline: each individual's expected count at its own cell states.
//!
//! Per topic and gene, the pseudobulk table is fit by a rank-1 Poisson model
//!
//! ```text
//!   y(i, p) ~ Poisson( n(i, p) * mu(p) * omega(i) )
//! ```
//!
//! with `mu(p)` the cell-state rate of pseudobulk `p` and `omega(i)` a free
//! multiplier per individual. It uses no exposure labels, so it is computed
//! once and shared by every permutation draw. Pseudobulks pool individuals,
//! so the individual x pseudobulk table identifies `mu` and `omega` up to one
//! scale per gene, which cancels in every reported effect. The fit
//! alternates the two closed-form updates
//!
//! ```text
//!   mu(p)     = (y(., p) + a0) / (sum_i omega(i) n(i, p) + b0)
//!   omega(i)  = (y(i, .) + a0) / (sum_p mu(p) n(i, p) + b0)
//! ```
//!
//! (Gamma(a0, b0) pseudo-counts keep sparse genes finite), with `omega`
//! rescaled to mean one per gene after every sweep. Stage 2 takes the offset
//! `m(i) = sum_p mu(p) n(i, p)`.

use super::*;

#[cfg(test)]
mod tests;

/// Stop once the largest relative change of any offset falls below this.
const OFFSET_TOL: f32 = 1e-5;

/// The stage-1 fit of one topic.
pub struct Baseline {
    /// m(d,i) = sum_p mu(d,p) n(i,p), gene x individual
    pub indv_offset: Mat,
    /// log mean count per cell per gene over all the topic's cells
    pub log_mean: DVec,
}

impl CocoaStat {
    /// Fit the rank-1 baseline of every topic, in parallel over topics.
    pub fn estimate_baseline(&self) -> Vec<Baseline> {
        (0..self.n_topics())
            .into_par_iter()
            .map(|k| self.baseline_one_topic(k))
            .collect()
    }

    fn baseline_one_topic(&self, k: usize) -> Baseline {
        let y_dp = self.y1_stat(k);
        let y_di = self.indv_y1_stat(k);
        let n_ip = self.indv_size_stat(k);
        let n_pi = n_ip.transpose();
        let (n_genes, n_indv) = y_di.shape();
        let (a0, b0) = (self.a0, self.b0);

        let mut omega = Mat::from_element(n_genes, n_indv, 1.0);
        let mut mu = Mat::zeros(n_genes, n_ip.ncols());
        let mut m = Mat::zeros(n_genes, n_indv);
        let mut prev = Mat::from_element(n_genes, n_indv, f32::INFINITY);
        for _ in 0..self.n_opt_iter {
            // mu(d,p) = (y(d,p) + a0) / (sum_i omega(d,i) n(i,p) + b0)
            omega.mul_to(n_ip, &mut mu);
            mu.zip_apply(y_dp, |den, y| *den = (y + a0) / (*den + b0));
            // m(d,i) = sum_p mu(d,p) n(i,p);  omega = (y(d,i) + a0) / (m + b0)
            mu.mul_to(&n_pi, &mut m);
            omega.copy_from(y_di);
            omega.zip_apply(&m, |w, den| *w = (*w + a0) / (den + b0));
            // fix the gene scale: omega has mean one per gene
            for d in 0..n_genes {
                let s = omega.row(d).mean();
                if s > 0.0 {
                    omega.row_mut(d).scale_mut(1.0 / s);
                    mu.row_mut(d).scale_mut(s);
                    m.row_mut(d).scale_mut(s);
                }
            }
            let done = m
                .iter()
                .zip(prev.iter())
                .all(|(now, was)| (now - was).abs() <= OFFSET_TOL * was.abs().max(1e-8));
            if done {
                break;
            }
            prev.copy_from(&m);
        }

        // log mean count per cell per gene; half a count keeps unexpressed
        // genes finite
        let total_cells = n_ip.sum().max(f32::EPSILON);
        let log_mean = y_di.column_sum().map(|y| ((y + 0.5) / total_cells).ln());
        Baseline {
            indv_offset: m,
            log_mean,
        }
    }
}
