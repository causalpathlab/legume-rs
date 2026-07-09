//! Online quantile estimation via the **P² algorithm** (Jain & Chlamtac, 1985):
//! an arbitrary target quantile from a stream using five markers, O(1) memory,
//! no stored samples. Companion to the running mean/variance in
//! [`crate::traits::RunningStatOps`] for cases (e.g. MCMC credible intervals)
//! where storing every draw to sort for a quantile is too expensive.

/// Single-quantile P² estimator for a stream of scalars.
///
/// ```
/// use matrix_util::running_quantile::P2Quantile;
/// let mut med = P2Quantile::new(0.5);
/// for i in 0..1000 { med.add(f64::from(i)); }
/// assert!((med.quantile() - 500.0).abs() < 25.0);
/// ```
#[derive(Clone, Debug)]
pub struct P2Quantile {
    p: f64,
    /// First ≤5 observations, buffered until the markers are initialized.
    init: Vec<f64>,
    q: [f64; 5],  // marker heights
    n: [f64; 5],  // marker positions (integer-valued)
    np: [f64; 5], // desired marker positions
    dn: [f64; 5], // desired-position increments
    count: usize,
}

impl P2Quantile {
    #[must_use]
    pub fn new(p: f64) -> Self {
        let p = p.clamp(0.0, 1.0);
        Self {
            p,
            init: Vec::with_capacity(5),
            q: [0.0; 5],
            n: [1.0, 2.0, 3.0, 4.0, 5.0],
            np: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            count: 0,
        }
    }

    pub fn add(&mut self, x: f64) {
        self.count += 1;
        if self.count <= 5 {
            self.init.push(x);
            if self.count == 5 {
                self.init.sort_by(f64::total_cmp);
                self.q.copy_from_slice(&self.init);
            }
            return;
        }
        // 1. Locate the cell k containing x, adjusting the extreme markers.
        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x >= self.q[4] {
            self.q[4] = x;
            3
        } else {
            (0..4)
                .find(|&i| self.q[i] <= x && x < self.q[i + 1])
                .unwrap_or(3)
        };
        // 2. Increment positions of the markers above the cell.
        for i in (k + 1)..5 {
            self.n[i] += 1.0;
        }
        // 3. Advance the desired positions.
        for i in 0..5 {
            self.np[i] += self.dn[i];
        }
        // 4. Adjust the three interior markers (parabolic, or linear if it
        //    would break monotonicity).
        for i in 1..4 {
            let d = self.np[i] - self.n[i];
            let go_up = d >= 1.0 && (self.n[i + 1] - self.n[i]) > 1.0;
            let go_dn = d <= -1.0 && (self.n[i - 1] - self.n[i]) < -1.0;
            if go_up || go_dn {
                let ds = if d >= 0.0 { 1.0 } else { -1.0 };
                let qp = self.parabolic(i, ds);
                self.q[i] = if self.q[i - 1] < qp && qp < self.q[i + 1] {
                    qp
                } else {
                    self.linear(i, ds)
                };
                self.n[i] += ds;
            }
        }
    }

    fn parabolic(&self, i: usize, d: f64) -> f64 {
        let (q, n) = (&self.q, &self.n);
        q[i] + d / (n[i + 1] - n[i - 1])
            * ((n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
                + (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / (n[i] - n[i - 1]))
    }

    fn linear(&self, i: usize, d: f64) -> f64 {
        let j = if d > 0.0 { i + 1 } else { i - 1 };
        self.q[i] + d * (self.q[j] - self.q[i]) / (self.n[j] - self.n[i])
    }

    /// Current estimate of the target quantile.
    #[must_use]
    pub fn quantile(&self) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        if self.count < 5 {
            let mut v = self.init.clone();
            v.sort_by(f64::total_cmp);
            let idx = (((v.len() - 1) as f64) * self.p).round() as usize;
            return v[idx.min(v.len() - 1)];
        }
        self.q[2]
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Per-row P² trackers for a set of target quantiles, fed one dense column
/// (observation vector) at a time — mirrors the column-add convention of
/// [`crate::sparse_stat::SparseRunningStatistics`]. Row `r`'s estimate of the
/// `qi`-th quantile is `quantile(qi)[r]`.
pub struct RunningQuantiles {
    nrows: usize,
    quantiles: Vec<f64>,
    est: Vec<P2Quantile>, // row-major: [row * n_quantiles + qi]
}

impl RunningQuantiles {
    #[must_use]
    pub fn new(nrows: usize, quantiles: &[f64]) -> Self {
        let est = (0..nrows)
            .flat_map(|_| quantiles.iter().map(|&p| P2Quantile::new(p)))
            .collect();
        Self {
            nrows,
            quantiles: quantiles.to_vec(),
            est,
        }
    }

    /// Feed one observation vector (length `nrows`).
    pub fn add_dense_column(&mut self, values: &[f32]) {
        assert_eq!(values.len(), self.nrows, "column length != nrows");
        let nq = self.quantiles.len();
        for (row, &v) in values.iter().enumerate() {
            for qi in 0..nq {
                self.est[row * nq + qi].add(f64::from(v));
            }
        }
    }

    /// Per-row estimate of the `qi`-th target quantile.
    #[must_use]
    pub fn quantile(&self, qi: usize) -> Vec<f32> {
        let nq = self.quantiles.len();
        (0..self.nrows)
            .map(|row| self.est[row * nq + qi].quantile() as f32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic near-uniform permutation of 0..m so the empirical CDF is
    // known: the p-quantile is ≈ p·m.
    fn perm(m: u64) -> impl Iterator<Item = f64> {
        (0..m).map(move |i| ((i.wrapping_mul(7919)) % m) as f64)
    }

    #[test]
    fn recovers_quantiles_within_tolerance() {
        let m = 10_007u64;
        for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            let mut est = P2Quantile::new(p);
            for x in perm(m) {
                est.add(x);
            }
            let want = p * m as f64;
            let got = est.quantile();
            assert!(
                (got - want).abs() < 0.02 * m as f64,
                "p={p}: got {got:.1}, want ≈ {want:.1}"
            );
        }
    }

    #[test]
    fn running_quantiles_per_row() {
        // Two rows: row 0 ~ U(0,m), row 1 = row0 + m (shifted). Check both.
        let m = 5_003u64;
        let mut rq = RunningQuantiles::new(2, &[0.5]);
        for x in perm(m) {
            rq.add_dense_column(&[x as f32, x as f32 + m as f32]);
        }
        let med = rq.quantile(0);
        assert!((med[0] - 0.5 * m as f32).abs() < 0.02 * m as f32);
        assert!((med[1] - 1.5 * m as f32).abs() < 0.02 * m as f32);
    }

    #[test]
    fn handles_fewer_than_five() {
        let mut est = P2Quantile::new(0.5);
        est.add(3.0);
        est.add(1.0);
        est.add(2.0);
        assert!((est.quantile() - 2.0).abs() < 1e-9);
    }
}
