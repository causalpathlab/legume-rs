//! Continuous output during sampling: a fraction trace, per-sweep fit scalars,
//! and periodic checkpoints of the running posterior mean.
//!
//! Disk is cheap next to a thousand-sweep run, so the sampler writes as it goes
//! rather than only at the end. The trace covers warmup as well as the retained
//! draws — burn-in is the part worth looking at, and a reference drifting off
//! its prior shows up there first.

use crate::embed_common::Mat;
use anyhow::{Context, Result};
use matrix_util::common_io::open_buf_writer;
use std::io::Write;

/// How the trace and checkpoints are emitted.
pub struct MonitorConfig {
    /// Write a trace row every this many sweeps; 0 disables the trace.
    pub trace_every: usize,
    /// Re-write the fraction tables every this many sweeps; 0 disables it.
    pub checkpoint_every: usize,
}

pub struct Monitor {
    cfg: MonitorConfig,
    out: String,
    sample_names: Vec<Box<str>>,
    celltype_names: Vec<Box<str>>,
    trace: Option<Box<dyn Write>>,
    fit: Option<Box<dyn Write>>,
    /// Which chain is currently sampling. Pooled chains each restart their sweep
    /// counter, so without this the trace cannot be split back apart.
    chain: usize,
}

impl Monitor {
    pub fn new(
        cfg: MonitorConfig,
        out: &str,
        sample_names: &[Box<str>],
        celltype_names: &[Box<str>],
    ) -> Result<Self> {
        let (mut trace, mut fit) = (None, None);
        if cfg.trace_every > 0 {
            let path = format!("{out}.trace.tsv.gz");
            let mut w = open_buf_writer(&path).with_context(|| format!("creating {path}"))?;
            writeln!(w, "chain\tsweep\tphase\tsample\tcelltype\tfraction")?;
            trace = Some(w);

            let path = format!("{out}.trace_fit.tsv");
            let mut w = open_buf_writer(&path).with_context(|| format!("creating {path}"))?;
            writeln!(w, "chain\tsweep\tphase\tmean_anchor_drift")?;
            fit = Some(w);
        }
        Ok(Self {
            cfg,
            out: out.to_string(),
            sample_names: sample_names.to_vec(),
            celltype_names: celltype_names.to_vec(),
            trace,
            fit,
            chain: 0,
        })
    }

    /// Disabled monitor, for unit tests and callers that want no side files.
    #[cfg(test)]
    #[must_use]
    pub fn silent() -> Self {
        Self {
            cfg: MonitorConfig {
                trace_every: 0,
                checkpoint_every: 0,
            },
            out: String::new(),
            sample_names: Vec::new(),
            celltype_names: Vec::new(),
            trace: None,
            fit: None,
            chain: 0,
        }
    }

    pub fn begin_chain(&mut self, chain: usize) {
        self.chain = chain;
    }

    /// Append one trace block for sweep `it`. `frac_flat` is `[si*C + ct]`.
    pub fn record(
        &mut self,
        it: usize,
        past_warmup: bool,
        frac_flat: &[f32],
        drift: f32,
    ) -> Result<()> {
        if self.cfg.trace_every == 0 || !it.is_multiple_of(self.cfg.trace_every) {
            return Ok(());
        }
        let phase = if past_warmup { "draw" } else { "warmup" };
        let (chain, c) = (self.chain, self.celltype_names.len());
        if let Some(w) = self.trace.as_mut() {
            for (si, sname) in self.sample_names.iter().enumerate() {
                for (ct, cname) in self.celltype_names.iter().enumerate() {
                    writeln!(
                        w,
                        "{chain}\t{it}\t{phase}\t{sname}\t{cname}\t{:.6}",
                        frac_flat[si * c + ct]
                    )?;
                }
            }
        }
        if let Some(w) = self.fit.as_mut() {
            writeln!(w, "{chain}\t{it}\t{phase}\t{drift:.6}")?;
        }
        Ok(())
    }

    /// Re-write the posterior-mean fractions from the running sums, so a long
    /// run stays inspectable and can be interrupted without losing everything.
    ///
    /// Uses the same writer as the final table, so a checkpoint and the finished
    /// file cannot drift into different formats.
    pub fn checkpoint(&mut self, it: usize, frac_sum: &[f64], n_collect: usize) -> Result<()> {
        if self.cfg.checkpoint_every == 0 || !it.is_multiple_of(self.cfg.checkpoint_every) {
            return Ok(());
        }
        let (s, c) = (self.sample_names.len(), self.celltype_names.len());
        let nc = n_collect as f64;
        let mat = Mat::from_fn(s, c, |si, ct| (frac_sum[si * c + ct] / nc) as f32);
        super::io::write_wide_tsv(
            &format!("{}.fractions.tsv", self.out),
            "sample",
            &self.sample_names,
            &self.celltype_names,
            &mat,
        )
    }

    /// Flush and close the trace writers.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(w) = self.trace.as_mut() {
            w.flush()?;
        }
        if let Some(w) = self.fit.as_mut() {
            w.flush()?;
        }
        Ok(())
    }
}
