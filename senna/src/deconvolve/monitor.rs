//! Continuous output during sampling: a fraction trace, per-sweep fit scalars,
//! and periodic checkpoints of the running posterior mean.
//!
//! Disk is cheap next to a thousand-sweep run, so the sampler writes as it goes
//! rather than only at the end. The trace covers warmup as well as the retained
//! draws — burn-in is the part worth looking at, and a drifting reference shows
//! up there first.

use crate::embed_common::Mat;
use anyhow::{Context, Result};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{BufWriter, Write};

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
    trace: Option<BufWriter<GzEncoder<File>>>,
    fit: Option<BufWriter<File>>,
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
            let file =
                File::create(&path).with_context(|| format!("creating trace file {path}"))?;
            let mut w = BufWriter::new(GzEncoder::new(file, Compression::default()));
            writeln!(w, "sweep\tphase\tsample\tcelltype\tfraction")?;
            trace = Some(w);

            let path = format!("{out}.trace_fit.tsv");
            let file = File::create(&path).with_context(|| format!("creating fit file {path}"))?;
            let mut w = BufWriter::new(file);
            writeln!(w, "sweep\tphase\tmean_anchor_drift")?;
            fit = Some(w);
        }
        Ok(Self {
            cfg,
            out: out.to_string(),
            sample_names: sample_names.to_vec(),
            celltype_names: celltype_names.to_vec(),
            trace,
            fit,
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
        }
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
        let c = self.celltype_names.len();
        if let Some(w) = self.trace.as_mut() {
            for (si, sname) in self.sample_names.iter().enumerate() {
                for (ct, cname) in self.celltype_names.iter().enumerate() {
                    writeln!(
                        w,
                        "{it}\t{phase}\t{sname}\t{cname}\t{:.6}",
                        frac_flat[si * c + ct]
                    )?;
                }
            }
        }
        if let Some(w) = self.fit.as_mut() {
            writeln!(w, "{it}\t{phase}\t{drift:.6}")?;
        }
        Ok(())
    }

    /// Re-write the posterior-mean fractions from the running sums, so a long
    /// run stays inspectable and can be interrupted without losing everything.
    pub fn checkpoint(
        &mut self,
        it: usize,
        frac_sum: &[f64],
        _frac_flat: &[f32],
        n_collect: usize,
    ) -> Result<()> {
        if self.cfg.checkpoint_every == 0 || !it.is_multiple_of(self.cfg.checkpoint_every) {
            return Ok(());
        }
        let (s, c) = (self.sample_names.len(), self.celltype_names.len());
        let nc = n_collect as f64;
        let mat = Mat::from_fn(s, c, |si, ct| (frac_sum[si * c + ct] / nc) as f32);
        let path = format!("{}.fractions.tsv", self.out);
        let file = File::create(&path).with_context(|| format!("checkpointing {path}"))?;
        let mut w = BufWriter::new(file);
        write!(w, "sample")?;
        for name in &self.celltype_names {
            write!(w, "\t{name}")?;
        }
        writeln!(w)?;
        for (si, name) in self.sample_names.iter().enumerate() {
            write!(w, "{name}")?;
            for ct in 0..c {
                write!(w, "\t{:.6}", mat[(si, ct)])?;
            }
            writeln!(w)?;
        }
        Ok(())
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
