//! BGZF BED output for peak→gene links.

use crate::common::*;

/// One fine-mapped peak→gene link.
pub struct LinkRecord {
    pub chr: Box<str>,
    pub start: i64,
    pub end: i64,
    pub peak_id: Box<str>,
    pub gene_id: Box<str>,
    pub pip: f32,
    pub effect_mean: f32,
    pub effect_std: f32,
    pub z: f32,
    pub distance: i64,
}

/// Write links as a sorted, BGZF-compressed BED (`{out}.results.bed.gz`).
///
/// Columns: `#chr start end peak_id gene_id pip effect_mean effect_std z distance`.
/// All tested pairs are written; `pip_threshold` only drives the summary log.
pub fn write_bed(records: &mut [LinkRecord], pip_threshold: f32, path: &str) -> anyhow::Result<()> {
    use rust_htslib::bgzf;
    use std::io::Write;

    records.sort_by(|a, b| (&*a.chr, a.start, a.end).cmp(&(&*b.chr, b.start, b.end)));

    let mut writer = bgzf::Writer::from_path(path)?;
    writeln!(
        writer,
        "#chr\tstart\tend\tpeak_id\tgene_id\tpip\teffect_mean\teffect_std\tz\tdistance"
    )?;

    let mut n_sig = 0usize;
    for r in records.iter() {
        if r.pip >= pip_threshold {
            n_sig += 1;
        }
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.4}\t{}",
            r.chr,
            r.start,
            r.end,
            r.peak_id,
            r.gene_id,
            r.pip,
            r.effect_mean,
            r.effect_std,
            r.z,
            r.distance,
        )?;
    }
    writer.flush()?;

    info!(
        "Wrote {} ({} links, {} with pip >= {:.3})",
        path,
        records.len(),
        n_sig,
        pip_threshold
    );
    Ok(())
}
