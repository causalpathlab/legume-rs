//! Gene positions from a simple TSV (`gene`, `chr`, `tss`, with a header), as
//! the multiome simulator writes them.

use crate::common::*;
use genomic_data::coordinates::GeneTss;

/// Load gene TSS positions from a simple TSV file (gene\tchr\ttss).
/// Produced by sim-link as {out}.gene_coords.tsv.gz.
pub fn load_gene_coords_tsv(
    path: &str,
    gene_names: &[Box<str>],
) -> anyhow::Result<Vec<Option<GeneTss>>> {
    use legume_numeric::matrix::common_io::open_buf_reader;
    use std::io::BufRead;

    let reader = open_buf_reader(path)?;
    let mut tss_map: rustc_hash::FxHashMap<Box<str>, GeneTss> = Default::default();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }
        let gene: Box<str> = fields[0].into();
        let chr: Box<str> = fields[1].into();
        let tss: i64 = fields[2].parse()?;
        tss_map.insert(gene, GeneTss { chr, tss });
    }

    info!(
        "Loaded {} gene positions from {}, matching against {} genes",
        tss_map.len(),
        path,
        gene_names.len()
    );

    let result: Vec<Option<GeneTss>> = gene_names
        .iter()
        .map(|name| tss_map.get(name).cloned())
        .collect();

    let matched = result.iter().filter(|x| x.is_some()).count();
    info!(
        "Matched {}/{} genes to coordinates",
        matched,
        gene_names.len()
    );

    Ok(result)
}
