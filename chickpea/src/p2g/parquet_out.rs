//! E2G-like parquet writers: peaks, clusters, peak_gene (chr-partitioned).
//!
//! Target columns (Engreitz E2G-inspired):
//! - `peaks.parquet`: id, chromosome, start, end, class
//! - `clusters.parquet`: id, name
//! - `peak_gene/chr*.parquet`: id, score, target_gene_id, target_gene_name,
//!   target_gene_tss, enhancer_gene_distance, model, chromosome, enhancer_id,
//!   cell_type_id (= cluster id)

use crate::p2g::refine::ClusterLink;
use genomic_data::coordinates::{chr_stripped, GeneTss, PeakCoord};
use legume_numeric::matrix::parquet::{write_table, Column};
use log::info;
use rustc_hash::FxHashMap;
use std::path::Path;

/// One peak catalog row (`enhancers.parquet` analog).
#[derive(Clone, Debug)]
pub struct PeakRow {
    pub id: i32,
    pub chromosome: Box<str>,
    pub start: i32,
    pub end: i32,
    pub class: Box<str>,
}

/// One cluster catalog row (`cell_types.parquet` analog).
#[derive(Clone, Debug)]
pub struct ClusterRow {
    pub id: Box<str>,
    pub name: Box<str>,
}

/// Write E2G-like tables under `out_dir`:
/// - `peaks.parquet`
/// - `clusters.parquet`
/// - `peak_gene/chr*.parquet`
pub fn write_e2g_tables(
    out_dir: &str,
    peaks: &[PeakRow],
    clusters: &[ClusterRow],
    links: &[ClusterLink],
    peak_coords: &[Option<PeakCoord>],
    gene_tss: &[Option<GeneTss>],
    gene_names: &[Box<str>],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let peaks_path = Path::new(out_dir).join("peaks.parquet");
    write_peaks(peaks_path.to_str().unwrap(), peaks)?;

    let clusters_path = Path::new(out_dir).join("clusters.parquet");
    write_clusters(clusters_path.to_str().unwrap(), clusters)?;

    let pg_dir = Path::new(out_dir).join("peak_gene");
    std::fs::create_dir_all(&pg_dir)?;
    write_peak_gene_partitions(
        pg_dir.to_str().unwrap(),
        links,
        peak_coords,
        gene_tss,
        gene_names,
    )?;

    info!(
        "Wrote E2G-like tables under {out_dir} ({} peaks, {} clusters, {} links)",
        peaks.len(),
        clusters.len(),
        links.len()
    );
    Ok(())
}

/// Build peak catalog rows from coordinates (1-based ids).
pub fn peaks_from_coords(peak_coords: &[Option<PeakCoord>]) -> Vec<PeakRow> {
    peak_coords
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let c = c.as_ref()?;
            Some(PeakRow {
                id: (i + 1) as i32,
                chromosome: c.chr.clone(),
                start: c.start as i32,
                end: c.end as i32,
                class: "peak".into(),
            })
        })
        .collect()
}

fn write_peaks(path: &str, peaks: &[PeakRow]) -> anyhow::Result<()> {
    let id: Vec<i32> = peaks.iter().map(|p| p.id).collect();
    let chromosome: Vec<Box<str>> = peaks.iter().map(|p| p.chromosome.clone()).collect();
    let start: Vec<i32> = peaks.iter().map(|p| p.start).collect();
    let end: Vec<i32> = peaks.iter().map(|p| p.end).collect();
    let class: Vec<Box<str>> = peaks.iter().map(|p| p.class.clone()).collect();
    write_table(
        path,
        &[
            ("id".into(), Column::I32(&id)),
            ("chromosome".into(), Column::Str(&chromosome)),
            ("start".into(), Column::I32(&start)),
            ("end".into(), Column::I32(&end)),
            ("class".into(), Column::Str(&class)),
        ],
    )
}

fn write_clusters(path: &str, clusters: &[ClusterRow]) -> anyhow::Result<()> {
    let id: Vec<Box<str>> = clusters.iter().map(|c| c.id.clone()).collect();
    let name: Vec<Box<str>> = clusters.iter().map(|c| c.name.clone()).collect();
    write_table(
        path,
        &[
            ("id".into(), Column::Str(&id)),
            ("name".into(), Column::Str(&name)),
        ],
    )
}

fn write_peak_gene_partitions(
    pg_dir: &str,
    links: &[ClusterLink],
    peak_coords: &[Option<PeakCoord>],
    gene_tss: &[Option<GeneTss>],
    gene_names: &[Box<str>],
) -> anyhow::Result<()> {
    let mut by_chr: FxHashMap<Box<str>, Vec<usize>> = FxHashMap::default();
    for (i, link) in links.iter().enumerate() {
        let Some(pc) = peak_coords.get(link.edge.peak).and_then(|c| c.as_ref()) else {
            continue;
        };
        by_chr.entry(pc.chr.clone()).or_default().push(i);
    }

    for (chr, idxs) in by_chr {
        let stem = chr_stripped(chr.as_ref());
        let path = format!("{pg_dir}/chr{stem}.parquet");

        let mut id = Vec::with_capacity(idxs.len());
        let mut score = Vec::with_capacity(idxs.len());
        let mut target_gene_id = Vec::with_capacity(idxs.len());
        let mut target_gene_name = Vec::with_capacity(idxs.len());
        let mut target_gene_tss_col = Vec::with_capacity(idxs.len());
        let mut enhancer_gene_distance = Vec::with_capacity(idxs.len());
        let mut chromosome = Vec::with_capacity(idxs.len());
        let mut enhancer_id = Vec::with_capacity(idxs.len());
        let mut cell_type_id = Vec::with_capacity(idxs.len());
        let model: Vec<Box<str>> = idxs.iter().map(|_| "chickpea".into()).collect();

        for (row_i, &li) in idxs.iter().enumerate() {
            let link = &links[li];
            let pc = peak_coords[link.edge.peak]
                .as_ref()
                .expect("peak filtered above");
            let g = link.edge.gene;
            anyhow::ensure!(g < gene_names.len(), "gene index {g} out of range");
            let tss = gene_tss
                .get(g)
                .and_then(|t| t.as_ref())
                .map(|t| t.tss)
                .unwrap_or(0);
            let mid = (pc.start + pc.end) / 2;
            let dist = (mid - tss).unsigned_abs() as i32;

            id.push((row_i + 1) as i32);
            score.push(link.edge.weight);
            target_gene_id.push(gene_names[g].clone());
            target_gene_name.push(gene_names[g].clone());
            target_gene_tss_col.push(tss as i32);
            enhancer_gene_distance.push(dist);
            chromosome.push(pc.chr.clone());
            enhancer_id.push((link.edge.peak + 1) as i32);
            cell_type_id.push(link.cluster.to_string().into_boxed_str());
        }

        write_table(
            &path,
            &[
                ("id".into(), Column::I32(&id)),
                ("score".into(), Column::F32(&score)),
                ("target_gene_id".into(), Column::Str(&target_gene_id)),
                ("target_gene_name".into(), Column::Str(&target_gene_name)),
                ("target_gene_tss".into(), Column::I32(&target_gene_tss_col)),
                (
                    "enhancer_gene_distance".into(),
                    Column::I32(&enhancer_gene_distance),
                ),
                ("model".into(), Column::Str(&model)),
                ("chromosome".into(), Column::Str(&chromosome)),
                ("enhancer_id".into(), Column::I32(&enhancer_id)),
                ("cell_type_id".into(), Column::Str(&cell_type_id)),
            ],
        )?;
    }
    Ok(())
}
