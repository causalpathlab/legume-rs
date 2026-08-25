//! The regression Stage 0 exists to prevent, stated end to end.
//!
//! `cage` scored a splice-channelized matrix as if each channel row were its own
//! gene: the gene axis doubled, and a gene's positives were drawn from half its
//! evidence twice over rather than from the gene once. The property that says it
//! is fixed is an EQUALITY — a two-channel matrix must produce exactly the
//! activities of the single-channel matrix whose counts are its per-gene sums.

use super::gene_gating::{build_gene_active_fine_edges, ActivityNorm, GeneActiveEdges};
use crate::util::common::*;
use crate::util::gene_axis::GeneAxis;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};

const N_CELLS: usize = 6;
const EDGES: [(u32, u32); 5] = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5)];

/// `(gene, cell) -> (spliced, unspliced)`. Deliberately uneven: some cells carry
/// only one track, which is where a per-row activity and a per-gene one differ
/// most.
const COUNTS: [(usize, usize, f32, f32); 10] = [
    (0, 0, 5.0, 2.0),
    (0, 1, 3.0, 0.0),
    (0, 3, 0.0, 4.0),
    (1, 0, 10.0, 1.0),
    (1, 1, 20.0, 6.0),
    (1, 4, 15.0, 0.0),
    (1, 5, 0.0, 25.0),
    (2, 2, 7.0, 3.0),
    (2, 4, 1.0, 1.0),
    (2, 5, 2.0, 0.0),
];

fn write(
    dir: &tempfile::TempDir,
    tag: &str,
    rows: &[Box<str>],
    triplets: &[(u64, u64, f32)],
) -> anyhow::Result<SparseIoVec> {
    let path = dir.path().join(format!("{tag}.zarr"));
    let mut backend = create_sparse_from_triplets(
        triplets,
        (rows.len(), N_CELLS, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )?;
    backend.register_row_names_vec(rows);
    let cells: Vec<Box<str>> = (0..N_CELLS).map(|i| format!("c{i}").into()).collect();
    backend.register_column_names_vec(&cells);
    let mut v = SparseIoVec::new();
    v.push(std::sync::Arc::from(backend), None)?;
    Ok(v)
}

/// One row per gene, holding `spliced + unspliced`.
fn pooled_input(dir: &tempfile::TempDir) -> anyhow::Result<SparseIoVec> {
    let rows: Vec<Box<str>> = (0..3).map(|g| format!("GENE{g}").into()).collect();
    let triplets: Vec<(u64, u64, f32)> = COUNTS
        .iter()
        .map(|&(g, c, s, u)| (g as u64, c as u64, s + u))
        .collect();
    write(dir, "pooled", &rows, &triplets)
}

/// Two rows per gene. The row ORDER is interleaved and puts a gene's nascent row
/// before its mature one, because nothing about the grammar promises otherwise
/// and a fold that only works on sorted rows would pass a friendlier fixture.
fn channelized_input(dir: &tempfile::TempDir) -> anyhow::Result<SparseIoVec> {
    let mut rows: Vec<Box<str>> = Vec::new();
    for g in 0..3 {
        rows.push(format!("GENE{g}/count/unspliced").into());
        rows.push(format!("GENE{g}/count/spliced").into());
    }
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for &(g, c, s, u) in COUNTS.iter() {
        if u > 0.0 {
            triplets.push((2 * g as u64, c as u64, u));
        }
        if s > 0.0 {
            triplets.push((2 * g as u64 + 1, c as u64, s));
        }
    }
    triplets.sort_by_key(|&(r, c, _)| (c, r));
    write(dir, "channelized", &rows, &triplets)
}

fn activities(data: &SparseIoVec, norm: ActivityNorm) -> anyhow::Result<GeneActiveEdges> {
    let axis = GeneAxis::resolve(&data.row_names()?)?;
    build_gene_active_fine_edges(data, &EDGES, None, norm, &axis)
}

/// Break it by indexing `gene_active_edges` with the row instead of the gene and
/// this fails on the very first assert: six entries where there are three genes.
#[test]
fn two_channels_give_exactly_the_pooled_single_channel_activities() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let pooled = pooled_input(&dir)?;
    let channelized = channelized_input(&dir)?;

    // The axis itself, first: same genes, same order, from row names alone.
    let axis = GeneAxis::resolve(&channelized.row_names()?)?;
    assert!(axis.is_channelized());
    assert_eq!(axis.gene_names(), pooled.row_names()?.as_slice());

    for norm in [ActivityNorm::Log1p, ActivityNorm::L1, ActivityNorm::L2] {
        let a = activities(&pooled, norm)?;
        let b = activities(&channelized, norm)?;

        assert_eq!(a.gene_active_edges.len(), 3, "{norm:?}: one entry per gene");
        assert_eq!(b.gene_active_edges, a.gene_active_edges, "{norm:?}");
        assert_eq!(
            b.gene_active_edge_weights, a.gene_active_edge_weights,
            "{norm:?}"
        );
    }
    Ok(())
}

/// The half-evidence failure, made visible: summing the two tracks AFTER the
/// `log1p` is not the same number, so an implementation that pooled activities
/// instead of counts would pass the shape checks and still be wrong.
#[test]
fn pooling_happens_on_counts_not_on_log1p_activities() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let a = activities(&channelized_input(&dir)?, ActivityNorm::Log1p)?;

    // GENE0 at cell 0 is (spliced 5, unspliced 2); edge (0,1) has GENE0 active
    // at both endpoints, where cell 1 is (3, 0).
    let w = 7f32.ln_1p() * 3f32.ln_1p();
    let wrong = (5f32.ln_1p() + 2f32.ln_1p()) * 3f32.ln_1p();
    let e0 = a.gene_active_edges[0]
        .iter()
        .position(|&e| e == 0)
        .expect("GENE0 is active on edge (0, 1)");
    let got = a.gene_active_edge_weights[0][e0];

    assert!((got - w).abs() < 1e-6, "expected {w}, got {got}");
    assert!(
        (got - wrong).abs() > 1e-3,
        "log1p is not additive across tracks"
    );
    Ok(())
}
