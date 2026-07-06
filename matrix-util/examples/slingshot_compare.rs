//! Validation harness: run matrix-util's Slingshot principal curves on a fixed
//! embedding + fixed cluster labels + fixed root, so the result can be compared
//! against reference R `slingshot` on the *same* labels/centroids/MST/root.
//!
//! Usage:
//!   cargo run -p matrix-util --example slingshot_compare -- \
//!       <embedding.parquet> <labels.csv> <root_cluster> <out_prefix>
//!
//! Writes `<out_prefix>_lineage_pt.csv` (N × L per-lineage pseudotime, blank for
//! non-members) and prints the MST edges + lineage node paths to stdout.

use std::fs;
use std::io::Write;

use matrix_util::dmatrix_io::DMatrix;
use matrix_util::principal_curve::{fit_principal_curves, PrincipalCurveArgs};
use matrix_util::principal_graph::{mst_from_sqdist, pairwise_sqdist_rows_to_rows};
use matrix_util::traits::IoOps;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 5,
        "usage: slingshot_compare <embedding.parquet> <labels.csv> <root_cluster> <out_prefix>"
    );
    let emb_path = &args[1];
    let labels_path = &args[2];
    let root_cluster: usize = args[3].parse()?;
    let out_prefix = &args[4];

    // Embedding (N × D) with cell row names.
    let emb = DMatrix::<f32>::from_parquet(emb_path)?;
    let z = emb.mat;
    let cell_names = emb.rows;
    let n = z.nrows();

    // Labels: CSV with a header line, one integer per row.
    let raw = fs::read_to_string(labels_path)?;
    let labels: Vec<usize> = raw
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;
    anyhow::ensure!(
        labels.len() == n,
        "labels ({}) != cells ({n})",
        labels.len()
    );
    let k = labels.iter().copied().max().unwrap() + 1;

    // Centroids = mean per label (exactly what slingshot computes from labels).
    let d = z.ncols();
    let mut centroids = DMatrix::<f32>::zeros(k, d);
    let mut counts = vec![0usize; k];
    for (i, &c) in labels.iter().enumerate() {
        for j in 0..d {
            centroids[(c, j)] += z[(i, j)];
        }
        counts[c] += 1;
    }
    for c in 0..k {
        for j in 0..d {
            centroids[(c, j)] /= counts[c].max(1) as f32;
        }
    }

    // Euclidean MST on centroids (slingshot's default), then Slingshot curves.
    let (edges, _w) = mst_from_sqdist(&pairwise_sqdist_rows_to_rows(&centroids, &centroids));
    println!("MST edges (cluster pairs):");
    let mut sorted = edges.clone();
    sorted.sort_unstable();
    for (a, b) in &sorted {
        println!("  {a} - {b}");
    }

    let curves = fit_principal_curves(
        &z,
        &centroids,
        &edges,
        root_cluster,
        &PrincipalCurveArgs::default(),
    )?;

    println!("\nlineages (node paths, root -> leaf):");
    for (l, c) in curves.curves.iter().enumerate() {
        println!("  L{l}: {:?}", c.node_path);
    }

    // Per-lineage pseudotime, N × L (blank where the cell is not a member).
    let n_lin = curves.n_lineages();
    let path = format!("{out_prefix}_lineage_pt.csv");
    let mut f = fs::File::create(&path)?;
    write!(f, "cell")?;
    for l in 0..n_lin {
        write!(f, ",L{l}")?;
    }
    writeln!(f)?;
    for (i, name) in cell_names.iter().enumerate() {
        write!(f, "{name}")?;
        for l in 0..n_lin {
            let v = curves.lineage_pseudotime[(i, l)];
            if v.is_finite() && curves.weights[(i, l)] > 0.0 {
                write!(f, ",{v}")?;
            } else {
                write!(f, ",")?;
            }
        }
        writeln!(f)?;
    }
    println!("\nwrote {path}  ({n} cells x {n_lin} lineages)");
    Ok(())
}
