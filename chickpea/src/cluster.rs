//! `chickpea cluster` — kNN + Leiden over a cell-embedding parquet.
//!
//! Inputs come from `--e-cell` directly or from a `chickpea embed-graph`
//! run manifest via `--from <prefix>.senna.json`. In manifest mode the
//! cluster path is written back so downstream `senna annotate` /
//! `senna layout` / `senna plot --from` see it.

use crate::common::*;
use crate::manifest::{self, manifest_path, RunManifest};
use matrix_util::knn_graph::{self, KnnGraph, KnnGraphArgs};
use matrix_util::traits::{IoOps, MatOps};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct ClusterArgs {
    #[arg(
        long,
        conflicts_with = "from",
        help = "Cell-embedding parquet (e.g., {prefix}.e_cell.parquet)"
    )]
    e_cell: Option<Box<str>>,

    #[arg(
        long,
        conflicts_with = "e_cell",
        help = "chickpea embed-graph run manifest ({prefix}.senna.json). \
                Reads e_cell from outputs.latent and writes cluster path back."
    )]
    from: Option<Box<str>>,

    #[arg(long, default_value_t = 30, help = "k for the kNN graph")]
    knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution (auto-rescaled to CPM)"
    )]
    resolution: f64,

    #[arg(
        long,
        help = "If set, binary-search resolution to approximate this many clusters"
    )]
    target_clusters: Option<usize>,

    #[arg(long, default_value_t = 1, help = "Random seed")]
    seed: u64,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix; produces {out}.clusters.parquet"
    )]
    out: Box<str>,
}

pub fn cluster(args: &ClusterArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let (e_cell_path, manifest_state) = resolve_input(args)?;

    let loaded = nalgebra::DMatrix::<f32>::from_parquet(&e_cell_path)?;
    let (n, d) = (loaded.mat.nrows(), loaded.mat.ncols());
    info!("Loaded {n} cells × {d} embedding dims from {e_cell_path}");

    let mut latent = loaded.mat;
    latent.scale_columns_inplace();

    let graph = KnnGraph::from_rows(
        &latent,
        KnnGraphArgs {
            knn: args.knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;
    info!(
        "KNN graph: {} nodes, {} edges",
        graph.num_nodes(),
        graph.num_edges()
    );

    let (network, total_edge_weight) = graph.to_leiden_network();
    let res_scaled = knn_graph::modularity_to_cpm_resolution(args.resolution, total_edge_weight);
    let seed = Some(args.seed as usize);

    let labels = match args.target_clusters {
        Some(k) => {
            info!("Tuning resolution to ~{k} clusters");
            knn_graph::tune_leiden_resolution(&network, n, k, res_scaled, seed)
        }
        None => {
            info!("Running Leiden at scaled resolution={res_scaled:.6e}");
            knn_graph::run_leiden(&network, n, res_scaled, seed)
        }
    };

    let mut compact = labels;
    knn_graph::compact_labels(&mut compact);
    let k = compact.iter().copied().max().unwrap_or(0) + 1;
    info!("Leiden: {k} clusters");

    let clusters_path = format!("{}.clusters.parquet", args.out);
    save_clusters(&clusters_path, &compact, &loaded.rows)?;
    info!("Wrote {clusters_path}");

    if let Some((mut m, manifest_dir, manifest_path_str)) = manifest_state {
        m.cluster.clusters = Some(manifest::rel_to_manifest(&manifest_dir, &clusters_path));
        manifest::save(&m, &manifest_path_str)?;
        info!("Updated manifest {manifest_path_str}");
    }

    Ok(())
}

type ManifestState = (RunManifest, PathBuf, String);

fn resolve_input(args: &ClusterArgs) -> anyhow::Result<(String, Option<ManifestState>)> {
    if let Some(p) = &args.e_cell {
        return Ok((p.to_string(), None));
    }
    let from = args
        .from
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("either --e-cell or --from is required"))?;
    let from_path = if Path::new(from).exists() {
        from.to_string()
    } else {
        manifest_path(from)
    };
    let (m, manifest_dir) = manifest::load(&from_path)?;
    let latent_rel = m
        .outputs
        .latent
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("manifest {from_path} has no outputs.latent"))?;
    let e_cell_path = manifest::resolve(&manifest_dir, latent_rel)
        .to_string_lossy()
        .into_owned();
    Ok((e_cell_path, Some((m, manifest_dir, from_path))))
}

fn save_clusters(path: &str, labels: &[usize], cell_names: &[Box<str>]) -> anyhow::Result<()> {
    let n = labels.len();
    if cell_names.len() != n {
        anyhow::bail!("labels {} != cell_names {}", n, cell_names.len());
    }
    let col_data: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
    let mat = nalgebra::DMatrix::<f32>::from_column_slice(n, 1, &col_data);
    let cols = vec![Box::<str>::from("cluster")];
    mat.to_parquet_with_names(path, (Some(cell_names), Some("cell")), Some(&cols))?;
    Ok(())
}
