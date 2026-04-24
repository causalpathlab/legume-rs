//! `cocoa simulate-spatial` — minimal spatial-DE simulator.
//!
//! Generates ground-truth inputs to validate `cocoa spatial-diff`.
//! See `main.rs` long_about for the generative model and output layout.

use crate::common::*;

use clap::Parser;
use matrix_util::common_io::{mkdir, write_lines};
use matrix_util::traits::IoOps;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Gamma as GammaDist, Poisson};
use rustc_hash::FxHashSet;

#[derive(Parser, Debug, Clone)]
pub struct SimSpatialArgs {
    #[arg(long, default_value_t = 30, help = "Grid width (number of x bins).")]
    grid_x: usize,

    #[arg(long, default_value_t = 30, help = "Grid height (number of y bins).")]
    grid_y: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Number of genes in the simulated panel."
    )]
    n_genes: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of latent topics (Dirichlet components)."
    )]
    n_topics: usize,

    #[arg(
        long,
        default_value_t = 4,
        help = "Number of individuals (contiguous x-blocks of the grid)."
    )]
    n_indv: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of ground-truth spatial DE markers to inject.",
        long_help = "Each marker is a random (gene, topic, region) triple.\n\
                     The gene is elevated in cells that are simultaneously\n\
                     HIGH in the target topic and inside the region.\n\
                     Each gene is chosen at most once across markers."
    )]
    n_spatial_markers: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Baseline Poisson rate for non-marker cells."
    )]
    baseline_rate: f32,

    #[arg(
        long,
        default_value_t = 15.0,
        help = "Elevated Poisson rate inside a marker region."
    )]
    effect_size: f32,

    #[arg(
        long,
        default_value_t = 0.75,
        help = "θ_k quantile defining HIGH cells for marker injection."
    )]
    topic_high_quantile: f32,

    #[arg(long, default_value_t = 42, help = "Random seed.")]
    rseed: u64,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Sparse matrix backend for counts (`zarr` or `hdf5`)."
    )]
    backend: SparseIoBackend,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix (all files share this prefix)."
    )]
    out: Box<str>,
}

/// One injected spatial marker: gene `g` is elevated in cells whose
/// θ_k ≥ HIGH and (x, y) lie in `region`.
struct SpatialMarker {
    gene: usize,
    topic: usize,
    region: Region,
}

#[derive(Clone, Copy)]
enum Region {
    XLeft(f32),
    XRight(f32),
    YBottom(f32),
    YTop(f32),
}

impl Region {
    fn contains(&self, x: f32, y: f32) -> bool {
        match *self {
            Region::XLeft(thr) => x < thr,
            Region::XRight(thr) => x >= thr,
            Region::YBottom(thr) => y < thr,
            Region::YTop(thr) => y >= thr,
        }
    }
    fn label(&self) -> String {
        match *self {
            Region::XLeft(t) => format!("x<{t}"),
            Region::XRight(t) => format!("x>={t}"),
            Region::YBottom(t) => format!("y<{t}"),
            Region::YTop(t) => format!("y>={t}"),
        }
    }
}

pub fn run_sim_spatial(args: SimSpatialArgs) -> anyhow::Result<()> {
    let n_cells = args.grid_x * args.grid_y;
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.rseed);

    info!(
        "Simulating {} cells on {}x{} grid, {} genes, {} topics, {} individuals",
        n_cells, args.grid_x, args.grid_y, args.n_genes, args.n_topics, args.n_indv
    );

    // Coords (row-major: (xi, yi) for cell i = xi * grid_y + yi)
    let coords: Vec<(f32, f32)> = (0..n_cells)
        .map(|i| {
            let xi = (i / args.grid_y) as f32;
            let yi = (i % args.grid_y) as f32;
            (xi, yi)
        })
        .collect();

    // Per-cell Dirichlet topic θ
    let gu = GammaDist::new(1.0, 1.0).unwrap();
    let mut theta = Mat::zeros(n_cells, args.n_topics);
    for i in 0..n_cells {
        let mut s = 0f32;
        let mut row = vec![0f32; args.n_topics];
        for r in row.iter_mut() {
            *r = gu.sample(&mut rng) as f32;
            s += *r;
        }
        for (k, &r) in row.iter().enumerate() {
            theta[(i, k)] = r / s;
        }
    }

    // Per-cell individual: split along x into n_indv blocks
    let indv_ids: Vec<Box<str>> = (0..args.n_indv)
        .map(|i| format!("indv{i}").into_boxed_str())
        .collect();
    let cell_to_indv: Vec<Box<str>> = (0..n_cells)
        .map(|i| {
            let xi = i / args.grid_y;
            let bin = (xi * args.n_indv) / args.grid_x;
            indv_ids[bin.min(args.n_indv - 1)].clone()
        })
        .collect();

    // Per-topic HIGH quantile thresholds
    let mut hi_thr = vec![0f32; args.n_topics];
    for k in 0..args.n_topics {
        let mut col: Vec<f32> = (0..n_cells).map(|i| theta[(i, k)]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        hi_thr[k] = col[((n_cells as f32) * args.topic_high_quantile) as usize];
    }

    // Pick markers: random (gene, topic, region)
    let x_median = (args.grid_x as f32 - 1.0) * 0.5;
    let y_median = (args.grid_y as f32 - 1.0) * 0.5;
    let region_menu = [
        Region::XLeft(x_median),
        Region::XRight(x_median),
        Region::YBottom(y_median),
        Region::YTop(y_median),
    ];
    let mut markers: Vec<SpatialMarker> = Vec::with_capacity(args.n_spatial_markers);
    let mut used_genes: FxHashSet<usize> = Default::default();
    while markers.len() < args.n_spatial_markers && used_genes.len() < args.n_genes {
        let g: usize = rng.random_range(0..args.n_genes);
        if !used_genes.insert(g) {
            continue;
        }
        let k: usize = rng.random_range(0..args.n_topics);
        let r = region_menu[rng.random_range(0..region_menu.len())];
        markers.push(SpatialMarker {
            gene: g,
            topic: k,
            region: r,
        });
    }
    info!(
        "Injected {} spatial markers (quantile > {})",
        markers.len(),
        args.topic_high_quantile
    );

    // Generate counts (triplets)
    let baseline = Poisson::new(args.baseline_rate).unwrap();
    let elevated = Poisson::new(args.effect_size).unwrap();
    let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
    for i in 0..n_cells {
        let (xi, yi) = coords[i];
        for g in 0..args.n_genes {
            let mut hit = false;
            for m in &markers {
                if m.gene == g
                    && theta[(i, m.topic)] >= hi_thr[m.topic]
                    && m.region.contains(xi, yi)
                {
                    hit = true;
                    break;
                }
            }
            let y: f32 = if hit {
                elevated.sample(&mut rng)
            } else {
                baseline.sample(&mut rng)
            };
            if y > 0.0 {
                triplets.push((g as u64, i as u64, y));
            }
        }
    }

    // Write out
    let out = args.out.clone();
    mkdir(&out)?;

    let backend_file = match args.backend {
        SparseIoBackend::HDF5 => format!("{out}.h5"),
        SparseIoBackend::Zarr => format!("{out}.zarr"),
    };

    let gene_names: Vec<Box<str>> = (0..args.n_genes)
        .map(|g| format!("gene{g}").into_boxed_str())
        .collect();
    let cell_names: Vec<Box<str>> = (0..n_cells)
        .map(|i| format!("cell{i}").into_boxed_str())
        .collect();

    let mut sp = create_sparse_from_triplets(
        &triplets,
        (args.n_genes, n_cells, triplets.len()),
        Some(&backend_file),
        Some(&args.backend),
    )?;
    sp.register_row_names_vec(&gene_names);
    sp.register_column_names_vec(&cell_names);
    info!("Wrote sparse counts to {backend_file}");

    // Topic propensity parquet (N × K, cell-name as row index, "topic{k}" columns)
    let topic_cols: Vec<Box<str>> = (0..args.n_topics)
        .map(|k| format!("topic{k}").into_boxed_str())
        .collect();
    let theta_path = format!("{out}.topic.parquet");
    theta.to_parquet_with_names(
        &theta_path,
        (Some(&cell_names), Some("cell")),
        Some(&topic_cols),
    )?;
    info!("Wrote topic propensity to {theta_path}");

    // Coords TSV.gz
    // Coords: row-order (no header, no cell-name column) so they align
    // with data column positions regardless of batch-tag suffixes.
    let coords_path = format!("{out}.coords.tsv.gz");
    let mut lines: Vec<Box<str>> = Vec::with_capacity(n_cells);
    for &(x, y) in &coords {
        lines.push(format!("{x}\t{y}").into());
    }
    write_lines(&lines, &coords_path)?;
    info!("Wrote coords to {coords_path}");

    // Individual labels TSV.gz (cell\tindv)
    let indv_path = format!("{out}.indv.tsv.gz");
    let mut ilines: Vec<Box<str>> = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        ilines.push(format!("{}\t{}", cell_names[i], cell_to_indv[i]).into());
    }
    write_lines(&ilines, &indv_path)?;
    info!("Wrote individual labels to {indv_path}");

    // Ground truth TSV.gz
    let gt_path = format!("{out}.ground_truth.tsv.gz");
    let mut gtl: Vec<Box<str>> = Vec::with_capacity(markers.len() + 1);
    gtl.push("gene\ttopic\tregion\teffect_size".into());
    for m in &markers {
        gtl.push(
            format!(
                "{}\t{}\t{}\t{}",
                gene_names[m.gene],
                topic_cols[m.topic],
                m.region.label(),
                args.effect_size
            )
            .into(),
        );
    }
    write_lines(&gtl, &gt_path)?;
    info!("Wrote ground truth to {gt_path}");

    info!("Spatial simulation complete.");
    Ok(())
}
