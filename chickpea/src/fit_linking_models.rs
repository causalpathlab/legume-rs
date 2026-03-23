use crate::common::*;
use crate::input::*;
use crate::linking::*;
use candle_util::sgvb::variant_tree::VariantTree;

#[derive(Args, Debug)]
pub struct LinkArgs {
    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "RNA expression files (sparse backends: zarr, h5)"
    )]
    rna: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "ATAC peak files (sparse backends: zarr, h5)"
    )]
    atac: Vec<Box<str>>,

    #[arg(long, short, required = true, help = "Output prefix")]
    out: Box<str>,

    #[arg(long, help = "GFF/GTF file for cis-window filtering")]
    gff: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 500_000,
        help = "Cis window in bp (default 500kb)"
    )]
    cis_window: i64,

    // Cell collapsing
    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension"
    )]
    proj_dim: usize,

    #[arg(long, default_value_t = 10, help = "Sort dimension for cell grouping")]
    sort_dim: usize,

    // Feature coarsening
    #[arg(long, default_value_t = 256, help = "Max gene modules for coarsening")]
    max_gene_modules: usize,

    #[arg(long, default_value_t = 512, help = "Max peak modules for coarsening")]
    max_peak_modules: usize,

    // Screening
    #[arg(long, default_value_t = 0.1, help = "Module correlation threshold")]
    corr_threshold: f32,

    // Tree structure
    #[arg(
        long,
        default_value_t = 8,
        help = "Tree branching factor for multi-resolution cascade"
    )]
    block_size: usize,

    // Cascade pruning
    #[arg(
        long,
        default_value_t = 0.05,
        help = "PIP threshold for pruning at each tree level"
    )]
    pip_prune_threshold: f64,

    // Regression model
    #[arg(long, value_enum, default_value_t = LinkModel::Gaussian)]
    model: LinkModel,

    // SuSiE params
    #[arg(long, default_value_t = 5, help = "SuSiE components L")]
    susie_components: usize,

    #[arg(long, default_value_t = 0.2, help = "SuSiE prior variance")]
    prior_var: f64,

    #[arg(long, default_value_t = 0.1, help = "PIP reporting threshold")]
    pip_threshold: f64,

    // SGVB-only params
    #[arg(long, default_value_t = 0.01, help = "SGVB learning rate")]
    learning_rate: f64,

    #[arg(long, default_value_t = 2000, help = "SGVB iterations")]
    num_sgvb_iter: usize,

    #[arg(long, default_value_t = 4, help = "SGVB MC samples")]
    num_sgvb_samples: usize,

    // I/O
    #[arg(long, default_value_t = 100, help = "Block size for I/O")]
    io_block_size: usize,

    #[arg(long, default_value_t = false, help = "Preload data into memory")]
    preload: bool,
}

pub fn run_link(args: &LinkArgs) -> anyhow::Result<()> {
    use rayon::prelude::*;

    // Phase 1: Load data and build pseudobulk
    info!("=== Phase 1: Loading data ===");
    let data = load_and_collapse(args)?;

    let n_genes = data.gene_names.len();
    let n_peaks = data.peak_names.len();

    // Phase 2: Feature coarsening + module hierarchy
    info!("=== Phase 2: Feature coarsening + tree ===");

    let gene_fc = if args.max_gene_modules < n_genes {
        let fc = compute_feature_coarsening(&data.gene_pb, args.max_gene_modules)?;
        info!("Gene coarsening: {} → {} modules", n_genes, fc.num_coarse);
        Some(fc)
    } else {
        None
    };

    let peak_fc = compute_feature_coarsening(&data.peak_pb, args.max_peak_modules)?;
    info!(
        "Peak coarsening: {} → {} modules",
        n_peaks, peak_fc.num_coarse
    );

    let module_pb = peak_fc.aggregate_rows_ds(&data.peak_pb);
    let tree = VariantTree::regular(peak_fc.num_coarse, args.block_size);
    let tree_maps = TreeLevelMaps::from_tree(&tree);
    info!(
        "Tree: {} modules, block_size={}, depth={}",
        peak_fc.num_coarse, args.block_size, tree.depth
    );

    let gene_coarse_owned;
    let gene_coarse_ref: &Mat = match &gene_fc {
        Some(fc) => {
            gene_coarse_owned = fc.aggregate_rows_ds(&data.gene_pb);
            &gene_coarse_owned
        }
        None => &data.gene_pb,
    };
    let screen = screen_module_links(gene_coarse_ref, &module_pb, args.corr_threshold);

    let gene_candidates = build_gene_candidates(
        &gene_fc,
        &peak_fc,
        &screen,
        n_genes,
        data.gene_tss.as_deref(),
        &data.peak_coords,
        args.cis_window,
    );

    let genes_with_cands = gene_candidates.iter().filter(|c| !c.is_empty()).count();
    let total_cands: usize = gene_candidates.iter().map(|c| c.len()).sum();
    info!(
        "{} genes with candidates, {} total candidate pairs",
        genes_with_cands, total_cands
    );

    info!("=== Phase 3: Multi-resolution cascade ===");

    let cascade_params = CascadeParams {
        num_components: args.susie_components,
        prior_var: args.prior_var,
        prune_threshold: args.pip_prune_threshold,
        pip_threshold: args.pip_threshold,
        sgvb: SgvbParams {
            learning_rate: args.learning_rate,
            num_iterations: args.num_sgvb_iter,
            num_samples: args.num_sgvb_samples,
        },
    };

    let tasks: Vec<CascadeTask> = gene_candidates
        .into_iter()
        .enumerate()
        .filter(|(_, c)| !c.is_empty())
        .map(|(gi, cands)| CascadeTask {
            y: data.gene_pb.rows(gi, 1).into(),
            candidate_peaks: cands,
            gene_idx: gi,
            gene_name: data.gene_names[gi].clone(),
        })
        .collect();

    info!(
        "Running cascade on {} genes (model={:?}, tree depth={})",
        tasks.len(),
        args.model,
        tree.depth
    );

    let hierarchy = FeatureHierarchy {
        fc: &peak_fc,
        tree: &tree,
        tree_maps: &tree_maps,
    };

    let results: Vec<GeneLinkResult> = tasks
        .par_iter()
        .filter_map(|task| {
            let res = run_cascade(
                task,
                &module_pb,
                &data.peak_pb,
                &hierarchy,
                args.model,
                &cascade_params,
            )?;
            Some(GeneLinkResult {
                gene_idx: task.gene_idx,
                gene_name: task.gene_name.clone(),
                peak_indices: res.peak_indices,
                pip: res.pip,
                effect_size: res.effect_size,
            })
        })
        .collect();

    info!("=== Phase 4: Writing outputs ===");
    write_link_results(
        &results,
        &data.peak_names,
        &args.out,
        gene_fc.as_ref(),
        Some(&peak_fc),
        &data.gene_names,
    )?;
    write_module_correlations(&screen.corr_matrix, &args.out)?;

    info!(
        "Done. {} genes with significant peak-gene links.",
        results.len()
    );

    Ok(())
}

struct LinkRow {
    pip: f64,
    effect: f64,
    gene_idx: usize,
    gene: Box<str>,
    peak: Box<str>,
}

struct CollapsedData {
    gene_pb: Mat,
    peak_pb: Mat,
    gene_names: Vec<Box<str>>,
    peak_names: Vec<Box<str>>,
    gene_tss: Option<Vec<Option<GeneTss>>>,
    peak_coords: Vec<Option<PeakCoord>>,
}

fn load_and_collapse(args: &LinkArgs) -> anyhow::Result<CollapsedData> {
    let data = load_dual_modality(&args.rna, &args.atac, args.preload)?;

    let n_shared = data.shared_rna_idx.len();

    let gene_tss = if let Some(ref gff) = args.gff {
        Some(load_gene_tss(gff, &data.gene_names)?)
    } else {
        None
    };
    let peak_coords = parse_peak_coordinates(&data.peak_names);

    info!("=== Cell collapsing ===");
    let proj_dim = args.proj_dim.min(data.gene_names.len());

    let proj_kn = project_shared_cells(
        &data.rna,
        &data.shared_rna_idx,
        proj_dim,
        args.io_block_size,
    )?;

    let cell_to_group = binary_sort_shared(&proj_kn, args.sort_dim);
    let actual_groups = *cell_to_group.iter().max().unwrap_or(&0) + 1;
    info!(
        "Cell grouping: {} shared cells → {} super-cells",
        n_shared, actual_groups
    );

    let gene_pb = build_pseudobulk_from_groups(
        &data.rna,
        &data.shared_rna_idx,
        &cell_to_group,
        actual_groups,
        args.io_block_size,
    )?;
    let peak_pb = build_pseudobulk_from_groups(
        &data.atac,
        &data.shared_atac_idx,
        &cell_to_group,
        actual_groups,
        args.io_block_size,
    )?;
    info!(
        "Pseudobulk: genes [{} × {}], peaks [{} × {}]",
        gene_pb.nrows(),
        gene_pb.ncols(),
        peak_pb.nrows(),
        peak_pb.ncols()
    );

    Ok(CollapsedData {
        gene_pb,
        peak_pb,
        gene_names: data.gene_names,
        peak_names: data.peak_names,
        gene_tss,
        peak_coords,
    })
}

/// Build per-gene candidate peak lists from module screening + cis filtering.
fn build_gene_candidates(
    gene_fc: &Option<FeatureCoarsening>,
    peak_fc: &FeatureCoarsening,
    screen: &ModuleLinkScreen,
    n_genes: usize,
    gene_tss: Option<&[Option<GeneTss>]>,
    peak_coords: &[Option<PeakCoord>],
    cis_window: i64,
) -> Vec<Vec<usize>> {
    let candidates = if let Some(gfc) = gene_fc {
        expand_module_candidates(gfc, peak_fc, screen)
    } else {
        let mut cands = vec![Vec::new(); n_genes];
        for &(gm, pm) in &screen.candidate_pairs {
            if gm < n_genes {
                cands[gm].extend_from_slice(&peak_fc.coarse_to_fine[pm]);
            }
        }
        for c in &mut cands {
            c.sort_unstable();
            c.dedup();
        }
        cands
    };

    if let Some(tss_vec) = gene_tss {
        candidates
            .iter()
            .enumerate()
            .map(|(gi, cands)| {
                cis_filter_candidates(cands, tss_vec[gi].as_ref(), peak_coords, cis_window)
            })
            .collect()
    } else {
        candidates
    }
}

/// Simple random projection for shared cells.
fn project_shared_cells(
    data: &SparseIoVec,
    shared_idx: &[usize],
    proj_dim: usize,
    block_size: usize,
) -> anyhow::Result<Mat> {
    use matrix_util::traits::SampleOps;

    let d = data.num_rows();
    let n = shared_idx.len();

    let mut basis = Mat::rnorm(d, proj_dim);
    basis.normalize_columns_inplace();

    let basis_t = basis.transpose();
    let mut proj = Mat::zeros(proj_dim, n);

    for chunk_start in (0..n).step_by(block_size) {
        let chunk_end = (chunk_start + block_size).min(n);
        let cols = shared_idx[chunk_start..chunk_end].iter().copied();
        let block = data.read_columns_dmatrix(cols)?;
        let block_log = block.map(|x| (x + 1.0).ln());
        let proj_block = &basis_t * &block_log;
        for (local_j, global_j) in (chunk_start..chunk_end).enumerate() {
            for k in 0..proj_dim {
                proj[(k, global_j)] = proj_block[(k, local_j)];
            }
        }
    }

    proj.scale_rows_inplace();
    proj.iter_mut().for_each(|x| *x = x.clamp(-4.0, 4.0));

    Ok(proj)
}

/// Binary sort cells into groups based on projection.
fn binary_sort_shared(proj_kn: &Mat, sort_dim: usize) -> Vec<usize> {
    use data_beans_alg::random_projection::binary_sort_columns;
    let k = proj_kn.nrows();
    let sd = sort_dim.min(k).min(20);
    binary_sort_columns(proj_kn, sd).unwrap_or_else(|_| vec![0; proj_kn.ncols()])
}

/// Write linking results to parquet.
fn write_link_results(
    results: &[GeneLinkResult],
    peak_names: &[Box<str>],
    out_prefix: &str,
    gene_fc: Option<&FeatureCoarsening>,
    peak_fc: Option<&FeatureCoarsening>,
    gene_names: &[Box<str>],
) -> anyhow::Result<()> {
    let mut links: Vec<LinkRow> = Vec::new();
    for res in results {
        for (i, &pi) in res.peak_indices.iter().enumerate() {
            links.push(LinkRow {
                pip: res.pip[i],
                effect: res.effect_size[i],
                gene_idx: res.gene_idx,
                gene: res.gene_name.clone(),
                peak: peak_names[pi].clone(),
            });
        }
    }

    links.sort_by(|a, b| {
        b.pip
            .partial_cmp(&a.pip)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.gene_idx.cmp(&b.gene_idx))
    });

    if links.is_empty() {
        info!("No significant links found.");
        return Ok(());
    }

    let n_links = links.len();
    let mut mat = Mat::zeros(n_links, 2);
    let mut row_labels: Vec<Box<str>> = Vec::with_capacity(n_links);

    for (i, row) in links.iter().enumerate() {
        mat[(i, 0)] = row.pip as f32;
        mat[(i, 1)] = row.effect as f32;
        row_labels.push(format!("{}|{}", row.gene, row.peak).into());
    }

    let col_labels: Vec<Box<str>> = vec!["pip".into(), "effect_size".into()];

    let link_path = format!("{}.peak_gene_links.parquet", out_prefix);
    mat.to_parquet_with_names(
        &link_path,
        (Some(&row_labels), Some("gene_peak")),
        Some(&col_labels),
    )?;
    info!("Wrote {} links to {}", n_links, link_path);

    if let Some(gfc) = gene_fc {
        write_module_mapping(gene_names, &gfc.fine_to_coarse, out_prefix, "gene")?;
    }
    if let Some(pfc) = peak_fc {
        write_module_mapping(peak_names, &pfc.fine_to_coarse, out_prefix, "peak")?;
    }

    Ok(())
}

fn write_module_correlations(corr_matrix: &Mat, out_prefix: &str) -> anyhow::Result<()> {
    let corr_path = format!("{}.module_correlations.parquet", out_prefix);
    corr_matrix.to_parquet_with_names(&corr_path, (None, None), None)?;
    info!("Wrote module correlations to {}", corr_path);
    Ok(())
}

fn write_module_mapping(
    names: &[Box<str>],
    fine_to_coarse: &[usize],
    out_prefix: &str,
    label: &str,
) -> anyhow::Result<()> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let path = format!("{}.{}_modules.tsv.gz", out_prefix, label);
    let file = std::fs::File::create(&path)?;
    let mut gz = GzEncoder::new(file, Compression::default());

    writeln!(gz, "{}\tmodule", label)?;
    for (i, name) in names.iter().enumerate() {
        writeln!(gz, "{}\t{}", name, fine_to_coarse[i])?;
    }
    gz.finish()?;

    info!("Wrote {} module mapping to {}", label, path);
    Ok(())
}
