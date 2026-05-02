use clap::Args;
use log::info;
use nalgebra::DVector;

use dashmap::DashMap as HashMap;
use indicatif::ParallelProgressIterator;

use data_beans::sparse_io::*;
use matrix_util::common_io::file_ext;
use matrix_util::dmatrix_util::concatenate_horizontal;
use matrix_util::traits::*;
use rand::distr::{weighted::WeightedIndex, Distribution};
use rand_distr::Gamma;
use rayon::prelude::*;

use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Args, Debug)]
pub struct SimConvArgs {
    /// single-cell data (`.zarr` or `.h5`)
    #[arg(short = 's', long, required = true)]
    sc_data_file: Box<str>,

    /// topic matrix with the first column corresponds to cell
    /// barcodes (`.parquet`, `.tsv.gz`, `.csv.gz`)
    #[arg(short = 't', long, required = true)]
    topic_file: Box<str>,

    /// number of cells per sample
    #[arg(short = 'c', long, default_value_t = 10)]
    cells_per_sample: usize,

    /// number of bulk samples
    #[arg(short = 'n', long, default_value_t = 100)]
    bulk_samples: usize,

    /// Dirichlet concentration parameter for topic mixing fractions
    /// (smaller = more skewed, e.g. 0.1-0.5; larger = more uniform)
    #[arg(short = 'a', long, default_value_t = 0.3)]
    dirichlet_alpha: f64,

    /// random seed
    #[arg(short, long, default_value_t = 42)]
    rseed: u64,

    /// output file header
    #[arg(short, long, required = true)]
    output: Box<str>,
}

pub fn generate_convoluted_data(args: &SimConvArgs) -> anyhow::Result<()> {
    type Mat = DMatrix<f32>;
    type DVec = DVector<f32>;

    // 1. read sc data and topic proportion file
    let sc_data = match file_ext(&args.sc_data_file)?.to_string().as_ref() {
        "h5" => open_sparse_matrix(&args.sc_data_file, &SparseIoBackend::HDF5),
        "zarr" => open_sparse_matrix(&args.sc_data_file, &SparseIoBackend::Zarr),
        ext => panic!("Unsupported file extension: {}", ext),
    }?;

    let MatWithNames {
        rows: cells,
        cols: topic_names,
        mat: topic_mat,
    } = match file_ext(&args.topic_file)?.as_ref() {
        "parquet" => Mat::from_parquet(&args.topic_file)?,
        _ => Mat::read_data(&args.topic_file, &['\t', ','], None, Some(0), None, None)?,
    };

    // 2. match SC cells to topic matrix rows
    let topic_cells = cells
        .into_iter()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect::<HashMap<_, _>>();

    let max_k = topic_mat.ncols();
    let sc_data_cells = sc_data.column_names()?;

    let matched_cells: Vec<(usize, usize)> = sc_data_cells
        .iter()
        .enumerate()
        .filter_map(|(sc_idx, sc)| topic_cells.get(sc).map(|pos| (sc_idx, *pos.value())))
        .collect();

    info!(
        "{} / {} cells matched between SC data and topic file",
        matched_cells.len(),
        sc_data_cells.len()
    );

    // 3. build per-topic soft cell sampling distributions
    //    weight of cell j for topic k = topic_mat[(topic_row_j, k)]
    info!("building per-topic weighted cell indices...");

    let per_topic_samplers: Vec<(Vec<usize>, WeightedIndex<f32>)> = (0..max_k)
        .map(|k| {
            let pairs: Vec<(usize, f32)> = matched_cells
                .iter()
                .map(|&(sc_idx, topic_row)| (sc_idx, topic_mat[(topic_row, k)]))
                .filter(|&(_, w)| w > 0.0)
                .collect();
            let sc_indices: Vec<usize> = pairs.iter().map(|&(i, _)| i).collect();
            let weights: Vec<f32> = pairs.iter().map(|&(_, w)| w).collect();
            let dist = WeightedIndex::new(&weights).expect("valid topic weights");
            (sc_indices, dist)
        })
        .collect();

    let n_bulk_samples = args.bulk_samples;
    let n_genes = sc_data
        .num_rows()
        .ok_or(anyhow::anyhow!("unknown # rows"))?;

    info!("simulating convoluted (bulk) data matrix...");

    // 4. build Gamma distributions for Dirichlet sampling
    let gamma_dist = Gamma::new(args.dirichlet_alpha, 1.0)
        .map_err(|e| anyhow::anyhow!("invalid Dirichlet alpha: {}", e))?;

    // 5. simulate bulk samples in parallel
    //    use a separate seed space from any prior step
    let bulk_seed_offset = 1_000_000u64;

    let results: Vec<(usize, DVec, DVec)> = (0..n_bulk_samples)
        .into_par_iter()
        .progress_count(n_bulk_samples as u64)
        .map(|s| -> anyhow::Result<(usize, DVec, DVec)> {
            let mut rng =
                StdRng::seed_from_u64(args.rseed.wrapping_add(bulk_seed_offset + s as u64));

            // sample Dirichlet fractions via Gamma trick
            let mut frac = DVec::zeros(max_k);
            for k in 0..max_k {
                frac[k] = gamma_dist.sample(&mut rng) as f32;
            }
            let frac_sum = frac.sum().max(f32::EPSILON);
            frac.unscale_mut(frac_sum);

            // sample topics for each cell slot from Dirichlet fractions
            let topic_dist = WeightedIndex::new(frac.as_slice())?;
            let mut cells_s = Vec::with_capacity(args.cells_per_sample);

            for _ in 0..args.cells_per_sample {
                let k = topic_dist.sample(&mut rng);
                let (ref sc_indices, ref cell_dist) = per_topic_samplers[k];
                let cell_idx = sc_indices[cell_dist.sample(&mut rng)];
                cells_s.push(cell_idx);
            }

            // sum selected cells into a bulk column
            let n_s = cells_s.len();
            let bulk_col = if n_s > 0 {
                sc_data.read_columns_csc(cells_s)? * DVec::from_element(n_s, 1.)
            } else {
                DVec::zeros(n_genes)
            };

            Ok((s, bulk_col, frac))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // 6. assemble output matrices
    let mut conv_ds = Mat::zeros(n_genes, n_bulk_samples);
    let mut fractions = Vec::with_capacity(n_bulk_samples);

    for (s, bulk_col, frac) in results {
        conv_ds.column_mut(s).copy_from(&bulk_col);
        fractions.push(frac);
    }

    // 2. introduce individual/sample-specific bias factors
    // let ln_delta_ds = Mat::rnorm(n_genes, n_bulk_samples).scale_columns();
    // if cell type factors are confounded with δ?

    let fractions_kn = concatenate_horizontal(&fractions)?;

    let frac_file = args.output.to_string() + ".fractions.parquet";
    let bulk_file = args.output.to_string() + ".bulk.parquet";

    let genes = sc_data.row_names()?;
    let samples = (0..args.bulk_samples)
        .map(|x| x.to_string().into_boxed_str())
        .collect::<Vec<_>>();

    fractions_kn.transpose().to_parquet_with_names(
        &frac_file,
        (Some(&samples), Some("sample")),
        Some(&topic_names),
    )?;

    conv_ds.to_parquet_with_names(
        &bulk_file,
        (Some(genes.as_ref()), Some("gene")),
        Some(&samples),
    )?;

    info!("done");
    Ok(())
}
