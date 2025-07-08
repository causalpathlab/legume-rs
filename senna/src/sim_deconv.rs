use crate::embed_common::*;
use crate::routines_pre_process::*;
use dashmap::DashMap as HashMap;
use indicatif::ParallelProgressIterator;
use matrix_util::common_io::extension;
use matrix_util::dmatrix_util::concatenate_horizontal;
use matrix_util::utils::partition_by_membership;
use rand_distr::Distribution;
use rayon::prelude::*;

use rand::distr::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Args, Debug)]
pub struct SimDeconvArgs {
    /// single-cell data (`.zarr` or `.h5`)
    #[arg(short = 's', long, value_delimiter = ',', required = true)]
    sc_data_files: Vec<Box<str>>,

    /// topic matrix with the first column corresponds to cell
    /// barcodes (`.parquet`, `.tsv.gz`, `.csv.gz`)
    #[arg(short = 't', long, required = true)]
    topic_file: Box<str>,

    /// number of cells per topic
    #[arg(short = 'c', long, default_value_t = 10)]
    cells_per_sample: usize,

    /// number of bulk samples
    #[arg(short = 'n', long, default_value_t = 100)]
    bulk_samples: usize,

    /// random seed
    #[arg(short, long, default_value_t = 42)]
    rseed: u64,

    /// output file header
    #[arg(short, long, required = true)]
    output: Box<str>,
}

pub fn sim_deconv(args: &SimDeconvArgs) -> anyhow::Result<()> {
    // 1. read sc data and topic proportion file
    let SparseDataWithBatch {
        data: mut sc_data,
        batch: _,
    } = read_sparse_data_with_membership(ReadArgs {
        data_files: args.sc_data_files.clone(),
        batch_files: None,
    })?;

    let MatWithNames {
        rows: cells,
        cols: topic_names,
        mat: topic_mat,
    } = match extension(&args.topic_file)?.as_ref() {
        "parquet" => Mat::from_parquet(&args.topic_file)?,
        _ => Mat::read_data(&args.topic_file, &['\t', ','], None, Some(0), None, None)?,
    };

    // 2. figure out matching cell barcodes/names and partition them into
    let topic_cells = cells
        .into_iter()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect::<HashMap<_, _>>();

    let max_k = topic_mat.ncols();
    let runif_k = Uniform::new(0, max_k)?;
    let sc_data_cells = sc_data.column_names()?;

    let rng = StdRng::seed_from_u64(args.rseed);

    let cell_to_topic = sc_data_cells
        .par_iter()
        .map_init(
            || rng.clone(), // Clone the RNG for each thread
            |local_rng, sc| {
                if let Some(pos) = topic_cells.get(sc) {
                    let theta = topic_mat.row(*pos.value());
                    runif_k.sample(local_rng) // Use the thread-local RNG
                } else {
                    max_k
                }
            },
        )
        .collect::<Vec<_>>();

    let topic_to_cells = partition_by_membership(&cell_to_topic, None);

    let n_bulk_samples = args.bulk_samples;

    info!("simulating convoluted data matrix");

    let mut out_ds = Mat::zeros(sc_data.num_rows()?, n_bulk_samples);
    let arc_conv = Arc::new(Mutex::new(&mut out_ds));
    let fractions = (0..n_bulk_samples)
        .par_bridge()
        .progress_count(n_bulk_samples as u64)
        .map(|s| -> anyhow::Result<DVec> {
            let mut rng = StdRng::seed_from_u64(args.rseed + (s as u64));

            let topics: Vec<usize> = (0..args.cells_per_sample)
                .map(|_| rng.random_range(0..max_k))
                .collect();

            let mut cells_s = Vec::with_capacity(args.cells_per_sample);
            let mut ncells_s = DVec::zeros(max_k);

            for k in topics {
                if let Some(cells) = topic_to_cells.get(&k) {
                    let n_k = cells.len();
                    if n_k > 0 {
                        cells_s.push(cells[rng.random_range(0..n_k)]);
                        ncells_s[k] += 1.0;
                    }
                }
            }

            let n_s = cells_s.len();
            if n_s > 0 {
                let x =
                    sc_data.read_columns_csc(cells_s.into_iter())? * DVec::from_element(n_s, 1.);
                let mut convoluted = arc_conv.lock().expect("lock");
                convoluted.column_mut(s).copy_from(&x);
            }

            let frac = ncells_s.unscale(ncells_s.sum().max(1.0));
            Ok(frac)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let fractions_kn = concatenate_horizontal(&fractions)?;

    let frac_file = args.output.to_string() + ".fractions.parquet";
    let bulk_file = args.output.to_string() + ".bulk.parquet";
    let genes = sc_data.row_names()?;
    let samples = (0..args.bulk_samples)
        .map(|x| x.to_string().into_boxed_str())
        .collect::<Vec<_>>();

    fractions_kn
        .transpose()
        .to_parquet(Some(&samples), Some(&topic_names), &frac_file)?;
    out_ds.to_parquet(Some(genes.as_ref()), Some(&samples), &bulk_file)?;

    info!("done");
    Ok(())
}
