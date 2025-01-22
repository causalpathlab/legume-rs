mod collapse_data;
mod common;
mod normalization;
mod random_projection;

use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;
use clap::{Args, Parser, Subcommand};
use collapse_data::CollapsingOps;
use matrix_param::traits::*;
use matrix_util::common_io::{extension, read_lines};
use random_projection::RandProjOps;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.commands {
        Commands::CollapseColumns(args) => {
            // 1. push data files and collect batch membership
            let file = args.data_files[0].as_ref();
            let backend_from_first = match extension(file)?.to_string().as_str() {
                "h5" => SparseIoBackend::HDF5,
                "zarr" => SparseIoBackend::Zarr,
                _ => SparseIoBackend::Zarr,
            };

            let backend = args.backend.clone().unwrap_or(backend_from_first);

            let mut data_vec = SparseIoVec::new();
            for data_file in args.data_files.iter() {
                let data = open_sparse_matrix(&data_file, &backend)?;
                data_vec.push(Arc::from(data))?;
            }

            let mut batch_name_to_id = HashMap::new();
            let mut batch_membership = vec![];

            if let Some(batch_files) = &args.batch_files {
                if batch_files.len() != args.data_files.len() {
                    return Err(anyhow::anyhow!("# batch files != # of data files"));
                }

                for batch_file in batch_files.iter() {
                    for s in read_lines(&batch_file)? {
                        batch_membership.push(s.to_string());
                        // if let Some(&id) = batch_name_to_id.get(&s) {
                        //     batch_membership.push(id);
                        // } else {
                        //     let new_id = batch_name_to_id.len();
                        //     batch_name_to_id.insert(s.clone(), new_id);
                        //     batch_membership.push(new_id);
                        // }
                    }
                }
            } else {
                for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
                    batch_membership.extend(vec![id.to_string(); nn]);
                    batch_name_to_id.insert(id.to_string().into_boxed_str(), id);
                }
            }

            if batch_membership.len() != data_vec.num_columns()? {
                return Err(anyhow::anyhow!(
                    "# batch membership {} != # of columns {}",
                    batch_membership.len(),
                    data_vec.num_columns()?
                ));
            }

            // 2. randomly project the columns
            let proj_res = data_vec.project_columns(args.proj_dim, args.block_size.clone())?;
            proj_res
                .basis
                .to_tsv(&(args.out.to_string() + ".basis.gz"))?;

            let proj_kn = proj_res.proj;

            // let num_batches = data_vec.num_batches();
            // if num_batches > 1 {
            //     let mut proj_kb = DMatrix::<f32>::zeros(proj_kn.nrows(), num_batches);
            // }

            proj_kn
                .transpose()
                .to_tsv(&(args.out.to_string() + ".proj.gz"))?;

            data_vec.assign_columns_to_samples(Some(&proj_kn), None)?;

            // 3. register batch membership
            data_vec.register_batches(&proj_kn, &batch_membership)?;

            dbg!(data_vec.num_batches());

            // 4. final collapsing
            let ret = data_vec.collapse_columns(args.down_sample, args.knn, args.iter_opt)?;

            ret.mu.write_tsv(&(args.out.to_string() + ".mu"))?;

            if let Some(delta) = &ret.delta {
                delta.write_tsv(&(args.out.to_string() + ".delta"))?;
            }

            if let Some(gamma) = &ret.gamma {
                gamma.write_tsv(&(args.out.to_string() + ".gamma"))?;
            }
        }
    }

    Ok(())
}

#[derive(Parser)]
#[command(version, about, long_about=None)]
#[command(propagate_version = true)]
///
/// Basic utility functions for processing sparse matrices
///
/// - RP:
///
struct Cli {
    #[command(subcommand)]
    commands: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build faster backend data from mtx
    CollapseColumns(RunCollapseArgs),
}

#[derive(Args)]
pub struct RunCollapseArgs {
    /// Data files of either `.zarr` or `.h5` format. All the formats
    /// should be identical. We can convert `.mtx` to `.zarr` or `.h5`
    /// using `asap-data build`
    #[arg(required = true)]
    data_files: Vec<Box<str>>,

    /// Projection dimension
    #[arg(long, short, required = true)]
    proj_dim: usize,

    /// #k-nearest neighbours within each batch
    #[arg(long, default_value = "10")]
    knn: Option<usize>,

    /// #downsampling columns per each collapsed sample
    #[arg(long, default_value = "100")]
    down_sample: Option<usize>,

    /// optimization iterations
    #[arg(long, default_value = "100")]
    iter_opt: Option<usize>,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,

    #[arg(long)]
    /// Batch membership files. Each bach file should correspond to
    /// each data file.
    batch_files: Option<Vec<Box<str>>>,

    /// Block_size for parallel processing
    #[arg(long, value_enum, default_value = "100")]
    block_size: Option<usize>,

    /// backend to use
    #[arg(long, value_enum, default_value = "zarr")]
    backend: Option<SparseIoBackend>,
}
