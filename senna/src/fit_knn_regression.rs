use crate::embed_common::*;
use crate::senna_input::*;

use data_beans::sparse_data_visitors::VisitColumnsOps;
use matrix_util::common_io::*;
use matrix_util::dmatrix_util::*;
use matrix_util::utils::partition_by_membership;

use fnv::FnvHashMap as HashMap;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;

#[derive(Args, Debug)]
pub struct KnnImputeArgs {
    #[arg(
        short = 'x',
        long = "x-data",
        value_delimiter(','),
        required = true,
        help = "Data files of predictor cells (X)",
        long_help = "These data files provide a richer basis set \n\
		     than the other files that will be predicted/imputed\n\
		     based on k-nearest neighbourhood mapping."
    )]
    x_data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files for X data",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file.\n\
		     Example: batch1.csv,batch2.csv"
    )]
    x_batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'y',
        long = "y-data",
        value_delimiter(','),
        required = true,
        help = "Data files of to-be-impute cells/columns (Y)",
        long_help = "Target data sets to be matched with the cells in X\n\
		     so that we have the same set of columns/cells."
    )]
    y_data_files: Vec<Box<str>>,

    #[arg(
        long,
        short,
        value_delimiter(','),
        help = "Batch membership files for Y data",
        long_help = "Batch membership files (comma-separated names).\n\
		     Each batch file should correspond to each data file.\n\
		     Example: batch1.gz,batch2.gz"
    )]
    y_batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output header",
        long_help = "Output header for the imputed files of Y data:\n\
		     \n\
		     `{out}_by_{basename}.{backend}`\n\
		     \n\
		     where `{basename}` and `{backend}` will be\n\
		     inferred from `x-data` to have the same backend.\n\
		     If there are more backend files for `x-data`,\n\
		     a unique index will be added to distinguish them."
    )]
    out: Box<str>,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension",
        long_help = "Random projection dimension to project the data.\n\
		     Controls the dimensionality of the random projection step.\n\
		     We will match target Y to source X columns/cells based on.\n\
		     this projection matrix."
    )]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of k-nearest neighbours in Y data",
        long_help = "Number of k-nearest neighbours in Y data.\n\
		     Controls the number of cells considered \n\
		     for nearest neighbour search for each column of X.\n\
		     We match each x(j) vector with K y vectors and predict\n\
		     yhat(j) that corresponds to x(j) for joint analysis."
    )]
    knn_cells: usize,

    #[arg(
        long,
        default_value_t = 1e-2,
        help = "Cutoff value to determine zero imputation",
        long_help = "Cutoff value to determine zero imputation."
    )]
    impute_cutoff: f32,

    #[arg(
        long,
        short,
        help = "Verbosity",
        long_help = "Enable verbose output.\n\
		     Prints additional information during execution."
    )]
    verbose: bool,
}

pub fn fit_knn_regression(args: &KnnImputeArgs) -> anyhow::Result<()> {
    if args.verbose {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    ////////////////////////////////////////////
    // 1. Read the data with batch membership //
    ////////////////////////////////////////////

    let SparseDataWithBatch {
        data: mut x_data,
        batch: x_batch_membership,
        nbatch: _x_nbatch,
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.x_data_files.clone(),
        batch_files: args.x_batch_files.clone(),
        preload: false,
    })?;

    let SparseDataWithBatch {
        data: mut y_data,
        batch: y_batch_membership,
        nbatch: _y_nbatch,
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.y_data_files.clone(),
        batch_files: args.y_batch_files.clone(),
        preload: false,
    })?;

    ////////////////////////////////////////////
    // 2. figure out features present in both //
    ////////////////////////////////////////////
    let x_features: HashMap<Box<str>, usize> = x_data
        .row_names()?
        .into_iter()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect();

    let mut x_row_index = HashMap::default();
    let mut y_row_index = HashMap::default();
    let mut common_rows = vec![];
    {
        let mut v = 0_usize;
        for (y_pos, y) in y_data.row_names()?.into_iter().enumerate() {
            if let Some(&x_pos) = x_features.get(&y) {
                x_row_index.insert(x_pos, v);
                y_row_index.insert(y_pos, v);
                common_rows.push(y);
                v += 1;
            }
        }
        info!("found {} common features", v);

        if v == 0 {
            return Err(anyhow::anyhow!(
                "no common features found between the X and Y"
            ));
        }
    }

    /////////////////////////////////////////////////
    // 3. project X and Y data on the shared basis //
    /////////////////////////////////////////////////

    let n_common_rows = x_row_index.len();
    let basis_dk = Mat::rnorm(n_common_rows, args.proj_dim);

    let y_proj_kn = project_columns(
        &mut y_data,
        &y_batch_membership,
        &BasisOnSubsetRows {
            basis_dk: &basis_dk,
            subset: &y_row_index,
        },
    )?;

    let x_proj_kn = project_columns(
        &mut x_data,
        &x_batch_membership,
        &BasisOnSubsetRows {
            basis_dk: &basis_dk,
            subset: &x_row_index,
        },
    )?;

    let n_y = y_data.num_columns()?;
    let n_x = x_data.num_columns()?;
    let n_tot = n_x + n_y;

    info!("randomized SVD on the projected data");
    let mut combined_kn = concatenate_horizontal(&[x_proj_kn, y_proj_kn])?;
    combined_kn.scale_columns_inplace();
    let (_, _, rsvd_vt) = combined_kn.rsvd(args.proj_dim)?;
    let mut rsvd_v = rsvd_vt.transpose();
    rsvd_v.scale_columns_inplace();
    let x_proj_kn = subset_columns(&rsvd_v, 0..n_x)?;
    let y_proj_kn = subset_columns(&rsvd_v, n_x..n_tot)?;
    y_data.register_batches_dmatrix(&y_proj_kn, &y_batch_membership)?;

    //////////////////////////////////////////////
    // 4. Counterfactual Y for each column of X //
    //////////////////////////////////////////////

    let x_column_names = x_data.column_names()?;
    let y_row_names = y_data.row_names()?;
    let nnz_y = y_data.num_non_zeros()?;
    let avg_y = nnz_y.div_ceil(n_y).max(1);

    let x_backend_cols = x_data.take_backend_columns();
    let n_out = x_backend_cols.len();

    info!("Prediction of Y for each backend of X (N={})", n_out);

    for (didx, (x_backend_file, x_columns)) in x_backend_cols.into_iter().enumerate() {
        let ext = extension(&x_backend_file)?;
        let base = basename(&x_backend_file)?;
        let backend = match ext.as_ref() {
            "zarr" => SparseIoBackend::Zarr,
            "h5" => SparseIoBackend::HDF5,
            _ => return Err(anyhow::anyhow!("Unknown file format: {}", x_backend_file)),
        };

        let output_file = if n_out > 1 {
            format!("{}.{}_by_{}.{}", args.out, didx, base, ext)
        } else {
            format!("{}_by_{}.{}", args.out, base, ext)
        };

        if std::path::Path::new(&output_file).exists() {
            info!(
                "This existing backend file '{}' will be deleted",
                &output_file
            );
            remove_file(&output_file)?;
        }

        let n_x = x_columns.len();
        let expected = avg_y * 2 * n_x;
        let col_names: Vec<Box<str>> = x_columns
            .iter()
            .map(|&j| x_column_names[j].clone())
            .collect();

        let mut full_triplets = Vec::with_capacity(expected);
        let arc_triplets = Arc::new(Mutex::new(&mut full_triplets));
        x_columns
            .into_iter()
            .enumerate()
            .par_bridge()
            .progress_count(n_x as u64)
            .map(|(j_new, k)| -> anyhow::Result<()> {
                let (y, _cols, dd) =
                    y_data.query_columns_by_data_csc(x_proj_kn.column(k), args.knn_cells)?;

                let mut ww = CscMat::from_nonzero_triplets(
                    y.ncols(),
                    1,
                    &dd.into_iter()
                        .enumerate()
                        .map(|(i, d)| (i, 0, -d))
                        .collect::<Vec<_>>(),
                )?;
                ww.normalize_exp_logits_columns_inplace();

                let yhat = y * ww;

                let ret: Vec<(u64, u64, f32)> = yhat
                    .to_nonzero_triplets()?
                    .triplets
                    .into_iter()
                    .filter_map(|(i, _, x_ij)| {
                        if x_ij < args.impute_cutoff {
                            None
                        } else {
                            Some((i as u64, j_new as u64, x_ij))
                        }
                    })
                    .collect();

                let mut triplets = arc_triplets.lock().expect("failed to lock triplets");

                if ret.is_empty() {
                    // add a dummy column if needed
                    triplets.extend(vec![(0, j_new as u64, 0.0)]);
                } else {
                    triplets.extend(ret);
                }

                Ok(())
            })
            .collect::<anyhow::Result<()>>()?;

        let nrows = y_data.num_rows()?;
        let ncols = n_x;
        let nnz = full_triplets.len();

        let mut data = create_sparse_from_triplets(
            &full_triplets,
            (nrows, ncols, nnz),
            Some(&output_file),
            Some(&backend),
        )?;

        data.register_column_names_vec(&col_names);
        data.register_row_names_vec(&y_row_names);
        info!("created imputed data file: {}", output_file);
    }

    Ok(())
}

fn project_columns(
    data: &mut SparseIoVec,
    batch_membership: &[Box<str>],
    basis: &BasisOnSubsetRows,
) -> anyhow::Result<Mat> {
    let mut proj_kn = Mat::zeros(basis.basis_dk.ncols(), data.num_columns()?);

    data.visit_columns_by_block(&project_columns_visitor, basis, &mut proj_kn, None)?;

    let batches = partition_by_membership(batch_membership, None);

    for (_, cols) in batches.iter() {
        let xx = subset_columns(&proj_kn, cols.iter().cloned())?
            .transpose() // n x k
            .centre_columns() // adjust the mean
            .transpose(); // k x n
        assign_columns(&xx, cols.iter().cloned(), &mut proj_kn);
    }

    Ok(proj_kn)
}

struct BasisOnSubsetRows<'a> {
    basis_dk: &'a Mat,
    subset: &'a HashMap<usize, usize>,
}

fn project_columns_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    basis_on_subset_rows: &BasisOnSubsetRows,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;

    let basis_dk = basis_on_subset_rows.basis_dk;
    let subset = basis_on_subset_rows.subset;

    let xx_dm = data.read_columns_csc(lb..ub)?;
    let triplets_subset = xx_dm
        .triplet_iter()
        .filter_map(|(i, j, &x_ij)| {
            if let Some(&i_subset) = subset.get(&i) {
                Some((i_subset, j, x_ij))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut xx_dm = CscMat::from_nonzero_triplets(subset.len(), xx_dm.ncols(), &triplets_subset)?;
    for x in xx_dm.values_mut() {
        *x = x.ln_1p();
    }
    xx_dm.normalize_columns_inplace();
    let chunk = (xx_dm.transpose() * basis_dk).transpose();

    let mut proj_kn = arc_proj_kn.lock().expect("proj_kn lock");
    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);

    Ok(())
}
