#![allow(dead_code)]

use crate::srt_common::*;
use crate::SRTArgs;

use indicatif::ParallelProgressIterator;
use log::info;

use asap_alg::normalization::NormalizeDistance;
use asap_data::sparse_io::SparseIoBackend;
use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;

use matrix_util::common_io::{extension, read_lines};
use matrix_util::dmatrix_util::*;
use matrix_util::knn_match::ColumnDict;
use matrix_util::traits::*;
use matrix_util::utils::partition_by_membership;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

///
///
///
pub fn spectral_network_embedding(
    coordinates_nc: &Mat,
    knn: usize,
    max_rank: usize,
) -> anyhow::Result<Mat> {
    let nn = coordinates_nc.nrows();

    let points = coordinates_nc.transpose();

    let points = points.column_iter().collect::<Vec<_>>();

    let names = (0..nn).collect::<Vec<_>>();

    let dict = ColumnDict::from_dvector_views(points, names);

    let target_exp_sum = 2_f32.ln();
    let nquery = (knn + 1).min(nn);

    let triplets = (0..nn)
        .par_bridge()
        .progress_count(nn as u64)
        .map(|i| {
            let (neighbours, distances) = dict
                .search_others(&i, nquery)
                .expect("failed to search k-nearest neighbours");

            let weights = distances.into_iter().normalized_exp(target_exp_sum);

            neighbours
                .iter()
                .zip(weights)
                .filter_map(|(&j, w_ij)| if i == j { None } else { Some((i, j, w_ij)) })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    let adj = CscMat::from_nonzero_triplets(nn, nn, triplets)?;
    let adj_t = adj.transpose();
    let adj = adj * 0.5 + adj_t * 0.5;
    let (ret, _, _) = adj.rsvd(max_rank.min(nn))?;

    Ok(ret)
}

///
/// each position vector (1 x d) `*` random_proj (d x r)
///
pub fn positional_projection<T>(
    coords: &Mat,
    batch_membership: &Vec<T>,
    d: usize,
) -> anyhow::Result<Mat>
where
    T: Sync + Send + Clone + Eq + std::hash::Hash,
{
    if coords.nrows() != batch_membership.len() {
        return Err(anyhow::anyhow!("incompatible batch membership"));
    }

    let maxval = coords.max();
    let minval = coords.min();
    let coords = coords
        .map(|x| (x - minval) / (maxval - minval + 1.))
        .transpose();

    let batches = partition_by_membership(batch_membership, None);

    let mut ret = Mat::zeros(d, coords.ncols());

    for (_b, points) in batches.into_iter() {
        let rand_proj = Mat::rnorm(d, coords.nrows());

        points.into_iter().for_each(|p| {
            ret.column_mut(p)
                .copy_from(&(&rand_proj * coords.column(p)));
        });
    }

    let (lb, ub) = (-4., 4.);
    ret.scale_columns_inplace();

    if ret.max() > ub || ret.min() < lb {
        info!("Clamping values [{}, {}] after standardization", lb, ub);
        ret.iter_mut().for_each(|x| {
            *x = x.clamp(lb, ub);
        });
        ret.scale_columns_inplace();
    }
    Ok(ret)
}

pub fn read_data_vec(args: SRTArgs) -> anyhow::Result<(SparseIoVec, Mat, Vec<Box<str>>)> {
    // push data files and collect batch membership
    let file = args.data_files[0].as_ref();
    let backend = match extension(file)?.to_string().as_str() {
        "h5" => SparseIoBackend::HDF5,
        "zarr" => SparseIoBackend::Zarr,
        _ => SparseIoBackend::Zarr,
    };

    if args.coord_files.len() != args.data_files.len() {
        return Err(anyhow::anyhow!("# coordinate files != # of data files"));
    }

    let mut data_vec = SparseIoVec::new();

    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        match extension(data_file)?.as_ref() {
            "zarr" => {
                assert_eq!(backend, SparseIoBackend::Zarr);
            }
            "h5" => {
                assert_eq!(backend, SparseIoBackend::HDF5);
            }
            _ => return Err(anyhow::anyhow!("Unknown file format: {}", data_file)),
        };

        let data = open_sparse_matrix(data_file, &backend)?;
        data_vec.push(Arc::from(data))?;
    }

    // check if row names are the same
    let row_names = data_vec[0].row_names()?;

    for j in 1..data_vec.len() {
        let row_names_j = data_vec[j].row_names()?;
        if row_names != row_names_j {
            return Err(anyhow::anyhow!("Row names are not the same"));
        }
    }

    let mut coord_vec = Vec::with_capacity(args.coord_files.len());

    for coord_file in args.coord_files.iter() {
        info!("Reading coordinate file: {}", coord_file);
        let coord = Mat::read_file_delim(coord_file, vec!['\t', ',', ' '], None)?;
        coord_vec.push(coord);
    }

    let coord_nk = concatenate_vertical(coord_vec)?;

    // check batch membership
    let mut batch_membership = Vec::with_capacity(data_vec.len());

    if let Some(batch_files) = &args.batch_files {
        if batch_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!("# batch files != # of data files"));
        }

        for batch_file in batch_files.iter() {
            info!("Reading batch file: {}", batch_file);
            for s in read_lines(batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
        }
    } else {
        for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
            batch_membership.extend(vec![id.to_string().into_boxed_str(); nn]);
        }
    }

    if batch_membership.len() != data_vec.num_columns()? {
        return Err(anyhow::anyhow!(
            "# batch membership {} != # of columns {}",
            batch_membership.len(),
            data_vec.num_columns()?
        ));
    }

    // use batch index as another coordinate
    let uniq_batches = batch_membership.par_iter().collect::<HashSet<_>>();
    let n_batches = uniq_batches.len();
    let coord_nk = if n_batches > 1 {
        info!("attaching {} batch index coordinate(s)", n_batches);
        append_batch_coordinate(&coord_nk, &batch_membership)?
    } else {
        coord_nk
    };

    Ok((data_vec, coord_nk, batch_membership))
}

pub fn append_batch_coordinate<T>(coords: &Mat, batch_membership: &Vec<T>) -> anyhow::Result<Mat>
where
    T: Sync + Send + Clone + Eq + std::hash::Hash,
{
    if coords.nrows() != batch_membership.len() {
        return Err(anyhow::anyhow!("incompatible batch membership"));
    }

    let minval = coords.min();
    let maxval = coords.max();
    let width = (maxval - minval).max(1.);

    let uniq_batches = batch_membership.iter().collect::<HashSet<_>>();

    let batch_index = uniq_batches
        .into_iter()
        .enumerate()
        .map(|(k, v)| (v, k))
        .collect::<HashMap<_, _>>();

    let batch_coord = batch_membership
        .iter()
        .map(|k| {
            let b = batch_index[k];
            width * (b as f32)
        })
        .collect::<Vec<_>>();

    let bb = Mat::from_vec(coords.nrows(), 1, batch_coord);

    concatenate_horizontal(vec![coords.clone(), bb])
}
