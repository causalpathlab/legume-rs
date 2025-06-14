// #![allow(dead_code)]

use crate::srt_common::*;
use crate::SRTArgs;
use matrix_util::common_io::extension as file_ext;

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

    for (i, coord_file) in args.coord_files.iter().enumerate() {
        info!("Reading coordinate file: {}", coord_file);

        let cell_names = data_vec[i].column_names()?;

        let ext = file_ext(&coord_file)?;

        let coord = match ext.as_ref() {
            "parquet" | "csv.gz" | "tsv.gz" | "txt.gz" => {
                info!("parsing coordinate file: {}", &coord_file);

                let (coord_cell_names, _, data) = match ext.as_ref() {
                    "parquet" => Mat::from_parquet_with_indices_names(
                        &coord_file,
                        Some(0),
                        args.coord_columns.as_deref(),
                        Some(&args.coord_column_names),
                    )?,
                    _ => Mat::read_data(
                        &coord_file,
                        vec!['\t', ',', ' '],
                        Some(0),
                        Some(0),
                        args.coord_columns.as_deref(),
                        Some(&args.coord_column_names),
                    )?,
                };

                if cell_names == coord_cell_names {
                    data
                } else {

		    info!("reordering coordinate information");
		    
                    let coord_index_map: HashMap<&Box<str>, usize> = coord_cell_names
                        .iter()
                        .enumerate()
                        .map(|(index, name)| (name, index))
                        .collect();

                    let reordered_indices: Vec<usize> = cell_names
                        .iter()
                        .map(|name| {
                            coord_index_map
                                .get(name)
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "cell_name '{}' not found in coord_cell_names",
                                        name
                                    )
                                })
                                .map(|&index| index)
                        })
                        .collect::<anyhow::Result<_>>()?;

                    concatenate_vertical(
                        &reordered_indices
                            .iter()
                            .map(|&index| data.row(index))
                            .collect::<Vec<_>>(),
                    )?
                }
            }

            _ => Mat::read_file_delim(coord_file, vec!['\t', ',', ' '], None)?,
        };

        coord_vec.push(coord);
    }

    let coord_nk = concatenate_vertical(&coord_vec)?;

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

    info!(
        "Read {} x {} coordinates",
        coord_nk.nrows(),
        coord_nk.ncols()
    );

    Ok((data_vec, coord_nk, batch_membership))
}

fn append_batch_coordinate<T>(coords: &Mat, batch_membership: &Vec<T>) -> anyhow::Result<Mat>
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

    concatenate_horizontal(&[coords.clone(), bb])
}
