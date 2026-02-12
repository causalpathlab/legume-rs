use crate::srt_common::*;
use data_beans::convert::try_open_or_convert;

pub struct SRTReadArgs {
    pub data_files: Vec<Box<str>>,
    pub coord_files: Vec<Box<str>>,
    pub preload_data: bool,
    pub coord_columns: Vec<usize>,
    pub coord_column_names: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub header_in_coord: Option<usize>,
}

pub struct SRTData {
    pub data: SparseIoVec,
    pub coordinates: Mat,
    pub coordinate_names: Vec<Box<str>>,
    pub batches: Vec<Box<str>>,
}

pub fn read_expr_data(data_files: &[Box<str>]) -> anyhow::Result<SparseIoVec> {
    if data_files.is_empty() {
        return Err(anyhow::anyhow!("empty data files"));
    }

    let attach_data_name = data_files.len() > 1;
    let mut data_vec = SparseIoVec::new();

    for data_file in data_files.iter() {
        info!("Importing data file: {}", data_file);
        let data = try_open_or_convert(data_file)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;
        data_vec.push(Arc::from(data), data_name)?;
    }

    Ok(data_vec)
}

pub fn read_data_with_coordinates(args: SRTReadArgs) -> anyhow::Result<SRTData> {
    let attach_data_name = args.data_files.len() > 1;
    let mut data_vec = SparseIoVec::new();

    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        let mut data = try_open_or_convert(data_file)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;

        if args.preload_data {
            data.preload_columns()?;
        }

        data_vec.push(Arc::from(data), data_name)?;
    }

    // check if row names are the same across data
    let row_names = data_vec[0].row_names()?;

    for j in 1..data_vec.len() {
        let row_names_j = data_vec[j].row_names()?;
        if row_names != row_names_j {
            return Err(anyhow::anyhow!("Row names are not the same"));
        }
    }

    let mut coord_vec = Vec::with_capacity(args.coord_files.len());

    let mut coord_column_names = vec![];

    for (i, coord_file) in args.coord_files.iter().enumerate() {
        info!("Reading coordinate file: {}", coord_file);
        let ext = file_ext(coord_file)?;

        let MatWithNames {
            rows: coord_cell_names,
            cols: column_names,
            mat: data,
        } = match ext.as_ref() {
            "parquet" => Mat::from_parquet_with_indices_names(
                coord_file,
                Some(0),
                Some(&args.coord_columns),
                Some(&args.coord_column_names),
            )?,
            _ => Mat::read_data(
                coord_file,
                &['\t', ',', ' '],
                args.header_in_coord,
                Some(0),
                Some(&args.coord_columns),
                Some(&args.coord_column_names),
            )?,
        };

        let data_cell_names = data_vec[i].column_names()?;

        if coord_column_names.is_empty() {
            coord_column_names.extend(column_names);
        } else if coord_column_names != column_names {
            return Err(anyhow::anyhow!(
                "coordinate column names do not match with each other"
            ));
        }

        if data_cell_names == coord_cell_names {
            coord_vec.push(data);
        } else {
            info!("reordering coordinate information");

            let coord_index_map: HashMap<&Box<str>, usize> = coord_cell_names
                .iter()
                .enumerate()
                .map(|(index, name)| (name, index))
                .collect();

            let reordered_indices: Vec<usize> = data_cell_names
                .iter()
                .map(|name| {
                    coord_index_map
                        .get(name)
                        .ok_or_else(|| {
                            anyhow::anyhow!("cell '{}' not found in the file {}", name, coord_file)
                        })
                        .copied()
                })
                .collect::<anyhow::Result<_>>()?;

            coord_vec.push(concatenate_vertical(
                &reordered_indices
                    .iter()
                    .map(|&index| data.row(index))
                    .collect::<Vec<_>>(),
            )?);
        }
    }

    let coord_nk = concatenate_vertical(&coord_vec)?;

    // will incorporate batch label as an additional coordinate
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
        info!("Each data file will be considered a different batch.");
        for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
            batch_membership.extend(vec![id.to_string().into_boxed_str(); nn]);
        }
    }

    if batch_membership.len() != data_vec.num_columns() {
        return Err(anyhow::anyhow!(
            "# batch membership {} != # of columns {}",
            batch_membership.len(),
            data_vec.num_columns()
        ));
    }

    // use batch index as another coordinate
    let uniq_batches = batch_membership.par_iter().cloned().collect::<HashSet<_>>();
    let n_batches = uniq_batches.len();
    let coord_nk = if n_batches > 1 {
        info!("attaching {} batch index coordinate(s)", n_batches);
        coord_column_names.push("batch".to_string().into_boxed_str());
        append_batch_coordinate(&coord_nk, &batch_membership)?
    } else {
        coord_nk
    };

    info!(
        "Read {} x {} coordinates",
        coord_nk.nrows(),
        coord_nk.ncols()
    );

    Ok(SRTData {
        data: data_vec,
        coordinates: coord_nk,
        coordinate_names: coord_column_names,
        batches: batch_membership,
    })
}

fn append_batch_coordinate<T>(coords: &Mat, batch_membership: &[T]) -> anyhow::Result<Mat>
where
    T: Sync + Send + Clone + Eq + std::hash::Hash + std::fmt::Debug,
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
            let b = *batch_index
                .get(k)
                .ok_or_else(|| anyhow::anyhow!("batch key {:?} not found in index", k))?;
            Ok(width * (b as f32))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let bb = Mat::from_vec(coords.nrows(), 1, batch_coord);

    concatenate_horizontal(&[coords.clone(), bb])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_batch_coordinate_single_batch() {
        let coords = Mat::from_vec(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let batches = vec!["A", "A", "A"];
        let result = append_batch_coordinate(&coords, &batches).unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 3); // original 2 + 1 batch
                                       // Single batch → all batch coordinates should be 0.0
        for i in 0..3 {
            assert_eq!(result[(i, 2)], 0.0);
        }
    }

    #[test]
    fn test_append_batch_coordinate_two_batches() {
        let coords = Mat::from_vec(4, 2, vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]);
        let batches = vec!["A", "A", "B", "B"];
        let result = append_batch_coordinate(&coords, &batches).unwrap();
        assert_eq!(result.ncols(), 3);
        // Batch coordinates should differ between A and B
        let batch_a = result[(0, 2)];
        let batch_b = result[(2, 2)];
        assert_ne!(
            batch_a, batch_b,
            "different batches should have different coordinates"
        );
        // Same batch should have same coordinate
        assert_eq!(result[(0, 2)], result[(1, 2)]);
        assert_eq!(result[(2, 2)], result[(3, 2)]);
    }

    #[test]
    fn test_append_batch_coordinate_scaling() {
        // Width = max - min of all coords
        let coords = Mat::from_vec(2, 1, vec![0.0, 10.0]);
        let batches = vec!["X", "Y"];
        let result = append_batch_coordinate(&coords, &batches).unwrap();
        // Width = 10.0, one batch at 0*width, other at 1*width
        let vals: Vec<f32> = (0..2).map(|i| result[(i, 1)]).collect();
        let (lo, hi) = if vals[0] < vals[1] {
            (vals[0], vals[1])
        } else {
            (vals[1], vals[0])
        };
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 10.0); // 1 * width
    }

    #[test]
    fn test_append_batch_coordinate_mismatch() {
        let coords = Mat::from_vec(2, 1, vec![0.0, 1.0]);
        let batches = vec!["A"]; // wrong length
        assert!(append_batch_coordinate(&coords, &batches).is_err());
    }
}
