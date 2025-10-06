use crate::common::*;
use matrix_util::common_io::*;
use matrix_util::dmatrix_util::*;
use matrix_util::parquet::*;
use matrix_util::traits::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub struct InputDataArgs {
    pub data_files: Vec<Box<str>>,
    pub indv_files: Option<Vec<Box<str>>>,
    pub topic_assignment_files: Option<Vec<Box<str>>>,
    pub topic_proportion_files: Option<Vec<Box<str>>>,
    pub exposure_assignment_file: Option<Box<str>>,
    pub preload_data: bool,
}

pub struct InputData {
    pub sparse_data: SparseIoVec,
    pub cell_to_indv: Vec<Box<str>>,
    pub cell_topic: Mat,
    pub sorted_topic_names: Vec<Box<str>>,
    pub indv_to_exposure: Option<HashMap<Box<str>, Box<str>>>,
    pub exposure_id: Option<HashMap<Box<str>, usize>>,
}

pub fn read_input_data(args: InputDataArgs) -> anyhow::Result<InputData> {
    if let Some(indv_files) = args.indv_files.as_ref() {
        if indv_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!("# sample files != # of data files"));
        }
    }

    let data_files = args.data_files.clone();
    let indv_files = args.indv_files.clone();
    let topic_assignment_files = match args.topic_assignment_files {
        Some(vec) => vec.into_iter().map(Some).collect(),
        None => vec![None; data_files.len()],
    };
    let topic_proportion_files = match args.topic_proportion_files {
        Some(vec) => vec.into_iter().map(Some).collect(),
        None => vec![None; data_files.len()],
    };

    info!("Looking into the topic files...");

    let mut topic_names = HashSet::<Box<str>>::new();

    for (a_file, p_file) in topic_assignment_files
        .iter()
        .zip(topic_proportion_files.iter())
    {
        if let Some(a_file) = a_file {
            let _names = read_lines_of_words_delim(a_file.as_ref(), &['\t', ',', ' '], -1)?
                .lines
                .iter()
                .filter_map(|line| (line.len() > 1).then(|| line[1].clone()))
                .collect::<HashSet<_>>();
            for _x in _names {
                topic_names.insert(_x);
            }
        }

        if let Some(p_file) = p_file {
            let ext = extension(p_file)?;
            if ext.as_ref() == "parquet" {
                let _names = peek_parquet_field_names(p_file)?;
                for _x in _names {
                    topic_names.insert(_x);
                }
            }
        }
    }

    let mut sorted_topic_names: Vec<Box<str>> = topic_names.into_iter().collect();
    sorted_topic_names.sort();

    let topic_names: HashMap<Box<str>, usize> = sorted_topic_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();

    info!("Found {} topics", topic_names.len());

    ////////////////////////////////////
    // Read matched data, topic, indv //
    ////////////////////////////////////

    let mut topic_vec = vec![];
    let mut sparse_data = SparseIoVec::new();
    let mut cell_to_indv = vec![];

    for (f, this_data_file) in data_files.iter().enumerate() {
        let this_indv_file = indv_files.as_ref().map(|files| files[f].clone());

        info!("Importing: {}, {:?}", this_data_file, this_indv_file);

        let backend = match extension(this_data_file)?.to_string().as_str() {
            "h5" => SparseIoBackend::HDF5,
            "zarr" => SparseIoBackend::Zarr,
            _ => {
                return Err(anyhow::anyhow!("unknown backend"));
            }
        };

        let mut this_data = open_sparse_matrix(&this_data_file, &backend)?;

        if args.preload_data {
            this_data.preload_columns()?;
        }

        let ndata = this_data.num_columns().unwrap_or(0);

        let missing = "NA".to_string().into_boxed_str();

        let this_indv = match this_indv_file.as_ref() {
            Some(input_file) => read_lines_of_words_delim(input_file, &['\t', ',', ' '], -1)?.lines,
            None => vec![vec![missing.clone(); 1]; ndata],
        };

        if this_indv.len() != ndata {
            return Err(anyhow::anyhow!(
                "{:?} is unmatched with {}",
                this_indv_file.clone(),
                this_data_file.clone()
            ));
        }

        let this_indv: Vec<Box<str>> = match this_indv[0].len() {
            1 => this_indv.into_iter().map(|w| w[0].clone()).collect(),
            _ => {
                let cell_to_indv: HashMap<_, _> = this_indv
                    .into_iter()
                    .filter_map(|w| (w.len() > 1).then(|| (w[0].clone(), w[1].clone())))
                    .collect();

                this_data
                    .column_names()?
                    .iter()
                    .map(|c| {
                        cell_to_indv
                            .get(c)
                            .cloned()
                            .unwrap_or_else(|| missing.clone())
                    })
                    .collect()
            }
        };

        let topic_a_file = &topic_assignment_files[f];
        let topic_p_file = &topic_proportion_files[f];

        let mut this_topic = if topic_names.is_empty() {
            Mat::from_element(ndata, 1, 1.0)
        } else {
            Mat::zeros(ndata, topic_names.len())
        };

        let cells_to_rows = this_data
            .column_names()?
            .into_iter()
            .enumerate()
            .map(|(i, x)| (x, i))
            .collect::<HashMap<_, _>>();

        if let Some(a_file) = topic_a_file {
            info!("importing topic information from {}", a_file);

            let ReadLinesOut { lines, header: _ } =
                read_lines_of_words_delim(a_file.as_ref(), &['\t', ',', ' '], -1)?;

            for words in lines {
                if words.len() > 1 {
                    if let Some(&r) = cells_to_rows.get(&words[0]) {
                        if let Some(&c) = topic_names.get(&words[1]) {
                            this_topic[(r, c)] = 1.0;
                        }
                    }
                }
            }
        }

        if let Some(p_file) = topic_p_file {
            let ext = extension(p_file)?;
            if ext.as_ref() == "parquet" {
                info!("importing topic information from {}", p_file);

                let MatWithNames { mat, rows, cols } = Mat::from_parquet(&p_file)?;

                for j in 0..mat.ncols() {
                    if let Some(&c) = topic_names.get(&cols[j]) {
                        for i in 0..mat.nrows() {
                            if let Some(&r) = cells_to_rows.get(&rows[i]) {
                                this_topic[(r, c)] = mat[(i, j)];
                            }
                        }
                    }
                }
            }
        }

        let data_name = basename(&this_data_file)?;
        sparse_data.push(Arc::from(this_data), Some(data_name))?;
        cell_to_indv.extend(this_indv);
        topic_vec.push(this_topic);
    }

    let cell_topic = concatenate_vertical(topic_vec.as_slice())?;
    info!("Total {} data sets combined", sparse_data.len());

    if let Some(exposure_assignment_file) = args.exposure_assignment_file {
        let exposure =
            read_lines_of_words_delim(&exposure_assignment_file, &['\t', ',', ' '], -1)?.lines;

        let indv_to_exposure = exposure
            .into_iter()
            .filter(|w| w.len() > 1)
            .map(|w| (w[0].clone(), w[1].clone()))
            .collect::<HashMap<_, _>>();

        let exposure_id: HashMap<_, usize> = indv_to_exposure
            .values()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(id, val)| (val, id))
            .collect();

        let n_exposure = exposure_id.len();
        let mut exposure_name = vec![String::from("").into_boxed_str(); n_exposure];
        for (x, &id) in exposure_id.iter() {
            if id < n_exposure {
                exposure_name[id] = x.clone();
            }
        }
        info!("{} exposure groups", n_exposure);

        Ok(InputData {
            sparse_data,
            cell_to_indv,
            cell_topic,
            sorted_topic_names,
            indv_to_exposure: Some(indv_to_exposure),
            exposure_id: Some(exposure_id),
        })
    } else {
        Ok(InputData {
            sparse_data,
            cell_to_indv,
            cell_topic,
            sorted_topic_names,
            indv_to_exposure: None,
            exposure_id: None,
        })
    }
}
