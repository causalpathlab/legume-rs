use crate::common::*;
use clap::ValueEnum;
use matrix_util::common_io::*;
use matrix_util::dmatrix_util::*;
use matrix_util::parquet::*;
use matrix_util::traits::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum TopicValue {
    Prob,
    Logit,
}

pub struct InputDataArgs {
    pub data_files: Vec<Box<str>>,
    pub indv_files: Option<Vec<Box<str>>>,
    pub topic_assignment_files: Option<Vec<Box<str>>>,
    pub topic_proportion_files: Option<Vec<Box<str>>>,
    pub exposure_assignment_file: Option<Box<str>>,
    pub preload_data: bool,
    pub topic_value: TopicValue,
}

pub struct InputData {
    pub sparse_data: SparseIoVec,
    pub cell_to_indv: Vec<Box<str>>,
    pub cell_topic: Mat,
    pub sorted_topic_names: Vec<Box<str>>,
    pub indv_to_exposure: Option<HashMap<Box<str>, Box<str>>>,
    pub exposure_id: Option<HashMap<Box<str>, usize>>,
}

/// Collects topic names from assignment files
fn collect_topic_names_from_assignment(
    file_path: &str,
    topic_names: &mut HashSet<Box<str>>,
) -> anyhow::Result<()> {
    let lines = read_lines_of_words_delim(file_path, &['\t', ',', ' '], -1)?.lines;

    // Single pass: collect topics from both formats
    for line in lines {
        match line.len() {
            1 => {
                topic_names.insert(line[0].clone());
            }
            n if n > 1 => {
                topic_names.insert(line[1].clone());
            }
            _ => {}
        }
    }

    Ok(())
}

/// Collects topic names from proportion files (parquet)
fn collect_topic_names_from_proportion(
    file_path: &str,
    topic_names: &mut HashSet<Box<str>>,
) -> anyhow::Result<()> {
    let ext = file_ext(file_path)?;
    if ext.as_ref() == "parquet" {
        let names = peek_parquet_field_names(file_path)?;
        // Skip the first column (row names)
        if names.len() > 1 {
            topic_names.extend(names.into_iter().skip(1));
        }
    }
    Ok(())
}

/// Collects and sorts all unique topic names from assignment and proportion files
fn collect_all_topic_names(
    topic_assignment_files: &[Option<Box<str>>],
    topic_proportion_files: &[Option<Box<str>>],
) -> anyhow::Result<(Vec<Box<str>>, HashMap<Box<str>, usize>)> {
    info!("Looking into the topic files...");

    let mut topic_names = HashSet::<Box<str>>::new();

    for (a_file, p_file) in topic_assignment_files
        .iter()
        .zip(topic_proportion_files.iter())
    {
        if let Some(a_file) = a_file {
            collect_topic_names_from_assignment(a_file, &mut topic_names)?;
        }
        if let Some(p_file) = p_file {
            collect_topic_names_from_proportion(p_file, &mut topic_names)?;
        }
    }

    let mut sorted_topic_names: Vec<Box<str>> = topic_names.into_iter().collect();
    sorted_topic_names.sort();

    let topic_name_to_index: HashMap<Box<str>, usize> = sorted_topic_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();

    info!("Found {} topics", sorted_topic_names.len());

    Ok((sorted_topic_names, topic_name_to_index))
}

/// Reads topic assignments from a file and populates the topic matrix
fn read_topic_assignments(
    file_path: &str,
    topic_matrix: &mut Mat,
    topic_names: &HashMap<Box<str>, usize>,
    cells_to_rows: &HashMap<Box<str>, usize>,
) -> anyhow::Result<()> {
    info!("importing topic information from {}", file_path);

    let ReadLinesOut { lines, header: _ } =
        read_lines_of_words_delim(file_path, &['\t', ',', ' '], -1)?;

    for (i, words) in lines.iter().enumerate() {
        if words.len() > 1 {
            // Format: cell_id topic_name
            if let Some(&r) = cells_to_rows.get(&words[0]) {
                if let Some(&c) = topic_names.get(&words[1]) {
                    topic_matrix[(r, c)] = 1.0;
                }
            }
        } else if words.len() == 1 {
            // Format: topic_name (line number is the cell index)
            if let Some(&c) = topic_names.get(&words[0]) {
                topic_matrix[(i, c)] = 1.0;
            }
        }
    }

    Ok(())
}

/// Reads topic proportions from a parquet file and populates the topic matrix
fn read_topic_proportions(
    file_path: &str,
    topic_matrix: &mut Mat,
    topic_names: &HashMap<Box<str>, usize>,
    cells_to_rows: &HashMap<Box<str>, usize>,
    topic_value: &TopicValue,
) -> anyhow::Result<()> {
    let ext = file_ext(file_path)?;
    if ext.as_ref() != "parquet" {
        return Ok(());
    }

    info!("importing topic information from {}", file_path);

    let MatWithNames { mat, rows, cols } = Mat::from_parquet(file_path)?;

    for j in 0..mat.ncols() {
        if let Some(&c) = topic_names.get(&cols[j]) {
            for i in 0..mat.nrows() {
                if let Some(&r) = cells_to_rows.get(&rows[i]) {
                    topic_matrix[(r, c)] = match topic_value {
                        TopicValue::Logit => mat[(i, j)].exp(),
                        TopicValue::Prob => mat[(i, j)],
                    };
                }
            }
        }
    }

    Ok(())
}

/// Processes exposure assignment file and returns mapping and ID structures
fn process_exposure_assignments(
    file_path: &str,
) -> anyhow::Result<(HashMap<Box<str>, Box<str>>, HashMap<Box<str>, usize>)> {
    let exposure = read_lines_of_words_delim(file_path, &['\t', ',', ' '], -1)?.lines;

    let indv_to_exposure: HashMap<Box<str>, Box<str>> = exposure
        .into_iter()
        .filter(|w| w.len() > 1)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect();

    let exposure_id: HashMap<Box<str>, usize> = indv_to_exposure
        .values()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(id, val)| (val, id))
        .collect();

    info!("{} exposure groups", exposure_id.len());

    Ok((indv_to_exposure, exposure_id))
}

/// Reads individual assignments from a file or creates default assignments
fn read_individual_assignments(
    indv_file: Option<&Box<str>>,
    column_names: &[Box<str>],
    ndata: usize,
) -> anyhow::Result<Vec<Box<str>>> {
    const MISSING: &str = "NA";
    let missing = MISSING.to_string().into_boxed_str();

    let this_indv = match indv_file {
        Some(input_file) => read_lines_of_words_delim(input_file, &['\t', ',', ' '], -1)?.lines,
        None => vec![vec![missing.clone()]; ndata],
    };

    if this_indv.is_empty() {
        return Err(anyhow::anyhow!("Individual file is empty"));
    }

    if this_indv.len() != ndata {
        return Err(anyhow::anyhow!(
            "Individual file has {} entries but data has {} columns",
            this_indv.len(),
            ndata
        ));
    }

    let result = if this_indv[0].len() == 1 {
        // Simple format: one individual per line
        this_indv.into_iter().map(|w| w[0].clone()).collect()
    } else {
        // Cell-individual pairs format
        let cell_to_indv_map: HashMap<_, _> = this_indv
            .into_iter()
            .filter_map(|w| (w.len() > 1).then(|| (w[0].clone(), w[1].clone())))
            .collect();

        column_names
            .iter()
            .map(|c| {
                cell_to_indv_map
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| missing.clone())
            })
            .collect()
    };

    Ok(result)
}

/// Determines the appropriate sparse matrix backend from file extension
fn get_backend_from_extension(file_path: &str) -> anyhow::Result<SparseIoBackend> {
    match file_ext(file_path)?.to_string().as_str() {
        "h5" => Ok(SparseIoBackend::HDF5),
        "zarr" => Ok(SparseIoBackend::Zarr),
        ext => Err(anyhow::anyhow!(
            "Unsupported file extension '{}' for data file: {}. Expected 'h5' or 'zarr'",
            ext,
            file_path
        )),
    }
}

pub fn read_input_data(args: InputDataArgs) -> anyhow::Result<InputData> {
    // Validate input arguments
    if let Some(indv_files) = args.indv_files.as_ref() {
        if indv_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!(
                "Number of individual files ({}) does not match number of data files ({})",
                indv_files.len(),
                args.data_files.len()
            ));
        }
    }
    if let Some(topic_assignment_files) = args.topic_assignment_files.as_ref() {
        if topic_assignment_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!(
                "Number of topic assignment files ({}) does not match number of data files ({})",
                topic_assignment_files.len(),
                args.data_files.len()
            ));
        }
    }
    if let Some(topic_proportion_files) = args.topic_proportion_files.as_ref() {
        if topic_proportion_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!(
                "Number of topic proportion files ({}) does not match number of data files ({})",
                topic_proportion_files.len(),
                args.data_files.len()
            ));
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

    let (sorted_topic_names, topic_names) =
        collect_all_topic_names(&topic_assignment_files, &topic_proportion_files)?;

    ////////////////////////////////////
    // Read matched data, topic, indv //
    ////////////////////////////////////

    let mut topic_vec = vec![];
    let mut sparse_data = SparseIoVec::new();
    let mut cell_to_indv = vec![];

    for (f, this_data_file) in data_files.iter().enumerate() {
        let this_indv_file = indv_files.as_ref().map(|files| files[f].clone());

        info!("Importing: {}, {:?}", this_data_file, this_indv_file);

        let backend = get_backend_from_extension(this_data_file)?;
        let mut this_data = open_sparse_matrix(this_data_file, &backend)?;

        if args.preload_data {
            this_data.preload_columns()?;
        }

        let ndata = this_data.num_columns().unwrap_or(0);
        let column_names = this_data.column_names()?;

        let this_indv = read_individual_assignments(this_indv_file.as_ref(), &column_names, ndata)?;

        let topic_a_file = &topic_assignment_files[f];
        let topic_p_file = &topic_proportion_files[f];

        let mut this_topic = if topic_names.is_empty() {
            Mat::from_element(ndata, 1, 1.0)
        } else {
            Mat::zeros(ndata, topic_names.len())
        };

        let cells_to_rows: HashMap<_, _> = column_names
            .into_iter()
            .enumerate()
            .map(|(i, x)| (x, i))
            .collect();

        if let Some(a_file) = topic_a_file {
            read_topic_assignments(a_file, &mut this_topic, &topic_names, &cells_to_rows)?;
        }

        if let Some(p_file) = topic_p_file {
            read_topic_proportions(
                p_file,
                &mut this_topic,
                &topic_names,
                &cells_to_rows,
                &args.topic_value,
            )?;
        }

        let data_name = basename(&this_data_file)?;
        sparse_data.push(Arc::from(this_data), Some(data_name))?;
        cell_to_indv.extend(this_indv);
        topic_vec.push(this_topic);
    }

    let cell_topic = concatenate_vertical(topic_vec.as_slice())?;

    // If no topic files were given, we created a single default column;
    // make sure there's a matching name so downstream parquet output works
    let sorted_topic_names = if sorted_topic_names.is_empty() && cell_topic.ncols() == 1 {
        vec!["0".to_string().into_boxed_str()]
    } else {
        sorted_topic_names
    };

    info!("Total {} data sets combined", sparse_data.len());

    sparse_data.assign_groups(&cell_to_indv, None);

    let (indv_to_exposure, exposure_id) =
        if let Some(exposure_assignment_file) = args.exposure_assignment_file {
            let (indv_map, exp_id) = process_exposure_assignments(&exposure_assignment_file)?;
            (Some(indv_map), Some(exp_id))
        } else {
            (None, None)
        };

    Ok(InputData {
        sparse_data,
        cell_to_indv,
        cell_topic,
        sorted_topic_names,
        indv_to_exposure,
        exposure_id,
    })
}
