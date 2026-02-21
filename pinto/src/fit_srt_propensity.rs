use crate::srt_common::*;
use crate::srt_input::read_expr_data;
use clap::Parser;
use dmatrix_gamma::GammaMatrix;
use matrix_param::dmatrix_gamma;
use matrix_param::io::ParamIo;
use matrix_param::traits::TwoStatParam;
use matrix_util::parquet::{
    parquet_add_bytearray, parquet_add_numeric_column, parquet_add_string_column, ParquetWriter,
};
use parquet::basic::Type as ParquetType;

#[derive(Parser, Debug, Clone)]
pub struct SrtPropensityArgs {
    #[arg(
        long,
        help = "Number of edge clusters for K-means",
        long_help = "Number of edge clusters for K-means.\n\
                       Defaults to the number of latent dimensions if not specified."
    )]
    n_edge_clusters: Option<usize>,

    #[arg(
        long,
        default_value_t = 100,
        help = "Maximum K-means iterations for edge clustering"
    )]
    maxiter_clustering: usize,

    #[arg(
        short = 'z',
        long,
        required = true,
        help = "Latent edge representation file (.latent.parquet)"
    )]
    latent_data_file: Box<str>,

    #[arg(
        short = 'e',
        long,
        required = true,
        help = "Coordinate pair file (.coord_pairs.parquet)",
        long_help = "Coordinate pair file (.coord_pairs.parquet from delta-svd).\n\
                       Must contain left_cell and right_cell columns."
    )]
    coord_pair_file: Box<str>,

    #[arg(
        short = 'd',
        long,
        value_delimiter(','),
        help = "Expression data files (.zarr or .h5)",
        long_help = "Expression data files (.zarr or .h5, comma separated).\n\
                       Optional; used for additional per-vertex expression statistics."
    )]
    expr_data_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value = "left_cell",
        help = "Column name for left cell index in coord_pair_file"
    )]
    left_name: Box<str>,

    #[arg(
        long,
        default_value = "right_cell",
        help = "Column name for right cell index in coord_pair_file"
    )]
    right_name: Box<str>,

    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        help = "Coordinate column names in coord_pair_file",
        long_help = "Coordinate column names in coord_pair_file (comma separated).\n\
                       Looked up as left_{name} and right_{name}."
    )]
    coord_column_names: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing"
    )]
    block_size: usize,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Output file prefix.\n\
                       Generates: {out}.propensity.parquet, {out}.edge_cluster.parquet"
    )]
    out: Box<str>,
}

pub fn fit_srt_propensity(args: &SrtPropensityArgs) -> anyhow::Result<()> {
    let MatWithNames {
        rows,
        cols: _,
        mat: proj_mk,
    } = Mat::from_parquet(args.latent_data_file.as_ref())?;

    let pair_names = names_from_parquet(
        &args.coord_pair_file,
        &[args.left_name.clone(), args.right_name.clone()],
    )?;

    if pair_names.len() != rows.len() {
        anyhow::bail!(
            "pair names length {} != latent matrix rows {}",
            pair_names.len(),
            rows.len()
        );
    }

    let proj_km = proj_mk.transpose();

    info!("clustering edges");
    let num_clusters = args.n_edge_clusters.unwrap_or(proj_km.nrows());

    let edge_membership = proj_km.kmeans_columns(KmeansArgs {
        num_clusters,
        max_iter: args.maxiter_clustering,
    });

    info!("calibrating propensity");

    let mut vertices = pair_names.par_iter().flatten().cloned().collect::<Vec<_>>();
    vertices.par_sort();
    vertices.dedup();

    let vertex_index: HashMap<Box<str>, usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect();

    let nvertices = vertices.len();
    info!("{} vertices", nvertices);

    let mut prop_kn = Mat::zeros(num_clusters, nvertices);
    let arc_count_kn = Arc::new(Mutex::new(&mut prop_kn));

    pair_names
        .par_iter()
        .zip(edge_membership.par_iter())
        .for_each(|(vertices, &k)| {
            let indices = vertices
                .iter()
                .filter_map(|x| vertex_index.get(x).copied())
                .collect::<Vec<_>>();

            let mut count_kn = arc_count_kn.lock().expect("lock count kn");

            for v in indices {
                count_kn.column_mut(v)[k] += 1.0;
            }
        });

    prop_kn.sum_to_one_columns_inplace();

    // Dominant cluster per vertex (argmax of propensity)
    let cluster_col: Vec<f32> = (0..nvertices)
        .map(|j| prop_kn.column(j).iamax() as f32)
        .collect();
    let cluster_mat = Mat::from_column_slice(nvertices, 1, &cluster_col);

    // Build propensity column names: propensity_0 .. propensity_{K-1}, cluster
    let mut col_names: Vec<Box<str>> = (0..num_clusters)
        .map(|k| format!("propensity_{}", k).into_boxed_str())
        .collect();
    col_names.push("cluster".into());

    // Propensity output (optionally with coordinates)
    let prop_nk = prop_kn.transpose();
    if let Some(coord_column_names) = &args.coord_column_names {
        info!("Extracting vertex coordinates from coord_pair_file");

        let left_coord_names: Vec<Box<str>> = coord_column_names
            .iter()
            .map(|name| format!("left_{}", name).into_boxed_str())
            .collect();

        let MatWithNames {
            rows: _,
            cols: _,
            mat: left_coords,
        } = Mat::from_parquet_with_indices_names(
            &args.coord_pair_file,
            Some(0),
            Some(&[]),
            Some(&left_coord_names),
        )?;

        let n_coords = coord_column_names.len();
        let mut vertex_coords = Mat::zeros(nvertices, n_coords);
        for (pair_idx, pair) in pair_names.iter().enumerate() {
            if let Some(&v_idx) = vertex_index.get(&pair[0]) {
                vertex_coords
                    .row_mut(v_idx)
                    .copy_from(&left_coords.row(pair_idx));
            }
        }

        col_names.extend(coord_column_names.iter().cloned());
        let combined = concatenate_horizontal(&[prop_nk, cluster_mat, vertex_coords])?;

        combined.to_parquet_with_names(
            &(args.out.to_string() + ".propensity.parquet"),
            (Some(&vertices), Some("cell")),
            Some(&col_names),
        )?;
    } else {
        let combined = concatenate_horizontal(&[prop_nk, cluster_mat])?;

        combined.to_parquet_with_names(
            &(args.out.to_string() + ".propensity.parquet"),
            (Some(&vertices), Some("cell")),
            Some(&col_names),
        )?;
    }

    // Edge cluster assignments
    {
        info!("Writing edge cluster assignments");
        let n_edges = pair_names.len();
        let left_cells: Vec<Box<str>> = pair_names.iter().map(|p| p[0].clone()).collect();
        let right_cells: Vec<Box<str>> = pair_names.iter().map(|p| p[1].clone()).collect();
        let cluster_f32: Vec<f32> = edge_membership.iter().map(|&k| k as f32).collect();

        let ec_col_names: Vec<Box<str>> = vec!["right_cell".into(), "cluster".into()];
        let ec_col_types = vec![ParquetType::BYTE_ARRAY, ParquetType::FLOAT];

        let writer = ParquetWriter::new(
            &(args.out.to_string() + ".edge_cluster.parquet"),
            (n_edges, 2),
            (Some(&left_cells), Some(&ec_col_names)),
            Some(&ec_col_types),
            Some("left_cell"),
        )?;

        let row_names = writer.row_names_vec();
        let mut writer = writer.get_writer()?;
        let mut row_group = writer.next_row_group()?;

        parquet_add_bytearray(&mut row_group, row_names)?;
        parquet_add_string_column(&mut row_group, &right_cells)?;
        parquet_add_numeric_column(&mut row_group, &cluster_f32)?;

        row_group.close()?;
        writer.close()?;
    }

    if let Some(data_files) = args.expr_data_files.as_ref() {
        info!("Estimate cluster-specific gene expressions");
        let data_vec = read_expr_data(data_files)?;
        let genes = data_vec.row_names()?;
        let data_vertices = data_vec.column_names()?;

        let jobs = matrix_util::utils::generate_minibatch_intervals(
            data_vec.num_columns(),
            args.block_size,
        );

        let pb = new_progress_bar(
            jobs.len() as u64,
            "Gene modules {bar:40} {pos}/{len} blocks ({eta})",
        );
        let partial_stats = jobs
            .par_iter()
            .progress_with(pb.clone())
            .map(|&(lb, ub)| -> anyhow::Result<(Mat, DVec)> {
                let x_dn = data_vec.read_columns_csc(lb..ub)?;
                let mut p_kn = Mat::zeros(prop_kn.nrows(), x_dn.ncols());

                for (i, v) in data_vertices[lb..ub].iter().enumerate() {
                    if let Some(&j) = vertex_index.get(v) {
                        p_kn.column_mut(i).copy_from(&prop_kn.column(j));
                    }
                }

                let n_k = p_kn.column_sum();
                let sum_dk = x_dn * p_kn.transpose();

                Ok((sum_dk, n_k))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        pb.finish_and_clear();

        let mut sum_dk = Mat::zeros(genes.len(), prop_kn.nrows());
        let mut n_1k = Mat::zeros(1, prop_kn.nrows());

        for (s_dk, n_k1) in partial_stats {
            sum_dk += s_dk;
            n_1k += n_k1.transpose();
        }

        let mut gamma_param = GammaMatrix::new((sum_dk.nrows(), sum_dk.ncols()), 1.0, 1.0);

        let denom_dk = DVec::from_element(sum_dk.nrows(), 1.0) * n_1k;

        gamma_param.update_stat(&sum_dk, &denom_dk);
        gamma_param.calibrate();

        gamma_param.to_parquet_with_names(
            &(args.out.to_string() + ".genes.parquet"),
            (Some(&genes), Some("gene")),
            None,
        )?;
    }

    info!("Done");
    Ok(())
}
