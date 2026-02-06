use crate::srt_common::*;
use crate::srt_input::read_expr_data;
use clap::Parser;
use dmatrix_gamma::GammaMatrix;
use matrix_param::dmatrix_gamma;
use matrix_param::io::ParamIo;
use matrix_param::traits::TwoStatParam;
use rayon::prelude::*;

#[derive(Parser, Debug, Clone)]
///
/// Estimate vertex propensity
///
pub struct SrtPropensityArgs {
    /// number of (edge) clusters
    #[arg(short = 'k', long)]
    n_edge_clusters: Option<usize>,

    /// number of (edge) clusters
    #[arg(long, default_value_t = 100)]
    maxiter_clustering: usize,

    /// latent vectors for edges
    #[arg(short = 'z', long, required = true)]
    latent_data_file: Box<str>,

    /// coordinate pair file (edges)
    #[arg(short = 'e', long, required = true)]
    coord_pair_file: Box<str>,

    /// expression data
    #[arg(short = 'd', long, value_delimiter(','))]
    expr_data_files: Option<Vec<Box<str>>>,

    /// coordinate left name
    #[arg(long, default_value = "left_cell")]
    left_name: Box<str>,
    /// coordinate right name
    #[arg(long, default_value = "right_cell")]
    right_name: Box<str>,

    /// Block_size for parallel processing
    #[arg(long, default_value_t = 100)]
    block_size: usize,

    /// Output header
    #[arg(long, short, required = true)]
    out: Box<str>,
}

pub fn fit_srt_propensity(args: &SrtPropensityArgs) -> anyhow::Result<()> {
    let MatWithNames {
        rows,
        cols: _cols,
        mat: proj_mk,
    } = Mat::from_parquet(args.latent_data_file.as_ref())?;

    // just take left_cell and right_cell
    let pair_names = names_from_parquet(
        &args.coord_pair_file,
        &[args.left_name.clone(), args.right_name.clone()],
    )?;

    if pair_names.len() != rows.len() {
        return Err(anyhow::anyhow!(
            "Check the length of pair names, {} vs. the rows of the latent matrix, {}",
            pair_names.len(),
            rows.len()
        ));
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
        .zip(edge_membership)
        .for_each(|(vertices, k)| {
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

    prop_kn.transpose().to_parquet(
        Some(vertices.as_ref()),
        None,
        &(args.out.to_string() + ".propensity.parquet"),
    )?;

    if let Some(data_files) = args.expr_data_files.as_ref() {
        info!("Estimate cluster-specific gene expressions");
        let data_vec = read_expr_data(data_files)?;
        let genes = data_vec.row_names()?;
        let data_vertices = data_vec.column_names()?;

        let jobs = matrix_util::utils::generate_minibatch_intervals(
            data_vec.num_columns(),
            args.block_size,
        );

        let partial_stats = jobs
            .par_iter()
            .progress_count(jobs.len() as u64)
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

        gamma_param.to_parquet(
            Some(genes.as_ref()),
            None,
            &(args.out.to_string() + ".genes.parquet"),
        )?;
    }

    info!("Done");
    Ok(())
}
