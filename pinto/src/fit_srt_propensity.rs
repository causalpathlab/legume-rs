use crate::srt_common::*;
use crate::srt_input::read_expr_data;
use clap::Parser;
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
                .into_iter()
                .filter_map(|x| vertex_index.get(x).map(|ret| *ret.value()))
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

        // matched_indices
        let mut stat = Mat::zeros(num_clusters, genes.len());

        // let lb = 0;
        // let ub = 100;
        // let x_dn = data_vec.read_columns_csc(lb..ub)?;

	// this is from the prop_kn
	// data_vertices[lb..ub].iter().map(f)

	// otherwise zeros

        // let matched_indices: Vec<usize> = data_vertices
        //     .iter()
        //     .filter_map(|v| vertex_index.get(v).map(|x| x.value().clone()))
        //     .collect();


        // matched_indices.map(|j| {
        // });
    }

    info!("Done");
    Ok(())
}
