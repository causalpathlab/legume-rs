use crate::srt_common::*;
use clap::Parser;
use matrix_util::parquet::*;
use parquet::basic::Type as ParquetType;

#[derive(Parser, Debug, Clone)]
pub struct SrtGeneNetworkArgs {
    #[arg(
        short = 'g',
        long,
        required = true,
        help = "Gene graph parquet file (gene_graph.parquet)"
    )]
    gene_graph: Box<str>,

    #[arg(
        short = 'd',
        long,
        required = true,
        help = "Dictionary parquet file (dictionary.parquet)"
    )]
    dictionary: Box<str>,

    #[arg(
        short = 'k',
        long,
        help = "Number of edge clusters for K-means (default = n_topics)"
    )]
    n_clusters: Option<usize>,

    #[arg(
        long,
        default_value_t = 100,
        help = "Maximum K-means iterations for edge clustering"
    )]
    maxiter_clustering: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of Laplacian eigenvectors for spectral embedding"
    )]
    num_eigen: usize,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix",
        long_help = "Output file prefix.\n\
                       Generates: {out}.gene_coords.parquet, {out}.gene_pair_clusters.parquet"
    )]
    out: Box<str>,
}

pub fn fit_srt_gene_network(args: &SrtGeneNetworkArgs) -> anyhow::Result<()> {
    let _ = env_logger::try_init();

    // Step A: Read gene graph from parquet
    info!("Reading gene graph from {}", args.gene_graph);
    let graph_rows = names_from_parquet(&args.gene_graph, &["gene1".into(), "gene2".into()])?;

    let MatWithNames {
        rows: _,
        cols: _,
        mat: dist_col,
    } = Mat::from_parquet_with_names(&args.gene_graph, None, Some(&["distance".into()]))?;

    // Build gene index
    let mut gene_names: Vec<Box<str>> = graph_rows
        .iter()
        .flat_map(|pair| pair.iter().cloned())
        .collect();
    gene_names.sort();
    gene_names.dedup();
    let n_genes = gene_names.len();

    let gene_index: HashMap<Box<str>, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    info!("{} genes, {} edges", n_genes, graph_rows.len());

    // Build dense distance matrix (initialize to infinity)
    let mut dist_nn = Mat::from_element(n_genes, n_genes, f32::INFINITY);
    for i in 0..n_genes {
        dist_nn[(i, i)] = 0.0;
    }
    for (idx, pair) in graph_rows.iter().enumerate() {
        let i = gene_index[&pair[0]];
        let j = gene_index[&pair[1]];
        let d = dist_col[(idx, 0)];
        dist_nn[(i, j)] = d;
        dist_nn[(j, i)] = d;
    }

    // Step B: Compute 2D gene layout via fuzzy kernel + spectral embedding
    info!("Computing fuzzy kernel weights");
    let similarity = fuzzy_kernel_dense(&dist_nn);

    // Regularize: add small self-loops
    let eps = 1e-4;
    let mut sim_reg = similarity;
    for i in 0..n_genes {
        sim_reg[(i, i)] += eps;
    }

    info!("Spectral embedding with {} eigenvectors", args.num_eigen);
    let emb = spectral_embed(&sim_reg, args.num_eigen)?;
    let coords = reduce_to_2d(&emb);

    // Write gene coordinates
    let coord_file = args.out.to_string() + ".gene_coords.parquet";
    let coord_cols: Vec<Box<str>> = vec!["x".into(), "y".into()];
    coords.to_parquet_with_names(
        &coord_file,
        (Some(&gene_names), Some("gene")),
        Some(&coord_cols),
    )?;
    info!("Wrote {}", coord_file);

    // Step C: Read dictionary and cluster edges
    info!("Reading dictionary from {}", args.dictionary);
    let MatWithNames {
        rows: dict_rows,
        cols: _dict_cols,
        mat: dict_mk,
    } = Mat::from_parquet(args.dictionary.as_ref())?;

    let n_topics = dict_mk.ncols();
    let num_clusters = args.n_clusters.unwrap_or(n_topics);

    info!(
        "K-means clustering: {} edges into {} clusters",
        dict_mk.nrows(),
        num_clusters
    );
    let dict_km = dict_mk.transpose();
    let membership = dict_km.kmeans_columns(KmeansArgs {
        num_clusters,
        max_iter: args.maxiter_clustering,
    });

    // Step D: Write gene pair clusters with topic loadings
    let cluster_file = args.out.to_string() + ".gene_pair_clusters.parquet";

    // Parse "GENE1:GENE2" row names into gene1, gene2
    let gene1_names: Vec<Box<str>> = dict_rows
        .iter()
        .map(|name| {
            name.split(':')
                .next()
                .unwrap_or(name.as_ref())
                .to_string()
                .into_boxed_str()
        })
        .collect();

    let gene2_names: Vec<Box<str>> = dict_rows
        .iter()
        .map(|name| {
            name.split(':')
                .nth(1)
                .unwrap_or("")
                .to_string()
                .into_boxed_str()
        })
        .collect();

    let membership_i32: Vec<i32> = membership.iter().map(|&x| x as i32).collect();
    let n_edges = dict_rows.len();

    // Build schema: row_index, gene1, gene2, cluster, 0, 1, ...
    let mut col_names: Vec<Box<str>> = vec!["gene1".into(), "gene2".into(), "cluster".into()];
    for k in 0..n_topics {
        col_names.push(k.to_string().into_boxed_str());
    }

    let mut col_types: Vec<ParquetType> = vec![
        ParquetType::BYTE_ARRAY,
        ParquetType::BYTE_ARRAY,
        ParquetType::INT32,
    ];
    for _ in 0..n_topics {
        col_types.push(ParquetType::FLOAT);
    }

    let pw = ParquetWriter::new(
        &cluster_file,
        (n_edges, col_names.len()),
        (None, Some(&col_names)),
        Some(&col_types),
        None,
    )?;
    let row_names_ba = pw.row_names_vec();
    let mut writer = pw.get_writer()?;
    let mut rg = writer.next_row_group()?;

    parquet_add_bytearray(&mut rg, row_names_ba)?;
    parquet_add_string_column(&mut rg, &gene1_names)?;
    parquet_add_string_column(&mut rg, &gene2_names)?;
    parquet_add_numeric_column(&mut rg, &membership_i32)?;

    for k in 0..n_topics {
        let col_data: Vec<f32> = (0..n_edges).map(|i| dict_mk[(i, k)]).collect();
        parquet_add_numeric_column(&mut rg, &col_data)?;
    }

    rg.close()?;
    writer.close()?;
    info!("Wrote {}", cluster_file);

    info!("Done");
    Ok(())
}

/// Compute fuzzy kernel (UMAP-style) similarity from a dense distance matrix.
///
/// Per-gene rho (nearest neighbor distance) and sigma (binary search for target
/// perplexity), then symmetrize with fuzzy union.
fn fuzzy_kernel_dense(dist: &Mat) -> Mat {
    let n = dist.nrows();

    // Step 1-2: compute rho and sigma per node
    let mut rho = vec![0.0f32; n];
    let mut sigma = vec![1.0f32; n];

    for i in 0..n {
        // Collect finite non-self distances
        let dists: Vec<f32> = (0..n)
            .filter(|&j| j != i && dist[(i, j)].is_finite())
            .map(|j| dist[(i, j)])
            .collect();

        if dists.is_empty() {
            continue;
        }

        rho[i] = dists.iter().cloned().fold(f32::INFINITY, f32::min);
        let target = (dists.len() as f32).log2();
        sigma[i] = smooth_knn_sigma(&dists, rho[i], target);
    }

    // Step 3-4: directed weights + symmetrize
    let mut sim = Mat::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let w_ij = directed_umap_weight(dist[(i, j)], rho[i], sigma[i]);
            let w_ji = directed_umap_weight(dist[(j, i)], rho[j], sigma[j]);
            let w_sym = w_ij + w_ji - w_ij * w_ji;
            sim[(i, j)] = w_sym;
            sim[(j, i)] = w_sym;
        }
    }
    sim
}

/// Binary search for per-point sigma (UMAP's smooth_knn_dist).
fn smooth_knn_sigma(dists: &[f32], rho: f32, target: f32) -> f32 {
    const TOLERANCE: f32 = 1e-5;
    const MAX_ITER: usize = 64;

    let mean_dist: f32 = dists.iter().sum::<f32>() / dists.len().max(1) as f32;
    let min_sigma = 1e-3 * mean_dist;

    let mut lo = 0.0f32;
    let mut hi = f32::INFINITY;
    let mut mid = 1.0f32;

    for _ in 0..MAX_ITER {
        let mut psum = 0.0f32;
        for &d in dists {
            let gap = d - rho;
            if gap > 0.0 {
                psum += (-gap / mid).exp();
            } else {
                psum += 1.0;
            }
        }

        if (psum - target).abs() < TOLERANCE {
            break;
        }

        if psum > target {
            hi = mid;
            mid = (lo + hi) / 2.0;
        } else {
            lo = mid;
            if hi.is_infinite() {
                mid *= 2.0;
            } else {
                mid = (lo + hi) / 2.0;
            }
        }
    }

    mid.max(min_sigma)
}

/// Compute a single directed UMAP membership weight.
fn directed_umap_weight(d: f32, rho: f32, sigma: f32) -> f32 {
    if d.is_infinite() || sigma <= 0.0 {
        return 0.0;
    }
    let gap = d - rho;
    if gap <= 0.0 {
        1.0
    } else {
        (-gap / sigma).exp()
    }
}

/// Spectral embedding: compute k-dimensional embedding from similarity matrix.
/// Uses symmetric normalized Laplacian: L_sym = I - D^{-1/2} S D^{-1/2}
fn spectral_embed(similarity: &Mat, num_eigen: usize) -> anyhow::Result<Mat> {
    let n = similarity.nrows();
    let k = num_eigen.clamp(2, n - 1);
    anyhow::ensure!(n > k, "Need more than {} genes, got {}", k, n);

    let degree: DVec = DVec::from_iterator(n, similarity.row_iter().map(|r| r.sum()));
    let d_inv_sqrt = Mat::from_diagonal(&degree.map(|d| 1.0 / d.sqrt()));
    let laplacian = Mat::identity(n, n) - &d_inv_sqrt * similarity * &d_inv_sqrt;

    let eig = laplacian.symmetric_eigen();
    let idx = argsort(&eig.eigenvalues, true);
    let mut emb = Mat::zeros(n, k);
    for (j, &i) in idx[1..=k].iter().enumerate() {
        let w = 1.0 / eig.eigenvalues[i].max(1e-10);
        emb.column_mut(j)
            .copy_from(&(w * eig.eigenvectors.column(i)));
    }

    Ok(emb)
}

/// Reduce k-dimensional embedding to 2D via PCA
fn reduce_to_2d(emb: &Mat) -> Mat {
    let n = emb.nrows();
    let k = emb.ncols();

    if k == 2 {
        return emb.clone();
    }

    let mut centered = emb.clone();
    centered.centre_columns_inplace();
    let pca = (centered.transpose() * &centered).symmetric_eigen();
    let pca_idx = argsort(&pca.eigenvalues, false);
    let mut coords = Mat::zeros(n, 2);
    coords
        .column_mut(0)
        .copy_from(&(&centered * pca.eigenvectors.column(pca_idx[0])));
    coords
        .column_mut(1)
        .copy_from(&(&centered * pca.eigenvectors.column(pca_idx[1])));
    coords
}

/// Sort eigenvalue indices
fn argsort(vals: &DVec, asc: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| {
        let c = vals[a].partial_cmp(&vals[b]).unwrap();
        if asc {
            c
        } else {
            c.reverse()
        }
    });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::basic::Type as ParquetType;

    /// Create a synthetic gene_graph.parquet with 5 genes and 7 edges
    fn write_test_gene_graph(path: &str) -> anyhow::Result<()> {
        let genes = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"];
        let edges: Vec<(usize, usize, f32)> = vec![
            (0, 1, 1.0),
            (0, 2, 2.0),
            (1, 2, 1.5),
            (1, 3, 2.5),
            (2, 3, 1.2),
            (2, 4, 1.8),
            (3, 4, 0.9),
        ];

        let n_edges = edges.len();
        let col_names: Vec<Box<str>> = vec!["gene1".into(), "gene2".into(), "distance".into()];
        let col_types = vec![
            ParquetType::BYTE_ARRAY,
            ParquetType::BYTE_ARRAY,
            ParquetType::FLOAT,
        ];

        let pw = ParquetWriter::new(
            path,
            (n_edges, col_names.len()),
            (None, Some(&col_names)),
            Some(&col_types),
            None,
        )?;
        let row_names = pw.row_names_vec();
        let mut writer = pw.get_writer()?;
        let mut rg = writer.next_row_group()?;

        parquet_add_bytearray(&mut rg, row_names)?;

        let g1: Vec<Box<str>> = edges.iter().map(|&(i, _, _)| genes[i].into()).collect();
        parquet_add_string_column(&mut rg, &g1)?;

        let g2: Vec<Box<str>> = edges.iter().map(|&(_, j, _)| genes[j].into()).collect();
        parquet_add_string_column(&mut rg, &g2)?;

        let dists: Vec<f32> = edges.iter().map(|&(_, _, d)| d).collect();
        parquet_add_numeric_column(&mut rg, &dists)?;

        rg.close()?;
        writer.close()?;
        Ok(())
    }

    /// Create a synthetic dictionary.parquet with 7 edges x 3 topics
    fn write_test_dictionary(path: &str) -> anyhow::Result<()> {
        let genes = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"];
        let edges: Vec<(usize, usize)> =
            vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)];

        let row_names: Vec<Box<str>> = edges
            .iter()
            .map(|&(i, j)| format!("{}:{}", genes[i], genes[j]).into_boxed_str())
            .collect();

        let n_edges = edges.len();
        let n_topics = 3;

        // Deterministic values
        let mut dict = Mat::zeros(n_edges, n_topics);
        for i in 0..n_edges {
            for k in 0..n_topics {
                dict[(i, k)] = ((i * 7 + k * 3) % 11) as f32 / 10.0;
            }
        }

        dict.to_parquet_with_names(path, (Some(&row_names), Some("gene_pair")), None)?;
        Ok(())
    }

    #[test]
    fn test_fit_srt_gene_network() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let graph_path = tmp.path().join("test.gene_graph.parquet");
        let dict_path = tmp.path().join("test.dictionary.parquet");
        let out_prefix = tmp.path().join("test_out");

        write_test_gene_graph(graph_path.to_str().unwrap())?;
        write_test_dictionary(dict_path.to_str().unwrap())?;

        let args = SrtGeneNetworkArgs {
            gene_graph: graph_path.to_str().unwrap().into(),
            dictionary: dict_path.to_str().unwrap().into(),
            n_clusters: Some(2),
            maxiter_clustering: 50,
            num_eigen: 3,
            out: out_prefix.to_str().unwrap().into(),
        };

        fit_srt_gene_network(&args)?;

        // Check gene_coords output
        let coord_file = out_prefix.to_str().unwrap().to_string() + ".gene_coords.parquet";
        let MatWithNames { rows, cols, mat } = Mat::from_parquet(&coord_file)?;
        assert_eq!(rows.len(), 5, "expected 5 genes");
        assert_eq!(mat.ncols(), 2, "expected 2 coordinate columns");
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].as_ref(), "x");
        assert_eq!(cols[1].as_ref(), "y");

        // All coordinates should be finite
        for i in 0..mat.nrows() {
            for j in 0..mat.ncols() {
                assert!(
                    mat[(i, j)].is_finite(),
                    "non-finite coord at ({}, {})",
                    i,
                    j
                );
            }
        }

        // Check gene_pair_clusters output
        let cluster_file = out_prefix.to_str().unwrap().to_string() + ".gene_pair_clusters.parquet";
        let cluster_names = names_from_parquet(&cluster_file, &["gene1".into(), "gene2".into()])?;
        assert_eq!(cluster_names.len(), 7, "expected 7 edges");

        // Read the numeric part (cluster + topics)
        let MatWithNames {
            rows: _,
            cols: cluster_cols,
            mat: cluster_mat,
        } = Mat::from_parquet_with_names(
            &cluster_file,
            None,
            Some(&["cluster".into(), "0".into(), "1".into(), "2".into()]),
        )?;
        assert_eq!(cluster_cols.len(), 4, "expected cluster + 3 topics");
        assert_eq!(cluster_mat.nrows(), 7);

        // Cluster values should be 0 or 1 (since n_clusters=2)
        for i in 0..7 {
            let c = cluster_mat[(i, 0)] as i32;
            assert!(c == 0 || c == 1, "cluster {} not in {{0, 1}}", c);
        }

        Ok(())
    }

    #[test]
    fn test_smooth_knn_sigma() {
        // With uniform distances around rho, sigma should converge
        let dists = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rho = 1.0;
        let target = (dists.len() as f32).log2();
        let sigma = smooth_knn_sigma(&dists, rho, target);
        assert!(sigma > 0.0, "sigma must be positive");
        // Verify convergence: sum of weights should be close to target
        let psum: f32 = dists
            .iter()
            .map(|&d| {
                let gap = d - rho;
                if gap > 0.0 {
                    (-gap / sigma).exp()
                } else {
                    1.0
                }
            })
            .sum();
        assert!(
            (psum - target).abs() < 0.1,
            "psum={} should be close to target={}",
            psum,
            target
        );
    }

    #[test]
    fn test_smooth_knn_sigma_all_equal() {
        // All distances equal to rho → all weights = 1.0
        let dists = vec![2.0, 2.0, 2.0];
        let rho = 2.0;
        let target = 2.0;
        let sigma = smooth_knn_sigma(&dists, rho, target);
        assert!(sigma > 0.0);
    }

    #[test]
    fn test_directed_umap_weight_basic() {
        // d <= rho → weight = 1.0
        assert_eq!(directed_umap_weight(1.0, 2.0, 1.0), 1.0);
        assert_eq!(directed_umap_weight(2.0, 2.0, 1.0), 1.0);

        // d > rho → weight in (0, 1)
        let w = directed_umap_weight(3.0, 1.0, 1.0);
        assert!(w > 0.0 && w < 1.0, "weight={}", w);

        // infinite distance → weight = 0
        assert_eq!(directed_umap_weight(f32::INFINITY, 1.0, 1.0), 0.0);

        // sigma <= 0 → weight = 0
        assert_eq!(directed_umap_weight(3.0, 1.0, 0.0), 0.0);
        assert_eq!(directed_umap_weight(3.0, 1.0, -1.0), 0.0);
    }

    #[test]
    fn test_argsort_ascending() {
        let vals = DVec::from_vec(vec![3.0, 1.0, 4.0, 1.5, 2.0]);
        let idx = argsort(&vals, true);
        assert_eq!(idx, vec![1, 3, 4, 0, 2]);
    }

    #[test]
    fn test_argsort_descending() {
        let vals = DVec::from_vec(vec![3.0, 1.0, 4.0, 1.5, 2.0]);
        let idx = argsort(&vals, false);
        assert_eq!(idx, vec![2, 0, 4, 3, 1]);
    }

    #[test]
    fn test_spectral_embed_dimensions() -> anyhow::Result<()> {
        // Build a 6-node similarity matrix (block diagonal → 2 clusters)
        let n = 6;
        let mut sim = Mat::from_element(n, n, 0.01);
        for i in 0..3 {
            for j in 0..3 {
                sim[(i, j)] = 1.0;
            }
        }
        for i in 3..6 {
            for j in 3..6 {
                sim[(i, j)] = 1.0;
            }
        }
        // Add small self-loops
        for i in 0..n {
            sim[(i, i)] += 1e-4;
        }

        let k = 3;
        let emb = spectral_embed(&sim, k)?;
        assert_eq!(emb.nrows(), n);
        assert_eq!(emb.ncols(), k);
        // All values should be finite
        for i in 0..emb.nrows() {
            for j in 0..emb.ncols() {
                assert!(emb[(i, j)].is_finite());
            }
        }
        Ok(())
    }
}
