use crate::test_support::make_test_graph;
use crate::util::cell_pairs::*;
use crate::util::common::*;

#[test]
fn test_connected_components_single() {
    let graph = make_test_graph(5, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    let (labels, n_components) = connected_components(&graph);
    assert_eq!(n_components, 1);
    assert!(labels.iter().all(|&l| l == labels[0]));
}

#[test]
fn test_connected_components_two_cliques() {
    let graph = make_test_graph(6, vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]);
    let (labels, n_components) = connected_components(&graph);
    assert_eq!(n_components, 2);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[3], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

#[test]
fn test_connected_components_isolates() {
    let graph = make_test_graph(4, vec![]);
    let (labels, n_components) = connected_components(&graph);
    assert_eq!(n_components, 4);
    let unique: HashSet<usize> = labels.iter().cloned().collect();
    assert_eq!(unique.len(), 4);
}

/// `Both` means the pair is physically adjacent AND expression-similar. It
/// must read as spatial, because every consumer that filters on `spatial` is
/// asking whether the two cells are neighbours, and for a `Both` edge they
/// are. Reading it as `expression` would silently drop real adjacencies from
/// the mesh view and from the directional activity test.
#[test]
fn a_pair_in_both_graphs_reads_as_spatial() {
    use matrix_util::knn_graph::EdgeSource;
    assert_eq!(edge_kind_code(EdgeSource::Primary), EDGE_KIND_SPATIAL);
    assert_eq!(edge_kind_code(EdgeSource::Both), EDGE_KIND_SPATIAL);
    assert_eq!(edge_kind_code(EdgeSource::Secondary), EDGE_KIND_EXPRESSION);
    assert_ne!(EDGE_KIND_SPATIAL, EDGE_KIND_EXPRESSION);
}

///////////////////////////////////////////////////////
// The coord_pairs table: what leaves, what stays in //
///////////////////////////////////////////////////////

mod coord_pairs_table {
    use crate::test_support::make_test_graph;
    use crate::util::cell_pairs::*;
    use crate::util::common::*;
    use data_beans::sparse_io_vector::SparseIoVec;
    use matrix_util::knn_graph::EdgeSource;
    use matrix_util::parquet::peek_parquet_field_names;
    use ndarray::Array2;
    use std::sync::Arc;

    /// Shape and column names are all that matters; no test reads counts.
    fn make_data(n_cells: usize) -> anyhow::Result<SparseIoVec> {
        let raw = Array2::<f32>::zeros((3, n_cells));
        let mut sp = create_sparse_from_ndarray(&raw, None, None)?;
        sp.register_row_names_vec(
            &(0..3)
                .map(|i| format!("g{i}").into())
                .collect::<Vec<Box<str>>>(),
        );
        sp.register_column_names_vec(
            &(0..n_cells)
                .map(|i| format!("cell{i}").into())
                .collect::<Vec<Box<str>>>(),
        );
        sp.preload_columns()?;
        let mut data = SparseIoVec::new();
        data.push(Arc::from(sp), Some("b0".into()))?;
        Ok(data)
    }

    /// 4 cells in two batches. The third coordinate is the numeric batch
    /// pseudo-coordinate `read_data_with_coordinates` appends: 0 for the
    /// first pair of cells, a large offset for the second.
    fn fixture_coords() -> (Mat, Vec<Box<str>>, Vec<Box<str>>) {
        let coords = Mat::from_row_slice(
            4,
            3,
            &[
                0.0, 0.0, 0.0, //
                3.0, 4.0, 0.0, //
                10.0, 0.0, 1000.0, //
                10.0, 1.0, 1000.0, //
            ],
        );
        let names: Vec<Box<str>> = vec!["x".into(), "y".into(), "batch".into()];
        let labels: Vec<Box<str>> =
            vec!["left".into(), "left".into(), "right".into(), "right".into()];
        (coords, names, labels)
    }

    /// The batch MEMBERSHIP labels are what identify a core downstream, so
    /// they export as strings under `left_batch` / `right_batch`; the
    /// numeric pseudo-coordinate that used to squat on those names is
    /// internal scaffolding and must not be exported at all. It stays in
    /// the in-memory coordinates, which is what the auto-batch fold reads.
    #[test]
    fn batch_labels_export_as_strings_and_the_pseudo_coordinate_stays_internal(
    ) -> anyhow::Result<()> {
        let data = make_data(4)?;
        let (coords, names, labels) = fixture_coords();
        let graph = make_test_graph(4, vec![(0, 1), (2, 3), (1, 2)]);
        let pairs = SrtCellPairs::with_graph(&data, &coords, &graph, None, Some(&labels));

        // Internally all three coordinates remain: the fold judges
        // component containment on the offset column.
        assert_eq!(pairs.num_coordinates(), 3);

        let dir = tempfile::tempdir()?;
        let prefix = dir.path().join("t").to_str().unwrap().to_string();
        pairs.write_coord_pairs(&prefix, &names)?;

        let path = format!("{prefix}.coord_pairs.parquet");
        let fields: Vec<Box<str>> = peek_parquet_field_names(&path)?;
        let fields: Vec<&str> = fields.iter().map(|f| f.as_ref()).collect();
        assert_eq!(
            fields,
            vec![
                "cell_pair",
                "left_cell",
                "right_cell",
                "left_x",
                "left_y",
                "right_x",
                "right_y",
                "left_batch",
                "right_batch",
                "distance",
            ],
            "labels exported once, as strings; no left_batch coordinate"
        );

        // The production reader agrees with the writer: within-batch edges
        // carry their label, the straddling edge resolves to None.
        let mut edges: Vec<crate::lr_activity::io::EdgeRecord> = graph
            .edges
            .iter()
            .map(|&(l, r)| crate::lr_activity::io::EdgeRecord {
                left_cell: format!("cell{l}").into(),
                right_cell: format!("cell{r}").into(),
                community: 0,
                batch: None,
                is_spatial: true,
            })
            .collect();
        crate::lr_activity::io::attach_batch_from_coord_pairs(&mut edges, &path)?;
        assert_eq!(edges[0].batch.as_deref(), Some("left"));
        assert_eq!(edges[1].batch.as_deref(), Some("right"));
        assert_eq!(edges[2].batch, None, "a straddling edge belongs to no core");
        Ok(())
    }

    /// An augmented graph keeps `distance_rank` for kernels, but the
    /// physical length returns beside it: Euclidean over the REAL
    /// coordinates only (a spatial pair crossing the batch offset must not
    /// inherit it), and NaN for expression pairs, whose endpoints are not
    /// adjacent in any physical space.
    #[test]
    fn physical_distance_returns_beside_the_rank() -> anyhow::Result<()> {
        let data = make_data(4)?;
        let (coords, names, labels) = fixture_coords();
        let graph = make_test_graph(4, vec![(0, 1), (2, 3), (1, 2), (0, 3)]);
        let sources = [
            EdgeSource::Primary,
            EdgeSource::Primary,
            EdgeSource::Primary, // spatial, straddles the batch offset
            EdgeSource::Secondary,
        ];
        let pairs = SrtCellPairs::with_graph(&data, &coords, &graph, Some(&sources), Some(&labels));

        let dir = tempfile::tempdir()?;
        let prefix = dir.path().join("t").to_str().unwrap().to_string();
        pairs.write_coord_pairs(&prefix, &names)?;

        let path = format!("{prefix}.coord_pairs.parquet");
        let fields = peek_parquet_field_names(&path)?;
        assert!(fields.iter().any(|f| f.as_ref() == "distance_rank"));
        assert!(fields.iter().any(|f| f.as_ref() == "distance"));

        use matrix_util::parquet::ParquetReader;
        let read = ParquetReader::new(&path, Some(0), None, None)?;
        let di = read
            .column_names
            .iter()
            .position(|c| c.as_ref() == "distance")
            .expect("distance column");
        let ncol = read.column_names.len();
        let d = |i: usize| read.row_major_data[i * ncol + di] as f32;

        assert!(
            (d(0) - 5.0).abs() < 1e-5,
            "0-1: 3-4-5 triangle, got {}",
            d(0)
        );
        assert!((d(1) - 1.0).abs() < 1e-5, "2-3: unit apart, got {}", d(1));
        assert!(
            (d(2) - 65.0f32.sqrt()).abs() < 1e-4,
            "1-2 must ignore the 1000-unit batch offset, got {}",
            d(2)
        );
        assert!(d(3).is_nan(), "expression pair has no physical length");
        Ok(())
    }
}
