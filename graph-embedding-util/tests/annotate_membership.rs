//! End-to-end check that `annotate_embeddings` emits the cell→coarse-label
//! `membership.tsv` (consumed by `faba gem-summary` and `data-beans stat -g`)
//! in the expected 2-column, header-less format.

use graph_embedding_util::type_annotation::{annotate_embeddings, AnnotateProjConfig};
use matrix_util::dmatrix_io::DMatrix;
use std::fs;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

#[test]
fn writes_membership_tsv() {
    // 4 genes, H=2: GENE0/1 point +x, GENE2/3 point +y.
    let feature_emb =
        DMatrix::<f32>::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
    let gene_names = names(&["GENE0", "GENE1", "GENE2", "GENE3"]);

    // 6 cells: three near +x, three near +y → two resolvable clusters.
    let cell_emb = DMatrix::<f32>::from_row_slice(
        6,
        2,
        &[
            0.9, 0.1, 1.0, 0.0, 0.8, 0.2, // +x
            0.1, 0.9, 0.0, 1.0, 0.2, 0.8, // +y
        ],
    );
    let cell_names = names(&["cell0", "cell1", "cell2", "cell3", "cell4", "cell5"]);

    let dir = std::env::temp_dir().join("geu_annotate_membership_test");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("run.gem_annot");
    let prefix = prefix.to_str().unwrap();

    // Two marker types matching the two embedding directions.
    let markers_path = dir.join("markers.tsv");
    fs::write(
        &markers_path,
        "GENE0\tXtype\nGENE1\tXtype\nGENE2\tYtype\nGENE3\tYtype\n",
    )
    .unwrap();

    let cfg = AnnotateProjConfig {
        n_perm: 0, // raw cosine — fast & deterministic
        seed: 42,
        knn: 3,
        resolution: 1.0,
        coarsen: true,
        ..AnnotateProjConfig::default()
    };
    annotate_embeddings(
        &feature_emb,
        &gene_names,
        &cell_emb,
        &cell_names,
        markers_path.to_str().unwrap(),
        prefix,
        true,
        &cfg,
    )
    .expect("annotate_embeddings failed");

    let membership = fs::read_to_string(format!("{prefix}.membership.tsv")).unwrap();
    let lines: Vec<&str> = membership.lines().collect();
    assert_eq!(lines.len(), 6, "one membership line per cell");

    for (line, expected_cell) in lines.iter().zip(cell_names.iter()) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            cols.len(),
            2,
            "membership rows are `cell<TAB>label`: {line:?}"
        );
        assert_eq!(cols[0], expected_cell.as_ref(), "cell in input order");
        assert!(!cols[1].is_empty(), "non-empty coarse label");
    }
    // No header: first token is a real cell barcode, not the literal "cell".
    assert_ne!(lines[0].split('\t').next(), Some("cell"));

    let _ = fs::remove_dir_all(&dir);
}
