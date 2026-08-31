//! What impute's dispatch must guarantee: the reference negotiation keeps
//! explicit flags authoritative, unsupported run kinds are refused before
//! any data is read, and the svd projection is a genuine name-keyed remap
//! (row order must not matter) with the training-time transform.

use super::*;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};

/// A minimal manifest through the real writer, so the fixture tracks the
/// manifest schema instead of a hand-formatted JSON string.
fn write_manifest(dir: &tempfile::TempDir, name: &str, kind: RunKind) -> Box<str> {
    let prefix = dir.path().join(name);
    run_manifest::RunManifest::new(kind, name)
        .save(std::path::Path::new(&format!(
            "{}.senna.json",
            prefix.display()
        )))
        .unwrap();
    prefix.to_str().unwrap().into()
}

fn base_args(model: Box<str>, out: Box<str>) -> ImputeArgs {
    ImputeArgs {
        data_files: vec!["query.zarr".into()],
        model,
        out,
        reference: None,
        reference_latent: None,
        reference_data: None,
        reference_batch_files: None,
        batch_files: None,
        knn: 5,
        knn_temperature: 1.0,
        minibatch_size: 100,
        block_size: None,
        preload_data: false,
        verbose: false,
    }
}

#[test]
fn explicit_reference_flags_bypass_the_manifest() -> anyhow::Result<()> {
    // The model prefix does not exist on disk; if the negotiation touched a
    // manifest anyway, this would fail on the missing file.
    let mut args = base_args("no/such/model".into(), "out".into());
    args.reference_latent = Some("ref.latent.parquet".into());
    args.reference_data = Some(vec!["ref.zarr".into()]);
    let spec = resolve_reference(&args, RunKind::Topic, MatchingPlan::SoftmaxSimplex)?;
    assert_eq!(spec.latent.as_deref(), Some("ref.latent.parquet"));
    assert_eq!(spec.data_files, vec![Box::<str>::from("ref.zarr")]);
    Ok(())
}

#[test]
fn an_svd_model_needs_no_reference_latent() -> anyhow::Result<()> {
    let mut args = base_args("no/such/model".into(), "out".into());
    args.reference_data = Some(vec!["ref.zarr".into()]);
    let spec = resolve_reference(&args, RunKind::Svd, MatchingPlan::DictionaryProjection)?;
    assert!(spec.latent.is_none());
    Ok(())
}

#[test]
fn a_missing_manifest_names_the_explicit_flags_as_the_way_out() {
    // Topic model, no explicit latent → the negotiation must consult the
    // (absent) manifest and fail with actionable guidance.
    let mut args = base_args("no/such/model".into(), "out".into());
    args.reference_data = Some(vec!["ref.zarr".into()]);
    let err = resolve_reference(&args, RunKind::Topic, MatchingPlan::SoftmaxSimplex).unwrap_err();
    assert!(err.to_string().contains("--reference-latent"));
}

#[test]
fn a_reference_in_a_different_cell_space_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let bge_prefix = write_manifest(&dir, "embed_run", RunKind::Bge);
    let mut args = base_args("model".into(), "out".into());
    args.reference = Some(bge_prefix);
    // A topic model matched against a bge reference: simplex vs embedding.
    let err = resolve_reference(&args, RunKind::Topic, MatchingPlan::SoftmaxSimplex).unwrap_err();
    assert!(err.to_string().contains("different"), "{err}");
}

#[test]
fn kinds_without_a_projection_are_refused_up_front() {
    let dir = tempfile::tempdir().unwrap();
    for kind in [RunKind::Fne, RunKind::JointSvd] {
        let prefix = write_manifest(&dir, &format!("run_{kind}"), kind);
        let args = base_args(prefix, "out".into());
        let err = impute_model(&args).unwrap_err();
        assert!(
            err.to_string().contains("no query-side projection"),
            "{kind}: {err}"
        );
    }
}

#[test]
fn zero_rows_survive_normalization_untouched() {
    let mut m = Mat::from_row_slice(2, 2, &[3.0, 4.0, 0.0, 0.0]);
    l2_normalize_rows_inplace(&mut m);
    assert!((m.row(0).norm() - 1.0).abs() < 1e-6);
    assert!(m.row(1).iter().all(|&x| x == 0.0));
}

#[test]
fn missing_manifest_falls_back_to_the_default_scale() {
    assert_eq!(svd_column_sum_norm("no/such/prefix").unwrap(), 1e4);
}

fn make_backend(
    dir: &tempfile::TempDir,
    name: &str,
    gene_names: &[&str],
    triplets: &[(u64, u64, f32)],
    n_cells: usize,
) -> anyhow::Result<SparseIoVec> {
    let path = dir.path().join(name);
    let mut backend = create_sparse_from_triplets(
        triplets,
        (gene_names.len(), n_cells, triplets.len()),
        Some(path.to_str().unwrap()),
        Some(&SparseIoBackend::Zarr),
    )?;
    let genes: Vec<Box<str>> = gene_names.iter().map(|s| Box::from(*s)).collect();
    let cells: Vec<Box<str>> = (0..n_cells).map(|i| format!("cell_{i}").into()).collect();
    backend.register_row_names_vec(&genes);
    backend.register_column_names_vec(&cells);
    let mut data_vec = SparseIoVec::new();
    data_vec.push(Arc::from(backend), None)?;
    Ok(data_vec)
}

#[test]
fn svd_projection_is_keyed_by_gene_name_not_row_order() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let train_genes: Vec<Box<str>> = vec!["a".into(), "b".into(), "c".into()];
    let u_dk = Mat::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);

    // The same three cells twice: once on the training gene order, once
    // with the rows permuted (and the triplet row ids permuted to match).
    let aligned = make_backend(
        &dir,
        "aligned.zarr",
        &["a", "b", "c"],
        &[
            (0, 0, 5.0),
            (1, 0, 1.0),
            (1, 1, 4.0),
            (2, 1, 2.0),
            (0, 2, 3.0),
            (2, 2, 3.0),
        ],
        3,
    )?;
    let permuted = make_backend(
        &dir,
        "permuted.zarr",
        &["c", "a", "b"],
        &[
            (1, 0, 5.0),
            (2, 0, 1.0),
            (2, 1, 4.0),
            (0, 1, 2.0),
            (1, 2, 3.0),
            (0, 2, 3.0),
        ],
        3,
    )?;

    let proj_aligned =
        project_onto_svd_dictionary(&aligned, &train_genes, &u_dk, 1e4, None, "test")?;
    let proj_permuted =
        project_onto_svd_dictionary(&permuted, &train_genes, &u_dk, 1e4, None, "test")?;

    assert_eq!(proj_aligned.nrows(), 3);
    assert_eq!(proj_aligned.ncols(), 2);
    let diff = (&proj_aligned - &proj_permuted).norm();
    assert!(diff < 1e-5, "projection depends on row order: diff={diff}");
    Ok(())
}

#[test]
fn svd_projection_refuses_a_query_sharing_no_gene() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let train_genes: Vec<Box<str>> = vec!["a".into(), "b".into()];
    let u_dk = Mat::from_row_slice(2, 1, &[1.0, 1.0]);
    let query = make_backend(&dir, "query.zarr", &["x", "y"], &[(0, 0, 1.0)], 1)?;
    assert!(project_onto_svd_dictionary(&query, &train_genes, &u_dk, 1e4, None, "test").is_err());
    Ok(())
}
