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
        device: crate::embed_common::ComputeDevice::Cpu,
        device_no: 0,
    }
}

#[test]
fn explicit_reference_flags_bypass_the_manifest() -> anyhow::Result<()> {
    // The model prefix does not exist on disk; if the negotiation touched a
    // manifest anyway, this would fail on the missing file.
    let mut args = base_args("no/such/model".into(), "out".into());
    args.reference_latent = Some("ref.latent.parquet".into());
    args.reference_data = Some(vec!["ref.zarr".into()]);
    let spec = resolve_reference(&args, RunKind::Topic)?;
    assert_eq!(spec.latent.as_deref(), Some("ref.latent.parquet"));
    assert_eq!(spec.data_files, vec![Box::<str>::from("ref.zarr")]);
    Ok(())
}

#[test]
fn an_svd_model_needs_no_reference_latent() -> anyhow::Result<()> {
    let mut args = base_args("no/such/model".into(), "out".into());
    args.reference_data = Some(vec!["ref.zarr".into()]);
    let spec = resolve_reference(&args, RunKind::Svd)?;
    assert!(spec.latent.is_none());
    Ok(())
}

#[test]
fn a_missing_manifest_names_the_explicit_flags_as_the_way_out() {
    // Topic model, no explicit latent → the negotiation must consult the
    // (absent) manifest and fail with actionable guidance.
    let mut args = base_args("no/such/model".into(), "out".into());
    args.reference_data = Some(vec!["ref.zarr".into()]);
    let err = resolve_reference(&args, RunKind::Topic).unwrap_err();
    assert!(err.to_string().contains("--reference-latent"));
}

#[test]
fn a_reference_in_a_different_cell_space_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let bge_prefix = write_manifest(&dir, "embed_run", RunKind::Bge);
    let mut args = base_args("model".into(), "out".into());
    args.reference = Some(bge_prefix);
    // A topic model matched against a bge reference: simplex vs embedding.
    let err = resolve_reference(&args, RunKind::Topic).unwrap_err();
    assert!(err.to_string().contains("different"), "{err}");
}

#[test]
fn a_reference_with_a_different_matching_plan_is_refused() {
    // vae and svd share a Signed cell space, but NOT a matching plan: the
    // svd latent is whitened by a scale that is not persisted, so plan
    // equality — not cell-space equality — is the gate.
    let dir = tempfile::tempdir().unwrap();
    let svd_prefix = write_manifest(&dir, "svd_run", RunKind::Svd);
    let mut args = base_args("model".into(), "out".into());
    args.reference = Some(svd_prefix);
    let err = resolve_reference(&args, RunKind::Vae).unwrap_err();
    assert!(
        err.to_string().contains("different matching space"),
        "{err}"
    );
}

#[test]
fn kinds_without_a_projection_are_refused_up_front() {
    let dir = tempfile::tempdir().unwrap();
    for kind in [RunKind::Fne, RunKind::Gem, RunKind::GemEncoder] {
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
fn the_joint_families_are_refused_before_predict_is_reached() {
    // They write no encoder checkpoint, so the simplex arm would otherwise
    // fail inside `predict` with a missing-model.json error naming a path the
    // user never typed — after the reference had already been resolved.
    let dir = tempfile::tempdir().unwrap();
    for kind in [RunKind::JointTopic, RunKind::JointSvd] {
        let prefix = write_manifest(&dir, &format!("run_{kind}"), kind);
        let args = base_args(prefix, "out".into());
        let err = impute_model(&args).unwrap_err();
        assert!(
            err.to_string().contains("no encoder checkpoint"),
            "{kind}: {err}"
        );
    }
}

#[test]
fn missing_manifest_falls_back_to_the_default_scale() {
    assert_eq!(crate::svd::project::column_sum_norm("no/such/prefix"), 1e4);
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

    let proj_aligned = crate::svd::project::project_onto_dictionary(
        &aligned,
        &train_genes,
        &u_dk,
        1e4,
        &crate::topic::eval::QueryNameOpts::default(),
        None,
        "test",
    )?;
    let proj_permuted = crate::svd::project::project_onto_dictionary(
        &permuted,
        &train_genes,
        &u_dk,
        1e4,
        &crate::topic::eval::QueryNameOpts::default(),
        None,
        "test",
    )?;

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
    assert!(crate::svd::project::project_onto_dictionary(
        &query,
        &train_genes,
        &u_dk,
        1e4,
        &crate::topic::eval::QueryNameOpts::default(),
        None,
        "test"
    )
    .is_err());
    Ok(())
}

/// Only the `initialized` genes come through, in the alignment's order, and the
/// column is looked up by NAME in the rates table (whose order is the writer's).
#[test]
fn model_imputed_columns_keep_only_initialized_genes_by_name() {
    let al_genes: Vec<Box<str>> = ["A", "B", "U1", "U2"].iter().map(|s| (*s).into()).collect();
    let al_status: Vec<Box<str>> = ["matched", "missing", "initialized", "initialized"]
        .iter()
        .map(|s| (*s).into())
        .collect();
    // rates hold missing + initialized, in a different order than the alignment.
    let rate_genes: Vec<Box<str>> = ["U2", "B", "U1"].iter().map(|s| (*s).into()).collect();
    let rates = Mat::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let (names, m) = model_imputed_columns(&al_genes, &al_status, &rate_genes, &rates);
    assert_eq!(names, vec![Box::<str>::from("U1"), Box::<str>::from("U2")]);
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 2);
    assert_eq!(m[(0, 0)], 3.0); // U1, cell 0
    assert_eq!(m[(0, 1)], 1.0); // U2, cell 0
    assert_eq!(m[(1, 0)], 6.0);
    assert_eq!(m[(1, 1)], 4.0);
}

/// No initialized gene → an empty table, not an error.
#[test]
fn model_imputed_columns_are_empty_without_initialized_genes() {
    let al_genes: Vec<Box<str>> = ["A"].iter().map(|s| (*s).into()).collect();
    let al_status: Vec<Box<str>> = ["matched"].iter().map(|s| (*s).into()).collect();
    let (names, m) = model_imputed_columns(&al_genes, &al_status, &[], &Mat::zeros(2, 0));
    assert!(names.is_empty());
    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 0);
}

/// A simba run projects through the bge path, so it matches by cosine in
/// its cell embedding exactly as bge does.
#[test]
fn simba_shares_bges_matching_plan() {
    assert_eq!(
        matching_plan(RunKind::Simba).unwrap(),
        matching_plan(RunKind::Bge).unwrap()
    );
}
