//! Output writers for the faba simulator: per-modality `.zarr.zip`
//! matrices in faba's `{gene}/{modality}/{detail}` row convention plus
//! all the ground-truth parquets harnesses need.

use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use data_beans::zarr_io::{apply_zip_flag, finalize_zarr_output};
use log::info;
use matrix_util::common_io::write_lines;
use matrix_util::traits::*;
use nalgebra::DMatrix;

use super::latents::Latents;
use super::{FabaArgs, MODALITIES, SUBSTRATE_AXIS_NAMES};

/// Filenames used for each modality's sparse output (without backend suffix).
const FILE_STEMS: [&str; 4] = ["genes", "dartseq", "atoi", "apa"];

#[allow(clippy::too_many_arguments)]
pub fn write_all(
    args: &FabaArgs,
    lats: &Latents,
    held_out: &[Vec<bool>],
    batch_membership: &[usize],
    ln_delta_per_mod: &[DMatrix<f32>],
    count_triplets: &[(u64, u64, f32)],
    modifier_triplets: &[Vec<(u64, u64, f32)>],
    modifier_row_keys: &[Vec<(usize, usize)>],
) -> anyhow::Result<()> {
    let g = lats.beta_g.len();
    let n = lats.theta_kn.ncols();
    let backend = args.backend.clone();
    let backend_suffix = match backend {
        SparseIoBackend::Zarr => "zarr",
        SparseIoBackend::HDF5 => "h5",
    };

    ////////////////////////////////
    // Per-modality .zarr.zip //
    ////////////////////////////////
    // Count modality (m=0): rows are 2G, layout: spliced (0..G), unspliced (G..2G).
    {
        let row_names = build_count_row_names(&lats.gene_names);
        write_modality_zarr(
            args,
            FILE_STEMS[0],
            backend_suffix,
            count_triplets,
            (2 * g, n),
            &row_names,
            &lats.cell_names,
        )?;
    }
    // Modifier modalities: rows = (substrate-positive, non-held-out) genes × C_m.
    for (i, m) in (1..MODALITIES.len()).enumerate() {
        let row_keys = &modifier_row_keys[i];
        let row_names: Vec<Box<str>> = row_keys
            .iter()
            .map(|&(gi, ci)| {
                format!(
                    "{}/{}/{}",
                    lats.gene_names[gi],
                    MODALITIES[m],
                    lats.detail(m, ci)
                )
                .into_boxed_str()
            })
            .collect();
        write_modality_zarr(
            args,
            FILE_STEMS[m],
            backend_suffix,
            &modifier_triplets[i],
            (row_keys.len().max(1), n),
            &row_names,
            &lats.cell_names,
        )?;
    }

    ///////////////////////////////
    // Ground-truth parquets //
    ///////////////////////////////
    let modality_names: Vec<Box<str>> = MODALITIES.iter().map(|s| (*s).into()).collect();

    let substrate_axis_names: Vec<Box<str>> = SUBSTRATE_AXIS_NAMES
        .iter()
        .map(|name| (*name).into())
        .collect();

    write_dmatrix(
        &format!("{}.substrate.parquet", args.out),
        &lats.s_g,
        &lats.gene_names,
        Some("gene"),
        Some(&substrate_axis_names),
    )?;

    write_dmatrix(
        &format!("{}.substrate_weights.parquet", args.out),
        &lats.w_m,
        &modality_names,
        Some("modality"),
        Some(&substrate_axis_names),
    )?;

    // φ as a wide [G × M] binary matrix.
    let phi_mat = phi_to_matrix(&lats.phi, g);
    write_dmatrix(
        &format!("{}.substrate_mask.parquet", args.out),
        &phi_mat,
        &lats.gene_names,
        Some("gene"),
        Some(&modality_names),
    )?;

    let held_mat = phi_to_matrix(held_out, g);
    write_dmatrix(
        &format!("{}.held_out_mask.parquet", args.out),
        &held_mat,
        &lats.gene_names,
        Some("gene"),
        Some(&modality_names),
    )?;

    let program_names = indexed_names(lats.a_mk.ncols(), "program");
    write_dmatrix(
        &format!("{}.program_writer_editor.parquet", args.out),
        &lats.a_mk,
        &modality_names,
        Some("modality"),
        Some(&program_names),
    )?;

    write_dmatrix(
        &format!("{}.gene_program_loadings.parquet", args.out),
        &lats.z_gk,
        &lats.gene_names,
        Some("gene"),
        Some(&program_names),
    )?;

    let beta_g_mat = DMatrix::<f32>::from_iterator(g, 1, lats.beta_g.iter().copied());
    let one_label: [Box<str>; 1] = ["beta_g".into()];
    write_dmatrix(
        &format!("{}.gene_baseline.parquet", args.out),
        &beta_g_mat,
        &lats.gene_names,
        Some("gene"),
        Some(&one_label),
    )?;

    write_dmatrix(
        &format!("{}.modality_base.parquet", args.out),
        &lats.base_gm,
        &lats.gene_names,
        Some("gene"),
        Some(&modality_names),
    )?;

    let topic_names = indexed_names(lats.beta_topic_gk.ncols(), "topic");
    write_dmatrix(
        &format!("{}.topic_dictionary.parquet", args.out),
        &lats.beta_topic_gk,
        &lats.gene_names,
        Some("gene"),
        Some(&topic_names),
    )?;

    // θ written as [N × K_topic] for convenience (rows = cells).
    let theta_nk = lats.theta_kn.transpose();
    write_dmatrix(
        &format!("{}.topic_proportions.parquet", args.out),
        &theta_nk,
        &lats.cell_names,
        Some("cell"),
        Some(&topic_names),
    )?;

    // α mixture weights: one parquet per modality. Stored as [G × C_m].
    for (m, modality) in MODALITIES.iter().enumerate() {
        let alpha_gc = lats.alpha_per_mod[m].transpose();
        let comp_names: Vec<Box<str>> = (0..alpha_gc.ncols()).map(|c| lats.detail(m, c)).collect();
        write_dmatrix(
            &format!("{}.alpha_{}.parquet", args.out, modality),
            &alpha_gc,
            &lats.gene_names,
            Some("gene"),
            Some(&comp_names),
        )?;
    }

    /////////////////////////////////////////
    // Batch artefacts (only if B > 1) //
    /////////////////////////////////////////
    let bb = ln_delta_per_mod[0].ncols();
    if bb > 1 {
        let batch_lines: Vec<Box<str>> = batch_membership
            .iter()
            .map(|b| b.to_string().into_boxed_str())
            .collect();
        write_lines(&batch_lines, &format!("{}.batch.gz", args.out))?;

        let batch_names = indexed_names(bb, "batch");
        for (m, ln_delta) in ln_delta_per_mod.iter().enumerate() {
            write_dmatrix(
                &format!("{}.ln_batch_{}.parquet", args.out, MODALITIES[m]),
                ln_delta,
                &lats.gene_names,
                Some("gene"),
                Some(&batch_names),
            )?;
        }
    }

    //////////////////////////////////////////
    // Substrate intercepts (small TSV) //
    //////////////////////////////////////////
    write_intercepts(args, lats)?;

    //////////////////////////////////////////////
    // Cell barcodes file (faba-compatible) //
    //////////////////////////////////////////////
    write_lines(&lats.cell_names, &format!("{}.barcodes.txt", args.out))?;

    Ok(())
}

fn write_modality_zarr(
    args: &FabaArgs,
    stem: &str,
    backend_suffix: &str,
    triplets: &[(u64, u64, f32)],
    shape: (usize, usize),
    row_names: &[Box<str>],
    cell_names: &[Box<str>],
) -> anyhow::Result<()> {
    let backend = args.backend.clone();
    let backend_dir = format!("{}.{}.{}", args.out, stem, backend_suffix);
    let final_path = apply_zip_flag(&backend_dir, args.zip_output());
    let (nrows, ncols) = shape;
    let mut data = create_sparse_from_triplets(
        triplets,
        (nrows, ncols, triplets.len()),
        Some(&backend_dir),
        Some(&backend),
    )?;
    data.register_row_names_vec(row_names);
    data.register_column_names_vec(cell_names);
    finalize_zarr_output(&backend_dir, &final_path)?;
    info!(
        "wrote modality '{}': {} rows × {} cells → {}",
        stem, nrows, ncols, final_path
    );
    Ok(())
}

fn build_count_row_names(gene_names: &[Box<str>]) -> Vec<Box<str>> {
    let g = gene_names.len();
    let mut out: Vec<Box<str>> = Vec::with_capacity(2 * g);
    for name in gene_names {
        out.push(format!("{}/count/spliced", name).into_boxed_str());
    }
    for name in gene_names {
        out.push(format!("{}/count/unspliced", name).into_boxed_str());
    }
    out
}

fn phi_to_matrix(mask: &[Vec<bool>], g: usize) -> DMatrix<f32> {
    let m = mask.len();
    let mut out = DMatrix::<f32>::zeros(g, m);
    for mi in 0..m {
        for gi in 0..g {
            out[(gi, mi)] = if mask[mi][gi] { 1.0 } else { 0.0 };
        }
    }
    out
}

fn indexed_names(n: usize, prefix: &str) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{}_{}", prefix, i).into_boxed_str())
        .collect()
}

fn write_dmatrix(
    path: &str,
    m: &DMatrix<f32>,
    row_names: &[Box<str>],
    row_label: Option<&str>,
    col_names: Option<&[Box<str>]>,
) -> anyhow::Result<()> {
    m.to_parquet_with_names(path, (Some(row_names), row_label), col_names)?;
    info!("wrote {}", path);
    Ok(())
}

fn write_intercepts(args: &FabaArgs, lats: &Latents) -> anyhow::Result<()> {
    let path = format!("{}.intercepts.tsv.gz", args.out);
    let mut lines: Vec<Box<str>> = Vec::with_capacity(lats.intercept_m.len() + 1);
    lines.push("modality\tintercept\trealized_coverage".into());
    for (m, &b) in lats.intercept_m.iter().enumerate() {
        let cov =
            lats.phi[m].iter().filter(|x| **x).count() as f32 / lats.phi[m].len().max(1) as f32;
        lines.push(format!("{}\t{:.6}\t{:.6}", MODALITIES[m], b, cov).into_boxed_str());
    }
    write_lines(&lines, &path)?;
    Ok(())
}
