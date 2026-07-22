//! Every table `faba gem-encoder` puts on disk, and the two invariants that
//! guard them.
//!
//! **The non-finite gate.** Every artifact funnels through [`write_matrix`],
//! which refuses to persist NaN/Inf. A diverged run should fail loudly at the
//! write rather than leave a plausible-looking parquet behind for someone to
//! analyze.
//!
//! **Cell QC is applied in exactly one place**, [`write_cell_table`], so every
//! per-cell table this run writes is filtered by the same keep set and the rows
//! can never desync from the barcodes.
//!
//! Diagnostics and the model manifest are written by
//! [`crate::gem_encoder::report`]; the velocity tables by
//! [`crate::gem_encoder::velocity`], both through the primitives here.

use candle_util::candle_core::{Device, Tensor};
use candle_util::decoder::gem_etm::{GemEtmDecoder, Track};
use candle_util::encoder::gem_encoder::GemIndexedEncoder;
use log::info;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;

use crate::gem_encoder::load::PreparedData;

type Mat = DMatrix<f32>;

/// Pull a tensor to host and write it, refusing to persist a diverged artifact.
pub fn write_matrix(
    t: &Tensor,
    path: &str,
    row_names: &[Box<str>],
    row_axis: &str,
    col_prefix: &str,
) -> anyhow::Result<()> {
    let host = t.to_device(&Device::Cpu)?;
    let flat: Vec<f32> = host.flatten_all()?.to_vec1()?;
    let bad = flat.iter().filter(|x| !x.is_finite()).count();
    anyhow::ensure!(
        bad == 0,
        "refusing to write {path}: {bad} non-finite (NaN/Inf) entries — training diverged. \n\
         Check the log_likelihood trace and any \"skipped optimizer step\" warnings, and \n\
         re-run with a lower --learning-rate."
    );
    let n_cols = host.dims().last().copied().unwrap_or(0);
    let cols: Vec<Box<str>> = (0..n_cols)
        .map(|i| format!("{col_prefix}{i}").into_boxed_str())
        .collect();
    host.to_parquet_with_names(path, (Some(row_names), Some(row_axis)), Some(&cols))?;
    info!("wrote {path}");
    Ok(())
}

/// Write a flat `[n, k]` host buffer as a cell-keyed parquet.
pub fn write_cell_table(
    data: &[f32],
    n: usize,
    k: usize,
    path: &str,
    cell_names: &[Box<str>],
    col_prefix: &str,
    keep: Option<&[usize]>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        data.len() == n * k,
        "{path}: buffer is {} long but {n}×{k} was expected",
        data.len()
    );
    // Cell QC is applied HERE and only here, so every per-cell table this run
    // writes is filtered by the same set and the rows can never desync from the
    // barcodes — they are subset in one place, together. Mirrors
    // `graph_embedding_util::eval`'s `cell_keep_idx`, which `senna bge` uses.
    let (rows, names, n_out) = match keep {
        Some(keep) => {
            let mut buf = Vec::with_capacity(keep.len() * k);
            let mut nm = Vec::with_capacity(keep.len());
            for &i in keep {
                anyhow::ensure!(
                    i < n,
                    "{path}: QC keep index {i} is out of range for {n} cells"
                );
                buf.extend_from_slice(&data[i * k..(i + 1) * k]);
                nm.push(cell_names[i].clone());
            }
            (
                std::borrow::Cow::Owned(buf),
                std::borrow::Cow::Owned(nm),
                keep.len(),
            )
        }
        None => (
            std::borrow::Cow::Borrowed(data),
            std::borrow::Cow::Borrowed(cell_names),
            n,
        ),
    };
    let t = Tensor::from_vec(rows.into_owned(), (n_out, k), &Device::Cpu)?;
    write_matrix(&t, path, &names, "cell", col_prefix)
}

/// Gem row names for one track, the axis both feature-embedding files are keyed
/// on. `faba::gem::marker_embedding::load_gene_embedding` selects rows by this
/// suffix, so it is what makes a gem-encoder run readable by `faba annotate`.
fn track_row_names(gene_names: &[Box<str>], track: Track) -> Vec<Box<str>> {
    gene_names
        .iter()
        .map(|g| format!("{g}{}", track.row_suffix()).into_boxed_str())
        .collect()
}

/// The gene-side tables read straight off the fitted model: both dictionaries,
/// the per-gene dispersion, and the raw `δ`.
///
/// `{out}.delta_feature_embedding.parquet` is `δ` alone, so the nascent program
/// is recoverable as `raw_feature_embedding − delta`. Its SIGN convention is not
/// gem's — see [`crate::gem_encoder::report::save_model_metadata`].
pub fn write_dictionaries(
    decoder: &GemEtmDecoder,
    encoder: &GemIndexedEncoder,
    gene_names: &[Box<str>],
    out: &str,
) -> anyhow::Result<()> {
    write_matrix(
        &decoder.get_dictionary(Track::Mature)?,
        &format!("{out}.dictionary.parquet"),
        gene_names,
        "gene",
        "T",
    )?;
    write_matrix(
        &decoder.get_dictionary(Track::Nascent)?,
        &format!("{out}.dictionary_nascent.parquet"),
        gene_names,
        "gene",
        "T",
    )?;
    write_matrix(
        &decoder.phi_g2()?,
        &format!("{out}.dispersion.parquet"),
        gene_names,
        "gene",
        "phi",
    )?;
    write_matrix(
        encoder.delta_embeddings(),
        &format!("{out}.delta_feature_embedding.parquet"),
        gene_names,
        "gene",
        "H",
    )
}

/// Write `{out}.raw_feature_embedding.parquet` keyed by **gem row names**, so
/// `faba::gem::marker_embedding::load_gene_embedding` reads it unchanged.
///
/// Only the `/count/spliced` rows are written, and they carry `ρ + δ`. The
/// nascent program is NOT emitted here because it is exactly `ρ`, i.e. these
/// rows minus `{out}.delta_feature_embedding.parquet`; writing it too would put
/// `3·G·H` numbers on disk to carry `2·G·H` of information.
///
/// Returns `(mature, nascent)` — BOTH, because the co-embedded file does need
/// both (see [`write_coembedding`]); it is only this raw table that can skip the
/// redundant half.
///
/// NOTE `{out}.feature_embedding.parquet` — the one `faba annotate --mode
/// projection` reads — is a DIFFERENT file, written by [`write_coembedding`]
/// after this, because a nearest-centroid call needs genes on the cell manifold
/// and these raw rows are not.
pub fn write_feature_embedding(
    encoder: &GemIndexedEncoder,
    gene_names: &[Box<str>],
    out: &str,
) -> anyhow::Result<(Tensor, Tensor)> {
    let cpu = Device::Cpu;
    // Nascent is the base, so ρ IS the nascent program and the SPLICED one is ρ + δ.
    let rho = encoder.feature_embeddings().to_device(&cpu)?;
    let mature = (&rho + encoder.delta_embeddings().to_device(&cpu)?)?;
    // A gene whose input carried just one track still gets a row here — the
    // model has an embedding for it regardless.
    let rows = track_row_names(gene_names, Track::Mature);
    write_matrix(
        &mature,
        &format!("{out}.raw_feature_embedding.parquet"),
        &rows,
        "feature",
        "H",
    )?;
    Ok((mature, rho))
}

/// Write `{out}.cell_embedding.parquet` — cells placed in the GENE-embedding
/// space, `[N, H]`, so `faba annotate` / `lineage` / `plot` can read this run.
///
/// `cell_embedding = θ · α`. That is an identity, not an approximation: the
/// decoder's own per-gene score is
///
/// ```text
/// Σ_t θ_t ⟨α_t, ρ_g⟩ = ⟨ Σ_t θ_t α_t , ρ_g ⟩ = ⟨ θ·α , ρ_g ⟩
/// ```
///
/// so `θ·α` is exactly the vector whose inner product with a gene embedding
/// reproduces what the model predicts for that gene. Same construction pinto's
/// `lc-etm` uses to co-embed a topic fit.
///
/// **Why this matters beyond plumbing.** Every diagnostic this model reports —
/// effective rank, θ_max, between-cell variance, the splice-ratio check — is a
/// property of the fit itself, so when two of them disagree nothing adjudicates.
/// Marker-based annotation is the first EXTERNAL criterion available: it asks
/// whether the latent recovers cell types the model never saw.
///
/// `latent` is already log θ (see [`crate::gem_encoder::infer::Inferred`]), so
/// the map here is `exp`.
pub fn write_cell_embedding(
    latent: &[f32],
    alpha: &Tensor,
    cell_names: &[Box<str>],
    k: usize,
    out: &str,
    keep: Option<&[usize]>,
) -> anyhow::Result<Tensor> {
    let cpu = Device::Cpu;
    let n = cell_names.len();
    let h = alpha.dim(1)?;

    let theta = Tensor::from_vec(latent.to_vec(), (n, k), &cpu)?.exp()?;

    let z = theta.matmul(alpha)?; // [N, H]
    let emb: Vec<f32> = z.flatten_all()?.to_vec1()?;
    write_cell_table(
        &emb,
        n,
        h,
        &format!("{out}.cell_embedding.parquet"),
        cell_names,
        "H",
        keep,
    )?;
    // The FULL z is returned: the co-embedding places genes on the cell
    // manifold and wants every cell's geometry, QC-failed ones included. QC is
    // an output filter, not a change to what the model saw.
    Ok(z)
}

/// Write `{out}.feature_embedding.parquet` — genes placed on the CELL manifold.
///
/// This file must be CO-EMBEDDED, not raw `ρ+δ`.
///
/// Cells live in the convex hull of α while ρ fans out off that simplex, so
/// the two clouds are not in a shared metric space — measured on a six-file
/// fit, `cos(mean cell, mean gene) = −0.92`, which made every cell roughly
/// antipodal to every gene and left `faba annotate` unable to assign a single
/// cell. The per-gene `logit_bias` makes this worse by construction: β only
/// depends on gene-to-gene DIFFERENCES, so once `b_g` absorbs the level the
/// absolute direction of ⟨α,ρ⟩ is a gauge freedom the likelihood never pins.
///
/// The SIMBA `si.tl.embed` transform re-places each gene at the
/// softmax-over-cells weighted mean of the CELL embeddings, landing it on the
/// cell manifold. Same treatment `senna rest` applies for the same reason,
/// where the raw ρ "is the disjoint off-manifold cloud and is not written".
/// Post-hoc on that fit it moved the cosine to +0.77 and took the annotation
/// null from degenerate (67–100 % of terms untestable) to λ_perm = 0.76.
///
/// # Both tracks, one manifold
///
/// BOTH programs are co-embedded and written here — `ρ+δ` as `/count/spliced`
/// rows and `ρ` as `/count/unspliced` rows — matching the layout `faba gem`
/// emits and `faba::gem::marker_embedding::load_gene_embedding` selects on.
///
/// This is what makes `faba annotate --track velocity` (and therefore its
/// `both` DEFAULT) work on a gem-encoder run. Writing only the spliced rows made
/// that path fail outright, and the tempting shortcut — pointing the velocity
/// track at the raw `{out}.delta_feature_embedding.parquet` — is exactly the
/// off-manifold comparison the measurement above rules out: `δ` is a raw model
/// parameter, so it lands in the same disjoint cloud raw `ρ` does, and a
/// nearest-centroid call against cell velocity would be scoring across two
/// different metric spaces.
///
/// Both tracks are placed against the SAME cell cloud `z` in one call, so the
/// softmax temperature and the effective-cells target are shared and the two
/// halves stay directly comparable.
pub fn write_coembedding(
    z: &Tensor,
    mature: &Tensor,
    nascent: &Tensor,
    gene_names: &[Box<str>],
    k: usize,
    out: &str,
) -> anyhow::Result<()> {
    use anyhow::Context;

    let (_labels, target_eff) =
        graph_embedding_util::cell_clusters(z, Some(k)).context("cell clusters")?;

    let both = Tensor::cat(&[mature, nascent], 0)?; // [2G, H]
    let mut rows = track_row_names(gene_names, Track::Mature);
    rows.extend(track_row_names(gene_names, Track::Nascent));

    graph_embedding_util::write_feature_coembedding(out, z, &both, &rows, target_eff)
        .context("feature co-embedding")
}

/// Write the two pseudobulk tables that `faba annotate --mode enrichment`
/// needs on top of the dictionary and the latent.
///
/// Enrichment annotation asks a different question from nearest-centroid:
/// instead of "which marker centroid is this cell closest to in H-space", it
/// asks "which factor's gene program is over-represented for this cell type",
/// then carries the answer to cells through θ. That routes the call through
/// `β` and `θ` — the two things a topic model actually estimates — and never
/// forms a cell↔gene inner product, so it is immune to the gauge freedom that
/// makes `⟨z_c, ρ_g⟩` unreliable here: `β = softmax_g(b_g + ⟨α_t, ρ_g⟩)`
/// depends only on gene-to-gene DIFFERENCES, so `b_g` can absorb the level and
/// the absolute cell↔gene direction is never pinned by the likelihood.
///
/// The permutation null in `enrichment::annotate` needs a level ABOVE the
/// cell: it recomputes `β̃ = pb_gene · pb_membership[π]` under shuffled
/// pseudobulk labels, which is what makes the null respect the correlation
/// structure between genes instead of treating cells as exchangeable.
///
/// - `{out}.pb_gene.parquet` `[G, P]` — the finest pseudobulk's mature-track
///   posterior mean, on the same gene axis as `dictionary.parquet`.
/// - `{out}.pb_latent.parquet` `[P, K]` — **log θ per pseudobulk**, the mean of
///   its cells' θ. Log, like `{out}.latent.parquet`, so the two files carry the
///   one contract; averaging happens on the simplex and the log is taken after.
pub fn write_pseudobulk_tables(
    prepared: &PreparedData,
    finest: &data_beans_alg::collapse_data::CollapsedOut,
    latent: &[f32],
    k: usize,
    out: &str,
) -> anyhow::Result<()> {
    use matrix_param::traits::Inference;

    let mu_dp = finest.mu_observed.posterior_mean(); // [rows, P]
    let n_pb = mu_dp.ncols();
    let g = prepared.map.n_genes;
    let (_, mature_row) = prepared.map.per_gene_rows();

    // Genes lacking a mature row keep a zero column-slice: the model still has
    // an embedding for them, and a zero pseudobulk profile simply never
    // contributes to any enrichment score.
    // Pseudobulk OUTER, gene inner: both matrices are column-major, so this
    // order walks each column sequentially on both sides. Gene-outer strides
    // both by `n_pb`/`n_rows` per step — ~34 M cache-missing accesses.
    let mut pb_gene = Mat::zeros(g, n_pb);
    for p in 0..n_pb {
        for (gene, row) in mature_row.iter().enumerate() {
            if let Some(r) = row {
                pb_gene[(gene, p)] = mu_dp[(*r as usize, p)];
            }
        }
    }

    let n = prepared.data_vec.num_columns();
    let groups = prepared.data_vec.get_group_membership(0..n)?;
    anyhow::ensure!(
        groups.len() == n,
        "group membership covers {} of {n} cells",
        groups.len()
    );
    let mut pb_theta = Mat::zeros(n_pb, k);
    let mut counts = vec![0f32; n_pb];
    for (cell, &grp) in groups.iter().enumerate() {
        if grp >= n_pb {
            continue;
        }
        counts[grp] += 1.0;
        for t in 0..k {
            // `latent` is log θ; average on the simplex, not in log space.
            pb_theta[(grp, t)] += latent[cell * k + t].exp();
        }
    }
    for p in 0..n_pb {
        if counts[p] > 0.0 {
            for t in 0..k {
                pb_theta[(p, t)] = (pb_theta[(p, t)] / counts[p]).max(1e-20).ln();
            }
        }
    }

    // Through `write_matrix` so both tables get the same non-finite guard and
    // column-naming as every other artifact this run writes.
    let cpu = Device::Cpu;
    let pb_names: Vec<Box<str>> = (0..n_pb)
        .map(|p| format!("PB{p}").into_boxed_str())
        .collect();
    write_matrix(
        &matrix_util::traits::ConvertMatOps::to_tensor(&pb_gene, &cpu)?,
        &format!("{out}.pb_gene.parquet"),
        &track_row_names(&prepared.gene_names, Track::Mature),
        "gene",
        "PB",
    )?;
    write_matrix(
        &matrix_util::traits::ConvertMatOps::to_tensor(&pb_theta, &cpu)?,
        &format!("{out}.pb_latent.parquet"),
        &pb_names,
        "pb",
        "T",
    )?;
    info!("pseudobulk tables: pb_gene [{g}×{n_pb}], pb_latent [{n_pb}×{k}]");
    Ok(())
}

#[cfg(test)]
#[path = "qc_filter_tests.rs"]
mod qc_filter_tests;
