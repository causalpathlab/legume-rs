//! `senna deconvolve` — projection-based hierarchical-Bayes bulk deconvolution.
//!
//! Builds a cell-type reference from a feature embedding (`bge --skip-etm` or
//! `masked-topic`): each type's gene profile is reconstructed as
//! `μ_{g,c} = exp(ρ_g·t_c + a_g)`, with anchors `t_c` drawn from the
//! annotate-by-projection marker centroids and their scatter. Bulk samples are
//! projected into the same latent, then a full Gibbs sampler (Gamma-Poisson
//! conjugate fractions + multinomial gene allocation + elliptical-slice anchor
//! updates) yields BayesPrism-style deliverables — per-sample cell-type
//! fractions with credible intervals and a per-cell-type expression tensor.

mod anchors;
mod args;
mod gibbs;
mod io;
mod project;
mod source;
#[cfg(test)]
mod tests;

pub use args::DeconvolveArgs;

use crate::embed_common::read_bulk_data_aligned;
use crate::run_manifest;
use anyhow::Result;
use log::info;
use matrix_util::common_io::mkdir_parent;
use source::EmbeddingSource;

pub fn run(args: &DeconvolveArgs) -> Result<()> {
    let out: String = match args.out.as_deref() {
        Some(o) => o.to_string(),
        None => format!("{}.deconv", run_manifest::derive_out_prefix(&args.from)),
    };
    mkdir_parent(&out)?;

    // 1. Reference: embedding ρ + gene offset, and marker-anchored prior.
    let src = EmbeddingSource::load(&args.from)?;
    let prior = anchors::build_anchor_prior(
        &src.anchor_emb,
        &src.feature_names,
        &args.markers,
        &args.anchor_config(),
    )?;

    // 2. Bulk: load and align to the reference gene order.
    let bulk = read_bulk_data_aligned(&args.bulk, &src.feature_names)?;
    info!(
        "deconvolve: {} bulk samples × {} genes, {} cell types",
        bulk.samples.len(),
        bulk.genes.len(),
        prior.names.len()
    );

    // 3. Project bulk into the shared latent + warm-start fractions.
    let cfg = args.sampler_config();
    let sample_z = project::project_bulk(&src.rho, &src.gene_offset, &bulk.data, cfg.project_ridge);
    let init_w = project::init_fractions(&prior.mean, &sample_z, cfg.init_iters);

    // 4. Hierarchical Gibbs.
    let result = gibbs::run_gibbs(&src, &bulk.data, &prior, &init_w, &cfg)?;

    // 5. Write deliverables.
    let meta = io::RunMeta {
        from: &args.from,
        markers: &args.markers,
        kind: src.kind.to_string(),
        exact: src.exact,
        warmup: cfg.warmup,
        draws: cfg.draws,
        bulk_files: &args.bulk,
    };
    io::write_outputs(&out, &bulk.samples, &bulk.genes, &sample_z, &result, &meta)?;
    info!("senna deconvolve complete → {out}.*");
    Ok(())
}
