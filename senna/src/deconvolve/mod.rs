//! `senna deconvolve` — hierarchical-Bayes bulk deconvolution.
//!
//! Bulk counts are split across empirical reference components by a full Gibbs
//! sampler (Gamma-Poisson conjugate abundances + multinomial gene allocation),
//! yielding BayesPrism-style deliverables: per-sample cell-type fractions with
//! credible intervals, and a per-cell-type expression tensor for within-type DE.
//!
//! The reference is built by collapsing annotated single cells into archetypes
//! and mapping those onto cell types through the annotation posterior. Profiles
//! are measured rather than reconstructed, so nothing caps how well a
//! composition can fit the way a low-rank reconstruction does. Several archetype
//! granularities are run and pooled, so the partition — a nuisance parameter the
//! data does not pin down — is averaged over rather than conditioned on.

mod archetypes;
mod args;
mod gibbs;
mod io;
mod monitor;
mod reference;
mod result;
pub(crate) mod source;
#[cfg(test)]
mod tests;

pub use args::DeconvolveArgs;

use crate::embed_common::{read_bulk_data_aligned, Mat};
use crate::run_manifest;
use anyhow::Result;
use gibbs::PosteriorAccum;
use log::info;
use matrix_util::common_io::mkdir_parent;
use monitor::Monitor;
use reference::Reference;
use source::EmbeddingSource;

/// Seed stride between independent streams — an odd, well-mixed constant so the
/// per-sample streams of different chains cannot coincide.
pub(super) const SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

pub fn run(args: &DeconvolveArgs) -> Result<()> {
    let out: String = match args.out.as_deref() {
        Some(o) => o.to_string(),
        None => format!("{}.deconv", run_manifest::derive_out_prefix(&args.from)),
    };
    mkdir_parent(&out)?;

    // 1. Embedding source and bulk, aligned on the gene axis.
    let src = EmbeddingSource::load(&args.from)?;
    let bulk = read_bulk_data_aligned(&args.bulk, &src.feature_names, &args.bulk_table.opts())?;
    let mut cfg = args.sampler_config();
    let (d, s) = (bulk.genes.len(), bulk.samples.len());

    // 2. One chain per archetype granularity, all pooling into one posterior.
    // The references are built together, because the parquet loads, the cell
    // alignment and the streaming gene sums are the same for every granularity
    // and must not be re-done per chain.
    anyhow::ensure!(
        !args.archetypes.is_empty(),
        "deconvolve: `--archetypes` needs at least one target"
    );
    let references = {
        let (refs, membership) = archetypes::build_all(
            &args.archetype_config(),
            &src,
            &bulk.genes,
            &args.archetypes,
        )?;
        // Written here, not in `write_outputs`: it needs the references, which
        // the chain loop consumes. Dropping the membership at the end of this
        // block also keeps it off the heap for the whole sampling run.
        io::write_archetype_diagnostics(&out, &refs, &membership)?;
        refs
    };
    anyhow::ensure!(!references.is_empty(), "deconvolve: no reference built");
    let n_chains = references.len();

    // Everything the pooled posterior needs is fixed by the first reference, so
    // it is read once here rather than deferred into the loop behind an Option.
    let first = &references[0];
    let celltype_names = first.celltype_names.clone();
    let coord_dim = first.coords.ncols();
    let n_comp_total: usize = references.iter().map(Reference::n_comp).sum();
    // One prior for every chain. Chains differ by partition and by RNG stream —
    // nothing else — so a between-chain disagreement is attributable to the
    // partition, which is the whole point of pooling over it.
    if args.frac_prior_shape.is_none() {
        cfg.a0 = reference::default_prior_shape(&bulk.data, n_comp_total / n_chains.max(1));
        info!(
            "deconvolve: per-component prior shape a0 = {:.1}, shared by all {n_chains} chains",
            cfg.a0
        );
    }
    let mut accum = PosteriorAccum::new(s, first.n_types(), d, n_comp_total);
    let mut monitor = Monitor::new(args.monitor_config(), &out, &bulk.samples, &celltype_names)?;
    let mut comp_names: Vec<Box<str>> = Vec::new();
    let mut comp_coords: Vec<f32> = Vec::new();

    for (chain, reference) in references.into_iter().enumerate() {
        anyhow::ensure!(
            reference.celltype_names == celltype_names && reference.coords.ncols() == coord_dim,
            "deconvolve: chain {} does not agree with the first chain on cell types or \
             embedding dimension",
            chain + 1
        );
        // How to start follows from how the reference scales its profiles.
        let init_w = reference.init_abundances(&bulk.data);
        info!(
            "deconvolve: chain {}/{n_chains} — {s} samples × {d} genes, {} archetypes → {} cell \
             types",
            chain + 1,
            reference.n_comp(),
            reference.n_types()
        );

        // Each chain gets its own RNG stream. Sharing one would make two chains
        // over the same reference bit-identical, which triple-counts the same
        // draws and reports R̂ = 1 from no independent information.
        cfg.seed = args
            .seed
            .wrapping_add((chain as u64).wrapping_mul(SEED_STRIDE));
        monitor.begin_chain(chain);
        gibbs::run_chain(
            &reference,
            &bulk.data,
            &init_w,
            &cfg,
            &mut monitor,
            &mut accum,
        )?;

        // Component-axis bookkeeping for the output table, prefixed per chain so
        // pooled partitions stay distinguishable.
        let coords = reference.coords;
        for m in 0..coords.nrows() {
            comp_names.push(reference::comp_label(
                n_chains,
                chain,
                &reference.comp_names[m],
            ));
            comp_coords.extend(coords.row(m).iter().copied());
        }
    }

    monitor.finish()?;
    let coords = Mat::from_row_slice(comp_names.len(), coord_dim, &comp_coords);
    let result = gibbs::finalize(
        accum,
        &bulk.data,
        gibbs::ComponentTable {
            coords,
            names: comp_names,
        },
        celltype_names,
    )?;

    // 3. Write deliverables.
    let meta = io::RunMeta {
        from: &args.from,
        kind: src.kind.to_string(),
        n_components: result.anchor_names.len(),
        n_chains,
        warmup: cfg.warmup,
        draws: cfg.draws,
        bulk_files: &args.bulk,
        traced: args.trace_every > 0,
    };
    io::write_outputs(&out, &bulk.samples, &bulk.genes, &result, &meta)?;
    info!("senna deconvolve complete → {out}.*");
    Ok(())
}
