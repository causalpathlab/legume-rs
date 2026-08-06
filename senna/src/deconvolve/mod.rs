//! `senna deconvolve` — hierarchical-Bayes bulk deconvolution on a feature embedding.
//!
//! Bulk counts are split across reference components by a full Gibbs sampler
//! (Gamma-Poisson conjugate abundances + multinomial gene allocation), yielding
//! BayesPrism-style deliverables: per-sample cell-type fractions with credible
//! intervals, and a per-cell-type expression tensor for within-type DE.
//!
//! Two references are available, sharing the sampler and the outputs:
//!
//! * `--reference low-rank` reconstructs one profile per cell type from the
//!   embedding, `μ_{g,c} = exp(ρ_g·t_c + a_g)`, with anchors `t_c` drawn from
//!   the annotate-by-projection marker centroids and resampled by elliptical
//!   slice sampling. Annotation uncertainty widens the fraction posterior.
//! * `--reference archetype` measures the profiles instead, collapsing
//!   annotated cells into empirical archetypes mapped onto cell types by the
//!   annotation posterior. It needs the single-cell counts, and it is not
//!   subject to the reconstruction's rank ceiling. Several archetype
//!   granularities are run and pooled, so the partition — a nuisance parameter
//!   the data does not pin down — is averaged over rather than conditioned on.

mod anchors;
mod archetypes;
mod args;
mod gibbs;
mod io;
mod monitor;
mod project;
mod reference;
mod result;
mod source;
#[cfg(test)]
mod tests;

pub use args::DeconvolveArgs;

use crate::embed_common::{read_bulk_data_aligned, Mat};
use crate::run_manifest;
use anyhow::Result;
use args::ReferenceMode;
use gibbs::{AnchorSampler, PosteriorAccum};
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
    let bulk = read_bulk_data_aligned(&args.bulk, &src.feature_names)?;
    let mut cfg = args.sampler_config();
    let (d, s) = (bulk.genes.len(), bulk.samples.len());

    // The projected bulk position is carried through as a QC artifact in both
    // modes. It is not used to place the bulk against cells — a mixture does not
    // sit near its constituents in a log-linear embedding.
    let sample_z = project::project_bulk(&src.rho, &src.gene_offset, &bulk.data, cfg.project_ridge);

    // The low-rank marker prior, when that reference is selected.
    let prior = match args.reference {
        ReferenceMode::LowRank => {
            let markers = args.markers.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "deconvolve --reference low-rank needs `--markers`; the archetype reference \
                     takes `--annotation` instead"
                )
            })?;
            Some(anchors::build_anchor_prior(
                &src.anchor_emb,
                &src.feature_names,
                markers,
                &args.anchor_config(),
            )?)
        }
        ReferenceMode::Archetype => None,
    };

    // 2. One chain per reference, all pooling into one posterior. The low-rank
    // reference has nothing to partition, so it is a single chain; the archetype
    // references are built together, because the parquet loads, the cell
    // alignment and the streaming gene sums are the same for every granularity
    // and must not be re-done per chain.
    let references = match prior.as_ref() {
        Some(prior) => vec![Reference::low_rank(&src, prior, d)],
        None => {
            anyhow::ensure!(
                !args.archetypes.is_empty(),
                "deconvolve: `--archetypes` needs at least one target"
            );
            let (refs, membership) = archetypes::build_all(
                &args.archetype_config(),
                &src,
                &bulk.genes,
                &args.archetypes,
            )?;
            // Written here, not in `write_outputs`: it needs the references,
            // which the chain loop consumes. Dropping the membership at the end
            // of this arm also keeps it off the heap for the whole sampling run.
            io::write_archetype_diagnostics(&out, &refs, &membership)?;
            refs
        }
    };
    anyhow::ensure!(!references.is_empty(), "deconvolve: no reference built");
    let n_chains = references.len();

    // Everything the pooled posterior needs is fixed by the first reference, so
    // it is read once here rather than deferred into the loop behind an Option.
    let first = &references[0];
    let celltype_names = first.celltype_names.clone();
    let (axis, units) = (first.axis, first.units);
    let coord_dim = first.coords.ncols();
    let n_comp_total: usize = references.iter().map(Reference::n_comp).sum();
    let mut accum = PosteriorAccum::new(s, first.n_types(), d, n_comp_total);
    let mut monitor = Monitor::new(args.monitor_config(), &out, &bulk.samples, &celltype_names)?;
    let mut comp_names: Vec<Box<str>> = Vec::new();
    let mut comp_coords: Vec<f32> = Vec::new();

    for (chain, mut reference) in references.into_iter().enumerate() {
        anyhow::ensure!(
            reference.celltype_names == celltype_names && reference.coords.ncols() == coord_dim,
            "deconvolve: chain {} does not agree with the first chain on cell types or \
             embedding dimension",
            chain + 1
        );
        // How to start and how strongly to hold each component follow from how
        // the reference scales its profiles, so the reference decides both.
        let init_w = match prior.as_ref() {
            Some(prior) => project::init_fractions(&prior.mean, &sample_z, cfg.init_iters),
            None => reference.init_abundances(&bulk.data),
        };
        if args.frac_prior_shape.is_none() {
            cfg.a0 = reference.default_prior_shape(&bulk.data);
        }
        info!(
            "deconvolve: chain {}/{n_chains} — {s} samples × {d} genes, {} components → {} cell \
             types ({:?} reference, a0 = {:.1})",
            chain + 1,
            reference.n_comp(),
            reference.n_types(),
            args.reference,
            cfg.a0
        );

        // Each chain gets its own RNG stream. Sharing one would make two chains
        // over the same reference bit-identical, which triple-counts the same
        // draws and reports R̂ = 1 from no independent information.
        cfg.seed = args
            .seed
            .wrapping_add((chain as u64).wrapping_mul(SEED_STRIDE));
        let anchor_sampler = prior
            .as_ref()
            .map(|p| AnchorSampler::new(&src, p, cfg.seed));
        monitor.begin_chain(chain);
        let posterior_anchors = gibbs::run_chain(
            &mut reference,
            &bulk.data,
            &init_w,
            &cfg,
            anchor_sampler,
            &mut monitor,
            &mut accum,
        )?;

        // Component-axis bookkeeping for the output table, prefixed per chain so
        // pooled partitions stay distinguishable. Anchors report their posterior
        // position; archetypes are fixed, so theirs is the reference coordinate.
        let coords = posterior_anchors.unwrap_or(reference.coords);
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
            axis,
            units,
        },
        celltype_names,
    )?;

    // 3. Write deliverables.
    let meta = io::RunMeta {
        from: &args.from,
        markers: args.markers.as_deref(),
        kind: src.kind.to_string(),
        reference: match args.reference {
            ReferenceMode::LowRank => "low-rank",
            ReferenceMode::Archetype => "archetype",
        },
        n_components: result.anchor_names.len(),
        n_chains,
        fraction_units: result.units.as_str(),
        warmup: cfg.warmup,
        draws: cfg.draws,
        bulk_files: &args.bulk,
        traced: args.trace_every > 0,
    };
    io::write_outputs(&out, &bulk.samples, &bulk.genes, &sample_z, &result, &meta)?;
    info!("senna deconvolve complete → {out}.*");
    Ok(())
}
