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
use reference::{identity_readout, Reference};
use source::EmbeddingSource;

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

    // 2. One chain per reference partition, all pooling into one posterior.
    // The low-rank reference has nothing to partition, so it is a single chain.
    let targets: Vec<usize> = match args.reference {
        ReferenceMode::LowRank => vec![0],
        ReferenceMode::Archetype => {
            anyhow::ensure!(
                !args.archetypes.is_empty(),
                "deconvolve: `--archetypes` needs at least one target"
            );
            args.archetypes.clone()
        }
    };

    let mut accum: Option<PosteriorAccum> = None;
    let mut monitor: Option<Monitor> = None;
    let mut celltype_names: Vec<Box<str>> = Vec::new();
    let mut comp_names: Vec<Box<str>> = Vec::new();
    let mut comp_coords: Vec<Vec<f32>> = Vec::new();
    let mut rates_sum: Option<Mat> = None;

    for (chain, &target) in targets.iter().enumerate() {
        let (mut reference, init_w) = match prior.as_ref() {
            Some(prior) => {
                let c = prior.mean.nrows();
                let reference = Reference {
                    mu_gm: vec![0.0; d * c],
                    n_genes: d,
                    n_comp: c,
                    readout: identity_readout(c),
                    coords: prior.mean.clone(),
                    comp_names: prior.names.clone(),
                    celltype_names: prior.names.clone(),
                };
                let init_w = project::init_fractions(&prior.mean, &sample_z, cfg.init_iters);
                (reference, init_w)
            }
            None => {
                let mut acfg = args.archetype_config();
                acfg.target = target;
                // A different seed per chain so the partitions are genuinely
                // distinct draws, not the same clustering at another resolution.
                acfg.seed = args.seed.wrapping_add(chain as u64 * 0x9E37_79B9);
                let reference = archetypes::build(&acfg, &src, &bulk.genes)?;
                let init_w = flat_init(&bulk.data, reference.n_comp);
                (reference, init_w)
            }
        };

        if args.reference == ReferenceMode::Archetype && args.frac_prior_shape.is_none() {
            cfg.a0 = auto_prior_shape(&bulk.data, reference.n_comp);
        }
        info!(
            "deconvolve: chain {}/{} — {s} samples × {d} genes, {} components → {} cell types \
             ({:?} reference, a0 = {:.1})",
            chain + 1,
            targets.len(),
            reference.n_comp,
            reference.n_types(),
            args.reference,
            cfg.a0
        );

        if accum.is_none() {
            celltype_names = reference.celltype_names.clone();
            accum = Some(PosteriorAccum::new(s, reference.n_types(), d));
            monitor = Some(Monitor::new(
                args.monitor_config(),
                &out,
                &bulk.samples,
                &celltype_names,
            )?);
        }
        anyhow::ensure!(
            reference.celltype_names == celltype_names,
            "deconvolve: chain {} reports different cell types than the first chain",
            chain + 1
        );

        let anchor_sampler = prior
            .as_ref()
            .map(|p| AnchorSampler::new(&src, p, cfg.seed));
        let posterior_anchors = gibbs::run_chain(
            &mut reference,
            &bulk.data,
            &init_w,
            &cfg,
            anchor_sampler,
            monitor.as_mut().expect("monitor initialised above"),
            accum.as_mut().expect("accumulator initialised above"),
        )?;

        // Component-axis bookkeeping for the output table, prefixed per chain so
        // pooled partitions stay distinguishable. Anchors report their posterior
        // position; archetypes are fixed, so theirs is the reference coordinate.
        let coords = posterior_anchors.unwrap_or_else(|| reference.coords.clone());
        for m in 0..reference.n_comp {
            comp_names.push(if targets.len() > 1 {
                format!("c{chain}_{}", reference.comp_names[m]).into_boxed_str()
            } else {
                reference.comp_names[m].clone()
            });
            comp_coords.push(coords.row(m).iter().copied().collect());
        }

        // Per-type rates for the residual report, averaged over the partitions
        // that produced them.
        let rates = reference.type_rates();
        rates_sum = Some(match rates_sum.take() {
            Some(acc) => acc + rates,
            None => rates,
        });
    }

    let accum = accum.expect("at least one chain");
    if let Some(m) = monitor.as_mut() {
        m.finish()?;
    }
    let type_rates = rates_sum.expect("at least one chain") / targets.len() as f32;
    let h = comp_coords.first().map_or(0, Vec::len);
    let coords = Mat::from_fn(comp_coords.len(), h, |i, j| comp_coords[i][j]);
    let result = gibbs::finalize(
        &accum,
        &bulk.data,
        &type_rates,
        coords,
        comp_names,
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
        n_chains: targets.len(),
        fraction_units: fraction_units(args.reference),
        warmup: cfg.warmup,
        draws: cfg.draws,
        bulk_files: &args.bulk,
    };
    io::write_outputs(&out, &bulk.samples, &bulk.genes, &sample_z, &result, &meta)?;
    info!("senna deconvolve complete → {out}.*");
    Ok(())
}

/// Fractions are mRNA-mass shares when the reference profiles are normalised to
/// sum to one over genes (the archetype mode), and cell shares when the
/// per-type exposure carries the total transcript load (the low-rank mode).
/// Recorded in the manifest so a units mismatch is not read as a model error.
fn fraction_units(mode: ReferenceMode) -> &'static str {
    match mode {
        ReferenceMode::LowRank => "cell",
        ReferenceMode::Archetype => "mrna",
    }
}

/// Per-component prior shape for the archetype reference: two standard
/// deviations of the counts a component would hold under a uniform split.
///
/// The allocation step is winner-take-all among overlapping profiles, so a
/// component that falls behind early can be extinguished. Holding it a couple of
/// sampling-noise units off zero is what keeps that from happening, and that
/// scale is `sqrt(N/R)` — which is why the default tracks depth instead of being
/// a constant.
fn auto_prior_shape(bulk: &Mat, r: usize) -> f32 {
    let s = bulk.ncols().max(1);
    let mean_total: f64 = (0..bulk.ncols())
        .map(|si| f64::from(bulk.column(si).iter().sum::<f32>()))
        .sum::<f64>()
        / s as f64;
    let per_component = (mean_total / r as f64).max(0.0);
    (2.0 * per_component.sqrt()).max(1.0) as f32
}

/// Warm start with each sample's counts spread evenly over the components.
/// Archetype profiles sum to one over genes, so an abundance is on the scale of
/// allocated counts and `total/R` is the uniform-composition point.
fn flat_init(bulk: &Mat, r: usize) -> Mat {
    Mat::from_fn(bulk.ncols(), r, |si, _| {
        let total: f32 = bulk.column(si).iter().sum();
        (total / r as f32).max(1e-6)
    })
}
